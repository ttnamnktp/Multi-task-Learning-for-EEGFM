import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pretrain_module.eegpt.utils import augmentation, make_masks

class OriginalBYOLTask(nn.Module):
    def __init__(self, online_encoder, d_model, proj_dim=256, hidden_dim=4096, tau_base=0.996, total_steps=100000):
        super().__init__()
        
        self.online_encoder = online_encoder
        in_features = d_model  # 512
        
        # Bộ chiếu (Projector) và Bộ dự đoán (Predictor) chuẩn JAX Pseudo-code
        self.online_projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Mạng Target
        self.target_encoder = copy.deepcopy(online_encoder)
        self.target_projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        
        # Khóa Gradient Target (Stop Gradient tương đương với toán tử sg trong bài báo)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
        self.tau_base = tau_base
        self.total_steps = total_steps
        self.current_step = 0

    @torch.no_grad()
    def update_target_network(self):
        """Cập nhật mạng Target bằng EMA (tương đương ema_update trong JAX pseudo-code)"""
        tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * self.current_step / self.total_steps) + 1) / 2
        self.current_step = min(self.current_step + 1, self.total_steps)
        
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1.0 - tau)
            
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1.0 - tau)

    def regression_loss(self, q, z):
        """Hàm loss MSE chuẩn hóa chính xác theo cấu trúc toán học của JAX Pseudo-code"""
        q = F.normalize(q, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        # Công thức: mean(2 - 2 * sum(q * z, axis=-1))
        return (2.0 - 2.0 * (q * z).sum(dim=-1)).mean()

    def _pool_representation(self, x_tensor):
        return x_tensor.mean(dim=(1, 2))
    
    # Lifecycle hook
    def on_train_batch_end(self, **kwargs):
        self.update_target_network()

    def forward(self, shared_output, x, x_aug, mask_x=None, mask_y=None):
        """
        shared_output: Đầu ra online nhận x_aug kèm mask_x (View 1)
        """
        # --- VIEW 1 ---
        h_online_1 = self._pool_representation(shared_output) 
        z_online_1 = self.online_projector(h_online_1)
        q_online_1 = self.online_predictor(z_online_1)
        
        # --- VIEW 2 ---
        x_aug_2 = augmentation(x) 
        
        # Để tránh việc mô hình gian lận thông tin qua mask, ta tạo một mask ngẫu nhiên độc lập cho View 2
        if mask_x is not None:
            mask_x_2, _ = make_masks(self.online_encoder.encoder.num_patches, p_n_y=0.4, p_c_y=0.2)
            mask_x_2 = mask_x_2.to(x_aug_2.device)
        else:
            mask_x_2 = None
            
        chan_ids = self.online_encoder.chans_id.to(x_aug_2)
        shared_output_2 = self.online_encoder(x=x_aug_2, chan_ids=chan_ids, mask_x=mask_x_2)
        
        h_online_2 = self._pool_representation(shared_output_2)
        z_online_2 = self.online_projector(h_online_2)
        q_online_2 = self.online_predictor(z_online_2)
        
        # --- NHÁNH TARGET (Chặn gradient hoàn toàn qua khối torch.no_grad) ---
        with torch.no_grad():
            # Chạy View 1 (x_aug) qua Target với mask ban đầu của nó
            target_output_1 = self.target_encoder(x=x_aug, chan_ids=chan_ids, mask_x=mask_x)
            h_target_1 = self._pool_representation(target_output_1)
            z_target_1 = self.target_projector(h_target_1)
            
            # Chạy View 2 (x_aug_2) qua Target với mask độc lập của nó
            target_output_2 = self.target_encoder(x=x_aug_2, chan_ids=chan_ids, mask_x=mask_x_2)
            h_target_2 = self._pool_representation(target_output_2)
            z_target_2 = self.target_projector(h_target_2)
            
        # --- HÀM LOSS ĐỐI XỨNG (.detach() hoạt động giống hệt toán tử stop_gradient trong JAX) ---
        loss_1 = self.regression_loss(q_online_1, z_target_2.detach()) 
        loss_2 = self.regression_loss(q_online_2, z_target_1.detach()) 
        
        total_loss = (loss_1 + loss_2) / 2.0
        
        return {"loss": total_loss}