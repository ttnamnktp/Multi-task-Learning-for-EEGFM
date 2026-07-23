# ============================================
# byol_reg.py
# ============================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_task import BaseTask
from src.pretrain_module.eegpt.utils import augmentation, make_masks

# ============================================
# Projection Head
# ============================================

class ProjectionHead(nn.Module):

    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.layer2(x)
        return x

# ============================================
# Predictor
# ============================================

class Predictor(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=512):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


# ============================================
# BYOL Task
# ============================================

class BYOLRegTask(BaseTask):

    def __init__(
        self,
        online_encoder,
        d_model,
        proj_dim=256,
        hidden_dim=512,
        tau=0.996,
        tau_end=0.999,
        total_steps=None,
        temperature=0.1,       # thêm: nhiệt độ cho id_loss
        reg_weight=1,      
        decoupled=False,
    ):
        super().__init__()

        # Tạo momentum scheduler giống chương trình B nếu total_steps được cung cấp
        self.tau = tau
        self.tau_end = tau_end
        self.temperature = temperature
        self.reg_weight = reg_weight
        self.decoupled = decoupled

        if total_steps is not None:
            self._momentum_scheduler = iter(
                tau + i * (tau_end - tau) / total_steps
                for i in range(total_steps + 1)
            )
        else:
            self._momentum_scheduler = None  # fallback về tau cố định

        # =====================================
        # ONLINE ENCODER REFERENCE
        # =====================================
        self.online_encoder = online_encoder

        # =====================================
        # TARGET ENCODER (EMA)
        # =====================================
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # =====================================
        # ONLINE PROJECTOR
        # =====================================
        self.projector = ProjectionHead(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            out_dim=proj_dim
        )

        # =====================================
        # ONLINE PREDICTOR
        # =====================================
        self.predictor = Predictor(
            in_dim=proj_dim,
            hidden_dim=hidden_dim
        )

        # =====================================
        # TARGET PROJECTOR (EMA)
        # =====================================
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # =====================================
        # PRINT TASK INITIALIZATION PARAMETERS
        # =====================================
        print("\n" + "="*50)
        print(f"🚀 INITIALIZING TASK: {self.__class__.__name__}")
        print("="*50)
        print(f" 🔹 proj_dim         : {proj_dim}")
        print(f" 🔹 hidden_dim       : {hidden_dim}")
        print(f" 🔹 tau (EMA start)  : {tau}")
        print(f" 🔹 tau_end (EMA end): {tau_end}")
        print(f" 🔹 total_steps      : {total_steps} (Scheduler: {'Enabled' if total_steps else 'Disabled'})")
        print(f" 🔹 temperature      : {temperature}")
        print(f" 🔹 reg_weight       : {reg_weight}")
        print(f" 🔹 decoupled        : {decoupled}")
        print("="*50 + "\n")

    # ============================================
    # Pooling
    # ============================================

    def pool(self, h):
        """
        h: [B, C, N, D] hoặc [B, N, D] phụ thuộc vào chiều của tensor đầu ra.
        Đoạn code gốc: out_enc.mean(dim=[1, 2]) áp dụng cho tensor 4D [B, C, N, D]
        """
        if h.dim() == 4:
            return h.mean(dim=(1, 2))
        elif h.dim() == 3:
            return h.mean(dim=1)
        return h

    # ============================================
    # EMA UPDATE
    # ============================================

    @torch.no_grad()
    def update_ema(self):
        
        # Dùng momentum scheduler nếu có, không thì dùng tau cố định
        if self._momentum_scheduler is not None:
            try:
                m = next(self._momentum_scheduler)
            except StopIteration:
                m = self.tau_end  # sau khi hết scheduler, giữ ở tau_end
        else:
            m = self.tau

        # encoder EMA
        for online, target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target.data.mul_(m).add_(online.data, alpha=1 - m)

        # projector EMA
        for online, target in zip(
            self.projector.parameters(),
            self.target_projector.parameters()
        ):
            target.data.mul_(m).add_(online.data, alpha=1 - m)

    # ============================================
    # Lifecycle hook
    # ============================================

    def on_train_batch_end(self, **kwargs):
        self.update_ema()

    # ============================================
    # FORWARD (Chuẩn hóa theo flow code gốc)
    # ============================================

    def forward(self, shared_output, x, x_aug, mask_x, mask_y):
        """
        shared_output: out_enc của nhánh Online nhận x_aug [B, C, N, D]
        batch[0]: Dữ liệu gốc x (chưa qua augmentation)
        """
        B = x.shape[0] # Lấy kích thước Batch thực tế
        device = x.device
        # Tự động khởi tạo mảng ID tuần tự giống hệt như code mẫu làm trong training_step
        id_tensor = torch.arange(B, device=device)

        # =====================================
        # 1. NHÁNH ONLINE (Từ x_aug)
        # =====================================
        # shared_output chính là kết quả của Online Encoder khi nhận x_aug
        z_online = self.projector(shared_output)   
        p_online = self.predictor(z_online)    
        p_online = self.pool(p_online)

        # =====================================
        # 2. NHÁNH TARGET EMA 
        # =====================================
        x2 = augmentation(x)
        with torch.no_grad():
            # ema_mask = mask_x
            ema_mask, _ = make_masks(
                self.online_encoder.encoder.num_patches, 
                p_n_y=0.1, 
                p_c_y=0.05
            )

            # Khởi chạy Target Encoder trên dữ liệu gốc x
            h_target = self.target_encoder(x2, self.online_encoder.chans_id.to(x2) , mask_x=ema_mask)
            p_target = self.target_projector(h_target) 
            p_target = self.pool(p_target)

        # =====================================
        # 3. TÍNH TOÁN LOSS
        # =====================================
        # Detach target giống BYOL — chỉ online branch được backprop
        loss, loss_reg = id_loss(
            z1=p_online,
            z2=p_target.detach(),
            id=id_tensor,
            temperature=self.temperature,
            decoupled=self.decoupled,
            reg_weight=self.reg_weight,
        )
 
        return {
            "loss": loss,
            "loss_koleo": loss_reg,   # optional: để BaseModule log nếu muốn
        }
        

 # ============================================
# Loss functions 
# ============================================

def mean_pairwise_sim_reg(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Regularizer thay thế KoLeo — luôn >= 0.
    Phạt khi các embedding quá giống nhau (cosine similarity cao).
    
    Args:
        z1, z2: [B, D] — embedding đã normalize (hoặc chưa, sẽ normalize bên trong)
    Returns:
        scalar >= 0
    """
    B = z1.size(0)
    if B < 2:
        return z1.new_tensor(0.0)

    z1_n = F.normalize(z1, dim=1)   # [B, D]
    z2_n = F.normalize(z2, dim=1)   # [B, D]

    # Ma trận cosine similarity [B, B]
    sim1 = torch.mm(z1_n, z1_n.T)   # similarity nội bộ z1
    sim2 = torch.mm(z2_n, z2_n.T)   # similarity nội bộ z2

    # Loại đường chéo (sim(i,i) = 1 luôn luôn, không phải signal)
    mask = ~torch.eye(B, dtype=torch.bool, device=z1.device)

    # Trung bình bình phương của off-diagonal similarities
    reg1 = sim1[mask].pow(2).mean()
    reg2 = sim2[mask].pow(2).mean()

    return 0.5 * (reg1 + reg2)

def id_loss(z1, z2, id, temperature=0.1, decoupled=False, reg_weight=1):
    '''
    Tính Contrastive Loss dựa trên Subject ID pairing kết hợp với KoLeo (Đúng logic mẫu).
    '''
    device = z1.device
    B, D = z1.shape
    
    # ── Regularizer mới (thay KoLeo) ──────────────────────────────────
    # Tính TRƯỚC khi normalize vì bên trong hàm đã normalize lại
    # loss_reg = mean_pairwise_sim_reg(z1, z2)
    
    # Chuẩn hóa l2 phục vụ cho Contrastive Loss
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    id = id.to(device)

    def one_direction_loss(exp_sim, id):
        loss = 0.0
        num_valid_anchors = 0
        
        # Tạo mask tương tác ID: pos_mask[i, j] = True nếu id[i] == id[j]
        # Sử dụng vòng lặp theo đúng logic xử lý mask và phân tách mẫu (decoupled) của code mẫu
        for i in range(B):
            pos_mask = (id == id[i])
            num_pos = pos_mask.sum().item()
            if num_pos == 0:
                continue
                
            pos_exp = exp_sim[i][pos_mask]
            all_sum = exp_sim[i].sum()
            
            if decoupled:
                denoms = all_sum - pos_exp
                denoms = torch.clamp(denoms, min=1e-6)
                log_probs = torch.log(pos_exp / denoms)
            else:
                denom = all_sum
                log_probs = torch.log(pos_exp / denom)
                
            loss += -log_probs.mean()
            num_valid_anchors += 1
            
        if num_valid_anchors == 0:
            return torch.tensor(0.0, device=device)
        return loss / num_valid_anchors

    # Hướng 1: z1 làm anchor, z2 làm targets
    sim12 = torch.mm(z1_norm, z2_norm.T) / temperature
    exp_sim12 = torch.exp(sim12)
    l12 = one_direction_loss(exp_sim12, id)

    # Hướng 2: z2 làm anchor, z1 làm targets
    sim21 = torch.mm(z2_norm, z1_norm.T) / temperature
    exp_sim21 = torch.exp(sim21)
    l21 = one_direction_loss(exp_sim21, id)

    # Tính toán contrastive loss tổng hợp từ 2 hướng
    loss_nce = (l12 + l21) / 2
    
    # Kết hợp tổng loss với trọng số tương tự cấu trúc hiện tại của bạn
    # loss_total = loss_nce + reg_weight * loss_reg
    loss_total = loss_nce
    print(f"\nLoss_nce: {loss_nce}")
    # print(f"Loss_reg: {loss_reg}")

    return loss_total
