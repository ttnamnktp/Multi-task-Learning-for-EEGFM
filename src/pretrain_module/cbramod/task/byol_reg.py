# ============================================
# contrastive_byol_task.py
# ============================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_task import BaseTask
from src.pretrain_module.cbramod.utils import augmentation, make_mask

# ============================================
# Projection Head
# ============================================

class ProjectionHead(nn.Module):

    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# Predictor
# ============================================

class Predictor(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)


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
        reg_weight=1,      # thêm: trọng số KoLeo, giống hệ số 0.1 trong B
    ):
        super().__init__()

        # Tạo momentum scheduler giống chương trình B nếu total_steps được cung cấp
        self.tau = tau
        self.tau_end = tau_end
        self.temperature = temperature
        self.reg_weight = reg_weight

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
        # TARGET PROJECTOR
        # =====================================
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

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

    def forward(self, shared_output, batch, mask):
        """
        shared_output: out_enc của nhánh Online nhận x_aug [B, C, N, D]
        batch[0]: Dữ liệu gốc x (chưa qua augmentation)
        """
        x = batch
        id_tensor = torch.arange(x.shape[0], device=x.device)

        # =====================================
        # 1. NHÁNH ONLINE (Từ x_aug)
        # =====================================
        # shared_output chính là kết quả của Online Encoder khi nhận x_aug
        z_online = self.projector(shared_output)    # -> [B, proj_dim]
        p_online = self.predictor(z_online)    # -> [B, proj_dim] (z_cons trong code cũ)
        h_online = self.pool(p_online)    # -> [B, D]

        # =====================================
        # 2. NHÁNH TARGET EMA (Từ x sạch)
        # =====================================
        with torch.no_grad():
            # Tạo mask nhẹ cho nhánh EMA giống hệt tỷ lệ code trước (mask / 4)
            # Giả định mask gốc là 0.2 thì ema_mask_ratio là 0.05
            ema_mask = make_mask(x, 0.1) 
            
            # Khởi chạy Target Encoder trên dữ liệu gốc x
            h_target_enc = self.target_encoder(x, mask=ema_mask)
            
            # Xử lý nếu đầu ra của target_encoder trả về dict hoặc tensor thuần
            if isinstance(h_target_enc, dict):
                h_target_enc = h_target_enc.get("latent", h_target_enc.get("out_enc", h_target_enc))
                
            z_target = self.target_projector(h_target_enc) 
            h_target = self.pool(z_target)      

        # =====================================
        # 3. TÍNH TOÁN LOSS
        # =====================================

        loss= id_loss(
            z1=h_online,
            z2=h_target.detach(),
            id=id_tensor,
            temperature=self.temperature,
            reg_weight=self.reg_weight,
        )

        return {
            "loss": loss,
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
    device = z1.device
    B, D = z1.shape

    # ── Regularizer mới (thay KoLeo) ──────────────────────────────────
    # Tính TRƯỚC khi normalize vì bên trong hàm đã normalize lại
    loss_reg = mean_pairwise_sim_reg(z1, z2)   # >= 0

    # ── Contrastive NCE (giữ nguyên) ──────────────────────────────────
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    id = id.to(device)

    def one_direction_loss(exp_sim, id):
        loss = 0.0
        num_valid_anchors = 0
        for i in range(B):
            pos_mask = (id == id[i])
            num_pos = pos_mask.sum().item()
            if num_pos == 0:
                continue
            pos_exp = exp_sim[i][pos_mask]
            all_sum = exp_sim[i].sum()
            if decoupled:
                denoms = torch.clamp(all_sum - pos_exp, min=1e-6)
                log_probs = torch.log(pos_exp / denoms)
            else:
                log_probs = torch.log(pos_exp / all_sum)
            loss += -log_probs.mean()
            num_valid_anchors += 1
        if num_valid_anchors == 0:
            return torch.tensor(0.0, device=device)
        return loss / num_valid_anchors

    sim12 = torch.mm(z1_norm, z2_norm.T) / temperature
    l12 = one_direction_loss(torch.exp(sim12), id)

    sim21 = torch.mm(z2_norm, z1_norm.T) / temperature
    l21 = one_direction_loss(torch.exp(sim21), id)

    loss_nce = (l12 + l21) / 2

    # ── Tổng loss ──────────────────────────────────────────────────────
    loss_total = loss_nce + reg_weight * loss_reg
    print(f"Loss_nce: {loss_nce}")
    print(f"Loss_reg: {loss_reg}")

    return loss_total