import torch.nn as nn
import torch.nn.functional as F
from .base_task import BaseTask

class ContrastiveTask(BaseTask):

    def __init__(self, model, d_model, proj_dim=128, temperature=0.5):
        super().__init__()

        self.model = model
        self.projector = ProjectionHead(d_model, proj_dim)
        self.loss_fn = NTXentLoss(temperature)

    def forward(self, shared_output, batch):

        # 1. create 2 views
        x1 = batch[0]
        x2 = augmentation(batch[0])

        # 2. forward backbone
        h1 = self.model(x1, return_mask=False)["latent"]
        h2 = self.model(x2, return_mask=False)["latent"]

        # 3. pooling
        h1 = h1.mean(dim=1)
        h2 = h2.mean(dim=1)

        # 4. projection
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # 5. loss
        loss = self.loss_fn(z1, z2)

        return {"loss": loss}

class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D]
        """
        B = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z1 = z1.mean(dim=1)
        z2 = z2.mean(dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim = torch.matmul(z, z.T) / self.t  # [2B, 2B]

        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        pos = torch.cat([
            torch.arange(B, 2*B),
            torch.arange(0, B)
        ]).to(z.device)

        loss = F.cross_entropy(sim, pos)
        return loss

import random
import torch

def augmentation(x, prob=0.25):
    """
    Đầu vào x đã là 4 chiều chuẩn: [B, C, N, patch_size]
    Hàm áp dụng DUY NHẤT một phép tăng cường cho mỗi batch.
    """
    # 1. Trích xuất kích thước từ Tensor 4 chiều
    B, C, N, patch_size = x.shape
    T = N * patch_size  # Tổng số điểm thời gian (Time-steps)
    device = x.device

    # 2. Ép phẳng tạm thời về 3 chiều [B, C, T] để thực hiện biến đổi
    x_3d = x.view(B, C, T)
    out = x_3d.clone()

    r = random.random()

    # -------------------------------
    # 1) Magnitude scaling (25% cơ hội)
    # -------------------------------
    if r < prob:
        scale = torch.empty(1, device=device).uniform_(0.5, 2.0)
        out = out * scale
        flag = "scale_mag"

    # -------------------------------
    # 2) Gaussian noise (25% cơ hội)
    # -------------------------------
    elif r < 2 * prob:
        noise = torch.randn_like(out) * 0.2  
        out = out + noise
        flag = "noise"

    # -------------------------------
    # 3) Interpolation (25% cơ hội)
    # -------------------------------
    elif r < 3 * prob:
        delete_prob = 0.2
        keep_n = int(T * (1 - delete_prob))
        if keep_n > 0:
            idx = torch.randperm(T, device=device)[:keep_n]
            idx = torch.sort(idx)[0]
            values_kept = out[:, :, idx]
            t = torch.arange(T, dtype=out.dtype, device=device)
            t_kept = idx.to(out.dtype)

            left_idx = torch.searchsorted(t_kept, t, right=True) - 1

            is_before = left_idx < 0
            is_after = left_idx >= (keep_n - 1)
            is_interp = ~(is_before | is_after)

            if is_before.any():
                num_before = is_before.sum()
                out[:, :, is_before] = values_kept[:, :, 0:1].repeat(1, 1, num_before)

            if is_after.any():
                num_after = is_after.sum()
                out[:, :, is_after] = values_kept[:, :, -1:].repeat(1, 1, num_after)

            if is_interp.any():
                left_interp = left_idx[is_interp]
                t_interp = t[is_interp]
                t_left = t_kept[left_interp]
                t_right = t_kept[left_interp + 1]
                delta = t_right - t_left
                delta = torch.where(delta == 0, torch.ones_like(delta), delta)
                weight = (t_interp - t_left) / delta

                values_left = values_kept[:, :, left_interp]
                values_right = values_kept[:, :, left_interp + 1]
                values_interp = values_left + weight[None, None, :] * (values_right - values_left)
                out[:, :, is_interp] = values_interp
        flag = "interpolate"

    # -------------------------------
    # 4) Frequency masking (25% cơ hội)
    # -------------------------------
    elif r < 4 * prob:
        fft = torch.fft.rfft(out, dim=2)
        freq_bins = fft.shape[2]
        mask_n = int(freq_bins * 0.2)
        mask_idx = torch.randperm(freq_bins, device=device)[:mask_n]
        fft[:, :, mask_idx] = 0
        out = torch.fft.irfft(fft, n=T, dim=2)
        flag = "freq_mask"
    else:
        flag = "keep"

    # 3. KHÔI PHỤC LẠI dạng 4 chiều [B, C, N, patch_size] cho out
    out = out.view(B, C, N, patch_size)
        
    return out