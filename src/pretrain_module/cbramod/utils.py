
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

def make_mask(x, mask_ratio=0.5):
        # print(x.shape)
        bz, ch, patch, _ = x.shape

        mask = torch.zeros((bz, ch, patch), device=x.device)
        mask = mask.bernoulli_(mask_ratio)

        return mask