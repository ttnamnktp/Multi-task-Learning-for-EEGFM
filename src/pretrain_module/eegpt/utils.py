import random
import torch

def make_masks(num_patchs, mC_x=12, p_n_y=0.4, p_c_y=0.2):
        
        C, N = num_patchs
        mC_x = C - int(C * p_c_y)
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i * C

                if random.random() > p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)

            if len(mask_x) == 0: continue
            if len(mask_y_bx) == 0: continue
            
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            if len(mask_y_bx) == 0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y + [mask_y_bx], dim=0)

def augmentation(x, prob=0.25):
    """
    Applies exactly ONE augmentation per sample.
    """
    B, C, T = x.shape
    device = x.device
    out = x.clone()
    r = random.random()
    flag = "keep"

    # Dùng IF - ELIF để ép buộc chỉ chọn duy nhất một nhánh xử lý
    if r < prob:
        scale = torch.empty(1, device=device).uniform_(0.5, 2.0)
        out = out * scale
        flag = "scale_mag"

    elif r < 2 * prob:
        noise = torch.randn_like(out) * 0.2
        out = out + noise
        flag = "noise"

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
                out[:, :, is_before] = values_kept[:, :, 0:1].repeat(1, 1, is_before.sum())
            if is_after.any():
                out[:, :, is_after] = values_kept[:, :, -1:].repeat(1, 1, is_after.sum())
            if is_interp.any():
                left_interp = left_idx[is_interp]
                weight = (t[is_interp] - t_kept[left_interp]) / (t_kept[left_interp + 1] - t_kept[left_interp])
                out[:, :, is_interp] = values_kept[:, :, left_interp] + weight[None, None, :] * (values_kept[:, :, left_interp + 1] - values_kept[:, :, left_interp])
        flag = "interpolate"

    elif r < 4 * prob:
        fft = torch.fft.rfft(out, dim=2)
        mask_n = int(fft.shape[2] * 0.2)
        mask_idx = torch.randperm(fft.shape[2], device=device)[:mask_n]
        fft[:, :, mask_idx] = 0
        out = torch.fft.irfft(fft, n=T, dim=2)
        flag = "freq_mask"

    # return out, flag
    return out

# def augmentation(x, prob=0.25, mC_x=12, p_n_y=0.2, p_c_y=0.2):
#         """
#         x: Tensor [B, C, T]
#         Applies exactly ONE augmentation per sample
#         """
#         B, C, T = x.shape
#         device = x.device
#         out = x.clone()

#         r = random.random()

#         # -------------------------------
#         # 1) Magnitude scaling (20%)
#         # -------------------------------
#         # if random.random() < prob :
#         if r < prob :
#             # scale = torch.empty(1, device=device).uniform_(0.5, 2.0)
#             scale = torch.empty(1, device=device).uniform_(0.5, 2.0)
#             out = out * scale
#             flag = "scale_mag"

#         # -------------------------------
#         # 2) Gaussian noise (20%)
#         # -------------------------------
#         # if random.random() < prob:
#         if r < 2*prob :
#             noise = torch.randn_like(out) * 0.2  # standard deviation = 0.1
#             # noise = torch.randn_like(out) * 0.1
#             out = out + noise
#             flag = "noise"

#         # -------------------------------
#         # 3) Interpolation (20%)
#         # -------------------------------
#         # if random.random() < prob:
#         if r < 3*prob :
#             # delete_prob = 0.2
#             delete_prob = 0.2
#             keep_n = int(T * (1 - delete_prob))
#             if keep_n == 0:
#                 # Rare case: all deleted, keep original or set to zero (adjust as needed)
#                 flag = "interpolate"
#             else:
#                 idx = torch.randperm(T, device=device)[:keep_n]
#                 idx = torch.sort(idx)[0]  # [keep_n]
#                 values_kept = out[:, :, idx]  # [B, C, keep_n]
#                 t = torch.arange(T, dtype=out.dtype, device=device)  # [T]
#                 t_kept = idx.to(out.dtype)  # [keep_n]

#                 # Find left index for each t (largest k where t_kept[k] <= t)
#                 left_idx = torch.searchsorted(t_kept, t, right=True) - 1  # [T]

#                 is_before = left_idx < 0
#                 is_after = left_idx >= (keep_n - 1)
#                 is_interp = ~(is_before | is_after)

#                 # Handle before: constant with first kept
#                 if is_before.any():
#                     num_before = is_before.sum()
#                     out[:, :, is_before] = values_kept[:, :, 0:1].repeat(1, 1, num_before)

#                 # Handle after: constant with last kept
#                 if is_after.any():
#                     num_after = is_after.sum()
#                     out[:, :, is_after] = values_kept[:, :, -1:].repeat(1, 1, num_after)

#                 # Handle interp regions: linear between left and right
#                 if is_interp.any():
#                     left_interp = left_idx[is_interp]  # [num_interp]
#                     t_interp = t[is_interp]
#                     t_left = t_kept[left_interp]
#                     t_right = t_kept[left_interp + 1]
#                     delta = t_right - t_left
#                     # Avoid div by zero (shouldn't happen with unique idx)
#                     delta = torch.where(delta == 0, torch.ones_like(delta), delta)
#                     weight = (t_interp - t_left) / delta  # [num_interp]

#                     values_left = values_kept[:, :, left_interp]  # [B, C, num_interp]
#                     values_right = values_kept[:, :, left_interp + 1]  # [B, C, num_interp]
#                     values_interp = values_left + weight[None, None, :] * (values_right - values_left)
#                     out[:, :, is_interp] = values_interp

#             flag = "interpolate"

#         # -------------------------------
#         # 4) Frequency masking (20%) - mask 10% frequency
#         # -------------------------------
#         # if random.random() < prob:
#         if r < 4*prob :
#             fft = torch.fft.rfft(out, dim=2)  # FFT along T
#             freq_bins = fft.shape[2]

#             mask_n = int(freq_bins * 0.2)
#             # mask_n = int(freq_bins * 0.4)
#             mask_idx = torch.randperm(freq_bins, device=device)[:mask_n]

#             fft[:, :, mask_idx] = 0
#             out = torch.fft.irfft(fft, n=T, dim=2)
#             flag = "freq_mask"
#         else:
#             flag = "keep"
            
#         return out

class GradientMonitor:

    def __init__(self, pl_module):
        self.model = pl_module.model
        self.pl_module = pl_module

    def collect_modules(self):
        modules = {}

        for i, blk in enumerate(self.model.encoder.blocks):
            modules[f"enc.block.{i}"] = blk.attn
            modules[f"enc.mlp.{i}"] = blk.mlp

        return modules

    def get_grads(self, loss, modules):
        params = []
        names = []

        for name, m in modules.items():
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(name)

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True
        )

        out = {}
        for name, g in zip(names, grads):
            if g is None:
                continue
            out.setdefault(name, []).append(g.detach().flatten())

        return {k: torch.cat(v) for k, v in out.items()}

    def log_conflict(self, loss_a, loss_b, step=True):
        modules = self.collect_modules()

        ga = self.get_grads(loss_a, modules)
        gb = self.get_grads(loss_b, modules)

        for name in modules:
            if name not in ga or name not in gb:
                continue

            cos = torch.dot(ga[name], gb[name]) / (
                torch.norm(ga[name]) * torch.norm(gb[name]) + 1e-8
            )

            self.pl_module.log(
                f"conflict/{name}/cos",
                cos,
                on_step=True
            )