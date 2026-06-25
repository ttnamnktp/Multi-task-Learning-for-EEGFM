import torch.nn.functional as F
import torch
import torch.nn as nn

from src.pretrain_module.registry import register_module
from src.pretrain_module.base_module import BaseModule
from src.models.registry import get_model
from src.models.eegpt.eegpt import EEGPTModel
from .task.reconstruction import ReconstructionTask
from .task.contrastive import ContrastiveTask
from .task.byol import OriginalBYOLTask
from .task.byol_nce import ContrastiveBYOLTask
from .task.byol_reg import BYOLRegTask
from .config_builder import EEGPTConfigBuilder
from .utils import augmentation, make_masks


@register_module("eegpt_pretrain_module")
class EEGPretrain(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_training()  # Initialize model, tasks, and weight method

    def _build_model(self, cfg) -> nn.Module:
        """Build EEGPT model."""
        builder = EEGPTConfigBuilder()
        model_cfg = builder.build(cfg)
        model = EEGPTModel(model_cfg)
        # Store model_cfg for task building
        self.model_cfg = model_cfg
        return model

    def _build_tasks(self, cfg, model_cfg=None) -> nn.ModuleDict:
        """Build EEGPT-specific tasks."""
        if model_cfg is None:
            model_cfg = self.model_cfg
        
        tasks = nn.ModuleDict()
        task_configs = cfg.get("tasks", {})

        # ==============================================================
        # Reconstruction task
        if task_configs.get("reconstruction", {}).get("enabled", False):
            tasks["reconstruction"] = ReconstructionTask(
                online_encoder=self.model,
                models_configs=self.model_cfg
            )
            print("[INFO] ✓ Reconstruction task enabled")
        
        # ==============================================================
        # Contrastive task
        if task_configs.get("contrastive", {}).get("enabled", False):
            contrastive_cfg = task_configs["contrastive"]
            tasks["contrastive"] = ContrastiveTask(
                model=self.model,
                d_model=model_cfg["embed_dim"],
                proj_dim=contrastive_cfg.get("proj_dim", 128),
                temperature=contrastive_cfg.get("temperature", 0.5)
            )
            print("[INFO] ✓ Contrastive task enabled")
        
        try:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            max_epochs = self.trainer.max_epochs
            total_steps = int(steps_per_epoch * max_epochs) + 1
            print(f"[DEBUG] Total step: {total_steps}")
        except Exception:
            total_steps = 100000
            print(f"[DEBUG] Total step is set to {total_steps}")

        # ==============================================================
        # BYOL task
        if task_configs.get("byol", {}).get("enabled", False):
            byol_cfg = task_configs["byol"]
            
            tasks["byol"] = ContrastiveBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["embed_dim"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996),
                tau_end=byol_cfg.get("tau_end", 0.999),   # FIX Bug: thêm tham số mới
                total_steps=total_steps,                   # FIX Bug: truyền total_steps
                temperature=byol_cfg.get("temperature", 0.1),
                decoupled=byol_cfg.get("decoupled", False),
            )
            print("[INFO] ✓ BYOL task enabled")

        # ==============================================================
        # BYOL REG task
        if task_configs.get("byol_reg", {}).get("enabled", False):
            byol_cfg = task_configs["byol_reg"]
 
            tasks["byol_reg"] = BYOLRegTask(
                online_encoder=self.model,
                d_model=model_cfg["embed_dim"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996),
                tau_end=byol_cfg.get("tau_end", 0.999),  
                total_steps=total_steps,       
                temperature=byol_cfg.get("temperature", 0.1),
                reg_weight=byol_cfg.get("reg_weight", 1),
                decoupled=byol_cfg.get("decoupled", False),
            )
            print("[INFO] ✓ BYOL REG task enabled")

        # ==============================================================
        # Original BYOL Task đúng chuẩn bài báo gốc
        if task_configs.get("byol_original", {}).get("enabled", False):
            byol_orig_cfg = task_configs["byol_original"]
            
            tasks["byol_original"] = OriginalBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["embed_dim"],
                proj_dim=byol_orig_cfg.get("proj_dim", 256),
                hidden_dim=byol_orig_cfg.get("hidden_dim", 4096),  # Mặc định bài gốc là 4096
                tau_base=byol_orig_cfg.get("tau", 0.996),          # Mặc định bài gốc là 0.996
                total_steps=total_steps
            )
            print("[INFO] ✓ Original BYOL task enabled")

        # ==============================================================
        if len(tasks) == 0:
            raise ValueError("At least one task must be enabled!")
        
        print(f"[INFO] Total active tasks: {len(tasks)}\n")
        return tasks

    # ==============================================================
    # Shared step
    # ==============================================================
    def shared_step(self, batch):
        x, _ = batch
        x_aug = augmentation(x)

        mask_x, mask_y = make_masks(
            self.model.encoder.num_patches, 
            p_n_y=0.4, 
            p_c_y=0.2
        )

        # Forward pass
        shared_output = self.model(
            x=x_aug, 
            chan_ids=self.model.chans_id.to(x_aug), 
            mask_x=mask_x
        )
        
        # Compute task losses
        loss_dict = {}
        for task_name, task in self.tasks.items():
            task_output = task(shared_output=shared_output, x=x, x_aug=x_aug, mask_x=mask_x, mask_y=mask_y)
            loss_dict[task_name] = task_output["loss"]
        
        return loss_dict

# Check forward

# import torch
# from omegaconf import OmegaConf


# def debug_forward_original_byol(module, x):
#     """
#     Chạy check forward cho task byol_original.
#     x: tensor shape [B, 19, 3200]
#     """
#     device = next(module.parameters()).device
#     x = x.to(device)

#     print("=" * 100)
#     print("[DEBUG] Start Original BYOL forward check")
#     print(f"[DEBUG] Input x.shape = {x.shape}")

#     if "byol_original" not in module.tasks:
#         raise ValueError("Task 'byol_original' is not enabled in module.tasks")

#     byol_task = module.tasks["byol_original"]

#     # ------------------------------------------------------------
#     # View 1
#     # ------------------------------------------------------------
#     x_aug = augmentation(x)
#     print(f"[DEBUG] x_aug.shape = {x_aug.shape}")

#     mask_x, mask_y = make_masks(
#         module.model.encoder.num_patches,
#         p_n_y=0.4,
#         p_c_y=0.2
#     )

#     if mask_x is not None:
#         mask_x = mask_x.to(device)
#     if mask_y is not None:
#         mask_y = mask_y.to(device)

#     print(f"[DEBUG] mask_x.shape = {None if mask_x is None else mask_x.shape}")
#     print(f"[DEBUG] mask_y.shape = {None if mask_y is None else mask_y.shape}")

#     chan_ids = module.model.chans_id.to(device)
#     print(f"[DEBUG] chan_ids.shape = {chan_ids.shape}")

#     shared_output = module.model(
#         x=x_aug,
#         chan_ids=chan_ids,
#         mask_x=mask_x
#     )

#     print(f"[DEBUG] shared_output.shape = {shared_output.shape}")

#     # ------------------------------------------------------------
#     # Kiểm tra shape đúng với giả định của OriginalBYOLTask
#     # expected: [B, C, P, D] = [B, 19, 16, 512]
#     # ------------------------------------------------------------
#     assert shared_output.dim() == 4, \
#         f"Expected shared_output dim = 4, got shape {shared_output.shape}"

#     B, C, P, D = shared_output.shape
#     print(f"[DEBUG] Parsed shared_output dims:")
#     print(f"        B = {B}, C = {C}, P = {P}, D = {D}")

#     # ------------------------------------------------------------
#     # Xem online branch view 1 trước
#     # ------------------------------------------------------------
#     h_online_1 = byol_task._pool_representation(shared_output)
#     print(f"[DEBUG] h_online_1.shape = {h_online_1.shape}")

#     z_online_1 = byol_task.online_projector(h_online_1)
#     print(f"[DEBUG] z_online_1.shape = {z_online_1.shape}")

#     q_online_1 = byol_task.online_predictor(z_online_1)
#     print(f"[DEBUG] q_online_1.shape = {q_online_1.shape}")

#     # ------------------------------------------------------------
#     # Chạy full BYOL task
#     # ------------------------------------------------------------
#     out = byol_task(
#         shared_output=shared_output,
#         x=x,
#         x_aug=x_aug,
#         mask_x=mask_x,
#         mask_y=mask_y
#     )

#     if not isinstance(out, dict) or "loss" not in out:
#         raise ValueError("OriginalBYOLTask forward must return {'loss': ...}")

#     loss = out["loss"]
#     print(f"[DEBUG] BYOL loss = {loss.item():.6f}")
#     print("[DEBUG] Forward success.")
#     print("=" * 100)


# def main():
#     """
#     Main test để chạy riêng file eegpt_pretrain_module.py
#     """
#     # ============================================================
#     # 1) Config tối thiểu để build EEGPretrain + bật đúng Original BYOL
#     # ============================================================
#     cfg = OmegaConf.create({
#         "model": {
#             "img_size": [19, 3200],
#             "patch_size": 200,
#             "embed_dim": 512,
#             "embed_num": 4,
#             "num_heads": 8,
#             "encoder_depth": 8,
#             "predictor_depth": 2,
#             "reconstructor_depth": 2,
#             "mlp_ratio": 4.0,
#             "qkv_bias": True,
#             "drop_rate": 0.0,
#             "attn_drop_rate": 0.0,
#             "drop_path_rate": 0.0,
#             "init_std": 0.02,
#         },

#         # chỉ bật task này
#         "tasks": {
#             "byol_original": {
#                 "enabled": True,
#                 "proj_dim": 256,
#                 "hidden_dim": 4096,
#                 "tau": 0.996
#             }
#         },

#         # nếu BaseModule/setup_training của bạn cần optimizer/lr thì thêm vào
#         # nếu không cần thì có thể bỏ
#         "optimizer": {
#             "lr": 1e-4
#         }
#     })

#     # ============================================================
#     # 2) Build module
#     # ============================================================
#     device = "cpu"
#     print(f"[INFO] Using device: {device}")

#     module = EEGPretrain(cfg)
#     module = module.to(device)
#     module.eval()

#     print("[INFO] EEGPretrain module created successfully.")
#     print(f"[INFO] Active tasks: {list(module.tasks.keys())}")

#     # ============================================================
#     # 3) Tạo batch giả
#     # data sample của bạn có shape [19, 3200]
#     # batch -> [B, 19, 3200]
#     # ============================================================
#     B = 2
#     x = torch.randn(B, 19, 3200, device=device)

#     print

#     # ============================================================
#     # 4) Debug forward cho Original BYOL
#     # ============================================================
#     with torch.no_grad():
#         debug_forward_original_byol(module, x)


# if __name__ == "__main__":
#     main()