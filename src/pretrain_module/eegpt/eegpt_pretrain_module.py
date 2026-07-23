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

# ============================================================================
# CONTEXT
# ============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class SharedForwardContext:
    x: torch.Tensor
    x_aug: torch.Tensor
    mask_x: Optional[torch.Tensor]
    mask_y: Optional[torch.Tensor]
    chan_ids: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ForwardContext:
    shared: SharedForwardContext
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
    def shared_step(self, ctx):
        x_aug = ctx.shared.x_aug
        mask_x = ctx.shared.mask_x
        chan_ids = ctx.shared.chan_ids

        # Forward pass
        shared_output = self.model(x=x_aug, chan_ids=chan_ids, mask_x=mask_x)
        
        # Compute task losses
        loss_dict = {}
        for task_name, task in self.tasks.items():
            task_output = task(shared_output=shared_output, ctx=ctx) # truyền context cho task
            loss_dict[task_name] = task_output["loss"]
        
        return loss_dict
    
    # ==============================================================
    # BUILD CONTEXT
    # ==============================================================
    def build_shared_context(self, batch) -> SharedForwardContext:
        x, _ = batch
        x_aug = augmentation(x)
        mask_x, mask_y = make_masks(
            self.model.encoder.num_patches,
            p_n_y=0.4,
            p_c_y=0.2
        )

        return SharedForwardContext(
            x=x,
            x_aug=x_aug,
            mask_x=mask_x,
            mask_y=mask_y,
            chan_ids=self.model.chans_id.to(x_aug)
        )
    
    def build_forward_context(self, batch) -> ForwardContext:
        shared_ctx = self.build_shared_context(batch)

        task_ctxs = {}
        for task_name, task in self.tasks.items():
            task_ctxs[task_name] = task.build_task_context(
                batch=batch,
                shared_ctx=shared_ctx,
                module=self,
            )

        return ForwardContext(shared=shared_ctx, tasks=task_ctxs)

# Check forward
from omegaconf import OmegaConf

def main():
    cfg = OmegaConf.create({
        "model": {
            "img_size": [19, 3200],
            "patch_size": 200,
            "embed_dim": 512,
            "embed_num": 4,
            "num_heads": 8,
            "encoder_depth": 8,
            "predictor_depth": 2,
            "reconstructor_depth": 2,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "init_std": 0.02,
        },
        "tasks": {
            "reconstruction": {"enabled": True},
            "contrastive": {"enabled": False},
            "byol": {
                "enabled": True,
                "proj_dim": 256,
                "hidden_dim": 512,
                "tau": 0.996,
                "tau_end": 0.999,
                "temperature": 0.1,
                "decoupled": False,
            },
            "byol_reg": {"enabled": False},
            "byol_original": {"enabled": False},
        }
    })

    device = "cpu"

    module = EEGPretrain(cfg).to(device)

    print("\n[INFO] Active tasks:", list(module.tasks.keys()))
    print("[INFO] Model mode:", module.training)

    # =========================
    # Fake batch
    # =========================
    B = 2
    batch = (
        torch.randn(B, 19, 3200, device=device),
        torch.zeros(B, dtype=torch.long, device=device)
    )

    # =========================
    # Build context
    # =========================
    ctx = module.build_forward_context(batch)

    print("\n========== CONTEXT DEBUG ==========")
    print(f"x        : {ctx.shared.x.shape}")
    print(f"x_aug    : {ctx.shared.x_aug.shape}")
    print(f"mask_x   : {ctx.shared.mask_x.shape}")
    print(f"mask_y   : {ctx.shared.mask_y.shape}")
    print(f"chan_ids : {ctx.shared.chan_ids.shape}")

    # =========================
    # TASK CONTEXT DEBUG
    # =========================
    print("\n========== TASK CONTEXT ==========")
    for task_name, task_ctx in ctx.tasks.items():
        print(f"\n👉 Task: {task_name}")
        for k, v in task_ctx.items():
            if torch.is_tensor(v):
                print(f"  {k:15s}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"  {k:15s}: {v}")

    # =========================
    # FORWARD TEST
    # =========================
    print("\n========== FORWARD TEST ==========")

    module.eval()

    with torch.no_grad():

        loss_dict_eval = module.shared_step(ctx)

        print("\n[DEBUG] EVAL MODE LOSSES:")
        for k, v in loss_dict_eval.items():
            print(f"  {k:20s}: {v.item():.6f}")

    # =========================
    # TRAIN MODE TEST (IMPORTANT)
    # =========================
    print("\n========== TRAIN MODE TEST ==========")

    module.train()

    loss_dict_train = module.shared_step(ctx)

    print("\n[DEBUG] TRAIN MODE LOSSES:")
    for k, v in loss_dict_train.items():
        print(f"  {k:20s}: {v.item():.6f}")

    # =========================
    # COMPARE (CRITICAL FOR FAMO)
    # =========================
    print("\n========== TRAIN vs EVAL ==========")

    for k in loss_dict_train:
        print(
            f"{k:20s} | train={loss_dict_train[k].item():.6f} | eval={loss_dict_eval[k].item():.6f}"
        )

    # =========================
    # CHECK FOR COLLAPSE
    # =========================
    print("\n========== COLLAPSE CHECK ==========")

    for k, v in loss_dict_eval.items():
        if v.item() == 0:
            print(f"[WARNING] {k} eval loss = 0 ❌ (collapse)")
        elif abs(v.item() - loss_dict_train[k].item()) < 1e-6:
            print(f"[OK] {k} stable")
        else:
            print(f"[INFO] {k} differs train/eval → stochastic or BN effect")

    print("\n[DEBUG] Forward check finished.")

# def main():
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
#         "tasks": {
#             "reconstruction": {"enabled": True},
#             "contrastive":    {"enabled": False},
#             "byol": {
#                 "enabled":     False,
#                 "proj_dim":    256,
#                 "hidden_dim":  512,
#                 "tau":         0.996,
#                 "tau_end":     0.999,
#                 "temperature": 0.1,
#                 "decoupled":   False,
#             },
#             "byol_reg":      {"enabled": False},
#             "byol_original": {"enabled": True},
#         }
#     })

#     device = "cpu"
#     module = EEGPretrain(cfg)
#     module = module.to(device)
#     module.eval()

#     print(f"[INFO] Active tasks: {list(module.tasks.keys())}")

#     # Tạo batch giả
#     B = 2
#     batch = (
#         torch.randn(B, 19, 3200, device=device),
#         torch.zeros(B, dtype=torch.long, device=device)
#     )

#     with torch.no_grad():
#         # Build context — sampling augmentation + mask xảy ra ở đây
#         ctx = module.build_forward_context(batch)

#         print(f"[DEBUG] x.shape       = {ctx.shared.x.shape}")
#         print(f"[DEBUG] x_aug.shape   = {ctx.shared.x_aug.shape}")
#         print(f"[DEBUG] mask_x.shape  = {ctx.shared.mask_x.shape}")
#         print(f"[DEBUG] mask_y.shape  = {ctx.shared.mask_y.shape}")
#         print(f"[DEBUG] chan_ids.shape = {ctx.shared.chan_ids.shape}")

#         print("-" * 60)
#         print("=== TASKS CONTEXT ===")
#         if not ctx.tasks:
#             print("[DEBUG] No task-specific contexts found.")
#         else:
#             for task_name, task_ctx in ctx.tasks.items():
#                 print(f"\n👉 Task: [{task_name}]")
#                 for key, val in task_ctx.items():
#                     if isinstance(val, torch.Tensor):
#                         print(f"  └─ [DEBUG] {key:12s} : Tensor shape = {val.shape}, dtype = {val.dtype}")
#                     else:
#                         print(f"  └─ [DEBUG] {key:12s} : {val}")
#         print("-" * 60)

#         # Forward
#         loss_dict = module.shared_step(ctx)

#     print("=" * 60)
#     for name, loss in loss_dict.items():
#         print(f"[DEBUG] {name:20s} loss = {loss.item():.6f}")
#     print("[DEBUG] Forward check passed.")


if __name__ == "__main__":
    main()