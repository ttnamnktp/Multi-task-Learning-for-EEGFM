# src/pretrain_module/cbramod/cbramod_pretrain.py
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.pretrain_module.base_module import BaseModule
from src.pretrain_module.registry import register_module
from src.models.registry import get_model
from src.models.cbramod.cbramod import CBraMod
from .task.reconstruction import ReconstructionTask
from .task.contrastive import ContrastiveTask
from .task.byol_nce import ContrastiveBYOLTask
from .task.byol_reg import BYOLRegTask
from .task.byol import OriginalBYOLTask


from .utils import augmentation, make_mask

# ============================================================================
# CONTEXT
# ============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class SharedForwardContext:
    x: torch.Tensor
    x_aug: torch.Tensor
    mask: Optional[torch.Tensor]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ForwardContext:
    shared: SharedForwardContext
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@register_module("cbramod_pretrain_module")
class CBraModPretrain(BaseModule):
    """CBraMod-specific pretraining module."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_training()  # Initialize model, tasks, and weight method
    
    def _build_model(self, cfg) -> nn.Module:
        """Build CBraMod model."""
        model_cfg = OmegaConf.to_container(
            cfg.model,
            resolve=True
        )
        self.model_cfg = model_cfg
        return CBraMod(**model_cfg)
    
    def _build_tasks(self, cfg, model_cfg=None) -> nn.ModuleDict:
        """Build CBraMod-specific tasks."""
        if model_cfg is None:
            model_cfg = self.model_cfg
        
        tasks = nn.ModuleDict()
        task_configs = cfg.get("tasks", {})
        
        # ==============================================================
        # Reconstruction task
        if task_configs.get("reconstruction", {}).get("enabled", False):
            tasks["reconstruction"] = ReconstructionTask(
                d_model=model_cfg["d_model"],
                out_dim=model_cfg["out_dim"]
            )
            print("[INFO] ✓ Reconstruction task enabled")
        
        # ==============================================================
        # Contrastive task
        if task_configs.get("contrastive", {}).get("enabled", False):
            contrastive_cfg = task_configs["contrastive"]
            tasks["contrastive"] = ContrastiveTask(
                model=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=contrastive_cfg.get("proj_dim", 128),
                temperature=contrastive_cfg.get("temperature", 0.5)
            )
            print("[INFO] ✓ Contrastive task enabled")
        
        total_steps = None
        try:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            max_epochs = self.trainer.max_epochs
            total_steps = int(steps_per_epoch * max_epochs) + 1
            print(f"[DEBUG] Total step: {total_steps}")
        except Exception:
            total_steps = 50000
            print(f"[DEBUG] Total step is set to {total_steps}")

        # ==============================================================
        # BYOL task
        if task_configs.get("byol", {}).get("enabled", False):
            byol_cfg = task_configs["byol"]
 
            tasks["byol"] = ContrastiveBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996),
                tau_end=byol_cfg.get("tau_end", 0.999),
                total_steps=total_steps,   
            )
            print("[INFO] ✓ BYOL task enabled")

        # ==============================================================
        # BYOL Reg task
        if task_configs.get("byol_reg", {}).get("enabled", False):
            byol_cfg = task_configs["byol_reg"]
 
            tasks["byol_reg"] = BYOLRegTask(
                online_encoder=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996),
                tau_end=byol_cfg.get("tau_end", 0.999),
                total_steps=total_steps,
            )
            print("[INFO] ✓ BYOL REG task enabled")

        # ==============================================================
        # Original BYOL Task
        if task_configs.get("byol_original", {}).get("enabled", False):
            byol_orig_cfg = task_configs["byol_original"]
            
            tasks["byol_original"] = OriginalBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=byol_orig_cfg.get("proj_dim", 256),
                hidden_dim=byol_orig_cfg.get("hidden_dim", 4096),  # Mặc định bài gốc là 4096
                tau_base=byol_orig_cfg.get("tau", 0.996),          # Mặc định bài gốc là 0.996
                total_steps=total_steps
            )
            print("[INFO] ✓ Original BYOL task enabled")
        
        if len(tasks) == 0:
            raise ValueError("At least one task must be enabled!")
        
        print(f"[INFO] Total active tasks: {len(tasks)}\n")
        return tasks
    
    def shared_step(self, ctx):
        """Compute all task losses for CBraMod."""
        x_aug = ctx.shared.x_aug
        mask = ctx.shared.mask
        
        # Forward pass
        shared_output = self.model(x_aug, mask=mask)
        
        # Compute task losses
        loss_dict = {}
        for task_name, task in self.tasks.items():
            task_output = task(shared_output, ctx)
            loss_dict[task_name] = task_output["loss"]
        
        return loss_dict
    
    # ==============================================================
    # BUILD CONTEXT
    # ==============================================================
    def build_shared_context(self, batch) -> SharedForwardContext:
        x = batch[0]
        x = x.view(x.size(0), x.size(1), -1, self.model_cfg["patch_size"])
        x_aug = augmentation(x)
        mask = make_mask(x, mask_ratio=0.4)

        return SharedForwardContext(
            x=x,
            x_aug=x_aug,
            mask=mask
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
            "in_dim": 200,
            "out_dim": 200,
            "d_model": 200,
            "dim_feedforward": 800,
            "n_layer": 12,
            "nhead": 8,
            "need_mask": True,
            "mask_ratio": 0.5,
            "patch_size": 200,
            "seq_len": 16,
        },
        "tasks": {
            "reconstruction": {"enabled": True},
            "contrastive":    {"enabled": False},
            "byol": {
                "enabled":     False,
                "proj_dim":    256,
                "hidden_dim":  512,
                "tau":         0.996,
                "tau_end":     0.999,
                "temperature": 0.1,
                "decoupled":   False,
            },
            "byol_reg":      {"enabled": False},
            "byol_original": {"enabled": True},
        }
    })

    device = "cpu"
    module = CBraModPretrain(cfg)
    module = module.to(device)
    module.eval()

    print(f"[INFO] Active tasks: {list(module.tasks.keys())}")

    # Tạo batch giả
    B = 2
    batch = (
        torch.randn(B, 19, 3200, device=device),
        torch.zeros(B, dtype=torch.long, device=device)
    )

    with torch.no_grad():
        # Build context — sampling augmentation + mask xảy ra ở đây
        ctx = module.build_forward_context(batch)

        print(f"[DEBUG] x.shape       = {ctx.shared.x.shape}")
        print(f"[DEBUG] x_aug.shape   = {ctx.shared.x_aug.shape}")
        print(f"[DEBUG] mask.shape  = {ctx.shared.mask.shape}")

        print("-" * 60)
        print("=== TASKS CONTEXT ===")
        if not ctx.tasks:
            print("[DEBUG] No task-specific contexts found.")
        else:
            for task_name, task_ctx in ctx.tasks.items():
                print(f"\n👉 Task: [{task_name}]")
                for key, val in task_ctx.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  └─ [DEBUG] {key:12s} : Tensor shape = {val.shape}, dtype = {val.dtype}")
                    else:
                        print(f"  └─ [DEBUG] {key:12s} : {val}")
        print("-" * 60)

        # Forward
        loss_dict = module.shared_step(ctx)

    print("=" * 60)
    for name, loss in loss_dict.items():
        print(f"[DEBUG] {name:20s} loss = {loss.item():.6f}")
    print("[DEBUG] Forward check passed.")


if __name__ == "__main__":
    main()