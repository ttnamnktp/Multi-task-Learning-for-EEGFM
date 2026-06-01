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
from .task.byol import ContrastiveBYOLTask
from .utils import augmentation, make_mask

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
        
        # Reconstruction task
        if task_configs.get("reconstruction", {}).get("enabled", False):
            tasks["reconstruction"] = ReconstructionTask(
                d_model=model_cfg["d_model"],
                out_dim=model_cfg["out_dim"]
            )
            print("[INFO] ✓ Reconstruction task enabled")
        
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
        
        # BYOL task
        if task_configs.get("byol", {}).get("enabled", False):
            byol_cfg = task_configs["byol"]
            tasks["byol"] = ContrastiveBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996)
            )
            print("[INFO] ✓ BYOL task enabled")
        
        if len(tasks) == 0:
            raise ValueError("At least one task must be enabled!")
        
        print(f"[INFO] Total active tasks: {len(tasks)}\n")
        return tasks
    
    def shared_step(self, batch):
        """Compute all task losses for CBraMod."""
        x = batch[0]
        x = x.view(x.size(0), x.size(1), -1, self.model_cfg["patch_size"])

        # CBraMod-specific preprocessing
        x_aug = augmentation(x)
        mask = make_mask(x, mask_ratio=0.5)

        print(f"mask shape: {mask.shape}, x shape: {x.shape}, x_aug shape: {x_aug.shape}")
        
        # Forward pass
        shared_output = self.model(x_aug, mask=mask)
        
        # Compute task losses
        loss_dict = {}
        for task_name, task in self.tasks.items():
            task_output = task(shared_output, x, mask)
            loss_dict[task_name] = task_output["loss"]
        
        return loss_dict