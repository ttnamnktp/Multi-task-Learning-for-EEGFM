import torch.nn.functional as F
import torch
import torch.nn as nn

from src.pretrain_module.registry import register_module
from src.pretrain_module.base_module import BaseModule
from src.models.registry import get_model
from src.models.eegpt.eegpt import EEGPTModel
from .task.reconstruction import ReconstructionTask
from .task.contrastive import ContrastiveTask
from .task.byol import ContrastiveBYOLTask
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

        # Reconstruction task
        if task_configs.get("reconstruction", {}).get("enabled", False):
            tasks["reconstruction"] = ReconstructionTask(
                online_encoder=self.model,
                models_configs=self.model_cfg
            )
            print("[INFO] ✓ Reconstruction task enabled")
        
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
        
        # BYOL task
        if task_configs.get("byol", {}).get("enabled", False):
            byol_cfg = task_configs["byol"]
            try:
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                max_epochs = self.trainer.max_epochs
                total_steps = int(steps_per_epoch * max_epochs) + 1
            except Exception:
                total_steps = None
 
            tasks["byol"] = ContrastiveBYOLTask(
                online_encoder=self.model,
                d_model=model_cfg["embed_dim"],
                proj_dim=byol_cfg.get("proj_dim", 256),
                hidden_dim=byol_cfg.get("hidden_dim", 512),
                tau=byol_cfg.get("tau", 0.996),
                tau_end=byol_cfg.get("tau_end", 0.999),   # FIX Bug: thêm tham số mới
                total_steps=total_steps,                   # FIX Bug: truyền total_steps
            )
            print("[INFO] ✓ BYOL task enabled")
        
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
