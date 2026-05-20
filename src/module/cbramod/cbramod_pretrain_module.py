import torch
import torch.nn.functional as F
import torch.nn as nn

from .config_builder import CBraModConfigBuilder
from src.module.base_module import BaseModule
from src.module.registry import register_module
from src.models.registry import get_model
from .task.reconstruction import ReconstructionTask
from .task.contrastive import ContrastiveTask

@register_module("cbramod_pretrain_module")
class CBraModPretrain(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

        builder = CBraModConfigBuilder()
        model_cfg = builder.build(cfg)

        model_cls = get_model(cfg.model.name)
        self.model = model_cls(**model_cfg)

        self.tasks = nn.ModuleDict({

            "reconstruction": ReconstructionTask(
                d_model=model_cfg["d_model"],
                out_dim=model_cfg["out_dim"]
            ),

            "contrastive": ContrastiveTask(
                model=self.model,
                d_model=model_cfg["d_model"],
                proj_dim=128,
                temperature=0.5
            ),
        })

        self.task_weights = {
            "reconstruction": 1.0,
            "contrastive": 1.0,
        }

    # =========================
    # shared step
    # =========================
    def shared_step(self, batch):

        shared_output = self.model(
            batch[0],
            return_mask=True
        )

        total_loss = 0
        loss_dict = {}

        for task_name, task in self.tasks.items():

            task_output = task(shared_output, batch)

            task_loss = task_output["loss"]

            weighted_loss = (
                self.task_weights[task_name]
                * task_loss
            )

            total_loss += weighted_loss

            loss_dict[f"{task_name}_loss"] = task_loss

        loss_dict["loss"] = total_loss

        return loss_dict

    # =========================
    # train
    # =========================
    def training_step(self, batch, batch_idx):

        loss_dict = self.shared_step(batch)

        for k, v in loss_dict.items():

            self.log(
                f"train/{k}",
                v,
                prog_bar=(k == "loss"),
                on_epoch=True,
                on_step=False
            )

        return loss_dict["loss"]

    # =========================
    # VALIDATION 
    # =========================
    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)
        loss = loss_dict["loss"]

        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    # =========================
    # optimizer
    # =========================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }