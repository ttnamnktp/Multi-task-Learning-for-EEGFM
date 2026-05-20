import torch.nn.functional as F
import torch

from src.module.registry import register_module
from src.module.base_module import BaseModule
from src.models.registry import get_model
from src.weighting.registry import get_weighting

from .config_builder import EEGPTConfigBuilder
from .gradient_monitor import GradientMonitor

@register_module("eegpt_pretrain_module")
class EEGPretrain(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

        # ===== model =====
        config_builder = EEGPTConfigBuilder() # resolve config for model
        model_cfg = config_builder.build(cfg)

        model_cls = get_model(cfg.model.name)
        self.model = model_cls(model_cfg)

        # ===== loss weighting strategy =====
        weighting_cls = get_weighting(cfg.weighting.name)
        self.loss_weighting = weighting_cls(
            loss_names=["loss1", "loss2"],
            **cfg.weighting.get("params", {})
        )

    # ==============================================================
    # Shared step
    # ==============================================================
    def shared_step(self, batch):
        x, _ = batch

        mask_x, mask_y = self.model.make_masks(
            self.model.encoder.num_patches
        )

        h, y = self.model.forward_target(x, mask_y)
        z, r = self.model.forward_context(x, mask_x, mask_y)

        loss_dict = self._compute_raw_losses(h, z, y, r)
        
        return loss_dict

    # ==============================================================
    # Loss
    # ==============================================================
    def _compute_raw_losses(self, h, z, y, r):
        """Compute individual losses WITHOUT weighting"""
        loss1 = F.mse_loss(h, z)
        loss2 = F.mse_loss(y, r)

        return {
            "loss1": loss1,
            "loss2": loss2,
        }
    
    # ==============================================================
    # Train / Val
    # ==============================================================
    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)

        # Let weighting strategy handle updates
        self.loss_weighting.on_train_step(loss_dict, batch_idx)

        # Compute final weighted loss
        total_loss = self.loss_weighting.get_weighted_loss(loss_dict)
        loss_dict["loss"] = total_loss

        # Log loss
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_epoch=True, on_step=False, sync_dist=True
        )
        # Log gradient conflicts
        if batch_idx % 50 == 0:
            self.grad_monitor.log_conflict(
                loss_dict["loss1"],
                loss_dict["loss2"]
            )

        # Log current weights
        for k, w in self.loss_weighting.weights.items():
            self.log(f"weight_{k}", w, on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)

        # Let weighting strategy accumulate val stats
        self.loss_weighting.on_validation_step(loss_dict, batch_idx)

        total_loss = self.loss_weighting.get_weighted_loss(loss_dict)
        loss_dict["loss"] = total_loss

        self.log_dict(
            {f"valid_{k}": v for k, v in loss_dict.items()},
            on_epoch=True, on_step=False, sync_dist=True
        )
        return loss_dict["loss"]

    # ==============================================================
    # EMA update
    # ==============================================================
    def on_fit_start(self):
        self.grad_monitor = GradientMonitor(self)

        self.loss_weighting.on_fit_start()

        total_steps = self.trainer.estimated_stepping_batches
        self.momentum_scheduler = (
            0.996 + i * (1.0 - 0.996) / total_steps
            for i in range(total_steps + 1)
        )

    def on_train_batch_end(self, *args):
        with torch.no_grad():
            m = next(self.momentum_scheduler)

            for q, k in zip(self.model.encoder.parameters(),
                            self.model.target_encoder.parameters()):
                k.data.mul_(m).add_((1 - m) * q.detach())

    def on_validation_epoch_end(self):
        """Trigger weight update after validation epoch completes"""
        self.loss_weighting.on_validation_epoch_end()

    # ==============================================================
    # Optimizer
    # ==============================================================
    def configure_optimizers(self):

        def is_decay(n, p):
            return not (("bias" in n) or (len(p.shape) == 1))

        decay, no_decay = [], []

        for module in [self.model.encoder,
                       self.model.predictor,
                       self.model.reconstructor]:

            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (decay if is_decay(n, p) else no_decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-2},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=6e-5
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.cfg.trainer.max_epochs,
            pct_start=0.2,
            div_factor=2,
            final_div_factor=8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
