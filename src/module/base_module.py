import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from src.utils.metrics import accuracy
from src.utils.optimizers import build_optimizer
from src.utils.schedulers import build_scheduler

class BaseModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def compute_loss(self, logits, y):
        raise NotImplementedError

    def build_metrics(self):
        return None

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # Log learning rates
        opt = self.optimizers()
        for i, group in enumerate(opt.param_groups):
            name = group.get("name", f"group_{i}") # if forget naming, auto name it group
            self.log(f"lr_{name}", group["lr"], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):

        param_groups = self.get_param_groups()
        optimizer = build_optimizer(self.cfg, param_groups)
        scheduler = build_scheduler(self.cfg, optimizer)

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def get_param_groups(self):
        return self.model.get_param_groups()