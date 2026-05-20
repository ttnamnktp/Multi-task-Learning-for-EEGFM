import torch.nn as nn

from .base_task import BaseTask

class ReconstructionTask(BaseTask):

    def __init__(self, d_model, out_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, out_dim)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, shared_output, batch):

        z = shared_output["latent"]
        mask = shared_output["mask"]

        x = batch[0]

        pred = self.decoder(z)

        if mask is not None:
            pred = pred[mask == 1]
            target = x[mask == 1]
        else:
            target = x

        loss = self.loss_fn(pred, target)

        return {
            "loss": loss,
            "pred": pred
        }