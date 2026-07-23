import torch.nn as nn

from .base_task import BaseTask

class ReconstructionTask(BaseTask):

    def __init__(self, d_model, out_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.LayerNorm(d_model*4),
            nn.Linear(d_model*4, out_dim)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, shared_output, ctx):
        x = ctx.shared.x
        mask = ctx.shared.mask

        z = shared_output
        pred = self.decoder(z)

        masked_x = x[mask == 1]
        masked_y = pred[mask == 1]
        loss = self.loss_fn(masked_x, masked_y)
        
        return {
            "loss": loss,
            "pred": pred
        }