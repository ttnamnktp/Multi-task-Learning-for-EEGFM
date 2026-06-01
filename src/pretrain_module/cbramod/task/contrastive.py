import torch.nn as nn
import torch.nn.functional as F
from .base_task import BaseTask

class ContrastiveTask(BaseTask):

    def __init__(self, model, d_model, proj_dim=128, temperature=0.5):
        super().__init__()

        self.model = model
        self.projector = ProjectionHead(d_model, proj_dim)
        self.loss_fn = NTXentLoss(temperature)

    def forward(self, shared_output, batch):

        # 1. create 2 views
        x1 = batch[0]
        x2 = augmentation(batch[0])

        # 2. forward backbone
        h1 = self.model(x1, return_mask=False)["latent"]
        h2 = self.model(x2, return_mask=False)["latent"]

        # 3. pooling
        h1 = h1.mean(dim=1)
        h2 = h2.mean(dim=1)

        # 4. projection
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # 5. loss
        loss = self.loss_fn(z1, z2)

        return {"loss": loss}

class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D]
        """
        B = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z1 = z1.mean(dim=1)
        z2 = z2.mean(dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim = torch.matmul(z, z.T) / self.t  # [2B, 2B]

        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        pos = torch.cat([
            torch.arange(B, 2*B),
            torch.arange(0, B)
        ]).to(z.device)

        loss = F.cross_entropy(sim, pos)
        return loss

