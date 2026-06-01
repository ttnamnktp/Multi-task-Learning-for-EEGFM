from src.models.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_components import *

@register_model("cbramod")
class CBraMod(nn.Module):

    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=16,
        patch_size=200,
        n_layer=12,
        nhead=8,
        need_mask=True,
        mask_ratio=0.5,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.need_mask = need_mask
        self.mask_ratio = mask_ratio

        self.patch_embedding = PatchEmbedding(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            seq_len=seq_len
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )

        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )

        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        z = self.patch_embedding(x, mask)
        z = self.encoder(z)
        return z

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
