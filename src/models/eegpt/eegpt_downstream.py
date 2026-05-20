from src.models.base_model import BaseModel
from src.models.registry import register_model
from .eegpt import *
import torch.nn as nn
import torch

@register_model("eegpt_downstream")
class EEGPTDownstream(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_channels_num = cfg.dataset.use_chans_num
        self.use_channels_names = cfg.dataset.use_channels_names
        self.num_classes = cfg.dataset.num_classes
        self.time_points = cfg.dataset.time_points
        self.encoder = encoder = EEGTransformer(
            img_size=[self.use_channels_num, self.time_points],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.chans_id = encoder.prepare_chan_ids(self.use_channels_names)

        self.load_pretrained_encoder(cfg)

        self.adapter = nn.Sequential(
            nn.Conv1d(cfg.dataset.num_channels , self.use_channels_num, kernel_size=1)
        )

        self.linear_probe1 = nn.Linear(2048, 16)
        self.linear_probe2 = nn.Linear(16*16, self.num_classes)
        self.drop = torch.nn.Dropout(p=0.50)

    def load_pretrained_encoder(self, cfg):
        path = cfg.model.get("pretrained_ckpt", None)

        if path is None:
            print("No checkpoint provided. Random init encoder.")
            return

        try:
            ckpt = torch.load(path, map_location="cpu")

            self.encoder.load_state_dict(
                ckpt["state_dict"],
                strict=False
            )

            print(f"Loaded checkpoint: {path}")

        except FileNotFoundError:
            print(f"Checkpoint not found: {path}")
            print("Use random initialized encoder.")

        except Exception as e:
            raise RuntimeError(f"Failed loading checkpoint: {e}")

    def forward(self, x):

        feat = self.adapter(x)

        feat = self.encoder(feat, self.chans_id.to(feat))

        feat = feat.flatten(2)
        
        feat = self.linear_probe1(self.drop(feat))

        feat = feat.flatten(1)

        logits = self.linear_probe2(feat)

        return logits

    def get_param_groups(self):

        return [
            {
                "params": self.encoder.parameters(),
                "lr": self.cfg.optimizer.encoder_lr,
                "name": "encoder"
            },
            {
                "params": self.adapter.parameters(),
                "lr": self.cfg.optimizer.adapter_lr,
                "name": "adapter"
            },
            {
                "params": list(self.linear_probe1.parameters()) + list(self.linear_probe2.parameters()),
                "lr": self.cfg.optimizer.head_lr,
                "name": "head"
            }
        ]