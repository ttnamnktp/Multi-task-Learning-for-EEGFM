from src.models.base_model import BaseModel
from src.models.registry import register_model

import torch
import torch.nn as nn

from .model_components import *
from .cbramod import *

@register_model("cbramod_downstream")
class CBraModDownstream(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_channels_num = cfg.dataset.use_chans_num
        self.num_classes = cfg.dataset.num_classes
        self.time_points = cfg.dataset.time_points

        self.patch_size = cfg.model.patch_size
        self.seq_len = cfg.dataset.time_points // cfg.model.d_model

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------
        self.encoder = CBraMod(
            in_dim=cfg.model.in_dim,
            out_dim=cfg.model.out_dim,
            d_model=cfg.model.d_model,
            dim_feedforward=cfg.model.dim_feedforward,
            seq_len=self.seq_len,
            n_layer=cfg.model.n_layer,
            nhead=cfg.model.nhead,
        )

        self.load_pretrained_encoder(cfg)

        # --------------------------------------------------
        # Channel adapter
        # Same logic as LitCBraMod
        # [B,C,S,P]
        #   -> reshape
        # [B,C,S*P]
        #   -> Conv1d
        # [B,C_model,S*P]
        # --------------------------------------------------
        self.adapter = nn.Conv1d(
            cfg.dataset.num_channels,
            self.use_channels_num,
            kernel_size=1,
            bias=True
        )

        # --------------------------------------------------
        # Classification head
        # Encoder output:
        # [B,C_model,S,D]
        # --------------------------------------------------
        self.linear_probe1 = nn.Linear(
            cfg.model.d_model,
            16
        )

        self.linear_probe2 = nn.Linear(
            16 * self.use_channels_num * self.seq_len,
            self.num_classes
        )

        self.drop = nn.Dropout(p=0.5)

    # ======================================================
    # Pretrained loading
    # ======================================================

    def load_pretrained_encoder(self, cfg):

        path = cfg.model.pretrained_ckpt

        if path is None:
            print("[ENC LOAD] No checkpoint provided → random init")
            return

        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            model_state = self.encoder.state_dict()

            # --- ĐOẠN CODE THÊM VÀO ĐỂ DEBUG ---
            print("\n🔍 --- DEBUG KEY MISMATCH ---")
            print(f"Ví dụ 3 keys đầu tiên trong CHECKPOINT:")
            print(list(state.keys())[:3])
            print(f"Ví dụ 3 keys đầu tiên trong MODEL HIỆN TẠI (self.encoder):")
            print(list(model_state.keys())[:3])
            print("---------------------------------\n")
            # -----------------------------------

            prefix = "model."
            enc_state = {}
            not_in_model = 0
            shape_mismatch = 0

            for k, v in state.items():

                if not isinstance(k, str):
                    continue
                if not k.startswith(prefix):
                    continue

                k = k[len(prefix):]

                if k not in model_state:
                    not_in_model += 1
                    continue

                if model_state[k].shape != v.shape:
                    shape_mismatch += 1
                    continue

                enc_state[k] = v

            if len(enc_state) == 0:
                print("[ENC LOAD] ❌ No compatible parameters found")
                return

            missing, unexpected = self.encoder.load_state_dict(
                enc_state,
                strict=False
            )

            coverage = (
                100.0 * len(enc_state)
                / len(model_state)
            )

            print("\n========== [ENCODER LOAD] ==========")
            print(f"Checkpoint: {path}")
            print(
                f"Loaded: {len(enc_state)}/{len(model_state)} "
                f"({coverage:.2f}%)"
            )
            print(f"Not in model: {not_in_model}")
            print(f"Shape mismatch: {shape_mismatch}")
            print(f"Missing (torch): {len(missing)}")
            print(f"Unexpected (torch): {len(unexpected)}")
            print("====================================\n")

        except FileNotFoundError:
            print(f"[ENC LOAD] ❌ Checkpoint not found: {path}")

        except Exception as e:
            raise RuntimeError(
                f"[ENC LOAD] Failed loading checkpoint: {e}"
            )

    # ======================================================
    # Channel Adapter
    # ======================================================

    def apply_channel_adapter(self, x):

        """
        Input:
            [B,C,S,P]

        Output:
            [B,C_model,S,P]
        """

        B, C, S, P = x.shape

        x = x.reshape(B, C, S * P)

        x = self.adapter(x)

        x = x.reshape(
            B,
            self.use_channels_num,
            S,
            P
        )

        return x

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x):

        """
        x:
            [B,C,T]
        """

        B, C, T = x.shape

        assert (
            T % self.patch_size == 0
        ), (
            f"time_points={T} "
            f"is not divisible by "
            f"patch_size={self.patch_size}"
        )

        # -----------------------------------------
        # [B,C,T]
        # ->
        # [B,C,S,P]
        # -----------------------------------------
        feat = x.view(
            B,
            C,
            -1,
            self.patch_size
        )

        # -----------------------------------------
        # Channel adapter
        # -----------------------------------------
        feat = self.apply_channel_adapter(feat)

        # -----------------------------------------
        # CBraMod Encoder
        # Expected:
        # [B,C_model,S,D]
        # -----------------------------------------
        feat = self.encoder(feat)

        # -----------------------------------------
        # Linear Probe
        # [B,C,S,D]
        # ->
        # [B,C,S,16]
        # -----------------------------------------
        feat = self.linear_probe1(
            self.drop(feat)
        )

        # -----------------------------------------
        # Flatten
        # [B,C,S,16]
        # ->
        # [B,C*S*16]
        # -----------------------------------------
        feat = feat.flatten(1)

        logits = self.linear_probe2(feat)

        return logits

    # ======================================================
    # Optimizer Groups
    # ======================================================

    def get_param_groups(self):

        return [
            {
                "params": self.encoder.parameters(),
                "lr": self.cfg.optimizer.encoder_lr,
                "name": "encoder",
            },
            {
                "params": self.adapter.parameters(),
                "lr": self.cfg.optimizer.adapter_lr,
                "name": "adapter",
            },
            {
                "params":
                    list(self.linear_probe1.parameters())
                    + list(self.linear_probe2.parameters()),
                "lr": self.cfg.optimizer.head_lr,
                "name": "head",
            },
        ]

def main():
    # --------------------------------------------------
    # Fake config (bạn có thể thay bằng cfg thật nếu muốn)
    # --------------------------------------------------
    class CFG:
        class dataset:
            use_chans_num = 32
            num_classes = 5
            # time_points = 800 #3200
            time_points = 1000
            num_channels = 19 #32

        class model:
            patch_size = 200
            # seq_len = 16 #16
            in_dim = 200
            out_dim = 200
            d_model = 200
            dim_feedforward = 800
            n_layer = 12
            nhead = 8
            pretrained_ckpt = "/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/857003/2026-06-19/17-11-30/checkpoints/best-epoch=28-valid/average_valid_loss=0.9141.ckpt"  # set path nếu có checkpoint

        class optimizer:
            encoder_lr = 1e-4
            adapter_lr = 1e-4
            head_lr = 1e-3

    cfg = CFG()

    # --------------------------------------------------
    # Init model
    # --------------------------------------------------
    model = CBraModDownstream(cfg)

    model.eval()

    # --------------------------------------------------
    # Build fake EEG input
    # shape: [B, C, T]
    # --------------------------------------------------
    B = 2
    C = cfg.dataset.num_channels
    T = cfg.dataset.time_points  # 200 * 16 = 3200

    x = torch.randn(B, C, T)

    print("\n==============================")
    print("INPUT SHAPE:", x.shape)
    print("==============================\n")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(x)

    print("\n==============================")
    print("OUTPUT SHAPE:", logits.shape)
    print("==============================\n")

    print("Forward pass SUCCESS ✅")


if __name__ == "__main__":
    main()