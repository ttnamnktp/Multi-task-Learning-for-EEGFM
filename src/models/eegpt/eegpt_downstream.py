from src.models.base_model import BaseModel
from src.models.registry import register_model
from .eegpt import *
import torch.nn as nn
import torch
from functools import partial

@register_model("eegpt_downstream")
class EEGPTDownstream(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_channels_num = cfg.dataset.use_chans_num
        self.use_channels_names = cfg.dataset.use_channels_names
        self.num_classes = cfg.dataset.num_classes
        self.time_points = cfg.dataset.time_points

        # =========================================================
        # Encoder
        # =========================================================
        self.encoder = encoder = EEGTransformer(
            img_size=[self.use_channels_num, self.time_points],
            patch_size=self.time_points // cfg.model.get("patch_num", 16),
            embed_num=cfg.model.get("embed_num", 4),
            embed_dim=cfg.model.get("embed_dim", 512),
            depth=cfg.model.get("encoder_depth", 8),
            num_heads=cfg.model.get("num_heads", 8),
            mlp_ratio=cfg.model.get("mlp_ratio", 4.0),
            drop_rate=cfg.model.get("drop_rate", 0.0),
            attn_drop_rate=cfg.model.get("attn_drop_rate", 0.0),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
            init_std=cfg.model.get("init_std", 0.02),
            qkv_bias=cfg.model.get("qkv_bias", True),
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.chans_id = encoder.prepare_chan_ids(self.use_channels_names)

        # =========================================================
        # Lưu các parameter của encoder bị missing khi load ckpt
        # - phần này sẽ dùng adapter_lr
        # =========================================================
        self.missing_encoder_keys = set()

        # load pretrained encoder
        self.load_pretrained_encoder(cfg)

        # =========================================================
        # Adapter + Head
        # =========================================================
        self.adapter = nn.Sequential(
            nn.Conv1d(cfg.dataset.num_channels , self.use_channels_num, kernel_size=1)
        )

        self.linear_probe1 = nn.Linear(cfg.model.get("embed_dim", 512) * cfg.model.get("embed_num", 4), 16)
        self.linear_probe2 = nn.Linear(16*cfg.model.get("patch_num", 16), self.num_classes)
        self.drop = torch.nn.Dropout(p=0.50)

        # =========================================================
        # Nếu encoder_lr = 0:
        # chỉ freeze phần encoder đã load từ checkpoint
        # còn phần missing vẫn train bằng adapter_lr
        # =========================================================
        if cfg.get("optimizer", {}).get("encoder_lr", 0) == 0:
            print("[INFO] Encoder LR is 0. Freezing ONLY pretrained-loaded encoder parameters...")

            frozen_count = 0
            trainable_missing_count = 0

            for name, param in self.encoder.named_parameters():
                if name not in self.missing_encoder_keys:
                    # param này đã load từ checkpoint -> freeze
                    param.requires_grad = False
                    frozen_count += param.numel()
                else:
                    # param này missing -> vẫn train
                    param.requires_grad = True
                    trainable_missing_count += param.numel()
        
            print(f"[INFO] Frozen pretrained encoder params: {frozen_count:,}")
            print(f"[INFO] Trainable missing encoder params: {trainable_missing_count:,}")


    def load_pretrained_encoder(self, cfg):
        """
        Load pretrained weights cho encoder.

        Logic:
        - chỉ lấy key có prefix 'model.encoder.'
        - chỉ load key tồn tại trong encoder hiện tại
        - chỉ load nếu shape khớp
        - các key missing sau load_state_dict(strict=False) sẽ được lưu vào
          self.missing_encoder_keys để train bằng adapter_lr
        """

        path = cfg.model.pretrained_ckpt

        # ---------------------------------------------------------
        # Không có checkpoint:
        # -> không có khái niệm "missing so với pretrained"
        # -> để missing_encoder_keys = rỗng
        # -> toàn bộ encoder sẽ dùng encoder_lr
        # ---------------------------------------------------------
        if path is None:
            print("[ENC LOAD] No checkpoint provided → random init")
            self.missing_encoder_keys = set()
            return

        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            prefix = "model.encoder."
            model_state = self.encoder.state_dict()

            enc_state = {}
            not_in_model = 0
            shape_mismatch = 0

            for k, v in state.items(): # Kiểm tra các điều kiện để load ckpt vào encoder
                if not isinstance(k, str):
                    continue
                if not k.startswith(prefix):
                    continue

                key = k[len(prefix):]

                # 1. check existence
                if key not in model_state:
                    not_in_model += 1
                    continue

                # 2. check shape
                if model_state[key].shape != v.shape:
                    shape_mismatch += 1
                    continue

                enc_state[key] = v

            if len(enc_state) == 0:
                print("[ENC LOAD] ❌ No compatible parameters found")
                print(f"[DEBUG] not_in_model={not_in_model}, shape_mismatch={shape_mismatch}")
                print("[ENC LOAD] Fallback: encoder stays random init -> train all encoder params with encoder_lr")

                # Không load được gì => coi như encoder random init toàn bộ
                # => train toàn bộ bằng encoder_lr
                self.missing_encoder_keys = set()
                return

            missing, unexpected = self.encoder.load_state_dict(
                enc_state,
                strict=False
            )

            # -----------------------------------------------------
            # Chỉ giữ lại những key missing thuộc ENCODER
            # vì load_state_dict có thể trả cả buffer
            # -----------------------------------------------------
            encoder_param_names = {name for name, _ in self.encoder.named_parameters()}
            self.missing_encoder_keys = {
                key for key in missing if key in encoder_param_names
            }

            model_state = self.encoder.state_dict()
            coverage = 100.0 * len(enc_state) / len(model_state)

            print("\n========== [ENCODER LOAD] ==========")
            print(f"Checkpoint: {path}")
            print(f"Loaded: {len(enc_state)}/{len(model_state)} ({coverage:.2f}%)")
            print(f"Not in model: {not_in_model}")
            print(f"Shape mismatch: {shape_mismatch}")
            print(f"Missing (torch): {len(missing)}")
            print(f"Missing encoder PARAMETERS: {len(self.missing_encoder_keys)}")
            print(f"Unexpected (torch): {len(unexpected)}")

            if len(self.missing_encoder_keys) > 0:
                print("\n[Missing encoder param sample]")
                print(list(self.missing_encoder_keys)[:10])

            print("====================================\n")

        except FileNotFoundError:
            print(f"[ENC LOAD] ❌ Checkpoint not found: {path}")
            print("[ENC LOAD] Fallback: encoder stays random init -> train all encoder params with encoder_lr")

        except Exception as e:
            raise RuntimeError(f"[ENC LOAD] Failed loading checkpoint: {e}")

    
    def forward(self, x):

        feat = self.adapter(x)

        feat = self.encoder(feat, self.chans_id.to(feat))

        feat = feat.flatten(2)
        
        feat = self.linear_probe1(self.drop(feat))

        feat = feat.flatten(1)

        logits = self.linear_probe2(feat)

        return logits

    def get_param_groups(self):
        """
        Trả về param groups cho optimizer:

        1) encoder_loaded  -> encoder_lr
        2) encoder_missing -> adapter_lr
        3) adapter         -> adapter_lr
        4) head            -> head_lr

        Lưu ý:
        - nếu encoder_lr = 0 thì phần loaded encoder đã bị freeze,
          nên sẽ không xuất hiện trong optimizer vì requires_grad=False.
        """

        loaded_encoder_params = []
        missing_encoder_params = []

        for name, param in self.encoder.named_parameters():
            if not param.requires_grad: # nếu freeze learning rate encoder
                continue 

            if name in self.missing_encoder_keys: # nếu KHÔNG freeze learning rate encoder và param mismatch shape/không loaded từ checkpoint
                missing_encoder_params.append(param)
            else: # nếu KHÔNG freeze learning rate encoder và encoder params được loaded từ checkpoint
                loaded_encoder_params.append(param)

        param_groups = [] # return this

        # ---------------------------------------------------------
        # 1) Encoder params đã load từ checkpoint
        # ---------------------------------------------------------
        if len(loaded_encoder_params) > 0: # nếu KHÔNG freeze learning rate encoder và encoder params được loaded từ checkpoint
            param_groups.append({
                "params": loaded_encoder_params,
                "lr": self.cfg.optimizer.encoder_lr,
                "name": "encoder_loaded"
            })

        # ---------------------------------------------------------
        # 2) Encoder params missing -> train bằng adapter_lr
        # ---------------------------------------------------------
        if len(missing_encoder_params) > 0: # nếu KHÔNG freeze learning rate encoder và param mismatch shape/không loaded từ checkpoint
            param_groups.append({
                "params": missing_encoder_params,
                "lr": self.cfg.optimizer.adapter_lr,
                "name": "encoder_missing"
            })

        # ---------------------------------------------------------
        # 3) Adapter
        # ---------------------------------------------------------
        adapter_params = [p for p in self.adapter.parameters() if p.requires_grad]
        if len(adapter_params) > 0:
            param_groups.append({
                "params": adapter_params,
                "lr": self.cfg.optimizer.adapter_lr,
                "name": "adapter"
            })

        # ---------------------------------------------------------
        # 4) Head
        # ---------------------------------------------------------
        head_params = (
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters())
        )
        head_params = [p for p in head_params if p.requires_grad]

        if len(head_params) > 0:
            param_groups.append({
                "params": head_params,
                "lr": self.cfg.optimizer.head_lr,
                "name": "head"
            })

        # ---------------------------------------------------------
        # Debug print
        # ---------------------------------------------------------
        print("\n========== [PARAM GROUPS] ==========")
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            print(f"{group['name']:>16s} | lr = {group['lr']} | params = {n_params:,}")
        print("====================================\n")

        return param_groups