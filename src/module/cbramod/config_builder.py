# module/cbramod/config_builder.py
import yaml
from omegaconf import OmegaConf

class CBraModConfigBuilder:

    def __init__(self):
        self.variants = self._load_variants()

    def _load_variants(self):
        with open("/home/infres/ttran-25/eegfm/configs/model/cbramod/variants.yaml", "r") as f:
            return yaml.safe_load(f)

    def build(self, cfg):
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

        variant = model_cfg.get("variant", "base")
        base_cfg = self.variants["variants"][variant].copy()

        # merge override
        for k, v in model_cfg.items():
            if v is not None:
                base_cfg[k] = v

        # derive values
        seq_len = base_cfg["seq_len"]

        return {
            # core model config
            "seq_len": seq_len,
            "in_dim": base_cfg["in_dim"],
            "out_dim": base_cfg["out_dim"],
            "d_model": base_cfg["d_model"],
            "dim_feedforward": base_cfg["dim_feedforward"],
            "n_layer": base_cfg["n_layer"],
            "nhead": base_cfg["nhead"],
            "need_mask": base_cfg.get("need_mask", True),
            "mask_ratio": base_cfg.get("mask_ratio", 0.5),
        }