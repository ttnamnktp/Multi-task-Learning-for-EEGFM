from src.module.base_module import BaseModule
from src.models.registry import get_model
import torch.nn.functional as F
from src.module.registry import register_module

@register_module("classification")
class ClassificationModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        model_cls = get_model(cfg.model.name)
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)