# src/weighting/static_weighting.py

from src.weighting.base_weighting import BaseLossWeighting
from src.weighting.registry import register_weighting

@register_weighting("static")
class StaticWeighting(BaseLossWeighting):
    """
    Fixed weights - no adaptation
    """
    
    def __init__(self, loss_names, weights=None, **kwargs):
        super().__init__(loss_names)
        
        if weights is not None:
            assert set(weights.keys()) == set(loss_names)
            self.weights = weights
    
    def on_train_step(self, loss_dict, batch_idx):
        pass
    
    def on_validation_step(self, loss_dict, batch_idx):
        pass
    
    def on_fit_start(self):
        pass
    
    def on_validation_epoch_end(self):
        pass