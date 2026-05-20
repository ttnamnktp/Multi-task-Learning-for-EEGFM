# src/weighting/base_weighting.py

from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseLossWeighting(ABC):
    """
    Base class for all loss weighting strategies
    """
    
    def __init__(self, loss_names, **kwargs):
        self.loss_names = loss_names
        self.weights = {k: 1.0 / len(loss_names) for k in loss_names}
    
    @abstractmethod
    def on_train_step(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int):
        """Called after each training step"""
        pass
    
    @abstractmethod
    def on_validation_step(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int):
        """Called after each validation step"""
        pass
    
    @abstractmethod
    def on_fit_start(self):
        """Called when training starts"""
        pass

    @abstractmethod
    def on_validation_epoch_end(self):
        """Called after validation epoch ends"""
        pass
    
    def get_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses"""
        total = sum(self.weights[k] * loss_dict[k] for k in self.loss_names)
        return total