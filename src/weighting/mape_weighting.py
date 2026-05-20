# src/weighting/mape_weighting.py

import torch
from src.weighting.base_weighting import BaseLossWeighting
from src.weighting.registry import register_weighting

@register_weighting("mape")
class MAPEWeighting(BaseLossWeighting):
    """
    MAPE-based adaptive loss weighting
    """

    def __init__(self, loss_names, eps=1e-8, **kwargs):
        super().__init__(loss_names)
        
        self.eps = eps

        # statistics
        self.init_train = {}
        self.init_val = {}
        self.train_buf = {k: [] for k in loss_names}
        self.val_buf = {k: [] for k in loss_names}

        # control
        self.global_step = 0
        self.update_step = 0
        self._initialized = False

    def on_fit_start(self):
        """Reset state at start of training"""
        self._initialized = False
        self.global_step = 0

    def on_train_step(self, loss_dict, batch_idx):
        """Update statistics and initialize if needed"""
        # Initialize on first step
        if not self._initialized:
            self._set_initial_losses(loss_dict, is_train=True)
            self._initialized = True
        
        # Log losses
        self.global_step += 1
        for k in self.loss_names:
            self.train_buf[k].append(loss_dict[k].detach())

    def on_validation_step(self, loss_dict, batch_idx):
        """Accumulate validation losses"""
        # Initialize validation baseline if needed
        if not self.init_val:
            self._set_initial_losses(loss_dict, is_train=False)
        
        for k in self.loss_names:
            self.val_buf[k].append(loss_dict[k].detach())

    def on_validation_epoch_end(self):
        """Update weights after validation epoch"""
        if len(self.val_buf[self.loss_names[0]]) > 0:  # Check if we have validation data
            self._update_weights()

    def _set_initial_losses(self, loss_dict, is_train=True):
        """Initialize L_i(0) for train or validation"""
        target_dict = self.init_train if is_train else self.init_val
        for k in self.loss_names:
            target_dict[k] = loss_dict[k].detach()

    def _should_update(self):
        """Check if we have enough data to update"""
        # Check both train and val buffers have data
        has_train = len(self.train_buf[self.loss_names[0]]) > 0
        has_val = len(self.val_buf[self.loss_names[0]]) > 0
        return has_train and has_val

    def _compute_statistics(self):
        """Compute overfitting (O) and generalization (G) metrics"""
        O, G = {}, {}

        # compute for each loss
        for k in self.loss_names:

            Lt_raw = torch.stack(self.train_buf[k]).mean()
            Lv_raw = torch.stack(self.val_buf[k]).mean()
            # normalize
            Lt = (Lt_raw) / (self.init_train[k] + self.eps)
            Lv = (Lv_raw) / (self.init_val[k] + self.eps)

            O[k] = (1 - Lt) - (1 - Lv)
            G[k] = (1 - Lv)

        return O, G

    def _update_weights(self):
        """MAPE weight update rule"""
        if not self._should_update():
            return
        
        O, G = self._compute_statistics()
        O_vec = torch.stack([O[k] for k in self.loss_names])
        G_vec = torch.stack([G[k] for k in self.loss_names])

        A = torch.outer(O_vec, O_vec) + 0.1 * torch.eye(len(O_vec), device=O_vec.device)
        w_vec = 0.5 * torch.linalg.solve(A, G_vec)

        # Ensure non-negative weights
        w_vec = torch.clamp(w_vec, min=0.0)

        # Normalize
        total = torch.sum(w_vec)
        new_w = {}
        for i, k in enumerate(self.loss_names):
            new_w[k] = (w_vec[i] / (total + self.eps)).item()
        self.weights = new_w

        # Reset buffers (keep initial losses for next epoch)
        self.train_buf = {k: [] for k in self.loss_names}
        self.val_buf = {k: [] for k in self.loss_names}
        self.update_step += 1