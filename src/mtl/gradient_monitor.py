from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


class MTLGradientMonitor:
    """
    Monitor for MTL diagnostics.

    Two groups of metrics:
    1) Gradient metrics: computed on training steps from gradients
    2) Loss dynamics metrics: computed on epoch-averaged losses
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        log_every_n: int = 50,
    ):
        self.ema_alpha = ema_alpha
        self.log_every_n = log_every_n

        # Gradient logging step counter is no longer necessary for loss dynamics,
        # but can still be kept if you want.
        self._step = 0

        # ===== Epoch-level loss dynamics state =====
        self._prev_epoch_loss: Dict[str, float] = {}
        self._initial_epoch_loss: Dict[str, float] = {}
        self._ema_epoch_rate: Dict[str, float] = {}
        self._ema_epoch_speed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Epoch-level loss dynamics
    # ------------------------------------------------------------------

    def compute_epoch_loss_dynamics(
        self,
        epoch_loss_dict: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss dynamics using epoch-averaged task losses.

        Args:
            epoch_loss_dict:
                Dict[str, float], e.g.
                {
                    "task_a": 0.523,
                    "task_b": 1.214,
                }

        Returns:
            Dict[str, torch.Tensor]
        """
        metrics: Dict[str, torch.Tensor] = {}

        if len(epoch_loss_dict) == 0:
            return metrics

        # tensor of epoch mean losses
        loss_values = torch.tensor(
            list(epoch_loss_dict.values()),
            dtype=torch.float32
        )

        # --------------------------------------------------
        # global imbalance
        # --------------------------------------------------
        metrics["loss_dynamics/variance"] = loss_values.var()

        mean_loss = float(loss_values.mean())

        # --------------------------------------------------
        # GradNorm-style training rates
        # --------------------------------------------------
        training_rates: Dict[str, float] = {}

        for name, val in epoch_loss_dict.items():
            if name not in self._initial_epoch_loss:
                self._initial_epoch_loss[name] = val

            training_rates[name] = (
                val / (self._initial_epoch_loss[name] + 1e-8)
            )

        mean_training_rate = (
            sum(training_rates.values()) / max(len(training_rates), 1)
        )

        # --------------------------------------------------
        # per-task metrics
        # --------------------------------------------------
        for name, val in epoch_loss_dict.items():

            # relative scale wrt mean task loss of this epoch
            if mean_loss > 1e-8:
                metrics[f"loss_dynamics/relative_scale_{name}"] = torch.tensor(
                    val / mean_loss
                )

            # current / initial epoch loss
            training_rate = training_rates[name]
            metrics[f"loss_dynamics/training_rate_{name}"] = torch.tensor(
                training_rate
            )

            inverse_training_rate = (
                training_rate / (mean_training_rate + 1e-8)
            )
            metrics[f"loss_dynamics/inverse_training_rate_{name}"] = torch.tensor(
                inverse_training_rate
            )

            # epoch-to-epoch change
            if name in self._prev_epoch_loss:
                prev = self._prev_epoch_loss[name]

                # absolute decrease
                delta = prev - val
                metrics[f"loss_dynamics/rate_{name}"] = torch.tensor(delta)

                # EMA(rate)
                if name not in self._ema_epoch_rate:
                    self._ema_epoch_rate[name] = delta
                else:
                    self._ema_epoch_rate[name] = (
                        self.ema_alpha * delta
                        + (1.0 - self.ema_alpha) * self._ema_epoch_rate[name]
                    )

                metrics[f"loss_dynamics/ema_rate_{name}"] = torch.tensor(
                    self._ema_epoch_rate[name]
                )

                # relative decrease
                speed = delta / (prev + 1e-8)
                metrics[f"loss_dynamics/speed_{name}"] = torch.tensor(speed)

                # EMA(speed)
                if name not in self._ema_epoch_speed:
                    self._ema_epoch_speed[name] = speed
                else:
                    self._ema_epoch_speed[name] = (
                        self.ema_alpha * speed
                        + (1.0 - self.ema_alpha) * self._ema_epoch_speed[name]
                    )

                metrics[f"loss_dynamics/ema_speed_{name}"] = torch.tensor(
                    self._ema_epoch_speed[name]
                )

            self._prev_epoch_loss[name] = val

        return metrics

    # ------------------------------------------------------------------
    # Gradient metrics (step-level)
    # ------------------------------------------------------------------

    def compute_gradient_metrics(self, grad_info):
        metrics = {}

        task_grads = grad_info["task_grads"]
        total_grad = grad_info["total_grad"]

        total_norm = total_grad.norm().item()
        metrics["grad/total_norm"] = total_norm

        for task_name, g in task_grads.items():
            task_norm = g.norm().item()
            metrics[f"grad/{task_name}_norm"] = task_norm

            if task_norm > 0 and total_norm > 0:
                cos_sim = F.cosine_similarity(
                    g.unsqueeze(0),
                    total_grad.unsqueeze(0),
                    dim=1
                ).item()

                metrics[f"grad/{task_name}_cos_total"] = cos_sim

                proj = (
                    torch.dot(g, total_grad)
                    / (total_grad.norm() + 1e-8)
                ).item()

                metrics[f"grad/{task_name}_proj_total"] = proj
            else:
                metrics[f"grad/{task_name}_cos_total"] = 0.0
                metrics[f"grad/{task_name}_proj_total"] = 0.0

        return metrics