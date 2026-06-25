# src/mtl/gradient_monitor.py
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

class MTLGradientMonitor:
    """
    Compute and return MTL diagnostic metrics after backward pass.

    Args:
        encoder:        The shared encoder module (gradients accumulated here).
        layer_names:    List of encoder submodule names to track.
                        If None, auto-discovers all named children.
        ema_alpha:      Smoothing factor for loss-decrease-rate EMA (0 < α ≤ 1).
        log_every_n:    Only compute heavy metrics every n steps (gradient metrics
                        require an extra backward per task — expensive).  Set to 1
                        to compute every step.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        log_every_n: int = 50,
    ):
        self.ema_alpha = ema_alpha
        self.log_every_n = log_every_n

        # State
        self._step = 0

        # absolute loss decrease EMA
        self._ema_loss: Dict[str, float] = {}

        # relative loss decrease EMA
        self._ema_speed: Dict[str, float] = {}

        # previous step loss
        self._prev_loss: Dict[str, float] = {}

        # first observed loss (for GradNorm-style training rate)
        self._initial_loss: Dict[str, float] = {}
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        loss_dict: Dict[str, torch.Tensor],
        task_grads,
    ) -> Dict[str, torch.Tensor]:
        """
        Call this inside on_after_backward (gradients are populated, no
        optimizer step yet).

        Args:
            module:     The LightningModule (used to re-run per-task backward).
            loss_dict:  Task losses from shared_step.
            task_names: Ordered list of task names.

        Returns:
            Flat dict of scalar metrics ready for self.log().
        """
        self._step += 1
        metrics: Dict[str, torch.Tensor] = {}

        # Loss-dynamics metrics (cheap — no extra backward)
        metrics.update(self._loss_dynamics(loss_dict))

        # Gradient metrics (expensive — one backward per task)
        # if self._step % self.log_every_n == 0:
        if task_grads is not None:
            metrics.update(
                # self._gradient_metrics(loss_dict, task_names)
                self._gradient_metrics(task_grads)
            )

        return metrics

    # ------------------------------------------------------------------
    # Loss dynamics
    # ------------------------------------------------------------------

    def _loss_dynamics(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        metrics: Dict[str, torch.Tensor] = {}

        loss_values = torch.stack(
            [loss.detach() for loss in loss_dict.values()]
        )

        # --------------------------------------------------
        # global imbalance
        # --------------------------------------------------

        metrics["loss_dynamics/variance"] = loss_values.var()

        mean_loss = loss_values.mean()

        # --------------------------------------------------
        # inverse training rates
        # --------------------------------------------------

        training_rates: Dict[str, float] = {}

        for name, loss in loss_dict.items():

            val = float(loss.detach())

            if name not in self._initial_loss:
                self._initial_loss[name] = val

            training_rates[name] = (
                val / (self._initial_loss[name] + 1e-8)
            )

        mean_training_rate = (
            sum(training_rates.values())
            / max(len(training_rates), 1)
        )

        # --------------------------------------------------
        # per-task metrics
        # --------------------------------------------------

        for name, loss in loss_dict.items():

            val = float(loss.detach())

            # ----------------------------------------------
            # store initial loss
            # ----------------------------------------------

            if name not in self._initial_loss:
                self._initial_loss[name] = val

            # ----------------------------------------------
            # relative scale
            # ----------------------------------------------

            if mean_loss.item() > 1e-8:
                metrics[f"loss_dynamics/relative_scale_{name}"] = (
                    loss.detach() / mean_loss
                )

            # ----------------------------------------------
            # GradNorm-style training rate
            #
            # current loss / initial loss
            # ----------------------------------------------

            training_rate = training_rates[name]

            metrics[f"loss_dynamics/training_rate_{name}"] = torch.tensor(
                training_rate
            )

            inverse_training_rate = (
                training_rate
                / (mean_training_rate + 1e-8)
            )

            metrics[
                f"loss_dynamics/inverse_training_rate_{name}"
            ] = torch.tensor(
                inverse_training_rate
            )

            # ----------------------------------------------
            # step-to-step metrics
            # ----------------------------------------------

            if name in self._prev_loss:

                prev = self._prev_loss[name]

                # absolute decrease
                delta = prev - val

                metrics[f"loss_dynamics/rate_{name}"] = torch.tensor(delta)

                # EMA(delta)
                if name not in self._ema_loss:
                    self._ema_loss[name] = delta
                else:
                    self._ema_loss[name] = (
                        self.ema_alpha * delta
                        + (1.0 - self.ema_alpha) * self._ema_loss[name]
                    )

                metrics[f"loss_dynamics/ema_rate_{name}"] = torch.tensor(
                    self._ema_loss[name]
                )

                # ------------------------------------------
                # relative training speed
                #
                # (prev - current)/prev
                # ------------------------------------------

                speed = delta / (prev + 1e-8)

                metrics[f"loss_dynamics/speed_{name}"] = torch.tensor(
                    speed
                )

                # EMA(speed)

                if name not in self._ema_speed:
                    self._ema_speed[name] = speed
                else:
                    self._ema_speed[name] = (
                        self.ema_alpha * speed
                        + (1.0 - self.ema_alpha) * self._ema_speed[name]
                    )

                metrics[f"loss_dynamics/ema_speed_{name}"] = torch.tensor(
                    self._ema_speed[name]
                )

            self._prev_loss[name] = val

        return metrics

    # ------------------------------------------------------------------
    # Gradient metrics
    # ------------------------------------------------------------------

    def _gradient_metrics(self, grad_info):

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

                # projection lên hướng total_grad
                proj = (
                    torch.dot(g, total_grad)
                    / (total_grad.norm() + 1e-8)
                ).item()

                metrics[f"grad/{task_name}_proj_total"] = proj

            else:
                metrics[f"grad/{task_name}_cos_total"] = 0.0
                metrics[f"grad/{task_name}_proj_total"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_flat(
        grad_dict: Dict[str, torch.Tensor], layer_name: str
    ) -> Optional[torch.Tensor]:
        """Flatten and concatenate all gradients from a given layer prefix."""
        parts = [
            g.flatten()
            for k, g in grad_dict.items()
            if k.startswith(layer_name)
        ]
        if not parts:
            return None
        return torch.cat(parts)

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denom = (a.norm() * b.norm()).clamp(min=1e-8)
        return (a @ b) / denom