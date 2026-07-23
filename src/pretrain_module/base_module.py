# src/pretrain_module/base_module.py
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.mtl.mtl import WeightMethodManager
from src.mtl.gradient_monitor import MTLGradientMonitor
from src.mtl.mtl import LinearScalarWeighting, FAMOWeighting, OGRWeighting

class BaseModule(L.LightningModule, ABC):
    """
    Abstract base class for all pretraining modules.
    
    Subclasses must implement:
    - _build_model()
    - _build_tasks()
    - shared_step()
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Will be initialized by subclasses
        self.model = None
        self.tasks = None
        self.weight_method_manager = None

        self._current_losses = None
        self._current_forward_context = None
        self._train_losses_accum: Dict[str, List[float]] = {}
        self._val_losses_accum:   Dict[str, List[float]] = {}
        self._last_train_epoch_losses: Dict[str, float] = {} # thêm
    
    @abstractmethod
    def _build_model(self, cfg) -> nn.Module:
        """
        Build and return the main model.
        
        Args:
            cfg: Configuration object
            
        Returns:
            nn.Module: The main model
        """
        pass
    
    @abstractmethod
    def _build_tasks(self, cfg, model_cfg: Optional[Dict] = None) -> nn.ModuleDict:
        """
        Build and return task modules.
        
        Args:
            cfg: Configuration object
            model_cfg: Optional model configuration dict
            
        Returns:
            nn.ModuleDict: Dictionary of task modules
        """
        pass
    
    @abstractmethod
    def shared_step(self, ctx) -> Dict[str, torch.Tensor]:
        """
        Compute all task losses for a batch.
        
        Args:
            ctx: Input batch + randomness
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to loss values
        """
        pass

    @abstractmethod
    def build_forward_context(self, batch):
        pass
    
    def _build_weight_method(self, cfg) -> WeightMethodManager:
        """
        Factory method to create weight method from config.
        Can be overridden by subclasses if needed.
        """
        
        method_name = cfg.get("weight_method", {}).get("name", "linear")
        task_names = list(self.tasks.keys())
        n_tasks = len(task_names)
        
        if n_tasks == 0:
            raise ValueError("At least one task must be enabled!")
        
        if n_tasks == 1:
            print(f"[INFO] Single task detected: {task_names[0]}")
            print("[INFO] Using LinearScalarWeighting with weight=1.0\n")
            weight_method = LinearScalarWeighting(
                n_tasks=1,
                device=self.device,
                task_weights={task_names[0]: 1.0}
            )
        elif method_name == "linear":
            task_weights = cfg.get("weight_method", {}).get("task_weights", {})
            filtered_weights = {k: v for k, v in task_weights.items() if k in task_names}
            weight_method = LinearScalarWeighting(
                n_tasks=n_tasks,
                device=self.device,
                task_weights=filtered_weights
            )
        # elif method_name == "famo":
        #     params = cfg.get("weight_method", {})
        #     weight_method = FAMOWeighting(
        #         n_tasks=n_tasks,
        #         device=self.device,
        #         task_names=task_names,
        #         gamma=params.get("gamma", 1e-3),
        #         w_lr=params.get("w_lr", 0.025),
        #     )
        elif method_name == "famo":
            params = cfg.get("weight_method", {})
            weight_method = FAMOWeighting(
                n_tasks=n_tasks,
                device=self.device,
                task_names=task_names,
                forward_losses_fn=self.forward_losses_no_grad,
                gamma=params.get("gamma", 1e-3),
                w_lr=params.get("w_lr", 0.025),
            )
        elif method_name == "ogr":
            params = cfg.get("weight_method", {})
            weight_method = OGRWeighting(
                n_tasks=n_tasks,
                device=self.device,
                task_names=task_names,
                softmax_temp=params.get("softmax_temp", 0.01),
            )
        else:
            raise ValueError(f"Unknown weight method: {method_name}")
        
        print(f"[INFO] Initialized MTL Weight Method: {method_name.upper()}\n")
        return WeightMethodManager(weight_method)
    
    def setup_training(self):
        """
        Initialize all components for training.
        Called after __init__ to ensure proper initialization order.
        """
        # 1. Build model
        self.model = self._build_model(self.cfg)
        
        # 2. Build tasks
        self.tasks = self._build_tasks(self.cfg)
        
        # 3. Initialize weight method
        self.weight_method_manager = self._build_weight_method(self.cfg)

        # 4. Gradient monitor (opt-in via config)
        monitor_cfg = self.cfg.get("grad_monitor", {})
        if monitor_cfg.get("enabled", False):
            self.grad_monitor = MTLGradientMonitor(
                ema_alpha=monitor_cfg.get("ema_alpha", 0.1),
                log_every_n=monitor_cfg.get("log_every_n", 50),
            )
            print(f"[INFO] MTLGradientMonitor enabled — log_every_n={self.grad_monitor.log_every_n}")
        else:
            self.grad_monitor = None
        
    # ==========================================
    # Lightning Lifecycle Hooks
    # ==========================================
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        if self.weight_method_manager is not None:
            self.weight_method_manager.to(self.device)
            self.weight_method_manager.trigger_lifecycle_hooks(
                "on_fit_start",
                losses=None,
                model=None
            )
    
    
    def training_step(self, batch, batch_idx):
        """Standard training step with MTL support."""
        # Step 1: Compute task losses
        ctx = self.build_forward_context(batch)
        loss_dict = self.shared_step(ctx)
        self._current_forward_context = ctx
        self._current_losses = loss_dict
        
        # Step 2: Pre-backward hook (FAMO updates weights here)
        self.weight_method_manager.trigger_lifecycle_hooks(
            "on_before_backward",
            losses=loss_dict,
            model=self.model,
            forward_context=self._current_forward_context
        )
        
        # Step 3: Compute loss
        weighted_loss, weighted_losses = (
            self.weight_method_manager.compute_weighted_loss(
                losses=loss_dict,
                model=self.model,
                batch=batch,
                tasks=self.tasks
            )
        )

        # Step 4: COMPUTE TASK GRADIENTS (BEFORE BACKWARD)
        grad_metrics = None
        if (
            self.grad_monitor is not None
            and self.global_step % self.grad_monitor.log_every_n == 0
        ):
            task_grads = self._compute_task_gradients(
                weighted_losses=weighted_losses,
                total_loss=weighted_loss,
            )
            grad_metrics = self.grad_monitor.compute_gradient_metrics(task_grads)

        # Step 5: Logging
        self._log_training_metrics(
            loss_dict=loss_dict,
            grad_metrics=grad_metrics,
        )

        for task_name, loss in loss_dict.items():
            self._train_losses_accum.setdefault(task_name, []).append(loss.detach().item())
        
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        """Standard validation step."""
        ctx = self.build_forward_context(batch)
        loss_dict = self.shared_step(ctx)

        for task_name, loss in loss_dict.items():
            self.log(
                f"valid/{task_name}_valid_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
        
        valid_loss = torch.stack([l.detach() for l in loss_dict.values()]).mean()
        self.log(
            "valid/average_valid_loss",
            valid_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        for task_name, loss in loss_dict.items():
            self._val_losses_accum.setdefault(task_name, []).append(loss.detach().item())

        return valid_loss
    
    def on_before_optimizer_step(self, optimizer):
        """Lightning hook - called before optimizer.step()."""
        if self._current_losses is not None:
            self.weight_method_manager.trigger_lifecycle_hooks(
                "on_before_optimizer_step",
                losses=self._current_losses,
                model=self.model,
                optimizer=optimizer,
                tasks=self.tasks
            )
    
    def on_after_backward(self):
        """Lightning hook - called after backward."""
        if self._current_losses is not None:
            self.weight_method_manager.trigger_lifecycle_hooks(
                "on_after_backward",
                losses=self._current_losses,
                model=self.model,
                tasks=self.tasks
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Lightning hook - called after training_step."""

        for task_name, task in self.tasks.items():
            if hasattr(task, 'on_train_batch_end'):
                task.on_train_batch_end(
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx
                )

        self.weight_method_manager.trigger_lifecycle_hooks(
                "on_train_batch_end",
                losses=self._current_losses,
                model=self.model,
                tasks=self.tasks
            )
        
        self._current_losses = None
        self._current_forward_context = None

    def on_train_epoch_end(self):
        """Compute epoch-level training loss dynamics."""
        train_losses = {
            k: sum(v) / len(v)
            for k, v in self._train_losses_accum.items()
            if v
        }

        self._last_train_epoch_losses = train_losses

        if self.grad_monitor is not None and len(train_losses) > 0:
            metrics = self.grad_monitor.compute_epoch_loss_dynamics(train_losses)
            for k, v in metrics.items():
                self.log(
                    f"gradient_monitor/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )

        self._train_losses_accum.clear()

    def on_validation_epoch_end(self):
        """Called after validation epoch ends."""
        val_losses = {
            k: sum(v) / len(v)
            for k, v in self._val_losses_accum.items() if v
        }

        # Trigger OGR (or any method needing train/val epoch losses)
        self.weight_method_manager.trigger_lifecycle_hooks(
            "on_validation_epoch_end",
            losses={},
            model=self.model,
            train_losses=self._last_train_epoch_losses,
            val_losses=val_losses,
        )

        self._val_losses_accum.clear()
    
    def on_save_checkpoint(self, checkpoint):

        checkpoint["weight_method"] = (
            self.weight_method_manager.checkpoint_state()
        )

    def on_load_checkpoint(self, checkpoint):

        state = checkpoint.get("weight_method", None)

        if state is not None:
            self.weight_method_manager.restore_checkpoint(state)
            
    # ==========================================
    # Optimizer Configuration
    # ==========================================
    
    def configure_optimizers(self):
        """
        Optimizer chỉ lấy:
        - model params
        - task params
        - weight method params (nếu có)
        """
        # 1. Model parameters
        model_params = list(self.model.parameters())

        # 2. Task parameters
        task_params = []
        for task in self.tasks.values():
            task_params += list(task.parameters())

        # 3. Weight method parameters 
        weight_method_param_groups = self.weight_method_manager.get_optimizer_parameters()

        # 4. Combine
        param_groups = [
            {
                "params": model_params + task_params,
                "weight_decay": self.cfg.optimizer.weight_decay,
            }
        ]

        # add weight method param groups (nếu có)
        param_groups.extend(weight_method_param_groups)
        optimizer = torch.optim.AdamW(param_groups)

        # Define lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
            max_lr = self.cfg.scheduler.max_lr, 
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()), 
            epochs = self.trainer.max_epochs,
            div_factor = self.cfg.scheduler.div_factor,
            final_div_factor = self.cfg.scheduler.final_div_factor,
            pct_start = self.cfg.scheduler.pct_start,
        )

        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            'interval': 'step', # The unit of the scheduler's step size, could also be 'step'
            'frequency': 1, # The frequency of the scheduler
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_dict
        }
    
    # ==========================================
    # Helper Methods
    # ==========================================
    def forward_losses_no_grad(self, ctx):
        print("\n[DEBUG] === forward_losses_no_grad CALLED ===")
        print("[DEBUG] model.training BEFORE:", self.training)

        was_training = self.training

        self.eval()

        with torch.inference_mode():
            loss_dict = self.shared_step(ctx)

        print("[DEBUG] loss_dict keys:", loss_dict.keys())
        for k, v in loss_dict.items():
            print(f"[DEBUG] {k}: {v.item():.6f}")

        if was_training:
            self.train()

        print("[DEBUG] model.training AFTER:", self.training)
        print("[DEBUG] === END forward_losses_no_grad ===\n")

        return loss_dict
    
    def _log_training_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        grad_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Log training metrics. Can be overridden for custom logging."""
        # Log individual task losses
        for task_name, loss in loss_dict.items():
            self.log(
                f"train/{task_name}_loss",
                loss,
                on_epoch=True,
                on_step=False
            )

        weight_method_metrics = (
            self.weight_method_manager
                .get_monitoring_metrics(loss_dict)
        )

        for key, value in weight_method_metrics.items():
            self.log(
                f"weight_method/{key}",
                value,
                on_epoch=True,
                on_step=False,
                prog_bar=(key == "average_loss")
            )

        # Log gradient metrics only
        if grad_metrics is not None:
            for k, v in grad_metrics.items():
                self.log(
                    f"gradient_monitor/{k}",
                    v,
                    on_step=False,
                    on_epoch=True
                )
            
    def _compute_task_gradients(
        self,
        total_loss=None,
        weighted_losses=None,
    ):
        """
        Compute per-task gradients on encoder BEFORE backward.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Weighted-task grads
        task_grads = {}

        for task_name, loss in weighted_losses.items():
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            flat = []
            for g, p in zip(grads, params):
                if g is None:
                    flat.append(torch.zeros_like(p).flatten())
                else:
                    flat.append(g.detach().flatten())
            task_grads[task_name] = torch.cat(flat)

        # total grads
        grads = torch.autograd.grad(
            total_loss,
            params,
            retain_graph=True,
            allow_unused=True
        )
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(torch.zeros_like(p).flatten())
            else:
                flat.append(g.detach().flatten())
        total_grad = torch.cat(flat)

        return {
            "task_grads": task_grads,
            "total_grad": total_grad,
        }