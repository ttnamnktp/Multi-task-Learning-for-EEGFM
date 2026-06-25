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
        self._train_losses_accum: Dict[str, List[float]] = {}
        self._val_losses_accum:   Dict[str, List[float]] = {}
    
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
    def shared_step(self, batch) -> Dict[str, torch.Tensor]:
        """
        Compute all task losses for a batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to loss values
        """
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
        elif method_name == "famo":
            params = cfg.get("weight_method", {})
            weight_method = FAMOWeighting(
                n_tasks=n_tasks,
                device=self.device,
                task_names=task_names,
                gamma=params.get("gamma", 1e-5),
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
    
    def training_step(self, batch, batch_idx):
        """Standard training step with MTL support."""
        # Step 1: Compute task losses
        loss_dict = self.shared_step(batch)
        self._current_losses = loss_dict
        _monitor_losses = {k: v for k, v in loss_dict.items()}
        
        # Step 2: Pre-backward hook (FAMO updates weights here)
        self.weight_method_manager.trigger_lifecycle_hooks(
            "on_before_backward",
            losses=loss_dict,
            model=self.model,
            batch=batch
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
        if (self.grad_monitor is not None and self.grad_monitor._step % self.grad_monitor.log_every_n == 0):
            _task_grads = self._compute_task_gradients(
                weighted_losses=weighted_losses,
                total_loss=weighted_loss,
            )
        else:
            _task_grads = None

        # Step 5: Logging
        self._log_training_metrics(loss_dict, _task_grads, _monitor_losses)

        for task_name, loss in loss_dict.items():
            self._train_losses_accum.setdefault(task_name, []).append(loss.detach().item())
        
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        """Standard validation step."""
        loss_dict = self.shared_step(batch)

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
        self._current_losses = None

        for task_name, task in self.tasks.items():
            if hasattr(task, 'on_train_batch_end'):
                task.on_train_batch_end(
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx
                )

    def on_validation_epoch_end(self):
        """Hook Lightning — gọi sau khi toàn bộ validation epoch kết thúc."""
        # Average losses của epoch
        train_losses = {
            k: sum(v) / len(v)
            for k, v in self._train_losses_accum.items() if v
        }
        val_losses = {
            k: sum(v) / len(v)
            for k, v in self._val_losses_accum.items() if v
        }

        # Trigger OGR (hoặc bất kỳ method nào implement hook này)
        self.weight_method_manager.trigger_lifecycle_hooks(
            "on_validation_epoch_end",
            losses={},          # placeholder (OGR không dùng)
            model=self.model,
            train_losses=train_losses,
            val_losses=val_losses,
        )

        # Reset buffers
        self._train_losses_accum.clear()
        self._val_losses_accum.clear()
        
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
    
    def _log_training_metrics(self, loss_dict: Dict[str, torch.Tensor], _task_grads=None, _monitor_losses=None):
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

        # if _task_grads is not None and _monitor_losses is not None and self.grad_monitor is not None:
        if self.grad_monitor is not None:
            metrics = self.grad_monitor.compute(
                loss_dict=_monitor_losses,
                task_grads=_task_grads
            )
            for k, v in metrics.items():
                self.log(f"gradient_monitor/{k}", v, on_epoch=True, on_step=False)

        if _task_grads is not None:
            print("[DEBUG]"
                f"LOG STEP={self.global_step}, "
                f"monitor_step={self.grad_monitor._step}"
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