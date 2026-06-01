# ============================================
# 1. Base Weight Method Interface
# ============================================
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class BaseWeightMethod(ABC, nn.Module):
    """
    Base class cho tất cả weight methods.
    Định nghĩa các lifecycle hooks mà Lightning module có thể gọi.
    """
    
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        
    @abstractmethod
    def compute_weighted_loss(
        self, 
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> torch.Tensor:
        """
        Tính weighted loss từ dict of task losses.
        
        Args:
            losses: Dict[task_name, loss_value]
            model: The model being trained
            
        Returns:
            weighted_loss: Scalar tensor
        """
        pass
    
    def on_before_backward(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> None:
        """Hook được gọi trước khi backward pass."""
        pass
    
    def on_after_backward(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> None:
        """Hook được gọi sau backward nhưng trước optimizer step."""
        pass
    
    def on_before_optimizer_step(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> None:
        """Hook được gọi trước optimizer.step()."""
        pass
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """
        Trả về list of parameter groups để thêm vào optimizer.
        Cho phép weight method có learnable parameters.
        """
        return []
    
    # def get_logging_dict(self) -> Dict[str, float]:
    #     """Trả về metrics để log."""
    #     return {}
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Metrics dùng cho logging.
        Không ảnh hưởng optimization.
        """
        return {}


# ============================================
# 2. Concrete Weight Methods
# ============================================

class LinearScalarWeighting(BaseWeightMethod):
    """Simple fixed weights."""
    
    def __init__(
        self, 
        n_tasks: int, 
        device: torch.device,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(n_tasks, device)
        self.task_weights = task_weights or {}
        
    def compute_weighted_loss(
        self, 
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> torch.Tensor:
        total_loss = 0.0
        for task_name, loss in losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss
            print(f"Task: {task_name:15} | Weight: {weight:<5} | Unweighted Loss: {loss.item():.4f}")
            
        # In ra loss tổng
        print(f"=> TOTAL LOSS: {total_loss.item():.4f}")
        print("-" * 32)
        return total_loss
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        total_loss = 0.0

        for task_name, loss in losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss

        return {
            "average_loss": total_loss.detach(),
            # "optimization_loss": total_loss.detach(),
        }


class FAMOWeighting(BaseWeightMethod):
    """
    FAMO weight method với update TRƯỚC backward pass.
    
    Luồng:
    1. Step đầu tiên:
       - Tính losses dict
       - Khởi tạo weights (logits = 0, softmax = [0.5, 0.5, ...])
       - compute_weighted_loss với weights ban đầu
       - backward + optimizer.step()
       - Lưu losses làm prev_losses
    
    2. Step tiếp theo:
       - Tính losses dict
       - Update weights dựa trên prev_losses và losses hiện tại (trong on_before_backward)
       - compute_weighted_loss với weights đã update
       - backward + optimizer.step()
       - Lưu losses làm prev_losses
    
    Note: min_losses là lower bound lý thuyết của mỗi task (thường = 0),
          KHÔNG phải loss thực tế ở step đầu tiên.
    """
    
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_names: List[str],
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        max_norm: float = 1.0,
        min_losses: Optional[torch.Tensor] = None,  # Có thể truyền vào hoặc mặc định = 0
    ):
        super().__init__(n_tasks, device)
        self.task_names = task_names
        self.gamma = gamma
        self.w_lr = w_lr
        self.max_norm = max_norm
        
        # Initialize FAMO parameters - logits ban đầu = 0
        self.w = nn.Parameter(
            torch.zeros(n_tasks),
            requires_grad=True
        )
        self.w_opt = torch.optim.Adam(
            [self.w], 
            lr=w_lr, 
            weight_decay=gamma
        )
        
        # Cache
        self.prev_losses = None  # Lưu losses từ step trước
        self.current_weights = None  # Lưu weights hiện tại để log
        self.is_first_step = True  # Flag để biết step đầu tiên
        self.last_weighted_loss = 0.0

        # Min losses = lower bound lý thuyết (thường = 0 cho tất cả tasks)
        if min_losses is None:
            self.register_buffer("min_losses", torch.zeros(n_tasks))
        else:
            self.register_buffer("min_losses", min_losses)
        
        print(f"[FAMO] Initialized with min_losses (lower bounds): {self.min_losses}")
    
    def on_before_backward(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> None:
        """
        FAMO weight update - chạy TRƯỚC backward pass.
        
        Luồng:
        - Step đầu tiên: chỉ lưu losses, không update weights
        - Step tiếp theo: update weights dựa trên prev_losses và losses hiện tại
        """
        # Convert dict to tensor theo thứ tự task_names
        loss_tensor = torch.stack([
            losses[name] for name in self.task_names
        ])
        
        if self.is_first_step:
            # Step đầu tiên: chỉ lưu losses, không update
            self.is_first_step = False
            print(f"[FAMO] First step - losses: {loss_tensor}")
        else:
            # Step tiếp theo: update weights
            if self.prev_losses is not None:
                # Tính delta = log(L_t-1 - L_min) - log(L_t - L_min)
                # L_min là lower bound (thường = 0), không phải loss thực tế
                delta = (
                    (self.prev_losses - self.min_losses + 1e-8).log() - 
                    (loss_tensor.detach() - self.min_losses + 1e-8).log()
                )
                
                # Tính gradient của softmax(w) theo w
                with torch.enable_grad():
                    d = torch.autograd.grad(
                        torch.softmax(self.w, dim=-1),
                        self.w,
                        grad_outputs=delta.detach()
                    )[0]
                
                # Update w parameters
                self.w_opt.zero_grad()
                self.w.grad = d
                self.w_opt.step()
        
        # Lưu losses hiện tại làm prev_losses cho step sau
        self.prev_losses = loss_tensor.detach()
    
    def compute_weighted_loss(
        self, 
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> torch.Tensor:
        """
        Tính weighted loss bằng FAMO formula.
        Được gọi SAU on_before_backward (weights đã được update).
        """
        # Convert dict to tensor
        loss_tensor = torch.stack([
            losses[name] for name in self.task_names
        ])
        
        # FAMO weighting: z = softmax(w)
        z = torch.softmax(self.w, dim=-1)
        
        # D_i = L_i - L_min + eps
        D = loss_tensor - self.min_losses + 1e-8

        print(f"D: {D}")
        
        # c = sum(z_i / D_i) - normalization constant
        c = (z / D).sum().detach()
        
        # Weighted loss = sum(log(D_i) * z_i / c)
        weighted_loss = (D.log() * z / c).sum()
        self.last_weighted_loss = weighted_loss.detach()
        
        # Cache weights for logging
        self.current_weights = z.detach()

        print(f"loss components: {loss_tensor} - average loss: {loss_tensor.mean()}")
        print(f"current weights: {self.current_weights}")
        print(f"gradient loss: {weighted_loss}")
        
        return weighted_loss
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """FAMO không cần thêm params vào main optimizer (có optimizer riêng)."""
        return []
    
    # def get_logging_dict(self) -> Dict[str, float]:
    #     """Return current weights for logging."""
    #     if self.current_weights is None:
    #         return {}
        
    #     log_dict = {}
    #     for i, task_name in enumerate(self.task_names):
    #         log_dict[f"weight/{task_name}"] = self.current_weights[i].item()
        
    #     # Optional: log raw logits để debug
    #     # log_dict["weight/logits_norm"] = self.w.norm().item()
        
    #     return log_dict
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        loss_tensor = torch.stack(
            [losses[name] for name in self.task_names]
        )

        metrics = {
            "average_loss": loss_tensor.mean().detach(),
            "log_loss": self.last_weighted_loss,
        }

        if self.current_weights is not None:
            for i, name in enumerate(self.task_names):
                metrics[f"weight/{name}"] = self.current_weights[i].detach()

        return metrics


# ============================================
# 3. Weight Method Manager
# ============================================

class WeightMethodManager(nn.Module):
    """
    Quản lý weight method và expose unified interface cho Lightning.
    """
    
    def __init__(self, weight_method: BaseWeightMethod):
        super().__init__()
        self.weight_method = weight_method
        
    def compute_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> torch.Tensor:
        """Delegate to weight method."""
        return self.weight_method.compute_weighted_loss(
            losses, model, **kwargs
        )
    
    def trigger_lifecycle_hooks(
        self,
        hook_name: str,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> None:
        """
        Trigger một lifecycle hook nếu weight method implement nó.
        
        Args:
            hook_name: Tên của hook (e.g., 'on_before_backward')
            losses: Task losses
            model: The model
            **kwargs: Additional context
        """
        hook_method = getattr(self.weight_method, hook_name, None)
        if hook_method and callable(hook_method):
            hook_method(losses, model, **kwargs)
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """Get learnable parameters from weight method."""
        return self.weight_method.get_optimizer_parameters()
    
    # def get_logging_dict(self) -> Dict[str, float]:
    #     """Get logging metrics."""
    #     return self.weight_method.get_logging_dict()
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.weight_method.get_monitoring_metrics(losses)