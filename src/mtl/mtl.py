# ========================================================================================
# Base Weight Method Interface
# ========================================================================================
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
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
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """
        Tính weighted loss từ dict of task losses.
        
        Args:
            losses: Dict[task_name, loss_value]
            model: The model being trained
            
        Returns:
            weighted_loss: Scalar tensor
        """
        pass

    def on_fit_start(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ) -> None:
        """Hook được gọi trước khi backward pass."""
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

    def on_train_batch_end(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs,
    ) -> None:
        """Hook gọi sau optimizer.step(), cuối mỗi training batch."""
        pass
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """
        Trả về list of parameter groups để thêm vào optimizer.
        Cho phép weight method có learnable parameters.
        """
        return []
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Metrics dùng cho logging.
        Không ảnh hưởng optimization.
        """
        return {}
    
    def checkpoint_state(self):
        """
        State cần lưu khi save checkpoint.
        """
        return {}

    def restore_checkpoint(self, state):
        """
        Restore state sau khi load checkpoint.
        """
        pass

# ========================================================================================
# Linear Scalar Weighting
# ========================================================================================

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
    ):
        weighted_losses = {}
        total_loss = 0.0
        for task_name, loss in losses.items():
            weight = self.task_weights.get(task_name, 1.0 / max(len(losses), 1))
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss
            total_loss += weighted_loss
            print(f"Task: {task_name:15} | Weight: {weight:<5} | Unweighted Loss: {loss.item():.4f}")
            
        # In ra loss tổng
        print(f"=> TOTAL LOSS: {total_loss.item():.4f}")
        print("-" * 32)
        return total_loss, weighted_losses
    
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
        }

# ========================================================================================
# FAMO Weighting
# ========================================================================================

class EcoFAMOWeighting(BaseWeightMethod):
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
        min_losses: Optional[torch.Tensor] = None,  # Có thể truyền vào hoặc mặc định = 0
    ):
        super().__init__(n_tasks, device)
        self.task_names = task_names
        self.gamma = gamma
        self.w_lr = w_lr
        
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
        
        print(f"[FAMO] Initialized with min_losses (lower bounds): {self.min_losses} and weight learning rate {self.w_lr}")
    
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
        # Convert dict to tensor theo thứ tự task_names. 
        # Detach hoàn toàn các giá trị loss scalar để làm dữ liệu tính toán độc lập cho FAMO
        loss_tensor = torch.stack([
            losses[name].detach() for name in self.task_names 
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
                    (loss_tensor - self.min_losses + 1e-8).log()
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
                print("===== BEFORE w_opt.step() =====")
                print("w device:", self.w.device)
                print("grad device:", self.w.grad.device)

                for state in self.w_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k}: {v.device}")
                self.w_opt.step()
        
        # Lưu losses hiện tại làm prev_losses cho step sau
        self.prev_losses = loss_tensor
    
    def compute_weighted_loss(
        self, 
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ):
        """
        Tính weighted loss bằng FAMO formula.
        Được gọi SAU on_before_backward (weights đã được update).
        """
        # Convert dict to tensor
        loss_tensor = torch.stack([
            losses[name] for name in self.task_names
        ])
        
        weighted_losses = {}
        
        # FAMO weighting: z = softmax(w)
        z = torch.softmax(self.w, dim=-1)
        
        # D_i = L_i - L_min + eps
        D = loss_tensor - self.min_losses + 1e-8
        
        # c = sum(z_i / D_i) - normalization constant
        c = (z / D).sum().detach()
        
        # Weighted loss = sum(log(D_i) * z_i / c)
        # weighted_loss = (D.log() * z / c).sum()

        for i, name in enumerate(self.task_names):
            weighted_losses[name] = (
                D[i].log() * z[i] / c
            )

        weighted_loss = torch.stack(
            list(weighted_losses.values())
        ).sum()

        self.last_weighted_loss = weighted_loss.detach().item()
        
        # Cache weights for logging
        self.current_weights = z.detach()

        print(f"loss components: {loss_tensor} - average loss: {loss_tensor.mean()}")
        print(f"current weights: {self.current_weights}")
        print(f"gradient loss: {weighted_loss}")
        
        return weighted_loss, weighted_losses
    
    def get_optimizer_parameters(self) -> List[Dict[str, Any]]:
        """FAMO không cần thêm params vào main optimizer (có optimizer riêng)."""
        return []
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        loss_tensor = torch.stack(
            [losses[name].detach() for name in self.task_names]
        )

        metrics = {
            "average_loss": loss_tensor.mean().item(),
            "log_loss": self.last_weighted_loss,
        }

        if self.current_weights is not None:
            for i, name in enumerate(self.task_names):
                metrics[f"famo_weight/{name}"] = self.current_weights[i].item()

        return metrics

# ========================================================================================
# FAMO Original version
# ========================================================================================

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

class FAMOWeighting(BaseWeightMethod):
    """
    Exact FAMO (Algorithm 1/2) với tính năng theo dõi Loss t và Loss t+1.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_names: List[str],
        forward_losses_fn,
        gamma: float = 1e-3,
        w_lr: float = 0.025,
        min_losses: Optional[torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device)
        self.task_names = task_names
        self.forward_losses_fn = forward_losses_fn
        self.gamma = gamma
        self.w_lr = w_lr
        
        self.w = nn.Parameter(torch.zeros(n_tasks), requires_grad=True)
        self.w_opt = torch.optim.Adam(
            [self.w],
            lr=w_lr,
            weight_decay=gamma,
        )
        
        if min_losses is None:
            self.register_buffer("min_losses", torch.zeros(n_tasks))
        else:
            self.register_buffer("min_losses", min_losses)
            
        # ==========================
        # runtime state
        # ==========================
        self._current_ctx = None
        self._current_loss_t = None
        
        self._last_loss_t = None      # Loss của từng task tại thời điểm t (trước update)
        self._last_loss_t1 = None     # Loss của từng task tại thời điểm t+1 (sau update)
        
        self.current_weights = None
        self.last_weighted_loss = 0.
        
        print(f"[FAMO] Initialized (w_lr={w_lr}, gamma={gamma})")

    # =====================================================
    # Before backward
    # =====================================================
    def on_before_backward(
        self,
        losses,
        model,
        forward_context,
        **kwargs,
    ):
        self._current_ctx = forward_context
        # Lưu lại loss tại thời điểm t để tính delta toán học
        self._current_loss_t = torch.stack(
            [losses[name].detach() for name in self.task_names]
        )
        # Đồng thời clone sang biến chuyên tracking để tránh bị xóa ở cuối batch
        self._last_loss_t = self._current_loss_t.clone()

    # =====================================================
    # After optimizer.step()
    # =====================================================
    def on_train_batch_end(
        self,
        losses,
        model,
        **kwargs,
    ):
        if self._current_ctx is None or self._current_loss_t is None:
            print("[DEBUG] ctx and loss are not initialized")
            return
            
        ####################################################
        # Exact re-forward
        with torch.no_grad():
            loss_dict_next = self.forward_losses_fn(self._current_ctx)
            
        loss_t1 = torch.stack(
            [loss_dict_next[name].detach() for name in self.task_names]
        )
        # Lưu lại loss tại thời điểm t+1 để tracking
        self._last_loss_t1 = loss_t1.clone()
        
        ####################################################
        # delta
        eps = 1e-8
        delta = (
            (self._current_loss_t - self.min_losses + eps).log()
            -
            (loss_t1 - self.min_losses + eps).log()
        )
        print("Loss_t:", self._current_loss_t)
        print("Loss_t+1:", loss_t1)
        print("delta:", delta)
        
        ####################################################
        # update ξ
        with torch.enable_grad():
            softmax_w = torch.softmax(self.w, dim=-1)
            grad = torch.autograd.grad(
                softmax_w,
                self.w,
                grad_outputs=delta.detach(),
            )[0]
            
        self.w_opt.zero_grad()
        self.w.grad = grad
        self.w_opt.step()

        print("w before:", self.w)
        print("grad:", grad)
        
        ####################################################
        # cleanup (Chỉ xóa context, giữ lại các biến _last_loss_ để log)
        ####################################################
        self._current_ctx = None
        self._current_loss_t = None

    # =====================================================
    # weighted loss
    # =====================================================
    def compute_weighted_loss(
        self,
        losses,
        model,
        **kwargs,
    ):
        loss_tensor = torch.stack([losses[name] for name in self.task_names])
        z = torch.softmax(self.w, dim=-1)
        D = loss_tensor - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        
        weighted_losses = {}
        for i, name in enumerate(self.task_names):
            weighted_losses[name] = D[i].log() * z[i] / c
            
        weighted_loss = torch.stack(list(weighted_losses.values())).sum()
        
        self.current_weights = z.detach()
        self.last_weighted_loss = weighted_loss.detach().item()
        return weighted_loss, weighted_losses

    # =====================================================
    def get_optimizer_parameters(self):
        return []

    # =====================================================
    # Monitoring Metrics
    def get_monitoring_metrics(self, losses):
        # 1. Các metrics cơ bản có sẵn của bạn
        loss_tensor = torch.stack([losses[name].detach() for name in self.task_names])
        metrics = {
            "average_loss": loss_tensor.mean().item(),
            "log_loss": self.last_weighted_loss,
        }
        
        if self.current_weights is not None:
            for i, name in enumerate(self.task_names):
                metrics[f"famo_weight/{name}"] = self.current_weights[i].item()

        # PRINT RA CONSOLE 
        print("\n" + "="*50)
        print(f"[FAMO MONITORING] - Avg Loss: {metrics['average_loss']:.4f} | Log Loss: {metrics['log_loss']:.4f}")
        print("-"*50)
        
        for name in self.task_names:
            w_val = metrics.get(f"famo_weight/{name}", 0.0)
            # Ký hiệu mũi tên xanh/đỏ hoặc hướng tăng giảm loss để dễ nhìn bằng mắt
            print(f" > Task [{name}]:")
            print(f"   • Weight       : {w_val:.6f}")
        print("="*50 + "\n")

        return metrics
    
    # ========================================================
    def on_fit_start(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs
    ):
        """
        Move all optimizer internal states (exp_avg, exp_avg_sq, ...)
        to the same device as self.w.
        """
        device = self.w.device

        for state in self.w_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"[FAMO] Optimizer state moved to {device}")

    def checkpoint_state(self):
        return {
            "w": self.w.data,
            "w_opt": self.w_opt.state_dict(),
            "last_loss_t": self._last_loss_t,
            "last_loss_t1": self._last_loss_t1,
            "current_weights": self.current_weights,
        }

    def restore_checkpoint(self, state):
        self.w.data.copy_(state["w"])
        self.w_opt.load_state_dict(state["w_opt"])
        
        # 3. Khôi phục các biến tracking khác
        self._last_loss_t = state["last_loss_t"]
        self._last_loss_t1 = state["last_loss_t1"]
        self.current_weights = state["current_weights"]
    
# ========================================================================================
# OGR Weighting
# ========================================================================================

class OGRWeighting(BaseWeightMethod):
    """
    Overfitting-to-Generalization Ratio (OGR) weight method.
    
    Paper: Wang et al., 2020 — "What makes training multi-modal classification hard?"
    Adapted for self-supervised multi-objective pretraining (MAPE paper, baseline OGR).

    Update rule (Eq. 6 in MAPE paper):
        w_i* = G_i / (O_i)^2

    Với normalized losses (Li_bar = Li / Li(0)):
        G_i = L_val_i(t0) - L_val_i(tN)   # val improvement
        O_i = (train_drop_i) - (val_drop_i) # overfitting gap

    Weights được normalize bằng softmax với temperature để tránh:
      - âm (crash pretraining)
      - quá sharp (unstable)

    Update diễn ra sau mỗi epoch (on_validation_epoch_end).
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_names: List[str],
        softmax_temp: float = 0.02,
    ):
        super().__init__(n_tasks, device)
        self.task_names = task_names
        self.softmax_temp = softmax_temp

        # Current weights (uniform init)
        self.register_buffer(
            "current_weights",
            torch.ones(n_tasks) / n_tasks
        )

        # Epoch-start snapshots (set at beginning of each epoch)
        self._train_loss_epoch_start: Optional[Dict[str, float]] = None
        self._val_loss_epoch_start:   Optional[Dict[str, float]] = None

        # Normalization denominators (set from first epoch losses)
        self._train_loss_init: Optional[Dict[str, float]] = None
        self._val_loss_init:   Optional[Dict[str, float]] = None

        print(f"[OGR] Initialized — softmax_temp={softmax_temp}")

    # ------------------------------------------------------------------
    # Core: compute weighted loss (uses current_weights, static this epoch)
    # ------------------------------------------------------------------

    def compute_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weighted_losses = {}
        total_loss = torch.tensor(0.0, device=self.current_weights.device)

        for i, name in enumerate(self.task_names):
            w = self.current_weights[i]
            wl = w * losses[name]
            weighted_losses[name] = wl
            total_loss = total_loss + wl

        return total_loss, weighted_losses

    # ------------------------------------------------------------------
    # Lifecycle hook: called after each validation epoch
    # Receives aggregated val losses + train losses from BaseModule
    # ------------------------------------------------------------------
    def on_validation_epoch_end(
        self,
        losses: Dict[str, torch.Tensor],   # Không dùng
        model: nn.Module,
        train_losses: Optional[Dict[str, float]] = None,
        val_losses:   Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """
        Cập nhật OGR weights dựa trên toàn bộ tiến trình từ Epoch 0 đến Epoch N hiện tại.
        """
        if train_losses is None or val_losses is None:
            print("[OGR] Skipping weight update: losses not provided.")
            return

        # 1. Khởi tạo loss gốc (Epoch 0) để làm mẫu số chuẩn hóa
        if self._train_loss_init is None:
            self._train_loss_init = {k: max(v, 1e-8) for k, v in train_losses.items()}
            self._val_loss_init   = {k: max(v, 1e-8) for k, v in val_losses.items()}
            print(f"[OGR] Initial baseline losses recorded. Weights stay uniform for now.")
            return

        # 2. Tính toán G_i và O_i tích lũy (từ Epoch 0 -> Epoch hiện tại)
        new_weights = []
        for name in self.task_names:
            L0_tr  = self._train_loss_init[name]
            L0_val = self._val_loss_init[name]

            # Normalized losses tại epoch hiện tại (Li_bar = Li(t) / Li(0))
            tr_curr  = train_losses[name] 
            val_curr = val_losses[name]   

            # G_i: L_val(0)_bar - L_val(t)_bar
            G_i =  L0_val - val_curr
            
            # O_i: Khoảng cách Overfitting Gap chuẩn hóa tại thời điểm t
            O_i = L0_tr - tr_curr - (L0_val - val_curr)

            # Công thức gốc Eq. 6: w_i = G_i / (O_i^2)
            w_i = G_i / (O_i ** 2 + 1e-6)
            new_weights.append(w_i)

        print(f"[OGR] Anormalized weights: {new_weights}")

        # 3. Normalize bằng Softmax kết hợp Temperature
        w_tensor = torch.tensor(new_weights, dtype=torch.float32,
                                device=self.current_weights.device)
        
        # Áp dụng softmax để đưa về dạng phân phối xác suất tổng bằng 1
        self.current_weights = torch.softmax(w_tensor * self.softmax_temp, dim=0)
        
        print(f"[OGR] Updated weights: "
              + ", ".join(f"{n}={self.current_weights[i].item():.4f}"
                          for i, n in enumerate(self.task_names)))
        
    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_tensor = torch.stack(
            [losses[n].detach() for n in self.task_names]
        )
        metrics = {"average_loss": loss_tensor.mean()}
        for i, name in enumerate(self.task_names):
            metrics[f"ogr_weight/{name}"] = self.current_weights[i]
        return metrics
    
# ========================================================================================
# Weight Method Manager
# ========================================================================================

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
    ):
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
    
    def get_monitoring_metrics(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.weight_method.get_monitoring_metrics(losses)
    
    def checkpoint_state(self):
        return self.weight_method.checkpoint_state()

    def restore_checkpoint(self, state):
        self.weight_method.restore_checkpoint(state)
    

# =====================================================================
# BẠN CÓ THỂ CHÈN ĐOẠN KHỐI MAIN NÀY VÀO CUỐI FILE MTL CỦA BẠN
# =====================================================================
if __name__ == "__main__":
    print("--- KHỞI CHẠY UNIT TEST FOR FAMO WEIGHTING ---")
    import torch
    import torch.nn as nn

    # ==========================
    # Disable CUDA completely
    # ==========================
    torch.cuda.is_available = lambda: False
    torch.cuda.is_current_stream_capturing = lambda: False
    
    # 1. Cấu hình ban đầu
    device = torch.device('cpu')
    task_names = ["task_1", "task_2"]
    n_tasks = len(task_names)
    
    # Định nghĩa giá trị Loss_t và Loss_t+1 theo đề bài của bạn
    # (Tự động chuyển về 'cpu' nếu máy bạn đang test không có GPU CUDA)
    loss_t_values = torch.tensor([0.8564, 2.3708], device=device)
    loss_t1_values = torch.tensor([0.8551, 2.3472], device=device)
    
    # Chuyển đổi thành dictionary giống định dạng đầu ra của mô hình MTL thông thường
    losses_t_dict = {
        "task_1": loss_t_values[0],
        "task_2": loss_t_values[1]
    }
    
    losses_t1_dict = {
        "task_1": loss_t1_values[0],
        "task_2": loss_t1_values[1]
    }
    
    # 2. Giả lập hàm forward_losses_fn
    # Hàm này sẽ được gọi bên trong on_train_batch_end và trả về Loss tại thời điểm t+1
    def dummy_forward_losses_fn(context):
        print("[MOCK] forward_losses_fn được gọi để lấy Loss_t+1")
        return losses_t1_dict

    # 3. Khởi tạo class FAMOWeighting
    famo = FAMOWeighting(
        n_tasks=n_tasks,
        device=device,
        task_names=task_names,
        forward_losses_fn=dummy_forward_losses_fn,
        w_lr=0.025,
        gamma=1e-3
    )
    
    # Giả lập model rỗng và context rỗng phục vụ test
    dummy_model = nn.Linear(10, 2).to(device)
    dummy_context = {"batch_idx": 42} 

    print("\n--- BƯỚC 1: Gọi on_before_backward (Nạp Loss_t) ---")
    famo.on_before_backward(
        losses=losses_t_dict,
        model=dummy_model,
        forward_context=dummy_context
    )
    print("Đã lưu _current_loss_t:", famo._current_loss_t)

    print("\n--- BƯỚC 2: Gọi on_train_batch_end (Tính toán Delta & Update Weight) ---")
    # In ra giá trị w trước khi update để đối chiếu
    print("w ban đầu:", famo.w.data)
    
    # Gọi hàm test chính
    famo.on_train_batch_end(
        losses=losses_t_dict, # Lưu ý: hàm thực tế lấy data mới từ forward_losses_fn(ctx)
        model=dummy_model
    )
    
    print("\n--- BƯỚC 3: Kiểm tra tracking sau khi kết thúc batch ---")
    print("Loss_t đã được lưu lại ở biến tracking:", famo._last_loss_t)
    print("Loss_t+1 đã được lưu lại ở biến tracking:", famo._last_loss_t1)
    print("w sau khi update:", famo.w.data)
    
    print("\n--- HOÀN THÀNH TEST ---")