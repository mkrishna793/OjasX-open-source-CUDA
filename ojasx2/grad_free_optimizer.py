"""
ojasx2/grad_free_optimizer.py

Fused Backward-to-Optimizer Engine (`HolonomicGradFreeOptimizer`).
Uses PyTorch's `register_post_accumulate_grad_hook` to immediately apply parameter updates
(SGD or AdamW) in-place as soon as each layer's cotangent contraction finishes during `.backward()`,
setting `param.grad = None` immediately so full-model `.grad` buffers never accumulate in RAM.

Upgraded for OjasX v0.4.0 (Requirement R1):
1. Subclasses `torch.optim.Optimizer` for 100% PyTorch ecosystem compatibility.
2. Supports multiple `param_groups`.
3. Dynamically inspects `group['lr']`, `group['weight_decay']`, etc., on every backward hook
   invocation, ensuring 100% interoperability with `torch.optim.lr_scheduler`.
4. In-hook gradient norm clipping (`max_grad_norm`).
5. Complete `state_dict()` and `load_state_dict()` serialization.
6. Safe FP16/BF16 mixed-precision training with master FP32 momentum buffers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer


class HolonomicGradFreeOptimizer(Optimizer):
    """
    In-backward fused optimizer eliminating persistent `.grad` memory across the model.
    Subclasses `torch.optim.Optimizer` and provides mathematical parity with `torch.optim.SGD`
    and `torch.optim.AdamW` while keeping memory footprint at zero for gradient tensors.
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        algorithm: str = "adamw",
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        if algorithm not in ("sgd", "adamw"):
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'sgd' or 'adamw'.")

        self.algorithm = algorithm
        self._param_to_group: Dict[nn.Parameter, Dict[str, Any]] = {}
        self._handles: List[Any] = []
        self.last_step_freed_bytes: int = 0
        self.last_step_updated_params: int = 0
        defaults = dict(
            lr=float(lr),
            algorithm=algorithm,
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            max_grad_norm=float(max_grad_norm) if max_grad_norm is not None else None,
        )

        super().__init__(params, defaults)

    def _refresh_param_groups_and_hooks(self) -> None:
        self.remove_hooks()
        self._param_to_group.clear()

        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, nn.Parameter):
                    raise TypeError(f"optimizer can only optimize Parameters, got {type(p)}")
                self._param_to_group[p] = group
                if p.requires_grad:
                    if p not in self.state:
                        self.state[p] = {"step": 0}
                    handle = p.register_post_accumulate_grad_hook(self._make_hook(p))
                    self._handles.append(handle)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group["params"]:
            self._param_to_group[p] = group
            if p.requires_grad:
                if p not in self.state:
                    self.state[p] = {"step": 0}
                handle = p.register_post_accumulate_grad_hook(self._make_hook(p))
                self._handles.append(handle)

    def _make_hook(self, param: nn.Parameter) -> Any:
        @torch.no_grad()
        def _hook(_p: nn.Parameter) -> None:
            grad = _p.grad
            if grad is None:
                return

            grad_bytes = grad.numel() * grad.element_size()
            group = self._param_to_group.get(_p, self.param_groups[0])

            # Dynamic hyperparameter retrieval (100% lr_scheduler compatible!)
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            betas = group.get("betas", (0.9, 0.999))
            eps = float(group.get("eps", 1e-8))
            momentum = float(group.get("momentum", 0.0))
            max_grad_norm = group.get("max_grad_norm")
            algo = group.get("algorithm", self.algorithm)

            # In-hook gradient norm clipping
            if max_grad_norm is not None and max_grad_norm > 0:
                norm = torch.linalg.vector_norm(grad)
                if norm > max_grad_norm:
                    grad = grad * (max_grad_norm / (norm + 1e-6))

            st = self.state[_p]
            st["step"] = st.get("step", 0) + 1
            step = st["step"]

            is_half = _p.dtype in (torch.float16, torch.bfloat16)

            if algo == "sgd":
                d_p = grad.float() if is_half else grad
                p_data = _p.float() if is_half else _p

                if weight_decay != 0.0:
                    d_p = d_p.add(p_data, alpha=weight_decay)

                if momentum != 0.0:
                    buf = st.get("momentum_buffer")
                    if buf is None:
                        buf = st["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                if is_half:
                    _p.copy_((p_data - lr * d_p).to(_p.dtype))
                else:
                    _p.add_(d_p, alpha=-lr)

            elif algo == "adamw":
                beta1, beta2 = betas
                p_f32 = _p.float() if is_half else _p
                g_f32 = grad.float() if is_half else grad

                if weight_decay != 0.0:
                    if is_half:
                        p_f32 = p_f32 * (1.0 - lr * weight_decay)
                    else:
                        _p.mul_(1.0 - lr * weight_decay)

                exp_avg = st.get("exp_avg")
                exp_avg_sq = st.get("exp_avg_sq")
                if exp_avg is None:
                    # Allocate FP32 master momentum buffers to avoid underflow
                    exp_avg = st["exp_avg"] = torch.zeros_like(
                        p_f32, memory_format=torch.preserve_format
                    )
                    exp_avg_sq = st["exp_avg_sq"] = torch.zeros_like(
                        p_f32, memory_format=torch.preserve_format
                    )
                elif is_half and exp_avg.dtype != torch.float32:
                    exp_avg = st["exp_avg"] = exp_avg.float()
                    exp_avg_sq = st["exp_avg_sq"] = exp_avg_sq.float()

                exp_avg.mul_(beta1).add_(g_f32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                delta = -step_size * (exp_avg / denom)
                if is_half:
                    _p.copy_((p_f32 + delta).to(_p.dtype))
                else:
                    _p.add_(delta)

            # Immediately release parameter gradient tensor from RAM!
            _p.grad = None
            self.last_step_freed_bytes += grad_bytes
            self.last_step_updated_params += 1

        return _hook

    def reset_step_metrics(self) -> None:
        self.last_step_freed_bytes = 0
        self.last_step_updated_params = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.reset_step_metrics()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Updates applied during backward hook. Calling step() is a valid no-op for API compatibility."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def persistent_grad_bytes(self) -> int:
        """Returns bytes currently occupied by persistent `.grad` tensors (should be 0 after backward)."""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total += p.grad.numel() * p.grad.element_size()
        return total

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        for p, st in self.state.items():
            if p.dtype in (torch.float16, torch.bfloat16):
                for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    if k in st and isinstance(st[k], torch.Tensor) and st[k].dtype != torch.float32:
                        st[k] = st[k].float()
        self._refresh_param_groups_and_hooks()

    @property
    def params(self) -> List[nn.Parameter]:
        """Convenience property returning all trainable parameters across all groups."""
        res = []
        for group in self.param_groups:
            for p in group["params"]:
                res.append(p)
        return res
