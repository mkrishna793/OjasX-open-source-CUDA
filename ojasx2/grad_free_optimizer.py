"""
ojasx2/grad_free_optimizer.py

Fused Backward-to-Optimizer Engine (`HolonomicGradFreeOptimizer`).
Uses PyTorch's `register_post_accumulate_grad_hook` to immediately apply parameter updates
(SGD or AdamW) in-place as soon as each layer's cotangent contraction finishes during `.backward()`,
setting `param.grad = None` immediately so full-model `.grad` buffers never accumulate in RAM.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional
import torch
import torch.nn as nn


class HolonomicGradFreeOptimizer:
    """
    In-backward fused optimizer eliminating persistent `.grad` memory across the model.
    Supports exact mathematical parity with `torch.optim.SGD` and `torch.optim.AdamW`.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        algorithm: str = "adamw",
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        if algorithm not in ("sgd", "adamw"):
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'sgd' or 'adamw'.")
        self.params: List[nn.Parameter] = [p for p in params if p.requires_grad]
        self.param_groups: List[Dict[str, Any]] = [{"params": self.params}]
        self.lr = float(lr)
        self.algorithm = algorithm
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)

        self.state: Dict[nn.Parameter, Dict[str, Any]] = {}
        self._handles: List[Any] = []
        self.last_step_freed_bytes: int = 0
        self.last_step_updated_params: int = 0

        self._register_hooks()

    def _register_hooks(self) -> None:
        for p in self.params:
            self.state[p] = {"step": 0}
            handle = p.register_post_accumulate_grad_hook(self._make_hook())
            self._handles.append(handle)

    def _make_hook(self) -> Any:
        @torch.no_grad()
        def _hook(param: nn.Parameter) -> None:
            grad = param.grad
            if grad is None:
                return

            grad_bytes = grad.numel() * grad.element_size()
            st = self.state[param]
            st["step"] += 1
            step = st["step"]

            if self.algorithm == "sgd":
                d_p = grad
                if self.weight_decay != 0.0:
                    d_p = d_p.add(param, alpha=self.weight_decay)
                if self.momentum != 0.0:
                    buf = st.get("momentum_buffer")
                    if buf is None:
                        buf = st["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf.mul_(self.momentum).add_(d_p)
                    d_p = buf
                param.add_(d_p, alpha=-self.lr)

            elif self.algorithm == "adamw":
                beta1, beta2 = self.betas
                if self.weight_decay != 0.0:
                    param.mul_(1.0 - self.lr * self.weight_decay)

                exp_avg = st.get("exp_avg")
                exp_avg_sq = st.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = st["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    exp_avg_sq = st["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

            # Immediately release the parameter's gradient tensor!
            param.grad = None
            self.last_step_freed_bytes += grad_bytes
            self.last_step_updated_params += 1

        return _hook

    def reset_step_metrics(self) -> None:
        self.last_step_freed_bytes = 0
        self.last_step_updated_params = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.reset_step_metrics()
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self) -> None:
        """No-op for drop-in compatibility: updates are already applied in-place during backward hooks."""
        return None

    def persistent_grad_bytes(self) -> int:
        """Returns bytes currently occupied by persistent `.grad` tensors (should be 0 after backward)."""
        return sum(
            p.grad.numel() * p.grad.element_size()
            for p in self.params
            if p.grad is not None
        )

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
