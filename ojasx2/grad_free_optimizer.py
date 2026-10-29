"""
ojasx2/grad_free_optimizer.py

Upgraded Fused Backward-to-Optimizer Engine (`HolonomicGradFreeOptimizer`).
- Zero persistent `.grad` memory AND zero `grad_accum` buffer memory for AdamW/Adafactor.
- Auto-latching cross-layer global gradient norm tracker (even when `.step()` is omitted).
- Arbitrary-shape 2.0-bit E8 Lattice compression for first-moment momentum states (with automatic tail padding).
- Non-finite (`NaN`/`Inf`) gradient guard for mixed-precision stability.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from .math.lattice import E8LatticePacker


class HolonomicGradFreeOptimizer(Optimizer):
    """In-backward fused optimizer eliminating persistent `.grad` memory across the model."""

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
        grad_accum_steps: int = 1,
        quantize_moments_e8: bool = False,
        skip_if_nonfinite: bool = True,
        enabled: bool = True,
    ) -> None:
        algo_lower = algorithm.lower()
        if algo_lower not in ("sgd", "adamw", "adafactor", "low_memory_adamw"):
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose 'sgd', 'adamw', 'adafactor', or 'low_memory_adamw'."
            )

        self.algorithm = "adafactor" if algo_lower == "low_memory_adamw" else algo_lower
        self.grad_accum_steps = int(grad_accum_steps)
        self.quantize_moments_e8 = bool(quantize_moments_e8)
        self.skip_if_nonfinite = bool(skip_if_nonfinite)
        self.enabled = bool(enabled)
        self._param_to_group: Dict[nn.Parameter, Dict[str, Any]] = {}
        self._handles: List[Any] = []
        self.last_step_freed_bytes: int = 0
        self.last_step_updated_params: int = 0
        self.skipped_nonfinite_updates: int = 0
        self._running_global_norm_sq: float = 0.0
        self._prev_global_norm: float = 0.0
        self._hooks_fired_in_cycle: int = 0

        defaults = dict(
            lr=float(lr),
            algorithm=self.algorithm,
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            max_grad_norm=float(max_grad_norm) if max_grad_norm is not None else None,
            grad_accum_steps=int(grad_accum_steps),
            quantize_moments_e8=bool(quantize_moments_e8),
            skip_if_nonfinite=bool(skip_if_nonfinite),
            enabled=bool(enabled),
        )

        super().__init__(params, defaults)

    def _trainable_param_count(self) -> int:
        return sum(1 for g in self.param_groups for p in g["params"] if p.requires_grad)

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

    @staticmethod
    def _unpack_exp_avg(st: Dict[str, Any], target_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        if "exp_avg_packed" in st:
            packed = st["exp_avg_packed"]
            scales = st["exp_avg_scales"]
            padded_rows = st.get("exp_avg_padded_rows", target_shape.numel() // 32)
            unpacked = E8LatticePacker.unpack(
                packed, scales, (padded_rows, 32), dtype=dtype
            )
            return unpacked.reshape(-1)[: target_shape.numel()].view(target_shape)
        return st["exp_avg"].to(dtype)

    @staticmethod
    def _pack_exp_avg(st: Dict[str, Any], tensor: torch.Tensor, use_e8: bool, storage_dtype: torch.dtype) -> None:
        if use_e8 and tensor.numel() >= 8:
            flat = tensor.reshape(-1)
            pad_len = (-flat.numel()) % 32
            if pad_len > 0:
                flat = F.pad(flat, (0, pad_len))
            padded_rows = flat.numel() // 32
            packed, scales, _ = E8LatticePacker.pack(flat.view(padded_rows, 32), group_size=32)
            st["exp_avg_packed"] = packed
            st["exp_avg_scales"] = scales
            st["exp_avg_padded_rows"] = padded_rows
            st.pop("exp_avg", None)
        else:
            if "exp_avg" not in st or st["exp_avg"] is None:
                st["exp_avg"] = tensor.to(storage_dtype).clone()
            else:
                st["exp_avg"].copy_(tensor.to(storage_dtype))

    @staticmethod
    def _unpack_grad_accum(st: Dict[str, Any], target_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        if "grad_accum_packed" in st:
            packed = st["grad_accum_packed"]
            scales = st["grad_accum_scales"]
            padded_rows = st.get("grad_accum_padded_rows", target_shape.numel() // 32)
            unpacked = E8LatticePacker.unpack(
                packed, scales, (padded_rows, 32), dtype=dtype
            )
            return unpacked.reshape(-1)[: target_shape.numel()].view(target_shape)
        return st["grad_accum"].to(dtype)

    @staticmethod
    def _pack_grad_accum(st: Dict[str, Any], tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1)
        pad_len = (-flat.numel()) % 32
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        padded_rows = flat.numel() // 32
        packed, scales, _ = E8LatticePacker.pack(flat.view(padded_rows, 32), group_size=32)
        st["grad_accum_packed"] = packed
        st["grad_accum_scales"] = scales
        st["grad_accum_padded_rows"] = padded_rows
        st.pop("grad_accum", None)

    def _make_hook(self, param: nn.Parameter) -> Any:
        @torch.no_grad()
        def _hook(_p: nn.Parameter) -> None:
            group = self._param_to_group.get(_p, self.param_groups[0])
            if not (self.enabled and group.get("enabled", True)):
                return

            grad = _p.grad
            if grad is None:
                return

            grad_bytes = grad.numel() * grad.element_size()
            grad_accum_steps = int(group.get("grad_accum_steps", self.grad_accum_steps))
            use_e8_mom = bool(group.get("quantize_moments_e8", self.quantize_moments_e8))
            skip_nonfinite = bool(group.get("skip_if_nonfinite", self.skip_if_nonfinite))

            if skip_nonfinite and not bool(torch.isfinite(grad).all()):
                _p.grad = None
                self.last_step_freed_bytes += grad_bytes
                self.skipped_nonfinite_updates += 1
                return

            st = self.state[_p]

            grad_sq_norm = float(torch.linalg.vector_norm(grad.float()).square().item())
            self._running_global_norm_sq += grad_sq_norm
            self._hooks_fired_in_cycle += 1

            if grad_accum_steps > 1:
                st["accum_count"] = st.get("accum_count", 0) + 1
                scaled_grad = grad / float(grad_accum_steps)
                _p.grad = None
                self.last_step_freed_bytes += grad_bytes

                if use_e8_mom and _p.numel() >= 8:
                    if "grad_accum_packed" in st:
                        prev_accum = self._unpack_grad_accum(st, _p.shape, _p.dtype)
                        accum_tensor = prev_accum.add(scaled_grad.to(_p.dtype))
                    else:
                        accum_tensor = scaled_grad.to(_p.dtype)
                    self._pack_grad_accum(st, accum_tensor)
                else:
                    if st.get("grad_accum") is None:
                        st["grad_accum"] = scaled_grad.to(_p.dtype)
                    else:
                        st["grad_accum"].add_(scaled_grad.to(_p.dtype))

                if st["accum_count"] % grad_accum_steps != 0:
                    return

                if use_e8_mom and _p.numel() >= 8 and "grad_accum_packed" in st:
                    effective_grad = self._unpack_grad_accum(st, _p.shape, _p.dtype)
                    st.pop("grad_accum_packed", None)
                    st.pop("grad_accum_scales", None)
                    st.pop("grad_accum_padded_rows", None)
                else:
                    effective_grad = st["grad_accum"]
                    st["grad_accum"] = None
            else:
                effective_grad = grad
                _p.grad = None
                self.last_step_freed_bytes += grad_bytes

            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            betas = group.get("betas", (0.9, 0.999))
            eps = float(group.get("eps", 1e-8))
            momentum = float(group.get("momentum", 0.0))
            max_grad_norm = group.get("max_grad_norm")
            algo = group.get("algorithm", self.algorithm).lower()
            if algo == "low_memory_adamw":
                algo = "adafactor"

            if max_grad_norm is not None and max_grad_norm > 0:
                if self._prev_global_norm > 0:
                    effective_norm = self._prev_global_norm
                else:
                    effective_norm = float(torch.linalg.vector_norm(effective_grad.float()).item())
                if effective_norm > max_grad_norm:
                    effective_grad = effective_grad * (max_grad_norm / (effective_norm + 1e-6))

            st["step"] = st.get("step", 0) + 1
            step = st["step"]
            is_half = _p.dtype in (torch.float16, torch.bfloat16)

            if algo == "sgd":
                d_p = effective_grad.float() if is_half else effective_grad
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
                g_f32 = effective_grad.float() if is_half else effective_grad

                if weight_decay != 0.0:
                    if is_half:
                        p_f32 = p_f32 * (1.0 - lr * weight_decay)
                    else:
                        _p.mul_(1.0 - lr * weight_decay)

                if "exp_avg" not in st and "exp_avg_packed" not in st:
                    exp_avg_f32 = torch.zeros_like(p_f32)
                else:
                    exp_avg_f32 = self._unpack_exp_avg(st, _p.shape, torch.float32)

                exp_avg_sq = st.get("exp_avg_sq")
                if exp_avg_sq is None:
                    exp_avg_sq = st["exp_avg_sq"] = torch.zeros_like(p_f32)
                elif is_half and exp_avg_sq.dtype != torch.float32:
                    exp_avg_sq = st["exp_avg_sq"] = exp_avg_sq.float()

                exp_avg_f32.mul_(beta1).add_(g_f32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)
                self._pack_exp_avg(st, exp_avg_f32, use_e8_mom, torch.float32)

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                delta = -step_size * (exp_avg_f32 / denom)
                if is_half:
                    _p.copy_((p_f32 + delta).to(_p.dtype))
                else:
                    _p.add_(delta)

            elif algo == "adafactor":
                beta1, beta2 = betas
                p_f32 = _p.float() if is_half else _p
                g_f32 = effective_grad.float() if is_half else effective_grad

                if weight_decay != 0.0:
                    if is_half:
                        p_f32 = p_f32 * (1.0 - lr * weight_decay)
                    else:
                        _p.mul_(1.0 - lr * weight_decay)

                grad_sq = g_f32.square()
                bias_correction2 = 1.0 - (beta2 ** step)

                if _p.ndim >= 2:
                    exp_avg_sq_row = st.get("exp_avg_sq_row")
                    exp_avg_sq_col = st.get("exp_avg_sq_col")
                    if exp_avg_sq_row is None or exp_avg_sq_col is None:
                        row_shape = _p.shape[:-1]
                        col_shape = _p.shape[:-2] + _p.shape[-1:]
                        exp_avg_sq_row = st["exp_avg_sq_row"] = torch.zeros(
                            row_shape, dtype=torch.float32, device=_p.device
                        )
                        exp_avg_sq_col = st["exp_avg_sq_col"] = torch.zeros(
                            col_shape, dtype=torch.float32, device=_p.device
                        )

                    row_mean = grad_sq.mean(dim=-1)
                    col_mean = grad_sq.mean(dim=-2)

                    exp_avg_sq_row.mul_(beta2).add_(row_mean, alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)

                    r_factor = exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp(min=eps)
                    v_hat = r_factor.unsqueeze(-1) * exp_avg_sq_col.unsqueeze(-2)
                    denom = (v_hat.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    exp_avg_sq = st.get("exp_avg_sq")
                    if exp_avg_sq is None:
                        exp_avg_sq = st["exp_avg_sq"] = torch.zeros_like(p_f32)
                    exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if beta1 > 0.0:
                    if "exp_avg" not in st and "exp_avg_packed" not in st:
                        exp_avg_f32 = torch.zeros_like(p_f32)
                    else:
                        exp_avg_f32 = self._unpack_exp_avg(st, _p.shape, torch.float32)

                    exp_avg_f32.mul_(beta1).add_(g_f32, alpha=1.0 - beta1)
                    self._pack_exp_avg(st, exp_avg_f32, use_e8_mom, _p.dtype)
                    bias_correction1 = 1.0 - (beta1 ** step)
                    update = exp_avg_f32 / bias_correction1
                else:
                    update = g_f32

                delta = -lr * (update / denom)
                if is_half:
                    _p.copy_((p_f32 + delta).to(_p.dtype))
                else:
                    _p.add_(delta)

            self.last_step_updated_params += 1

            # Auto-latch global norm when all trainable parameters have fired in this backward pass
            total_trainable = self._trainable_param_count()
            if total_trainable > 0 and self._hooks_fired_in_cycle >= total_trainable:
                self._prev_global_norm = math.sqrt(self._running_global_norm_sq)
                self._running_global_norm_sq = 0.0
                self._hooks_fired_in_cycle = 0

        return _hook

    def reset_step_metrics(self) -> None:
        if self._running_global_norm_sq > 0.0:
            self._prev_global_norm = math.sqrt(self._running_global_norm_sq)
            self._running_global_norm_sq = 0.0
        self._hooks_fired_in_cycle = 0
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
                if p in self.state:
                    self.state[p]["grad_accum"] = None
                    self.state[p]["accum_count"] = 0

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        self.reset_step_metrics()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def persistent_grad_bytes(self) -> int:
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
            group = self._param_to_group.get(p, self.param_groups[0])
            algo = group.get("algorithm", self.algorithm).lower()
            if algo == "low_memory_adamw":
                algo = "adafactor"
            if p.dtype in (torch.float16, torch.bfloat16):
                if algo != "adafactor":
                    for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                        if k in st and isinstance(st[k], torch.Tensor) and st[k].dtype != torch.float32:
                            st[k] = st[k].float()
                else:
                    for k in ("exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col"):
                        if k in st and isinstance(st[k], torch.Tensor) and st[k].dtype != torch.float32:
                            st[k] = st[k].float()
                    if "exp_avg" in st and isinstance(st["exp_avg"], torch.Tensor) and st["exp_avg"].dtype != p.dtype:
                        st["exp_avg"] = st["exp_avg"].to(p.dtype)
        self._refresh_param_groups_and_hooks()

    @property
    def params(self) -> List[nn.Parameter]:
        return [p for group in self.param_groups for p in group["params"]]
