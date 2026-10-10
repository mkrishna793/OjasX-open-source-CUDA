"""OjasX Smart Workload Router & Selective Execution Cost Engine (`ojasx2.smart_router`).

Automatically inspects any PyTorch `nn.Module` and decides per-layer:
  1. What NOT to execute / save (`SKIP_BACKWARD_FROZEN_WEIGHTS` for LoRA / frozen backbones)
  2. When to use C++17 1-Bit Ballot & L1 Kernels (`OJASX_CPP_BITPACKED_1BIT`, `OJASX_CPP_L1_RMSNORM`)
  3. When to use Holonomic Math Engine (`OJASX_HOLONOMIC_FUSED`, `OJASX_SELECTIVE_VOCAB_HEAD`)
  4. When to keep Native PyTorch Fast-Path (`NATIVE_PYTORCH_FAST_BYPASS` for tiny tensors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .holonomic_math_compiler import (
    HolonomicGeGLU,
    HolonomicLinear,
    HolonomicRMSNorm,
    HolonomicSwiGLU,
)
from .tile_hal import (
    HolonomicBitPackedLinear,
    HolonomicNativeRMSNorm,
    SiliconCapabilityProfile,
    TileSchedulePlanner,
    probe_hardware_profile,
)


@dataclass
class SmartLayerDecision:
    module_name: str
    original_type: str
    decision: str
    replacement_type: str
    trainable_params: int
    frozen_params: int
    tile_schedule: str
    reason: str
    estimated_saved_bytes_per_token: float


@dataclass
class SmartWorkloadReport:
    workload_profile: str
    silicon_profile: SiliconCapabilityProfile
    total_params: int
    trainable_params: int
    frozen_params: int
    decisions: List[SmartLayerDecision] = field(default_factory=list)

    @property
    def action_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.decisions:
            counts[d.decision] = counts.get(d.decision, 0) + 1
        return counts

    def format_table(self) -> str:
        lines = [
            "=" * 98,
            f"OjasX Smart Workload Router — Detected Profile: {self.workload_profile}",
            f"Silicon Target: {self.silicon_profile.chip_class} | L1d: {self.silicon_profile.l1d_cache_bytes // 1024} KB | "
            f"L2: {self.silicon_profile.l2_cache_bytes // (1024 * 1024)} MB | AMX: {self.silicon_profile.has_apple_amx}",
            f"Parameters: Total={self.total_params:,} | Trainable={self.trainable_params:,} | Frozen={self.frozen_params:,}",
            "-" * 98,
            f"{'Module':<28} | {'Decision':<28} | {'Replacement':<24} | {'Tile (M,N,K)':<14}",
            "-" * 98,
        ]
        for d in self.decisions:
            lines.append(
                f"{d.module_name[:28]:<28} | {d.decision:<28} | {d.replacement_type[:24]:<24} | {d.tile_schedule:<14}"
            )
        lines.append("=" * 98)
        return "\n".join(lines)


class _FrozenLinearZeroTapeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Never save `x` on the autograd tape because `weight` is frozen (requires_grad=False)!
        ctx.weight = weight.detach()
        return torch.nn.functional.linear(x, ctx.weight, bias.detach() if bias is not None else None)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        dx = torch.matmul(grad_out, ctx.weight) if ctx.needs_input_grad[0] else None
        return dx, None, None


class FrozenLinearZeroTape(nn.Module):
    """Zero-Tape Linear wrapper for frozen backbone layers (`weight.requires_grad=False`).

    Never saves input `x` on the autograd tape because `dW` is not needed!
    Only propagates `dX = grad_out @ W` when the upstream input itself requires grad.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FrozenLinearZeroTapeFn.apply(x, self.weight, self.bias)


class SmartWorkloadRouter:
    """Inspects and selectively optimizes PyTorch models using hardware & mathematical cost rules."""

    def __init__(
        self,
        tiny_param_threshold: int = 1024,
        large_vocab_threshold: int = 8192,
    ) -> None:
        self.tiny_param_threshold = tiny_param_threshold
        self.large_vocab_threshold = large_vocab_threshold
        self.silicon = probe_hardware_profile()
        self.planner = TileSchedulePlanner(self.silicon)

    def detect_workload_profile(self, model: nn.Module) -> Tuple[str, int, int]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        if not model.training:
            return "INFERENCE_EVAL", trainable_params, frozen_params
        if total_params > 0 and (frozen_params / total_params) >= 0.50:
            return "LORA_FROZEN_BACKBONE", trainable_params, frozen_params

        has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
        has_llm_ops = any(
            isinstance(m, (HolonomicSwiGLU, HolonomicGeGLU, HolonomicRMSNorm, HolonomicNativeRMSNorm))
            or "rms" in m.__class__.__name__.lower()
            or "swiglu" in m.__class__.__name__.lower()
            or (isinstance(m, nn.Linear) and m.out_features >= self.large_vocab_threshold)
            for m in model.modules()
        )
        if has_llm_ops:
            return "LLM_TRANSFORMER", trainable_params, frozen_params
        if has_conv:
            return "VISION_CONV_MLP", trainable_params, frozen_params
        return "GENERAL_DENSE_WORKLOAD", trainable_params, frozen_params

    def optimize_model(
        self,
        model: nn.Module,
        batch_tokens: int = 512,
        prefer_1bit_relu2_for_relu: bool = False,
    ) -> Tuple[nn.Module, SmartWorkloadReport]:
        profile_name, trainable_total, frozen_total = self.detect_workload_profile(model)
        report = SmartWorkloadReport(
            workload_profile=profile_name,
            silicon_profile=self.silicon,
            total_params=trainable_total + frozen_total,
            trainable_params=trainable_total,
            frozen_params=frozen_total,
        )

        self._rewrite_children(
            model,
            prefix="",
            batch_tokens=batch_tokens,
            prefer_1bit_relu2_for_relu=prefer_1bit_relu2_for_relu,
            report=report,
        )
        return model, report

    def _rewrite_children(
        self,
        parent: nn.Module,
        prefix: str,
        batch_tokens: int,
        prefer_1bit_relu2_for_relu: bool,
        report: SmartWorkloadReport,
    ) -> None:
        if isinstance(parent, nn.Sequential):
            self._fuse_sequential_in_place(parent, prefix, batch_tokens, prefer_1bit_relu2_for_relu, report)

        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (HolonomicBitPackedLinear, HolonomicLinear, HolonomicNativeRMSNorm, FrozenLinearZeroTape)):
                continue

            if isinstance(child, nn.Linear):
                replacement, decision = self._decide_linear(child, full_name, batch_tokens)
                if replacement is not child:
                    setattr(parent, name, replacement)
                report.decisions.append(decision)
            elif child.__class__.__name__.lower() in ("rmsnorm", "holonomicrmsnorm") and hasattr(child, "weight"):
                dim = int(child.weight.numel())
                eps = float(getattr(child, "eps", 1e-6))
                rep = HolonomicNativeRMSNorm(dim, eps=eps, device=child.weight.device, dtype=child.weight.dtype)
                rep.weight = child.weight
                setattr(parent, name, rep)
                report.decisions.append(
                    SmartLayerDecision(
                        module_name=full_name,
                        original_type=child.__class__.__name__,
                        decision="OJASX_CPP_L1_RMSNORM",
                        replacement_type="HolonomicNativeRMSNorm",
                        trainable_params=dim if child.weight.requires_grad else 0,
                        frozen_params=0 if child.weight.requires_grad else dim,
                        tile_schedule=f"(1,{dim},1)",
                        reason="Single-pass L1 C++ RMSNorm saving 4B scalar mu2",
                        estimated_saved_bytes_per_token=float(dim * 4 - 4),
                    )
                )
            else:
                self._rewrite_children(child, full_name, batch_tokens, prefer_1bit_relu2_for_relu, report)

    def _fuse_sequential_in_place(
        self,
        seq: nn.Sequential,
        prefix: str,
        batch_tokens: int,
        prefer_1bit_relu2_for_relu: bool,
        report: SmartWorkloadReport,
    ) -> None:
        children = list(seq.named_children())
        i = 0
        while i < len(children) - 1:
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            full_name = f"{prefix}.{name_a}" if prefix else name_a

            if isinstance(mod_a, nn.Linear) and isinstance(mod_b, (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU)):
                params = mod_a.in_features * mod_a.out_features
                w_grad = mod_a.weight.requires_grad
                sched = self.planner.plan_gemm_tile(
                    batch_tokens, mod_a.out_features, mod_a.in_features,
                    device_type=mod_a.weight.device.type,
                )
                tile_str = f"({sched.tile_m},{sched.tile_n},{sched.tile_k})"

                if params >= self.tiny_param_threshold and w_grad:
                    if isinstance(mod_b, nn.ReLU) and prefer_1bit_relu2_for_relu:
                        rep = HolonomicBitPackedLinear(
                            mod_a.in_features, mod_a.out_features,
                            bias=mod_a.bias is not None,
                            device=mod_a.weight.device, dtype=mod_a.weight.dtype,
                        )
                        rep.weight = mod_a.weight
                        if mod_a.bias is not None:
                            rep.bias = mod_a.bias
                        setattr(seq, name_a, rep)
                        setattr(seq, name_b, nn.Identity())
                        report.decisions.append(
                            SmartLayerDecision(
                                module_name=full_name,
                                original_type=f"Linear+{mod_b.__class__.__name__}",
                                decision="OJASX_CPP_BITPACKED_1BIT",
                                replacement_type="HolonomicBitPackedLinear",
                                trainable_params=params,
                                frozen_params=0,
                                tile_schedule=tile_str,
                                reason="C++ AMX + 1-Bit Ballot Sign Mask (32x compression)",
                                estimated_saved_bytes_per_token=float(mod_a.out_features * 4 * (31.0 / 32.0)),
                            )
                        )
                        i += 2
                        continue

                    act_map = {
                        nn.GELU: "gelu_tanh",
                        nn.SiLU: "silu",
                        nn.ReLU: "relu",
                        nn.LeakyReLU: "leaky_relu",
                    }
                    act_name = act_map[type(mod_b)]
                    rep = HolonomicLinear(
                        mod_a.in_features, mod_a.out_features,
                        bias=mod_a.bias is not None,
                        act=act_name,
                        device=mod_a.weight.device, dtype=mod_a.weight.dtype,
                    )
                    rep.weight = mod_a.weight
                    if mod_a.bias is not None:
                        rep.bias = mod_a.bias
                    setattr(seq, name_a, rep)
                    setattr(seq, name_b, nn.Identity())
                    report.decisions.append(
                        SmartLayerDecision(
                            module_name=full_name,
                            original_type=f"Linear+{mod_b.__class__.__name__}",
                            decision="OJASX_HOLONOMIC_FUSED",
                            replacement_type=f"HolonomicLinear({act_name})",
                            trainable_params=params,
                            frozen_params=0,
                            tile_schedule=tile_str,
                            reason="Fused Linear+Activation eliminating pre-activation tensor",
                            estimated_saved_bytes_per_token=float(mod_a.out_features * 4),
                        )
                    )
                    i += 2
                    continue
            i += 1

    def _decide_linear(
        self,
        linear: nn.Linear,
        full_name: str,
        batch_tokens: int,
    ) -> Tuple[nn.Module, SmartLayerDecision]:
        params = linear.in_features * linear.out_features
        if linear.bias is not None:
            params += linear.out_features
        w_grad = linear.weight.requires_grad or (linear.bias is not None and linear.bias.requires_grad)

        sched = self.planner.plan_gemm_tile(
            batch_tokens, linear.out_features, linear.in_features,
            device_type=linear.weight.device.type,
        )
        tile_str = f"({sched.tile_m},{sched.tile_n},{sched.tile_k})"

        # Rule 1: What NOT to execute / save — Frozen Backbone Layer!
        if not w_grad:
            rep = FrozenLinearZeroTape(linear)
            return rep, SmartLayerDecision(
                module_name=full_name,
                original_type="nn.Linear (Frozen)",
                decision="SKIP_BACKWARD_FROZEN_WEIGHTS",
                replacement_type="FrozenLinearZeroTape",
                trainable_params=0,
                frozen_params=params,
                tile_schedule=tile_str,
                reason="Weight requires_grad=False: skip saving input X on autograd tape & skip dW GEMM",
                estimated_saved_bytes_per_token=float(linear.in_features * 4),
            )

        # Rule 2: Tiny Tensor Fast-Bypass — Never add wrapper overhead to tiny layers!
        if (linear.in_features * linear.out_features) < self.tiny_param_threshold:
            return linear, SmartLayerDecision(
                module_name=full_name,
                original_type="nn.Linear (Tiny)",
                decision="NATIVE_PYTORCH_FAST_BYPASS",
                replacement_type="nn.Linear (Unchanged)",
                trainable_params=params,
                frozen_params=0,
                tile_schedule=tile_str,
                reason=f"Matrix size ({linear.in_features}x{linear.out_features}) < {self.tiny_param_threshold}: keep native C++ fast-path",
                estimated_saved_bytes_per_token=0.0,
            )

        # Rule 3: Trainable Dense Linear -> HolonomicLinear
        rep = HolonomicLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            act="none",
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        rep.weight = linear.weight
        if linear.bias is not None:
            rep.bias = linear.bias
        return rep, SmartLayerDecision(
            module_name=full_name,
            original_type="nn.Linear",
            decision="OJASX_HOLONOMIC_LINEAR",
            replacement_type="HolonomicLinear(none)",
            trainable_params=params,
            frozen_params=0,
            tile_schedule=tile_str,
            reason="L1/L2 tiled Holonomic pullback",
            estimated_saved_bytes_per_token=float(linear.out_features * 2),
        )

