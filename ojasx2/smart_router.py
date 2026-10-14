"""OjasX Smart Workload Router & Selective Execution Cost Engine (`ojasx2.smart_router`).

Automatically inspects any PyTorch `nn.Module` and decides per-layer:
  1. What NOT to execute / save (`SKIP_BACKWARD_FROZEN_WEIGHTS` for LoRA / frozen backbones)
  2. When to use C++17 1-Bit Ballot & L1 Kernels (`OJASX_CPP_BITPACKED_1BIT`, `OJASX_CPP_L1_RMSNORM`)
  3. When to use Universal Holonomic GLU Engine:
     - All 10 variants: SwiGLU, GeGLU, ReGLU, ReLU2GLU, SigmoidGLU, BilinearGLU, QuickGeGLU, MishGLU, TanhGLU, LeakyReGLU
     - Both layouts: Split-projection (`gate_proj` + `up_proj` + `down_proj`) and Chunked-projection (`fc_in` -> 2H chunk -> `fc_out` / `nn.GLU`)
  4. When to rewrite normalization layers (`OJASX_HOLONOMIC_LAYERNORM`, `OJASX_HOLONOMIC_GROUPNORM`)
  5. When to wrap frozen convolutions with zero activation tape (`SKIP_BACKWARD_FROZEN_CONV`)
  6. When to replace activations with 1-bit masks (`OJASX_PACKED_RELU`, `OJASX_HOLONOMIC_ACTIVATION`)
  7. When to adaptively optimize narrow / tabular MLPs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .holonomic_layers import (
    FrozenConvZeroTape,
    HolonomicActivationLens,
    HolonomicFusedBilinearGLUMLP,
    HolonomicFusedChunkedGLUMLP,
    HolonomicFusedGLUMLP,
    HolonomicFusedGeGLUMLP,
    HolonomicFusedReGLUMLP,
    HolonomicFusedReLU2GLUMLP,
    HolonomicFusedSigmoidGLUMLP,
    HolonomicFusedSwiGLUMLP,
    HolonomicGLU,
    HolonomicGroupNorm,
    HolonomicLayerNorm,
    PackedReLU,
)
from .holonomic_math_compiler import (
    HolonomicGeGLU,
    HolonomicLinear,
    HolonomicRMSNorm,
    HolonomicSwiGLU,
)
from .tile_hal import (
    HolonomicBitPackedLinear,
    HolonomicNativeRMSNorm,
    HolonomicUnifiedTransformerMLP,
    HolonomicUnifiedTransformerMLPFunction,
    SiliconCapabilityProfile,
    TileSchedulePlanner,
    probe_hardware_profile,
)
from .dagger_attention import HolonomicDaggerAttention
from .fused_cross_entropy import HolonomicFusedLinearCrossEntropy


class HolonomicFusedLinearCrossEntropyHead(HolonomicFusedLinearCrossEntropy):
    """Smart router drop-in replacement for LM head.

    Supports both:
    1. Unsupervised / inference call: `logits = head(h)` (returns standard linear projection)
    2. Supervised training call: `loss = head(h, targets)` (returns fused streaming cross-entropy with 0 DRAM logits)
    """

    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if targets is None:
            import torch.nn.functional as F
            return F.linear(h, self.weight, self.bias)
        return super().forward(h, targets, return_lse=return_lse)


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
    mlir_pipeline: Optional[Dict[str, object]] = None

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
        ctx.save_for_backward(weight)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (weight,) = ctx.saved_tensors
        dx = torch.matmul(grad_out, weight) if ctx.needs_input_grad[0] else None
        return dx, None, None


class FrozenLinearZeroTape(nn.Module):
    """Zero-Tape Linear wrapper for frozen backbone layers (`weight.requires_grad=False`)."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FrozenLinearZeroTapeFn.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None} (FrozenZeroTape)"


def _extract_glu_activation(module: nn.Module) -> Tuple[str, float]:
    """Inspects module activation attributes and resolves to one of the 10 GLU variant strings."""
    act_fn = (
        getattr(module, "act_fn", None)
        or getattr(module, "activation", None)
        or getattr(module, "act", None)
    )
    neg_slope = 0.01

    if act_fn is not None:
        if isinstance(act_fn, str):
            act_str = act_fn.lower()
        else:
            act_str = act_fn.__class__.__name__.lower()
            neg_slope = float(getattr(act_fn, "negative_slope", 0.01))

        if "silu" in act_str or "swish" in act_str:
            return "swiglu", neg_slope
        elif "gelu" in act_str:
            approx = getattr(act_fn, "approximate", "none") if not isinstance(act_fn, str) else "none"
            return ("geglu", neg_slope) if approx == "tanh" else ("gelu_exact", neg_slope)
        elif "quickgelu" in act_str or "quickgeglu" in act_str:
            return "quickgeglu", neg_slope
        elif "relu2" in act_str or "squared" in act_str:
            return "relu2glu", neg_slope
        elif "leaky" in act_str:
            return "leaky_reglu", neg_slope
        elif "relu" in act_str:
            return "reglu", neg_slope
        elif "sigmoid" in act_str:
            return "sigmoidglu", neg_slope
        elif "bilinear" in act_str or "identity" in act_str:
            return "bilinearglu", neg_slope
        elif "mish" in act_str:
            return "mishglu", neg_slope
        elif "tanh" in act_str:
            return "tanhglu", neg_slope

    mod_name = module.__class__.__name__.lower()
    if "geglu" in mod_name:
        return "geglu", neg_slope
    elif "reglu" in mod_name:
        return "reglu", neg_slope
    elif "mish" in mod_name:
        return "mishglu", neg_slope
    elif "tanh" in mod_name:
        return "tanhglu", neg_slope

    # Default for modern LLMs (LLaMA, Mistral, Qwen) is SwiGLU
    return "swiglu", neg_slope


class SmartWorkloadRouter:
    """Inspects and selectively optimizes PyTorch models across all 5 major workload families."""

    def __init__(
        self,
        tiny_param_threshold: int = 1024,
        large_vocab_threshold: int = 8192,
        prefer_unified_mlp: bool = False,
    ) -> None:
        self.tiny_param_threshold = tiny_param_threshold
        self.large_vocab_threshold = large_vocab_threshold
        self.prefer_unified_mlp = prefer_unified_mlp
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
            isinstance(
                m,
                (
                    HolonomicSwiGLU,
                    HolonomicGeGLU,
                    HolonomicRMSNorm,
                    HolonomicNativeRMSNorm,
                    HolonomicGLU,
                    HolonomicFusedGLUMLP,
                    HolonomicFusedChunkedGLUMLP,
                    HolonomicDaggerAttention,
                    HolonomicFusedLinearCrossEntropy,
                    HolonomicUnifiedTransformerMLP,
                ),
            )
            or isinstance(m, nn.GLU)
            or "rms" in m.__class__.__name__.lower()
            or "glu" in m.__class__.__name__.lower()
            or "attention" in m.__class__.__name__.lower()
            or (isinstance(m, nn.Linear) and m.out_features >= self.large_vocab_threshold)
            or any(
                hasattr(m, g) and hasattr(m, u)
                for g, u in [
                    ("gate_proj", "up_proj"),
                    ("w1", "w3"),
                    ("gate", "up"),
                    ("in_proj", "out_proj"),
                    ("q_proj", "k_proj"),
                ]
            )
            for m in model.modules()
        )
        if has_llm_ops:
            return "MODERN_LLM_MLP", trainable_params, frozen_params

        has_layernorm = any(isinstance(m, (nn.LayerNorm, nn.GroupNorm)) for m in model.modules())
        if has_layernorm and not has_conv:
            return "TRANSFORMER_VIT_LAYERNORM", trainable_params, frozen_params

        if has_conv:
            return "VISION_CONV_UNET", trainable_params, frozen_params

        # Detect tabular / narrow MLP: shallow depth, small linear dimensions (all features <= 512)
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linear_layers and all(m.in_features <= 512 and m.out_features <= 512 for m in linear_layers):
            return "TABULAR_CLASSICAL_ML", trainable_params, frozen_params

        return "GENERAL_DENSE_WORKLOAD", trainable_params, frozen_params

    def optimize_model(
        self,
        model: nn.Module,
        batch_tokens: int = 512,
        prefer_1bit_relu2_for_relu: bool = False,
        prefer_unified_mlp: Optional[bool] = None,
        compile_mlir: bool = False,
        mlir_target: str = "auto",
    ) -> Tuple[nn.Module, SmartWorkloadReport]:
        if prefer_unified_mlp is not None:
            self.prefer_unified_mlp = prefer_unified_mlp

        profile_name, trainable_total, frozen_total = self.detect_workload_profile(model)
        report = SmartWorkloadReport(
            workload_profile=profile_name,
            silicon_profile=self.silicon,
            total_params=trainable_total + frozen_total,
            trainable_params=trainable_total,
            frozen_params=frozen_total,
        )

        effective_tiny_threshold = 64 if profile_name == "TABULAR_CLASSICAL_ML" else self.tiny_param_threshold

        # Check if root model itself is attention, lm_head, unified mlp, or split/chunked GLU MLP
        attn_detected, attn_lens = self._detect_and_fuse_attention(model, "root_attn", batch_tokens, report)
        if attn_detected and attn_lens is not None:
            return attn_lens, report

        if isinstance(model, nn.Linear) and (getattr(model, "out_features", 0) >= self.large_vocab_threshold):
            lm_detected, lm_lens = self._detect_and_fuse_lm_head(model, "root_lm_head", batch_tokens, report)
            if lm_detected and lm_lens is not None:
                return lm_lens, report

        unified_detected, unified_lens = self._detect_and_fuse_unified_mlp(model, "root_unified_mlp", batch_tokens, report)
        if unified_detected and unified_lens is not None:
            return unified_lens, report

        mlp_detected, mlp_lens = self._detect_and_fuse_llm_mlp(model, "root_mlp", batch_tokens, report)
        if mlp_detected and mlp_lens is not None:
            return mlp_lens, report

        chunked_detected, chunked_lens = self._detect_and_fuse_chunked_mlp(model, "root_chunked_mlp", batch_tokens, report)
        if chunked_detected and chunked_lens is not None:
            return chunked_lens, report

        self._rewrite_children(
            model,
            prefix="",
            batch_tokens=batch_tokens,
            prefer_1bit_relu2_for_relu=prefer_1bit_relu2_for_relu,
            effective_tiny_threshold=effective_tiny_threshold,
            report=report,
        )
        if compile_mlir:
            try:
                from .mlir_dialects import compile_mlir_executable
                exec_mod = compile_mlir_executable(model, target=mlir_target)
                report.mlir_pipeline = exec_mod.mlir_artifacts.to_dict()
                model._ojasx_mlir_artifacts = exec_mod.mlir_artifacts
            except Exception:
                pass
        return model, report

    def _rewrite_children(
        self,
        parent: nn.Module,
        prefix: str,
        batch_tokens: int,
        prefer_1bit_relu2_for_relu: bool,
        effective_tiny_threshold: int,
        report: SmartWorkloadReport,
    ) -> None:
        if isinstance(parent, nn.Sequential):
            self._fuse_sequential_in_place(
                parent, prefix, batch_tokens, prefer_1bit_relu2_for_relu, effective_tiny_threshold, report
            )
        else:
            self._fuse_adjacent_linear_act(
                parent, prefix, batch_tokens, prefer_1bit_relu2_for_relu, effective_tiny_threshold, report, is_sequential=False
            )

        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(
                child,
                (
                    HolonomicBitPackedLinear,
                    HolonomicLinear,
                    HolonomicNativeRMSNorm,
                    HolonomicLayerNorm,
                    HolonomicGroupNorm,
                    FrozenLinearZeroTape,
                    FrozenConvZeroTape,
                    PackedReLU,
                    HolonomicFusedGLUMLP,
                    HolonomicFusedSwiGLUMLP,
                    HolonomicFusedGeGLUMLP,
                    HolonomicFusedChunkedGLUMLP,
                    HolonomicGLU,
                    HolonomicActivationLens,
                    HolonomicDaggerAttention,
                    HolonomicFusedLinearCrossEntropy,
                    HolonomicFusedLinearCrossEntropyHead,
                    HolonomicUnifiedTransformerMLP,
                ),
            ):
                continue

            # 0a. Modern Transformer Attention
            attn_detected, attn_lens = self._detect_and_fuse_attention(child, full_name, batch_tokens, report)
            if attn_detected and attn_lens is not None:
                setattr(parent, name, attn_lens)
                continue

            # 0b. Model Output LM Head
            if isinstance(child, nn.Linear) and (
                name in ("lm_head", "embed_out", "output_layer", "head")
                or child.out_features >= self.large_vocab_threshold
            ):
                lm_detected, lm_lens = self._detect_and_fuse_lm_head(child, full_name, batch_tokens, report)
                if lm_detected and lm_lens is not None:
                    setattr(parent, name, lm_lens)
                    continue

            # 0c. Unified Transformer Block (Norm + SwiGLU + Linear)
            unified_detected, unified_lens = self._detect_and_fuse_unified_mlp(child, full_name, batch_tokens, report)
            if unified_detected and unified_lens is not None:
                setattr(parent, name, unified_lens)
                continue

            # 1. Universal GLU Engine: Layout A (Split-Projection: gate_proj + up_proj + down_proj)
            mlp_detected, mlp_lens = self._detect_and_fuse_llm_mlp(child, full_name, batch_tokens, report)
            if mlp_detected and mlp_lens is not None:
                setattr(parent, name, mlp_lens)
                continue

            # 2. Universal GLU Engine: Layout B (Chunked-Projection: fc_in 2*H + fc_out)
            chunked_detected, chunked_lens = self._detect_and_fuse_chunked_mlp(child, full_name, batch_tokens, report)
            if chunked_detected and chunked_lens is not None:
                setattr(parent, name, chunked_lens)
                continue

            # 3. Standalone nn.GLU replacement
            if isinstance(child, nn.GLU):
                rep = HolonomicGLU(dim=child.dim, act="sigmoid")
                setattr(parent, name, rep)
                report.decisions.append(
                    SmartLayerDecision(
                        module_name=full_name,
                        original_type="nn.GLU",
                        decision="OJASX_HOLONOMIC_GLU",
                        replacement_type="HolonomicGLU(sigmoid)",
                        trainable_params=0,
                        frozen_params=0,
                        tile_schedule="(glu_fused)",
                        reason="Zero-tape Holonomic GLU activation kernel",
                        estimated_saved_bytes_per_token=4.0,
                    )
                )
                continue

            # 4. Linear Layers
            if isinstance(child, nn.Linear):
                replacement, decision = self._decide_linear(
                    child, full_name, batch_tokens, effective_tiny_threshold
                )
                if replacement is not child:
                    setattr(parent, name, replacement)
                report.decisions.append(decision)

            # 5. LayerNorm
            elif isinstance(child, nn.LayerNorm):
                rep = HolonomicLayerNorm(
                    child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    bias=child.bias is not None,
                    device=child.weight.device if child.weight is not None else None,
                    dtype=child.weight.dtype if child.weight is not None else None,
                )
                if child.elementwise_affine:
                    rep.weight = child.weight
                    if child.bias is not None:
                        rep.bias = child.bias
                setattr(parent, name, rep)
                dim = child.weight.numel() if child.weight is not None else 0
                report.decisions.append(
                    SmartLayerDecision(
                        module_name=full_name,
                        original_type="nn.LayerNorm",
                        decision="OJASX_HOLONOMIC_LAYERNORM",
                        replacement_type="HolonomicLayerNorm",
                        trainable_params=dim if child.elementwise_affine and child.weight.requires_grad else 0,
                        frozen_params=0 if (not child.elementwise_affine or child.weight.requires_grad) else dim,
                        tile_schedule="(1,compact_stats,1)",
                        reason="Compact statistics (mean, rstd) eliminating normalized intermediate tensors",
                        estimated_saved_bytes_per_token=float(dim * 4),
                    )
                )

            # 6. GroupNorm
            elif isinstance(child, nn.GroupNorm):
                rep = HolonomicGroupNorm(
                    child.num_groups,
                    child.num_channels,
                    eps=child.eps,
                    affine=child.affine,
                    device=child.weight.device if child.weight is not None else None,
                    dtype=child.weight.dtype if child.weight is not None else None,
                )
                if child.affine:
                    rep.weight = child.weight
                    if child.bias is not None:
                        rep.bias = child.bias
                setattr(parent, name, rep)
                dim = child.weight.numel() if child.weight is not None else 0
                report.decisions.append(
                    SmartLayerDecision(
                        module_name=full_name,
                        original_type="nn.GroupNorm",
                        decision="OJASX_HOLONOMIC_GROUPNORM",
                        replacement_type="HolonomicGroupNorm",
                        trainable_params=dim if child.affine and child.weight.requires_grad else 0,
                        frozen_params=0 if (not child.affine or child.weight.requires_grad) else dim,
                        tile_schedule="(1,group_stats,1)",
                        reason="Group statistics (mean, rstd) eliminating normalized intermediate tensor",
                        estimated_saved_bytes_per_token=float(dim * 4),
                    )
                )

            # 7. RMSNorm
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

            # 8. Convolutions (Frozen Conv Zero Tape)
            elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                is_frozen = not child.weight.requires_grad and (child.bias is None or not child.bias.requires_grad)
                if is_frozen:
                    rep = FrozenConvZeroTape(child)
                    setattr(parent, name, rep)
                    params = child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
                    report.decisions.append(
                        SmartLayerDecision(
                            module_name=full_name,
                            original_type=child.__class__.__name__,
                            decision="SKIP_BACKWARD_FROZEN_CONV",
                            replacement_type="FrozenConvZeroTape",
                            trainable_params=0,
                            frozen_params=params,
                            tile_schedule="(conv_tiled)",
                            reason="Frozen conv backbone: zero input activation tape retained during forward",
                            estimated_saved_bytes_per_token=float(child.in_channels * 4),
                        )
                    )
                else:
                    self._rewrite_children(
                        child, full_name, batch_tokens, prefer_1bit_relu2_for_relu, effective_tiny_threshold, report
                    )

            # 9. Standalone Activations
            elif isinstance(child, nn.ReLU):
                if not child.inplace:
                    rep = PackedReLU()
                    setattr(parent, name, rep)
                    report.decisions.append(
                        SmartLayerDecision(
                            module_name=full_name,
                            original_type="nn.ReLU",
                            decision="OJASX_PACKED_RELU",
                            replacement_type="PackedReLU",
                            trainable_params=0,
                            frozen_params=0,
                            tile_schedule="(1-bit ballot)",
                            reason="1-bit boolean sign mask retaining 1/32th of FP32 storage",
                            estimated_saved_bytes_per_token=4.0 * (31.0 / 32.0),
                        )
                    )
            elif isinstance(child, (nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Mish, nn.ELU)):
                act_name = child.__class__.__name__.lower()
                neg_slope = getattr(child, "negative_slope", 0.01)
                approx = getattr(child, "approximate", "none")
                rep = HolonomicActivationLens(act_name, negative_slope=neg_slope, approximate=approx)
                setattr(parent, name, rep)
                report.decisions.append(
                    SmartLayerDecision(
                        module_name=full_name,
                        original_type=child.__class__.__name__,
                        decision="OJASX_HOLONOMIC_ACTIVATION",
                        replacement_type="HolonomicActivationLens",
                        trainable_params=0,
                        frozen_params=0,
                        tile_schedule="(autograd lens)",
                        reason=f"Standalone {child.__class__.__name__} routed to holonomic activation lens",
                        estimated_saved_bytes_per_token=4.0,
                    )
                )

            else:
                self._rewrite_children(
                    child, full_name, batch_tokens, prefer_1bit_relu2_for_relu, effective_tiny_threshold, report
                )

    def _detect_and_fuse_llm_mlp(
        self,
        module: nn.Module,
        full_name: str,
        batch_tokens: int,
        report: SmartWorkloadReport,
    ) -> Tuple[bool, Optional[nn.Module]]:
        """Detects Split-Projection GLU MLP submodules and fuses them into Holonomic GLU lenses."""
        gate_proj = getattr(module, "gate_proj", None) or getattr(module, "w1", None) or getattr(module, "gate", None)
        up_proj = getattr(module, "up_proj", None) or getattr(module, "w3", None) or getattr(module, "up", None)
        down_proj = getattr(module, "down_proj", None) or getattr(module, "w2", None) or getattr(module, "down", None)

        if not (isinstance(gate_proj, nn.Linear) and isinstance(up_proj, nn.Linear) and isinstance(down_proj, nn.Linear)):
            return False, None

        # Ensure module does not contain unrelated Linear/Conv/Attention layers that would be dropped
        allowed_linears = {id(gate_proj), id(up_proj), id(down_proj)}
        for _, sub in module.named_children():
            if isinstance(sub, nn.Linear) and id(sub) not in allowed_linears:
                return False, None
            if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, HolonomicDaggerAttention)):
                return False, None

        act_name, neg_slope = _extract_glu_activation(module)
        total_p = sum(p.numel() for p in module.parameters())
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        if act_name == "swiglu":
            lens = HolonomicFusedSwiGLUMLP(gate_proj, up_proj, down_proj)
            rep_type = "HolonomicFusedSwiGLUMLP"
        elif act_name == "geglu":
            lens = HolonomicFusedGeGLUMLP(gate_proj, up_proj, down_proj, approximate="tanh")
            rep_type = "HolonomicFusedGeGLUMLP"
        elif act_name in ("geglu_exact", "gelu_exact"):
            lens = HolonomicFusedGeGLUMLP(gate_proj, up_proj, down_proj, approximate="none")
            rep_type = "HolonomicFusedGeGLUMLP(exact)"
        elif act_name == "reglu":
            lens = HolonomicFusedReGLUMLP(gate_proj, up_proj, down_proj)
            rep_type = "HolonomicFusedReGLUMLP"
        elif act_name == "relu2glu":
            lens = HolonomicFusedReLU2GLUMLP(gate_proj, up_proj, down_proj)
            rep_type = "HolonomicFusedReLU2GLUMLP"
        elif act_name == "sigmoidglu":
            lens = HolonomicFusedSigmoidGLUMLP(gate_proj, up_proj, down_proj)
            rep_type = "HolonomicFusedSigmoidGLUMLP"
        elif act_name == "bilinearglu":
            lens = HolonomicFusedBilinearGLUMLP(gate_proj, up_proj, down_proj)
            rep_type = "HolonomicFusedBilinearGLUMLP"
        else:
            lens = HolonomicFusedGLUMLP(gate_proj, up_proj, down_proj, act=act_name, negative_slope=neg_slope)
            rep_type = f"HolonomicFusedGLUMLP({act_name})"

        decision_label = f"OJASX_HOLONOMIC_{act_name.upper()}_MLP"

        report.decisions.append(
            SmartLayerDecision(
                module_name=full_name,
                original_type=module.__class__.__name__,
                decision=decision_label,
                replacement_type=rep_type,
                trainable_params=trainable_p,
                frozen_params=total_p - trainable_p,
                tile_schedule="(fused split glu)",
                reason=f"Universal GLU Engine ({act_name}) rewritten to fused zero-tape lens",
                estimated_saved_bytes_per_token=float(gate_proj.out_features * 4 * 2),
            )
        )
        return True, lens

    def _detect_and_fuse_chunked_mlp(
        self,
        module: nn.Module,
        full_name: str,
        batch_tokens: int,
        report: SmartWorkloadReport,
    ) -> Tuple[bool, Optional[nn.Module]]:
        """Detects Chunked-Projection GLU MLP submodules (e.g. fc_in -> 2*H chunk -> act(g)*u -> fc_out)."""
        in_proj = (
            getattr(module, "in_proj", None)
            or getattr(module, "gate_up_proj", None)
            or getattr(module, "fc_in", None)
            or getattr(module, "fc1", None)
            or getattr(module, "w12", None)
        )
        out_proj = (
            getattr(module, "out_proj", None)
            or getattr(module, "down_proj", None)
            or getattr(module, "fc_out", None)
            or getattr(module, "fc2", None)
            or getattr(module, "w3", None)
        )

        if not isinstance(in_proj, nn.Linear):
            return False, None

        # Verify module is genuinely a chunked GLU block and not an ordinary MLP
        has_explicit_chunk_attr = (
            getattr(module, "gate_up_proj", None) is not None
            or getattr(module, "w12", None) is not None
            or any(isinstance(m, (nn.GLU, HolonomicGLU)) for m in module.children())
            or "glu" in module.__class__.__name__.lower()
            or "chunk" in module.__class__.__name__.lower()
        )
        if not has_explicit_chunk_attr:
            try:
                import inspect
                fwd_src = inspect.getsource(module.__class__.forward)
                has_explicit_chunk_attr = (".chunk(" in fwd_src or "torch.chunk(" in fwd_src or "glu(" in fwd_src)
            except Exception:
                has_explicit_chunk_attr = False
        if not has_explicit_chunk_attr:
            return False, None

        # Ensure no unrelated Linear layers exist on module
        allowed_linears = {id(in_proj)} | ({id(out_proj)} if isinstance(out_proj, nn.Linear) else set())
        for _, sub in module.named_children():
            if isinstance(sub, nn.Linear) and id(sub) not in allowed_linears:
                return False, None

        # Chunked GLU requires out_features of in_proj to be 2 * hidden
        if out_proj is not None and isinstance(out_proj, nn.Linear):
            if in_proj.out_features != 2 * out_proj.in_features:
                return False, None
        else:
            if in_proj.out_features % 2 != 0:
                return False, None

        act_name, neg_slope = _extract_glu_activation(module)
        total_p = sum(p.numel() for p in module.parameters())
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        lens = HolonomicFusedChunkedGLUMLP(
            in_proj, out_proj=out_proj, act=act_name, negative_slope=neg_slope, dim=-1
        )
        decision_label = f"OJASX_HOLONOMIC_CHUNKED_{act_name.upper()}_MLP"

        report.decisions.append(
            SmartLayerDecision(
                module_name=full_name,
                original_type=module.__class__.__name__,
                decision=decision_label,
                replacement_type=f"HolonomicFusedChunkedGLUMLP({act_name})",
                trainable_params=trainable_p,
                frozen_params=total_p - trainable_p,
                tile_schedule="(fused chunked glu)",
                reason=f"Universal Chunked GLU ({act_name}) rewritten to fused zero-tape lens",
                estimated_saved_bytes_per_token=float(in_proj.out_features * 4),
            )
        )
        return True, lens

    def _detect_and_fuse_attention(
        self,
        module: nn.Module,
        full_name: str,
        batch_tokens: int,
        report: SmartWorkloadReport,
    ) -> Tuple[bool, Optional[nn.Module]]:
        """Detects Transformer attention layers and rewrites them into Holonomic Dagger Attention."""
        if isinstance(module, HolonomicDaggerAttention):
            return False, None

        if isinstance(module, nn.MultiheadAttention):
            first_p = next(module.parameters(), None)
            dev = first_p.device if first_p is not None else None
            dt = first_p.dtype if first_p is not None else None
            total_p = sum(p.numel() for p in module.parameters())
            trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
            rep = HolonomicDaggerAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
                device=dev,
                dtype=dt,
            )
            if module.in_proj_weight is not None:
                d = module.embed_dim
                rep.q_proj.weight.data.copy_(module.in_proj_weight[:d])
                rep.k_proj.weight.data.copy_(module.in_proj_weight[d : 2 * d])
                rep.v_proj.weight.data.copy_(module.in_proj_weight[2 * d : 3 * d])
                if module.in_proj_bias is not None:
                    rep.q_proj.bias.data.copy_(module.in_proj_bias[:d])
                    rep.k_proj.bias.data.copy_(module.in_proj_bias[d : 2 * d])
                    rep.v_proj.bias.data.copy_(module.in_proj_bias[2 * d : 3 * d])
            if module.out_proj is not None and hasattr(module.out_proj, "weight"):
                rep.o_proj.weight.data.copy_(module.out_proj.weight)
                if module.out_proj.bias is not None and rep.o_proj.bias is not None:
                    rep.o_proj.bias.data.copy_(module.out_proj.bias)

            report.decisions.append(
                SmartLayerDecision(
                    module_name=full_name,
                    original_type="nn.MultiheadAttention",
                    decision="HOLONOMIC_DAGGER_ATTENTION",
                    replacement_type="HolonomicDaggerAttention",
                    trainable_params=trainable_p,
                    frozen_params=total_p - trainable_p,
                    tile_schedule="(online_softmax_tiles)",
                    reason="In-register adjoint online softmax tiling (0 bytes DRAM attention scores)",
                    estimated_saved_bytes_per_token=float(module.num_heads * batch_tokens * 4),
                )
            )
            return True, rep

        # Modern LLM Attention (LlamaAttention, Qwen2Attention, GemmaAttention, or modules with q_proj, k_proj, v_proj)
        q_proj = getattr(module, "q_proj", getattr(module, "q", None))
        k_proj = getattr(module, "k_proj", getattr(module, "k", None))
        v_proj = getattr(module, "v_proj", getattr(module, "v", None))
        o_proj = getattr(module, "o_proj", getattr(module, "out_proj", getattr(module, "c_proj", None)))

        is_attn_class = "attention" in module.__class__.__name__.lower()
        if (
            q_proj is not None
            and k_proj is not None
            and v_proj is not None
            and isinstance(q_proj, nn.Linear)
            and isinstance(k_proj, nn.Linear)
            and isinstance(v_proj, nn.Linear)
        ):
            total_p = sum(p.numel() for p in module.parameters())
            trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

            num_heads = getattr(
                module,
                "num_heads",
                getattr(module, "n_heads", getattr(module, "num_attention_heads", 8)),
            )
            num_kv_heads = getattr(
                module,
                "num_key_value_heads",
                getattr(module, "n_kv_heads", getattr(module, "num_kv_heads", num_heads)),
            )
            head_dim = getattr(module, "head_dim", None)
            if head_dim is None and num_heads:
                head_dim = q_proj.out_features // num_heads

            rep = HolonomicDaggerAttention(
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                o_proj=o_proj,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                is_causal=getattr(module, "is_causal", True),
            )
            report.decisions.append(
                SmartLayerDecision(
                    module_name=full_name,
                    original_type=module.__class__.__name__,
                    decision="HOLONOMIC_DAGGER_ATTENTION",
                    replacement_type="HolonomicDaggerAttention",
                    trainable_params=trainable_p,
                    frozen_params=total_p - trainable_p,
                    tile_schedule="(online_softmax_tiles)",
                    reason="In-register adjoint online softmax tiling (0 bytes DRAM attention scores)",
                    estimated_saved_bytes_per_token=float(num_heads * batch_tokens * 4),
                )
            )
            return True, rep

        return False, None

    def _detect_and_fuse_lm_head(
        self,
        module: nn.Linear,
        full_name: str,
        batch_tokens: int,
        report: SmartWorkloadReport,
    ) -> Tuple[bool, Optional[nn.Module]]:
        """Detects LM head output projection and rewrites to Holonomic Fused Linear Cross-Entropy."""
        total_p = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        rep = HolonomicFusedLinearCrossEntropyHead(
            in_features=module.in_features,
            vocab_size=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        rep.weight = module.weight
        if module.bias is not None:
            rep.bias = module.bias

        report.decisions.append(
            SmartLayerDecision(
                module_name=full_name,
                original_type=f"nn.Linear({module.in_features}->{module.out_features})",
                decision="FUSED_LINEAR_CROSS_ENTROPY",
                replacement_type="HolonomicFusedLinearCrossEntropyHead",
                trainable_params=trainable_p,
                frozen_params=total_p - trainable_p,
                tile_schedule=f"(streaming 128x4096, V={module.out_features})",
                reason="Dual-pass streaming token-vocab accumulation (0 bytes DRAM logits tensor)",
                estimated_saved_bytes_per_token=float(module.out_features * 4 * 2),
            )
        )
        return True, rep

    def _detect_and_fuse_unified_mlp(
        self,
        module: nn.Module,
        full_name: str,
        batch_tokens: int,
        report: SmartWorkloadReport,
    ) -> Tuple[bool, Optional[nn.Module]]:
        """Detects combined Transformer block (Norm + Gate/Up/Down Proj) and fuses into HolonomicUnifiedTransformerMLP."""
        if isinstance(module, HolonomicUnifiedTransformerMLP):
            return False, None

        gate_proj = (
            getattr(module, "gate_proj", None)
            or getattr(module, "w1", None)
            or getattr(module, "gate", None)
        )
        up_proj = (
            getattr(module, "up_proj", None)
            or getattr(module, "w3", None)
            or getattr(module, "up", None)
        )
        down_proj = (
            getattr(module, "down_proj", None)
            or getattr(module, "w2", None)
            or getattr(module, "down", None)
        )

        if not (
            isinstance(gate_proj, nn.Linear)
            and isinstance(up_proj, nn.Linear)
            and isinstance(down_proj, nn.Linear)
        ):
            return False, None

        norm = (
            getattr(module, "norm", None)
            or getattr(module, "rms_norm", None)
            or getattr(module, "input_norm", None)
            or getattr(module, "post_attention_layernorm", None)
            or getattr(module, "ffn_norm", None)
        )

        if norm is None and not getattr(self, "prefer_unified_mlp", False):
            return False, None

        allowed_linears = {id(gate_proj), id(up_proj), id(down_proj)}
        for _, sub in module.named_children():
            if isinstance(sub, nn.Linear) and id(sub) not in allowed_linears:
                return False, None
            if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, HolonomicDaggerAttention)):
                return False, None

        if not getattr(self, "prefer_unified_mlp", False):
            if isinstance(norm, nn.LayerNorm):
                return False, None
            if gate_proj.bias is not None or up_proj.bias is not None or down_proj.bias is not None:
                return False, None

        dim = gate_proj.in_features
        intermediate_dim = gate_proj.out_features
        eps = float(getattr(norm, "eps", 1e-6)) if norm is not None else 1e-6
        norm_w = norm.weight if (norm is not None and hasattr(norm, "weight")) else None

        rep = HolonomicUnifiedTransformerMLP(
            dim=dim,
            intermediate_dim=intermediate_dim,
            eps=eps,
            norm_weight=norm_w,
            gate_weight=gate_proj.weight,
            up_weight=up_proj.weight,
            down_weight=down_proj.weight,
            device=gate_proj.weight.device,
            dtype=gate_proj.weight.dtype,
        )

        total_p = sum(p.numel() for p in module.parameters())
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        report.decisions.append(
            SmartLayerDecision(
                module_name=full_name,
                original_type=module.__class__.__name__,
                decision="OJASX_UNIFIED_TRANSFORMER_MLP",
                replacement_type="HolonomicUnifiedTransformerMLP",
                trainable_params=trainable_p,
                frozen_params=total_p - trainable_p,
                tile_schedule=f"({dim},{intermediate_dim})",
                reason="Single compiled block sweep bypassing autograd dispatch overhead (1 node on tape)",
                estimated_saved_bytes_per_token=float(intermediate_dim * 4 * 4),
            )
        )
        return True, rep

    def _is_safe_to_fuse_linear_activation(
        self, parent: nn.Module, name_a: str, name_b: str
    ) -> bool:
        """Verifies that activation name_b in non-sequential parent is dedicated to name_a and not reused."""
        try:
            import inspect, ast, textwrap
            forward_fn = getattr(parent.__class__, "forward", None)
            if forward_fn is None:
                return False
            src = textwrap.dedent(inspect.getsource(forward_fn))
            tree = ast.parse(src)

            # Count occurrences of self.<name_a> and self.<name_b>
            a_refs = 0
            b_refs = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                    if node.attr == name_a:
                        a_refs += 1
                    elif node.attr == name_b:
                        b_refs += 1

            if a_refs == 1 and b_refs == 1:
                return True

            return False
        except Exception:
            return False

    def _fuse_adjacent_linear_act(
        self,
        parent: nn.Module,
        prefix: str,
        batch_tokens: int,
        prefer_1bit_relu2_for_relu: bool,
        effective_tiny_threshold: int,
        report: SmartWorkloadReport,
        is_sequential: bool = False,
    ) -> None:
        children = list(parent.named_children())
        i = 0
        while i < len(children) - 1:
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            full_name = f"{prefix}.{name_a}" if prefix else name_a

            if isinstance(mod_a, nn.Linear) and isinstance(mod_b, (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU)):
                # In non-sequential modules, verify that mod_b is safe to fuse and replace with Identity
                if not is_sequential and not self._is_safe_to_fuse_linear_activation(parent, name_a, name_b):
                    i += 1
                    continue

                params = mod_a.in_features * mod_a.out_features
                w_grad = mod_a.weight.requires_grad
                sched = self.planner.plan_gemm_tile(
                    batch_tokens, mod_a.out_features, mod_a.in_features,
                    device_type=mod_a.weight.device.type,
                )
                tile_str = f"({sched.tile_m},{sched.tile_n},{sched.tile_k})"

                if params >= effective_tiny_threshold and w_grad:
                    if isinstance(mod_b, nn.ReLU) and prefer_1bit_relu2_for_relu:
                        rep = HolonomicBitPackedLinear(
                            mod_a.in_features, mod_a.out_features,
                            bias=mod_a.bias is not None,
                            device=mod_a.weight.device, dtype=mod_a.weight.dtype,
                        )
                        rep.weight = mod_a.weight
                        if mod_a.bias is not None:
                            rep.bias = mod_a.bias
                        setattr(parent, name_a, rep)
                        setattr(parent, name_b, nn.Identity())
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

                    negative_slope = 0.01
                    if isinstance(mod_b, nn.GELU):
                        approx = getattr(mod_b, "approximate", "none")
                        act_name = "gelu_tanh" if approx == "tanh" else "gelu_exact"
                    elif isinstance(mod_b, nn.SiLU):
                        act_name = "silu"
                    elif isinstance(mod_b, nn.ReLU):
                        act_name = "relu"
                    elif isinstance(mod_b, nn.LeakyReLU):
                        act_name = "leaky_relu"
                        negative_slope = float(getattr(mod_b, "negative_slope", 0.01))
                    else:
                        act_name = "none"

                    rep = HolonomicLinear(
                        mod_a.in_features, mod_a.out_features,
                        bias=mod_a.bias is not None,
                        act=act_name,
                        negative_slope=negative_slope,
                        device=mod_a.weight.device, dtype=mod_a.weight.dtype,
                    )
                    rep.weight = mod_a.weight
                    if mod_a.bias is not None:
                        rep.bias = mod_a.bias
                    setattr(parent, name_a, rep)
                    setattr(parent, name_b, nn.Identity())
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

    def _fuse_sequential_in_place(
        self,
        seq: nn.Sequential,
        prefix: str,
        batch_tokens: int,
        prefer_1bit_relu2_for_relu: bool,
        effective_tiny_threshold: int,
        report: SmartWorkloadReport,
    ) -> None:
        self._fuse_adjacent_linear_act(
            seq, prefix, batch_tokens, prefer_1bit_relu2_for_relu, effective_tiny_threshold, report, is_sequential=True
        )

    def _decide_linear(
        self,
        linear: nn.Linear,
        full_name: str,
        batch_tokens: int,
        effective_tiny_threshold: int,
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

        # Rule 1: Frozen Backbone Layer
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

        # Rule 2: Tiny Tensor Fast-Bypass (evaluated against adaptive effective_tiny_threshold)
        if (linear.in_features * linear.out_features) < effective_tiny_threshold:
            return linear, SmartLayerDecision(
                module_name=full_name,
                original_type="nn.Linear (Tiny)",
                decision="NATIVE_PYTORCH_FAST_BYPASS",
                replacement_type="nn.Linear (Unchanged)",
                trainable_params=params,
                frozen_params=0,
                tile_schedule=tile_str,
                reason=f"Matrix size ({linear.in_features}x{linear.out_features}) < {effective_tiny_threshold}: keep native C++ fast-path",
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


def optimize_model(
    model: nn.Module,
    batch_tokens: int = 512,
    prefer_1bit_relu2_for_relu: bool = False,
    prefer_unified_mlp: bool = False,
    tiny_param_threshold: int = 1024,
    large_vocab_threshold: int = 8192,
    compile_mlir: bool = False,
    mlir_target: str = "auto",
    **router_kwargs,
) -> Tuple[nn.Module, SmartWorkloadReport]:
    """Top-level 3-line quickstart model optimizer.

    Usage:
        import ojasx2
        model, report = ojasx2.optimize_model(model)
    """
    router = SmartWorkloadRouter(
        tiny_param_threshold=tiny_param_threshold,
        large_vocab_threshold=large_vocab_threshold,
        prefer_unified_mlp=prefer_unified_mlp,
        **router_kwargs,
    )
    return router.optimize_model(
        model,
        batch_tokens=batch_tokens,
        prefer_1bit_relu2_for_relu=prefer_1bit_relu2_for_relu,
        prefer_unified_mlp=prefer_unified_mlp,
        compile_mlir=compile_mlir,
        mlir_target=mlir_target,
    )

