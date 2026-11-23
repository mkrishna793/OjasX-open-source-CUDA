"""
Phase 5: Memory-Minimal `DualRailLoRALinear` Layer for Up-to-9B Training on 24 GB M5.

Replaces `nn.Linear` in any HuggingFace `AutoModelForCausalLM` (Qwen3-8B,
Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B, Mistral-7B) with:
  - Frozen base weights stored strictly as **2.125-bit Dual-Rail `(P, N)` uint8 bitplanes**
    + FP16 per-group scales (7.53x smaller than BF16; zero FP16/BF16 base copy kept!).
  - Optional runtime Metal 4 shader dispatch (`torch.mps.compile_shader`, zero Xcode
    needed) via `use_metal_shader=True`, or chunked MPS/CUDA/CPU unpack fallback.
  - Trainable BF16 LoRA adapters (`lora_A`, `lora_B`) with full PyTorch autograd support.
"""

from __future__ import annotations

import gc
import math
from typing import Dict, Iterable, Optional, Tuple

from ojasx_logic.backend.metal_mps_runtime import mps_dual_rail_linear_forward


DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _pack_weight_to_uint8_dual_rail(
    weight_f32: "torch.Tensor",
    group_size: int = 128,
    threshold_ratio: float = 0.7,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int, float]:
    """Pack a 2D weight tensor [O, I] into uint8 Dual-Rail (P, N) bitplanes (100% MPS/CUDA/CPU safe)."""
    import torch

    O, I = weight_f32.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad > 0:
        w = torch.nn.functional.pad(weight_f32, (0, pad))
    else:
        w = weight_f32
    I_pad = w.shape[1]
    n_groups = I_pad // group_size
    groups = w.view(O, n_groups, group_size)

    abs_g = groups.abs()
    thresh = threshold_ratio * abs_g.mean(dim=-1, keepdim=True)
    q_pos = groups > thresh
    q_neg = groups < -thresh
    nz_mask = q_pos | q_neg

    nz_sum = (abs_g * nz_mask).sum(dim=-1)
    nz_cnt = nz_mask.sum(dim=-1).clamp(min=1)
    scales = (nz_sum / nz_cnt).to(torch.float16)
    sparsity = float(1.0 - nz_mask.float().mean().item())

    n_bytes = I_pad // 8
    shifts = torch.arange(8, dtype=torch.uint8, device=w.device)
    p_bytes = (q_pos.view(O, n_bytes, 8).to(torch.uint8) << shifts).sum(dim=-1, dtype=torch.uint8)
    n_bytes_t = (q_neg.view(O, n_bytes, 8).to(torch.uint8) << shifts).sum(dim=-1, dtype=torch.uint8)
    return p_bytes.contiguous(), n_bytes_t.contiguous(), scales.contiguous(), I_pad, sparsity


def _make_dual_rail_lora_linear_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DualRailMatmulFn(torch.autograd.Function):
        """Custom autograd keeping base weights frozen & packed (never saving unpacked W for backward)."""

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            pos_bytes: torch.Tensor,
            neg_bytes: torch.Tensor,
            scales: torch.Tensor,
            bias: Optional[torch.Tensor],
            in_features: int,
            group_size: int,
            chunk_rows: int,
            use_metal_shader: bool,
        ) -> torch.Tensor:
            ctx.save_for_backward(pos_bytes, neg_bytes, scales)
            ctx.in_features = in_features
            ctx.group_size = group_size
            ctx.chunk_rows = chunk_rows

            O = pos_bytes.shape[0]
            orig_shape = x.shape[:-1]
            x_2d = x.reshape(-1, in_features)

            if use_metal_shader and x_2d.device.type == "mps":
                metal_out = mps_dual_rail_linear_forward(
                    x_2d, pos_bytes, neg_bytes, scales, in_features, group_size
                )
                if metal_out is not None:
                    if bias is not None:
                        metal_out = metal_out + bias.to(metal_out.dtype)
                    return metal_out.view(*orig_shape, O)

            out_2d = torch.empty((x_2d.shape[0], O), dtype=x.dtype, device=x.device)
            shifts = torch.arange(8, dtype=torch.uint8, device=x.device)

            for r0 in range(0, O, chunk_rows):
                r1 = min(r0 + chunk_rows, O)
                pb = pos_bytes[r0:r1]
                nb = neg_bytes[r0:r1]
                sc = scales[r0:r1].to(x.dtype)
                p = ((pb.unsqueeze(-1) >> shifts) & 1).to(x.dtype)
                n = ((nb.unsqueeze(-1) >> shifts) & 1).to(x.dtype)
                w_chunk = (p - n).view(r1 - r0, -1, group_size) * sc.unsqueeze(-1)
                w_chunk = w_chunk.view(r1 - r0, -1)[:, :in_features]
                out_2d[:, r0:r1] = F.linear(x_2d, w_chunk, bias[r0:r1] if bias is not None else None)

            return out_2d.view(*orig_shape, O)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            pos_bytes, neg_bytes, scales = ctx.saved_tensors
            in_features = ctx.in_features
            group_size = ctx.group_size
            chunk_rows = ctx.chunk_rows

            O = pos_bytes.shape[0]
            orig_shape = grad_output.shape[:-1]
            g_2d = grad_output.reshape(-1, O)
            grad_x = torch.zeros((g_2d.shape[0], in_features), dtype=grad_output.dtype, device=grad_output.device)
            shifts = torch.arange(8, dtype=torch.uint8, device=grad_output.device)

            for r0 in range(0, O, chunk_rows):
                r1 = min(r0 + chunk_rows, O)
                pb = pos_bytes[r0:r1]
                nb = neg_bytes[r0:r1]
                sc = scales[r0:r1].to(grad_output.dtype)
                p = ((pb.unsqueeze(-1) >> shifts) & 1).to(grad_output.dtype)
                n = ((nb.unsqueeze(-1) >> shifts) & 1).to(grad_output.dtype)
                w_chunk = (p - n).view(r1 - r0, -1, group_size) * sc.unsqueeze(-1)
                w_chunk = w_chunk.view(r1 - r0, -1)[:, :in_features]
                grad_x.addmm_(g_2d[:, r0:r1], w_chunk)

            return grad_x.view(*orig_shape, in_features), None, None, None, None, None, None, None, None

    class DualRailLoRALinear(nn.Module):
        """Drop-in 2.125-bit Dual-Rail frozen base + trainable BF16 LoRA linear layer."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            lora_r: int = 16,
            lora_alpha: float = 32.0,
            group_size: int = 128,
            chunk_rows: int = 1024,
            has_bias: bool = False,
            compute_dtype: torch.dtype = torch.bfloat16,
            use_metal_shader: bool = False,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            self.lora_scaling = lora_alpha / float(max(lora_r, 1))
            self.group_size = group_size
            self.chunk_rows = chunk_rows
            self.use_metal_shader = use_metal_shader
            self.zero_sparsity = 0.0

            pad = (group_size - (in_features % group_size)) % group_size
            self.in_padded = in_features + pad

            self.register_buffer(
                "pos_bytes",
                torch.zeros((out_features, self.in_padded // 8), dtype=torch.uint8),
            )
            self.register_buffer(
                "neg_bytes",
                torch.zeros((out_features, self.in_padded // 8), dtype=torch.uint8),
            )
            self.register_buffer(
                "scales",
                torch.zeros((out_features, self.in_padded // group_size), dtype=torch.float16),
            )
            if has_bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=compute_dtype))
            else:
                self.bias = None

            if lora_r > 0:
                self.lora_A = nn.Parameter(torch.empty((lora_r, in_features), dtype=compute_dtype))
                self.lora_B = nn.Parameter(torch.zeros((out_features, lora_r), dtype=compute_dtype))
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            else:
                self.lora_A = None
                self.lora_B = None

        @classmethod
        def from_linear(
            cls,
            linear: nn.Linear,
            *,
            lora_r: int = 16,
            lora_alpha: float = 32.0,
            group_size: int = 128,
            chunk_rows: int = 1024,
            compute_dtype: torch.dtype = torch.bfloat16,
            use_metal_shader: bool = False,
        ) -> "DualRailLoRALinear":
            layer = cls(
                linear.in_features,
                linear.out_features,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                group_size=group_size,
                chunk_rows=chunk_rows,
                has_bias=linear.bias is not None,
                compute_dtype=compute_dtype,
                use_metal_shader=use_metal_shader,
            )
            with torch.no_grad():
                p_b, n_b, sc, _, sp = _pack_weight_to_uint8_dual_rail(
                    linear.weight.detach().float(),
                    group_size=group_size,
                )
                layer.pos_bytes.copy_(p_b)
                layer.neg_bytes.copy_(n_b)
                layer.scales.copy_(sc)
                layer.zero_sparsity = sp
                if linear.bias is not None and layer.bias is not None:
                    layer.bias.copy_(linear.bias.detach().to(compute_dtype))
            return layer

        @property
        def packed_weight_bytes(self) -> int:
            return int(
                self.pos_bytes.numel() * self.pos_bytes.element_size()
                + self.neg_bytes.numel() * self.neg_bytes.element_size()
                + self.scales.numel() * self.scales.element_size()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base = _DualRailMatmulFn.apply(
                x,
                self.pos_bytes,
                self.neg_bytes,
                self.scales,
                self.bias,
                self.in_features,
                self.group_size,
                self.chunk_rows,
                self.use_metal_shader,
            )
            if self.lora_A is not None and self.lora_B is not None:
                lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
                return base + lora_out
            return base

    return DualRailLoRALinear


def DualRailLoRALinear(*args, **kwargs):
    cls = _make_dual_rail_lora_linear_class()
    return cls(*args, **kwargs)


def quantize_module_to_dual_rail_lora(
    model: "torch.nn.Module",
    *,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    group_size: int = 128,
    chunk_rows: int = 1024,
    target_suffixes: Iterable[str] = DEFAULT_LORA_TARGETS,
    device: Optional[str] = None,
    compute_dtype_name: str = "bfloat16",
    use_metal_shader: bool = False,
) -> Dict[str, float]:
    """
    Stream-quantize every matching `nn.Linear` in-place to 2.125-bit `DualRailLoRALinear`.
    Frees each BF16 linear weight immediately so 8B–9B models fit cleanly on 24 GB Apple M5.
    """
    import torch
    import torch.nn as nn

    cls = _make_dual_rail_lora_linear_class()
    compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
    suffix_set = set(target_suffixes)

    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    bf16_bytes_total = 0
    packed_bytes_total = 0
    sparsities = []

    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if suffix_set and child_name not in suffix_set:
                continue
            bf16_bytes_total += child.weight.numel() * 2
            q_layer = cls.from_linear(
                child,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                group_size=group_size,
                chunk_rows=chunk_rows,
                compute_dtype=compute_dtype,
                use_metal_shader=use_metal_shader,
            )
            if device is not None:
                q_layer = q_layer.to(device)
            setattr(module, child_name, q_layer)
            del child
            replaced += 1
            packed_bytes_total += q_layer.packed_weight_bytes
            sparsities.append(q_layer.zero_sparsity)
            if replaced % 28 == 0:
                gc.collect()

    gc.collect()
    if device is not None:
        model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "layers_quantized": float(replaced),
        "bf16_linear_gb": bf16_bytes_total / (1024.0 ** 3),
        "packed_dual_rail_gb": packed_bytes_total / (1024.0 ** 3),
        "compression_ratio": bf16_bytes_total / max(packed_bytes_total, 1),
        "mean_zero_sparsity": sum(sparsities) / max(len(sparsities), 1),
        "trainable_lora_params": float(trainable_params),
    }


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "DualRailLoRALinear",
    "quantize_module_to_dual_rail_lora",
]
