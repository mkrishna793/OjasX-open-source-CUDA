"""
Phase 5: Compressed 2.125-Bit Dual-Rail Ternary Weight Packer (`{-1, 0, +1}`).

Fixes the 9-bitplane MX-NAF memory overhead (`18 bits/weight` > FP16) for frozen
LLM base weights by packing each weight into 1 Dual-Rail bitplane `(P, N)`:
  - `pos_bits`: 1 bit/weight (packed uint32, 32 weights/word)
  - `neg_bits`: 1 bit/weight (packed uint32, 32 weights/word)
  - `scales`:   FP16 per-group scale (group_size=128 -> 0.125 bits/weight)

Total footprint: **2.125 bits/weight** (7.53x smaller than FP16/BF16!).
A full 9B model (~8.2B linear weights) compresses from 16.4 GB BF16 -> **2.18 GB**,
leaving >16 GB headroom for LoRA adapters, AdamW states, and activations on a 24 GB M5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class TernaryDualRailTensor:
    pos_bits: np.ndarray  # uint32 [out_features, in_padded // 32]
    neg_bits: np.ndarray  # uint32 [out_features, in_padded // 32]
    scales: np.ndarray    # float16 [out_features, in_padded // group_size]
    out_features: int
    in_features: int
    in_padded: int
    group_size: int
    zero_sparsity: float

    @property
    def nbytes(self) -> int:
        return int(self.pos_bits.nbytes + self.neg_bits.nbytes + self.scales.nbytes)

    @property
    def bits_per_weight(self) -> float:
        return (self.nbytes * 8.0) / float(self.out_features * self.in_features)


def pack_ternary_dual_rail(
    W: np.ndarray,
    *,
    group_size: int = 128,
    threshold_ratio: float = 0.7,
) -> TernaryDualRailTensor:
    """Pack a 2D weight matrix [O, I] into 2.125-bit Dual-Rail (P, N) uint32 bitplanes."""
    arr = np.asarray(W, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {arr.shape}")
    if group_size % 32 != 0:
        raise ValueError(f"group_size must be a multiple of 32, got {group_size}")

    O, I = arr.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad > 0:
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
    I_pad = arr.shape[1]
    n_groups = I_pad // group_size
    groups = arr.reshape(O, n_groups, group_size)

    # AbsMean BitNet-style ternary quantization: {-1, 0, +1} * scale
    abs_g = np.abs(groups)
    mean_abs = np.mean(abs_g, axis=-1, keepdims=True)  # [O, n_groups, 1]
    thresh = threshold_ratio * mean_abs
    q_pos = groups > thresh
    q_neg = groups < -thresh
    nz_mask = q_pos | q_neg

    # Optimal least-squares scale over non-zero entries per group
    nz_sum = np.sum(abs_g * nz_mask, axis=-1)
    nz_cnt = np.sum(nz_mask, axis=-1)
    scales = np.where(nz_cnt > 0, nz_sum / np.maximum(nz_cnt, 1), 0.0).astype(np.float16)

    zero_sparsity = float(1.0 - np.mean(nz_mask[:, :, :]))

    # Pack booleans into uint32 words along K (LSB-first per 32-element word)
    n_words = I_pad // 32
    pos_bits_u8 = q_pos.reshape(O, n_words, 32).astype(np.uint32)
    neg_bits_u8 = q_neg.reshape(O, n_words, 32).astype(np.uint32)
    shifts = np.arange(32, dtype=np.uint32)
    pos_words = np.bitwise_or.reduce(pos_bits_u8 << shifts, axis=-1).astype(np.uint32)
    neg_words = np.bitwise_or.reduce(neg_bits_u8 << shifts, axis=-1).astype(np.uint32)

    return TernaryDualRailTensor(
        pos_bits=pos_words,
        neg_bits=neg_words,
        scales=scales,
        out_features=O,
        in_features=I,
        in_padded=I_pad,
        group_size=group_size,
        zero_sparsity=zero_sparsity,
    )


def unpack_ternary_dual_rail(t: TernaryDualRailTensor) -> np.ndarray:
    """Unpack 2.125-bit Dual-Rail (P, N) bitplanes back to float32 [out_features, in_features]."""
    O, n_words = t.pos_bits.shape
    shifts = np.arange(32, dtype=np.uint32)
    p = ((t.pos_bits[..., None] >> shifts) & np.uint32(1)).astype(np.float32)
    n = ((t.neg_bits[..., None] >> shifts) & np.uint32(1)).astype(np.float32)
    signed = (p - n).reshape(O, t.in_padded // t.group_size, t.group_size)
    out = (signed * t.scales.astype(np.float32)[..., None]).reshape(O, t.in_padded)
    return out[:, : t.in_features]


def estimate_llm_memory_budget(
    total_params_b: float,
    linear_fraction: float = 0.88,
    lora_rank: int = 16,
    group_size: int = 128,
) -> Dict[str, float]:
    """Compare BF16 full-weight VRAM vs. 2.125-bit Dual-Rail + BF16 LoRA training footprint (GB)."""
    total_params = total_params_b * 1e9
    linear_params = total_params * linear_fraction
    embed_norm_params = total_params * (1.0 - linear_fraction)

    bf16_weights_gb = (total_params * 2.0) / (1024.0 ** 3)
    bits_per_w = 2.0 + (16.0 / float(group_size))
    packed_linear_gb = (linear_params * (bits_per_w / 8.0)) / (1024.0 ** 3)
    embed_bf16_gb = (embed_norm_params * 2.0) / (1024.0 ** 3)

    # Approximate LoRA params (~0.5% of linear params at r=16) + AdamW FP32 states (12 bytes/LoRA param)
    lora_params = linear_params * (0.005 * (lora_rank / 16.0))
    lora_and_opt_gb = (lora_params * 14.0) / (1024.0 ** 3)
    total_train_weights_gb = packed_linear_gb + embed_bf16_gb + lora_and_opt_gb

    return {
        "total_params_b": total_params_b,
        "bf16_weights_gb": round(bf16_weights_gb, 2),
        "dual_rail_packed_linear_gb": round(packed_linear_gb, 2),
        "embed_and_norm_bf16_gb": round(embed_bf16_gb, 2),
        "lora_and_adamw_gb": round(lora_and_opt_gb, 2),
        "total_static_train_gb": round(total_train_weights_gb, 2),
        "compression_ratio_vs_bf16": round(bf16_weights_gb / max(packed_linear_gb + embed_bf16_gb, 1e-6), 2),
        "fits_24gb_m5": float(total_train_weights_gb < 14.0),
    }


__all__ = [
    "TernaryDualRailTensor",
    "pack_ternary_dual_rail",
    "unpack_ternary_dual_rail",
    "estimate_llm_memory_budget",
]
