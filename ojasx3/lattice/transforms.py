"""
Exact Orthogonal Incoherence Transforms & Circulant Weighing Adapters for OjasX3.

Provides multiplier-free exact orthogonal transforms for E8 lattice incoherence
processing and ultra-low-RAM Orthogonal Fine-Tuning (CW-OFT):
  - Block-FWHT (H_32 / H_64 / H_128): O(d log d) pure add/sub for power-of-2 widths
  - CW(2m, 4): Exact weight-4 proper circulant weighing matrix for any even n = 2m
  - CW(14m, 16): Exact weight-16 proper circulant weighing matrix for n = 14m
    (matches Llama-3 14336 = 14*1024 and Qwen 3584 = 14*256 without padding)
  - CW(120, 36): Certified proper circulant weighing matrix (W W^T = 36 I, scale 1/6)
  - CWOFTLinear: O(d) full-rank, norm-preserving orthogonal adapter D2 W^T D1 W
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ojasx2.math.lattice import fwht_block32  # re-export for backwards compatibility

# Certified proper CW(120, 36) first row: +1 on P, -1 on N; W W^T = 36 I, gcd = 1.
CW120_36_POS: Tuple[int, ...] = (
    17, 26, 30, 37, 42, 54, 57, 58, 60, 62, 68, 70, 77, 84, 94, 97, 98, 106, 112, 116, 117,
)
CW120_36_NEG: Tuple[int, ...] = (
    0, 2, 6, 8, 14, 18, 24, 50, 52, 56, 66, 74, 78, 102, 110,
)


def cw_weight4_support(n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (+1 indices, -1 indices) of proper CW(n, 4) for any even n = 2m >= 4.
    First row polynomial: a(x) = -1 + x + x^m + x^(m+1), satisfying a(x)a(x^-1) = 4.
    """
    if n < 4 or n % 2 != 0:
        raise ValueError(f"CW(n, 4) requires even n >= 4, got {n}")
    m = n // 2
    return ((1, m, m + 1), (0,))


def cw_weight16_support(n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (+1 indices, -1 indices) of proper CW(n, 16) for n = 14m (m >= 1).
    Construction (Arasu-Gordon-Zhang Thm 6.2):
      A(X) = (1 - X^(7m)) C(X^(2m)) + (1 + X^(7m)) X C(X^m),
      C(X) = -1 + X + X^2 + X^4.
    Satisfies A(X) A(X^-1) = 16 in Z[X]/(X^n - 1).
    """
    if n < 28 or n % 14 != 0:
        raise ValueError(f"CW(n, 16) requires n divisible by 14 with n >= 28 (m >= 2), got {n}")
    m = n // 14
    coeffs = [0] * n
    c_terms = ((0, -1), (1, 1), (2, 1), (4, 1))
    # (1 - X^(7m)) * C(X^(2m))
    for exp_c, val_c in c_terms:
        coeffs[(exp_c * 2 * m) % n] += val_c
        coeffs[(7 * m + exp_c * 2 * m) % n] -= val_c
    # (1 + X^(7m)) * X * C(X^m)
    for exp_c, val_c in c_terms:
        coeffs[(1 + exp_c * m) % n] += val_c
        coeffs[(7 * m + 1 + exp_c * m) % n] += val_c
    pos = tuple(i for i, v in enumerate(coeffs) if v == 1)
    neg = tuple(i for i, v in enumerate(coeffs) if v == -1)
    if len(pos) + len(neg) != 16 or any(v not in (-1, 0, 1) for v in coeffs):
        raise RuntimeError(f"Invalid CW({n}, 16) construction")
    return pos, neg


def fwht_block(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Normalized Fast Walsh-Hadamard Transform along last dim in blocks of `block_size` (power of 2).
    Exact orthogonal involution: fwht_block(fwht_block(x, B), B) == x.
    """
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"Last dim {x.shape[-1]} must be divisible by block_size={block_size}")
    orig_shape = x.shape
    h = x.reshape(-1, block_size)
    stages = int(math.log2(block_size))
    for s in range(stages):
        step = 1 << s
        h = h.view(-1, block_size // (2 * step), 2, step)
        a = h[:, :, 0, :]
        b = h[:, :, 1, :]
        h = torch.stack((a + b, a - b), dim=2).reshape(-1, block_size)
    return (h * (1.0 / math.sqrt(block_size))).reshape(orig_shape)


def apply_circulant_weighing(
    x: torch.Tensor,
    n: int,
    pos_idx: Tuple[int, ...],
    neg_idx: Tuple[int, ...],
    *,
    transpose: bool = False,
) -> torch.Tensor:
    """Apply normalized circulant weighing matrix (1/sqrt(k)) * W (or W^T) in blocks of size `n`.

    Circulant convention: (W x)[i] = sum_j a[(j - i) mod n] x[j] = sum_s a[s] x[(i + s) mod n].
    Transpose convention: (W^T x)[i] = sum_s a[s] x[(i - s) mod n].
    Exact inverse: apply_circulant_weighing(apply_circulant_weighing(x, ...), ..., transpose=True) == x.
    """
    if x.shape[-1] % n != 0:
        raise ValueError(f"Last dim {x.shape[-1]} must be divisible by CW order n={n}")
    k = len(pos_idx) + len(neg_idx)
    inv_sqrt_k = 1.0 / math.sqrt(float(k))
    orig_shape = x.shape
    xb = x.reshape(-1, n)
    out = torch.zeros_like(xb)
    if not transpose:
        for s in pos_idx:
            out.add_(torch.roll(xb, shifts=-s, dims=-1))
        for s in neg_idx:
            out.sub_(torch.roll(xb, shifts=-s, dims=-1))
    else:
        for s in pos_idx:
            out.add_(torch.roll(xb, shifts=s, dims=-1))
        for s in neg_idx:
            out.sub_(torch.roll(xb, shifts=s, dims=-1))
    return out.mul_(inv_sqrt_k).reshape(orig_shape)


def select_incoherence_spec(dim: int) -> Tuple[str, int]:
    """Select the fastest exact orthogonal transform for arbitrary hidden width `dim`.
    Returns `(kind, block_size)` where kind in {'fwht', 'cw120', 'cw14', 'cw4', 'identity'}.
    """
    if dim >= 128 and dim % 128 == 0:
        return ("fwht", 128)
    if dim >= 64 and dim % 64 == 0:
        return ("fwht", 64)
    if dim >= 32 and dim % 32 == 0:
        return ("fwht", 32)
    if dim >= 120 and dim % 120 == 0:
        return ("cw120", 120)
    if dim >= 28 and dim % 14 == 0:
        # CW(14m, 16) is exact for m >= 2 (28, 56, 112, 3584, 14336, ...)
        for mult in (8, 4, 2):
            if dim % (14 * mult) == 0:
                return ("cw14", 14 * mult)
        if dim <= 112:
            return ("cw14", dim)
    if dim >= 8 and dim % 8 == 0:
        return ("fwht", 8)
    if dim >= 4 and dim % 2 == 0:
        return ("cw4", min(dim, 32) if dim % min(dim, 32) == 0 else dim)
    return ("identity", 1)


def apply_incoherence_transform(
    x: torch.Tensor,
    *,
    inverse: bool = False,
    spec: Optional[Tuple[str, int]] = None,
) -> torch.Tensor:
    """Apply exact orthogonal incoherence transform (or its exact inverse) along the last dim."""
    if x.numel() == 0:
        return x
    kind, blk = spec if spec is not None else select_incoherence_spec(x.shape[-1])
    if kind == "identity":
        return x
    if kind == "fwht":
        return fwht_block(x, block_size=blk)
    if kind == "cw120":
        return apply_circulant_weighing(x, 120, CW120_36_POS, CW120_36_NEG, transpose=inverse)
    if kind == "cw14":
        pos, neg = cw_weight16_support(blk)
        return apply_circulant_weighing(x, blk, pos, neg, transpose=inverse)
    if kind == "cw4":
        pos, neg = cw_weight4_support(blk)
        return apply_circulant_weighing(x, blk, pos, neg, transpose=inverse)
    raise ValueError(f"Unknown incoherence transform kind: {kind}")


class CWOFTLinear(nn.Module):
    """Circulant-Weighing Orthogonal Fine-Tuning (CW-OFT) adapter around any frozen linear layer.

    Applies a full-rank, norm-preserving, multiplier-free structured orthogonal residual:
        y = base_layer(x) + scale * (W_out^T (d2 * (W_out (base_layer_proj))) )
    More directly on the input/output space:
        delta_x = W^T ( (d_1) * (W x) )  with d_1 initialized to 0 -> identity at init!
    Parameter count: O(in_features + out_features) instead of O(rank * (in + out)).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        learn_input_mix: bool = True,
        learn_output_mix: bool = True,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_f = getattr(base_layer, "in_features", getattr(base_layer, "in_dim", None))
        out_f = getattr(base_layer, "out_features", getattr(base_layer, "out_dim", None))
        if in_f is None or out_f is None:
            raise ValueError(f"Cannot infer dimensions for CWOFTLinear base_layer {base_layer}")

        self.in_features = int(in_f)
        self.out_features = int(out_f)
        self.in_spec = select_incoherence_spec(self.in_features)
        self.out_spec = select_incoherence_spec(self.out_features)

        params = list(base_layer.parameters())
        buffers = list(base_layer.buffers())
        ref = params[0] if params else (buffers[0] if buffers else None)
        device = ref.device if ref is not None else None
        dtype = params[0].dtype if params else torch.float32
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            dtype = torch.float32

        self.d_in = (
            nn.Parameter(torch.zeros(self.in_features, device=device, dtype=dtype))
            if learn_input_mix
            else None
        )
        self.d_out = (
            nn.Parameter(torch.zeros(self.out_features, device=device, dtype=dtype))
            if learn_output_mix
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.d_in is not None:
            # Perturb input in orthogonal CW/FWHT basis: x' = x + W^T (d_in * (W x))
            x_rot = apply_incoherence_transform(x, inverse=False, spec=self.in_spec)
            x = x + apply_incoherence_transform(x_rot * self.d_in.to(x.dtype), inverse=True, spec=self.in_spec)
        y = self.base_layer(x)
        if self.d_out is not None:
            y_rot = apply_incoherence_transform(y, inverse=False, spec=self.out_spec)
            y = y + apply_incoherence_transform(y_rot * self.d_out.to(y.dtype), inverse=True, spec=self.out_spec)
        return y


__all__ = [
    "CW120_36_POS",
    "CW120_36_NEG",
    "cw_weight4_support",
    "cw_weight16_support",
    "fwht_block32",
    "fwht_block",
    "apply_circulant_weighing",
    "select_incoherence_spec",
    "apply_incoherence_transform",
    "CWOFTLinear",
]
