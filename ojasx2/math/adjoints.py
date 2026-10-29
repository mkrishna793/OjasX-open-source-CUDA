"""
ojasx2/math/adjoints.py

Structured Low-Rank Adjoints, Normalization Residuals & Reference Block Algebra
===============================================================================
Implements canonical representation and computation for structured Jacobians:
    J = diag(d) + U V^T
where:
    d in R^n        (diagonal operator)
    U in R^{n x r}  (left low-rank factor)
    V in R^{n x r}  (right low-rank factor)
    r << n          (low rank, typically r in {0, 1, 2})

Also consolidates `NormalizationResidual`, `normalization_residual`,
`activation_derivative`, `rank7_reference`, and `winograd_reference` so that
`ojasx2.math` exposes the complete mathematical foundation without shadowing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _vector_domain(x: torch.Tensor) -> None:
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("Expected a floating-point tensor with at least one coordinate axis")


def _normalization_domain(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> None:
    _vector_domain(x)
    if x.shape[-1] == 0:
        raise ValueError("Normalization dimension must be nonempty")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected finite positive scalar epsilon")
    info = torch.finfo(x.dtype)
    if eps < info.tiny * info.eps or eps > info.max:
        raise ValueError("Epsilon must lie in the representable positive dtype range")
    if gamma.dtype != x.dtype or gamma.device != x.device:
        raise ValueError("Gain must share input dtype and device")
    if torch.broadcast_shapes(x.shape, gamma.shape) != x.shape:
        raise ValueError("Gain must broadcast without expanding the input shape")


# =============================================================================
# 1. Unified StructuredDerivative Representation
# =============================================================================

@dataclass(frozen=True)
class StructuredDerivative:
    """
    Structured linear operator J = diag(diagonal) + left @ right^T.

    Attributes:
        diagonal: Tensor of shape [..., n] containing the diagonal elements d.
        left:     Tensor of shape [..., n, r] containing the left factor U.
        right:    Tensor of shape [..., n, r] containing the right factor V.
    """
    diagonal: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        _vector_domain(self.diagonal)
        if self.left.ndim != self.diagonal.ndim + 1 or self.right.ndim != self.diagonal.ndim + 1:
            raise ValueError(
                f"Factors must have shape [..., n, r] matching diagonal [..., n]. "
                f"Got diagonal {self.diagonal.shape}, left {self.left.shape}, right {self.right.shape}"
            )
        if self.left.shape != self.right.shape:
            raise ValueError(f"Left and right factors must share shape, got {self.left.shape} vs {self.right.shape}")
        if self.left.shape[:-1] != self.diagonal.shape:
            raise ValueError(f"Factor batch+dim {self.left.shape[:-1]} must match diagonal {self.diagonal.shape}")
        if any(t.dtype != self.diagonal.dtype or t.device != self.diagonal.device for t in (self.left, self.right)):
            raise ValueError("All derivative tensors must share dtype and device")

    @property
    def dim(self) -> int:
        return self.diagonal.shape[-1]

    @property
    def rank(self) -> int:
        return self.left.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.diagonal.shape[:-1]

    def _validate_vector(self, x: torch.Tensor) -> None:
        if x.shape != self.diagonal.shape or x.dtype != self.diagonal.dtype or x.device != self.diagonal.device:
            raise ValueError("Direction/cotangent must match diagonal shape, dtype and device")

    def pull(self, cotangent: torch.Tensor) -> torch.Tensor:
        """Adjoint pullback (VJP): J^T @ cotangent = d * w + V (U^T w) in O(n r) FLOPs."""
        self._validate_vector(cotangent)
        if self.rank == 0:
            return self.diagonal * cotangent
        u_t_w = torch.matmul(self.left.transpose(-2, -1), cotangent.unsqueeze(-1))
        low_rank = torch.matmul(self.right, u_t_w).squeeze(-1)
        return self.diagonal * cotangent + low_rank

    def push(self, tangent: torch.Tensor) -> torch.Tensor:
        """Tangent pushforward (JVP): J @ tangent = d * v + U (V^T v) in O(n r) FLOPs."""
        self._validate_vector(tangent)
        if self.rank == 0:
            return self.diagonal * tangent
        v_t_v = torch.matmul(self.right.transpose(-2, -1), tangent.unsqueeze(-1))
        low_rank = torch.matmul(self.left, v_t_v).squeeze(-1)
        return self.diagonal * tangent + low_rank

    def materialize(self) -> torch.Tensor:
        """Materialize the full explicit [..., n, n] Jacobian matrix J = diag(d) + U @ V^T."""
        dense = torch.diag_embed(self.diagonal)
        if self.rank > 0:
            dense = dense + torch.matmul(self.left, self.right.transpose(-2, -1))
        return dense


# =============================================================================
# 2. Concrete Factor Constructors
# =============================================================================

def diagonal_derivative(diagonal: torch.Tensor) -> StructuredDerivative:
    """Construct a purely diagonal derivative (rank r=0)."""
    _vector_domain(diagonal)
    empty = diagonal.new_empty((*diagonal.shape, 0))
    return StructuredDerivative(diagonal, empty, empty)


def activation_derivative(x: torch.Tensor, kind: str) -> StructuredDerivative:
    """Construct a diagonal structured derivative for pointwise activations."""
    _vector_domain(x)
    if kind == "relu":
        d = (x > 0).to(x.dtype)
    elif kind == "silu":
        s = x.sigmoid()
        d = s + x * s * (1.0 - s)
    elif kind == "gelu":
        d = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    elif kind == "tanh":
        d = 1.0 - x.tanh().square()
    elif kind == "sigmoid":
        s = x.sigmoid()
        d = s * (1.0 - s)
    elif kind == "square":
        d = 2.0 * x
    else:
        raise ValueError(f"Unsupported activation derivative: {kind}")
    return diagonal_derivative(d)


def softmax_derivative(probabilities: torch.Tensor) -> StructuredDerivative:
    """Construct J = diag(p) - p p^T (rank r=1) from softmax output probabilities."""
    _vector_domain(probabilities)
    p = probabilities
    return StructuredDerivative(p, -p.unsqueeze(-1), p.unsqueeze(-1))


def rmsnorm_derivative(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> StructuredDerivative:
    """Construct the structured Jacobian factor for RMSNorm (rank r=1)."""
    gain = x.new_ones((x.shape[-1],)) if (gamma is None and x.ndim >= 1) else gamma
    _normalization_domain(x, gain, eps)
    n = x.shape[-1]
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    z = x * r
    d = (r * gain).expand_as(x)
    left = (-r * gain * z / n).unsqueeze(-1)
    right = z.unsqueeze(-1)
    return StructuredDerivative(d, left, right)


rmsnorm_input_derivative = rmsnorm_derivative


def layernorm_derivative(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> StructuredDerivative:
    """Construct the structured Jacobian factor for LayerNorm (rank r=2)."""
    gain = x.new_ones((x.shape[-1],)) if (gamma is None and x.ndim >= 1) else gamma
    _normalization_domain(x, gain, eps)
    n = x.shape[-1]
    c = x - x.mean(-1, keepdim=True)
    r = torch.rsqrt(c.square().mean(-1, keepdim=True) + eps)
    z = c * r
    d = (r * gain).expand_as(x)
    left = torch.stack((-d / n, -r * gain * z / n), dim=-1)
    right = torch.stack((torch.ones_like(x), z), dim=-1)
    return StructuredDerivative(d, left, right)


layernorm_input_derivative = layernorm_derivative


# =============================================================================
# 3. Compact Normalization Residual & Pullbacks
# =============================================================================

@dataclass(frozen=True)
class NormalizationResidual:
    """First-order sufficient normalization residual with pull() for dx, dgamma, dbeta."""
    kind: str
    normalized: torch.Tensor
    inverse_scale: torch.Tensor
    gamma: Optional[torch.Tensor]
    has_bias: bool

    def __post_init__(self) -> None:
        z = self.normalized
        r = self.inverse_scale
        _vector_domain(z)
        if self.kind not in ("layernorm", "rmsnorm") or z.shape[-1] == 0:
            raise ValueError("Expected supported normalization with nonempty coordinates")
        if r.shape != (*z.shape[:-1], 1) or r.dtype != z.dtype or r.device != z.device:
            raise ValueError("Inverse scale must match one value per normalization row")
        if self.gamma is not None and (
            self.gamma.shape != (z.shape[-1],) or self.gamma.dtype != z.dtype or self.gamma.device != z.device
        ):
            raise ValueError("Gain must match shared coordinate shape, dtype and device")
        if not isinstance(self.has_bias, bool):
            raise ValueError("Bias-presence flag must be Boolean")

    @property
    def residual_elements(self) -> int:
        return self.normalized.numel() + self.inverse_scale.numel()

    def pull(self, upstream: torch.Tensor):
        z = self.normalized
        r = self.inverse_scale
        if upstream.shape != z.shape or upstream.dtype != z.dtype or upstream.device != z.device:
            raise ValueError("Cotangent must match normalization output")
        q = upstream if self.gamma is None else upstream * self.gamma
        correction = z * (q * z).mean(-1, keepdim=True)
        if self.kind == "layernorm":
            dx = r * ((q - q.mean(-1, keepdim=True)) - correction)
        else:
            dx = r * (q - correction)
        dgamma = None if self.gamma is None else (upstream * z).sum_to_size(self.gamma.shape)
        dbeta = upstream.sum_to_size((z.shape[-1],)) if self.has_bias else None
        return dx, dgamma, dbeta


def normalization_residual(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    kind: str = "layernorm",
) -> NormalizationResidual:
    if kind not in ("layernorm", "rmsnorm"):
        raise ValueError("Unsupported normalization kind")
    gain = x.new_ones((x.shape[-1],)) if (gamma is None and x.ndim >= 1) else gamma
    _normalization_domain(x, gain, eps)
    if gamma is not None and gamma.shape != (x.shape[-1],):
        raise ValueError("Compact gain must have the shared coordinate shape")
    if beta is not None and (beta.shape != (x.shape[-1],) or beta.dtype != x.dtype or beta.device != x.device):
        raise ValueError("Compact bias must match shared coordinate shape, dtype and device")
    centered = x - x.mean(-1, keepdim=True) if kind == "layernorm" else x
    r = torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
    return NormalizationResidual(kind, centered * r, r, gamma, beta is not None)


def softmax_pullback(
    cotangent_or_probs: torch.Tensor,
    probabilities_or_cotangent: torch.Tensor,
) -> torch.Tensor:
    """Exact Softmax adjoint pullback in O(n) FLOPs."""
    t1, t2 = cotangent_or_probs, probabilities_or_cotangent
    if torch.all(t2 >= 0) and torch.allclose(t2.sum(dim=-1), torch.ones_like(t2.sum(dim=-1)), atol=1e-3):
        cotangent, p = t1, t2
    elif torch.all(t1 >= 0) and torch.allclose(t1.sum(dim=-1), torch.ones_like(t1.sum(dim=-1)), atol=1e-3):
        p, cotangent = t1, t2
    else:
        cotangent, p = t1, t2
    dot = (p * cotangent).sum(dim=-1, keepdim=True)
    return p * (cotangent - dot)


def rmsnorm_pullback(
    cotangent: torch.Tensor,
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact RMSNorm input cotangent pullback in O(n) FLOPs."""
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    z = x * r
    g_w = cotangent if gamma is None else cotangent * gamma
    correction = z * (g_w * z).mean(-1, keepdim=True)
    return r * (g_w - correction)


def layernorm_pullback(
    cotangent: torch.Tensor,
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact LayerNorm input cotangent pullback in O(n) FLOPs."""
    mean = x.mean(-1, keepdim=True)
    c = x - mean
    r = torch.rsqrt(c.square().mean(-1, keepdim=True) + eps)
    z = c * r
    g_w = cotangent if gamma is None else cotangent * gamma
    mean_gw = g_w.mean(-1, keepdim=True)
    corr_z = z * (g_w * z).mean(-1, keepdim=True)
    return r * (g_w - mean_gw - corr_z)


# =============================================================================
# 4. Reference Rank-7 Block Matrix Algebra (Strassen & Winograd)
# =============================================================================

def rank7_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One Strassen level on 2D matrices with positive even dimensions."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible matrices")
    if any(d <= 0 or d % 2 for d in (*a.shape, *b.shape)):
        raise ValueError("Expected positive even dimensions")
    if a.device != b.device or a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64):
        raise ValueError("Operands must share dtype and device")
    m, k, n = a.shape[0] // 2, a.shape[1] // 2, b.shape[1] // 2
    a00, a01, a10, a11 = a[:m, :k], a[:m, k:], a[m:, :k], a[m:, k:]
    b00, b01, b10, b11 = b[:k, :n], b[:k, n:], b[k:, :n], b[k:, n:]
    p1 = (a00 + a11) @ (b00 + b11)
    p2 = (a10 + a11) @ b00
    p3 = a00 @ (b01 - b11)
    p4 = a11 @ (b10 - b00)
    p5 = (a00 + a01) @ b11
    p6 = (a10 - a00) @ (b00 + b01)
    p7 = (a01 - a11) @ (b10 + b11)
    return torch.cat((torch.cat((p1 + p4 - p5 + p7, p3 + p5), 1), torch.cat((p2 + p4, p1 - p2 + p3 + p6), 1)), 0)


def winograd_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Known 7-product, 15-addition Winograd variant on 2D even matrices."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible matrices")
    if (
        any(d <= 0 or d % 2 for d in (*a.shape, *b.shape))
        or a.device != b.device
        or a.dtype != b.dtype
        or a.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Expected positive even dimensions and matching dtype/device")
    m, k, n = a.shape[0] // 2, a.shape[1] // 2, b.shape[1] // 2
    a00, a01, a10, a11 = a[:m, :k], a[:m, k:], a[m:, :k], a[m:, k:]
    b00, b01, b10, b11 = b[:k, :n], b[:k, n:], b[k:, :n], b[k:, n:]
    s1 = a10 + a11
    s2 = s1 - a00
    s3 = a00 - a10
    s4 = a01 - s2
    t1 = b01 - b00
    t2 = b11 - t1
    t3 = b11 - b01
    t4 = t2 - b10
    p1 = a00 @ b00
    p2 = a01 @ b10
    p3 = s4 @ b11
    p4 = a11 @ t4
    p5 = s1 @ t1
    p6 = s2 @ t2
    p7 = s3 @ t3
    u1 = p1 + p6
    u2 = u1 + p7
    u3 = u1 + p5
    return torch.cat((torch.cat((p1 + p2, u3 + p3), 1), torch.cat((u2 - p4, u2 + p5), 1)), 0)


__all__ = [
    "StructuredDerivative",
    "diagonal_derivative",
    "activation_derivative",
    "softmax_derivative",
    "rmsnorm_derivative",
    "rmsnorm_input_derivative",
    "layernorm_derivative",
    "layernorm_input_derivative",
    "NormalizationResidual",
    "normalization_residual",
    "softmax_pullback",
    "rmsnorm_pullback",
    "layernorm_pullback",
    "rank7_reference",
    "winograd_reference",
]
