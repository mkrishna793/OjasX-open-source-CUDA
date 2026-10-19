"""
ojasx2/math/adjoints.py

Structured Low-Rank Adjoints & O(nr) FLOP Pullback Engine
=========================================================
Implements canonical representation and computation for structured Jacobians:
    J = diag(d) + U V^T
where:
    d in R^n        (diagonal operator)
    U in R^{n x r}  (left low-rank factor)
    V in R^{n x r}  (right low-rank factor)
    r << n          (low rank, typically r in {1, 2})

Key Mathematical Properties:
1. Tangent Pushforward:
       push(v) = J v = d * v + U (V^T v)
   Computed in O(n r) FLOPs rather than O(n^2).
2. Cotangent Pullback (Adjoint VJP):
       pull(w) = J^T w = d * w + V (U^T w)
   Computed in O(n r) FLOPs rather than O(n^2).
3. Concrete Factor Constructors:
   - Softmax:   J = diag(p) - p p^T               (rank r=1, FLOPs: 4n)
   - RMSNorm:   J = diag(g*r) - (g*r*z/n) z^T      (rank r=1, FLOPs: 4n)
   - LayerNorm: J = diag(g*r) - u1 1^T - u2 z^T    (rank r=2, FLOPs: 8n)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch


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
    
    Supports arbitrary leading batch dimensions [..., n].
    """
    diagonal: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        if self.diagonal.ndim < 1:
            raise ValueError(f"Diagonal must have at least 1 dimension, got {self.diagonal.ndim}")
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
        """Coordinate dimension n."""
        return self.diagonal.shape[-1]

    @property
    def rank(self) -> int:
        """Low-rank factor dimension r."""
        return self.left.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions."""
        return self.diagonal.shape[:-1]

    def pull(self, cotangent: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pullback (VJP): J^T @ cotangent in O(n r) FLOPs.
            J^T w = d * w + V (U^T w)
        """
        if cotangent.shape != self.diagonal.shape:
            raise ValueError(f"Cotangent shape {cotangent.shape} must match diagonal shape {self.diagonal.shape}")
        if self.rank == 0:
            return self.diagonal * cotangent

        # 1. Project cotangent onto left factor U: s = U^T w in R^r (2 * n * r FLOPs)
        # left: [..., n, r], cotangent: [..., n] -> [..., r, 1]
        u_t_w = torch.matmul(self.left.transpose(-2, -1), cotangent.unsqueeze(-1))

        # 2. Re-project along right factor V: low_rank = V s in R^n (2 * n * r FLOPs)
        low_rank = torch.matmul(self.right, u_t_w).squeeze(-1)

        # 3. Add diagonal contribution (n FLOPs)
        return self.diagonal * cotangent + low_rank

    def push(self, tangent: torch.Tensor) -> torch.Tensor:
        """
        Tangent pushforward (JVP): J @ tangent in O(n r) FLOPs.
            J v = d * v + U (V^T v)
        """
        if tangent.shape != self.diagonal.shape:
            raise ValueError(f"Tangent shape {tangent.shape} must match diagonal shape {self.diagonal.shape}")
        if self.rank == 0:
            return self.diagonal * tangent

        # 1. Project tangent onto right factor V: t = V^T v in R^r (2 * n * r FLOPs)
        v_t_v = torch.matmul(self.right.transpose(-2, -1), tangent.unsqueeze(-1))

        # 2. Re-project along left factor U: low_rank = U t in R^n (2 * n * r FLOPs)
        low_rank = torch.matmul(self.left, v_t_v).squeeze(-1)

        # 3. Add diagonal contribution (n FLOPs)
        return self.diagonal * tangent + low_rank

    def materialize(self) -> torch.Tensor:
        """
        Materialize the full explicit [..., n, n] Jacobian matrix for debugging or testing.
        J = diag(d) + U @ V^T in O(n^2 r) FLOPs.
        """
        dense = torch.diag_embed(self.diagonal)
        if self.rank > 0:
            dense = dense + torch.matmul(self.left, self.right.transpose(-2, -1))
        return dense


# =============================================================================
# 2. Concrete Factor Constructors
# =============================================================================

def diagonal_derivative(diagonal: torch.Tensor) -> StructuredDerivative:
    """Construct a purely diagonal derivative (rank r=0)."""
    empty = diagonal.new_empty((*diagonal.shape, 0))
    return StructuredDerivative(diagonal, empty, empty)


def softmax_derivative(probabilities: torch.Tensor) -> StructuredDerivative:
    """
    Construct the structured Jacobian factor for Softmax:
        J_{ij} = p_i delta_{ij} - p_i p_j
        J = diag(p) - p p^T (rank r=1)
    
    Args:
        probabilities: Softmax output probabilities p in [..., n] (caller supplies p).
    """
    if probabilities.ndim < 1:
        raise ValueError("Probabilities must have at least 1 coordinate dimension")
    p = probabilities
    left = -p.unsqueeze(-1)   # [..., n, 1]
    right = p.unsqueeze(-1)   # [..., n, 1]
    return StructuredDerivative(p, left, right)


def rmsnorm_derivative(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> StructuredDerivative:
    """
    Construct the structured Jacobian factor for RMSNorm:
        y = gamma * (x * r) where r = rsqrt(mean(x^2) + eps)
        J = diag(g * r) - (1/n * g * r * z) z^T (rank r=1)
        where z = x * r is the normalized state.
    """
    if x.ndim < 1:
        raise ValueError("Input x must have at least 1 coordinate dimension")
    n = x.shape[-1]
    if n == 0:
        raise ValueError("Coordinate dimension must be non-empty")

    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    z = x * r
    g = gamma if gamma is not None else torch.ones_like(x[..., :1])
    d = (g * r).expand_as(x)

    left = (-r * g * z / n).unsqueeze(-1)  # [..., n, 1]
    right = z.unsqueeze(-1)                # [..., n, 1]
    return StructuredDerivative(d, left, right)


def layernorm_derivative(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> StructuredDerivative:
    """
    Construct the structured Jacobian factor for LayerNorm:
        c = x - mean(x)
        r = rsqrt(mean(c^2) + eps)
        z = c * r
        J = diag(g * r) - (d/n) 1^T - (r * g * z / n) z^T (rank r=2)
    """
    if x.ndim < 1:
        raise ValueError("Input x must have at least 1 coordinate dimension")
    n = x.shape[-1]
    if n == 0:
        raise ValueError("Coordinate dimension must be non-empty")

    mean = x.mean(-1, keepdim=True)
    c = x - mean
    r = torch.rsqrt(c.square().mean(-1, keepdim=True) + eps)
    z = c * r
    g = gamma if gamma is not None else torch.ones_like(x[..., :1])
    d = (g * r).expand_as(x)

    # Rank-2 factors: u1 = -d/n, v1 = 1; u2 = -r*g*z/n, v2 = z
    ones = torch.ones_like(x)
    u1 = -d / n
    u2 = -r * g * z / n
    left = torch.stack([u1, u2], dim=-1)   # [..., n, 2]
    right = torch.stack([ones, z], dim=-1) # [..., n, 2]
    return StructuredDerivative(d, left, right)


# =============================================================================
# 3. Pullback Convenience Functions (VJP)
# =============================================================================

def softmax_pullback(
    cotangent_or_probs: torch.Tensor,
    probabilities_or_cotangent: torch.Tensor,
) -> torch.Tensor:
    """
    Exact Softmax adjoint pullback in O(n) FLOPs:
        pull(w) = p * (w - <p, w>)
    Accepts either argument order (cotangent, probabilities) or (probabilities, upstream).
    """
    # Heuristic disambiguation: probabilities are non-negative and sum to ~1 along last axis
    t1, t2 = cotangent_or_probs, probabilities_or_cotangent
    if torch.all(t2 >= 0) and torch.allclose(t2.sum(dim=-1), torch.ones_like(t2.sum(dim=-1)), atol=1e-3):
        cotangent, p = t1, t2
    elif torch.all(t1 >= 0) and torch.allclose(t1.sum(dim=-1), torch.ones_like(t1.sum(dim=-1)), atol=1e-3):
        p, cotangent = t1, t2
    else:
        # Default signature (cotangent, probabilities)
        cotangent, p = t1, t2

    dot = (p * cotangent).sum(dim=-1, keepdim=True)
    return p * (cotangent - dot)


def rmsnorm_pullback(
    cotangent: torch.Tensor,
    x: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Exact RMSNorm input cotangent pullback in O(n) FLOPs:
        dx = r * (g * w - z * mean((g * w) * z))
    where z = x * r and r = rsqrt(mean(x^2) + eps).
    """
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
    """
    Exact LayerNorm input cotangent pullback in O(n) FLOPs:
        dx = r * (g*w - mean(g*w) - z * mean((g*w) * z))
    where c = x - mean(x), r = rsqrt(mean(c^2) + eps), and z = c * r.
    """
    mean = x.mean(-1, keepdim=True)
    c = x - mean
    r = torch.rsqrt(c.square().mean(-1, keepdim=True) + eps)
    z = c * r
    g_w = cotangent if gamma is None else cotangent * gamma
    mean_gw = g_w.mean(-1, keepdim=True)
    corr_z = z * (g_w * z).mean(-1, keepdim=True)
    return r * (g_w - mean_gw - corr_z)


__all__ = [
    "StructuredDerivative",
    "diagonal_derivative",
    "softmax_derivative",
    "rmsnorm_derivative",
    "layernorm_derivative",
    "softmax_pullback",
    "rmsnorm_pullback",
    "layernorm_pullback",
]
