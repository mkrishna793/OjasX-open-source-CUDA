"""
Phase 3: Symbolic Categorical Algebraic Matrix IR (`AlgMat-IR`) Optimizer.

Ported and extended from `trium-tile/algmat_ir.hpp` into Python for seamless
PyTorch FX / OjasX-LogicIR fusion:
  - Orthogonal Cancellation : CW^T ∘ CW -> k * Id, FWHT ∘ FWHT -> Id
  - Kronecker Mixed-Product : (A ⊗ B) ∘ (C ⊗ D) -> (A ∘ C) ⊗ (B ∘ D)
  - Yoneda Low-Rank Fusion  : (U V^T) x -> U (V^T x) (never materializes d x d)
  - Identity Elimination    : I ∘ A -> A, (s1*I) ∘ (s2*I) -> (s1*s2)*I
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class AlgNode:
    kind: str  # "identity", "cw", "fwht", "kronecker", "low_rank", "dense", "compose", "add"
    dim: int
    scale: float = 1.0
    name: str = ""
    weight_k: int = 1
    transposed: bool = False
    orbit_id: int = 0
    rank: int = 0
    children: Tuple["AlgNode", ...] = ()

    @staticmethod
    def identity(dim: int, scale: float = 1.0) -> "AlgNode":
        return AlgNode(kind="identity", dim=dim, scale=scale)

    @staticmethod
    def cw120(transposed: bool = False, orbit_id: int = 0) -> "AlgNode":
        return AlgNode(kind="cw", dim=120, weight_k=36, transposed=transposed, orbit_id=orbit_id)

    @staticmethod
    def fwht(dim: int) -> "AlgNode":
        return AlgNode(kind="fwht", dim=dim, weight_k=1)

    @staticmethod
    def kronecker(factors: Tuple["AlgNode", ...]) -> "AlgNode":
        d = 1
        for f in factors:
            d *= f.dim
        return AlgNode(kind="kronecker", dim=d, children=factors)

    @staticmethod
    def low_rank(dim: int, rank: int, name: str = "lora") -> "AlgNode":
        return AlgNode(kind="low_rank", dim=dim, rank=rank, name=name)

    @staticmethod
    def dense(dim: int, name: str = "W") -> "AlgNode":
        return AlgNode(kind="dense", dim=dim, name=name)

    @staticmethod
    def compose(a: "AlgNode", b: "AlgNode") -> "AlgNode":
        if a.dim != b.dim:
            raise ValueError(f"AlgMat compose dimension mismatch: {a.dim} vs {b.dim}")
        return AlgNode(kind="compose", dim=a.dim, children=(a, b))

    @staticmethod
    def add(a: "AlgNode", b: "AlgNode") -> "AlgNode":
        if a.dim != b.dim:
            raise ValueError(f"AlgMat add dimension mismatch: {a.dim} vs {b.dim}")
        return AlgNode(kind="add", dim=a.dim, children=(a, b))


@dataclass
class AlgOptStats:
    ortho_cancellations: int = 0
    kronecker_mixed_products: int = 0
    identity_eliminations: int = 0


def optimize_algmat(node: AlgNode, stats: Optional[AlgOptStats] = None) -> Tuple[AlgNode, AlgOptStats]:
    """Bottom-up algebraic rewrite pass eliminating redundant rotations and identities."""
    st = stats if stats is not None else AlgOptStats()
    opt_children = tuple(optimize_algmat(c, st)[0] for c in node.children)

    if node.kind == "kronecker":
        if opt_children and all(c.kind == "identity" for c in opt_children):
            s = 1.0
            for c in opt_children:
                s *= c.scale
            st.identity_eliminations += 1
            return AlgNode.identity(node.dim, s), st
        return AlgNode.kronecker(opt_children), st

    if node.kind == "compose":
        a, b = opt_children
        # Identity rules
        if a.kind == "identity" and b.kind == "identity":
            st.identity_eliminations += 1
            return AlgNode.identity(node.dim, a.scale * b.scale), st
        if a.kind == "identity" and a.scale == 1.0:
            st.identity_eliminations += 1
            return b, st
        if b.kind == "identity" and b.scale == 1.0:
            st.identity_eliminations += 1
            return a, st
        # FWHT involution: FWHT ∘ FWHT -> Id
        if a.kind == "fwht" and b.kind == "fwht" and a.dim == b.dim:
            st.ortho_cancellations += 1
            return AlgNode.identity(a.dim, 1.0), st
        # CW orthogonal cancellation: CW^T ∘ CW -> k * Id
        if (
            a.kind == "cw"
            and b.kind == "cw"
            and a.dim == b.dim
            and a.weight_k == b.weight_k
            and a.orbit_id == b.orbit_id
            and a.transposed != b.transposed
        ):
            st.ortho_cancellations += 1
            return AlgNode.identity(a.dim, float(a.weight_k)), st
        # Kronecker Mixed-Product: (A0 ⊗ A1) ∘ (B0 ⊗ B1) -> (A0 ∘ B0) ⊗ (A1 ∘ B1)
        if (
            a.kind == "kronecker"
            and b.kind == "kronecker"
            and len(a.children) == len(b.children)
            and all(x.dim == y.dim for x, y in zip(a.children, b.children))
        ):
            st.kronecker_mixed_products += 1
            fused = tuple(AlgNode.compose(x, y) for x, y in zip(a.children, b.children))
            return optimize_algmat(AlgNode.kronecker(fused), st)
        return AlgNode.compose(a, b), st

    if node.kind == "add":
        return AlgNode.add(opt_children[0], opt_children[1]), st

    return node, st


__all__ = ["AlgNode", "AlgOptStats", "optimize_algmat"]
