"""M0/M1 structured-cotangent representations and topology certificates.

A uniform cotangent is the ones morphism scaled by one scalar.  It has a large
logical tensor shape but only one scalar of managed storage.  Dense boundaries
are explicit IR nodes, so representation changes cannot hide an allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import torch

from .contracts import (
    DENSE_COTANGENT,
    UNIFORM_COTANGENT,
    ProofRef,
    ProofSystem,
)

if TYPE_CHECKING:
    from .ir import JointGraph


class UniformCotangentAcceptor:
    """Mixin for backward implementations qualified for uniform upstreams."""

    def accepts_cotangent(self, encoding: str) -> bool:
        return encoding in {DENSE_COTANGENT, UNIFORM_COTANGENT}


@dataclass(frozen=True)
class RepresentationRule:
    name: str
    family: str
    operators: Tuple[str, ...]
    proofs: Tuple[ProofRef, ...]


def _proofs(name: str) -> Tuple[ProofRef, ...]:
    return (
        ProofRef(ProofSystem.LEAN, f"OjasXMath.{name}"),
        ProofRef(ProofSystem.ROCQ, f"Ojasx.StructuredTopology.{name}"),
        ProofRef(ProofSystem.Z3, name),
    )


REPRESENTATION_RULES = (
    RepresentationRule("structured_sum_uniform", "reduction", ("sum",), _proofs("structured_sum_uniform")),
    RepresentationRule("structured_mean_uniform", "reduction", ("mean",), _proofs("structured_mean_uniform")),
    RepresentationRule("structured_uniform_add", "additive_fanout", ("add", "cotangent_add"), _proofs("structured_uniform_add")),
    RepresentationRule("structured_uniform_reindex", "topology", ("reshape", "transpose"), _proofs("structured_uniform_reindex")),
    RepresentationRule("structured_materialize_identity", "boundary", ("cotangent_materialize",), _proofs("structured_materialize_identity")),
    RepresentationRule(
        "structured_uniform_region_storage",
        "topology",
        ("cotangent_add", "cotangent_materialize"),
        (
            ProofRef(ProofSystem.ROCQ, "Ojasx.StructuredTopology.structured_uniform_region_storage"),
            ProofRef(ProofSystem.Z3, "structured_uniform_region_storage"),
        ),
    ),
)


def nonempty(shape: Sequence[int]) -> bool:
    return math.prod(tuple(shape)) > 0


def uniform_scalar(value: torch.Tensor) -> torch.Tensor:
    """Read the scalar witness without flattening or copying the tensor."""

    if value.numel() == 0:
        raise ValueError("Empty tensors have no uniform scalar witness")
    return value[(0,) * value.ndim]


def clone_uniform(value: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Return an independently owned scalar expanded over ``shape``."""

    if not nonempty(shape):
        raise ValueError("Uniform representations require a nonempty shape")
    return uniform_scalar(value).clone().expand(tuple(shape))


def divide_uniform(value: torch.Tensor, divisor: int, shape: Sequence[int]) -> torch.Tensor:
    """Scale once, then broadcast; used by the mean adjoint."""

    if divisor <= 0:
        raise ValueError("A uniform mean adjoint requires a positive divisor")
    if not nonempty(shape):
        raise ValueError("Uniform representations require a nonempty shape")
    return (uniform_scalar(value) / divisor).expand(tuple(shape))


def add_uniform(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Add two uniform fields with one scalar addition."""

    if left.shape != right.shape or not nonempty(left.shape):
        raise ValueError("Uniform addition requires equal nonempty shapes")
    return (uniform_scalar(left) + uniform_scalar(right)).expand(left.shape)


@dataclass(frozen=True)
class CotangentTopology:
    uniform_values: Tuple[int, ...]
    uniform_regions: Tuple[Tuple[int, ...], ...]
    boundary_nodes: Tuple[int, ...]
    materialization_nodes: Tuple[int, ...]
    logical_bytes: int
    managed_bytes: int
    saved_bytes: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze_cotangent_topology(graph: "JointGraph") -> CotangentTopology:
    """Compute maximal uniform regions and their dense representation cuts.

    Region connectivity follows producer/consumer edges only when both values
    are uniform.  The graph validator has already checked every local encoding
    rule and required every public output to be dense.
    """

    graph.validate()
    uniform = {
        value_id
        for value_id, value in graph.values.items()
        if value.encoding == UNIFORM_COTANGENT
    }
    parent = {value_id: value_id for value_id in uniform}

    def find(value_id: int) -> int:
        while parent[value_id] != value_id:
            parent[value_id] = parent[parent[value_id]]
            value_id = parent[value_id]
        return value_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    boundaries: List[int] = []
    for node in graph.nodes:
        inputs = [value_id for value_id in node.inputs if value_id in uniform]
        outputs = [value_id for value_id in node.outputs if value_id in uniform]
        for left in inputs:
            for right in outputs:
                union(left, right)
        if inputs and (len(outputs) != len(node.outputs) or node.kind == "cotangent_materialize"):
            boundaries.append(node.id)

    groups: Dict[int, List[int]] = {}
    for value_id in sorted(uniform):
        groups.setdefault(find(value_id), []).append(value_id)
    regions = tuple(tuple(groups[key]) for key in sorted(groups))
    values = [graph.values[value_id] for value_id in sorted(uniform)]
    logical = sum(value.nbytes for value in values)
    managed = sum(value.storage_bytes for value in values)
    return CotangentTopology(
        uniform_values=tuple(sorted(uniform)),
        uniform_regions=regions,
        boundary_nodes=tuple(sorted(boundaries)),
        materialization_nodes=tuple(
            node.id for node in graph.nodes if node.kind == "cotangent_materialize"
        ),
        logical_bytes=logical,
        managed_bytes=managed,
        saved_bytes=logical - managed,
    )


__all__ = [
    "CotangentTopology",
    "RepresentationRule",
    "REPRESENTATION_RULES",
    "UniformCotangentAcceptor",
    "add_uniform",
    "analyze_cotangent_topology",
    "clone_uniform",
    "divide_uniform",
    "nonempty",
    "uniform_scalar",
]
