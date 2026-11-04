"""Declarative semantics for the certified OjasX core.

The registry deliberately contains metadata and implementations together.  A
string annotation such as ``zero_tape`` is not evidence: every residual policy
must name the pullback slots it satisfies and the proof obligations that justify
it.  Runtime proof-check results belong in generated evidence reports, not in a
hard-coded ``verified=True`` flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .ir import JointGraph, Node
    from .jets import Jet


class AccuracyMode(str, Enum):
    """Declared numerical relationship to the pinned PyTorch reference."""

    STRICT = "strict"
    REPRODUCIBLE = "reproducible"
    BOUNDED_ERROR = "bounded_error"


class ProofSystem(str, Enum):
    LEAN = "lean4"
    ROCQ = "rocq"
    Z3 = "z3"


DENSE_COTANGENT = "dense"
UNIFORM_COTANGENT = "uniform"
COTANGENT_ENCODINGS = frozenset({DENSE_COTANGENT, UNIFORM_COTANGENT})


@dataclass(frozen=True)
class ProofRef:
    system: ProofSystem
    identifier: str
    premises: Tuple[str, ...] = ()
    artifact: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.identifier or any(not premise for premise in self.premises):
            raise ValueError("Proof references and their premises must be nonempty")


@dataclass(frozen=True)
class ResidualPolicy:
    """A sufficient representation of forward information for backward."""

    name: str
    accuracy: AccuracyMode
    lossless: bool
    satisfies: Tuple[str, ...]
    proofs: Tuple[ProofRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.satisfies or any(not slot for slot in self.satisfies):
            raise ValueError("A residual policy must name nonempty satisfied pullback slots")
        if self.accuracy == AccuracyMode.STRICT and not self.lossless:
            raise ValueError("Strict residual policies must be lossless")


@dataclass(frozen=True)
class OperatorContract:
    name: str
    min_arity: int
    max_arity: int
    accuracy: AccuracyMode
    deterministic: bool
    max_derivative_order: int
    residual_policies: Tuple[ResidualPolicy, ...]
    derivative_domain: str = "finite smooth inputs"
    proofs: Tuple[ProofRef, ...] = ()
    attribute_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.min_arity < 1 or self.max_arity < self.min_arity:
            raise ValueError("Invalid operator name or arity")
        if not 1 <= self.max_derivative_order <= 4 or not self.derivative_domain:
            raise ValueError("The core supports explicit derivative orders one through four")
        if any(not name for name in self.attribute_names) or len(set(self.attribute_names)) != len(self.attribute_names):
            raise ValueError(f"Invalid attribute metadata on {self.name}")
        names = [policy.name for policy in self.residual_policies]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate residual policy on {self.name}")

    def policy(self, name: str) -> ResidualPolicy:
        for policy in self.residual_policies:
            if policy.name == name:
                return policy
        raise KeyError(f"Operator {self.name!r} has no residual policy {name!r}")


@dataclass(frozen=True)
class ResidualBuild:
    role: str
    encoding: str
    gradient_slots: Tuple[int, ...]
    source_index: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    nbytes: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            not self.role
            or not self.encoding
            or not self.gradient_slots
            or any(slot < 0 for slot in self.gradient_slots)
            or len(set(self.gradient_slots)) != len(self.gradient_slots)
        ):
            raise ValueError("Residual role, encoding, and gradient slots must be valid")
        if self.source_index is not None and self.source_index < -1:
            raise ValueError("Residual source -1 denotes the output; nonnegative indices denote inputs")
        if self.source_index is None:
            if self.shape is None or self.dtype is None or self.nbytes is None:
                raise ValueError("A newly allocated residual needs shape, dtype, and byte size")
        elif any(value is not None for value in (self.shape, self.dtype, self.nbytes)):
            raise ValueError("An aliased residual takes its layout from its source value")


class CoreOperator(ABC):
    """Executable implementation attached to one declarative contract."""

    contract: OperatorContract

    @abstractmethod
    def output_shape(self, inputs: Sequence["Value"], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        """Return the primal output shape or raise on an invalid signature."""

    @abstractmethod
    def residual_layout(
        self,
        inputs: Sequence["Value"],
        demand: Tuple[bool, ...],
        policy: str,
    ) -> Tuple["ResidualBuild", ...]:
        """Describe sufficient residual values without executing the operation."""

    def output_storage(self, inputs: Sequence["Value"], attrs: Mapping[str, Any]) -> Optional[int]:
        """Return an input index aliased by the primal output, if guaranteed."""

        return None

    def jet_forward(self, inputs: Sequence["Jet"], attrs: Mapping[str, Any]) -> "Jet":
        """Evaluate truncated Taylor coefficients; no implicit fallback."""

        raise NotImplementedError(f"{self.contract.name} has no jet implementation")

    @abstractmethod
    def forward_workspace(self, inputs: Sequence["Value"], demand: Tuple[bool, ...], policy: str) -> int:
        """Conservative explicit-temporary bytes for the reference execution."""

    @abstractmethod
    def backward_workspace(self, inputs: Sequence["Value"], demand: Tuple[bool, ...], policy: str) -> int:
        """Conservative explicit-temporary bytes for the pullback execution."""

    def backward_workspace_for_node(
        self,
        inputs: Sequence["Value"],
        node: "Node",
    ) -> int:
        """Node-aware workspace hook for attribute-dependent replay policies."""

        return self.backward_workspace(inputs, node.demand, node.residual_policy)

    def accepts_cotangent(self, encoding: str) -> bool:
        """Declare a backward implementation safe for a structured upstream."""

        return encoding == DENSE_COTANGENT

    def cotangent_encoding(
        self,
        graph: "JointGraph",
        node: "Node",
        output_position: int,
    ) -> str:
        """Return the representation produced by one pullback output."""

        return DENSE_COTANGENT

    @abstractmethod
    def forward(
        self,
        tensors: Sequence[torch.Tensor],
        attrs: Mapping[str, Any],
        policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the primal value and residual payloads."""

    @abstractmethod
    def backward(
        self,
        graph: "JointGraph",
        node: "Node",
        upstream: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Compute cotangents in ``node.gradient_input_indices`` order."""


@dataclass(frozen=True)
class Value:
    id: int
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    storage_id: str
    storage_bytes: int
    producer: Optional[int]
    requires_grad: bool
    kind: str
    role: str = ""
    encoding: str = DENSE_COTANGENT


class OperatorRegistry:
    """Unique, inspectable registry with no implicit fallback registration."""

    def __init__(self) -> None:
        self._operators: Dict[str, CoreOperator] = {}

    def register(self, operator: CoreOperator) -> None:
        name = operator.contract.name
        if name in self._operators:
            raise ValueError(f"Operator {name!r} is already registered")
        self._operators[name] = operator

    def get(self, name: str) -> CoreOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"Unknown certified-core operator {name!r}") from exc

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._operators))

    def snapshot(self) -> Mapping[str, OperatorContract]:
        return MappingProxyType({name: self._operators[name].contract for name in self.names()})

    def __len__(self) -> int:
        return len(self._operators)


CORE_REGISTRY = OperatorRegistry()


def tensor_nbytes(shape: Iterable[int], dtype: torch.dtype) -> int:
    dimensions = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"Negative tensor dimension in {dimensions}")
    count = 1
    for dimension in dimensions:
        count *= dimension
    # torch.empty is meta, so this obtains the element size without storage.
    return int(count * torch.empty((), dtype=dtype, device="meta").element_size())


__all__ = [
    "AccuracyMode",
    "CORE_REGISTRY",
    "COTANGENT_ENCODINGS",
    "DENSE_COTANGENT",
    "UNIFORM_COTANGENT",
    "CoreOperator",
    "OperatorContract",
    "OperatorRegistry",
    "ProofRef",
    "ProofSystem",
    "ResidualBuild",
    "ResidualPolicy",
    "Value",
    "tensor_nbytes",
]
