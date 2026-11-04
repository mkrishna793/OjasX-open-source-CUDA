"""Typed global execution objectives and backward-inclusive logical costs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from .contracts import AccuracyMode, ProofRef, ProofSystem

if TYPE_CHECKING:
    from .memory import MemoryCertificate


PLANNER_PROOFS = {
    "original_fallback": (
        ProofRef(ProofSystem.LEAN, "OjasXMath.m4_original_retained_without_strict_improvement"),
        ProofRef(ProofSystem.ROCQ, "Ojasx.M4Equality.m4_original_retained_without_strict_improvement"),
        ProofRef(ProofSystem.Z3, "m4_original_retained_without_strict_improvement"),
    ),
    "bounded_error_budget": (
        ProofRef(ProofSystem.LEAN, "OjasXMath.m4_bounded_error_budget_nonnegative"),
        ProofRef(ProofSystem.ROCQ, "Ojasx.M4Equality.m4_bounded_error_budget_nonnegative"),
        ProofRef(ProofSystem.Z3, "m4_bounded_error_budget_nonnegative"),
    ),
}


@dataclass(frozen=True)
class ExecutionObjective:
    """Requested optimization order and admitted numerical semantics.

    Reproducible rules are valid in every mode. Exact-real rules retain the
    legacy explicit opt-in, but are not accepted as strict or bounded-error
    proofs. A bounded-error rule additionally needs a concrete bound no larger
    than the request's budget.
    """

    primary: str = "memory"
    accuracy: AccuracyMode = AccuracyMode.REPRODUCIBLE
    error_budget: Optional[float] = None

    def __post_init__(self) -> None:
        if self.primary not in {"memory", "work", "traffic"}:
            raise ValueError("Execution objective must be memory, work, or traffic")
        if not isinstance(self.accuracy, AccuracyMode):
            object.__setattr__(self, "accuracy", AccuracyMode(self.accuracy))
        if self.accuracy == AccuracyMode.BOUNDED_ERROR:
            if (
                self.error_budget is None
                or not math.isfinite(self.error_budget)
                or self.error_budget < 0
            ):
                raise ValueError("Bounded-error execution requires a finite nonnegative budget")
        elif self.error_budget is not None:
            raise ValueError("An error budget is valid only in bounded-error mode")

    def admits(
        self,
        semantics: str,
        error_bound: Optional[float] = None,
        *,
        allow_exact_real: bool = False,
    ) -> bool:
        if semantics == "core_reproducible":
            return True
        if semantics == "exact_real":
            return (
                allow_exact_real
                and self.accuracy == AccuracyMode.REPRODUCIBLE
            )
        if semantics == "bounded_error":
            return (
                self.accuracy == AccuracyMode.BOUNDED_ERROR
                and self.error_budget is not None
                and error_bound is not None
                and math.isfinite(error_bound)
                and 0 <= error_bound <= self.error_budget
            )
        raise ValueError(f"Unknown rewrite semantics {semantics!r}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary": self.primary,
            "accuracy": self.accuracy.value,
            "error_budget": self.error_budget,
        }


@dataclass(frozen=True)
class LogicalCost:
    """One comparable point in the global execution-cost space."""

    memory_bytes: int
    work_units: int
    traffic_bytes: int
    final_bytes: int

    @classmethod
    def from_certificate(cls, certificate: "MemoryCertificate") -> "LogicalCost":
        return cls(
            certificate.peak_bytes,
            certificate.work_units,
            certificate.traffic_bytes,
            certificate.final_bytes,
        )

    def key(self, objective: ExecutionObjective) -> Tuple[int, int, int, int]:
        if objective.primary == "memory":
            return self.memory_bytes, self.work_units, self.traffic_bytes, self.final_bytes
        if objective.primary == "work":
            return self.work_units, self.memory_bytes, self.traffic_bytes, self.final_bytes
        return self.traffic_bytes, self.work_units, self.memory_bytes, self.final_bytes

    def fits_sram_budget(self, sram_budget_bytes: int) -> bool:
        if sram_budget_bytes <= 0:
            return True
        return self.memory_bytes <= sram_budget_bytes

    def to_dict(self) -> Dict[str, int]:
        return {
            "memory_bytes": self.memory_bytes,
            "work_units": self.work_units,
            "traffic_bytes": self.traffic_bytes,
            "final_bytes": self.final_bytes,
        }


__all__ = ["ExecutionObjective", "LogicalCost", "PLANNER_PROOFS"]
