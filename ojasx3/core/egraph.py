"""Small, deterministic equality-saturation engine with proof-gated rewrites."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .contracts import AccuracyMode, ProofRef
from .planning import ExecutionObjective


Attrs = Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple["Expr", ...] = ()
    attrs: Attrs = ()
    policy: str = ""
    shape: Tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32

    def structural(self) -> str:
        attributes = f", attrs={self.attrs!r}" if self.attrs else ""
        policy = f", policy={self.policy!r}" if self.policy else ""
        if not self.args:
            return f"{self.op}({dict(self.attrs)!r})" if self.attrs else f"{self.op}()"
        return (
            f"{self.op}(" + ", ".join(arg.structural() for arg in self.args)
            + attributes + policy + ")"
        )


@dataclass(frozen=True)
class ENode:
    op: str
    args: Tuple[int, ...]
    attrs: Attrs
    policy: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    obligation: Optional["ProofObligation"] = field(
        default=None,
        compare=False,
        hash=False,
    )

    def key(self) -> Tuple[object, ...]:
        return (
            self.op,
            self.args,
            self.attrs,
            self.policy,
            self.shape,
            self.dtype,
        )


@dataclass(frozen=True)
class ProofObligation:
    rule: str
    family: str
    before: str
    after: str
    semantics: str
    proofs: Tuple[ProofRef, ...]
    side_conditions: Tuple[str, ...] = ()
    error_bound: Optional[float] = None


@dataclass(frozen=True)
class Rewrite:
    node: ENode
    before: str
    after: str
    side_conditions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteRule:
    name: str
    family: str
    semantics: str
    proofs: Tuple[ProofRef, ...]
    apply: Callable[["EGraph", int, ENode], Iterable[Rewrite]]
    error_bound: Optional[float] = None


class UnprovedRewriteError(RuntimeError):
    pass


@dataclass
class EClass:
    id: int
    parent: int
    rank: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    nodes: set[ENode]
    # Conservative provenance closure for every equality that merged into this
    # class. Extraction includes it even when the selected enode predates a merge.
    obligations: set[ProofObligation] = field(default_factory=set)


@dataclass(frozen=True)
class SaturationReport:
    iterations: int
    saturated: bool
    eclasses: int
    enodes: int
    generated_obligations: int


@dataclass(frozen=True)
class ExtractedCandidate:
    expr: Expr
    obligations: Tuple[ProofObligation, ...]
    work_units: int
    traffic_bytes: int
    residual_bytes: int

    @property
    def additive_key(self) -> Tuple[object, ...]:
        return (
            self.work_units,
            self.traffic_bytes,
            self.residual_bytes,
            _depth(self.expr),
            self.expr.structural(),
        )


def _depth(expr: Expr) -> int:
    return 1 + max((_depth(arg) for arg in expr.args), default=0)


class EGraph:
    """Hash-consed e-classes with congruence repair and bounded extraction."""

    def __init__(
        self,
        *,
        allow_exact_real: bool = False,
        objective: Optional[ExecutionObjective] = None,
    ) -> None:
        self.objective = objective or ExecutionObjective()
        if allow_exact_real and self.objective.accuracy != AccuracyMode.REPRODUCIBLE:
            raise ValueError("Exact-real rewrites require the explicit reproducible-mode opt-in")
        self.allow_exact_real = allow_exact_real
        self.classes: Dict[int, EClass] = {}
        self._hashcons: Dict[Tuple[object, ...], int] = {}
        self._next_id = 0
        self.rules: List[RewriteRule] = []
        self.generated_obligations: List[ProofObligation] = []

    def find(self, class_id: int) -> int:
        if class_id not in self.classes:
            raise KeyError(f"Unknown e-class {class_id}")
        parent = self.classes[class_id].parent
        if parent != class_id:
            self.classes[class_id].parent = self.find(parent)
        return self.classes[class_id].parent

    def canonical(self, node: ENode) -> ENode:
        return ENode(
            node.op,
            tuple(self.find(arg) for arg in node.args),
            node.attrs,
            node.policy,
            node.shape,
            node.dtype,
            node.obligation,
        )

    def spec(self, class_id: int) -> Tuple[Tuple[int, ...], torch.dtype]:
        root = self.find(class_id)
        return self.classes[root].shape, self.classes[root].dtype

    def intern(self, node: ENode) -> int:
        node = self.canonical(node)
        existing = self._hashcons.get(node.key())
        if existing is not None:
            return self.find(existing)
        class_id = self._next_id
        self._next_id += 1
        self.classes[class_id] = EClass(
            class_id,
            class_id,
            0,
            node.shape,
            node.dtype,
            {node},
        )
        self._hashcons[node.key()] = class_id
        return class_id

    def add_expr(self, expr: Expr) -> int:
        args = tuple(self.add_expr(arg) for arg in expr.args)
        return self.intern(
            ENode(
                expr.op,
                args,
                expr.attrs,
                expr.policy,
                expr.shape,
                expr.dtype,
            )
        )

    def union(self, left: int, right: int) -> bool:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return False
        a = self.classes[left]
        b = self.classes[right]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Only e-classes with identical tensor signatures may merge")
        if a.rank < b.rank or (a.rank == b.rank and left > right):
            left, right = right, left
            a, b = b, a
        b.parent = left
        a.nodes.update(b.nodes)
        a.obligations.update(b.obligations)
        if a.rank == b.rank:
            a.rank += 1
        return True

    def rebuild(self) -> None:
        changed = True
        while changed:
            changed = False
            roots = [class_id for class_id in self.classes if self.find(class_id) == class_id]
            seen: Dict[Tuple[object, ...], int] = {}
            for class_id in roots:
                eclass = self.classes[class_id]
                canonical_nodes = {self.canonical(node) for node in eclass.nodes}
                eclass.nodes = canonical_nodes
                for node in sorted(canonical_nodes, key=lambda item: repr(item.key())):
                    key = node.key()
                    other = seen.get(key)
                    if other is None:
                        seen[key] = class_id
                    elif self.find(other) != self.find(class_id):
                        changed = self.union(other, class_id) or changed
        self._hashcons = {}
        for class_id, eclass in self.classes.items():
            if self.find(class_id) == class_id:
                for node in eclass.nodes:
                    canonical = self.canonical(node)
                    self._hashcons[canonical.key()] = class_id

    def register_rule(self, rule: RewriteRule) -> None:
        if any(existing.name == rule.name for existing in self.rules):
            raise ValueError(f"Duplicate rewrite rule {rule.name!r}")
        self.rules.append(rule)

    def _apply_rule(self, rule: RewriteRule, class_id: int, node: ENode) -> int:
        if not self.objective.admits(
            rule.semantics,
            rule.error_bound,
            allow_exact_real=self.allow_exact_real,
        ):
            return 0
        if not rule.proofs:
            raise UnprovedRewriteError(
                f"Rewrite {rule.name!r} has no registered proof and cannot be applied"
            )
        additions = 0
        root = self.find(class_id)
        for rewrite in rule.apply(self, root, node):
            obligation = ProofObligation(
                rule.name,
                rule.family,
                rewrite.before,
                rewrite.after,
                rule.semantics,
                rule.proofs,
                rewrite.side_conditions,
                rule.error_bound,
            )
            proven = ENode(
                rewrite.node.op,
                rewrite.node.args,
                rewrite.node.attrs,
                rewrite.node.policy,
                rewrite.node.shape,
                rewrite.node.dtype,
                obligation,
            )
            new_class = self.intern(proven)
            if new_class == root:
                continue
            # If the node was already present in the target class, the original
            # proof-free alternative is sufficient and no duplicate obligation
            # is needed for extraction.
            self.union(root, new_class)
            self.classes[self.find(root)].obligations.add(obligation)
            self.generated_obligations.append(obligation)
            additions += 1
        return additions

    def saturate(self, *, max_iterations: int = 12) -> SaturationReport:
        if max_iterations <= 0:
            raise ValueError("Equality saturation requires a positive iteration bound")
        saturated = False
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            before = (
                sum(self.find(class_id) == class_id for class_id in self.classes),
                sum(len(eclass.nodes) for eclass in self.classes.values()),
            )
            snapshot = sorted(
                [
                    (self.find(class_id), node)
                    for class_id in list(self.classes)
                    if self.find(class_id) == class_id
                    for node in tuple(self.classes[self.find(class_id)].nodes)
                ],
                key=lambda item: (item[0], repr(item[1].key())),
            )
            for rule in self.rules:
                for class_id, node in snapshot:
                    if self.find(class_id) == class_id:
                        self._apply_rule(rule, class_id, node)
                    self.rebuild()
            self.rebuild()
            after = (
                sum(self.find(class_id) == class_id for class_id in self.classes),
                sum(len(eclass.nodes) for eclass in self.classes.values()),
            )
            if before == after:
                saturated = True
                break
        return SaturationReport(
            iterations,
            saturated,
            sum(self.find(class_id) == class_id for class_id in self.classes),
            sum(
                len(eclass.nodes)
                for class_id, eclass in self.classes.items()
                if self.find(class_id) == class_id
            ),
            len(self.generated_obligations),
        )

    def extract(
        self,
        root: int,
        *,
        local_cost: Callable[[ENode], Tuple[int, int, int]],
        beam_width: int = 32,
        max_expansions_per_node: int = 256,
    ) -> Tuple[List[ExtractedCandidate], bool]:
        if beam_width <= 0 or max_expansions_per_node <= 0:
            raise ValueError("Extraction bounds must be positive")
        memo: Dict[int, List[ExtractedCandidate]] = {}
        truncated = False
        visiting = set()

        def visit(class_id: int) -> List[ExtractedCandidate]:
            nonlocal truncated
            root_id = self.find(class_id)
            if root_id in memo:
                return memo[root_id]
            if root_id in visiting:
                # Equalities such as transpose(transpose(A)) = A naturally make
                # cyclic e-classes. Skip this alternative, retain an acyclic
                # representative, and conservatively clear extraction optimality.
                truncated = True
                return []
            visiting.add(root_id)
            candidates: Dict[Expr, ExtractedCandidate] = {}
            for node in sorted(self.classes[root_id].nodes, key=lambda item: repr(item.key())):
                child_candidates = [visit(arg) for arg in node.args]
                products = itertools.islice(
                    itertools.product(*child_candidates),
                    max_expansions_per_node + 1,
                )
                count = 0
                for children in products:
                    count += 1
                    if count > max_expansions_per_node:
                        truncated = True
                        break
                    expr = Expr(
                        node.op,
                        tuple(child.expr for child in children),
                        node.attrs,
                        node.policy,
                        node.shape,
                        node.dtype,
                    )
                    work, traffic, residual = local_cost(node)
                    obligations = tuple(
                        dict.fromkeys(
                            tuple(
                                sorted(
                                    self.classes[root_id].obligations,
                                    key=lambda item: (item.rule, item.before, item.after),
                                )
                            )
                            + ((node.obligation,) if node.obligation else ())
                            + tuple(
                                obligation
                                for child in children
                                for obligation in child.obligations
                            )
                        )
                    )
                    candidate = ExtractedCandidate(
                        expr,
                        obligations,
                        work + sum(child.work_units for child in children),
                        traffic + sum(child.traffic_bytes for child in children),
                        residual + sum(child.residual_bytes for child in children),
                    )
                    previous = candidates.get(expr)
                    if previous is None or len(candidate.obligations) < len(previous.obligations):
                        candidates[expr] = candidate
            visiting.remove(root_id)
            ordered = sorted(candidates.values(), key=lambda item: item.additive_key)
            if len(ordered) > beam_width:
                truncated = True
                ordered = ordered[:beam_width]
            if ordered:
                memo[root_id] = ordered
            return ordered

        return visit(root), truncated


__all__ = [
    "EGraph",
    "ENode",
    "Expr",
    "ExtractedCandidate",
    "ProofObligation",
    "Rewrite",
    "RewriteRule",
    "SaturationReport",
    "UnprovedRewriteError",
]
