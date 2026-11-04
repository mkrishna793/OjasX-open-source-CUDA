"""M4 proof-gated equality saturation and backward-inclusive schedule extraction."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .activation_math import ACTIVATION_NAMES, activation_program
from .contracts import CORE_REGISTRY, ProofRef, ProofSystem
from .egraph import (
    EGraph,
    ENode,
    Expr,
    ExtractedCandidate,
    ProofObligation,
    Rewrite,
    RewriteRule,
    SaturationReport,
)
from .ir import JointGraph, JointGraphBuilder
from .memory import MemoryBudgetError, MemoryCertificate, graph_digest, plan_memory
from .operators import register_core_operators
from .planning import ExecutionObjective, LogicalCost, PLANNER_PROOFS
from .structured import analyze_cotangent_topology


register_core_operators()


@dataclass(frozen=True)
class M4Input:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not self.name or any(dimension < 0 for dimension in self.shape):
            raise ValueError("M4 inputs require a name and nonnegative dimensions")
        if not self.dtype.is_floating_point:
            raise ValueError("M4 currently supports real floating tensors")


@dataclass(frozen=True)
class M4Problem:
    inputs: Tuple[M4Input, ...]
    root: Expr
    gradient_inputs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.inputs)
        if len(set(names)) != len(names):
            raise ValueError("M4 input names must be unique")
        requested = self.gradient_inputs or tuple(
            item.name for item in self.inputs if item.requires_grad
        )
        used = _input_names(self.root)
        if any(name not in names or name not in used for name in requested):
            raise ValueError("Every requested M4 gradient input must be used by the root")
        if any(not next(item for item in self.inputs if item.name == name).requires_grad for name in requested):
            raise ValueError("Cannot request an M4 gradient for a nondifferentiable input")
        object.__setattr__(self, "gradient_inputs", requested)


@dataclass(frozen=True)
class M4Config:
    max_iterations: int = 12
    beam_width: int = 32
    max_candidates: int = 64
    max_expansions_per_node: int = 256
    exact_node_limit: int = 12
    allow_exact_real: bool = False
    budget: Optional[int] = None
    objective: ExecutionObjective = ExecutionObjective()


@dataclass
class M4Plan:
    graph: JointGraph
    certificate: MemoryCertificate
    baseline_graph: JointGraph
    baseline_certificate: MemoryCertificate
    selected_expression: Expr
    baseline_expression: Expr
    proof_obligations: Tuple[ProofObligation, ...]
    generated_obligations: Tuple[ProofObligation, ...]
    saturation: SaturationReport
    evaluated_candidates: int
    optimal_for_searched_space: bool
    method: str
    objective: ExecutionObjective
    selected_cost: LogicalCost
    baseline_cost: LogicalCost

    @property
    def saved_bytes(self) -> int:
        return self.baseline_certificate.peak_bytes - self.certificate.peak_bytes

    @property
    def discovered(self) -> bool:
        return self.selected_expression != self.baseline_expression

    @property
    def rewrite_families(self) -> Tuple[str, ...]:
        return tuple(sorted({item.family for item in self.proof_obligations}))

    def to_dict(self) -> Dict[str, object]:
        return {
            "graph_digest": graph_digest(self.graph),
            "baseline_graph_digest": graph_digest(self.baseline_graph),
            "selected_expression": self.selected_expression.structural(),
            "baseline_expression": self.baseline_expression.structural(),
            "discovered": self.discovered,
            "baseline_peak_bytes": self.baseline_certificate.peak_bytes,
            "peak_bytes": self.certificate.peak_bytes,
            "saved_bytes": self.saved_bytes,
            "work_units": self.certificate.work_units,
            "traffic_bytes": self.certificate.traffic_bytes,
            "replay_evaluations": self.certificate.replay_evaluations,
            "objective": {
                "primary": self.objective.primary,
                "accuracy": self.objective.accuracy.value,
                "error_budget": self.objective.error_budget,
                "selected_key": list(self.selected_cost.key(self.objective)),
                "baseline_key": list(self.baseline_cost.key(self.objective)),
            },
            "selected_cost": self.selected_cost.to_dict(),
            "baseline_cost": self.baseline_cost.to_dict(),
            "planner_proofs": {
                name: [
                    {"system": proof.system.value, "identifier": proof.identifier}
                    for proof in proofs
                ]
                for name, proofs in PLANNER_PROOFS.items()
            },
            "structured_topology": analyze_cotangent_topology(self.graph).to_dict(),
            "residual_policies": {
                str(node.id): node.residual_policy
                for node in self.graph.nodes
                if node.kind == "primal" and node.residual_policy
            },
            "layout_view_nodes": [
                node.id
                for node in self.graph.nodes
                if node.kind == "primal" and node.op_name in {"reshape", "transpose"}
            ],
            "rewrite_families": list(self.rewrite_families),
            "proof_obligations": [_obligation_dict(item) for item in self.proof_obligations],
            "generated_obligations": len(self.generated_obligations),
            "saturation": {
                "iterations": self.saturation.iterations,
                "saturated": self.saturation.saturated,
                "eclasses": self.saturation.eclasses,
                "enodes": self.saturation.enodes,
            },
            "evaluated_candidates": self.evaluated_candidates,
            "optimal_for_searched_space": self.optimal_for_searched_space,
            "method": self.method,
            "scope": self.certificate.scope,
        }


def _obligation_dict(item: ProofObligation) -> Dict[str, object]:
    return {
        "rule": item.rule,
        "family": item.family,
        "before": item.before,
        "after": item.after,
        "semantics": item.semantics,
        "side_conditions": list(item.side_conditions),
        "error_bound": item.error_bound,
        "proofs": [
            {"system": proof.system.value, "identifier": proof.identifier}
            for proof in item.proofs
        ],
    }


def _input_names(expr: Expr) -> set[str]:
    if expr.op == "input":
        return {str(dict(expr.attrs)["name"])}
    return set().union(*(_input_names(arg) for arg in expr.args)) if expr.args else set()


def _default_policy(op: str) -> str:
    return CORE_REGISTRY.get(op).contract.residual_policies[0].name


def tensor_input(spec: M4Input) -> Expr:
    return Expr("input", attrs=(("name", spec.name),), shape=spec.shape, dtype=spec.dtype)


def activation(
    value: Expr,
    kind: str,
    *,
    approximate: str = "none",
    residual_policy: Optional[str] = None,
) -> Expr:
    if kind not in ACTIVATION_NAMES or kind == "activation_chain":
        raise ValueError(f"Unsupported M4 activation {kind!r}")
    if kind != "gelu" and approximate != "none":
        raise ValueError(f"{kind} has no approximation parameter")
    attrs = (("approximate", approximate),) if kind == "gelu" else ()
    policy = residual_policy or _default_policy(kind)
    CORE_REGISTRY.get(kind).contract.policy(policy)
    return Expr(kind, (value,), attrs, policy, value.shape, value.dtype)


def activation_chain_expr(
    value: Expr,
    operations: Sequence[str],
    *,
    replay: str = "replay_streaming",
) -> Expr:
    program = activation_program({"operations": tuple(operations)})
    encoded = _encode_program(program)
    CORE_REGISTRY.get("activation_chain").contract.policy(replay)
    return Expr(
        "activation_chain",
        (value,),
        (("operations", encoded),),
        replay,
        value.shape,
        value.dtype,
    )


def matmul(left: Expr, right: Expr) -> Expr:
    if len(left.shape) != 2 or len(right.shape) != 2:
        raise ValueError("M4 contraction rewrites currently require rank-two matrices")
    if left.shape[1] != right.shape[0] or left.dtype != right.dtype:
        raise ValueError("Incompatible M4 matmul signatures")
    return Expr(
        "matmul",
        (left, right),
        policy=_default_policy("matmul"),
        shape=(left.shape[0], right.shape[1]),
        dtype=left.dtype,
    )


def transpose(value: Expr) -> Expr:
    if len(value.shape) != 2:
        raise ValueError("M4 structured-adjoint rewrites currently require rank-two matrices")
    return Expr(
        "transpose",
        (value,),
        policy=_default_policy("transpose"),
        shape=(value.shape[1], value.shape[0]),
        dtype=value.dtype,
    )


def add(left: Expr, right: Expr) -> Expr:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("M4 add requires identical tensor signatures")
    return Expr(
        "add",
        (left, right),
        policy=_default_policy("add"),
        shape=left.shape,
        dtype=left.dtype,
    )


def _encode_program(program: Sequence[Tuple[str, str]]) -> Tuple[str, ...]:
    return tuple(
        kind if kind != "gelu" or approximate == "none" else "gelu:tanh"
        for kind, approximate in program
    )


def _node_program(node: ENode) -> Optional[Tuple[Tuple[str, str], ...]]:
    if node.op == "activation_chain":
        return activation_program(dict(node.attrs))
    if node.op in ACTIVATION_NAMES:
        approximate = dict(node.attrs).get("approximate", "none")
        return ((node.op, str(approximate)),)
    return None


def _signature(
    graph: EGraph,
    op: str,
    args: Tuple[int, ...],
    attrs: Tuple[Tuple[str, object], ...],
) -> Tuple[Tuple[int, ...], torch.dtype]:
    specs = [graph.spec(arg) for arg in args]
    if op in ACTIVATION_NAMES or op == "activation_chain":
        return specs[0]
    if op == "add":
        if specs[0] != specs[1]:
            raise ValueError("Invalid add signature during rewrite")
        return specs[0]
    if op == "matmul":
        (left_shape, left_dtype), (right_shape, right_dtype) = specs
        if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
            raise ValueError("Invalid matmul signature during rewrite")
        if left_dtype != right_dtype:
            raise ValueError("Invalid matmul dtype during rewrite")
        return (left_shape[0], right_shape[1]), left_dtype
    if op == "transpose":
        shape, dtype = specs[0]
        if len(shape) != 2:
            raise ValueError("Invalid transpose signature during rewrite")
        return (shape[1], shape[0]), dtype
    raise ValueError(f"Unknown M4 rewrite operator {op!r}")


def _nodes(graph: EGraph, class_id: int) -> Tuple[ENode, ...]:
    root = graph.find(class_id)
    return tuple(sorted(graph.classes[root].nodes, key=lambda item: repr(item.key())))


def _make_node(
    graph: EGraph,
    op: str,
    args: Tuple[int, ...],
    *,
    attrs: Tuple[Tuple[str, object], ...] = (),
    policy: Optional[str] = None,
) -> ENode:
    shape, dtype = _signature(graph, op, args, attrs)
    return ENode(op, args, attrs, policy or _default_policy(op), shape, dtype)


_CHAIN_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.m4_activation_chain_split_fuse"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.M4Equality.m4_activation_chain_split_fuse"),
    ProofRef(ProofSystem.Z3, "m4_activation_chain_split_fuse"),
)
_REPLAY_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.m25_replay_tail_dagger"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.M25Activation.m25_replay_tail_pullback"),
    ProofRef(ProofSystem.Z3, "m25_replay_cotangent_chain"),
)
_CONTRACTION_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.linear_chain_forward"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.Matrix.mm_associative"),
    ProofRef(ProofSystem.Z3, "m4_matmul_reassociation"),
)
_TRANSPOSE_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.dagger_composition"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.Matrix.mm_transpose"),
    ProofRef(ProofSystem.Z3, "m4_transpose_matmul"),
)
_INVOLUTION_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.dagger_involution"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.M4Equality.m4_transpose_involution"),
    ProofRef(ProofSystem.Z3, "m4_transpose_involution"),
)
_RELU_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.relu_mask_sufficient"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.Nonlinear.relu_one_bit_sufficient"),
    ProofRef(ProofSystem.Z3, "packed_sign_roundtrip_and_pullback"),
)


def _fuse_activations(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    outer = _node_program(node)
    if outer is None:
        return ()
    rewrites = []
    for inner_node in _nodes(graph, node.args[0]):
        inner = _node_program(inner_node)
        if inner is None:
            continue
        combined = _encode_program(inner + outer)
        replacement = _make_node(
            graph,
            "activation_chain",
            (inner_node.args[0],),
            attrs=(("operations", combined),),
            policy="replay_streaming",
        )
        rewrites.append(
            Rewrite(
                replacement,
                f"{inner_node.op} then {node.op}",
                f"activation_chain[{len(combined)}]",
                ("pure elementwise composition", "intermediate e-classes remain available"),
            )
        )
    return rewrites


def _split_activations(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "activation_chain":
        return ()
    program = _node_program(node)
    assert program is not None
    rewrites = []
    for split in range(1, len(program)):
        prefix = _make_node(
            graph,
            "activation_chain",
            (node.args[0],),
            attrs=(("operations", _encode_program(program[:split])),),
            policy=node.policy,
        )
        prefix_class = graph.intern(prefix)
        suffix = _make_node(
            graph,
            "activation_chain",
            (prefix_class,),
            attrs=(("operations", _encode_program(program[split:])),),
            policy=node.policy,
        )
        rewrites.append(
            Rewrite(
                suffix,
                f"activation_chain[{len(program)}]",
                f"activation_chain[{split}] then activation_chain[{len(program)-split}]",
                ("program append is function composition",),
            )
        )
    return rewrites


def _replay_alternative(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "activation_chain":
        return ()
    policy = "replay_cached" if node.policy == "replay_streaming" else "replay_streaming"
    replacement = _make_node(
        graph,
        "activation_chain",
        node.args,
        attrs=node.attrs,
        policy=policy,
    )
    return (
        Rewrite(
            replacement,
            node.policy,
            policy,
            ("replay reconstructs the original tail residuals",),
        ),
    )


def _relu_reconstruction(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "relu":
        return ()
    policy = "packed_sign" if node.policy == "resident_input" else "resident_input"
    return (
        Rewrite(
            _make_node(graph, "relu", node.args, attrs=node.attrs, policy=policy),
            node.policy,
            policy,
            ("the packed sign/NaN mask is sufficient for the ReLU pullback",),
        ),
    )


def _reassociate_lr(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "matmul":
        return ()
    rewrites = []
    for inner in _nodes(graph, node.args[0]):
        if inner.op != "matmul":
            continue
        a, b = inner.args
        c = node.args[1]
        right = _make_node(graph, "matmul", (b, c))
        right_class = graph.intern(right)
        replacement = _make_node(graph, "matmul", (a, right_class))
        rewrites.append(
            Rewrite(replacement, "(A B) C", "A (B C)", ("rank-two exact-real matrices",))
        )
    return rewrites


def _reassociate_rl(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "matmul":
        return ()
    rewrites = []
    for inner in _nodes(graph, node.args[1]):
        if inner.op != "matmul":
            continue
        a = node.args[0]
        b, c = inner.args
        left = _make_node(graph, "matmul", (a, b))
        left_class = graph.intern(left)
        replacement = _make_node(graph, "matmul", (left_class, c))
        rewrites.append(
            Rewrite(replacement, "A (B C)", "(A B) C", ("rank-two exact-real matrices",))
        )
    return rewrites


def _transpose_product(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "transpose":
        return ()
    rewrites = []
    for inner in _nodes(graph, node.args[0]):
        if inner.op != "matmul":
            continue
        a, b = inner.args
        bt = graph.intern(_make_node(graph, "transpose", (b,)))
        at = graph.intern(_make_node(graph, "transpose", (a,)))
        replacement = _make_node(graph, "matmul", (bt, at))
        rewrites.append(
            Rewrite(replacement, "(A B)ᵀ", "Bᵀ Aᵀ", ("rank-two exact-real matrices",))
        )
    return rewrites


def _transpose_involution(graph: EGraph, class_id: int, node: ENode) -> Iterable[Rewrite]:
    if node.op != "transpose":
        return ()
    target = graph.find(node.args[0])
    rewrites = []
    for inner in _nodes(graph, target):
        if inner.op != "transpose":
            continue
        original = graph.find(inner.args[0])
        for original_node in _nodes(graph, original):
            rewrites.append(Rewrite(original_node, "Aᵀᵀ", "A", ("transpose is an involution",)))
            break
    return rewrites


def default_rewrite_rules() -> Tuple[RewriteRule, ...]:
    return (
        RewriteRule("activation_fuse", "streaming", "core_reproducible", _CHAIN_PROOFS, _fuse_activations),
        RewriteRule("activation_split", "streaming", "core_reproducible", _CHAIN_PROOFS, _split_activations),
        RewriteRule("replay_policy", "streaming", "core_reproducible", _REPLAY_PROOFS, _replay_alternative),
        RewriteRule("relu_residual_reconstruction", "reconstruction", "core_reproducible", _RELU_PROOFS, _relu_reconstruction),
        RewriteRule("matmul_reassociate_lr", "contraction", "exact_real", _CONTRACTION_PROOFS, _reassociate_lr),
        RewriteRule("matmul_reassociate_rl", "contraction", "exact_real", _CONTRACTION_PROOFS, _reassociate_rl),
        RewriteRule("structured_transpose_product", "structured_adjoint", "exact_real", _TRANSPOSE_PROOFS, _transpose_product),
        RewriteRule("structured_transpose_involution", "structured_adjoint", "core_reproducible", _INVOLUTION_PROOFS, _transpose_involution),
    )


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _nbytes(signature: Tuple[Tuple[int, ...], torch.dtype]) -> int:
    shape, dtype = signature
    return math.prod(shape) * _element_size(dtype)


def _local_cost(graph: EGraph, node: ENode) -> Tuple[int, int, int]:
    if node.op == "input":
        return 0, 0, 0
    output = _nbytes((node.shape, node.dtype))
    inputs = [_nbytes(graph.spec(arg)) for arg in node.args]
    if node.op == "activation_chain":
        program = activation_program(dict(node.attrs))
        layers = len(program)
        replay = layers if node.policy == "replay_cached" else layers * (layers + 1) // 2
        return 2 * layers + replay, (5 * layers + 2 * replay) * output, output
    if node.op == "matmul":
        traffic = 3 * sum(inputs) + 2 * output
        return 2, traffic, sum(inputs)
    if node.op == "add":
        return 2, 4 * sum(inputs) + 2 * output, 0
    if node.op == "transpose":
        return 2, 2 * sum(inputs) + 2 * output, 0
    if node.op in ACTIVATION_NAMES:
        residual = (math.prod(node.shape) + 7) // 8 if node.policy == "packed_sign" else output
        if node.op == "neg":
            residual = 0
        return 2, 3 * sum(inputs) + 2 * output, residual
    raise ValueError(f"No M4 cost model for {node.op!r}")


def _lower(
    problem: M4Problem,
    candidate: ExtractedCandidate,
    exact_node_limit: int,
) -> Tuple[JointGraph, MemoryCertificate]:
    builder = JointGraphBuilder()
    inputs = {
        spec.name: builder.input(
            spec.name,
            spec.shape,
            dtype=spec.dtype,
            requires_grad=spec.requires_grad,
        )
        for spec in problem.inputs
    }
    memo: Dict[Expr, int] = {}

    def emit(expr: Expr) -> int:
        if expr in memo:
            return memo[expr]
        if expr.op == "input":
            value_id = inputs[str(dict(expr.attrs)["name"])]
        else:
            value_id = builder.forward(
                expr.op,
                tuple(emit(arg) for arg in expr.args),
                residual_policy=expr.policy,
                attrs=dict(expr.attrs),
            )
        memo[expr] = value_id
        return value_id

    output = emit(candidate.expr)
    builder.set_objective(output)
    for name in problem.gradient_inputs:
        builder.request_gradient(inputs[name])
    graph = builder.generate_reverse()
    return graph, plan_memory(graph, exact_node_limit=exact_node_limit)


def plan_m4(
    problem: M4Problem,
    *,
    config: M4Config = M4Config(),
    rules: Optional[Sequence[RewriteRule]] = None,
) -> M4Plan:
    """Saturate, extract, and globally certify original and discovered candidates."""

    if config.budget is not None and config.budget < 0:
        raise ValueError("M4 memory budget must be nonnegative")
    if config.max_candidates <= 0:
        raise ValueError("M4 must evaluate at least one candidate")
    egraph = EGraph(
        allow_exact_real=config.allow_exact_real,
        objective=config.objective,
    )
    root = egraph.add_expr(problem.root)
    baseline = egraph.extract(
        root,
        local_cost=lambda node: _local_cost(egraph, node),
        beam_width=1,
        max_expansions_per_node=config.max_expansions_per_node,
    )[0][0]
    for rule in default_rewrite_rules() if rules is None else rules:
        egraph.register_rule(rule)
    report = egraph.saturate(max_iterations=config.max_iterations)
    candidates, truncated = egraph.extract(
        root,
        local_cost=lambda node: _local_cost(egraph, node),
        beam_width=config.beam_width,
        max_expansions_per_node=config.max_expansions_per_node,
    )
    unique: Dict[Expr, ExtractedCandidate] = {baseline.expr: baseline}
    for candidate in candidates:
        previous = unique.get(candidate.expr)
        if previous is None or len(candidate.obligations) < len(previous.obligations):
            unique[candidate.expr] = candidate
    ordered = sorted(unique.values(), key=lambda item: item.additive_key)
    if len(ordered) > config.max_candidates:
        ordered = ordered[: config.max_candidates]
        truncated = True
    if baseline not in ordered:
        ordered.append(baseline)

    evaluated = []
    for candidate in ordered:
        graph, certificate = _lower(problem, candidate, config.exact_node_limit)
        evaluated.append((candidate, graph, certificate))

    def cost(item) -> LogicalCost:
        return LogicalCost.from_certificate(item[2])

    baseline_item = next(item for item in evaluated if item[0] is baseline)
    baseline_cost = cost(baseline_item)
    baseline_key = baseline_cost.key(config.objective)
    best_key = min(cost(item).key(config.objective) for item in evaluated)
    if best_key < baseline_key:
        selected, graph, certificate = min(
            (item for item in evaluated if cost(item).key(config.objective) == best_key),
            key=lambda item: item[0].expr.structural(),
        )
    else:
        # Discovery never replaces the user's program without a strict
        # improvement under the requested backward-inclusive objective.
        selected, graph, certificate = baseline_item
    baseline_graph, baseline_certificate = baseline_item[1], baseline_item[2]
    if config.budget is not None and certificate.peak_bytes > config.budget:
        raise MemoryBudgetError(
            certificate.peak_bytes,
            config.budget,
            report.saturated and not truncated,
        )
    optimal = (
        report.saturated
        and not truncated
        and len(candidates) <= config.max_candidates
        and all(item[2].optimal_for_managed_model for item in evaluated)
    )
    return M4Plan(
        graph=graph,
        certificate=certificate,
        baseline_graph=baseline_graph,
        baseline_certificate=baseline_certificate,
        selected_expression=selected.expr,
        baseline_expression=baseline.expr,
        proof_obligations=selected.obligations,
        generated_obligations=tuple(egraph.generated_obligations),
        saturation=report,
        evaluated_candidates=len(evaluated),
        optimal_for_searched_space=optimal,
        method="equality_saturation_bounded_exact" if optimal else "equality_saturation_beam",
        objective=config.objective,
        selected_cost=LogicalCost.from_certificate(certificate),
        baseline_cost=baseline_cost,
    )


__all__ = [
    "M4Config",
    "M4Input",
    "M4Plan",
    "M4Problem",
    "activation",
    "activation_chain_expr",
    "add",
    "default_rewrite_rules",
    "matmul",
    "plan_m4",
    "tensor_input",
    "transpose",
]
