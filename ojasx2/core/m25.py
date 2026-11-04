"""M2.5 replay fusion and joint residual-policy/memory optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .activation_math import ACTIVATION_NAMES
from .contracts import CORE_REGISTRY
from .ir import JointGraph, JointGraphBuilder, Node
from .memory import MemoryBudgetError, MemoryCertificate, graph_digest, plan_memory


@dataclass(frozen=True)
class FusionGroup:
    original_nodes: Tuple[int, ...]
    original_output: int
    fused_output: int
    operations: Tuple[str, ...]


@dataclass(frozen=True)
class FusionResult:
    builder: JointGraphBuilder
    groups: Tuple[FusionGroup, ...]


@dataclass
class M25Plan:
    graph: JointGraph
    certificate: MemoryCertificate
    baseline_graph: JointGraph
    baseline_certificate: MemoryCertificate
    unfused_optimized_peak: int
    fusion_groups: Tuple[FusionGroup, ...]
    policy_choices: Tuple[Tuple[int, str], ...]
    replay_evaluations: int
    evaluated_candidates: int
    optimal_for_considered_graphs: bool
    method: str

    @property
    def saved_bytes(self) -> int:
        return self.baseline_certificate.peak_bytes - self.certificate.peak_bytes

    @property
    def fusion_selected(self) -> bool:
        return bool(self.fusion_groups) and any(
            node.op_name == "activation_chain" for node in self.graph.nodes
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "graph_digest": graph_digest(self.graph),
            "baseline_graph_digest": graph_digest(self.baseline_graph),
            "baseline_peak_bytes": self.baseline_certificate.peak_bytes,
            "unfused_optimized_peak_bytes": self.unfused_optimized_peak,
            "peak_bytes": self.certificate.peak_bytes,
            "saved_bytes": self.saved_bytes,
            "fusion_selected": self.fusion_selected,
            "fusion_groups": [
                {
                    "original_nodes": list(group.original_nodes),
                    "operations": list(group.operations),
                }
                for group in self.fusion_groups
            ],
            "policy_choices": [[node, policy] for node, policy in self.policy_choices],
            "replay_evaluations": self.replay_evaluations,
            "evaluated_candidates": self.evaluated_candidates,
            "optimal_for_considered_graphs": self.optimal_for_considered_graphs,
            "method": self.method,
            "scope": self.certificate.scope,
        }


def _operation_name(node: Node) -> str:
    if node.op_name == "gelu" and dict(node.attrs).get("approximate", "none") == "tanh":
        return "gelu:tanh"
    return node.op_name


def fuse_activation_chains(builder: JointGraphBuilder) -> FusionResult:
    """Rewrite maximal safe elementwise chains without hiding requested values."""

    if builder._reverse_generated:
        raise ValueError("Activation fusion requires an ungenerated primal builder")
    graph = builder.graph
    if graph.objective is None:
        raise ValueError("Set an objective before M2.5 activation fusion")
    objective_node = graph.node(graph.values[graph.objective].producer)
    if objective_node.op_name != "sum" or len(objective_node.inputs) != 1:
        raise ValueError("M2.5 fusion requires the standard scalar objective sum")

    consumers: Dict[int, List[int]] = {}
    for node in graph.nodes:
        for value_id in node.inputs:
            consumers.setdefault(value_id, []).append(node.id)

    def eligible(node: Node) -> bool:
        return node.kind == "primal" and node.op_name in ACTIVATION_NAMES

    chains: List[Tuple[Node, ...]] = []
    claimed = set()
    for node in graph.nodes:
        if node.id in claimed or not eligible(node):
            continue
        members = [node]
        current = node
        while current.outputs[0] not in builder._gradient_requests:
            next_nodes = consumers.get(current.outputs[0], [])
            if len(next_nodes) != 1:
                break
            following = graph.node(next_nodes[0])
            if not eligible(following) or following.outputs[0] in builder._gradient_requests:
                break
            members.append(following)
            current = following
        if len(members) >= 2:
            chains.append(tuple(members))
            claimed.update(member.id for member in members)

    chain_starts = {chain[0].id: chain for chain in chains}
    fused = JointGraphBuilder()
    value_map: Dict[int, int] = {}
    storage_roots: Dict[str, int] = {}
    for old_id in graph.inputs:
        value = graph.values[old_id]
        alias_of = storage_roots.get(value.storage_id)
        new_id = fused.input(
            value.name,
            value.shape,
            dtype=value.dtype,
            requires_grad=value.requires_grad,
            alias_of=alias_of,
        )
        storage_roots.setdefault(value.storage_id, new_id)
        value_map[old_id] = new_id

    groups: List[FusionGroup] = []
    for node in graph.nodes:
        if node.id == objective_node.id:
            continue
        if node.id in claimed and node.id not in chain_starts:
            continue
        if node.id in chain_starts:
            chain = chain_starts[node.id]
            terminal = chain[-1]
            output = fused.forward(
                "activation_chain",
                (value_map[node.inputs[0]],),
                name=terminal.name,
                residual_policy="replay_streaming",
                attrs={"operations": tuple(_operation_name(member) for member in chain)},
            )
            value_map[terminal.outputs[0]] = output
            groups.append(
                FusionGroup(
                    original_nodes=tuple(member.id for member in chain),
                    original_output=terminal.outputs[0],
                    fused_output=output,
                    operations=tuple(_operation_name(member) for member in chain),
                )
            )
            continue
        output = fused.forward(
            node.op_name,
            tuple(value_map[value_id] for value_id in node.inputs),
            name=node.name,
            residual_policy=node.residual_policy,
            attrs=dict(node.attrs),
        )
        value_map[node.outputs[0]] = output

    fused.set_objective(value_map[objective_node.inputs[0]])
    for target in sorted(builder._gradient_requests):
        if target not in value_map:
            raise RuntimeError("A requested value was hidden by activation fusion")
        fused.request_gradient(value_map[target])
    return FusionResult(fused, tuple(groups))


def _policy_nodes(builder: JointGraphBuilder) -> Tuple[Node, ...]:
    return tuple(
        node
        for node in builder.graph.nodes
        if len(CORE_REGISTRY.get(node.op_name).contract.residual_policies) > 1
    )


def _replay_work(graph: JointGraph) -> int:
    work = 0
    for node in graph.nodes:
        if node.kind == "primal" and node.op_name == "activation_chain" and any(node.demand):
            work += CORE_REGISTRY.get("activation_chain").replay_evaluations(node)
    return work


def _candidate(
    builder: JointGraphBuilder,
    assignment: Tuple[Tuple[int, str], ...],
    exact_node_limit: int,
) -> Tuple[JointGraph, MemoryCertificate]:
    candidate = copy.deepcopy(builder)
    for node_id, policy in assignment:
        candidate.graph.node(node_id).residual_policy = policy
    graph = candidate.generate_reverse()
    return graph, plan_memory(graph, exact_node_limit=exact_node_limit)


def _optimize_policies(
    builder: JointGraphBuilder,
    *,
    exact_node_limit: int,
    exact_policy_node_limit: int,
    max_exact_candidates: int,
) -> Tuple[JointGraph, MemoryCertificate, Tuple[Tuple[int, str], ...], bool, int]:
    nodes = _policy_nodes(builder)
    defaults = tuple(
        (node.id, node.residual_policy)
        for node in nodes
    )
    policy_space = 1
    for node in nodes:
        policy_space *= len(CORE_REGISTRY.get(node.op_name).contract.residual_policies)
    exact = len(nodes) <= exact_policy_node_limit and policy_space <= max_exact_candidates
    schedule_limit = exact_node_limit if exact else 0
    evaluated = 0

    if exact:
        choices = [
            tuple(policy.name for policy in CORE_REGISTRY.get(node.op_name).contract.residual_policies)
            for node in nodes
        ]
        assignments = (
            tuple((node.id, policy) for node, policy in zip(nodes, policies))
            for policies in itertools.product(*choices)
        )
    else:
        assignments = (defaults,)

    best = None
    for assignment in assignments:
        graph, certificate = _candidate(builder, assignment, schedule_limit)
        evaluated += 1
        key = (certificate.peak_bytes, _replay_work(graph), certificate.final_bytes, assignment)
        if best is None or key < best[0]:
            best = (key, graph, certificate, assignment)

    if not exact:
        # Deterministic coordinate improvement over policy choices. Every
        # candidate uses greedy scheduling, so the result is not marked exact.
        current = best[3]
        for _pass in range(2):
            improved = False
            for node in nodes:
                local_best = best
                policies = CORE_REGISTRY.get(node.op_name).contract.residual_policies
                for policy in policies:
                    assignment = tuple(
                        (node_id, policy.name if node_id == node.id else selected)
                        for node_id, selected in current
                    )
                    graph, certificate = _candidate(builder, assignment, 0)
                    evaluated += 1
                    key = (certificate.peak_bytes, _replay_work(graph), certificate.final_bytes, assignment)
                    if key < local_best[0]:
                        local_best = (key, graph, certificate, assignment)
                if local_best[0] < best[0]:
                    best = local_best
                    current = best[3]
                    improved = True
            if not improved:
                break

    _, graph, certificate, assignment = best
    optimal = exact and certificate.optimal_for_managed_model
    return graph, certificate, assignment, optimal, evaluated


def plan_m25_memory(
    builder: JointGraphBuilder,
    *,
    fuse: bool = True,
    budget: Optional[int] = None,
    exact_node_limit: int = 12,
    exact_policy_node_limit: int = 4,
    max_exact_candidates: int = 64,
) -> M25Plan:
    """Optimize original/fused residual graphs without claiming unsearched rewrites."""

    if builder._reverse_generated:
        raise ValueError("M2.5 planning requires an ungenerated primal builder")
    if budget is not None and budget < 0:
        raise ValueError("Memory budget must be non-negative")
    baseline_graph, baseline_certificate = _candidate(builder, (), exact_node_limit)

    original_graph, original_certificate, original_choices, original_optimal, original_count = _optimize_policies(
        builder,
        exact_node_limit=exact_node_limit,
        exact_policy_node_limit=exact_policy_node_limit,
        max_exact_candidates=max_exact_candidates,
    )
    candidates = [(original_graph, original_certificate, original_choices, original_optimal, ())]
    evaluated = original_count

    if fuse:
        fusion = fuse_activation_chains(builder)
        if fusion.groups:
            fused_graph, fused_certificate, fused_choices, fused_optimal, fused_count = _optimize_policies(
                fusion.builder,
                exact_node_limit=exact_node_limit,
                exact_policy_node_limit=exact_policy_node_limit,
                max_exact_candidates=max_exact_candidates,
            )
            candidates.append((fused_graph, fused_certificate, fused_choices, fused_optimal, fusion.groups))
            evaluated += fused_count

    def key(candidate):
        graph, certificate, choices, _, groups = candidate
        return (
            certificate.peak_bytes,
            _replay_work(graph),
            certificate.final_bytes,
            bool(groups),
            choices,
        )

    selected = min(candidates, key=key)
    graph, certificate, choices, optimal, groups = selected
    if budget is not None and certificate.peak_bytes > budget:
        raise MemoryBudgetError(certificate.peak_bytes, budget, optimal)
    return M25Plan(
        graph=graph,
        certificate=certificate,
        baseline_graph=baseline_graph,
        baseline_certificate=baseline_certificate,
        unfused_optimized_peak=original_certificate.peak_bytes,
        fusion_groups=groups,
        policy_choices=choices,
        replay_evaluations=_replay_work(graph),
        evaluated_candidates=evaluated + 1,
        optimal_for_considered_graphs=all(candidate[3] for candidate in candidates),
        method="bounded_exact_original_and_fused" if all(candidate[3] for candidate in candidates) else "deterministic_greedy_policy_search",
    )


__all__ = ["FusionGroup", "FusionResult", "M25Plan", "fuse_activation_chains", "plan_m25_memory"]
