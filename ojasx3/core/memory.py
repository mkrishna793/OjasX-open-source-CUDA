"""Alias-aware liveness and bounded-optimal joint-graph scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .activation_math import activation_program
from .contracts import CORE_REGISTRY
from .ir import JointGraph, Node


@dataclass(frozen=True)
class MemoryStep:
    node_id: int
    node_name: str
    before_bytes: int
    new_storage_bytes: int
    workspace_bytes: int
    during_bytes: int
    after_bytes: int
    live_storage_after: Tuple[str, ...]
    work_units: int
    traffic_bytes: int
    replay_evaluations: int


@dataclass(frozen=True)
class MemoryCertificate:
    schedule: Tuple[int, ...]
    steps: Tuple[MemoryStep, ...]
    peak_bytes: int
    final_bytes: int
    work_units: int
    traffic_bytes: int
    replay_evaluations: int
    optimal_for_managed_model: bool
    method: str
    graph_digest: str
    scope: str = (
        "logical live storage plus declared explicit operator workspace; "
        "work counts operator/local derivative invocations and traffic counts logical value reads/writes; "
        "excludes allocator retention, library-internal workspace, Python, hardware traffic, and runtime overhead"
    )

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["schedule"] = list(self.schedule)
        return result

    def validate(self, graph: JointGraph) -> None:
        """Recheck schedule and liveness fields against the current graph.

        This authenticates internal consistency, not an untrusted cryptographic
        provenance chain. Execution does not rerun the exponential optimality DP.
        """

        if self.graph_digest != graph_digest(graph):
            raise ValueError("Memory certificate belongs to a different graph")
        expected = _certificate_from_schedule(
            graph,
            self.schedule,
            self.optimal_for_managed_model,
            self.method,
        )
        if (
            self.schedule != expected.schedule
            or self.steps != expected.steps
            or self.peak_bytes != expected.peak_bytes
            or self.final_bytes != expected.final_bytes
            or self.work_units != expected.work_units
            or self.traffic_bytes != expected.traffic_bytes
            or self.replay_evaluations != expected.replay_evaluations
        ):
            raise ValueError("Memory certificate schedule or liveness fields are inconsistent")


class MemoryBudgetError(RuntimeError):
    def __init__(self, required: int, budget: int, optimal: bool) -> None:
        self.required = required
        self.budget = budget
        self.optimal = optimal
        qualifier = "minimum admissible" if optimal else "selected"
        super().__init__(
            f"The {qualifier} managed-memory schedule requires {required} bytes, "
            f"above the {budget}-byte budget"
        )


def graph_digest(graph: JointGraph) -> str:
    payload = {
        "values": [
            (
                value.id,
                value.name,
                value.shape,
                str(value.dtype),
                value.storage_id,
                value.storage_bytes,
                value.producer,
                value.requires_grad,
                value.kind,
                value.role,
                value.encoding,
            )
            for value in graph.values.values()
        ],
        "nodes": [
            (
                node.id,
                node.name,
                node.op_name,
                node.kind,
                node.inputs,
                node.outputs,
                node.workspace_bytes,
                node.forward_inputs,
                node.residual_ids,
                node.residual_slots,
                node.residual_policy,
                node.demand,
                node.gradient_input_indices,
                node.attrs,
            )
            for node in graph.nodes
        ],
        "inputs": graph.inputs,
        "seeds": graph.seeds,
        "outputs": graph.outputs,
        "objective": graph.objective,
        "gradient_targets": sorted(graph.gradient_targets.items()),
        "execution_mode": graph.execution_mode,
        "derivative_order": graph.derivative_order,
        "variable_inputs": graph.variable_inputs,
        "structured_cotangents": graph.structured_cotangents,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class _Liveness:
    def __init__(self, graph: JointGraph) -> None:
        self.graph = graph
        self.nodes = tuple(graph.nodes)
        self.count = len(self.nodes)
        self.index = {node.id: index for index, node in enumerate(self.nodes)}
        consumers: Dict[int, List[int]] = {value_id: [] for value_id in graph.values}
        for node in self.nodes:
            for value_id in node.inputs:
                consumers[value_id].append(node.id)
        self.consumers: Dict[int, Tuple[int, ...]] = {
            value_id: tuple(node_ids) for value_id, node_ids in consumers.items()
        }
        self.total_uses = {
            value_id: len(self.consumers[value_id]) + int(value_id in graph.outputs)
            for value_id in graph.values
        }
        self.storage_bytes = {
            value.storage_id: value.storage_bytes for value in graph.values.values()
        }
        self._live_cache: Dict[int, Tuple[frozenset[str], int]] = {0: self._compute_live(0)}

    def _compute_live(self, completed_mask: int) -> Tuple[frozenset[str], int]:
        completed_nodes = {
            self.nodes[index].id for index in range(self.count) if completed_mask & (1 << index)
        }
        live_storage: Set[str] = set()
        for value_id, value in self.graph.values.items():
            consumed = sum(node_id in completed_nodes for node_id in self.consumers[value_id])
            remaining = self.total_uses[value_id] - consumed
            produced = value.producer is None or value.producer in completed_nodes
            if produced and remaining > 0:
                live_storage.add(value.storage_id)
        frozen = frozenset(live_storage)
        return frozen, sum(self.storage_bytes[storage] for storage in frozen)

    def live(self, completed_mask: int) -> Tuple[frozenset[str], int]:
        if completed_mask not in self._live_cache:
            self._live_cache[completed_mask] = self._compute_live(completed_mask)
        return self._live_cache[completed_mask]

    def ready(self, completed_mask: int) -> Iterable[int]:
        for index, node in enumerate(self.nodes):
            if completed_mask & (1 << index):
                continue
            admissible = True
            for value_id in node.inputs:
                producer = self.graph.values[value_id].producer
                if producer is not None:
                    producer_index = self.index[producer]
                    if not completed_mask & (1 << producer_index):
                        admissible = False
                        break
            if admissible:
                yield index

    def transition(self, completed_mask: int, node_index: int) -> Tuple[int, int, int, int]:
        node = self.nodes[node_index]
        live_before, before_bytes = self.live(completed_mask)
        output_storage = {self.graph.values[value_id].storage_id for value_id in node.outputs}
        new_storage = output_storage - live_before
        new_bytes = sum(self.storage_bytes[storage] for storage in new_storage)
        during = before_bytes + new_bytes + node.workspace_bytes
        next_mask = completed_mask | (1 << node_index)
        _, after_bytes = self.live(next_mask)
        return next_mask, new_bytes, during, after_bytes


def _node_accounting(graph: JointGraph, node: Node) -> Tuple[int, int, int]:
    """Return backward-inclusive work, logical value traffic, and replay work."""

    input_values = {graph.values[value_id] for value_id in node.inputs}
    output_values = {graph.values[value_id] for value_id in node.outputs}
    traffic = sum(value.nbytes for value in input_values | output_values)
    work = 1
    replay = 0
    if node.op_name != "activation_chain" or node.kind not in {"primal", "pullback"}:
        return work, traffic, replay

    program = activation_program(dict(node.attrs))
    layers = len(program)
    source = graph.values[node.forward_inputs[0]]
    nbytes = source.nbytes
    if node.kind == "primal":
        # Initial forward evaluation of every layer, including intermediate
        # value reads/writes hidden by the fused operator boundary.
        work = layers
        traffic += 2 * (layers - 1) * nbytes
    else:
        # Every local derivative plus the selected prefix/cached replay.
        replay = CORE_REGISTRY.get("activation_chain").replay_evaluations(node)
        work = layers + replay
        traffic += 3 * (layers - 1) * nbytes + 2 * replay * nbytes
    return work, traffic, replay


def _certificate_from_schedule(graph: JointGraph, schedule: Sequence[int], optimal: bool, method: str) -> MemoryCertificate:
    liveness = _Liveness(graph)
    if sorted(schedule) != sorted(node.id for node in graph.nodes):
        raise ValueError("Schedule must contain every joint-IR node exactly once")
    mask = 0
    peak = liveness.live(0)[1]
    steps: List[MemoryStep] = []
    available = set(graph.inputs) | set(graph.seeds)
    for node_id in schedule:
        index = liveness.index[node_id]
        node = graph.nodes[index]
        if any(value_id not in available for value_id in node.inputs):
            raise ValueError(f"Schedule is not topological at {node.name}")
        available.update(node.outputs)
        before_storage, before_bytes = liveness.live(mask)
        next_mask, new_bytes, during, after_bytes = liveness.transition(mask, index)
        mask = next_mask
        _, after_bytes = liveness.live(mask)
        live_after, _ = liveness.live(mask)
        work_units, traffic_bytes, replay_evaluations = _node_accounting(graph, node)
        steps.append(
            MemoryStep(
                node_id=node.id,
                node_name=node.name,
                before_bytes=before_bytes,
                new_storage_bytes=new_bytes,
                workspace_bytes=node.workspace_bytes,
                during_bytes=during,
                after_bytes=after_bytes,
                live_storage_after=tuple(sorted(live_after)),
                work_units=work_units,
                traffic_bytes=traffic_bytes,
                replay_evaluations=replay_evaluations,
            )
        )
        peak = max(peak, during)
        del before_storage
    final_bytes = liveness.live(mask)[1]
    peak = max(peak, final_bytes)
    return MemoryCertificate(
        schedule=tuple(schedule),
        steps=tuple(steps),
        peak_bytes=peak,
        final_bytes=final_bytes,
        work_units=sum(step.work_units for step in steps),
        traffic_bytes=sum(step.traffic_bytes for step in steps),
        replay_evaluations=sum(step.replay_evaluations for step in steps),
        optimal_for_managed_model=optimal,
        method=method,
        graph_digest=graph_digest(graph),
    )


def plan_memory(
    graph: JointGraph,
    *,
    budget: Optional[int] = None,
    exact_node_limit: int = 12,
) -> MemoryCertificate:
    """Minimize peak logical live storage over legal topological schedules.

    For at most ``exact_node_limit`` nodes, subset dynamic programming is exact:
    completed-node subsets determine both readiness and current live storage.
    Larger graphs use a deterministic greedy topological scheduler and are not
    labeled optimal.
    """

    graph.validate()
    if budget is not None and budget < 0:
        raise ValueError("Memory budget must be non-negative")
    if exact_node_limit < 0:
        raise ValueError("exact_node_limit must be non-negative")
    liveness = _Liveness(graph)
    count = liveness.count
    if count == 0:
        raise ValueError("Cannot schedule an empty joint graph")

    if count <= exact_node_limit:
        size = 1 << count
        costs: List[Optional[int]] = [None] * size
        parents: List[Optional[Tuple[int, int]]] = [None] * size
        costs[0] = liveness.live(0)[1]
        for mask in range(size):
            current_cost = costs[mask]
            if current_cost is None:
                continue
            for index in liveness.ready(mask):
                next_mask, _, during, after = liveness.transition(mask, index)
                candidate = max(current_cost, during, after)
                known_cost = costs[next_mask]
                if known_cost is None or candidate < known_cost:
                    costs[next_mask] = candidate
                    parents[next_mask] = (mask, index)
        full = size - 1
        if parents[full] is None and count > 1:
            raise ValueError("Joint graph has no legal topological schedule")
        reversed_schedule = []
        cursor = full
        while cursor:
            parent = parents[cursor]
            if parent is None:
                raise ValueError("Broken schedule reconstruction")
            previous, index = parent
            reversed_schedule.append(liveness.nodes[index].id)
            cursor = previous
        schedule = tuple(reversed(reversed_schedule))
        certificate = _certificate_from_schedule(
            graph,
            schedule,
            True,
            f"subset_dp_exact_n_{count}",
        )
    else:
        mask = 0
        current_peak = liveness.live(0)[1]
        schedule_list = []
        while mask != (1 << count) - 1:
            candidates = []
            for index in liveness.ready(mask):
                next_mask, _, during, after = liveness.transition(mask, index)
                candidates.append((max(current_peak, during, after), after, liveness.nodes[index].id, index, next_mask))
            if not candidates:
                raise ValueError("Joint graph has no legal topological schedule")
            _, _, _, selected, next_mask = min(candidates)
            schedule_list.append(liveness.nodes[selected].id)
            current_peak = max(current_peak, liveness.transition(mask, selected)[2])
            mask = next_mask
        certificate = _certificate_from_schedule(
            graph,
            schedule_list,
            False,
            f"deterministic_greedy_n_{count}",
        )

    if budget is not None and certificate.peak_bytes > budget:
        raise MemoryBudgetError(certificate.peak_bytes, budget, certificate.optimal_for_managed_model)
    return certificate


def last_use_by_step(graph: JointGraph, schedule: Sequence[int]) -> Dict[int, int]:
    last: Dict[int, int] = {}
    for step, node_id in enumerate(schedule):
        node = graph.node(node_id)
        for value_id in node.inputs:
            last[value_id] = step
    terminal = len(schedule)
    for value_id in graph.outputs:
        last[value_id] = terminal
    return last


__all__ = [
    "MemoryBudgetError",
    "MemoryCertificate",
    "MemoryStep",
    "graph_digest",
    "last_use_by_step",
    "plan_memory",
]
