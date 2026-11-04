"""Abstract-analysis memory planning and scheduled execution for Taylor jets."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Tuple
import weakref

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

from .contracts import CORE_REGISTRY
from .ir import JointGraph, Node
from .jets import Jet, seed_jet
from .memory import MemoryCertificate, last_use_by_step, plan_memory


class _LiveTensorLedger(TorchDispatchMode):
    """Track explicit dispatched tensors without retaining them."""

    def __init__(self) -> None:
        self._tensors: Dict[int, weakref.ReferenceType] = {}
        self.peak = 0

    @staticmethod
    def _key(tensor: torch.Tensor) -> Tuple[str, int]:
        return str(tensor.device), tensor.untyped_storage()._cdata

    @classmethod
    def groups_for(cls, tensors: Sequence[torch.Tensor]) -> Dict[Tuple[str, int], int]:
        groups: Dict[Tuple[str, int], int] = {}
        for tensor in tensors:
            key = cls._key(tensor)
            groups[key] = max(groups.get(key, 0), tensor.numel() * tensor.element_size())
        return groups

    def groups(self) -> Dict[Tuple[str, int], int]:
        tensors = [reference() for reference in list(self._tensors.values())]
        return self.groups_for([tensor for tensor in tensors if tensor is not None])

    def current(self) -> int:
        return sum(self.groups().values())

    def _record(self, result) -> None:
        for tensor in tree_flatten(result)[0]:
            if isinstance(tensor, torch.Tensor) and id(tensor) not in self._tensors:
                identity = id(tensor)
                self._tensors[identity] = weakref.ref(
                    tensor,
                    lambda _reference, identity=identity: self._tensors.pop(identity, None),
                )
        self.peak = max(self.peak, self.current())

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        self._record(result)
        return result


def _jet_coefficients(jet: Jet) -> Tuple[torch.Tensor, ...]:
    return tuple(jet.coefficients.values())


def _execute_primal_jet(node: Node, environment: Mapping[int, Jet]) -> Jet:
    operator = CORE_REGISTRY.get(node.op_name)
    return operator.jet_forward([environment[value_id] for value_id in node.inputs], dict(node.attrs))


def _validate_primal_graph(graph: JointGraph) -> None:
    graph.validate()
    if graph.execution_mode != "first_order" or graph.seeds or graph.gradient_targets:
        raise ValueError("Jet planning requires an ungenerated primal graph")
    if any(node.kind != "primal" or node.residual_ids for node in graph.nodes):
        raise ValueError("Jet planning requires a residual-free primal graph")


def plan_jet_memory(
    graph: JointGraph,
    *,
    variable_inputs: Optional[Sequence[int]] = None,
    order: int = 4,
    budget: Optional[int] = None,
    exact_node_limit: int = 12,
) -> Tuple[JointGraph, MemoryCertificate]:
    """Analyze sparse jets on meta tensors, then optimize their live storage.

    The analysis uses the same operator jet implementations as execution. Its
    dispatch ledger observes actual explicit coefficient/view/temporary storage;
    no dense-coefficient multiplier or guessed per-operator workspace is used.
    """

    _validate_primal_graph(graph)
    if isinstance(order, bool) or not 1 <= order <= 4:
        raise ValueError("Jet order must be one through four")
    if variable_inputs is None:
        variables = tuple(value_id for value_id in graph.inputs if graph.values[value_id].requires_grad)
    else:
        variables = tuple(variable_inputs)
    if not variables or len(set(variables)) != len(variables) or any(value_id not in graph.inputs for value_id in variables):
        raise ValueError("Jet variable inputs must be unique external inputs")
    specialized = copy.deepcopy(graph)
    specialized.execution_mode = "jet"
    specialized.derivative_order = order
    specialized.variable_inputs = variables
    variable_index = {value_id: index for index, value_id in enumerate(variables)}

    schedule = tuple(node.id for node in specialized.nodes)
    last_use = last_use_by_step(specialized, schedule)
    with torch.no_grad(), _LiveTensorLedger() as ledger:
        environment: Dict[int, Jet] = {}
        meta_inputs: Dict[str, torch.Tensor] = {}
        for value_id in specialized.inputs:
            value = specialized.values[value_id]
            tensor = meta_inputs.setdefault(
                value.storage_id,
                torch.empty(value.shape, dtype=value.dtype, device="meta"),
            )
            index = variable_index.get(value_id)
            environment[value_id] = seed_jet(
                tensor,
                variables=len(variables),
                order=order,
                variable=index,
                direction=torch.ones_like(tensor) if index is not None else None,
            )
            ledger._record(tensor)
            if index is not None:
                ledger._record(environment[value_id].coefficients[((index, 1),)])

        input_groups: Dict[str, Dict[Tuple[str, int], int]] = {}
        for value_id in specialized.inputs:
            storage = specialized.values[value_id].storage_id
            groups = _LiveTensorLedger.groups_for(_jet_coefficients(environment[value_id]))
            merged = input_groups.setdefault(storage, {})
            for key, nbytes in groups.items():
                merged[key] = max(merged.get(key, 0), nbytes)
        for value_id in specialized.inputs:
            value = specialized.values[value_id]
            specialized.values[value_id] = replace(
                value,
                storage_bytes=sum(input_groups[value.storage_id].values()),
            )

        ledger.peak = ledger.current()
        for step, node_id in enumerate(schedule):
            node = specialized.node(node_id)
            before_groups = ledger.groups()
            before = sum(before_groups.values())
            ledger.peak = before
            output = _execute_primal_jet(node, environment)
            environment[node.outputs[0]] = output
            output_groups = _LiveTensorLedger.groups_for(_jet_coefficients(output))
            new_bytes = sum(nbytes for key, nbytes in output_groups.items() if key not in before_groups)
            node.workspace_bytes = max(0, ledger.peak - before - new_bytes)
            output_value = specialized.values[node.outputs[0]]
            if node.op_name in {"reshape", "transpose"}:
                input_value = specialized.values[node.inputs[0]]
                input_groups = _LiveTensorLedger.groups_for(
                    _jet_coefficients(environment[node.inputs[0]])
                )
                if set(output_groups) == set(input_groups):
                    storage_id = input_value.storage_id
                    storage_bytes = input_value.storage_bytes
                else:
                    # A reshape after a noncontiguous view may copy. Meta
                    # execution observes that distinction; the primal-only
                    # output_storage hook intentionally assumes contiguous
                    # first-order inputs and must not force an alias here.
                    storage_id = f"jet_storage_{output_value.id}"
                    storage_bytes = sum(output_groups.values())
                specialized.values[node.outputs[0]] = replace(
                    output_value,
                    storage_id=storage_id,
                    storage_bytes=storage_bytes,
                )
            else:
                specialized.values[node.outputs[0]] = replace(
                    output_value,
                    storage_bytes=sum(output_groups.values()),
                )

            for value_id in node.inputs:
                if last_use.get(value_id) == step and value_id not in specialized.outputs:
                    environment.pop(value_id, None)
            for value_id in node.outputs:
                if value_id not in last_use and value_id not in specialized.outputs:
                    environment.pop(value_id, None)

    specialized.validate()
    certificate = plan_memory(specialized, budget=budget, exact_node_limit=exact_node_limit)
    return specialized, certificate


def execute_jet_graph(
    graph: JointGraph,
    certificate: MemoryCertificate,
    inputs: Mapping[int, torch.Tensor],
    *,
    directions: Optional[Mapping[int, torch.Tensor]] = None,
) -> Dict[int, Jet]:
    """Execute a sparse-jet graph with certified schedule and last-use release."""

    graph.validate()
    if graph.execution_mode != "jet":
        raise ValueError("execute_jet_graph requires a specialized jet graph")
    certificate.validate(graph)
    directions = directions or {}
    if any(value_id not in graph.variable_inputs for value_id in directions):
        raise ValueError("A direction was supplied for a constant or unknown input")
    variable_index = {value_id: index for index, value_id in enumerate(graph.variable_inputs)}
    environment: Dict[int, Jet] = {}
    for value_id in graph.inputs:
        if value_id not in inputs:
            raise KeyError(f"Missing input tensor for {graph.values[value_id].name}")
        value = graph.values[value_id]
        tensor = inputs[value_id]
        if tuple(tensor.shape) != value.shape or tensor.dtype != value.dtype:
            raise ValueError(f"Input signature mismatch for {value.name}")
        if not tensor.is_contiguous():
            raise ValueError("Jet execution requires contiguous inputs")
        if graph.derivative_order > 1 and not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError("Higher-order jet execution requires finite inputs")
        index = variable_index.get(value_id)
        direction = directions.get(value_id)
        if direction is not None and not direction.is_contiguous():
            raise ValueError("Jet execution requires contiguous directions")
        if (
            direction is not None
            and graph.derivative_order > 1
            and not bool(torch.all(torch.isfinite(direction)))
        ):
            raise ValueError("Higher-order jet execution requires finite directions")
        environment[value_id] = seed_jet(
            tensor,
            variables=len(graph.variable_inputs),
            order=graph.derivative_order,
            variable=index,
            direction=direction,
        )
    aliased_inputs: Dict[str, torch.Tensor] = {}
    for value_id in graph.inputs:
        value = graph.values[value_id]
        tensor = inputs[value_id]
        previous = aliased_inputs.setdefault(value.storage_id, tensor)
        if previous is not tensor and tensor.numel() and previous.data_ptr() != tensor.data_ptr():
            raise ValueError(f"Inputs declared as aliases do not share storage: {value.name}")
    devices = {inputs[value_id].device for value_id in graph.inputs}
    if len(devices) > 1:
        raise ValueError("Jet execution requires all inputs on one device")

    last_use = last_use_by_step(graph, certificate.schedule)
    with torch.no_grad():
        for step, node_id in enumerate(certificate.schedule):
            node = graph.node(node_id)
            output = _execute_primal_jet(node, environment)
            expected = graph.values[node.outputs[0]]
            if output.shape != expected.shape or output.dtype != expected.dtype:
                raise RuntimeError(f"Jet primal signature mismatch at {node.name}")
            if graph.derivative_order > 1:
                for coefficient in output.coefficients.values():
                    if not bool(torch.all(torch.isfinite(coefficient))):
                        raise ValueError(
                            f"Nonfinite jet coefficient at {node.name}; "
                            "the evaluation left the qualified smooth domain"
                        )
            environment[node.outputs[0]] = output
            for value_id in node.inputs:
                if last_use.get(value_id) == step and value_id not in graph.outputs:
                    environment.pop(value_id, None)
            for value_id in node.outputs:
                if value_id not in last_use and value_id not in graph.outputs:
                    environment.pop(value_id, None)
    return {value_id: environment[value_id] for value_id in graph.outputs}


__all__ = ["execute_jet_graph", "plan_jet_memory"]
