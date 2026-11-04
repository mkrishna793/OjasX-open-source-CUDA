"""First-order executor for certified joint graphs and memory schedules."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from .contracts import CORE_REGISTRY, UNIFORM_COTANGENT
from .ir import JointGraph, Node
from .memory import MemoryCertificate, last_use_by_step
from .structured import add_uniform


def _execute_node(graph: JointGraph, node: Node, environment: Dict[int, torch.Tensor]) -> None:
    """Execute one node in a separate scope so Python locals obey liveness."""

    if node.kind == "cotangent_add":
        left, right = (environment[value_id] for value_id in node.inputs)
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            environment[node.outputs[0]] = add_uniform(left, right)
        else:
            environment[node.outputs[0]] = left + right
        return
    if node.kind == "cotangent_materialize":
        source = environment[node.inputs[0]]
        environment[node.outputs[0]] = source.clone(memory_format=torch.contiguous_format)
        return
    operator = CORE_REGISTRY.get(node.op_name)
    if node.kind == "primal":
        tensors = [environment[value_id] for value_id in node.inputs]
        output, residuals = operator.forward(
            tensors,
            {**dict(node.attrs), "demand": node.demand},
            node.residual_policy,
        )
        expected = graph.values[node.outputs[0]]
        if tuple(output.shape) != expected.shape or output.dtype != expected.dtype:
            raise RuntimeError(f"Primal signature mismatch at {node.name}")
        environment[node.outputs[0]] = output
        produced_residuals = set()
        for residual_id in node.residual_ids:
            role = graph.values[residual_id].role
            if role not in residuals:
                raise RuntimeError(f"Missing residual {role!r} at {node.name}")
            environment[residual_id] = residuals[role]
            produced_residuals.add(role)
        if set(residuals) != produced_residuals:
            raise RuntimeError(f"Unexpected residual roles at {node.name}: {set(residuals) - produced_residuals}")
        return
    if node.kind == "pullback":
        upstream = environment[node.inputs[0]]
        residuals = {
            graph.values[residual_id].role: environment[residual_id]
            for residual_id in node.residual_ids
        }
        gradients = operator.backward(graph, node, upstream, residuals)
        if len(gradients) != len(node.outputs):
            raise RuntimeError(f"Pullback arity mismatch at {node.name}")
        for value_id, gradient in zip(node.outputs, gradients):
            expected = graph.values[value_id]
            if tuple(gradient.shape) != expected.shape or gradient.dtype != expected.dtype:
                raise RuntimeError(f"Cotangent signature mismatch at {node.name}")
            environment[value_id] = gradient
        return
    raise RuntimeError(f"Unknown joint node kind {node.kind!r}")


def execute_joint_graph(
    graph: JointGraph,
    certificate: MemoryCertificate,
    inputs: Mapping[int, torch.Tensor],
    *,
    objective_upstream: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Execute a planned graph and release managed references at last use.

    This is the first-order semantic reference. It runs under ``no_grad`` so
    executing the explicit joint IR does not also construct a hidden PyTorch
    autograd tape. Higher-order execution uses the separate sparse-jet contract.
    """

    graph.validate()
    certificate.validate(graph)
    environment: Dict[int, torch.Tensor] = {}
    for value_id in graph.inputs:
        if value_id not in inputs:
            raise KeyError(f"Missing input tensor for {graph.values[value_id].name}")
        value = graph.values[value_id]
        tensor = inputs[value_id]
        if tuple(tensor.shape) != value.shape or tensor.dtype != value.dtype:
            raise ValueError(f"Input signature mismatch for {value.name}")
        environment[value_id] = tensor
    aliased_inputs: Dict[str, torch.Tensor] = {}
    for value_id in graph.inputs:
        value = graph.values[value_id]
        tensor = environment[value_id]
        previous = aliased_inputs.setdefault(value.storage_id, tensor)
        if previous is not tensor and tensor.numel() and previous.data_ptr() != tensor.data_ptr():
            raise ValueError(f"Inputs declared as aliases do not share storage: {value.name}")
    devices = {environment[value_id].device for value_id in graph.inputs}
    if len(devices) > 1:
        raise ValueError("The initial core executor requires all inputs on one device")
    seed_device = objective_upstream.device if objective_upstream is not None else next(
        (environment[value_id].device for value_id in graph.inputs), torch.device("cpu")
    )
    if objective_upstream is not None and devices and objective_upstream.device not in devices:
        raise ValueError("Objective upstream must be on the input device")
    for value_id in graph.seeds:
        value = graph.values[value_id]
        if objective_upstream is None:
            environment[value_id] = torch.ones((), dtype=value.dtype, device=seed_device)
        else:
            if objective_upstream.numel() != 1 or objective_upstream.dtype != value.dtype:
                raise ValueError("Objective upstream must be a scalar of the objective dtype")
            environment[value_id] = objective_upstream.reshape(())

    last_use = last_use_by_step(graph, certificate.schedule)
    with torch.no_grad():
        for step, node_id in enumerate(certificate.schedule):
            node = graph.node(node_id)
            _execute_node(graph, node, environment)
            for value_id in node.inputs:
                if last_use.get(value_id) == step and value_id not in graph.outputs:
                    environment.pop(value_id, None)
            # A value with no consumer and no requested output is dead at the
            # end of its producing step. Do not retain it until function exit.
            for value_id in node.outputs:
                if value_id not in last_use and value_id not in graph.outputs:
                    environment.pop(value_id, None)

    missing = [value_id for value_id in graph.outputs if value_id not in environment]
    if missing:
        names = [graph.values[value_id].name for value_id in missing]
        raise RuntimeError(f"Execution did not preserve requested outputs: {names}")
    return {value_id: environment[value_id] for value_id in graph.outputs}


__all__ = ["execute_joint_graph"]
