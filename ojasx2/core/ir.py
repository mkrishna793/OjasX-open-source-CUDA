"""Executable joint primal/cotangent IR for the certified core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from .contracts import (
    CORE_REGISTRY,
    COTANGENT_ENCODINGS,
    DENSE_COTANGENT,
    UNIFORM_COTANGENT,
    Value,
    tensor_nbytes,
)
from .operators import register_core_operators


@dataclass
class Node:
    id: int
    name: str
    op_name: str
    kind: str
    inputs: Tuple[int, ...]
    outputs: Tuple[int, ...]
    workspace_bytes: int = 0
    forward_inputs: Tuple[int, ...] = ()
    residual_ids: Tuple[int, ...] = ()
    residual_slots: Tuple[Tuple[int, ...], ...] = ()
    residual_policy: str = ""
    demand: Tuple[bool, ...] = ()
    gradient_input_indices: Tuple[int, ...] = ()
    attrs: Tuple[Tuple[str, object], ...] = ()


@dataclass
class JointGraph:
    values: Dict[int, Value] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    inputs: List[int] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    gradient_targets: Dict[int, int] = field(default_factory=dict)
    objective: Optional[int] = None
    execution_mode: str = "first_order"
    derivative_order: int = 1
    variable_inputs: Tuple[int, ...] = ()
    structured_cotangents: bool = True

    def node(self, node_id: int) -> Node:
        if 0 <= node_id < len(self.nodes) and self.nodes[node_id].id == node_id:
            return self.nodes[node_id]
        raise KeyError(f"Unknown joint-IR node {node_id}")

    def validate(self) -> None:
        if self.execution_mode not in {"first_order", "jet"}:
            raise ValueError("Unknown core execution mode")
        if self.execution_mode == "first_order" and (self.derivative_order != 1 or self.variable_inputs):
            raise ValueError("First-order graphs cannot carry jet specialization metadata")
        if self.execution_mode == "jet":
            if not 1 <= self.derivative_order <= 4 or not self.variable_inputs:
                raise ValueError("Jet graphs require order one through four and variable inputs")
            if len(set(self.variable_inputs)) != len(self.variable_inputs) or any(
                value_id not in self.inputs for value_id in self.variable_inputs
            ):
                raise ValueError("Jet variable inputs must be unique external inputs")
            if self.seeds or self.gradient_targets or any(node.kind != "primal" or node.residual_ids for node in self.nodes):
                raise ValueError("Jet graphs must be clean primal-only graphs")
        if self.objective is None or self.objective not in self.values:
            raise ValueError("Joint graph requires a scalar objective")
        if self.values[self.objective].shape != ():
            raise ValueError("Joint graph objective must be scalar")
        if not self.outputs or any(value_id not in self.values for value_id in self.outputs):
            raise ValueError("Joint graph outputs are incomplete")
        if len(set(self.outputs)) != len(self.outputs) or self.objective not in self.outputs:
            raise ValueError("Joint graph outputs must be unique and include the objective")
        if any(self.values[value_id].encoding != DENSE_COTANGENT for value_id in self.outputs):
            raise ValueError("Public graph outputs require dense materialization boundaries")
        if any(source not in self.values or gradient not in self.outputs for source, gradient in self.gradient_targets.items()):
            raise ValueError("Gradient target mapping is incomplete")
        for source, gradient in self.gradient_targets.items():
            if not self.values[source].requires_grad:
                raise ValueError("Gradient source was not marked differentiable")
            if self.values[gradient].kind not in {"cotangent", "seed"}:
                raise ValueError("Gradient target must be a cotangent or objective seed")
            source_value = self.values[source]
            gradient_value = self.values[gradient]
            if gradient_value.shape != source_value.shape or gradient_value.dtype != source_value.dtype:
                raise ValueError("Gradient target signature does not match its source")
        producers = {node.id: node for node in self.nodes}
        if len(producers) != len(self.nodes):
            raise ValueError("Duplicate joint-IR node id")
        if tuple(self.values) != tuple(range(len(self.values))):
            raise ValueError("Joint IR value ids must be stable and sequential")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Joint graph inputs and seeds must be unique")
        for value_id in self.inputs:
            if value_id not in self.values or self.values[value_id].producer is not None:
                raise ValueError("Invalid external input")
        for value_id in self.seeds:
            if value_id not in self.values or self.values[value_id].shape != () or self.values[value_id].kind != "seed":
                raise ValueError("Invalid objective seed")
        storage_types: Dict[str, Tuple[int, torch.dtype]] = {}
        for value in self.values.values():
            if value.nbytes != tensor_nbytes(value.shape, value.dtype):
                raise ValueError(f"Incorrect byte size for value {value.name}")
            if value.encoding not in COTANGENT_ENCODINGS:
                raise ValueError(f"Unknown cotangent encoding for {value.name}")
            if not self.structured_cotangents and value.encoding != DENSE_COTANGENT:
                raise ValueError("Structured cotangents are disabled for this graph")
            if value.encoding == UNIFORM_COTANGENT:
                element_bytes = torch.empty((), dtype=value.dtype, device="meta").element_size()
                if (
                    value.kind != "cotangent"
                    or value.nbytes <= 0
                    or value.storage_bytes != element_bytes
                ):
                    raise ValueError(f"Invalid uniform cotangent storage for {value.name}")
            elif value.storage_bytes < value.nbytes:
                raise ValueError(f"Storage is smaller than value {value.name}")
            previous = storage_types.setdefault(value.storage_id, (value.storage_bytes, value.dtype))
            if previous != (value.storage_bytes, value.dtype):
                raise ValueError(f"Inconsistent aliased storage {value.storage_id}")
            if value.producer is not None and value.producer not in producers:
                raise ValueError(f"Missing producer for {value.name}")
        for position, node in enumerate(self.nodes):
            if node.id != position:
                raise ValueError("Joint IR node ids must be stable and sequential")
            operator = (
                CORE_REGISTRY.get(node.op_name)
                if node.kind in {"primal", "pullback"}
                else None
            )
            if (
                self.execution_mode == "jet"
                and operator is not None
                and self.derivative_order > operator.contract.max_derivative_order
            ):
                raise ValueError(f"{node.op_name} does not implement the requested derivative order")
            if operator is not None:
                operator.contract.policy(node.residual_policy)
            if not node.outputs:
                raise ValueError(f"Node {node.name} has no outputs")
            if any(value_id not in self.values for value_id in (*node.inputs, *node.outputs)):
                raise ValueError(f"Node {node.name} references an unknown value")
            for value_id in node.inputs:
                producer = self.values[value_id].producer
                if producer is not None and producer >= node.id:
                    raise ValueError("Joint IR is not topologically ordered")
            for value_id in node.outputs:
                if self.values[value_id].producer != node.id:
                    raise ValueError(f"Node/output producer mismatch at {node.name}")
            if len(set(node.outputs)) != len(node.outputs) or node.workspace_bytes < 0:
                raise ValueError(f"Invalid outputs or workspace at {node.name}")
            if any(value_id not in self.values or self.values[value_id].kind != "residual" for value_id in node.residual_ids):
                raise ValueError(f"Residual values are inconsistent at {node.name}")
            if len(node.residual_slots) != len(node.residual_ids):
                raise ValueError(f"Residual slots are incomplete at {node.name}")
            if node.kind == "primal":
                if any(self.values[value_id].encoding != DENSE_COTANGENT for value_id in node.outputs):
                    raise ValueError(f"Primal outputs must be dense at {node.name}")
                if node.forward_inputs != node.inputs or len(node.demand) != len(node.inputs):
                    raise ValueError(f"Primal demand is inconsistent at {node.name}")
                if node.residual_ids != node.outputs[1:]:
                    raise ValueError(f"Primal residuals must follow the primal output at {node.name}")
                if any(
                    not slots or any(slot < 0 or slot >= len(node.demand) for slot in slots)
                    for slots in node.residual_slots
                ):
                    raise ValueError(f"Residual gradient slot is invalid at {node.name}")
            elif node.kind == "pullback":
                if not node.inputs or node.inputs[1:] != node.residual_ids:
                    raise ValueError(f"Pullback inputs are inconsistent at {node.name}")
                if len(node.demand) != len(node.forward_inputs):
                    raise ValueError(f"Pullback demand is inconsistent at {node.name}")
                if len(node.gradient_input_indices) != len(node.outputs):
                    raise ValueError(f"Pullback outputs are inconsistent at {node.name}")
                if len(set(node.gradient_input_indices)) != len(node.gradient_input_indices) or any(
                    index < 0 or index >= len(node.demand) for index in node.gradient_input_indices
                ):
                    raise ValueError(f"Pullback gradient slots are invalid at {node.name}")
                if any(not node.demand[index] for index in node.gradient_input_indices):
                    raise ValueError(f"Pullback emits an undemanded gradient at {node.name}")
                assert operator is not None
                for position, value_id in enumerate(node.outputs):
                    expected = (
                        operator.cotangent_encoding(self, node, position)
                        if self.structured_cotangents
                        else DENSE_COTANGENT
                    )
                    if self.values[value_id].encoding != expected:
                        raise ValueError(f"Pullback encoding mismatch at {node.name}")
            elif node.kind == "cotangent_add":
                if len(node.inputs) != 2 or len(node.outputs) != 1:
                    raise ValueError(f"Cotangent accumulation must be binary at {node.name}")
                left, right = (self.values[value_id] for value_id in node.inputs)
                expected = (
                    UNIFORM_COTANGENT
                    if self.structured_cotangents
                    and left.encoding == right.encoding == UNIFORM_COTANGENT
                    else DENSE_COTANGENT
                )
                if self.values[node.outputs[0]].encoding != expected:
                    raise ValueError(f"Cotangent-add encoding mismatch at {node.name}")
            elif node.kind == "cotangent_materialize":
                if len(node.inputs) != 1 or len(node.outputs) != 1:
                    raise ValueError(f"Cotangent materialization must be unary at {node.name}")
                source = self.values[node.inputs[0]]
                output = self.values[node.outputs[0]]
                if (
                    source.encoding != UNIFORM_COTANGENT
                    or output.encoding != DENSE_COTANGENT
                    or source.shape != output.shape
                    or source.dtype != output.dtype
                ):
                    raise ValueError(f"Invalid dense materialization boundary at {node.name}")
            else:
                raise ValueError(f"Unknown joint node kind {node.kind!r}")


class JointGraphBuilder:
    """Build first-order joint graphs with explicit residual information."""

    def __init__(self, *, structured_cotangents: bool = True) -> None:
        register_core_operators()
        self.graph = JointGraph(structured_cotangents=structured_cotangents)
        self._primal_nodes: List[int] = []
        self._gradient_requests: Set[int] = set()
        self._reverse_generated = False

    def _new_value(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        producer: Optional[int],
        requires_grad: bool,
        kind: str,
        storage_id: Optional[str] = None,
        storage_bytes: Optional[int] = None,
        role: str = "",
        encoding: str = DENSE_COTANGENT,
    ) -> int:
        value_id = len(self.graph.values)
        nbytes = tensor_nbytes(shape, dtype)
        if storage_bytes is None and encoding == UNIFORM_COTANGENT:
            storage_bytes = torch.empty((), dtype=dtype, device="meta").element_size()
        value = Value(
            id=value_id,
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            nbytes=nbytes,
            storage_id=storage_id or f"storage_{value_id}",
            storage_bytes=nbytes if storage_bytes is None else storage_bytes,
            producer=producer,
            requires_grad=requires_grad,
            kind=kind,
            role=role,
            encoding=encoding,
        )
        self.graph.values[value_id] = value
        return value_id

    def input(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        alias_of: Optional[int] = None,
    ) -> int:
        if self._reverse_generated:
            raise RuntimeError("Cannot add inputs after reverse generation")
        if alias_of is None:
            value_id = self._new_value(
                name,
                tuple(shape),
                dtype,
                producer=None,
                requires_grad=requires_grad,
                kind="input",
            )
        else:
            if alias_of not in self.graph.inputs:
                raise ValueError("Input aliases must refer to an existing external input")
            source = self.graph.values[alias_of]
            if tuple(shape) != source.shape or dtype != source.dtype:
                raise ValueError("M0–M2 input aliases must have identical shape and dtype")
            if source.requires_grad != requires_grad:
                raise ValueError("Aliased views of one autograd leaf must share gradient eligibility")
            value_id = self._new_value(
                name,
                tuple(shape),
                dtype,
                producer=None,
                requires_grad=requires_grad,
                kind="input",
                storage_id=source.storage_id,
                storage_bytes=source.storage_bytes,
            )
        self.graph.inputs.append(value_id)
        return value_id

    def forward(
        self,
        op_name: str,
        inputs: Sequence[int],
        *,
        name: Optional[str] = None,
        residual_policy: Optional[str] = None,
        attrs: Optional[Mapping[str, object]] = None,
    ) -> int:
        if self._reverse_generated:
            raise RuntimeError("Cannot add primal nodes after reverse generation")
        operator = CORE_REGISTRY.get(op_name)
        values = [self.graph.values[value_id] for value_id in inputs]
        if not operator.contract.min_arity <= len(values) <= operator.contract.max_arity:
            raise ValueError(f"Invalid arity for {op_name}")
        demand = tuple(value.requires_grad for value in values)
        policy = residual_policy or operator.contract.residual_policies[0].name
        operator.contract.policy(policy)
        attributes = dict(attrs or {})
        unknown_attributes = set(attributes) - set(operator.contract.attribute_names)
        if unknown_attributes:
            raise ValueError(
                f"Unsupported attributes for {op_name}: {sorted(unknown_attributes)}"
            )
        output_shape = operator.output_shape(values, attributes)
        storage_source = operator.output_storage(values, attributes)
        node_id = len(self.graph.nodes)
        source = values[storage_source] if storage_source is not None else None
        output = self._new_value(
            name or f"{op_name}_{node_id}",
            output_shape,
            values[0].dtype,
            producer=node_id,
            requires_grad=any(demand),
            kind="primal",
            storage_id=source.storage_id if source is not None else None,
            storage_bytes=source.storage_bytes if source is not None else None,
        )
        node = Node(
            id=node_id,
            name=name or f"{op_name}_{node_id}",
            op_name=op_name,
            kind="primal",
            inputs=tuple(inputs),
            outputs=(output,),
            workspace_bytes=operator.forward_workspace(values, demand, policy),
            forward_inputs=tuple(inputs),
            residual_policy=policy,
            demand=demand,
            attrs=tuple(sorted(attributes.items())),
        )
        self.graph.nodes.append(node)
        self._primal_nodes.append(node_id)
        return output

    def _materialize_residuals(self, node: Node) -> None:
        operator = CORE_REGISTRY.get(node.op_name)
        inputs = [self.graph.values[value_id] for value_id in node.inputs]
        layouts = operator.residual_layout(inputs, node.demand, node.residual_policy)
        residual_ids = []
        for layout in layouts:
            if layout.source_index is None:
                residual = self._new_value(
                    f"{node.op_name}_{node.id}_{layout.role}",
                    layout.shape or (),
                    layout.dtype or inputs[0].dtype,
                    producer=node.id,
                    requires_grad=False,
                    kind="residual",
                    role=layout.role,
                )
                if self.graph.values[residual].nbytes != layout.nbytes:
                    raise RuntimeError(f"Residual byte contract mismatch at {node.name}")
            else:
                source = (
                    self.graph.values[node.outputs[0]]
                    if layout.source_index == -1
                    else inputs[layout.source_index]
                )
                residual = self._new_value(
                    f"{node.op_name}_{node.id}_{layout.role}",
                    source.shape,
                    source.dtype,
                    producer=node.id,
                    requires_grad=False,
                    kind="residual",
                    storage_id=source.storage_id,
                    storage_bytes=source.storage_bytes,
                    role=layout.role,
                )
            residual_ids.append(residual)
        node.outputs = (node.outputs[0], *residual_ids)
        node.residual_ids = tuple(residual_ids)
        node.residual_slots = tuple(tuple(layout.gradient_slots) for layout in layouts)

    def request_gradient(self, value_id: int) -> None:
        if self._reverse_generated:
            raise RuntimeError("Request gradients before reverse generation")
        if not self.graph.values[value_id].requires_grad:
            raise ValueError("Cannot request a gradient for a frozen value")
        self._gradient_requests.add(value_id)

    def request_input_gradients(self) -> None:
        if self._reverse_generated:
            raise RuntimeError("Request gradients before reverse generation")
        for value_id in self.graph.inputs:
            if self.graph.values[value_id].requires_grad:
                self._gradient_requests.add(value_id)

    def set_objective(self, value_id: int) -> int:
        if self.graph.objective is not None:
            raise RuntimeError("Objective is already set")
        objective = self.forward("sum", (value_id,), name="objective_sum", residual_policy="shape_only")
        self.graph.objective = objective
        self.graph.outputs = [objective]
        return objective

    def _cotangent_add(self, left: int, right: int) -> int:
        left_value = self.graph.values[left]
        right_value = self.graph.values[right]
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            raise RuntimeError("Cotangent accumulation shape/dtype mismatch")
        node_id = len(self.graph.nodes)
        encoding = (
            UNIFORM_COTANGENT
            if self.graph.structured_cotangents
            and left_value.encoding == right_value.encoding == UNIFORM_COTANGENT
            else DENSE_COTANGENT
        )
        output = self._new_value(
            f"cotangent_add_{node_id}",
            left_value.shape,
            left_value.dtype,
            producer=node_id,
            requires_grad=False,
            kind="cotangent",
            encoding=encoding,
        )
        self.graph.nodes.append(
            Node(
                id=node_id,
                name=f"cotangent_add_{node_id}",
                op_name="add",
                kind="cotangent_add",
                inputs=(left, right),
                outputs=(output,),
            )
        )
        return output

    def _materialize_cotangent(self, value_id: int) -> int:
        source = self.graph.values[value_id]
        if source.encoding == DENSE_COTANGENT:
            return value_id
        if source.encoding != UNIFORM_COTANGENT:
            raise ValueError(f"Unknown structured cotangent {source.encoding!r}")
        node_id = len(self.graph.nodes)
        output = self._new_value(
            f"dense_cotangent_{node_id}",
            source.shape,
            source.dtype,
            producer=node_id,
            requires_grad=False,
            kind="cotangent",
            role="dense_boundary",
        )
        self.graph.nodes.append(
            Node(
                id=node_id,
                name=f"materialize_{node_id}",
                op_name="cotangent_materialize",
                kind="cotangent_materialize",
                inputs=(value_id,),
                outputs=(output,),
            )
        )
        return output

    def generate_reverse(self) -> JointGraph:
        if self._reverse_generated:
            return self.graph
        if self.graph.objective is None:
            raise RuntimeError("Set an objective before generating reverse mode")
        objective = self.graph.objective
        primal_ids = set(self._primal_nodes)
        objective_ancestors: Set[int] = set()

        objective_producer = self.graph.values[objective].producer
        if objective_producer is None:
            raise RuntimeError("Objective must be produced by a core operation")
        pending = [objective_producer]
        while pending:
            node_id = pending.pop()
            if node_id in objective_ancestors:
                continue
            objective_ancestors.add(node_id)
            for input_id in self.graph.node(node_id).inputs:
                producer = self.graph.values[input_id].producer
                if producer in primal_ids:
                    pending.append(producer)
        consumers: Dict[int, List[int]] = {}
        for node_id in self._primal_nodes:
            node = self.graph.node(node_id)
            for input_id in node.inputs:
                consumers.setdefault(input_id, []).append(node_id)

        def path_roots(target: int) -> Tuple[int, ...]:
            if target not in self.graph.inputs:
                return (target,)
            storage = self.graph.values[target].storage_id
            return tuple(
                value_id
                for value_id in self.graph.inputs
                if self.graph.values[value_id].storage_id == storage
            )

        active: Set[int] = set()
        requested_paths: Set[int] = set()
        for target in sorted(self._gradient_requests):
            roots = path_roots(target)
            requested_paths.update(roots)
            queue = list(roots)
            visited: Set[int] = set()
            connected = False
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current == objective:
                    connected = True
                    continue
                for consumer in consumers.get(current, []):
                    if consumer not in objective_ancestors:
                        continue
                    active.add(consumer)
                    queue.append(self.graph.node(consumer).outputs[0])
            if not connected:
                raise ValueError(f"Requested gradient for {self.graph.values[target].name} is disconnected")

        # Restrict both residual production and backward work to paths that can
        # reach a requested target. A requested intermediate value is a natural
        # boundary unless another requested target lies farther upstream.
        for forward_node_id in self._primal_nodes:
            forward_node = self.graph.node(forward_node_id)
            effective = []
            for index, source_id in enumerate(forward_node.inputs):
                source = self.graph.values[source_id]
                needed = (
                    forward_node_id in active
                    and forward_node.demand[index]
                    and (source_id in requested_paths or source.producer in active)
                )
                effective.append(bool(needed))
            forward_node.demand = tuple(effective)
            self._materialize_residuals(forward_node)
            original_inputs = [self.graph.values[value_id] for value_id in forward_node.inputs]
            forward_node.workspace_bytes = CORE_REGISTRY.get(forward_node.op_name).forward_workspace(
                original_inputs,
                forward_node.demand,
                forward_node.residual_policy,
            )

        seed = self._new_value(
            "objective_upstream",
            (),
            self.graph.values[objective].dtype,
            producer=None,
            requires_grad=False,
            kind="seed",
            role="upstream",
        )
        self.graph.seeds.append(seed)
        sources: Dict[int, List[int]] = {objective: [seed]}
        combined: Dict[int, int] = {}

        def accumulate(value_id: int) -> int:
            contributions = sources.get(value_id, [])
            if not contributions:
                raise RuntimeError(f"No cotangent contributions for {self.graph.values[value_id].name}")
            if len(contributions) == 1:
                return contributions[0]
            if value_id in combined:
                return combined[value_id]
            current = contributions[0]
            for contribution in contributions[1:]:
                current = self._cotangent_add(current, contribution)
            combined[value_id] = current
            return current

        for forward_node_id in reversed(self._primal_nodes):
            if forward_node_id not in active:
                continue
            forward_node = self.graph.node(forward_node_id)
            primal = forward_node.outputs[0]
            if primal not in sources:
                continue
            upstream = accumulate(primal)
            gradient_indices = tuple(index for index, needed in enumerate(forward_node.demand) if needed)
            if not gradient_indices:
                continue
            operator = CORE_REGISTRY.get(forward_node.op_name)
            upstream_encoding = self.graph.values[upstream].encoding
            if not operator.accepts_cotangent(upstream_encoding):
                upstream = self._materialize_cotangent(upstream)
            original_inputs = [self.graph.values[value_id] for value_id in forward_node.inputs]
            backward_workspace = operator.backward_workspace_for_node(
                original_inputs,
                forward_node,
            )
            node_id = len(self.graph.nodes)
            backward_node = Node(
                id=node_id,
                name=f"pullback_{forward_node.name}",
                op_name=forward_node.op_name,
                kind="pullback",
                inputs=(upstream, *forward_node.residual_ids),
                outputs=(),
                workspace_bytes=backward_workspace,
                forward_inputs=forward_node.inputs,
                residual_ids=forward_node.residual_ids,
                residual_slots=forward_node.residual_slots,
                residual_policy=forward_node.residual_policy,
                demand=forward_node.demand,
                gradient_input_indices=gradient_indices,
                attrs=forward_node.attrs,
            )
            contributions = []
            for position, input_index in enumerate(gradient_indices):
                source = self.graph.values[forward_node.inputs[input_index]]
                encoding = (
                    operator.cotangent_encoding(self.graph, backward_node, position)
                    if self.graph.structured_cotangents
                    else DENSE_COTANGENT
                )
                contributions.append(
                    self._new_value(
                        f"d_{forward_node.name}_{source.name}_{node_id}",
                        source.shape,
                        source.dtype,
                        producer=node_id,
                        requires_grad=False,
                        kind="cotangent",
                        role=f"input_{input_index}",
                        encoding=encoding,
                    )
                )
            backward_node.outputs = tuple(contributions)
            self.graph.nodes.append(backward_node)
            for input_index, contribution in zip(gradient_indices, contributions):
                sources.setdefault(forward_node.inputs[input_index], []).append(contribution)

        completed_alias_groups: Set[str] = set()
        for value_id in sorted(self._gradient_requests):
            roots = path_roots(value_id)
            if value_id in self.graph.inputs:
                storage = self.graph.values[value_id].storage_id
                if storage in completed_alias_groups:
                    continue
                completed_alias_groups.add(storage)
                contributions = [accumulate(root) for root in roots if sources.get(root)]
                if not contributions:
                    raise ValueError(f"Requested gradient for {self.graph.values[value_id].name} is disconnected")
                gradient = contributions[0]
                for contribution in contributions[1:]:
                    gradient = self._cotangent_add(gradient, contribution)
                requested_roots = [root for root in roots if root in self._gradient_requests]
            else:
                if value_id not in sources:
                    raise ValueError(f"Requested gradient for {self.graph.values[value_id].name} is disconnected")
                gradient = accumulate(value_id)
                requested_roots = [value_id]
            gradient = self._materialize_cotangent(gradient)
            if gradient not in self.graph.outputs:
                self.graph.outputs.append(gradient)
            for root in requested_roots:
                self.graph.gradient_targets[root] = gradient
        self._reverse_generated = True
        self.graph.validate()
        return self.graph


__all__ = ["JointGraph", "JointGraphBuilder", "Node"]
