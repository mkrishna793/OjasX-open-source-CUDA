"""Reference operators and lossless demanded-residual implementations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from ojasx2.residuals import pack_mask, unpack_mask

from .jets import Jet
from .contracts import (
    AccuracyMode,
    CORE_REGISTRY,
    CoreOperator,
    DENSE_COTANGENT,
    OperatorContract,
    ProofRef,
    ProofSystem,
    ResidualBuild,
    ResidualPolicy,
    UNIFORM_COTANGENT,
    Value,
    tensor_nbytes,
)
from .structured import (
    UniformCotangentAcceptor,
    clone_uniform,
    nonempty,
)


def _validate_tensors(
    tensors: Sequence[torch.Tensor],
    *,
    require_contiguous: bool = True,
) -> None:
    if not tensors:
        raise ValueError("An operator requires at least one tensor")
    device = tensors[0].device
    dtype = tensors[0].dtype
    if not dtype.is_floating_point:
        raise TypeError("Certified-core differentiable operators require floating tensors")
    for tensor in tensors:
        if tensor.device != device or tensor.dtype != dtype:
            raise ValueError("Core reference operators require matching device and dtype")
        if require_contiguous and not tensor.is_contiguous():
            raise ValueError("The reference memory contract requires contiguous tensors")


def _demand(attrs: Mapping[str, Any], arity: int) -> Tuple[bool, ...]:
    demand = tuple(bool(value) for value in attrs.get("demand", ()))
    if len(demand) != arity:
        raise RuntimeError("Internal gradient-demand metadata is incomplete")
    return demand


class LinearOperator(UniformCotangentAcceptor, CoreOperator):
    contract = OperatorContract(
        name="linear",
        min_arity=2,
        max_arity=3,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=4,
        derivative_domain="all finite real inputs",
        residual_policies=(
            ResidualPolicy(
                name="demand_resident",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input", "weight", "bias"),
                proofs=(
                    ProofRef(ProofSystem.LEAN, "OjasXMath.matrix_product_pullback"),
                    ProofRef(ProofSystem.ROCQ, "Ojasx.CoreOperators.core_linear_pairing"),
                    ProofRef(ProofSystem.Z3, "linear_demand_residual_sufficiency"),
                ),
            ),
        ),
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) not in (2, 3):
            raise ValueError("linear expects X, W, and optional bias")
        x, weight = inputs[:2]
        if len(x.shape) < 1 or len(weight.shape) != 2:
            raise ValueError("linear expects X [..., K] and W [N, K]")
        if x.shape[-1] != weight.shape[1]:
            raise ValueError("linear inner dimensions do not match")
        if any(value.dtype != x.dtype for value in inputs):
            raise TypeError("linear operands must share dtype")
        if len(inputs) == 3 and inputs[2].shape != (weight.shape[0],):
            raise ValueError("linear bias must have shape [N]")
        return (*x.shape[:-1], weight.shape[0])

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        result = inputs[0] @ inputs[1].transpose(-1, -2)
        if len(inputs) == 3:
            result = result + inputs[2]
        return result.set_value(F.linear(*(jet.value for jet in inputs)))

    def residual_layout(
        self,
        inputs: Sequence[Value],
        demand: Tuple[bool, ...],
        policy: str,
    ) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        if len(demand) != len(inputs):
            raise ValueError("linear demand arity mismatch")
        residuals = []
        if demand[1]:
            residuals.append(ResidualBuild("x", "resident_alias", (1,), source_index=0))
        if demand[0]:
            residuals.append(ResidualBuild("weight", "resident_alias", (0,), source_index=1))
        return tuple(residuals)

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        self.contract.policy(policy)
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        self.contract.policy(policy)
        return 0

    def forward(
        self,
        tensors: Sequence[torch.Tensor],
        attrs: Mapping[str, Any],
        policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, len(tensors))
        output = F.linear(tensors[0], tensors[1], tensors[2] if len(tensors) == 3 else None)
        residuals: Dict[str, torch.Tensor] = {}
        if demand[1]:
            residuals["x"] = tensors[0]
        if demand[0]:
            residuals["weight"] = tensors[1]
        return output, residuals

    def backward(
        self,
        graph: "JointGraph",
        node: "Node",
        upstream: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        forward_inputs = [graph.values[value_id] for value_id in node.forward_inputs]
        results = []
        for input_index in node.gradient_input_indices:
            if input_index == 0:
                results.append(torch.matmul(upstream, residuals["weight"]))
            elif input_index == 1:
                x = residuals["x"]
                out_features = forward_inputs[1].shape[0]
                in_features = forward_inputs[1].shape[1]
                batch = 1
                for dimension in forward_inputs[0].shape[:-1]:
                    batch *= dimension
                grad_2d = upstream.reshape(batch, out_features)
                x_2d = x.reshape(batch, in_features)
                results.append(grad_2d.transpose(0, 1).matmul(x_2d))
            elif input_index == 2:
                results.append(upstream.sum_to_size(forward_inputs[2].shape))
            else:
                raise RuntimeError("Invalid linear gradient slot")
        return tuple(results)


class ReluOperator(UniformCotangentAcceptor, CoreOperator):
    contract = OperatorContract(
        name="relu",
        min_arity=1,
        max_arity=1,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=4,
        derivative_domain="finite inputs away from zero for orders above one; order one uses the PyTorch kink convention",
        residual_policies=(
            ResidualPolicy(
                name="resident_input",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input",),
                proofs=(
                    ProofRef(ProofSystem.LEAN, "OjasXMath.relu_mask_sufficient"),
                    ProofRef(ProofSystem.ROCQ, "Ojasx.Nonlinear.relu_one_bit_sufficient"),
                ),
            ),
            ResidualPolicy(
                name="packed_sign",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input",),
                proofs=(
                    ProofRef(ProofSystem.LEAN, "OjasXMath.relu_mask_sufficient"),
                    ProofRef(ProofSystem.ROCQ, "Ojasx.Nonlinear.relu_one_bit_sufficient"),
                    ProofRef(ProofSystem.Z3, "packed_sign_roundtrip_and_pullback"),
                ),
            ),
        ),
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1 or not inputs[0].dtype.is_floating_point:
            raise ValueError("relu expects one floating input")
        return inputs[0].shape

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        value = inputs[0].value
        if not value.is_meta and inputs[0].order > 1 and (bool(torch.any(value == 0)) or not bool(torch.all(torch.isfinite(value)))):
            raise ValueError("Higher-order ReLU jets require finite inputs away from the kink")
        mask = (value > 0).to(value.dtype)
        return (inputs[0] * mask).set_value(F.relu(value))

    def residual_layout(
        self,
        inputs: Sequence[Value],
        demand: Tuple[bool, ...],
        policy: str,
    ) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        if len(demand) != 1:
            raise ValueError("relu demand arity mismatch")
        if not demand[0]:
            return ()
        if policy == "resident_input":
            return (ResidualBuild("input", "resident_alias", (0,), source_index=0),)
        packed_bytes = (inputs[0].nbytes // inputs[0].dtype.itemsize + 7) // 8
        return (
            ResidualBuild(
                "sign",
                "packed_boolean",
                (0,),
                shape=(packed_bytes,),
                dtype=torch.uint8,
                nbytes=packed_bytes,
            ),
        )

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        self.contract.policy(policy)
        if policy == "resident_input" or not demand[0]:
            return 0
        # Conservative upper bound for the explicit boolean -> uint8 -> padded
        # product/sum pack_mask reference. Native backends replace this with a
        # bounded bit-packing loop and their own checked workspace.
        elements = inputs[0].nbytes // inputs[0].dtype.itemsize
        packed = (elements + 7) // 8
        return int(4 * elements + 16 * packed)

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        self.contract.policy(policy)
        if not demand[0]:
            return 0
        elements = inputs[0].nbytes // inputs[0].dtype.itemsize
        if policy == "resident_input":
            # positive mask, NaN mask, and their Boolean disjunction
            return 3 * elements
        # unpack_mask materializes one uint8 and one Boolean [byte, 8] tensor.
        return 16 * ((elements + 7) // 8)

    def forward(
        self,
        tensors: Sequence[torch.Tensor],
        attrs: Mapping[str, Any],
        policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, 1)
        output = F.relu(tensors[0])
        if not demand[0]:
            return output, {}
        if policy == "resident_input":
            return output, {"input": tensors[0]}
        # Match PyTorch threshold-backward's NaN convention, not merely the
        # extended-real positive predicate: d relu(NaN) is one in PyTorch.
        mask = (tensors[0] > 0) | torch.isnan(tensors[0])
        return output, {"sign": pack_mask(mask)}

    def backward(
        self,
        graph: "JointGraph",
        node: "Node",
        upstream: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if node.residual_policy == "resident_input":
            residual = residuals["input"]
            mask = (residual > 0) | torch.isnan(residual)
        else:
            shape = graph.values[node.forward_inputs[0]].shape
            mask = unpack_mask(residuals["sign"], shape)
        return (torch.where(mask, upstream, 0),)


class AddOperator(UniformCotangentAcceptor, CoreOperator):
    contract = OperatorContract(
        name="add",
        min_arity=2,
        max_arity=2,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=4,
        derivative_domain="all finite real inputs",
        residual_policies=(
            ResidualPolicy(
                name="none",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("left", "right"),
                proofs=(
                    ProofRef(ProofSystem.LEAN, "OjasXMath.core_add_pullback"),
                    ProofRef(ProofSystem.ROCQ, "Ojasx.CoreOperators.core_add_pullback"),
                ),
            ),
        ),
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 2 or inputs[0].shape != inputs[1].shape or inputs[0].dtype != inputs[1].dtype:
            raise ValueError("core add currently requires identical shape and dtype")
        return inputs[0].shape

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0] + inputs[1]

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        return ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(
        self,
        tensors: Sequence[torch.Tensor],
        attrs: Mapping[str, Any],
        policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        return tensors[0] + tensors[1], {}

    def cotangent_encoding(self, graph: "JointGraph", node: "Node", output_position: int) -> str:
        upstream = graph.values[node.inputs[0]]
        source = graph.values[node.forward_inputs[0]]
        if upstream.encoding == UNIFORM_COTANGENT and nonempty(source.shape):
            return UNIFORM_COTANGENT
        return DENSE_COTANGENT

    def backward(
        self,
        graph: "JointGraph",
        node: "Node",
        upstream: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            shape = graph.values[node.outputs[0]].shape
            return tuple(clone_uniform(upstream, shape) for _ in node.gradient_input_indices)
        # Dense outputs own separate storage, matching the conservative model.
        return tuple(upstream.clone() for _ in node.gradient_input_indices)


class SumOperator(UniformCotangentAcceptor, CoreOperator):
    contract = OperatorContract(
        name="sum",
        min_arity=1,
        max_arity=1,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=4,
        derivative_domain="all finite real inputs, including empty reductions",
        residual_policies=(
            ResidualPolicy(
                name="shape_only",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input",),
                proofs=(
                    ProofRef(ProofSystem.LEAN, "OjasXMath.core_sum_pullback"),
                    ProofRef(ProofSystem.ROCQ, "Ojasx.CoreOperators.core_sum_pullback"),
                ),
            ),
        ),
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1 or not inputs[0].dtype.is_floating_point:
            raise ValueError("sum expects one floating input")
        return ()

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0].sum()

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        return ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(
        self,
        tensors: Sequence[torch.Tensor],
        attrs: Mapping[str, Any],
        policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Reduction semantics are layout-independent and do not require a copy.
        _validate_tensors(tensors, require_contiguous=False)
        return tensors[0].sum(), {}

    def cotangent_encoding(self, graph: "JointGraph", node: "Node", output_position: int) -> str:
        source = graph.values[node.forward_inputs[0]]
        return UNIFORM_COTANGENT if nonempty(source.shape) else DENSE_COTANGENT

    def backward(
        self,
        graph: "JointGraph",
        node: "Node",
        upstream: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        source = graph.values[node.forward_inputs[0]]
        if upstream.numel() != 1:
            raise ValueError("sum pullback requires a scalar upstream cotangent")
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            return (clone_uniform(upstream, source.shape),)
        return (upstream.expand(source.shape).clone(),)


def register_core_operators() -> None:
    from .extended import extended_operators

    operators = (LinearOperator(), ReluOperator(), AddOperator(), SumOperator(), *extended_operators())
    for operator in operators:
        if operator.contract.name not in CORE_REGISTRY.names():
            CORE_REGISTRY.register(operator)


__all__ = [
    "AddOperator",
    "LinearOperator",
    "ReluOperator",
    "SumOperator",
    "register_core_operators",
]
