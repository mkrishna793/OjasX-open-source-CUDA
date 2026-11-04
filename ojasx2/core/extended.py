"""Extended certified-core operators with first and fourth-order semantics."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .contracts import (
    AccuracyMode,
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
    divide_uniform,
    nonempty,
)
from .jets import Jet
from .operators import _demand, _validate_tensors
from .activation_math import (
    activation_backward,
    activation_forward,
    activation_jet,
    activation_program,
    cached_replay_workspace_elements,
    streaming_replay_evaluations,
    streaming_replay_workspace_elements,
)


def _contract(
    name: str,
    minimum: int = 1,
    maximum: Optional[int] = None,
    derivative_domain: str = "all finite real inputs",
    proofs: Tuple[ProofRef, ...] = (),
    attribute_names: Tuple[str, ...] = (),
) -> OperatorContract:
    return OperatorContract(
        name=name,
        min_arity=minimum,
        max_arity=minimum if maximum is None else maximum,
        accuracy=AccuracyMode.REPRODUCIBLE,
        deterministic=True,
        max_derivative_order=4,
        residual_policies=(
            ResidualPolicy(
                name="demanded",
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("requested_pullbacks",),
            ),
        ),
        derivative_domain=derivative_domain,
        proofs=proofs,
        attribute_names=attribute_names,
    )


JET_PRODUCT_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.core_normalized_product_order_four"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.CoreOperators.core_normalized_product_order_four"),
    ProofRef(ProofSystem.Z3, "jet_normalized_product_order_four"),
)


def _output_residual() -> ResidualBuild:
    return ResidualBuild("output", "output_alias", (0,), source_index=-1)


def _input_residual(role: str = "input", index: int = 0, slots: Tuple[int, ...] = (0,)) -> ResidualBuild:
    return ResidualBuild(role, "resident_alias", slots, source_index=index)


class UnaryOperator(UniformCotangentAcceptor, CoreOperator):
    OUTPUT_RESIDUAL = {"sigmoid", "tanh", "exp", "sqrt"}
    INPUT_RESIDUAL = {"log", "sin", "cos", "silu", "gelu", "square"}

    def __init__(self, kind: str) -> None:
        if kind not in self.OUTPUT_RESIDUAL | self.INPUT_RESIDUAL | {"neg"}:
            raise ValueError(f"Unknown unary operator {kind}")
        self.kind = kind
        domain = (
            "finite strictly positive inputs"
            if kind in {"log", "sqrt"}
            else "all finite real inputs"
        )
        self.contract = _contract(
            kind,
            derivative_domain=domain,
            proofs=JET_PRODUCT_PROOFS if kind == "square" else (),
            attribute_names=("approximate",) if kind == "gelu" else (),
        )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1 or not inputs[0].dtype.is_floating_point:
            raise ValueError(f"{self.kind} expects one floating input")
        if self.kind == "gelu" and attrs.get("approximate", "none") not in {"none", "tanh"}:
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
        return inputs[0].shape

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        if not demand[0] or self.kind == "neg":
            return ()
        if self.kind in self.OUTPUT_RESIDUAL:
            return (_output_residual(),)
        return (_input_residual(),)

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        if not demand[0] or self.kind == "neg":
            return 0
        return (8 if self.kind in {"gelu", "silu"} else 2) * inputs[0].nbytes

    def _forward_tensor(self, tensor: torch.Tensor, attrs: Mapping[str, Any]) -> torch.Tensor:
        return activation_forward(
            self.kind,
            tensor,
            attrs.get("approximate", "none"),
        )

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, 1)
        output = self._forward_tensor(tensors[0], attrs)
        if not demand[0] or self.kind == "neg":
            return output, {}
        if self.kind in self.OUTPUT_RESIDUAL:
            return output, {"output": output}
        return output, {"input": tensors[0]}

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        result = activation_backward(
            self.kind,
            residuals.get("input"),
            residuals.get("output"),
            upstream,
            dict(node.attrs).get("approximate", "none"),
        )
        return (result,)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return activation_jet(
            inputs[0],
            self.kind,
            attrs.get("approximate", "none"),
        )


class BinaryOperator(UniformCotangentAcceptor, CoreOperator):
    def __init__(self, kind: str) -> None:
        if kind not in {"mul", "div"}:
            raise ValueError(kind)
        self.kind = kind
        domain = (
            "finite inputs with every denominator element nonzero"
            if kind == "div"
            else "all finite real inputs"
        )
        self.contract = _contract(
            kind,
            2,
            derivative_domain=domain,
            proofs=JET_PRODUCT_PROOFS if kind == "mul" else (),
        )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 2 or inputs[0].dtype != inputs[1].dtype:
            raise ValueError(f"{self.kind} expects two same-dtype inputs")
        return tuple(torch.broadcast_shapes(inputs[0].shape, inputs[1].shape))

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        layouts = []
        if self.kind == "mul":
            if demand[0]:
                layouts.append(_input_residual("right", 1, (0,)))
            if demand[1]:
                layouts.append(_input_residual("left", 0, (1,)))
        else:
            if any(demand):
                layouts.append(_input_residual("right", 1, tuple(index for index in (0, 1) if demand[index])))
            if demand[1]:
                layouts.append(ResidualBuild("output", "output_alias", (1,), source_index=-1))
        return tuple(layouts)

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 2 * tensor_nbytes(self.output_shape(inputs, {}), inputs[0].dtype) if any(demand) else 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, 2)
        output = tensors[0] * tensors[1] if self.kind == "mul" else tensors[0] / tensors[1]
        residuals: Dict[str, torch.Tensor] = {}
        if self.kind == "mul":
            if demand[0]:
                residuals["right"] = tensors[1]
            if demand[1]:
                residuals["left"] = tensors[0]
        else:
            if any(demand):
                residuals["right"] = tensors[1]
            if demand[1]:
                residuals["output"] = output
        return output, residuals

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        results = []
        for index in node.gradient_input_indices:
            source = graph.values[node.forward_inputs[index]]
            if self.kind == "mul":
                gradient = upstream * (residuals["right"] if index == 0 else residuals["left"])
            elif index == 0:
                gradient = upstream / residuals["right"]
            else:
                gradient = -upstream * residuals["output"] / residuals["right"]
            results.append(gradient.sum_to_size(source.shape))
        return tuple(results)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0] * inputs[1] if self.kind == "mul" else inputs[0] / inputs[1]


def _matmul_shape(a: Value, b: Value) -> Tuple[int, ...]:
    if a.shape == () or b.shape == ():
        raise ValueError("matmul does not accept scalar operands")
    a_k, b_k = a.shape[-1], b.shape[-2] if len(b.shape) >= 2 else b.shape[0]
    if a_k != b_k:
        raise ValueError("matmul inner dimensions do not match")
    batch = torch.broadcast_shapes(a.shape[:-2] if len(a.shape) >= 2 else (), b.shape[:-2] if len(b.shape) >= 2 else ())
    if len(a.shape) == len(b.shape) == 1:
        return ()
    if len(a.shape) == 1:
        return (*batch, b.shape[-1])
    if len(b.shape) == 1:
        return (*batch, a.shape[-2])
    return (*batch, a.shape[-2], b.shape[-1])


class MatmulOperator(UniformCotangentAcceptor, CoreOperator):
    contract = _contract("matmul", 2)

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 2 or inputs[0].dtype != inputs[1].dtype:
            raise ValueError("matmul expects two same-dtype inputs")
        return _matmul_shape(inputs[0], inputs[1])

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        layouts = []
        if demand[0]:
            layouts.append(_input_residual("right", 1, (0,)))
        if demand[1]:
            layouts.append(_input_residual("left", 0, (1,)))
        return tuple(layouts)

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # torch.matmul defines strided semantics directly. Do not insert an
        # untracked contiguous copy when structured-adjoint rewrites produce
        # transposed operands; other first-order operators retain their
        # contiguous-input contract.
        _validate_tensors(tensors, require_contiguous=False)
        demand = _demand(attrs, 2)
        residuals = {}
        if demand[0]:
            residuals["right"] = tensors[1]
        if demand[1]:
            residuals["left"] = tensors[0]
        return torch.matmul(*tensors), residuals

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        a = graph.values[node.forward_inputs[0]]
        b = graph.values[node.forward_inputs[1]]
        results = []
        for index in node.gradient_input_indices:
            if index == 0:
                right = residuals["right"]
                if len(a.shape) == len(b.shape) == 1:
                    gradient = upstream * right
                elif len(b.shape) == 1:
                    gradient = upstream.unsqueeze(-1) * right
                elif len(a.shape) == 1:
                    gradient = torch.matmul(
                        upstream.unsqueeze(-2), right.transpose(-1, -2)
                    ).squeeze(-2)
                else:
                    gradient = torch.matmul(upstream, right.transpose(-1, -2))
                results.append(gradient.sum_to_size(a.shape))
            else:
                left = residuals["left"]
                if len(a.shape) == len(b.shape) == 1:
                    gradient = upstream * left
                elif len(a.shape) == 1:
                    gradient = left.unsqueeze(-1) * upstream.unsqueeze(-2)
                elif len(b.shape) == 1:
                    gradient = torch.matmul(left.transpose(-1, -2), upstream.unsqueeze(-1)).squeeze(-1)
                else:
                    gradient = torch.matmul(left.transpose(-1, -2), upstream)
                results.append(gradient.sum_to_size(b.shape))
        return tuple(results)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0] @ inputs[1]


class MeanOperator(UniformCotangentAcceptor, CoreOperator):
    contract = _contract(
        "mean",
        derivative_domain="finite inputs with at least one element; empty mean follows the PyTorch NaN convention outside the smooth domain",
    )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1:
            raise ValueError("mean expects one input")
        return ()

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        return ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        return tensors[0].mean(), {}

    def cotangent_encoding(self, graph: "JointGraph", node: "Node", output_position: int) -> str:
        source = graph.values[node.forward_inputs[0]]
        return UNIFORM_COTANGENT if nonempty(source.shape) else DENSE_COTANGENT

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        source = graph.values[node.forward_inputs[0]]
        count = math.prod(source.shape)
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            return (divide_uniform(upstream, count, source.shape),)
        return ((upstream.expand(source.shape).clone() / count),)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0].mean()


class SoftmaxOperator(UniformCotangentAcceptor, CoreOperator):
    def __init__(self, log_mode: bool = False) -> None:
        self.log_mode = log_mode
        self.contract = _contract(
            "log_softmax" if log_mode else "softmax",
            derivative_domain="finite nonscalar inputs with a nonempty normalized dimension",
            attribute_names=("dim",),
        )

    def _dim(self, attrs: Mapping[str, Any]) -> int:
        return int(attrs.get("dim", -1))

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1 or inputs[0].shape == ():
            raise ValueError("softmax requires a nonscalar input")
        dim = self._dim(attrs)
        if not -len(inputs[0].shape) <= dim < len(inputs[0].shape):
            raise ValueError("softmax dimension is out of range")
        if inputs[0].shape[dim] == 0:
            raise ValueError("softmax normalized dimension must be nonempty")
        return inputs[0].shape

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        return (_output_residual(),) if demand[0] else ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 2 * inputs[0].nbytes if demand[0] else 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, 1)
        function = F.log_softmax if self.log_mode else F.softmax
        output = function(tensors[0], dim=self._dim(attrs))
        return output, {"output": output} if demand[0] else {}

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        y = residuals["output"]
        dim = self._dim(dict(node.attrs))
        if self.log_mode:
            result = upstream - y.exp() * upstream.sum(dim=dim, keepdim=True)
        else:
            result = y * (upstream - (upstream * y).sum(dim=dim, keepdim=True))
        return (result,)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        x = inputs[0]
        dim = self._dim(attrs)
        shift = Jet.constant(x.value.amax(dim=dim, keepdim=True), like=x)
        exponential = (x - shift).exp()
        denominator = exponential.sum(dim=dim, keepdim=True)
        result = (x - shift) - denominator.log() if self.log_mode else exponential / denominator
        function = F.log_softmax if self.log_mode else F.softmax
        return result.set_value(function(x.value, dim=dim))


def _epsilon(attrs: Mapping[str, Any]) -> float:
    epsilon = float(attrs.get("eps", 1e-5))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Normalization epsilon must be finite and positive")
    return epsilon


def _normalization_tensor(inputs: Sequence[torch.Tensor], attrs: Mapping[str, Any], rms: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = inputs[0]
    if rms:
        inverse = torch.rsqrt(x.square().mean(-1, keepdim=True) + _epsilon(attrs))
        normalized = x * inverse
    else:
        centered = x - x.mean(-1, keepdim=True)
        inverse = torch.rsqrt(centered.square().mean(-1, keepdim=True) + _epsilon(attrs))
        normalized = centered * inverse
    output = normalized
    if len(inputs) >= 2:
        output = output * inputs[1]
    if len(inputs) == 3:
        output = output + inputs[2]
    return output, normalized, inverse


class NormalizationOperator(UniformCotangentAcceptor, CoreOperator):
    def __init__(self, rms: bool) -> None:
        self.rms = rms
        self.contract = _contract(
            "rms_norm" if rms else "layer_norm",
            1,
            2 if rms else 3,
            derivative_domain="finite inputs with a nonempty last dimension and finite epsilon > 0",
            attribute_names=("eps",),
        )

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if not 1 <= len(inputs) <= (2 if self.rms else 3):
            raise ValueError("Invalid normalization arity")
        x = inputs[0]
        if not x.shape or x.shape[-1] == 0:
            raise ValueError("Normalization requires a nonempty last dimension")
        _epsilon(attrs)
        for parameter in inputs[1:]:
            if parameter.shape != (x.shape[-1],) or parameter.dtype != x.dtype:
                raise ValueError("Normalization parameters must match the last dimension and dtype")
        return x.shape

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        layouts = []
        normalized_slots = tuple(index for index in range(len(inputs)) if demand[index] and index < 2)
        if normalized_slots:
            layouts.append(ResidualBuild("normalized", "dense", normalized_slots, shape=inputs[0].shape, dtype=inputs[0].dtype, nbytes=inputs[0].nbytes))
        if demand[0]:
            inverse_shape = (*inputs[0].shape[:-1], 1)
            layouts.append(ResidualBuild("inverse", "row_dense", (0,), shape=inverse_shape, dtype=inputs[0].dtype, nbytes=tensor_nbytes(inverse_shape, inputs[0].dtype)))
            if len(inputs) >= 2:
                layouts.append(_input_residual("gamma", 1, (0,)))
        return tuple(layouts)

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 6 * inputs[0].nbytes

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 4 * inputs[0].nbytes if demand[0] else inputs[0].nbytes

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        demand = _demand(attrs, len(tensors))
        output, normalized, inverse = _normalization_tensor(tensors, attrs, self.rms)
        residuals: Dict[str, torch.Tensor] = {}
        if any(demand[:2]):
            residuals["normalized"] = normalized
        if demand[0]:
            residuals["inverse"] = inverse
            if len(tensors) >= 2:
                residuals["gamma"] = tensors[1]
        return output, residuals

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        results = []
        for index in node.gradient_input_indices:
            if index == 0:
                z, inverse = residuals["normalized"], residuals["inverse"]
                q = upstream * residuals["gamma"] if len(node.forward_inputs) >= 2 else upstream
                if self.rms:
                    gradient = inverse * (q - z * (q * z).mean(-1, keepdim=True))
                else:
                    gradient = inverse * (q - q.mean(-1, keepdim=True) - z * (q * z).mean(-1, keepdim=True))
            elif index == 1:
                gradient = (upstream * residuals["normalized"]).sum_to_size(graph.values[node.forward_inputs[1]].shape)
            else:
                gradient = upstream.sum_to_size(graph.values[node.forward_inputs[2]].shape)
            results.append(gradient)
        return tuple(results)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        x = inputs[0]
        if self.rms:
            inverse = (x.pow_scalar(2).mean(-1, True) + _epsilon(attrs)).rsqrt()
            normalized = x * inverse
        else:
            centered = x - x.mean(-1, True)
            inverse = (centered.pow_scalar(2).mean(-1, True) + _epsilon(attrs)).rsqrt()
            normalized = centered * inverse
        result = normalized
        if len(inputs) >= 2:
            result = result * inputs[1]
        if len(inputs) == 3:
            result = result + inputs[2]
        primal = _normalization_tensor([jet.value for jet in inputs], attrs, self.rms)[0]
        return result.set_value(primal)


class ReshapeOperator(UniformCotangentAcceptor, CoreOperator):
    contract = _contract("reshape", attribute_names=("shape",))

    def _shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if "shape" not in attrs:
            raise ValueError("reshape requires a shape attribute")
        shape = tuple(int(dimension) for dimension in attrs["shape"])
        if math.prod(shape) != math.prod(inputs[0].shape):
            raise ValueError("reshape must preserve element count")
        return shape

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        return self._shape(inputs, attrs)

    def output_storage(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Optional[int]:
        return 0

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        return ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        return tensors[0].reshape(tuple(attrs["shape"])), {}

    def cotangent_encoding(self, graph: "JointGraph", node: "Node", output_position: int) -> str:
        upstream = graph.values[node.inputs[0]]
        source = graph.values[node.forward_inputs[0]]
        if upstream.encoding == UNIFORM_COTANGENT and nonempty(source.shape):
            return UNIFORM_COTANGENT
        return DENSE_COTANGENT

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        source = graph.values[node.forward_inputs[0]]
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            return (clone_uniform(upstream, source.shape),)
        return (upstream.reshape(source.shape),)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0].reshape(tuple(attrs["shape"]))


class TransposeOperator(UniformCotangentAcceptor, CoreOperator):
    contract = _contract("transpose", attribute_names=("dim0", "dim1"))

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if not inputs[0].shape:
            raise ValueError("transpose expects a nonscalar input")
        ndim = len(inputs[0].shape)
        dim0, dim1 = int(attrs.get("dim0", -2)), int(attrs.get("dim1", -1))
        if not -ndim <= dim0 < ndim or not -ndim <= dim1 < ndim:
            raise ValueError("transpose dimensions are out of range")
        shape = list(inputs[0].shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return tuple(shape)

    def output_storage(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Optional[int]:
        return 0

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        return ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        return 0

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        return tensors[0].transpose(int(attrs.get("dim0", -2)), int(attrs.get("dim1", -1))), {}

    def cotangent_encoding(self, graph: "JointGraph", node: "Node", output_position: int) -> str:
        upstream = graph.values[node.inputs[0]]
        source = graph.values[node.forward_inputs[0]]
        if upstream.encoding == UNIFORM_COTANGENT and nonempty(source.shape):
            return UNIFORM_COTANGENT
        return DENSE_COTANGENT

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        attrs = dict(node.attrs)
        source = graph.values[node.forward_inputs[0]]
        if graph.values[node.outputs[0]].encoding == UNIFORM_COTANGENT:
            return (clone_uniform(upstream, source.shape),)
        return (upstream.transpose(int(attrs.get("dim0", -2)), int(attrs.get("dim1", -1))),)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        return inputs[0].transpose(int(attrs.get("dim0", -2)), int(attrs.get("dim1", -1)))


ACTIVATION_CHAIN_PROOFS = (
    ProofRef(ProofSystem.LEAN, "OjasXMath.m25_replay_tail_dagger"),
    ProofRef(ProofSystem.ROCQ, "Ojasx.M25Activation.m25_replay_tail_pullback"),
    ProofRef(ProofSystem.Z3, "m25_replay_cotangent_chain"),
)


def _streaming_layer_backward(
    original: torch.Tensor,
    program: Sequence[Tuple[str, str]],
    target: int,
    upstream: torch.Tensor,
) -> torch.Tensor:
    """Replay one prefix in an isolated scope, retaining no prefix afterward."""

    value = original
    for index in range(target + 1):
        kind, approximate = program[index]
        output = activation_forward(kind, value, approximate)
        if index == target:
            return activation_backward(kind, value, output, upstream, approximate)
        value = output
    raise RuntimeError("Invalid activation replay index")


class ActivationChainOperator(UniformCotangentAcceptor, CoreOperator):
    """Elementwise composition with one input residual and explicit replay cost."""

    def __init__(self) -> None:
        base = _contract(
            "activation_chain",
            derivative_domain="finite inputs in every composed activation's smooth domain; ReLU kinks excluded above order one",
            proofs=ACTIVATION_CHAIN_PROOFS,
            attribute_names=("operations",),
        )
        policies = tuple(
            ResidualPolicy(
                name=name,
                accuracy=AccuracyMode.REPRODUCIBLE,
                lossless=True,
                satisfies=("input",),
                proofs=ACTIVATION_CHAIN_PROOFS,
            )
            for name in ("replay_streaming", "replay_cached")
        )
        self.contract = replace(base, residual_policies=policies)

    def output_shape(self, inputs: Sequence[Value], attrs: Mapping[str, Any]) -> Tuple[int, ...]:
        if len(inputs) != 1 or not inputs[0].dtype.is_floating_point:
            raise ValueError("activation_chain expects one floating input")
        activation_program(attrs)
        return inputs[0].shape

    def residual_layout(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> Tuple[ResidualBuild, ...]:
        self.contract.policy(policy)
        if len(demand) != 1:
            raise ValueError("activation_chain demand arity mismatch")
        return (_input_residual(),) if demand[0] else ()

    def forward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        self.contract.policy(policy)
        return 2 * inputs[0].nbytes

    def backward_workspace(self, inputs: Sequence[Value], demand: Tuple[bool, ...], policy: str) -> int:
        raise RuntimeError("activation_chain backward workspace requires node attributes")

    def forward(self, tensors: Sequence[torch.Tensor], attrs: Mapping[str, Any], policy: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_tensors(tensors)
        self.contract.policy(policy)
        demand = _demand(attrs, 1)
        value = tensors[0]
        for kind, approximate in activation_program(attrs):
            value = activation_forward(kind, value, approximate)
        return value, {"input": tensors[0]} if demand[0] else {}

    def backward(self, graph: "JointGraph", node: "Node", upstream: torch.Tensor, residuals: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        program = activation_program(dict(node.attrs))
        original = residuals["input"]
        if node.residual_policy == "replay_cached":
            values = [original]
            for kind, approximate in program:
                values.append(activation_forward(kind, values[-1], approximate))
            for index in range(len(program) - 1, -1, -1):
                output = values.pop()
                kind, approximate = program[index]
                upstream = activation_backward(
                    kind,
                    values[-1],
                    output,
                    upstream,
                    approximate,
                )
        else:
            for index in range(len(program) - 1, -1, -1):
                upstream = _streaming_layer_backward(original, program, index, upstream)
        return (upstream,)

    def jet_forward(self, inputs: Sequence[Jet], attrs: Mapping[str, Any]) -> Jet:
        value = inputs[0]
        for kind, approximate in activation_program(attrs):
            value = activation_jet(value, kind, approximate)
        return value

    def replay_evaluations(self, node: "Node") -> int:
        program = activation_program(dict(node.attrs))
        if node.residual_policy == "replay_cached":
            return len(program)
        return streaming_replay_evaluations(program)

    def backward_workspace_for_node(self, inputs: Sequence[Value], node: "Node") -> int:
        if not node.demand[0]:
            return 0
        program = activation_program(dict(node.attrs))
        elements = (
            cached_replay_workspace_elements(program)
            if node.residual_policy == "replay_cached"
            else streaming_replay_workspace_elements(program)
        )
        return elements * inputs[0].nbytes


def extended_operators() -> Tuple[CoreOperator, ...]:
    unary = tuple(UnaryOperator(kind) for kind in ("sigmoid", "tanh", "exp", "log", "sqrt", "sin", "cos", "silu", "gelu", "square", "neg"))
    return (
        *unary,
        ActivationChainOperator(),
        BinaryOperator("mul"),
        BinaryOperator("div"),
        MatmulOperator(),
        MeanOperator(),
        SoftmaxOperator(False),
        SoftmaxOperator(True),
        NormalizationOperator(False),
        NormalizationOperator(True),
        ReshapeOperator(),
        TransposeOperator(),
    )


__all__ = ["extended_operators"]
