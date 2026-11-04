"""Differential coverage for the expanded 25-operator first-order core."""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from ojasx2.core import JointGraphBuilder, execute_joint_graph, plan_memory


def _compare_requested(builder, sources, tensors, reference_function, *, rtol=1e-10, atol=1e-10):
    for source in sources:
        builder.request_gradient(source)
    graph = builder.generate_reverse()
    result = execute_joint_graph(graph, plan_memory(graph), dict(zip(sources, tensors)))
    references = [tensor.clone().requires_grad_(True) for tensor in tensors]
    reference = reference_function(*references)
    expected = torch.autograd.grad(reference, references)
    torch.testing.assert_close(result[graph.objective], reference.detach(), rtol=rtol, atol=atol, equal_nan=True)
    for source, gradient in zip(sources, expected):
        torch.testing.assert_close(result[graph.gradient_targets[source]], gradient, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("kind", ["sigmoid", "tanh", "exp", "log", "sqrt", "sin", "cos", "silu", "gelu", "square", "neg"])
def test_extended_unary_first_order(kind):
    builder = JointGraphBuilder()
    x = builder.input("x", (7,), dtype=torch.float64, requires_grad=True)
    output = builder.forward(kind, (x,))
    builder.set_objective(output)
    value = torch.linspace(0.4 if kind in {"log", "sqrt"} else -1.2, 1.3, 7, dtype=torch.float64)
    functions = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "sin": torch.sin,
        "cos": torch.cos,
        "silu": F.silu,
        "gelu": F.gelu,
        "square": torch.square,
        "neg": torch.neg,
    }
    _compare_requested(builder, (x,), (value,), lambda x: functions[kind](x).sum())


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_approximations_first_order(approximate):
    builder = JointGraphBuilder()
    x = builder.input("x", (7,), dtype=torch.float64, requires_grad=True)
    output = builder.forward("gelu", (x,), attrs={"approximate": approximate})
    builder.set_objective(output)
    value = torch.linspace(-1.2, 1.3, 7, dtype=torch.float64)
    _compare_requested(
        builder,
        (x,),
        (value,),
        lambda x: F.gelu(x, approximate=approximate).sum(),
    )


@pytest.mark.parametrize("kind", ["mul", "div"])
@pytest.mark.parametrize("left_shape,right_shape", [((2, 3), (3,)), ((3,), (2, 3)), ((), (2, 3))])
def test_broadcast_binary_all_demands(kind, left_shape, right_shape):
    for demand in itertools.product((False, True), repeat=2):
        builder = JointGraphBuilder()
        left = builder.input("left", left_shape, dtype=torch.float64, requires_grad=demand[0])
        right = builder.input("right", right_shape, dtype=torch.float64, requires_grad=demand[1])
        output = builder.forward(kind, (left, right))
        requested = tuple(value for value, needed in zip((left, right), demand) if needed)
        builder.set_objective(output)
        if not requested:
            graph = builder.generate_reverse()
            result = execute_joint_graph(
                graph,
                plan_memory(graph),
                {
                    left: torch.ones(left_shape, dtype=torch.float64),
                    right: torch.ones(right_shape, dtype=torch.float64),
                },
            )
            assert graph.objective in result
            continue
        left_tensor = torch.linspace(0.5, 1.5, math.prod(left_shape), dtype=torch.float64).reshape(left_shape)
        right_tensor = torch.linspace(1.5, 2.5, math.prod(right_shape), dtype=torch.float64).reshape(right_shape)
        sources, tensors = (left, right), (left_tensor, right_tensor)
        for source in requested:
            builder.request_gradient(source)
        graph = builder.generate_reverse()
        result = execute_joint_graph(graph, plan_memory(graph), dict(zip(sources, tensors)))
        references = [tensor.clone().requires_grad_(True) for tensor in tensors]
        reference = (references[0] * references[1] if kind == "mul" else references[0] / references[1]).sum()
        expected = torch.autograd.grad(reference, [reference for reference, needed in zip(references, demand) if needed])
        for source, gradient in zip(requested, expected):
            torch.testing.assert_close(result[graph.gradient_targets[source]], gradient)


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((3,), (3,)),
        ((3,), (3, 4)),
        ((2, 3), (3,)),
        ((2, 3), (3, 4)),
        ((2, 2, 3), (3, 4)),
        ((3,), (2, 3, 4)),
        ((2, 2, 3), (1, 3, 4)),
    ],
)
def test_matmul_vector_matrix_and_broadcast_demands(left_shape, right_shape):
    for demand in itertools.product((False, True), repeat=2):
        if not any(demand):
            continue
        builder = JointGraphBuilder()
        left = builder.input("left", left_shape, dtype=torch.float64, requires_grad=demand[0])
        right = builder.input("right", right_shape, dtype=torch.float64, requires_grad=demand[1])
        output = builder.forward("matmul", (left, right))
        requested = tuple(value for value, needed in zip((left, right), demand) if needed)
        for source in requested:
            builder.request_gradient(source)
        builder.set_objective(output)
        graph = builder.generate_reverse()
        tensors = (
            torch.randn(left_shape, dtype=torch.float64),
            torch.randn(right_shape, dtype=torch.float64),
        )
        result = execute_joint_graph(graph, plan_memory(graph), {left: tensors[0], right: tensors[1]})
        references = [tensor.clone().requires_grad_(True) for tensor in tensors]
        reference = torch.matmul(*references).sum()
        expected = torch.autograd.grad(reference, [reference for reference, needed in zip(references, demand) if needed])
        for source, gradient in zip(requested, expected):
            torch.testing.assert_close(result[graph.gradient_targets[source]], gradient, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_mean_first_order(shape):
    builder = JointGraphBuilder()
    x = builder.input("x", shape, dtype=torch.float64, requires_grad=True)
    output = builder.forward("mean", (x,))
    builder.set_objective(output)
    value = torch.randn(shape, dtype=torch.float64)
    _compare_requested(builder, (x,), (value,), lambda x: x.mean())


@pytest.mark.parametrize("kind", ["softmax", "log_softmax"])
def test_softmax_family_first_order(kind):
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    output = builder.forward(kind, (x,), attrs={"dim": -1})
    builder.set_objective(output)
    value = torch.randn(2, 3, dtype=torch.float64)
    function = F.log_softmax if kind == "log_softmax" else F.softmax
    _compare_requested(builder, (x,), (value,), lambda x: function(x, dim=-1).sum())


@pytest.mark.parametrize(
    "rms,arity",
    [(False, 1), (False, 2), (False, 3), (True, 1), (True, 2)],
)
def test_normalization_all_demands(rms, arity):
    for demand in itertools.product((False, True), repeat=arity):
        if not any(demand):
            continue
        builder = JointGraphBuilder()
        x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=demand[0])
        sources = [x]
        if arity >= 2:
            sources.append(builder.input("gamma", (3,), dtype=torch.float64, requires_grad=demand[1]))
        if arity == 3:
            sources.append(builder.input("beta", (3,), dtype=torch.float64, requires_grad=demand[2]))
        output = builder.forward("rms_norm" if rms else "layer_norm", sources, attrs={"eps": 1e-5})
        requested = tuple(value for value, needed in zip(sources, demand) if needed)
        for source in requested:
            builder.request_gradient(source)
        builder.set_objective(output)
        graph = builder.generate_reverse()
        tensors = [torch.randn(2, 3, dtype=torch.float64)] + [torch.randn(3, dtype=torch.float64) for _ in sources[1:]]
        result = execute_joint_graph(graph, plan_memory(graph), dict(zip(sources, tensors)))
        references = [tensor.clone().requires_grad_(True) for tensor in tensors]
        if rms:
            reference = F.rms_norm(references[0], (3,), references[1] if arity == 2 else None, 1e-5).sum()
        else:
            reference = F.layer_norm(
                references[0],
                (3,),
                references[1] if arity >= 2 else None,
                references[2] if arity == 3 else None,
                1e-5,
            ).sum()
        expected = torch.autograd.grad(reference, [reference for reference, needed in zip(references, demand) if needed])
        torch.testing.assert_close(result[graph.objective], reference.detach(), rtol=1e-9, atol=1e-9)
        for source, gradient in zip(requested, expected):
            torch.testing.assert_close(result[graph.gradient_targets[source]], gradient, rtol=1e-8, atol=1e-8)


def test_normalization_without_affine():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    output = builder.forward("layer_norm", (x,))
    builder.request_gradient(x)
    builder.set_objective(output)
    graph = builder.generate_reverse()
    value = torch.randn(2, 3, dtype=torch.float64)
    result = execute_joint_graph(graph, plan_memory(graph), {x: value})
    reference_value = value.clone().requires_grad_(True)
    reference = F.layer_norm(reference_value, (3,)).sum()
    expected, = torch.autograd.grad(reference, reference_value)
    torch.testing.assert_close(result[graph.gradient_targets[x]], expected, rtol=1e-8, atol=1e-8)


def test_reshape_and_transpose_pullbacks():
    for kind in ("reshape", "transpose"):
        builder = JointGraphBuilder()
        x = builder.input("x", (1, 3), dtype=torch.float64, requires_grad=True)
        attrs = {"shape": (3, 1)} if kind == "reshape" else {"dim0": 0, "dim1": 1}
        output = builder.forward(kind, (x,), attrs=attrs)
        builder.request_gradient(x)
        builder.set_objective(output)
        graph = builder.generate_reverse()
        value = torch.randn(1, 3, dtype=torch.float64)
        result = execute_joint_graph(graph, plan_memory(graph), {x: value})
        reference_value = value.clone().requires_grad_(True)
        transformed = reference_value.reshape(3, 1) if kind == "reshape" else reference_value.transpose(0, 1)
        reference = transformed.sum()
        expected, = torch.autograd.grad(reference, reference_value)
        torch.testing.assert_close(result[graph.gradient_targets[x]], expected)


def test_unknown_operator_attribute_is_rejected():
    builder = JointGraphBuilder()
    x = builder.input("x", (2,), dtype=torch.float64)
    with pytest.raises(ValueError, match="Unsupported attributes for exp"):
        builder.forward("exp", (x,), attrs={"dim": 0})


def test_first_order_rejects_noncontiguous_operator_input_explicitly():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64)
    transposed = builder.forward("transpose", (x,))
    squared = builder.forward("square", (transposed,))
    builder.set_objective(squared)
    graph = builder.generate_reverse()
    with pytest.raises(ValueError, match="contiguous"):
        execute_joint_graph(graph, plan_memory(graph), {x: torch.randn(2, 3, dtype=torch.float64)})
