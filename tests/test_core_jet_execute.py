"""Scheduled sparse-jet execution checked against mixed PyTorch derivatives."""

import copy
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from ojasx2.core import (
    JointGraphBuilder,
    MemoryBudgetError,
    execute_jet_graph,
    plan_jet_memory,
)
from ojasx2.core.activation_math import activation_forward


def _tensor(shape, low=-0.7, high=0.8):
    return torch.linspace(low, high, math.prod(shape), dtype=torch.float64).reshape(shape)


def _direction(shape, variable):
    values = torch.linspace(0.25, 0.9, math.prod(shape), dtype=torch.float64).reshape(shape)
    return values if variable % 2 == 0 else -0.7 * values


def _case(kind, shapes, attrs=None, ranges=None):
    return kind, shapes, attrs or {}, ranges or [(-0.7, 0.8)] * len(shapes)


CASES = [
    _case("activation_chain", [(7,)], {"operations": ("sigmoid", "tanh", "gelu:tanh")}),
    _case("add", [(2, 3), (2, 3)]),
    _case("linear", [(2, 3), (4, 3), (4,)]),
    _case("relu", [(7,)], ranges=[(-1.2, 1.3)]),
    _case("sum", [(2, 3)]),
    *[
        _case(kind, [(7,)], ranges=[(0.4, 1.3)] if kind in {"log", "sqrt"} else [(-1.2, 1.3)])
        for kind in ("sigmoid", "tanh", "exp", "log", "sqrt", "sin", "cos", "silu", "gelu", "square", "neg")
    ],
    _case("gelu", [(7,)], {"approximate": "tanh"}, [(-1.2, 1.3)]),
    _case("mul", [(2, 3), (3,)]),
    _case("div", [(2, 3), (3,)], ranges=[(-0.7, 0.8), (1.4, 2.1)]),
    _case("matmul", [(2, 2, 3), (1, 3, 4)]),
    _case("matmul", [(3,), (2, 3, 4)]),
    _case("mean", [(2, 3)]),
    _case("softmax", [(2, 3)], {"dim": -1}),
    _case("log_softmax", [(2, 3)], {"dim": 1}),
    _case("layer_norm", [(2, 3)], {"eps": 1e-4}),
    _case("layer_norm", [(2, 3), (3,)], {"eps": 1e-4}),
    _case("layer_norm", [(2, 3), (3,), (3,)], {"eps": 1e-4}),
    _case("rms_norm", [(2, 3)], {"eps": 1e-4}),
    _case("rms_norm", [(2, 3), (3,)], {"eps": 1e-4}),
    _case("reshape", [(2, 3)], {"shape": (3, 2)}),
    _case("transpose", [(2, 3)], {"dim0": 0, "dim1": 1}),
]


def _forward(kind, inputs, attrs):
    if kind == "activation_chain":
        result = inputs[0]
        for operation in attrs["operations"]:
            name, _, approximate = operation.partition(":")
            result = activation_forward(name, result, approximate or "none")
        return result
    if kind == "linear":
        return F.linear(*inputs)
    if kind == "relu":
        return F.relu(inputs[0])
    if kind in {"sigmoid", "tanh", "exp", "log", "sqrt", "sin", "cos", "square", "neg"}:
        return getattr(torch, kind)(inputs[0])
    if kind == "silu":
        return F.silu(inputs[0])
    if kind == "gelu":
        return F.gelu(inputs[0], approximate=attrs.get("approximate", "none"))
    if kind == "add":
        return inputs[0] + inputs[1]
    if kind == "mul":
        return inputs[0] * inputs[1]
    if kind == "div":
        return inputs[0] / inputs[1]
    if kind == "matmul":
        return torch.matmul(*inputs)
    if kind in {"sum", "mean"}:
        return getattr(inputs[0], kind)()
    if kind in {"softmax", "log_softmax"}:
        return getattr(F, kind)(inputs[0], dim=attrs.get("dim", -1))
    if kind == "layer_norm":
        # Torch 2.15's native layer_norm backward is not reliable above order
        # two. This explicit differentiable formula also agrees with central
        # finite differences of the native forward at orders three and four.
        x = inputs[0]
        centered = x - x.mean(-1, keepdim=True)
        result = centered * torch.rsqrt(
            centered.square().mean(-1, keepdim=True) + attrs.get("eps", 1e-5)
        )
        if len(inputs) >= 2:
            result = result * inputs[1]
        if len(inputs) == 3:
            result = result + inputs[2]
        return result
    if kind == "rms_norm":
        return F.rms_norm(
            inputs[0],
            (inputs[0].shape[-1],),
            inputs[1] if len(inputs) == 2 else None,
            attrs.get("eps", 1e-5),
        )
    if kind == "reshape":
        return inputs[0].reshape(attrs["shape"])
    if kind == "transpose":
        return inputs[0].transpose(attrs.get("dim0", -2), attrs.get("dim1", -1))
    raise AssertionError(kind)


def _mixed_directional(objective, references, directions, powers):
    current = objective
    for variable, power in enumerate(powers):
        for _ in range(power):
            if not current.requires_grad:
                return torch.zeros_like(objective)
            gradient = torch.autograd.grad(
                current,
                references[variable],
                grad_outputs=torch.ones_like(current),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is None:
                return torch.zeros_like(objective)
            current = (gradient * directions[variable]).sum()
    return current


@pytest.mark.parametrize("kind,shapes,attrs,ranges", CASES)
def test_every_operator_matches_all_mixed_directionals_through_order_four(kind, shapes, attrs, ranges):
    builder = JointGraphBuilder()
    sources = tuple(
        builder.input(f"x{index}", shape, dtype=torch.float64, requires_grad=True)
        for index, shape in enumerate(shapes)
    )
    output = builder.forward(kind, sources, attrs=attrs)
    builder.set_objective(output)
    specialized, certificate = plan_jet_memory(builder.graph, order=4)
    tensors = tuple(_tensor(shape, *interval) for shape, interval in zip(shapes, ranges))
    directions = {source: _direction(shape, index) for index, (source, shape) in enumerate(zip(sources, shapes))}
    result = execute_jet_graph(
        specialized,
        certificate,
        dict(zip(sources, tensors)),
        directions=directions,
    )[specialized.objective]

    references = tuple(tensor.clone().requires_grad_(True) for tensor in tensors)
    reference_directions = tuple(directions[source] for source in sources)
    objective = _forward(kind, references, attrs).sum()
    torch.testing.assert_close(result.value, objective.detach(), rtol=2e-10, atol=2e-10)
    for order in range(1, 5):
        for powers in itertools.product(range(order + 1), repeat=len(sources)):
            if sum(powers) != order:
                continue
            expected = _mixed_directional(objective, references, reference_directions, powers)
            actual = result.derivative(powers)
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6, msg=f"Mixed powers: {powers}")


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_requested_order_is_exact(order):
    builder = JointGraphBuilder()
    x = builder.input("x", (3,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("exp", (x,)))
    specialized, certificate = plan_jet_memory(builder.graph, order=order)
    value = _tensor((3,))
    result = execute_jet_graph(specialized, certificate, {x: value})[specialized.objective]
    assert result.order == order
    assert all(sum(power for _, power in index) <= order for index in result.coefficients)
    with pytest.raises(ValueError, match="exceeds"):
        result.derivative((order + 1,))


def test_jet_planning_rejects_invalid_order_and_generated_graph():
    builder = JointGraphBuilder()
    x = builder.input("x", (3,), dtype=torch.float64, requires_grad=True)
    output = builder.forward("exp", (x,))
    builder.set_objective(output)
    for order in (0, 5, True):
        with pytest.raises(ValueError, match="order"):
            plan_jet_memory(builder.graph, order=order)
    builder.request_gradient(x)
    with pytest.raises(ValueError, match="ungenerated|primal"):
        plan_jet_memory(builder.generate_reverse(), order=4)


def test_jet_budget_and_digest_binding():
    builder = JointGraphBuilder()
    x = builder.input("x", (3,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("exp", (x,)))
    specialized, certificate = plan_jet_memory(builder.graph, order=4)
    certificate.validate(specialized)
    assert certificate.optimal_for_managed_model
    assert certificate.peak_bytes > 0
    with pytest.raises(MemoryBudgetError) as error:
        plan_jet_memory(builder.graph, order=4, budget=certificate.peak_bytes - 1)
    assert error.value.required == certificate.peak_bytes

    tampered = copy.deepcopy(specialized)
    tampered.derivative_order = 3
    with pytest.raises(ValueError, match="different graph"):
        execute_jet_graph(tampered, certificate, {x: torch.ones(3, dtype=torch.float64)})


def test_variable_input_subset_and_direction_validation():
    builder = JointGraphBuilder()
    x = builder.input("x", (3,), dtype=torch.float64, requires_grad=True)
    y = builder.input("y", (3,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("mul", (x, y)))
    specialized, certificate = plan_jet_memory(builder.graph, order=4, variable_inputs=(x,))
    xv = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    yv = torch.tensor([1.1, 1.4, 1.8], dtype=torch.float64)
    direction = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    result = execute_jet_graph(
        specialized,
        certificate,
        {x: xv, y: yv},
        directions={x: direction},
    )[specialized.objective]
    torch.testing.assert_close(result.derivative((1,)), (direction * yv).sum())
    assert result.derivative((2,)) == 0
    with pytest.raises(ValueError, match="constant or unknown"):
        execute_jet_graph(specialized, certificate, {x: xv, y: yv}, directions={y: direction})
    with pytest.raises(ValueError, match="unique external"):
        plan_jet_memory(builder.graph, variable_inputs=(x, x))


def test_aliased_inputs_share_primal_storage_but_have_independent_directions():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    alias = builder.input("alias", (2, 3), dtype=torch.float64, requires_grad=True, alias_of=x)
    builder.set_objective(builder.forward("add", (x, alias)))
    specialized, certificate = plan_jet_memory(builder.graph, order=4)
    value = _tensor((2, 3))
    result = execute_jet_graph(specialized, certificate, {x: value, alias: value})[specialized.objective]
    torch.testing.assert_close(result.value, 2 * value.sum())
    torch.testing.assert_close(result.directional_derivative(1), torch.tensor(12.0, dtype=torch.float64))
    assert result.directional_derivative(2) == 0
    with pytest.raises(ValueError, match="declared as aliases"):
        execute_jet_graph(specialized, certificate, {x: value, alias: value.clone()})


def test_meta_specialization_distinguishes_view_and_copy_reshape():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    transposed = builder.forward("transpose", (x,))
    copied = builder.forward("reshape", (transposed,), attrs={"shape": (6,)})
    builder.set_objective(copied)
    specialized, certificate = plan_jet_memory(builder.graph, order=4)
    assert specialized.values[copied].storage_id != specialized.values[transposed].storage_id
    assert specialized.values[copied].storage_bytes > 0
    certificate.validate(specialized)
    value = _tensor((2, 3))
    result = execute_jet_graph(specialized, certificate, {x: value})[specialized.objective]
    torch.testing.assert_close(result.value, value.sum())
    torch.testing.assert_close(result.derivative((1,)), torch.tensor(6.0, dtype=torch.float64))

    view_builder = JointGraphBuilder()
    source = view_builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    viewed = view_builder.forward("reshape", (source,), attrs={"shape": (3, 2)})
    view_builder.set_objective(viewed)
    view_graph, view_certificate = plan_jet_memory(view_builder.graph, order=4)
    assert view_graph.values[viewed].storage_id == view_graph.values[source].storage_id
    view_certificate.validate(view_graph)


@pytest.mark.parametrize(
    "kind,value,order,message",
    [
        ("log", 0.0, 4, "strictly positive"),
        ("log", -1.0, 4, "strictly positive"),
        ("sqrt", 0.0, 4, "strictly positive"),
        ("relu", 0.0, 2, "away from the kink"),
    ],
)
def test_smooth_domain_violations_are_rejected(kind, value, order, message):
    builder = JointGraphBuilder()
    x = builder.input("x", (1,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward(kind, (x,)))
    specialized, certificate = plan_jet_memory(builder.graph, order=order)
    with pytest.raises(ValueError, match=message):
        execute_jet_graph(specialized, certificate, {x: torch.tensor([value], dtype=torch.float64)})


def test_first_order_relu_kink_and_negative_division_domain():
    builder = JointGraphBuilder()
    x = builder.input("x", (3,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("relu", (x,)))
    specialized, certificate = plan_jet_memory(builder.graph, order=1)
    result = execute_jet_graph(
        specialized,
        certificate,
        {x: torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)},
    )[specialized.objective]
    assert result.value == 2
    assert result.derivative((1,)) == 1

    division = JointGraphBuilder()
    numerator = division.input("numerator", (2,), dtype=torch.float64, requires_grad=True)
    denominator = division.input("denominator", (2,), dtype=torch.float64, requires_grad=True)
    division.set_objective(division.forward("div", (numerator, denominator)))
    div_graph, div_certificate = plan_jet_memory(division.graph, order=4)
    values = {
        numerator: torch.tensor([0.3, 0.5], dtype=torch.float64),
        denominator: torch.tensor([-1.5, -2.0], dtype=torch.float64),
    }
    div_result = execute_jet_graph(div_graph, div_certificate, values)[div_graph.objective]
    torch.testing.assert_close(div_result.value, (values[numerator] / values[denominator]).sum())
    with pytest.raises(ValueError, match="nonzero"):
        execute_jet_graph(
            div_graph,
            div_certificate,
            {numerator: values[numerator], denominator: torch.tensor([-1.5, 0.0], dtype=torch.float64)},
        )


def test_noncontiguous_direction_is_rejected():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 2), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("exp", (x,)))
    specialized, certificate = plan_jet_memory(builder.graph)
    value = _tensor((2, 2))
    with pytest.raises(ValueError, match="contiguous directions"):
        execute_jet_graph(
            specialized,
            certificate,
            {x: value},
            directions={x: value.transpose(0, 1)},
        )


def test_higher_order_nonfinite_partitions_are_rejected_explicitly():
    builder = JointGraphBuilder()
    x = builder.input("x", (1,), dtype=torch.float64, requires_grad=True)
    builder.set_objective(builder.forward("exp", (x,)))
    specialized, certificate = plan_jet_memory(builder.graph, order=2)
    finite = torch.tensor([0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="finite inputs"):
        execute_jet_graph(
            specialized,
            certificate,
            {x: torch.tensor([float("nan")], dtype=torch.float64)},
        )
    with pytest.raises(ValueError, match="finite directions"):
        execute_jet_graph(
            specialized,
            certificate,
            {x: finite},
            directions={x: torch.tensor([float("inf")], dtype=torch.float64)},
        )
    with pytest.raises(ValueError, match="Nonfinite jet coefficient"):
        execute_jet_graph(
            specialized,
            certificate,
            {x: torch.tensor([1000.0], dtype=torch.float64)},
        )
