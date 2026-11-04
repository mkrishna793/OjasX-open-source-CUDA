"""M1 exact requested-pullback and lossless residual-quotient tests."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from ojasx2.core import JointGraphBuilder, execute_joint_graph, plan_memory


def _run_linear(shape, demand, with_bias=True, out_features=4):
    builder = JointGraphBuilder()
    x = builder.input("x", shape, dtype=torch.float64, requires_grad=demand[0])
    weight = builder.input("w", (out_features, shape[-1]), dtype=torch.float64, requires_grad=demand[1])
    operands = [x, weight]
    tensors = [
        torch.randn(*shape, dtype=torch.float64),
        torch.randn(out_features, shape[-1], dtype=torch.float64),
    ]
    if with_bias:
        bias = builder.input("bias", (out_features,), dtype=torch.float64, requires_grad=demand[2])
        operands.append(bias)
        tensors.append(torch.randn(out_features, dtype=torch.float64))
    output = builder.forward("linear", operands)
    requested = [value for value, needed in zip(operands, demand) if needed]
    for value in requested:
        builder.request_gradient(value)
    builder.set_objective(output)
    graph = builder.generate_reverse()
    certificate = plan_memory(graph)
    result = execute_joint_graph(graph, certificate, dict(zip(operands, tensors)))

    references = [tensor.clone().requires_grad_(True) for tensor in tensors]
    reference = F.linear(*references).sum()
    assert torch.equal(result[graph.objective], reference.detach())
    if requested:
        requested_references = [tensor for tensor, needed in zip(references, demand) if needed]
        expected = torch.autograd.grad(reference, requested_references)
        for value, gradient, tensor in zip(requested, expected, requested_references):
            actual = result[graph.gradient_targets[value]]
            torch.testing.assert_close(actual, gradient, rtol=1e-12, atol=1e-12)
    else:
        assert not [node for node in graph.nodes if node.kind == "pullback"]
    return graph


@pytest.mark.parametrize("shape", [(2, 3), (3,), (0, 3), (2, 0)])
def test_linear_all_gradient_demands(shape):
    for demand in itertools.product((False, True), repeat=3):
        _run_linear(shape, demand)


def test_linear_zero_output_features_all_gradient_demands():
    for demand in itertools.product((False, True), repeat=3):
        _run_linear((2, 3), demand, out_features=0)


def test_linear_without_bias_all_gradient_demands():
    for demand in itertools.product((False, True), repeat=2):
        _run_linear((2, 3), (*demand, False), with_bias=False)


def test_linear_without_bias_and_requested_intermediate_boundary():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    weight = builder.input("w", (4, 3), dtype=torch.float64, requires_grad=True)
    y = builder.forward("linear", (x, weight))
    z = builder.forward("relu", (y,), residual_policy="packed_sign")
    builder.request_gradient(y)
    builder.set_objective(z)
    graph = builder.generate_reverse()

    assert y in graph.gradient_targets
    assert x not in graph.gradient_targets
    assert weight not in graph.gradient_targets
    assert not [node for node in graph.nodes if node.kind == "pullback" and node.op_name == "linear"]
    linear = graph.nodes[0]
    assert linear.demand == (False, False)
    assert linear.residual_ids == ()

    xv = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    wv = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    yv = F.linear(xv, wv)
    reference = F.relu(yv).sum()
    expected, = torch.autograd.grad(reference, yv)
    result = execute_joint_graph(graph, plan_memory(graph), {x: xv.detach(), weight: wv.detach()})
    torch.testing.assert_close(result[graph.gradient_targets[y]], expected)


@pytest.mark.parametrize("shape", [(0,), (1,), (17,), (2, 5)])
@pytest.mark.parametrize("policy", ["resident_input", "packed_sign"])
def test_relu_quotient_matches_torch_with_padding_and_empty_inputs(shape, policy):
    builder = JointGraphBuilder()
    x = builder.input("x", shape, requires_grad=True)
    y = builder.forward("relu", (x,), residual_policy=policy)
    builder.request_gradient(x)
    builder.set_objective(y)
    graph = builder.generate_reverse()
    tensor = torch.linspace(-3, 4, steps=torch.tensor(shape).prod().item()).reshape(shape)
    tensor = tensor.clone().requires_grad_(True)
    reference = tensor.relu().sum()
    expected, = torch.autograd.grad(reference, tensor)
    result = execute_joint_graph(graph, plan_memory(graph), {x: tensor.detach()})
    torch.testing.assert_close(result[graph.objective], reference.detach())
    torch.testing.assert_close(result[graph.gradient_targets[x]], expected)


def test_relu_special_values_and_packed_storage():
    builder = JointGraphBuilder()
    x = builder.input("x", (8,), requires_grad=True)
    y = builder.forward("relu", (x,), residual_policy="packed_sign")
    builder.request_gradient(x)
    builder.set_objective(y)
    graph = builder.generate_reverse()
    residual = graph.values[graph.nodes[0].residual_ids[0]]
    assert residual.nbytes == 1
    assert residual.nbytes * 32 == graph.values[x].nbytes

    tensor = torch.tensor([-0.0, 0.0, 1.0, -2.0, float("nan"), float("inf"), -float("inf"), 3.0])
    reference_tensor = tensor.clone().requires_grad_(True)
    reference = reference_tensor.relu().sum()
    expected, = torch.autograd.grad(reference, reference_tensor)
    result = execute_joint_graph(graph, plan_memory(graph), {x: tensor})
    torch.testing.assert_close(result[graph.objective], reference.detach(), equal_nan=True)
    torch.testing.assert_close(result[graph.gradient_targets[x]], expected, equal_nan=True)


@pytest.mark.parametrize("policy", ["resident_input", "packed_sign"])
@pytest.mark.parametrize("seed", [float("nan"), float("inf"), -0.0])
def test_relu_nonfinite_upstream_matches_pytorch(policy, seed):
    builder = JointGraphBuilder()
    x = builder.input("x", (6,), requires_grad=True)
    output = builder.forward("relu", (x,), residual_policy=policy)
    builder.request_gradient(x)
    builder.set_objective(output)
    graph = builder.generate_reverse()
    tensor = torch.tensor([-1.0, -0.0, 0.0, 1.0, float("nan"), -float("inf")])
    reference_tensor = tensor.clone().requires_grad_(True)
    reference = reference_tensor.relu().sum()
    upstream = torch.tensor(seed)
    expected, = torch.autograd.grad(reference, reference_tensor, grad_outputs=upstream)
    result = execute_joint_graph(
        graph,
        plan_memory(graph),
        {x: tensor},
        objective_upstream=upstream,
    )
    actual = result[graph.gradient_targets[x]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if seed == 0.0:
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_fanout_accumulates_before_shared_producer_pullback():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), requires_grad=True)
    left_weight = builder.input("left_w", (4, 3), requires_grad=True)
    right_weight = builder.input("right_w", (4, 3), requires_grad=True)
    left = builder.forward("linear", (x, left_weight), name="left")
    right = builder.forward("linear", (x, right_weight), name="right")
    joined = builder.forward("add", (left, right))
    activated = builder.forward("relu", (joined,), residual_policy="packed_sign")
    builder.request_input_gradients()
    builder.set_objective(activated)
    graph = builder.generate_reverse()
    assert any(node.kind == "cotangent_add" for node in graph.nodes)

    xt = torch.randn(2, 3)
    lw = torch.randn(4, 3)
    rw = torch.randn(4, 3)
    references = [tensor.clone().requires_grad_(True) for tensor in (xt, lw, rw)]
    xr, lr, rr = references
    reference = F.relu(F.linear(xr, lr) + F.linear(xr, rr)).sum()
    expected = torch.autograd.grad(reference, references)
    result = execute_joint_graph(graph, plan_memory(graph), {x: xt, left_weight: lw, right_weight: rw})
    for value, gradient in zip((x, left_weight, right_weight), expected):
        torch.testing.assert_close(result[graph.gradient_targets[value]], gradient)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_add_and_scalar_reduction_empty_and_scalar_shapes(shape):
    builder = JointGraphBuilder()
    left = builder.input("left", shape, dtype=torch.float64, requires_grad=True)
    right = builder.input("right", shape, dtype=torch.float64, requires_grad=True)
    output = builder.forward("add", (left, right))
    builder.request_input_gradients()
    builder.set_objective(output)
    graph = builder.generate_reverse()
    left_tensor = torch.randn(shape, dtype=torch.float64)
    right_tensor = torch.randn(shape, dtype=torch.float64)
    left_reference = left_tensor.clone().requires_grad_(True)
    right_reference = right_tensor.clone().requires_grad_(True)
    reference = (left_reference + right_reference).sum()
    expected = torch.autograd.grad(reference, (left_reference, right_reference))
    result = execute_joint_graph(
        graph,
        plan_memory(graph),
        {left: left_tensor, right: right_tensor},
    )
    torch.testing.assert_close(result[graph.objective], reference.detach())
    for value, gradient in zip((left, right), expected):
        torch.testing.assert_close(result[graph.gradient_targets[value]], gradient)


def test_nonfinite_linear_matches_pytorch_forward_and_pullbacks():
    builder = JointGraphBuilder()
    x = builder.input("x", (4,), requires_grad=True)
    weight = builder.input("w", (2, 4), requires_grad=True)
    bias = builder.input("b", (2,), requires_grad=True)
    output = builder.forward("linear", (x, weight, bias))
    builder.request_input_gradients()
    builder.set_objective(output)
    graph = builder.generate_reverse()
    tensors = (
        torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0]),
        torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]]),
        torch.tensor([0.0, 1.0]),
    )
    references = [tensor.clone().requires_grad_(True) for tensor in tensors]
    reference = F.linear(*references).sum()
    expected = torch.autograd.grad(reference, references)
    result = execute_joint_graph(graph, plan_memory(graph), dict(zip((x, weight, bias), tensors)))
    torch.testing.assert_close(result[graph.objective], reference.detach(), rtol=0, atol=0, equal_nan=True)
    for value, gradient in zip((x, weight, bias), expected):
        torch.testing.assert_close(
            result[graph.gradient_targets[value]],
            gradient,
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_nonfinite_add_and_reduction_match_pytorch():
    builder = JointGraphBuilder()
    left = builder.input("left", (4,), requires_grad=True)
    right = builder.input("right", (4,), requires_grad=True)
    output = builder.forward("add", (left, right))
    builder.request_input_gradients()
    builder.set_objective(output)
    graph = builder.generate_reverse()
    tensors = (
        torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0]),
        torch.tensor([1.0, float("inf"), 1.0, float("nan")]),
    )
    references = [tensor.clone().requires_grad_(True) for tensor in tensors]
    reference = (references[0] + references[1]).sum()
    expected = torch.autograd.grad(reference, references)
    result = execute_joint_graph(graph, plan_memory(graph), dict(zip((left, right), tensors)))
    torch.testing.assert_close(result[graph.objective], reference.detach(), equal_nan=True)
    for value, gradient in zip((left, right), expected):
        torch.testing.assert_close(result[graph.gradient_targets[value]], gradient)


def test_weight_only_request_does_not_compute_input_or_bias_gradients():
    builder = JointGraphBuilder()
    x = builder.input("x", (2, 3), requires_grad=True)
    weight = builder.input("w", (4, 3), requires_grad=True)
    bias = builder.input("b", (4,), requires_grad=True)
    y = builder.forward("linear", (x, weight, bias))
    builder.request_gradient(weight)
    builder.set_objective(y)
    graph = builder.generate_reverse()
    linear_backward = next(node for node in graph.nodes if node.kind == "pullback" and node.op_name == "linear")
    assert linear_backward.gradient_input_indices == (1,)
    assert graph.nodes[0].demand == (False, True, False)
    assert [graph.values[value_id].role for value_id in graph.nodes[0].residual_ids] == ["x"]
