import copy
import os
from pathlib import Path
import sysconfig

import pytest
import torch
import torch.nn.functional as F

from ojasx2.core import activation_chain, operator_schemas
from ojasx2.core.activation_math import ACTIVATION_NAMES, activation_forward
from ojasx2.residuals import SavedTensorAudit


POLICIES = ("replay_streaming", "replay_cached")
PROGRAMS = tuple(ACTIVATION_NAMES) + ("gelu:tanh",)


def _reference(x, operations):
    for operation in operations:
        kind, _, approximate = operation.partition(":")
        x = activation_forward(kind, x, approximate or "none")
    return x


def _value(operation, requires_grad=True):
    if operation in {"log", "sqrt"}:
        value = torch.linspace(0.15, 0.85, 12, dtype=torch.float64)
    elif operation == "exp":
        value = torch.linspace(-0.7, 0.6, 12, dtype=torch.float64)
    else:
        value = torch.linspace(-0.45, 0.55, 12, dtype=torch.float64)
    return value.requires_grad_(requires_grad)


@pytest.mark.parametrize("replay", POLICIES)
@pytest.mark.parametrize("operation", PROGRAMS)
def test_registered_chain_matches_eager_forward_and_native_gradient(operation, replay):
    operations = [operation, "sigmoid"]
    x = _value(operation)
    dy = torch.linspace(-0.4, 0.35, x.numel(), dtype=x.dtype)
    actual = activation_chain(x, operations, replay)
    expected = _reference(x, operations)
    assert torch.equal(actual, expected)
    gradient = torch.autograd.grad(actual, x, dy, retain_graph=True)[0]
    reference = torch.autograd.grad(expected, x, dy)[0]
    torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("replay", POLICIES)
def test_official_custom_op_contract(replay):
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.ojasx_verified.activation_chain.default,
        (x, ["sigmoid", "gelu:tanh"], replay),
    )
    assert set(result.values()) == {"SUCCESS"}
    assert len(operator_schemas()) == 2


def test_meta_and_noncontiguous_fake_behavior():
    x = torch.empty(3, 7, device="meta")
    y = activation_chain(x, ["sigmoid", "tanh"])
    assert y.device.type == "meta" and y.shape == x.shape and y.dtype == x.dtype
    upstream = torch.empty_like(x)
    gradient = torch.ops.ojasx_verified.activation_chain_backward(
        x, upstream, ["sigmoid", "tanh"], "replay_streaming"
    )
    assert gradient.device.type == "meta" and gradient.shape == x.shape

    real = torch.randn(5, 7, dtype=torch.float64).T.requires_grad_()
    output = activation_chain(real, ["sigmoid", "gelu:tanh"])
    expected = _reference(real, ["sigmoid", "gelu:tanh"])
    assert output.stride() == expected.stride()
    dy = torch.randn_like(output)
    torch.testing.assert_close(
        torch.autograd.grad(output, real, dy, retain_graph=True)[0],
        torch.autograd.grad(expected, real, dy)[0],
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize("replay", POLICIES)
def test_aot_compiled_forward_backward_and_dynamic_shapes(replay):
    operations = ["sigmoid", "gelu:tanh", "sin"]

    def function(x):
        return activation_chain(x, operations, replay).square().sum()

    compiled = torch.compile(function, backend="aot_eager", fullgraph=True)
    for shape in ((2, 3), (4, 3)):
        x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = compiled(x)
        expected = _reference(reference_x, operations).square().sum()
        assert torch.equal(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, x)[0],
            torch.autograd.grad(expected, reference_x)[0],
            rtol=2e-11,
            atol=2e-11,
        )


def test_export_contains_registered_operator():
    class Program(torch.nn.Module):
        def forward(self, x):
            return activation_chain(x, ["sigmoid", "tanh"])

    exported = torch.export.export(Program(), (torch.randn(2, 4),))
    x = torch.randn(2, 4)
    assert torch.equal(exported.module()(x), _reference(x, ["sigmoid", "tanh"]))
    graph = str(exported.graph)
    assert "ojasx_verified.activation_chain" in graph


def test_only_original_input_is_saved_and_mutation_is_rejected():
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    with SavedTensorAudit([x]) as audit:
        output = activation_chain(x, ["sigmoid"] * 4)
    summary = audit.summary()
    assert summary["saved_tensors"] == 1
    assert summary["fresh_saved_storage_bytes"] == 0

    # Identity saved_tensors_hooks can change PyTorch's normal saved-version
    # behavior, so audit storage and mutation safety with separate graphs.
    mutated = x.detach().clone().requires_grad_()
    output = activation_chain(mutated, ["sigmoid"] * 4)
    with torch.no_grad():
        mutated.add_(0.125)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_operator_does_not_consume_rng_and_compiled_random_regions_match():
    x = torch.randn(3, 4)
    torch.manual_seed(81)
    state = torch.get_rng_state()
    activation_chain(x, ["sigmoid", "tanh"])
    assert torch.equal(torch.get_rng_state(), state)

    def random_program(value):
        before = torch.rand_like(value)
        middle = activation_chain(value, ["sigmoid", "tanh"])
        after = torch.rand_like(value)
        return before, middle, after

    compiled = torch.compile(random_program, backend="aot_eager", fullgraph=True)
    torch.manual_seed(991)
    expected = random_program(x)
    torch.manual_seed(991)
    actual = compiled(x)
    for got, reference in zip(actual, expected):
        assert torch.equal(got, reference)


@pytest.mark.parametrize("operation", PROGRAMS)
def test_cpu_autocast_matches_unfused_program(operation):
    x = _value(operation).to(torch.float32)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = activation_chain(x, [operation])
        expected = _reference(reference_x, [operation])
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    dy = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, dy, retain_graph=True)[0],
        torch.autograd.grad(expected, reference_x, dy)[0],
        rtol=0.05,
        atol=0.05,
    )


def test_compiled_autocast_linear_boundary():
    weight = torch.randn(4, 3)

    def function(x, weight):
        hidden = F.linear(x, weight)
        return activation_chain(hidden, ["sigmoid", "tanh"])

    compiled = torch.compile(function, backend="aot_eager", fullgraph=True)
    x = torch.randn(2, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compiled(x, weight)
        expected = function(reference_x, weight)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), reference_x)[0],
        rtol=0.05,
        atol=0.05,
    )


def test_gradient_accumulation_matches_reference():
    operations = ["gelu:tanh", "sigmoid", "tanh"]
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    upstreams = [torch.randn(4, 5, dtype=torch.float64) for _ in range(2)]
    actual = activation_chain(x, operations)
    expected = _reference(reference_x, operations)
    for index, upstream in enumerate(upstreams):
        torch.autograd.backward(actual, upstream / 2, retain_graph=index == 0)
        torch.autograd.backward(expected, upstream / 2, retain_graph=index == 0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-11, atol=2e-11)


class _TrajectoryModel(torch.nn.Module):
    def __init__(self, operations, fused=True):
        super().__init__()
        torch.manual_seed(1234)
        self.input_projection = torch.nn.Linear(4, 6, dtype=torch.float64)
        self.output_projection = torch.nn.Linear(6, 3, dtype=torch.float64)
        self.operations = list(operations)
        self.fused = fused

    def forward(self, x):
        hidden = self.input_projection(x)
        if self.fused:
            hidden = activation_chain(hidden, self.operations)
        else:
            hidden = _reference(hidden, self.operations)
        return self.output_projection(hidden)


def _optimizer(name, parameters):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=0.03, momentum=0.8, weight_decay=0.01)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=0.01, weight_decay=0.01)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.02)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=0.01, momentum=0.5, weight_decay=0.01)
    raise ValueError(name)


def _assert_same_parameters(actual, expected):
    for got, reference in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(got, reference, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("optimizer_name", ("sgd", "adam", "adamw", "rmsprop"))
def test_standard_optimizer_multi_step_trajectories(optimizer_name):
    operations = ["gelu:tanh", "sigmoid"]
    actual = _TrajectoryModel(operations, fused=True)
    expected = copy.deepcopy(actual)
    expected.fused = False
    actual_optimizer = _optimizer(optimizer_name, actual.parameters())
    expected_optimizer = _optimizer(optimizer_name, expected.parameters())
    for step in range(4):
        torch.manual_seed(500 + step)
        x = torch.randn(5, 4, dtype=torch.float64)
        target = torch.randn(5, 3, dtype=torch.float64)
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        actual_loss = (actual(x) - target).square().mean()
        expected_loss = (expected(x) - target).square().mean()
        if step == 0:
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-10, atol=2e-10)
        actual_loss.backward()
        expected_loss.backward()
        for got, reference in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(got.grad, reference.grad, rtol=2e-10, atol=2e-10)
        actual_optimizer.step()
        expected_optimizer.step()
        _assert_same_parameters(actual, expected)


def test_compiled_training_and_accumulated_optimizer_step():
    operations = ["sigmoid", "tanh", "sin"]
    actual = _TrajectoryModel(operations, fused=True)
    expected = copy.deepcopy(actual)
    expected.fused = False
    compiled = torch.compile(actual, backend="aot_eager", fullgraph=True)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=0.01)
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=0.01)
    for step in range(3):
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        for microstep in range(2):
            torch.manual_seed(800 + step * 2 + microstep)
            x = torch.randn(3, 4, dtype=torch.float64)
            target = torch.randn(3, 3, dtype=torch.float64)
            actual_loss = (compiled(x) - target).square().mean() / 2
            expected_loss = (expected(x) - target).square().mean() / 2
            actual_loss.backward()
            expected_loss.backward()
        for got, reference in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(got.grad, reference.grad, rtol=2e-10, atol=2e-10)
        actual_optimizer.step()
        expected_optimizer.step()
        _assert_same_parameters(actual, expected)


def test_invalid_programs_and_backward_contract_fail_explicitly():
    with pytest.raises(ValueError, match="Unsupported"):
        activation_chain(torch.randn(2), ["not_an_activation"])
    with pytest.raises(ValueError, match="replay"):
        activation_chain(torch.randn(2), ["sigmoid"], "unknown")
    with pytest.raises(TypeError, match="floating"):
        activation_chain(torch.ones(2, dtype=torch.int64), ["sigmoid"])
    with pytest.raises(ValueError, match="matching dtype and device"):
        torch.ops.ojasx_verified.activation_chain_backward(
            torch.ones(2), torch.ones(2, dtype=torch.float64), ["sigmoid"], "replay_cached"
        )


@pytest.mark.skipif(
    not Path(sysconfig.get_paths()["include"], "Python.h").exists()
    and not os.environ.get("CPATH"),
    reason="Inductor requires CPython development headers; AOT compilation is tested separately",
)
def test_default_inductor_forward_backward():
    def chain(x):
        return activation_chain(x, ["sigmoid", "tanh"], "replay_cached")

    def function(x):
        return chain(x).square().sum()

    compiled_chain = torch.compile(chain, backend="inductor", fullgraph=True)
    compiled = torch.compile(function, backend="inductor", fullgraph=True)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()

    # The registered operator itself remains an exact eager call under Inductor.
    torch.testing.assert_close(compiled_chain(x), chain(reference_x), rtol=1e-12, atol=1e-12)
    actual = compiled(x)
    expected = function(reference_x)
    # Inductor may emit a different, but still float64-accurate, outer reduction.
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x)[0],
        torch.autograd.grad(expected, reference_x)[0],
        rtol=1e-12,
        atol=1e-12,
    )
