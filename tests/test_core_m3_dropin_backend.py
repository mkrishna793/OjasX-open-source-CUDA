"""Unchanged-model Dynamo/AOT backend and global-planner integration tests."""

import copy
import os
from pathlib import Path
import sysconfig

import pytest
import torch
import torch.nn.functional as F

from ojasx2.core import (
    BackendCompilationReport,
    BackendConfig,
    ExecutionObjective,
    HardwareTargetProfile,
    LogicalCost,
    M4Config,
    OjasXBackend,
    detect_hardware_target,
)


def _chain(value, count=4):
    for _ in range(count):
        value = torch.sigmoid(value)
    return value


def _scalar_program(value):
    return _chain(value).square().mean()


def _assert_close(actual, expected, tolerance=2e-11):
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)


def test_named_backend_retains_original_on_tie_and_executes_forward_backward():
    def function(value):
        return torch.tanh(torch.sigmoid(value)).square().mean()

    backend = OjasXBackend()
    compiled = torch.compile(function, backend="ojasx_verified", fullgraph=True)
    # Use a separate instance for report inspection; the named registration is
    # exercised above, while this instance runs the same unchanged function.
    inspected = torch.compile(function, backend=backend, fullgraph=True)
    for executable in (compiled, inspected):
        value = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64, requires_grad=True)
        reference = value.detach().clone().requires_grad_()
        actual = executable(value)
        expected = function(reference)
        assert torch.equal(actual, expected)
        actual.backward()
        expected.backward()
        assert torch.equal(value.grad, reference.grad)
    assert not backend.reports[-1].changed
    assert backend.reports[-1].regions[0].reason == "original_not_strictly_worse"


def test_backend_discovers_and_executes_a_proof_backed_schedule():
    backend = OjasXBackend()
    compiled = torch.compile(_scalar_program, backend=backend, fullgraph=True)
    value = torch.linspace(-0.3, 0.4, 8, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = compiled(value)
    expected = _scalar_program(reference)
    assert torch.equal(actual, expected)
    actual.backward()
    expected.backward()
    _assert_close(value.grad, reference.grad)
    report = backend.reports[-1]
    assert report.changed
    assert any(region.replaced and region.proof_obligations > 0 for region in report.regions)
    assert any(
        region.selected_cost["memory_bytes"] < region.baseline_cost["memory_bytes"]
        for region in report.regions
    )


def test_backend_requested_objective_controls_original_fallback():
    planner = M4Config(
        max_iterations=8,
        exact_node_limit=32,
        objective=ExecutionObjective(primary="work"),
    )
    backend = OjasXBackend(BackendConfig(planner=planner))

    def function(value):
        return _chain(value, count=4).square().mean()

    memory_backend = OjasXBackend()
    memory_compiled = torch.compile(function, backend=memory_backend, fullgraph=True)
    memory_value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    memory_compiled(memory_value)
    assert memory_backend.reports[-1].regions[0].replaced

    compiled = torch.compile(function, backend=backend, fullgraph=True)
    value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = compiled(value)
    expected = function(reference)
    _assert_close(actual, expected)
    actual.backward()
    expected.backward()
    _assert_close(value.grad, reference.grad)
    region = backend.reports[-1].regions[0]
    assert not region.replaced
    assert region.selected_cost["work_units"] == region.baseline_cost["work_units"]


def test_unsupported_nodes_fall_back_while_supported_neighbors_are_optimized():
    def mixed(value):
        hidden = _chain(value)
        ordered = torch.sort(hidden).values
        return ordered.square().mean()

    backend = OjasXBackend()
    compiled = torch.compile(mixed, backend=backend, fullgraph=True)
    value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = compiled(value)
    expected = mixed(reference)
    _assert_close(actual, expected)
    actual.backward()
    expected.backward()
    _assert_close(value.grad, reference.grad)
    assert backend.reports[-1].changed

    def unsupported_only(value):
        ordered, indices = torch.sort(torch.sigmoid(value))
        return ordered.sum() + indices.float().sum()

    fallback = OjasXBackend()
    compiled_fallback = torch.compile(unsupported_only, backend=fallback, fullgraph=True)
    actual = compiled_fallback(value.detach())
    expected = unsupported_only(value.detach())
    _assert_close(actual, expected)
    assert not fallback.reports[-1].changed
    assert not fallback.reports[-1].regions


def test_compiled_random_regions_preserve_rng_and_schedule_replacement():
    def random_program(value):
        before = torch.rand_like(value)
        hidden = _chain(value + before)
        return before, hidden, torch.rand_like(hidden)

    backend = OjasXBackend()
    compiled = torch.compile(random_program, backend=backend, fullgraph=True)
    value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(991)
    expected = random_program(value)
    torch.manual_seed(991)
    actual = compiled(value)
    assert all(torch.equal(got, reference) for got, reference in zip(actual, expected))
    assert backend.reports[-1].changed


def test_cpu_autocast_linear_boundary_and_gradients():
    def function(value, weight):
        hidden = F.linear(value, weight)
        return _chain(hidden).square().mean()

    backend = OjasXBackend()
    compiled = torch.compile(function, backend=backend, fullgraph=True)
    value = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compiled(value, weight)
        expected = function(reference_value, reference_weight)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=0.05, atol=0.05)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0.05, atol=0.05)
    assert backend.reports[-1].changed


def test_dynamic_meta_and_noncontiguous_execution():
    backend = OjasXBackend()
    compiled = torch.compile(_scalar_program, backend=backend, fullgraph=True, dynamic=True)
    for shape in ((2, 3), (4, 3)):
        value = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        reference = value.detach().clone().requires_grad_()
        actual = compiled(value)
        expected = _scalar_program(reference)
        _assert_close(actual, expected)
        actual.backward()
        expected.backward()
        _assert_close(value.grad, reference.grad)

    meta = compiled(torch.empty(2, 3, dtype=torch.float64, device="meta"))
    assert meta.device.type == "meta" and meta.shape == ()

    real = torch.randn(4, 6, dtype=torch.float64).T.requires_grad_()
    reference = real.detach().clone().requires_grad_()
    actual = compiled(real)
    expected = _scalar_program(reference)
    _assert_close(actual, expected)
    actual.backward()
    expected.backward()
    _assert_close(real.grad, reference.grad)


def test_input_mutation_behavior_matches_native_segmented_autograd():
    backend = OjasXBackend()
    compiled = torch.compile(_scalar_program, backend=backend, fullgraph=True)
    value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    original = value.detach().clone()
    actual = compiled(value)
    expected = _scalar_program(reference)
    assert torch.equal(value.detach(), original)
    with torch.no_grad():
        value.add_(0.125)
        reference.add_(0.125)
    # Native sigmoid saves outputs; the selected segmented schedule has the
    # same mutation behavior. A custom chain that directly saves an input is
    # separately tested to reject stale-version replay.
    actual.backward()
    expected.backward()
    _assert_close(value.grad, reference.grad)


class _NativeTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.input_projection = torch.nn.Linear(4, 6, dtype=torch.float64)
        self.output_projection = torch.nn.Linear(6, 3, dtype=torch.float64)

    def forward(self, value):
        hidden = self.input_projection(value)
        hidden = _chain(hidden)
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


@pytest.mark.parametrize("optimizer_name", ("sgd", "adam", "adamw", "rmsprop"))
def test_complete_model_accumulated_optimizer_training(optimizer_name):
    actual_model = _NativeTrainingModel()
    expected_model = copy.deepcopy(actual_model)
    backend = OjasXBackend()
    compiled = torch.compile(actual_model, backend=backend, fullgraph=True)
    actual_optimizer = _optimizer(optimizer_name, actual_model.parameters())
    expected_optimizer = _optimizer(optimizer_name, expected_model.parameters())

    for step in range(2):
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        for microstep in range(2):
            torch.manual_seed(500 + step * 2 + microstep)
            value = torch.randn(5, 4, dtype=torch.float64)
            target = torch.randn(5, 3, dtype=torch.float64)
            actual_loss = (compiled(value) - target).square().mean() / 2
            expected_loss = (expected_model(value) - target).square().mean() / 2
            _assert_close(actual_loss, expected_loss)
            actual_loss.backward()
            expected_loss.backward()
        for actual, expected in zip(actual_model.parameters(), expected_model.parameters()):
            _assert_close(actual.grad, expected.grad, tolerance=2e-10)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(actual_model.parameters(), expected_model.parameters()):
            _assert_close(actual, expected, tolerance=2e-9)
    assert any(report.changed for report in backend.reports)


@pytest.mark.skipif(
    not Path(sysconfig.get_paths()["include"], "Python.h").exists()
    and not os.environ.get("CPATH"),
    reason="Inductor requires CPython development headers; portable AOT is tested separately",
)
def test_inductor_inner_compiler_forward_backward():
    backend = OjasXBackend(BackendConfig(use_inductor=True))
    compiled = torch.compile(_scalar_program, backend=backend, fullgraph=True)
    value = torch.linspace(-0.3, 0.4, 8, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = compiled(value)
    expected = _scalar_program(reference)
    _assert_close(actual, expected, tolerance=1e-12)
    actual.backward()
    expected.backward()
    _assert_close(value.grad, reference.grad, tolerance=1e-12)
    assert backend.reports[-1].changed


def test_detect_hardware_target_profiles_all_architectures():
    auto_profile = detect_hardware_target()
    assert isinstance(auto_profile, HardwareTargetProfile)
    assert auto_profile.device_type in {"cuda", "rocm", "mps", "xpu", "cpu"}
    assert auto_profile.to_dict()["device_type"] == auto_profile.device_type

    cuda_profile = detect_hardware_target("cuda:0")
    assert cuda_profile.device_type == "cuda"
    assert cuda_profile.architecture == "nvidia_cuda"
    assert cuda_profile.preferred_executor == "tilelang"
    assert cuda_profile.supports_inductor and cuda_profile.supports_tilelang
    assert not cuda_profile.prefer_aot_boxed
    assert cuda_profile.vector_width_bits >= 512
    assert cuda_profile.sram_budget_bytes > 0

    for rocm_name in ("rocm", "hip"):
        rocm_profile = detect_hardware_target(rocm_name)
        assert rocm_profile.device_type == "rocm"
        assert rocm_profile.architecture == "amd_rocm"
        assert rocm_profile.preferred_executor == "tilelang"
        assert rocm_profile.supports_tilelang

    mps_profile = detect_hardware_target("mps")
    assert mps_profile.device_type == "mps"
    assert mps_profile.architecture == "apple_silicon_mps"
    assert mps_profile.simd_isa == "metal_simd"
    assert mps_profile.sram_budget_bytes == 32768

    xpu_profile = detect_hardware_target("xpu")
    assert xpu_profile.device_type == "xpu"
    assert xpu_profile.preferred_executor == "inductor"
    assert xpu_profile.supports_inductor and not xpu_profile.supports_tilelang

    openreg_profile = detect_hardware_target("openreg")
    assert openreg_profile.device_type == "openreg"
    assert openreg_profile.preferred_executor == "aot_boxed"
    assert openreg_profile.prefer_aot_boxed

    avx2_profile = detect_hardware_target("cpu:avx2")
    assert avx2_profile.device_type == "cpu"
    assert avx2_profile.simd_isa == "avx2"
    assert avx2_profile.vector_width_bits == 256

    neon_profile = detect_hardware_target("cpu:neon")
    assert neon_profile.device_type == "cpu"
    assert neon_profile.simd_isa == "neon"
    assert neon_profile.vector_width_bits == 128

    meta_tensor = torch.empty(4, device="meta")
    meta_profile = detect_hardware_target(meta_tensor)
    assert meta_profile.device_type == "meta"
    assert meta_profile.preferred_executor == "aot_boxed"
    assert meta_profile.sram_budget_bytes == 0


def test_backend_hardware_autodetection_and_compilation_report_fields():
    legacy_report = BackendCompilationReport(changed=False, regions=(), original_nodes=("x",))
    assert legacy_report.hardware_profile is None

    cost = LogicalCost(memory_bytes=4096, work_units=10, traffic_bytes=4096, final_bytes=1024)
    assert cost.fits_sram_budget(32768)
    assert not cost.fits_sram_budget(2048)

    def hw_program(value):
        return _chain(value, count=4).square().mean()

    backend_cpu = OjasXBackend(
        BackendConfig(target_device="cuda", auto_hardware=True, enable_tilelang=False)
    )
    compiled = torch.compile(hw_program, backend=backend_cpu, fullgraph=True)
    value = torch.linspace(-0.3, 0.4, 8, dtype=torch.float64, requires_grad=True)
    actual = compiled(value)
    expected = hw_program(value.detach().clone().requires_grad_())
    _assert_close(actual, expected)
    report = backend_cpu.reports[-1]
    assert report.changed
    assert report.hardware_profile is not None
    assert report.hardware_profile["device_type"] == "cuda"
    assert report.hardware_profile["supports_tilelang"] is False
    assert report.hardware_profile["preferred_executor"] == "inductor"

    auto_backend = OjasXBackend(BackendConfig(auto_hardware=True))
    auto_compiled = torch.compile(hw_program, backend=auto_backend, fullgraph=True)
    cpu_val = torch.linspace(-0.2, 0.3, 8, dtype=torch.float64, requires_grad=True)
    auto_compiled(cpu_val)
    auto_report = auto_backend.reports[-1]
    assert auto_report.hardware_profile is not None
    assert auto_report.hardware_profile["device_type"] == "cpu"

