"""Regression tests for correctness defects found by the full-repository audit."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.numerical_repairs import checked_value_and_grad
from ojasx2.trainer import LowRAMTrainer


def test_checked_value_and_grad_accepts_equivalent_candidate():
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    evidence = []
    value, (grad,) = checked_value_and_grad(
        lambda x: (x * x).sum(),
        lambda x: ((x + 1) * (x - 1)).sum() + x.numel(),
        (x,),
        evidence=evidence,
    )
    torch.testing.assert_close(value, (x * x).sum())
    torch.testing.assert_close(grad, 2 * x)
    assert evidence[-1]["path"] == "checked_candidate"
    assert evidence[-1]["raw_candidate_passed"] is True


def test_checked_value_and_grad_rejects_wrong_candidate_and_returns_native():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    evidence = []
    value, (grad,) = checked_value_and_grad(
        lambda x: (x * x).sum(),
        lambda x: (2 * x * x).sum(),
        (x,),
        evidence=evidence,
    )
    torch.testing.assert_close(value, torch.tensor(5.0))
    torch.testing.assert_close(grad, torch.tensor([2.0, 4.0]))
    assert evidence[-1]["path"] == "checked_native"
    assert evidence[-1]["raw_candidate_passed"] is False


def test_checked_value_and_grad_candidate_error_falls_back():
    x = torch.tensor([2.0], requires_grad=True)

    def bad_candidate(x):
        raise RuntimeError("candidate failure")

    value, (grad,) = checked_value_and_grad(lambda x: x.square().sum(), bad_candidate, (x,))
    torch.testing.assert_close(value, torch.tensor(4.0))
    torch.testing.assert_close(grad, torch.tensor([4.0]))


def test_checked_value_and_grad_higher_order_and_input_validation():
    x = torch.tensor([2.0], requires_grad=True)
    value, (grad,) = checked_value_and_grad(lambda x: (x**3).sum(), lambda x: (x**3).sum(), (x,), create_graph=True)
    second, = torch.autograd.grad(grad.sum(), x)
    torch.testing.assert_close(value, torch.tensor(8.0))
    torch.testing.assert_close(grad, torch.tensor([12.0]))
    torch.testing.assert_close(second, torch.tensor([12.0]))
    with pytest.raises(ValueError, match="scalar tensor loss"):
        checked_value_and_grad(lambda x: (x * 2).repeat(2), lambda x: (x * 2).repeat(2), (x,))
    with pytest.raises(TypeError, match="nonempty tuple of tensors"):
        checked_value_and_grad(lambda: torch.tensor(1.0), lambda: torch.tensor(1.0), ())


def _sgd_optimizer(param, accumulation=2):
    return HolonomicGradFreeOptimizer(
        [param],
        lr=0.1,
        algorithm="sgd",
        weight_decay=0.0,
        momentum=0.0,
        max_grad_norm=None,
        grad_accum_steps=accumulation,
        quantize_moments_e8=False,
    )


def test_accumulation_matches_single_scaled_pytorch_reference():
    actual = torch.tensor([1.0], requires_grad=True)
    expected = torch.tensor([1.0], requires_grad=True)
    opt = _sgd_optimizer(actual)
    ref = torch.optim.SGD([expected], lr=0.1)
    for _ in range(2):
        (2 * actual).backward()  # Optimizer performs the only 1/N scaling.
        (2 * expected / 2).backward()
    ref.step()
    torch.testing.assert_close(actual.detach(), expected.detach())
    torch.testing.assert_close(actual.detach(), torch.tensor([0.8]))
    assert opt.state[actual]["accum_count"] == 0
    assert opt.state[actual]["grad_accum"] is None


def test_partial_accumulation_flush_uses_actual_window_size():
    param = torch.tensor([1.0], requires_grad=True)
    opt = _sgd_optimizer(param)
    (2 * param).backward()
    torch.testing.assert_close(param.detach(), torch.tensor([1.0]))
    assert opt.state[param]["accum_count"] == 1
    assert opt.flush_accumulation() == 1
    torch.testing.assert_close(param.detach(), torch.tensor([0.8]))
    assert opt.state[param]["accum_count"] == 0
    assert opt.flush_accumulation() == 0
    torch.testing.assert_close(param.detach(), torch.tensor([0.8]))


def test_zero_grad_clears_dense_and_packed_accumulation():
    dense = torch.nn.Parameter(torch.ones(8))
    opt = HolonomicGradFreeOptimizer(
        [dense], algorithm="sgd", grad_accum_steps=2, quantize_moments_e8=True
    )
    dense.sum().backward()
    st = opt.state[dense]
    assert "grad_accum_packed" in st
    opt.zero_grad()
    assert st["accum_count"] == 0
    assert st["grad_accum"] is None
    assert "grad_accum_packed" not in st
    assert "grad_accum_scales" not in st
    assert "grad_accum_padded_rows" not in st


def test_trainer_three_microbatches_matches_reference_and_flushes():
    torch.manual_seed(7301)
    model = nn.Linear(2, 1, bias=False)
    reference = nn.Linear(2, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    batches = [
        (torch.tensor([0.5, -1.0]), torch.tensor(0)),
        (torch.tensor([1.5, 0.25]), torch.tensor(0)),
        (torch.tensor([-0.75, 2.0]), torch.tensor(0)),
    ]

    def make_sgd(params, **kwargs):
        return HolonomicGradFreeOptimizer(
            params,
            algorithm="sgd",
            lr=kwargs["lr"],
            weight_decay=0.0,
            max_grad_norm=None,
            grad_accum_steps=kwargs["grad_accum_steps"],
            quantize_moments_e8=False,
        )

    with patch("ojasx2.trainer.HolonomicGradFreeOptimizer", make_sgd):
        trainer = LowRAMTrainer(
            model, lr=0.05, weight_decay=0.0, grad_accum_steps=2,
            quantize_moments=False, fused_loss=False, device="cpu",
        )
    metrics = trainer.train_epoch(batches, log_interval=100)

    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05)
    for window in (batches[:2], batches[2:]):
        ref_opt.zero_grad()
        for x, target in window:
            loss = nn.functional.cross_entropy(reference(x).reshape(1, 1), target.reshape(1))
            (loss / len(window)).backward()
        ref_opt.step()

    torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-6)
    assert metrics["steps"] == 3
    assert metrics["optimizer_steps"] == 2
    assert metrics["flushed_parameter_updates"] == 1
    for param in model.parameters():
        assert trainer.optimizer.state[param]["accum_count"] == 0


@pytest.mark.parametrize("bitrate", [2.0, 1.5])
def test_non_row_divisible_native_layout_uses_correct_fallback(bitrate):
    import ojasx2.math.lattice as lattice

    torch.manual_seed(7400 + int(bitrate * 10))
    layer = lattice.E8LatticeLinear.from_linear(
        nn.Linear(48, 4), group_size=32, bitrate=bitrate, use_hadamard=False
    )
    x = torch.randn(2, 48, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    expected = F.linear(x_ref, layer.unpack_weights(), layer.bias)
    native = SimpleNamespace(
        fused_e8_gemm_fwd=Mock(side_effect=AssertionError("invalid native forward dispatch")),
        fused_e8_gemm_bwd_dx=Mock(side_effect=AssertionError("invalid native backward dispatch")),
        fused_ternary_e8_gemm_fwd=Mock(side_effect=AssertionError("invalid native forward dispatch")),
        fused_ternary_e8_gemm_bwd_dx=Mock(side_effect=AssertionError("invalid native backward dispatch")),
    )
    with patch.object(lattice, "_native", native), patch.object(lattice, "_HAS_NATIVE", True):
        actual = layer.forward_fused(x)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, x_ref.grad)
    native.fused_e8_gemm_fwd.assert_not_called()
    native.fused_e8_gemm_bwd_dx.assert_not_called()
    native.fused_ternary_e8_gemm_fwd.assert_not_called()
    native.fused_ternary_e8_gemm_bwd_dx.assert_not_called()
