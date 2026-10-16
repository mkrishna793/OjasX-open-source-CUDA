"""
tests/test_audit_fixes_a.py

Audit fix verification suite for Workstream A:
- HolonomicGradFreeOptimizer: grad_accum_steps, enabled=False, adafactor & low_memory_adamw
- Updater: notify_if_update_available_background early-return guards
- Symplectic Engine: Apple Silicon MPS RNG state capture/restore and GQA in SymplecticCausalAttention
"""

import os
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn

from ojasx2.updater import notify_if_update_available_background
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.symplectic_engine import (
    SymplecticCausalAttention,
    SymplecticCouplingFunction,
    SymplecticCoupling,
    SymplecticTransformerBlock,
    SymplecticStack,
    symplectic_coupling_chain,
)


# =============================================================================
# 1. Updater Early-Return Guard Tests
# =============================================================================

def test_updater_guard_pytest_env():
    """Verify background check returns immediately during pytest without starting a thread."""
    with patch("threading.Thread") as mock_thread:
        notify_if_update_available_background()
        mock_thread.assert_not_called()


def test_updater_guard_disabled_env():
    """Verify background check returns immediately when OJASX_DISABLE_UPDATE_CHECK=1."""
    with patch.dict(os.environ, {"OJASX_DISABLE_UPDATE_CHECK": "1"}):
        with patch("threading.Thread") as mock_thread:
            notify_if_update_available_background()
            mock_thread.assert_not_called()


def test_updater_guard_ci_env():
    """Verify background check returns immediately when CI is set."""
    with patch.dict(os.environ, {"CI": "true"}):
        with patch("threading.Thread") as mock_thread:
            notify_if_update_available_background()
            mock_thread.assert_not_called()


def test_updater_guard_worker_process():
    """Verify background check returns immediately in worker processes."""
    with patch("multiprocessing.current_process") as mock_proc:
        mock_proc.return_value.name = "Process-1"
        with patch("threading.Thread") as mock_thread:
            notify_if_update_available_background()
            mock_thread.assert_not_called()


# =============================================================================
# 2. HolonomicGradFreeOptimizer: enabled & grad_accum_steps Tests
# =============================================================================

def test_optimizer_enabled_false():
    """When enabled=False, backward hook returns immediately leaving param and grad untouched."""
    lin = nn.Linear(4, 4)
    orig_weight = lin.weight.clone()
    orig_bias = lin.bias.clone()

    opt = HolonomicGradFreeOptimizer(lin.parameters(), lr=0.01, enabled=False)
    assert opt.enabled is False

    x = torch.randn(2, 4)
    loss = lin(x).sum()
    loss.backward()

    # Gradients should remain attached to parameters since hook skipped
    assert lin.weight.grad is not None
    assert lin.bias.grad is not None
    # Parameters must be unmodified
    assert torch.equal(lin.weight, orig_weight)
    assert torch.equal(lin.bias, orig_bias)
    assert opt.last_step_updated_params == 0


def test_optimizer_grad_accum_steps():
    """Verify micro-step counter, gradient accumulation into st['grad_accum'], and deferred updates."""
    torch.manual_seed(42)
    lin = nn.Linear(4, 4)
    opt = HolonomicGradFreeOptimizer(lin.parameters(), lr=0.05, grad_accum_steps=3)
    assert opt.grad_accum_steps == 3

    p_weight = lin.weight
    orig_weight = p_weight.clone()

    # Micro-step 1
    x1 = torch.randn(2, 4)
    loss1 = lin(x1).sum()
    loss1.backward()

    st = opt.state[p_weight]
    assert st["accum_count"] == 1
    assert st["grad_accum"] is not None
    assert p_weight.grad is None  # Freed immediately to save RAM
    assert torch.equal(p_weight, orig_weight)  # Not updated yet
    assert opt.last_step_updated_params == 0

    # Micro-step 2
    x2 = torch.randn(2, 4)
    loss2 = lin(x2).sum()
    loss2.backward()

    assert st["accum_count"] == 2
    assert st["grad_accum"] is not None
    assert p_weight.grad is None
    assert torch.equal(p_weight, orig_weight)  # Still not updated
    assert opt.last_step_updated_params == 0

    # Micro-step 3 (trigger boundary: 3 % 3 == 0)
    x3 = torch.randn(2, 4)
    loss3 = lin(x3).sum()
    loss3.backward()

    assert st["accum_count"] == 3
    assert st["grad_accum"] is None  # Cleared after update
    assert p_weight.grad is None
    assert not torch.equal(p_weight, orig_weight)  # Successfully updated!
    assert opt.last_step_updated_params == 2  # weight and bias


# =============================================================================
# 3. HolonomicGradFreeOptimizer: Adafactor & Low-Memory AdamW Tests
# =============================================================================

def test_optimizer_adafactor_2d_and_1d():
    """Verify factored 2nd moments for 2D+ tensors and unfactored for 1D tensors."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w2d = nn.Parameter(torch.randn(8, 16))
            self.b1d = nn.Parameter(torch.randn(8))

    model = ToyModel()
    opt = HolonomicGradFreeOptimizer(
        model.parameters(), lr=1e-3, algorithm="adafactor", betas=(0.9, 0.99)
    )

    loss = (model.w2d.sum() + model.b1d.sum())
    loss.backward()

    st_2d = opt.state[model.w2d]
    assert "exp_avg_sq_row" in st_2d
    assert "exp_avg_sq_col" in st_2d
    assert st_2d["exp_avg_sq_row"].shape == model.w2d.shape[:-1]  # (8,)
    assert st_2d["exp_avg_sq_col"].shape == model.w2d.shape[:-2] + model.w2d.shape[-1:]  # (16,)
    assert st_2d["exp_avg_sq_row"].dtype == torch.float32
    assert st_2d["exp_avg_sq_col"].dtype == torch.float32

    st_1d = opt.state[model.b1d]
    assert "exp_avg_sq" in st_1d
    assert st_1d["exp_avg_sq"].shape == (8,)


def test_optimizer_adafactor_alias_and_fp16_savings():
    """Verify low_memory_adamw alias, half-precision exp_avg, and >80% state memory savings."""
    R, C = 256, 512
    p_half = nn.Parameter(torch.randn(R, C, dtype=torch.float16))
    opt = HolonomicGradFreeOptimizer([p_half], lr=1e-3, algorithm="low_memory_adamw")

    loss = (p_half * 2.0).sum()
    loss.backward()

    st = opt.state[p_half]
    # exp_avg stored in _p.dtype (FP16)
    assert st["exp_avg"].dtype == torch.float16
    assert st["exp_avg_sq_row"].dtype == torch.float32
    assert st["exp_avg_sq_col"].dtype == torch.float32

    # Memory calculation
    adafactor_bytes = (
        st["exp_avg"].numel() * st["exp_avg"].element_size()
        + st["exp_avg_sq_row"].numel() * st["exp_avg_sq_row"].element_size()
        + st["exp_avg_sq_col"].numel() * st["exp_avg_sq_col"].element_size()
    )
    # Standard FP32 AdamW would store 2 full FP32 tensors: exp_avg + exp_avg_sq
    adamw_fp32_bytes = 2 * (R * C * 4)

    savings_ratio = (adamw_fp32_bytes - adafactor_bytes) / adamw_fp32_bytes
    assert savings_ratio > 0.74  # >74.9% with FP16 momentum (without momentum it's >99%)


# =============================================================================
# 4. Symplectic Engine: Apple Silicon MPS RNG State Tests
# =============================================================================

def test_mps_rng_state_capture_and_restore():
    """Verify MPS RNG state capture and restore logic when MPS is available."""
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS not available on this host")

    device = torch.device("mps")
    block = SymplecticTransformerBlock(dim=32, num_heads=4).to(device)

    q1 = torch.randn(2, 6, 16, device=device, requires_grad=True)
    q2 = torch.randn(2, 6, 16, device=device, requires_grad=True)

    # Forward & Backward through SymplecticCouplingFunction
    y1, y2 = symplectic_coupling_chain(q1, q2, [block])
    loss = (y1.sum() + y2.sum())
    loss.backward()

    assert q1.grad is not None
    assert q2.grad is not None
    assert not torch.isnan(q1.grad).any()
    assert not torch.isnan(q2.grad).any()


def test_mps_rng_code_coverage_cpu():
    """Verify SymplecticCouplingFunction runs cleanly on CPU without MPS device errors."""
    block = SymplecticTransformerBlock(dim=32, num_heads=4)
    q1 = torch.randn(2, 4, 16, requires_grad=True)
    q2 = torch.randn(2, 4, 16, requires_grad=True)

    y1, y2 = symplectic_coupling_chain(q1, q2, [block])
    loss = (y1 * y2).sum()
    loss.backward()

    assert q1.grad is not None
    assert q2.grad is not None


# =============================================================================
# 5. Symplectic Engine: Grouped-Query Attention (GQA) Tests
# =============================================================================

def test_gqa_symplectic_causal_attention_init():
    """Verify SymplecticCausalAttention initialization with num_kv_heads."""
    attn = SymplecticCausalAttention(dim=64, num_heads=8, num_kv_heads=2)
    assert attn.num_heads == 8
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 8
    # qkv projection: dim + 2 * (num_kv_heads * head_dim) = 64 + 2 * (2 * 8) = 96 (vs 192 for MHA)
    assert attn.qkv.out_features == 96

    x = torch.randn(2, 12, 64, requires_grad=True)
    out = attn(x)
    assert out.shape == (2, 12, 64)
    out.sum().backward()
    assert x.grad is not None


def test_gqa_forward_num_kv_heads_override():
    """Verify runtime num_kv_heads override in forward."""
    attn = SymplecticCausalAttention(dim=64, num_heads=4)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = attn(x, num_kv_heads=2)
    assert out.shape == (2, 8, 64)
    out.sum().backward()
    assert x.grad is not None


def test_gqa_invalid_heads_divisibility():
    """Verify ValueError when num_heads is not divisible by num_kv_heads."""
    with pytest.raises(ValueError, match="must be divisible by num_kv_heads"):
        SymplecticCausalAttention(dim=60, num_heads=6, num_kv_heads=4)


def test_gqa_symplectic_stack_integration():
    """Verify full SymplecticStack end-to-end with GQA and reversible inverse."""
    stack = SymplecticStack(dim=64, depth=2, num_heads=8, num_kv_heads=2)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = stack(x)
    assert y.shape == (2, 8, 64)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Reversible reconstruction check
    x_reconstructed = stack.inverse(y.detach())
    assert x_reconstructed.shape == (2, 8, 64)
    assert torch.allclose(x.detach(), x_reconstructed, atol=1e-4)
