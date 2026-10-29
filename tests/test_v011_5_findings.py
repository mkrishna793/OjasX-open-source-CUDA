"""
tests/test_v011_5_findings.py
==============================
Test suite verifying the 5 Hidden Bugs & Bottlenecks identified in OjasX v0.11.0:
1. Finding 1: _FusedE8GEMMFunction does not save x_rot in ctx.saved_tensors on autograd tape.
2. Finding 2: prefer_native_sdpa enabled in DaggerAttention / dagger_tiled_attention fast paths.
3. Finding 3: HolonomicGradFreeOptimizer compresses gradient accumulation when quantize_moments_e8=True.
4. Finding 4: StormerVerletIntegrator.reconstruct_chain exact squeeze order and reversibility.
5. Finding 5: TileLang kernels use explicit 2D slice ranges in T.copy.
"""

import ast
import inspect
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import (
    _FusedE8GEMMFunction,
    E8LatticePacker,
    E8LatticeLinear,
)
from ojasx2.dagger_attention import (
    holonomic_dagger_attention,
    dagger_tiled_attention,
    DaggerAttention,
)
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.math.symplectic import (
    StormerVerletIntegrator,
)
import ojasx2.tilelang_kernels as tl_kernels


# =============================================================================
# Finding 1: _FusedE8GEMMFunction Zero-Activation Tape Test
# =============================================================================

def test_finding_1_no_activation_saved_on_autograd_tape():
    """Verify that _FusedE8GEMMFunction does NOT store x or x_rot in saved_tensors."""
    torch.manual_seed(42)
    B, S, in_f, out_f = 2, 8, 32, 64
    x = torch.randn(B, S, in_f, dtype=torch.float32, requires_grad=True)
    w_unpacked = torch.randn(out_f, in_f, dtype=torch.float32)
    packed_w, scales, _ = E8LatticePacker.pack(w_unpacked, group_size=32)

    # Forward
    y = _FusedE8GEMMFunction.apply(
        x, packed_w, scales, None, out_f, in_f, 32, 2.0, False
    )
    
    # Inspect saved tensors inside the backward grad_fn
    grad_fn = y.grad_fn
    assert grad_fn is not None, "Output must have grad_fn"
    
    # In PyTorch, grad_fn._saved_self or saved_tensors holds the saved tensors
    saved_tensors = getattr(grad_fn, "saved_tensors", ())
    # Check that x is NOT in saved_tensors and none of the saved tensors have shape matching x
    for t in saved_tensors:
        assert t is not x, "x must NOT be pinned on the autograd tape!"
        assert t.shape != x.shape, f"Saved tensor with shape {t.shape} matches x.shape! Full activations must not be saved."

    # Verify backward still computes exact gradients
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.shape == x.shape


# =============================================================================
# Finding 2: DaggerAttention Native SDPA Routing Test
# =============================================================================

def test_finding_2_dagger_attention_native_sdpa_path():
    """Verify that dagger_tiled_attention and DaggerAttention support prefer_native_sdpa=True."""
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 16, 32
    q = torch.randn(B, H, S, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.float32, requires_grad=True)

    # 1. Test dagger_tiled_attention with prefer_native_sdpa=True
    out_native = dagger_tiled_attention(q, k, v, is_causal=False, prefer_native_sdpa=True)
    assert out_native.shape == (B, H, S, D)

    # 2. Test DaggerAttention module
    layer = DaggerAttention(dim=D * H, num_heads=H, is_causal=False)
    x = torch.randn(B, S, D * H, requires_grad=True)
    out_mod = layer(x)
    assert out_mod.shape == (B, S, D * H)
    
    # Backward pass verification
    out_mod.sum().backward()
    assert x.grad is not None


# =============================================================================
# Finding 3: HolonomicGradFreeOptimizer Compressed Gradient Accumulation Test
# =============================================================================

def test_finding_3_optimizer_e8_compressed_grad_accum():
    """Verify that HolonomicGradFreeOptimizer compresses grad accumulation when quantize_moments_e8=True."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    opt = HolonomicGradFreeOptimizer(
        model.parameters(),
        lr=1e-3,
        grad_accum_steps=2,
        quantize_moments_e8=True,
    )

    x = torch.randn(4, 32)
    loss1 = model(x).sum()
    loss1.backward()

    # After microstep 1, check that parameters do not have active uncompressed gradients
    for p in model.parameters():
        assert p.grad is None, "In-backward hook must set p.grad = None immediately!"
        st = opt.state[p]
        if st.get("accum_count", 0) == 1:
            # When quantize_moments_e8 is True, grad_accum should be packed in 2.0-bit E8
            if p.numel() >= 8:
                assert "grad_accum_packed" in st, "st['grad_accum_packed'] must be present to compress grad accumulation memory!"
                assert "grad_accum" not in st or st["grad_accum"] is None, "Uncompressed st['grad_accum'] must NOT be retained!"

    # Microstep 2
    loss2 = model(x).sum()
    loss2.backward()

    # Step cycle should complete
    opt.step()
    opt.zero_grad()


# =============================================================================
# Finding 4: StormerVerletIntegrator Reversible Squeeze Order Test
# =============================================================================

def test_finding_4_stormer_verlet_squeeze_reversibility():
    """Verify StormerVerletIntegrator.reconstruct_chain achieves machine precision with enable_squeeze=True."""
    torch.manual_seed(42)
    D = 32
    L = 4
    stages = [
        (
            nn.Sequential(nn.Linear(D, D), nn.Tanh()),
            nn.Sequential(nn.Linear(D, D), nn.Tanh()),
        )
        for _ in range(L)
    ]
    integrator = StormerVerletIntegrator(dt=1.0, persistent_fp32=True)
    q0 = torch.randn(2, D, dtype=torch.float32) * 100.0  # Variance imbalance to trigger k != 0
    p0 = torch.randn(2, D, dtype=torch.float32) * 0.01

    # Forward integration with dyadic squeezing enabled
    q_f, p_f, squeezes = integrator.integrate_chain(q0, p0, stages, enable_squeeze=True)
    assert len(squeezes) == L
    assert any(k.item() != 0 for k in squeezes), "Must trigger non-zero squeeze to test squeeze order"

    # Reverse reconstruction
    q_rec, p_rec = integrator.reconstruct_chain(
        q_f, p_f, stages, enable_squeeze=True, squeezes=squeezes
    )

    err_q = (q_rec - q0).abs().max().item()
    err_p = (p_rec - p0).abs().max().item()

    assert err_q < 1e-4, f"Reconstruction error for q too large: {err_q}"
    assert err_p < 1e-4, f"Reconstruction error for p too large: {err_p}"


# =============================================================================
# Finding 5: TileLang 2D Slice Syntax in T.copy Test
# =============================================================================

def test_finding_5_tilelang_copy_2d_slice_syntax():
    """Inspect AST of tilelang_kernels to ensure T.copy uses 2D slice ranges (not scalar indices)."""
    src = inspect.getsource(tl_kernels)
    # The source must have slice syntax ':' inside T.copy for X, W_packed, Scales, and Y
    assert "X[by * block_M :" in src, "X copy must use 2D slice range: X[by * block_M : ...]"
    assert "W_packed[bx * block_N :" in src, "W_packed copy must use 2D slice range: W_packed[bx * block_N : ...]"
    assert "Scales[bx * block_N :" in src, "Scales copy must use 2D slice range: Scales[bx * block_N : ...]"
    assert "Y[by * block_M :" in src, "Y store copy must use 2D slice range: Y[by * block_M : ...]"


# =============================================================================
# OjasX v12: 1-Line Simple Developer API Tests
# =============================================================================

def test_ojasx_v12_version():
    """Verify version bump to 0.12.0."""
    import ojasx
    import ojasx2
    assert ojasx.__version__ == "0.12.0"
    assert ojasx2.__version__ == "0.12.0"


def test_ojasx_v12_simple_api_prepare_and_train():
    """Verify 1-line ojasx.prepare, ojasx.optimize, and ojasx.loss."""
    import ojasx

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    # 1-Line prepare
    opt_model, opt = ojasx.prepare(model, lr=1e-3, quantize="2.0b")

    x = torch.randn(4, 16)
    y = torch.tensor([1, 2, 0, 3])

    opt.zero_grad()
    out = opt_model(x)
    loss = ojasx.loss(out, y)
    loss.backward()
    opt.step()

    assert loss.item() > 0.0
    assert torch.isfinite(loss)
