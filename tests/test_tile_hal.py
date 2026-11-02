"""Comprehensive unit tests for OjasX TileLang Hardware HAL & Vector XOR Swizzling.

Verifies:
1. Vector-width-aware XOR bank swizzling for 128-bit vector types (`float4` / `uint4`, 16 bytes per thread).
2. Formal Z3 SMT prover for vector XOR swizzling (`verify_vector_xor_swizzle_smt`), proving 0 bank conflicts.
3. Scalar XOR swizzle and 1-bit ballot formal SMT verification (`verify_xor_swizzle_and_ballot_smt`).
4. Hardware silicon capability prober and cache schedule planner (decode vs prefill).
5. 1-bit hardware ballot packing and unpacking (32x compression).
6. AMX fused linear + ReLU2 forward and backward.
7. Fused L1-resident RMSNorm single-pass forward and backward.
8. TileLang 2D CPU tile decomposition and Metal adaptive micro-tiling.
9. Multi-dtype support (float32, float16, bfloat16) and non-contiguous views.
10. PyTorch official contract certification via `torch.library.opcheck`.
11. Universal L1-resident Gated Linear Unit (11 activation modes parity).
"""

from __future__ import annotations

import os
import time
import pytest
import torch
import torch.nn.functional as F

import ojasx2
import ojasx2.tile_hal as tile_hal
from ojasx2.tile_hal import (
    xor_swizzle_vector_offset,
    verify_vector_xor_swizzle_smt,
    verify_xor_swizzle_and_ballot_smt,
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# =============================================================================
# 1. Vector-Width-Aware XOR Bank Swizzling & Z3 SMT Formal Prover Tests
# =============================================================================

def test_xor_swizzle_vector_offset_formula_and_bijection():
    """Verify vector XOR swizzle formula: vec_col = col >> 2, bank index = ((vec_col ^ row) << 2) | (col & 3)."""
    num_banks = 32
    vec_width = 4  # 128-bit float4 / uint4 (4 32-bit words)

    for r in range(32):
        banks_seen = set()
        for c in range(32):
            vec_col = c >> 2
            expected_bank = (((vec_col ^ r) & 7) << 2) | (c & 3)

            # Test offset calculation
            computed_bank = xor_swizzle_vector_offset(r, c, num_banks=num_banks, vec_width=vec_width)
            assert computed_bank == expected_bank, f"Mismatch at row {r}, col {c}: {computed_bank} vs {expected_bank}"
            banks_seen.add(computed_bank % num_banks)

        # Every row must access strictly all 32 banks without collision
        assert len(banks_seen) == 32, f"Row {r} only accessed {len(banks_seen)} distinct banks instead of 32!"

    # Test stride calculation
    stride = 64
    off_strided = xor_swizzle_vector_offset(2, 5, num_banks=32, vec_width=4, stride=stride)
    expected_strided = 2 * stride + xor_swizzle_vector_offset(2, 5, num_banks=32, vec_width=4)
    assert off_strided == expected_strided


def test_verify_vector_xor_swizzle_smt_proof():
    """Verify formal Z3 SMT prover for 128-bit vector XOR swizzling.

    Proves:
    1. Inter-row conflict freedom: Any two distinct rows access disjoint banks (unsat).
    2. Intra-warp conflict freedom: Warp-level 32-bank access hits all 32 banks uniquely (unsat).
    """
    t0 = time.perf_counter()
    res = verify_vector_xor_swizzle_smt(vec_width=4)
    t1 = time.perf_counter()

    assert res["z3_status"] == "unsat", f"Expected z3_status 'unsat', got {res['z3_status']}"
    assert res["zero_bank_conflicts_proven"] is True
    assert res["inter_row_conflict_free"] is True
    assert res["intra_warp_conflict_free"] is True
    assert res["inter_row_unsat"] == "unsat"
    assert res["intra_warp_unsat"] == "unsat"
    assert res["vector_swizzle_bijection_verified"] is True
    assert res["num_banks"] == 32
    assert res["vec_width"] == 4
    assert res["proof_duration_sec"] < 2.0  # Formal proof finishes in milliseconds
    print(f"\n[Z3 Vector XOR Swizzle SMT] Proof duration: {res['proof_duration_sec']*1000:.2f}ms (Status: UNSAT)")


def test_verify_scalar_xor_swizzle_smt():
    """Verify scalar XOR swizzle and 1-bit ballot formal SMT verification."""
    res = verify_xor_swizzle_and_ballot_smt()
    assert res["zero_bank_conflicts_proven"] is True
    assert res["z3_status"] == "unsat"
    assert res["cpp_swizzle_bijection_verified"] is True
    assert res["num_banks"] == 32


# =============================================================================
# 2. Hardware Silicon Capability Prober & Tile Planner Tests
# =============================================================================

def test_silicon_capability_prober():
    """Verify hardware prober reads physical CPU cache, cores, and AMX capability."""
    profile = tile_hal.probe_hardware_profile()
    assert profile.l1d_cache_bytes > 0
    assert profile.l2_cache_bytes > 0
    assert profile.physical_cpu_cores >= 1
    assert isinstance(profile.has_apple_amx, bool)
    assert isinstance(profile.has_mps_gpu, bool)
    assert len(profile.chip_class) > 0


def test_tile_schedule_planner_decode_vs_prefill():
    """Verify TileSchedulePlanner generates adaptive schedules for decode (M<=8) vs prefill (M>=64)."""
    planner = tile_hal.TileSchedulePlanner()

    # 1. Autoregressive decode token (M = 1)
    sched_decode = planner.plan_gemm_tile(m=1, n=4096, k=1024, elem_bytes=4, device_type="mps")
    assert sched_decode.is_decode is True
    assert sched_decode.tile_m <= 8
    assert sched_decode.tile_n == 32
    assert sched_decode.pipeline_stages == 1
    if sched_decode.target_engine.startswith("Apple_Metal"):
        assert "8x32x16_Decode" in sched_decode.target_engine

    # 2. Batched prompt prefill (M = 128)
    sched_prefill = planner.plan_gemm_tile(m=128, n=4096, k=1024, elem_bytes=4, device_type="mps")
    assert sched_prefill.is_decode is False
    assert sched_prefill.tile_m >= 16
    assert sched_prefill.pipeline_stages >= 2
    if sched_prefill.target_engine.startswith("Apple_Metal"):
        assert "32x32x8_Prefill" in sched_prefill.target_engine


# =============================================================================
# 3. 1-Bit Hardware Ballot Packing Tests
# =============================================================================

def test_pack_and_unpack_1bit_ballot():
    """Verify 32x memory compaction with 1-bit hardware ballot packing."""
    x = torch.randn(8, 128, dtype=torch.float32)
    packed = torch.ops.ojasx_hal.pack_1bit_ballot(x)
    assert packed.shape == (8, 4)
    assert packed.dtype == torch.int32

    unpacked = torch.ops.ojasx_hal.unpack_1bit_ballot(packed, 128, torch.float32)
    expected = (x > 0).float()
    torch.testing.assert_close(unpacked, expected)


# =============================================================================
# 4. AMX Linear + ReLU2 Forward and Backward Tests
# =============================================================================

def test_amx_linear_relu2_forward_and_backward():
    """Verify AMX forward and direct Accelerate cblas_sgemm backward passes."""
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, K, dtype=torch.float32, requires_grad=True)
    b = torch.randn(N, dtype=torch.float32, requires_grad=True)

    # Reference forward & backward
    z_ref = F.linear(x, w, b)
    out_ref = F.relu(z_ref).square()
    loss_ref = out_ref.sum()
    loss_ref.backward()

    gx_ref = x.grad.clone()
    gw_ref = w.grad.clone()
    gb_ref = b.grad.clone()

    x.grad = None
    w.grad = None
    b.grad = None

    # Native TileHAL fused forward & autograd
    out_fused, packed_mask = tile_hal.amx_linear_relu2(x, w, b)
    torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)

    loss_fused = out_fused.sum()
    loss_fused.backward()

    torch.testing.assert_close(x.grad, gx_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w.grad, gw_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b.grad, gb_ref, atol=1e-4, rtol=1e-4)


# =============================================================================
# 5. Fused L1-Resident RMSNorm Tests
# =============================================================================

def test_fused_l1_rmsnorm_forward_and_backward():
    """Verify single-pass vDSP L1-resident RMSNorm forward and fused backward."""
    M, D = 8, 128
    x = torch.randn(M, D, dtype=torch.float32, requires_grad=True)
    w = torch.ones(D, dtype=torch.float32, requires_grad=True)
    eps = 1e-6

    # Reference forward & backward
    norm_ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    loss_ref = norm_ref.sum()
    loss_ref.backward()

    gx_ref = x.grad.clone()
    gw_ref = w.grad.clone()

    x.grad = None
    w.grad = None

    # Fused L1-resident RMSNorm forward & backward
    out_fused, rstd = tile_hal.l1_rmsnorm(x, w, eps)
    torch.testing.assert_close(out_fused, norm_ref, atol=1e-4, rtol=1e-4)

    loss_fused = out_fused.sum()
    loss_fused.backward()

    torch.testing.assert_close(x.grad, gx_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w.grad, gw_ref, atol=1e-4, rtol=1e-4)


# =============================================================================
# 6. TileLang Universal & Metal GEMM Tests
# =============================================================================

def test_tilelang_universal_gemm_2d_tile_decomposition():
    """Verify 2D (M, N) CPU tile decomposition on small M decoding ($M=1, N=128$)."""
    M, K, N = 1, 64, 128
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(N, K, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    out_univ, mask_univ = tile_hal.tilelang_universal_gemm_relu2(x, w, b, target_arch="cpu")
    out_ref = F.relu(F.linear(x, w, b)).square()

    torch.testing.assert_close(out_univ, out_ref, atol=1e-4, rtol=1e-4)
    unpacked = torch.ops.ojasx_hal.unpack_1bit_ballot(mask_univ, N, torch.float32)
    torch.testing.assert_close(unpacked, (F.linear(x, w, b) > 0).float())


def test_tilelang_metal_gemm_adaptive_micro_tiling():
    """Verify Metal GEMM with adaptive micro-tile sizing and zero-copy shared buffers."""
    for M in [1, 4, 8, 32]:
        K, N = 64, 64
        x = torch.randn(M, K, dtype=torch.float32)
        w = torch.randn(N, K, dtype=torch.float32)
        b = torch.randn(N, dtype=torch.float32)

        out_metal, mask_metal = tile_hal.tilelang_metal_gemm_relu2(x, w, b, panel_size=4)
        out_ref = F.relu(F.linear(x, w, b)).square()
        torch.testing.assert_close(out_metal, out_ref, atol=1e-4, rtol=1e-4)


def test_tilelang_universal_multi_dtype_support():
    """Verify clean execution across FP32, FP16, and BF16 dtypes."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(4, 32, dtype=dtype)
        w = torch.randn(64, 32, dtype=dtype)
        b = torch.randn(64, dtype=dtype)

        out, mask = tile_hal.tilelang_universal_gemm_relu2(x, w, b)
        assert out.dtype == dtype
        assert mask.dtype == torch.int32
        out_ref = F.relu(F.linear(x.float(), w.float(), b.float())).square().to(dtype)
        atol, rtol = (1e-3, 1e-3) if dtype == torch.float32 else (0.05, 0.05)
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


def test_non_contiguous_and_strided_inputs():
    """Verify non-contiguous slices and transposed tensors work with zero errors."""
    x_base = torch.randn(16, 64, dtype=torch.float32)
    w_base = torch.randn(64, 32, dtype=torch.float32)

    x_slice = x_base[::2, :]
    w_trans = w_base.t()
    assert not x_slice.is_contiguous()
    assert not w_trans.is_contiguous()

    out, mask = tile_hal.amx_linear_relu2(x_slice, w_trans)
    ref = F.relu(F.linear(x_slice, w_trans)).square()
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_zero_element_dimensions():
    """Verify robust handling of 0-element edge cases."""
    x_zero_m = torch.randn(0, 32, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    out_m, mask_m = tile_hal.amx_linear_relu2(x_zero_m, w)
    assert out_m.shape == (0, 16)
    assert mask_m.shape == (0, 1)

    x_zero_rms = torch.randn(0, 32, dtype=torch.float32)
    w_rms = torch.ones(32, dtype=torch.float32)
    out_rms, rstd_rms = tile_hal.l1_rmsnorm(x_zero_rms, w_rms)
    assert out_rms.shape == (0, 32)
    assert rstd_rms.shape == (0, 1)


# =============================================================================
# 7. PyTorch Official torch.library.opcheck Certification
# =============================================================================

def test_torch_library_opcheck_certification():
    """Strict PyTorch certification of all registered operators via `torch.library.opcheck`."""
    x = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 32, dtype=torch.float32, requires_grad=True)
    b = torch.randn(64, dtype=torch.float32, requires_grad=True)
    rw = torch.randn(32, dtype=torch.float32, requires_grad=True)
    g = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    u = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)

    torch.library.opcheck(torch.ops.ojasx_hal.pack_1bit_ballot.default, (x,))
    torch.library.opcheck(torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd.default, (x, w, b))
    torch.library.opcheck(torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd.default, (x, rw, 1e-6))
    torch.library.opcheck(torch.ops.ojasx_hal.fused_l1_glu_fwd.default, (g, u, 0))
    torch.library.opcheck(
        torch.ops.ojasx_hal.fused_l1_glu_bwd.default,
        (g.detach(), g.detach(), u.detach(), 0),
        test_utils=("test_schema", "test_faketensor"),
    )
    torch.library.opcheck(torch.ops.ojasx_hal.compress_boundary_int8.default, (x.detach(), 16))
    q, scale = torch.ops.ojasx_hal.compress_boundary_int8(x.detach(), 16)
    torch.library.opcheck(torch.ops.ojasx_hal.decompress_boundary_int8.default, (q, scale, [4, 32], torch.float32))
    torch.library.opcheck(torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd.default, (x, w, b, "cpu", 4))
    torch.library.opcheck(torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd.default, (x, w, b, 4))


# =============================================================================
# 8. Fused L1 Gated Linear Unit (11 Activation Modes) Parity
# =============================================================================

def _ref_glu_forward(g: torch.Tensor, u: torch.Tensor, mode: str) -> torch.Tensor:
    m = mode.lower()
    if m in ("swiglu", "silu"):
        phi = F.silu(g)
    elif m in ("geglu", "geglu_tanh", "gelu_tanh"):
        phi = F.gelu(g, approximate="tanh")
    elif m in ("geglu_exact", "gelu_exact"):
        phi = F.gelu(g, approximate="none")
    elif m in ("reglu", "relu"):
        phi = F.relu(g)
    elif m in ("relu2glu", "squared_reglu", "relu2"):
        phi = F.relu(g).square()
    elif m in ("sigmoidglu", "glu", "sigmoid"):
        phi = torch.sigmoid(g)
    elif m in ("bilinearglu", "bilinear", "identity"):
        phi = g
    elif m in ("quickgeglu", "quickgelu"):
        phi = g * torch.sigmoid(1.702 * g)
    elif m in ("mishglu", "mish"):
        phi = F.mish(g)
    elif m in ("tanhglu", "tanh"):
        phi = torch.tanh(g)
    elif m in ("leaky_reglu", "leaky_relu"):
        phi = F.leaky_relu(g, negative_slope=0.01)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return phi * u


def test_fused_l1_glu_all_modes_parity():
    """Verify all 11 GLU activation modes against PyTorch autograd reference."""
    modes = [
        "swiglu", "geglu_tanh", "geglu_exact", "reglu", "relu2glu",
        "sigmoidglu", "bilinearglu", "quickgeglu", "mishglu", "tanhglu", "leaky_reglu"
    ]
    for mode_idx, mode_name in enumerate(modes):
        for mode_arg in (mode_name, mode_idx):
            g = torch.randn(8, 32, dtype=torch.float32, requires_grad=True)
            u = torch.randn(8, 32, dtype=torch.float32, requires_grad=True)

            ref_out = _ref_glu_forward(g, u, mode_name)
            ref_loss = (ref_out * 0.5).sum()
            ref_loss.backward()

            ref_dg = g.grad.clone()
            ref_du = u.grad.clone()

            g.grad = None
            u.grad = None

            fused_out = tile_hal.fused_glu(g, u, act_mode=mode_arg)
            torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)

            fused_loss = (fused_out * 0.5).sum()
            fused_loss.backward()

            torch.testing.assert_close(g.grad, ref_dg, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(u.grad, ref_du, atol=1e-4, rtol=1e-4)
