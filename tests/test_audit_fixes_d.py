"""
tests/test_audit_fixes_d.py

Audit fix verification suite for Workstream D (C++ & Metal Kernels):
1. Metal Zero-Copy & Storage Offset Safety:
   - Verify MPS tensors with storage_offset > 0 (sliced tensors) execute correctly
     without reading offset 0 or aliasing.
   - Verify CPU tensors with arbitrary (non-page-aligned) pointers cleanly fall back
     to newBufferWithBytes and properly propagate outputs back to CPU memory.
2. Dtype Coverage (FP32, FP16, BF16) across CPU and MPS:
   - fused_amx_linear_relu2_fwd and backward on CPU and MPS across FP32, FP16, BF16.
   - fused_l1_rmsnorm_fwd and backward on CPU and MPS across FP32, FP16, BF16.
   - tilelang_metal_gemm_relu2_fwd across FP32, FP16, BF16 on CPU and MPS.
"""

import pytest
import torch
import ojasx2.tile_hal

HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
DEVICES = ["cpu", "mps"] if HAS_MPS else ["cpu"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# =============================================================================
# 1. Metal Buffer Zero-Copy & Offset Safety Tests
# =============================================================================

@pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
def test_metal_gemm_mps_storage_offset():
    """Verify that MPS tensors with storage_offset > 0 compute identical results to cloned tensors."""
    x_full = torch.randn(12, 32, device="mps", dtype=torch.float32)
    w_full = torch.randn(64, 32, device="mps", dtype=torch.float32)
    b_full = torch.randn(64, device="mps", dtype=torch.float32)

    # Slice creates storage_offset > 0
    x_slice = x_full[4:8]  # shape (4, 32), offset 128
    w_slice = w_full[16:48]  # shape (32, 32), offset 512
    b_slice = b_full[16:48]  # shape (32), offset 16

    assert x_slice.storage_offset() > 0
    assert w_slice.storage_offset() > 0
    assert b_slice.storage_offset() > 0

    out_slice, mask_slice = torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(
        x_slice, w_slice, b_slice, 16
    )

    # Reference with clean offset 0 tensors
    out_ref, mask_ref = torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(
        x_slice.clone(), w_slice.clone(), b_slice.clone(), 16
    )

    diff = (out_slice - out_ref).abs().max().item()
    assert diff == 0.0, f"Sliced MPS tensor had non-zero diff vs reference: {diff}"
    assert torch.equal(mask_slice, mask_ref)


def test_metal_gemm_cpu_unaligned_fallback():
    """Verify that CPU tensors without page alignment cleanly fall back and retrieve output."""
    # Small 4x16 tensor is not page-aligned (requires 16KB alignment)
    x = torch.randn(4, 16, device="cpu", dtype=torch.float32)
    w = torch.randn(32, 16, device="cpu", dtype=torch.float32)
    b = torch.randn(32, device="cpu", dtype=torch.float32)

    out, mask = torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(x, w, b, 16)
    ref = torch.relu(x @ w.T + b) ** 2

    assert out.shape == (4, 32)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# =============================================================================
# 2. fused_amx_linear_relu2 Tests across FP32, FP16, BF16 on CPU and MPS
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_amx_linear_relu2_fwd_dtypes(device, dtype):
    """Verify fused_amx_linear_relu2_fwd works accurately across FP32, FP16, BF16 on CPU & MPS."""
    x = torch.randn(4, 32, device=device, dtype=dtype)
    w = torch.randn(16, 32, device=device, dtype=dtype)
    b = torch.randn(16, device=device, dtype=dtype)

    out, mask = torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, w, b)
    assert out.shape == (4, 16)
    assert out.dtype == dtype
    assert out.device.type == device
    assert mask.shape == (4, 1)  # (16 + 31) // 32 = 1

    # Verify numerical accuracy
    ref = torch.relu(x.float() @ w.float().T + b.float()) ** 2
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_amx_linear_relu2_bwd_dtypes(device, dtype):
    """Verify fused_amx_linear_relu2_bwd works across FP32, FP16, BF16 on CPU & MPS."""
    x = torch.randn(4, 32, device=device, dtype=dtype)
    w = torch.randn(16, 32, device=device, dtype=dtype)
    b = torch.randn(16, device=device, dtype=dtype)

    out, mask = torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, w, b)
    go = torch.randn_like(out)

    dx, dw, db = torch.ops.ojasx_hal.fused_amx_linear_relu2_bwd(go, x, w, b, mask)

    assert dx.shape == x.shape and dx.dtype == dtype and dx.device.type == device
    assert dw.shape == w.shape and dw.dtype == dtype and dw.device.type == device
    assert db.shape == b.shape and db.dtype == dtype and db.device.type == device

    # Check finite gradients
    assert torch.isfinite(dx).all()
    assert torch.isfinite(dw).all()
    assert torch.isfinite(db).all()


@pytest.mark.parametrize("device", DEVICES)
def test_fused_amx_linear_relu2_no_bias(device):
    """Verify fused_amx_linear_relu2 handles optional None bias safely."""
    x = torch.randn(4, 16, device=device, dtype=torch.float32)
    w = torch.randn(32, 16, device=device, dtype=torch.float32)

    out, mask = torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, w, None)
    assert out.shape == (4, 32)
    ref = torch.relu(x @ w.T) ** 2
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    go = torch.randn_like(out)
    dx, dw, db = torch.ops.ojasx_hal.fused_amx_linear_relu2_bwd(go, x, w, None, mask)
    assert dx.shape == x.shape
    assert dw.shape == w.shape


# =============================================================================
# 3. fused_l1_rmsnorm Tests across FP32, FP16, BF16 on CPU and MPS
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_l1_rmsnorm_fwd_dtypes(device, dtype):
    """Verify fused_l1_rmsnorm_fwd handles FP32, FP16, BF16 on CPU and MPS with high precision."""
    dim = 64
    x = torch.randn(8, dim, device=device, dtype=dtype)
    w = torch.ones(dim, device=device, dtype=dtype)
    eps = 1e-5

    out, rstd = torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd(x, w, eps)

    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device.type == device
    assert rstd.shape == (8, 1)

    # Reference RMSNorm in float32
    x_f = x.float()
    w_f = w.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    ref = (x_f * inv_rms) * w_f

    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_l1_rmsnorm_3d_shape(device, dtype):
    """Verify fused_l1_rmsnorm preserves 3D batch sequence dimensions."""
    x = torch.randn(2, 4, 32, device=device, dtype=dtype)
    w = torch.ones(32, device=device, dtype=dtype)
    out, rstd = torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd(x, w, 1e-5)

    assert out.shape == (2, 4, 32)
    assert rstd.shape == (2, 4, 1)
    assert out.dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_l1_rmsnorm_bwd_dtypes(device, dtype):
    """Verify fused_l1_rmsnorm_bwd operates safely on CPU and MPS across dtypes."""
    dim = 32
    x = torch.randn(4, dim, device=device, dtype=dtype)
    w = torch.randn(dim, device=device, dtype=dtype)
    out, rstd = torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd(x, w, 1e-5)
    go = torch.randn_like(out)

    dx, dw = torch.ops.ojasx_hal.fused_l1_rmsnorm_bwd(go, x, w, rstd)

    assert dx.shape == x.shape and dx.dtype == dtype and dx.device.type == device
    assert dw.shape == w.shape and dw.dtype == dtype and dw.device.type == device
    assert torch.isfinite(dx).all()
    assert torch.isfinite(dw).all()


# =============================================================================
# 4. tilelang_metal_gemm_relu2_fwd Tests across FP32, FP16, BF16
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tilelang_metal_gemm_dtypes(device, dtype):
    """Verify tilelang_metal_gemm_relu2_fwd executes across FP32, FP16, BF16 on CPU and MPS."""
    x = torch.randn(4, 16, device=device, dtype=dtype)
    w = torch.randn(32, 16, device=device, dtype=dtype)
    b = torch.randn(32, device=device, dtype=dtype)

    out, mask = torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(x, w, b, 16)

    assert out.shape == (4, 32)
    assert out.dtype == dtype
    assert out.device.type == device
    assert mask.shape == (4, 1)

    ref = torch.relu(x.float() @ w.float().T + b.float()) ** 2
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
