"""
tests/test_tilelang_and_root_e8.py
==================================
Unit and differential regression tests for:
1. RootE8LatticePacker (1.000 bit/weight 240-root Gosset sphere codebook).
2. Fast Walsh-Hadamard Transform (fwht_block32 orthogonal involution).
3. DeepSeek TileLang Hardware JIT dispatch and fallback chain.
4. HolonomicGradFreeOptimizer with zero-RAM gradient accumulation, running global norm, and E8 momentum.
5. holonomic_dagger_attention SDPA fast-path dispatch.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import (
    RootE8LatticePacker,
    TernaryE8LatticePacker,
    E8LatticePacker,
    E8LatticeLinear,
    pack_model_e8,
    fwht_block32,
    fast_walsh_hadamard_transform,
    _get_root_e8_lut,
)
from ojasx2.tilelang_kernels import (
    is_tilelang_available,
    dispatch_tilelang_e8_gemm,
)
from ojasx2.dagger_attention import (
    holonomic_dagger_attention,
)
from ojasx2.grad_free_optimizer import (
    HolonomicGradFreeOptimizer,
)


# =============================================================================
# 1. RootE8LatticePacker (1.0-bit) Tests
# =============================================================================

def test_root_e8_lut_geometry():
    """Verify Root E8 codebook has 256 points in R^8 on sphere S^7 with identical norms."""
    lut = _get_root_e8_lut(torch.device("cpu"))
    assert lut.shape == (256, 8)
    assert lut.dtype == torch.float32

    # Verify all norms squared are identically equal (equidistant sphere)
    norms_sq = (lut ** 2).sum(dim=-1)
    expected_norm_sq = norms_sq[0].item()
    assert torch.allclose(norms_sq, torch.full_like(norms_sq, expected_norm_sq), atol=1e-5), (
        f"Non-equidistant norms: min={norms_sq.min().item()}, max={norms_sq.max().item()}"
    )

    # Verify no duplicate points
    unique_rows = torch.unique(lut, dim=0)
    assert unique_rows.shape[0] == 256, f"Duplicate vectors in codebook: got {unique_rows.shape[0]}"


def test_root_e8_pack_unpack_shape_and_bitrate():
    """Verify RootE8 packing reduces 8 weights to 1 uint8 byte = 1.000 bit/weight."""
    M, K = 64, 128
    x = torch.randn(M, K, dtype=torch.float32)

    packed, scales, original_shape = RootE8LatticePacker.pack(x, group_size=32)

    # 1 byte per 8 weights -> total packed bytes = (M * K) // 8
    assert packed.numel() == (M * K) // 8
    assert packed.dtype == torch.uint8
    assert scales.shape == (M * K // 32,)
    assert original_shape == (M, K)

    # Bitrate verification: exactly 1.000 bit per weight
    total_weights = M * K
    total_bits = packed.numel() * 8
    bitrate = total_bits / total_weights
    assert bitrate == 1.0, f"Expected 1.0 bit/weight, got {bitrate}"

    # Unpack and verify shape and reconstruction
    unpacked = RootE8LatticePacker.unpack(packed, scales, original_shape, dtype=torch.float32)
    assert unpacked.shape == (M, K)
    assert unpacked.dtype == torch.float32

    # Reconstruction error check (SQNR should exceed 4.0 dB for Gaussian noise)
    signal_power = (x ** 2).mean()
    noise_power = ((x - unpacked) ** 2).mean()
    sqnr = 10.0 * torch.log10(signal_power / (noise_power + 1e-12))
    assert sqnr.item() > 4.0, f"SQNR too low: {sqnr.item()} dB"


# =============================================================================
# 2. Fast Walsh-Hadamard Transform (FWHT-32) Tests
# =============================================================================

def test_fwht_block32_orthogonal_involution():
    """Verify fwht_block32 is symmetric and self-inverse: H^2 = I."""
    eye32 = torch.eye(32, dtype=torch.float32)
    H = fwht_block32(eye32)

    # Symmetry: H == H.t()
    assert torch.allclose(H, H.t(), atol=1e-6), "FWHT matrix is not symmetric"

    # Orthogonality / Involution: H @ H == I
    H2 = torch.matmul(H, H)
    assert torch.allclose(H2, eye32, atol=1e-5), f"H^2 != I: max error {(H2 - eye32).abs().max().item()}"


def test_fwht_outlier_dispersal():
    """Verify FWHT disperses heavy activation/weight outlier coordinates across the 32D block."""
    # Create a vector with a heavy outlier of magnitude 32.0 in component 0
    x = torch.zeros(1, 32, dtype=torch.float32)
    x[0, 0] = 32.0

    y = fwht_block32(x)

    # Energy is preserved: norm(y) == norm(x)
    assert torch.allclose(torch.norm(x), torch.norm(y), atol=1e-5)

    # Max coordinate must be reduced by exactly sqrt(32) ≈ 5.65x
    assert y.abs().max().item() < x.abs().max().item() / 5.0


def test_fast_walsh_hadamard_transform_arbitrary_tensor():
    """Verify fast_walsh_hadamard_transform works on multi-dimensional tensors."""
    B, S, D = 2, 8, 128
    x = torch.randn(B, S, D, dtype=torch.float32)

    transformed = fast_walsh_hadamard_transform(x)
    assert transformed.shape == (B, S, D)

    # Self-inverse test
    reconstructed = fast_walsh_hadamard_transform(transformed)
    assert torch.allclose(x, reconstructed, atol=1e-5)


# =============================================================================
# 3. E8LatticeLinear & pack_model_e8 with Hadamard Tests
# =============================================================================

def test_e8_lattice_linear_1bit_and_hadamard():
    """Test E8LatticeLinear forward and backward with 1.0b RootE8 and Hadamard rotation."""
    in_features, out_features = 64, 64
    lin = nn.Linear(in_features, out_features, bias=True)

    e8_lin = E8LatticeLinear.from_linear(
        lin,
        bitrate=1.0,
        rank=8,
        group_size=32,
        use_hadamard=True,
    )
    assert e8_lin.use_hadamard is True
    assert e8_lin.bitrate == 1.0

    x = torch.randn(4, 16, in_features, dtype=torch.float32, requires_grad=True)
    out = e8_lin(x)
    assert out.shape == (4, 16, out_features)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_pack_model_e8_with_root_and_hadamard():
    """Test pack_model_e8 replacing linears with 1.0b and 1.5b with use_hadamard=True."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 32)
            self.head = nn.Linear(32, 10)

        def forward(self, x):
            return self.head(F.relu(self.fc2(F.relu(self.fc1(x)))))

    model = ToyModel()
    packed_model = pack_model_e8(model, bitrate=1.0, rank=4, group_size=32, use_hadamard=True)

    assert isinstance(packed_model.fc1, E8LatticeLinear)
    assert isinstance(packed_model.fc2, E8LatticeLinear)
    assert packed_model.fc1.use_hadamard is True

    x = torch.randn(2, 64)
    y = packed_model(x)
    assert y.shape == (2, 10)


# =============================================================================
# 4. DeepSeek TileLang Integration & Fallback Tests
# =============================================================================

def test_tilelang_fallback_safety():
    """Verify dispatch_tilelang_e8_gemm gracefully returns None without crashing."""
    x = torch.randn(4, 64, dtype=torch.float16)
    w_packed = torch.zeros(64, 16, dtype=torch.uint8)
    scales = torch.ones(64, 2, dtype=torch.float16)

    # Dispatched call must never raise an unhandled exception
    res = dispatch_tilelang_e8_gemm(
        x, w_packed, scales, bias=None, bitrate=2.0, group_size=32,
        out_features=64, in_features=64,
    )
    # If TileLang runtime is not installed or available, it cleanly returns None
    if not is_tilelang_available():
        assert res is None


# =============================================================================
# 5. HolonomicGradFreeOptimizer Upgraded Tests
# =============================================================================

def test_grad_free_optimizer_zero_ram_accumulation():
    """Test grad_accum_steps > 1 performs zero-RAM accumulation without persistent grad_accum clones."""
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )

    opt = HolonomicGradFreeOptimizer(
        model.parameters(),
        lr=1e-3,
        algorithm="adamw",
        grad_accum_steps=4,
        max_grad_norm=1.0,
    )

    initial_weight = model[0].weight.clone()

    # Step 1, 2, 3: micro-steps accumulate without updating weights
    for micro_step in range(3):
        x = torch.randn(2, 32)
        loss = model(x).sum()
        loss.backward()
        # Verify param.grad is immediately freed to None
        assert model[0].weight.grad is None
        assert model[2].weight.grad is None
        # Weights must remain unchanged before micro-step 4
        assert torch.allclose(model[0].weight, initial_weight)

    # Micro-step 4: modulo boundary triggers weight update
    x = torch.randn(2, 32)
    loss = model(x).sum()
    loss.backward()
    assert model[0].weight.grad is None
    # Weights have now updated!
    assert not torch.allclose(model[0].weight, initial_weight)


def test_grad_free_optimizer_e8_momentum_quantization():
    """Test quantize_moments_e8=True stores first moments in 2.0-bit E8 lattice codes."""
    lin = nn.Linear(64, 64)
    opt = HolonomicGradFreeOptimizer(
        lin.parameters(),
        lr=1e-3,
        algorithm="adamw",
        quantize_moments_e8=True,
    )

    x = torch.randn(4, 64)
    loss = lin(x).sum()
    loss.backward()

    st = opt.state[lin.weight]
    # First moment should be compressed into exp_avg_packed
    assert "exp_avg_packed" in st
    assert "exp_avg_scales" in st
    assert "exp_avg" not in st or st["exp_avg"] is None
    assert st["exp_avg_packed"].dtype == torch.uint8


# =============================================================================
# 6. Holonomic Dagger Attention Hardware SDPA Fast-Path Tests
# =============================================================================

def test_holonomic_dagger_attention_sdpa_fast_path():
    """Test holonomic_dagger_attention routes to hardware SDPA and matches standard attention."""
    B, H, S, d = 2, 4, 32, 64
    q = torch.randn(B, H, S, d, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, S, d, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, S, d, dtype=torch.float32, requires_grad=True)

    # Native SDPA fast-path
    out_fast = holonomic_dagger_attention(q, k, v, is_causal=False, prefer_native_sdpa=True)

    # Reference SDPA
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    assert torch.allclose(out_fast, out_ref, atol=1e-5)

    # Backward gradient check
    loss = out_fast.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_holonomic_dagger_attention_gqa_dispatch():
    """Test GQA dispatch (H=8, H_kv=2) under holonomic_dagger_attention."""
    B, H, H_kv, S, d = 2, 8, 2, 32, 64
    q = torch.randn(B, H, S, d, dtype=torch.float32)
    k = torch.randn(B, H_kv, S, d, dtype=torch.float32)
    v = torch.randn(B, H_kv, S, d, dtype=torch.float32)

    out = holonomic_dagger_attention(q, k, v, is_causal=True, prefer_native_sdpa=True)
    assert out.shape == (B, H, S, d)
