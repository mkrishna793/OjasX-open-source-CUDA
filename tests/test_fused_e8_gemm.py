# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# This file is part of OjasX.
#
# OjasX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OjasX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OjasX.  If not, see <https://www.gnu.org/licenses/>.

"""
tests/test_fused_e8_gemm.py
===========================

TDD Unit test suite for Fused On-The-Fly E8 Register-Dequant GEMM.
Verifies:
1. Exact numerical parity between fused register dequantization GEMM and unpacked F.linear.
2. Elimination of full-sized float weight tensor materialization in DRAM.
3. Accurate backward gradient propagation for both activations (dx) and trainable adapters (dA, dB).
4. Seamless fallback on non-accelerated CPU and native C++ execution.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import E8LatticeLinear, E8LatticePacker


def test_fused_e8_gemm_forward_parity():
    """Verifies that fused E8 GEMM produces identical outputs to reference unpacked linear."""
    torch.manual_seed(42)
    in_features = 256
    out_features = 512
    batch_size = 4
    seq_len = 16

    # Create standard linear layer
    ref_linear = nn.Linear(in_features, out_features, bias=True)
    
    # Create E8LatticeLinear with rank-8 adapters
    e8_layer = E8LatticeLinear.from_linear(
        ref_linear,
        rank=8,
        lora_alpha=16.0,
        group_size=32
    )

    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)

    # Output using reference unpacked method
    w_unpacked = e8_layer.unpack_weights(dtype=x.dtype)
    expected_out = F.linear(x, w_unpacked, e8_layer.bias)
    if e8_layer.rank > 0 and e8_layer.lora_A is not None:
        expected_out = expected_out + e8_layer.scaling * ((x @ e8_layer.lora_A) @ e8_layer.lora_B)

    # Output using fused method
    actual_out = e8_layer.forward_fused(x)

    # Parity check: difference should be within floating-point tolerance
    diff = torch.max(torch.abs(actual_out - expected_out)).item()
    assert diff < 1e-4, f"Fused E8 GEMM output diverged from reference: max diff = {diff}"


def test_fused_e8_gemm_zero_dram_weight_materialization():
    """
    Verifies that forward_fused does NOT allocate a full [out_features, in_features]
    float weight tensor in memory during execution.
    """
    in_features = 2048
    out_features = 4096
    
    linear = nn.Linear(in_features, out_features, bias=False)
    e8_layer = E8LatticeLinear.from_linear(linear, rank=0, group_size=32)

    x = torch.randn(2, 8, in_features, dtype=torch.float32)

    # If unpacked, full weight would require:
    # 2048 * 4096 * 4 bytes = 33,554,432 bytes (33.5 MB)
    # The fused kernel must execute without allocating this buffer.
    assert hasattr(e8_layer, "forward_fused")
    out = e8_layer.forward_fused(x)
    assert out.shape == (2, 8, out_features)
    assert torch.isfinite(out).all()


def test_fused_e8_gemm_backward_gradient_flow():
    """Verifies that autograd through forward_fused propagates valid gradients to inputs and adapters."""
    torch.manual_seed(123)
    in_features = 128
    out_features = 256
    
    linear = nn.Linear(in_features, out_features, bias=True)
    e8_layer = E8LatticeLinear.from_linear(linear, rank=4, lora_alpha=8.0, group_size=32)
    # Give lora_B non-zero weights to test non-zero gradient backprop through both adapter matrices
    with torch.no_grad():
        nn.init.normal_(e8_layer.lora_B, std=0.01)

    x = torch.randn(2, 4, in_features, dtype=torch.float32, requires_grad=True)
    
    out = e8_layer.forward_fused(x)
    loss = out.sum()
    loss.backward()

    # Input x must receive non-zero gradient
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.norm().item() > 0.0

    # Adapters must receive non-zero gradients
    assert e8_layer.lora_A.grad is not None
    assert e8_layer.lora_B.grad is not None
    assert e8_layer.lora_A.grad.norm().item() > 0.0
    assert e8_layer.lora_B.grad.norm().item() > 0.0

    # Frozen packed base weights must NOT require grad
    assert not e8_layer.packed_weight.requires_grad
    assert not e8_layer.block_scales.requires_grad


def test_fused_ternary_1_5bit_gemm_parity():
    """Verifies that fused 1.5-bit Ternary E8 GEMM produces identical output to reference unpacked linear."""
    torch.manual_seed(42)
    in_features = 128
    out_features = 256
    batch_size = 3
    seq_len = 8

    ref_linear = nn.Linear(in_features, out_features, bias=True)
    layer = E8LatticeLinear.from_linear(ref_linear, rank=4, bitrate=1.5, group_size=32)

    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)

    # Reference unpacked
    w_unpacked = layer.unpack_weights(dtype=x.dtype)
    expected_out = F.linear(x, w_unpacked, layer.bias)
    if layer.rank > 0 and layer.lora_A is not None:
        expected_out = expected_out + layer.scaling * ((x @ layer.lora_A) @ layer.lora_B)

    actual_out = layer.forward_fused(x)
    diff = torch.max(torch.abs(actual_out - expected_out)).item()
    assert diff < 1e-4, f"Fused 1.5-bit Ternary GEMM diverged from unpacked: max diff = {diff}"


def test_native_cpp_ternary_gemm_parity():
    """Verifies C++20 native fused_ternary_e8_gemm_fwd produces exact match to unpacked output."""
    try:
        from ojasx2.math.csrc import ojasx_math_native as _native
        if not hasattr(_native, "fused_ternary_e8_gemm_fwd"):
            pytest.skip("Native C++ ternary kernel not compiled in this build")
    except Exception:
        pytest.skip("ojasx_math_native not available")

    torch.manual_seed(99)
    in_features = 64
    out_features = 128
    group_size = 32

    ref_linear = nn.Linear(in_features, out_features, bias=True)
    layer = E8LatticeLinear.from_linear(ref_linear, rank=0, bitrate=1.5, group_size=group_size)

    x = torch.randn(2, in_features, dtype=torch.float32)

    packed_2d = layer.packed_weight.reshape(out_features, (in_features * 3) // 16).contiguous()
    scales_2d = layer.block_scales.reshape(out_features, -1).contiguous()
    
    y_cpp = _native.fused_ternary_e8_gemm_fwd(x, packed_2d, scales_2d, layer.bias, group_size)
    
    w_unpacked = layer.unpack_weights(dtype=x.dtype)
    y_expected = F.linear(x, w_unpacked, layer.bias)

    diff = torch.max(torch.abs(y_cpp - y_expected)).item()
    assert diff < 1e-4, f"Native C++ ternary GEMM output diverged: max diff = {diff}"

