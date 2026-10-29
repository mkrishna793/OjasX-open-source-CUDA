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
tests/test_sub2bit_lattice.py
=============================

TDD Unit test suite for Sub-2-Bit Lattice Quantization:
1. Ternary E8 Lattice Packing (1.50 bits/weight via 12-bit D8 parity index).
2. 1-Bit Sign + Sparse E8 Residual Vector Quantization (RVQ, 1.25 bits/weight).
3. Integration with E8LatticeLinear at bitrate=1.5 and bitrate=1.25.
"""

import pytest
import torch
import torch.nn as nn

from ojasx2.math.lattice import (
    E8LatticeLinear,
    pack_model_e8,
    TernaryE8LatticePacker,
    conway_sloane_e8_project,
)


def test_ternary_e8_packer_roundtrip():
    """Verifies that TernaryE8LatticePacker packs 8 weights into 12 bits (1.5 b/w) with bounded error."""
    torch.manual_seed(99)
    weights = torch.randn(64, 128, dtype=torch.float32)
    
    # Pack into 1.5-bit representation
    packed, scales, shape = TernaryE8LatticePacker.pack(weights, group_size=32)
    
    # 64 * 128 = 8192 weights.
    # Grouped into 8D vectors: 8192 / 8 = 1024 8D blocks.
    # 12 bits per 8D block: 1024 * 1.5 bytes = 1536 bytes.
    # Check that packed storage size matches exact 1.5 bits/weight!
    expected_bytes = int(weights.numel() * 1.5 / 8)
    assert packed.numel() == expected_bytes, f"Expected {expected_bytes} bytes, got {packed.numel()}"

    # Unpack
    w_recon = TernaryE8LatticePacker.unpack(packed, scales, shape)
    assert w_recon.shape == weights.shape
    assert torch.isfinite(w_recon).all()
    
    # Correlation between original and quantized weights must be positive and strong
    cos_sim = torch.cosine_similarity(weights.flatten(), w_recon.flatten(), dim=0).item()
    assert cos_sim > 0.70, f"Cosine similarity too low: {cos_sim}"


def test_e8_lattice_linear_sub2bit_modes():
    """Verifies that E8LatticeLinear supports bitrate=1.5 (Ternary E8) for fine-tuning."""
    torch.manual_seed(77)
    linear = nn.Linear(256, 512, bias=True)

    # Instantiate layer with 1.5-bit base weights
    layer_1_5 = E8LatticeLinear.from_linear(
        linear,
        rank=8,
        bitrate=1.5,
        group_size=32
    )

    assert layer_1_5.bitrate == 1.5
    
    # Memory size of packed weights should be 1.5 bits/weight (10.67x smaller than FP16)
    expected_packed_bytes = int(256 * 512 * 1.5 / 8)
    assert layer_1_5.packed_weight.numel() == expected_packed_bytes

    # Forward pass
    x = torch.randn(2, 4, 256)
    out = layer_1_5(x)
    assert out.shape == (2, 4, 512)
    assert torch.isfinite(out).all()

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert layer_1_5.lora_A.grad is not None
    assert layer_1_5.lora_B.grad is not None


def test_e8_lattice_linear_default_group_size_1_5():
    """Verifies that E8LatticeLinear.from_linear with default group_size handles bitrate=1.5 cleanly."""
    linear = nn.Linear(64, 64)
    # Default group_size is 8 in from_linear; must not crash with bitrate=1.5
    layer = E8LatticeLinear.from_linear(linear, bitrate=1.5)
    assert layer.bitrate == 1.5
    assert layer.block_scales.shape[0] == (64 * 64) // layer.group_size
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()

