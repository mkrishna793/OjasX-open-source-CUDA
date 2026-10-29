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
Unit tests for ojasx2.math.lattice:
- Fast Walsh-Hadamard Transform (FWHT)
- Conway-Sloane E8 Lattice Projection
- 2.0-bit E8LatticePacker
- E8LatticeLinear (with trainable low-rank subspace)
- E8QuantizationAwareLinear (with Straight-Through Estimator)
- pack_model_e8
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import (
    fast_walsh_hadamard_transform,
    randomized_hadamard_transform,
    conway_sloane_e8_project,
    E8LatticePacker,
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    pack_model_e8,
)


def test_fwht_inversion_and_norm():
    """Verify FWHT is strictly self-inverting and preserves Frobenius norm."""
    torch.manual_seed(42)
    x = torch.randn(8, 64)
    h = fast_walsh_hadamard_transform(x)
    recon = fast_walsh_hadamard_transform(h)
    
    # Self-inversion
    diff = torch.max(torch.abs(x - recon)).item()
    assert diff < 1e-5, f"FWHT inversion error too high: {diff}"
    
    # Norm preservation (Parseval / isometry)
    orig_norm = torch.norm(x).item()
    rot_norm = torch.norm(h).item()
    assert abs(orig_norm - rot_norm) < 1e-4, "FWHT does not preserve norm!"


def test_conway_sloane_e8_membership():
    """Verify that every snapped point strictly satisfies the algebraic definition of E8."""
    torch.manual_seed(123)
    # Generate 50 random 8D query vectors
    x = torch.randn(50, 8) * 3.0
    p = conway_sloane_e8_project(x)
    
    assert p.shape == (50, 8)
    for i in range(50):
        pt = p[i]
        is_all_int = torch.allclose(pt, torch.round(pt), atol=1e-5)
        is_all_half = torch.allclose(pt - 0.5, torch.round(pt - 0.5), atol=1e-5)
        assert is_all_int or is_all_half, f"Point {i} is neither all-integer nor all-half-integer: {pt}"
        
        if is_all_int:
            s = int(torch.round(torch.sum(pt)).item())
            assert s % 2 == 0, f"Integer E8 point {i} has odd sum: {s}"
        else:
            s = int(torch.round(torch.sum(pt - 0.5)).item())
            assert s % 2 == 0, f"Half-integer E8 point {i} has odd shifted sum: {s}"


def test_e8_packer_unpack_roundtrip():
    """Verify 2.0-bit packing achieves exactly 8.0x reduction and valid reconstruction."""
    torch.manual_seed(42)
    # 64 x 128 weight matrix = 8,192 parameters
    w = torch.randn(64, 128) * 0.05
    numel = w.numel()
    
    packed_codes, scales, shape = E8LatticePacker.pack(w)
    
    # In 16-bit FP16, 8,192 weights consume 8,192 * 2 = 16,384 bytes
    # In 2.0-bit packed uint8, 8,192 weights consume 8,192 / 4 = 2,048 bytes!
    assert packed_codes.dtype == torch.uint8
    assert packed_codes.numel() == numel // 4, f"Expected {numel // 4} bytes, got {packed_codes.numel()}"
    
    # Scale has 1 float16 per 8-weight block
    assert scales.numel() == numel // 8
    
    # Unpack
    w_recon = E8LatticePacker.unpack(packed_codes, scales, shape)
    assert w_recon.shape == (64, 128)
    
    # Check high correlation (cosine similarity > 0.90)
    cos_sim = torch.sum(w * w_recon) / (torch.norm(w) * torch.norm(w_recon))
    assert cos_sim.item() > 0.90, f"Cosine similarity {cos_sim.item()} too low!"


def test_e8_lattice_linear_training_with_subspace():
    """Verify E8LatticeLinear forward and backward with trainable low-rank adapters."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 32
    orig_linear = nn.Linear(in_dim, out_dim)
    
    # Convert to E8LatticeLinear with rank-4 trainable adapter
    e8_layer = E8LatticeLinear.from_linear(orig_linear, rank=4, lora_alpha=8.0)
    
    # Check that base weight is packed and frozen
    assert hasattr(e8_layer, "packed_weight")
    assert e8_layer.packed_weight.dtype == torch.uint8
    assert not e8_layer.packed_weight.requires_grad
    assert e8_layer.lora_A.requires_grad
    assert e8_layer.lora_B.requires_grad
    
    # Forward pass
    x = torch.randn(4, in_dim, requires_grad=True)
    out = e8_layer(x)
    assert out.shape == (4, out_dim)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    # Gradients must populate for trainable subspace A and B
    assert e8_layer.lora_A.grad is not None
    assert e8_layer.lora_B.grad is not None
    assert x.grad is not None
    assert not torch.isnan(e8_layer.lora_A.grad).any()
    assert not torch.isnan(e8_layer.lora_B.grad).any()


def test_e8_quantization_aware_linear_ste():
    """Verify E8QuantizationAwareLinear updates continuous weights via STE."""
    torch.manual_seed(42)
    in_dim, out_dim = 32, 16
    layer = E8QuantizationAwareLinear(in_dim, out_dim)
    
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    
    initial_w = layer.weight.clone().detach()
    
    # Training step
    x = torch.randn(2, in_dim)
    y = layer(x)
    loss = (y ** 2).sum()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Verify STE gradient
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
    
    optimizer.step()
    
    # Weights must update
    updated_w = layer.weight.detach()
    diff = torch.max(torch.abs(initial_w - updated_w)).item()
    assert diff > 0.0, "Weights failed to update under STE!"


def test_pack_model_e8_end_to_end():
    """Verify pack_model_e8 converts standard PyTorch model and executes training loop."""
    torch.manual_seed(42)
    
    class ToyTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(64, 64)
            self.proj = nn.Linear(64, 64)
            self.mlp1 = nn.Linear(64, 128)
            self.mlp2 = nn.Linear(128, 64)
            
        def forward(self, x):
            h = self.proj(F.relu(self.qkv(x)))
            return self.mlp2(F.relu(self.mlp1(h)))

    model = ToyTransformerBlock()
    
    # Count original parameters in FP32
    orig_bytes = sum(p.numel() * 4 for p in model.parameters())
    
    # Pack entire model into 2-bit E8 lattice with rank-4 adapters
    model_e8 = pack_model_e8(model, rank=4, lora_alpha=8.0)
    
    # Verify all Linear modules became E8LatticeLinear
    assert isinstance(model_e8.qkv, E8LatticeLinear)
    assert isinstance(model_e8.proj, E8LatticeLinear)
    assert isinstance(model_e8.mlp1, E8LatticeLinear)
    assert isinstance(model_e8.mlp2, E8LatticeLinear)
    
    # Run forward and backward step
    x = torch.randn(2, 64)
    target = torch.randn(2, 64)
    
    opt = torch.optim.AdamW([p for p in model_e8.parameters() if p.requires_grad], lr=1e-3)
    out = model_e8(x)
    loss = F.mse_loss(out, target)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    assert not torch.isnan(loss).item()


def test_e8_with_holonomic_grad_free_optimizer():
    """Verify E8LatticeLinear integrates with HolonomicGradFreeOptimizer in-backward hooks."""
    from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
        def forward(self, x):
            return self.fc2(F.silu(self.fc1(x)))
            
    m = SimpleMLP()
    m_e8 = pack_model_e8(m, rank=8)
    
    # Trainable parameters must only be adapters and biases
    params = list(m_e8.parameters())
    assert all(p.requires_grad for p in params)
    
    opt = HolonomicGradFreeOptimizer(params, lr=1e-3, algorithm="adafactor")
    
    x = torch.randn(4, 64)
    loss = m_e8(x).sum()
    loss.backward()
    
    assert opt.last_step_updated_params > 0
    assert opt.last_step_freed_bytes > 0


def test_e8_with_smart_router_optimize_model():
    """Verify smart_router.optimize_model preserves E8 packed layers without double wrapping."""
    from ojasx2.smart_router import optimize_model
    
    class SimpleBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.norm = nn.LayerNorm(128)
            self.fc2 = nn.Linear(128, 64)
        def forward(self, x):
            return self.fc2(self.norm(F.relu(self.fc1(x))))
            
    m = SimpleBlock()
    m_e8 = pack_model_e8(m, rank=4)
    
    m_opt, report = optimize_model(m_e8)
    assert isinstance(m_opt.fc1, E8LatticeLinear)
    assert isinstance(m_opt.fc2, E8LatticeLinear)
    
    x = torch.randn(2, 64, requires_grad=True)
    out = m_opt(x)
    loss = out.sum()
    loss.backward()
    
    assert m_opt.fc1.lora_A.grad is not None
    assert m_opt.fc2.lora_A.grad is not None

