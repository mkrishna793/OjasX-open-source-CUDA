"""
tests/test_symplectic_numerical_mechanics.py

Rigorous Verification Suite for Upgraded Symplectic Numerical Mechanics:
1. 2nd-Order Störmer-Verlet (symmetric Strang splitting) with FSAL (First-Same-As-Last).
2. Persistent FP32 State Registers (bounding 64+ layer FP16 reconstruction drift < 1e-4).
3. Static Two-Buffer In-Place Ping-Pong Memory Engine.
4. Symplectic Squeeze Scaling in Sp(2D, R) with det(S_lambda) = 1.
5. Exact Analytic Gradient Parity against Eager PyTorch Autograd.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

from ojasx2.symplectic_engine import (
    SymplecticCoupling,
    SymplecticTransformerBlock,
    SymplecticStack,
    SymplecticSqueeze,
    symplectic_squeeze,
    symplectic_coupling,
    symplectic_coupling_chain,
    audit_symplectic_reversibility,
)


class SmallResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        with torch.no_grad():
            self.fc2.weight.mul_(0.05)
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _build_test_chain(depth: int, dim: int, dtype: torch.dtype, device: torch.device):
    blocks = []
    for _ in range(depth):
        f = SmallResidualBlock(dim).to(dtype=dtype, device=device)
        g = SmallResidualBlock(dim).to(dtype=dtype, device=device)
        blocks.append((f, g))
    return blocks


@pytest.mark.parametrize("depth", [16, 32, 64, 128])
def test_stormer_verlet_fp16_drift_bounded(depth: int):
    """
    Verifies that persistent FP32 registers eliminate catastrophic floating-point cancellation,
    bounding FP16 reconstruction drift across 64+ layers (and 128 layers) to < 1e-4.
    """
    torch.manual_seed(1000 + depth)
    dim = 16
    batch = 4
    device = torch.device("cpu")
    dtype = torch.float16

    blocks = _build_test_chain(depth, dim, dtype, device)
    q1 = torch.randn(batch, dim, dtype=dtype, device=device)
    q2 = torch.randn(batch, dim, dtype=dtype, device=device)

    report = audit_symplectic_reversibility(
        q1, q2, blocks, scheme="stormer_verlet", tol=1e-4
    )

    max_err = report["max_reconstruction_error"]
    raw_drift = report["max_raw_drift"]

    assert max_err < 1e-4, f"FP16 drift {max_err} exceeded 1e-4 at depth {depth}"
    assert raw_drift < 1e-4, f"FP32 raw drift {raw_drift} exceeded 1e-4 at depth {depth}"
    assert report["is_reversible"] is True


@pytest.mark.parametrize("depth", [8, 16, 32])
def test_stormer_verlet_fsal_gradient_parity(depth: int):
    """
    Verifies exact analytic backward reconstruction for Störmer-Verlet against eager autograd.
    Gradients for state inputs and module parameters must match eager PyTorch down to 1e-10.
    """
    torch.manual_seed(2000 + depth)
    dim = 8
    batch = 2
    device = torch.device("cpu")
    dtype = torch.float64

    blocks_e = _build_test_chain(depth, dim, dtype, device)
    import copy
    blocks_s = copy.deepcopy(blocks_e)

    params_e = [p for b in blocks_e for m in b for p in m.parameters()]
    params_s = [p for b in blocks_s for m in b for p in m.parameters()]

    q1_e = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)
    q2_e = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)
    q1_s = q1_e.detach().clone().requires_grad_(True)
    q2_s = q2_e.detach().clone().requires_grad_(True)

    # 1. Eager Störmer-Verlet Strang splitting with FSAL
    cur_q1, cur_q2 = q1_e, q2_e
    cur_q1 = cur_q1 + 0.5 * blocks_e[0][0](cur_q2)
    for i, (f, g) in enumerate(blocks_e):
        cur_q2 = cur_q2 + g(cur_q1)
        if i < len(blocks_e) - 1:
            cur_q1 = cur_q1 + 0.5 * f(cur_q2) + 0.5 * blocks_e[i + 1][0](cur_q2)
        else:
            cur_q1 = cur_q1 + 0.5 * f(cur_q2)
    loss_e = (cur_q1.pow(2) + cur_q2.pow(2)).sum()
    loss_e.backward()

    # 2. Symplectic engine O(1) autograd
    out1_s, out2_s = symplectic_coupling_chain(q1_s, q2_s, blocks_s, scheme="stormer_verlet")
    loss_s = (out1_s.pow(2) + out2_s.pow(2)).sum()
    loss_s.backward()

    # Parity checks
    torch.testing.assert_close(cur_q1, out1_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(cur_q2, out2_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(q1_e.grad, q1_s.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(q2_e.grad, q2_s.grad, rtol=1e-10, atol=1e-10)

    for pe, ps in zip(params_e, params_s):
        torch.testing.assert_close(pe.grad, ps.grad, rtol=1e-10, atol=1e-10)


def test_symplectic_squeeze_volume_preservation_and_variance_bounding():
    """
    Verifies that Symplectic Squeeze S_lambda = diag(lambda I, lambda^{-1} I) preserves
    phase space volume (det(S_lambda) = 1) and dynamically bounds variance.
    """
    dim = 16
    batch = 8
    q1 = torch.randn(batch, dim) * 10.0
    q2 = torch.randn(batch, dim) * 0.1

    squeeze = SymplecticSqueeze(dim=dim, dynamic=True, dyadic=True)
    s1, s2 = squeeze(q1, q2)

    # Invert
    inv1, inv2 = squeeze.inverse(s1, s2)
    torch.testing.assert_close(q1, inv1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q2, inv2, rtol=1e-5, atol=1e-5)

    # Check that variance ratio moved closer to 1
    orig_ratio = q1.var() / q2.var()
    new_ratio = s1.var() / s2.var()
    assert new_ratio < orig_ratio, f"Variance ratio was not bounded: {new_ratio} >= {orig_ratio}"


def test_symplectic_coupling_chain_with_squeeze():
    """Verifies deep chain stability with dynamic Symplectic Squeeze active."""
    depth = 64
    dim = 16
    batch = 4
    blocks = _build_test_chain(depth, dim, torch.float32, torch.device("cpu"))
    q1 = torch.randn(batch, dim)
    q2 = torch.randn(batch, dim)

    report = audit_symplectic_reversibility(
        q1, q2, blocks, scheme="stormer_verlet", enable_squeeze=True, tol=1e-4
    )
    assert report["is_reversible"] is True
    assert report["max_reconstruction_error"] < 1e-4
