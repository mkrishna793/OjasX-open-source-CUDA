"""
tests/test_math_canonical_subsystem.py

Comprehensive Verification Suite for Canonical Math Subsystem (ojasx2.math):
1. Symplectic Manifold Mechanics & 2nd-Order Störmer-Verlet (FSAL, det J = 1, FP32 registers).
2. Structured Low-Rank Adjoints (O(nr) pull/push vs autograd and materialized Jacobians).
3. Halley-Padé 3rd-Order Root Inversion & 1-bit boolean sign packing.
4. Streaming Linear Operators (tiled Dagger attention & fused linear cross entropy).
5. Native C++/Metal acceleration hooks.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx2.math as om


# =============================================================================
# 1. Symplectic Manifold Mechanics & Störmer-Verlet Tests
# =============================================================================

def test_symplectic_manifold_geometry():
    dim = 8
    manifold = om.SymplecticManifold(dim=dim)
    assert manifold.dim == dim
    assert manifold.phase_dim == 2 * dim

    J = manifold.canonical_matrix(dtype=torch.float64)
    # 1. Skew-symmetry: J^T = -J
    assert torch.allclose(J.T, -J)
    # 2. Involutive complex structure: J^2 = -I
    eye = torch.eye(2 * dim, dtype=torch.float64)
    assert torch.allclose(torch.matmul(J, J), -eye)
    # 3. Determinant = 1
    assert math.isclose(torch.linalg.det(J).item(), 1.0, rel_tol=1e-6)

    # 4. Invariance check
    assert manifold.is_symplectic_matrix(J)

    # 5. Split and merge
    x = torch.randn(4, 2 * dim)
    q, p = manifold.split(x)
    assert q.shape == (4, dim) and p.shape == (4, dim)
    x_rec = manifold.merge(q, p)
    assert torch.allclose(x, x_rec)

    # 6. Symplectic form evaluation
    u = torch.randn(4, 2 * dim)
    v = torch.randn(4, 2 * dim)
    omega_val = manifold.omega(u, v)
    assert omega_val.shape == (4,)


def test_stormer_verlet_single_step_inversion():
    torch.manual_seed(42)
    dim = 16
    q = torch.randn(2, dim)
    p = torch.randn(2, dim)

    class NonLinearForce(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        def forward(self, x):
            return 0.1 * torch.tanh(self.linear(x))

    f = NonLinearForce()
    g = NonLinearForce()

    integrator = om.StormerVerletIntegrator(dt=0.5, persistent_fp32=True)
    q1, p1 = integrator.step(q, p, f, g)
    q0_rec, p0_rec = integrator.inverse_step(q1, p1, f, g)

    assert torch.allclose(q, q0_rec, atol=1e-6)
    assert torch.allclose(p, p0_rec, atol=1e-6)


def test_stormer_verlet_multi_stage_fsal_chain():
    torch.manual_seed(101)
    dim = 16
    q = torch.randn(4, dim)
    p = torch.randn(4, dim)

    stages = []
    for _ in range(8):
        f = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        g = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        with torch.no_grad():
            f[2].weight.mul_(0.01)
            g[2].weight.mul_(0.01)
        stages.append((f, g))

    integrator = om.StormerVerletIntegrator(dt=1.0, persistent_fp32=True)
    q_out, p_out, squeezes = integrator.integrate_chain(q, p, stages, enable_squeeze=True)
    q_rec, p_rec = integrator.reconstruct_chain(q_out, p_out, stages, enable_squeeze=True, squeezes=squeezes)

    drift_q = (q_rec - q).abs().max().item()
    drift_p = (p_rec - p).abs().max().item()
    assert max(drift_q, drift_p) < 1e-5

    audit = om.audit_symplectic_reversibility(q, p, stages, dt=1.0)
    assert audit["is_reversible"]
    assert audit["max_reconstruction_error"] < 1e-5


def test_symplectic_squeeze_volume_preservation():
    torch.manual_seed(202)
    dim = 8
    q = torch.randn(4, dim) * 10.0
    p = torch.randn(4, dim) * 0.1

    squeeze = om.SymplecticSqueeze(dim=dim, dynamic=True, dyadic=True)
    q_s, p_s = squeeze(q, p)
    q_rec, p_rec = squeeze.inverse(q_s, p_s)

    assert torch.allclose(q, q_rec, atol=1e-6)
    assert torch.allclose(p, p_rec, atol=1e-6)


# =============================================================================
# 2. Structured Low-Rank Adjoints Tests
# =============================================================================

def test_structured_derivative_properties_and_push_pull():
    torch.manual_seed(303)
    batch, n, r = 2, 8, 2
    d = torch.randn(batch, n)
    left = torch.randn(batch, n, r)
    right = torch.randn(batch, n, r)

    sd = om.StructuredDerivative(d, left, right)
    assert sd.dim == n
    assert sd.rank == r
    assert sd.batch_shape == torch.Size([batch])

    J_dense = sd.materialize()
    assert J_dense.shape == (batch, n, n)

    # Test pull(w) vs J^T @ w
    w = torch.randn(batch, n)
    pull_sd = sd.pull(w)
    pull_dense = torch.matmul(J_dense.transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(pull_sd, pull_dense, atol=1e-5)

    # Test push(v) vs J @ v
    v = torch.randn(batch, n)
    push_sd = sd.push(v)
    push_dense = torch.matmul(J_dense, v.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(push_sd, push_dense, atol=1e-5)


def test_softmax_derivative_and_pullback_parity():
    torch.manual_seed(404)
    x = torch.randn(3, 10, requires_grad=True)
    probs = F.softmax(x, dim=-1)

    sd = om.softmax_derivative(probs)
    assert sd.rank == 1

    cotangent = torch.randn_like(probs)
    # Pull via StructuredDerivative
    dx_sd = sd.pull(cotangent)
    # Pull via softmax_pullback
    dx_pb = om.softmax_pullback(cotangent, probs)
    assert torch.allclose(dx_sd, dx_pb)

    # Ground truth via PyTorch autograd
    probs.backward(cotangent)
    assert torch.allclose(dx_pb, x.grad, atol=1e-5)


def test_rmsnorm_derivative_and_pullback_parity():
    torch.manual_seed(505)
    x = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(12, dtype=torch.float64)

    sd = om.rmsnorm_derivative(x, gamma=gamma, eps=1e-5)
    assert sd.rank == 1

    cotangent = torch.randn_like(x)
    dx_sd = sd.pull(cotangent)
    dx_pb = om.rmsnorm_pullback(cotangent, x, gamma=gamma, eps=1e-5)
    assert torch.allclose(dx_sd, dx_pb)

    # Ground truth via PyTorch autograd
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    y = x * r * gamma
    y.backward(cotangent)
    assert torch.allclose(dx_pb, x.grad, atol=1e-5)


def test_layernorm_derivative_and_pullback_parity():
    torch.manual_seed(606)
    x = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(16, dtype=torch.float64)

    sd = om.layernorm_derivative(x, gamma=gamma, eps=1e-5)
    assert sd.rank == 2

    cotangent = torch.randn_like(x)
    dx_sd = sd.pull(cotangent)
    dx_pb = om.layernorm_pullback(cotangent, x, gamma=gamma, eps=1e-5)
    assert torch.allclose(dx_sd, dx_pb)

    # Ground truth via PyTorch autograd
    c = x - x.mean(-1, keepdim=True)
    r = torch.rsqrt(c.square().mean(-1, keepdim=True) + 1e-5)
    y = c * r * gamma
    y.backward(cotangent)
    assert torch.allclose(dx_pb, x.grad, atol=1e-5)


# =============================================================================
# 3. Halley-Padé 3rd-Order Root Inversion Tests
# =============================================================================

def test_1bit_boolean_mask_pack_unpack():
    torch.manual_seed(707)
    for shape in [(16,), (4, 17), (2, 3, 35)]:
        mask = torch.rand(shape) > 0.5
        packed = om.pack_branch_bit(mask)
        unpacked = om.unpack_branch_bit(packed, shape)
        assert torch.equal(mask, unpacked)


@pytest.mark.parametrize("act", ["silu", "gelu", "gelu_exact", "sigmoid", "tanh", "relu", "leaky_relu", "relu2"])
def test_halley_invert_activation_accuracy(act: str):
    torch.manual_seed(808)
    # Range of values spanning negative and positive branches
    x = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5], dtype=torch.float32)
    z_star = om.get_activation_min_z(act)
    branch_bit = x >= z_star

    if act == "silu":
        y = F.silu(x)
    elif act == "gelu":
        y = F.gelu(x, approximate="tanh")
    elif act == "gelu_exact":
        y = F.gelu(x, approximate="none")
    elif act == "sigmoid":
        y = torch.sigmoid(x)
    elif act == "tanh":
        y = torch.tanh(x)
    elif act == "relu":
        y = F.relu(x)
    elif act == "leaky_relu":
        y = F.leaky_relu(x, negative_slope=0.01)
    elif act == "relu2":
        y = F.relu(x).square()

    x_rec = om.halley_invert_activation(y, branch_bit, act=act, max_iters=5)
    if act in ("relu", "relu2"):
        # For x >= 0, x_rec == x exactly. For x < 0, phi(x_rec) == y == 0.
        pos_mask = x >= 0
        assert (x_rec[pos_mask] - x[pos_mask]).abs().max().item() < 1e-4
        if act == "relu":
            assert torch.allclose(F.relu(x_rec), y, atol=1e-5)
        else:
            assert torch.allclose(F.relu(x_rec).square(), y, atol=1e-5)
    else:
        max_err = (x_rec - x).abs().max().item()
        assert max_err < 1e-4, f"Act {act} inversion error too high: {max_err}"



def test_halley_inverse_function_autograd_parity():
    torch.manual_seed(909)
    x = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    y_halley = om.HalleyInverseFunction.apply(x, "silu")

    grad_out = torch.randn_like(y_halley)
    y_halley.backward(grad_out)
    grad_halley = x.grad.clone()

    # Compare against eager autograd
    x_eager = x.detach().clone().requires_grad_(True)
    y_eager = F.silu(x_eager)
    y_eager.backward(grad_out)
    grad_eager = x_eager.grad

    assert torch.allclose(grad_halley, grad_eager, atol=1e-4)


# =============================================================================
# 4. Streaming Linear Operators Tests
# =============================================================================

def test_tiled_dagger_attention_execution():
    torch.manual_seed(1010)
    B, H, S, D = 2, 4, 32, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    out = om.tiled_dagger_attention(q, k, v, is_causal=True, tile_size=16)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_fused_linear_cross_entropy_execution():
    torch.manual_seed(1111)
    B, S, D, V = 2, 16, 32, 128
    h = torch.randn(B, S, D)
    w = torch.randn(V, D)
    targets = torch.randint(0, V, (B, S))

    loss = om.fused_linear_cross_entropy(h, w, targets, reduction="mean")
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


# =============================================================================
# 5. Native Acceleration Hook Tests
# =============================================================================

def test_native_backend_hooks():
    has_native = om.has_native_backend()
    assert isinstance(has_native, bool)

    backend_info = om.get_native_backend()
    assert isinstance(backend_info, dict)
    assert "has_native" in backend_info
    assert "tile_kernels_loaded" in backend_info
    assert "platform" in backend_info
