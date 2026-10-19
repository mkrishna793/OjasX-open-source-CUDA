import pytest
import torch
import ojasx2.math.csrc.ojasx_math_native as native


def test_native_import_and_metal_detection():
    assert native is not None
    assert hasattr(native, "stormer_verlet_fwd")
    assert hasattr(native, "structured_pullback_tensor")
    assert hasattr(native, "halley_invert_activation_tensor")
    assert hasattr(native, "is_metal_available")
    assert native.is_metal_available() is True


def test_stormer_verlet_fsal():
    q1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    q2 = torch.tensor([0.0, -1.0], dtype=torch.float32)

    def f(p):
        return p * 0.5

    def g(q):
        return -q * 0.2

    q1_out, q2_out = native.stormer_verlet_fwd(q1, q2, f, g, 0.1)
    assert q1_out.shape == q1.shape
    assert q2_out.shape == q2.shape


def test_structured_pullback():
    n, r = 64, 4
    torch.manual_seed(42)
    v = torch.randn(2, n)
    d = torch.randn(n)
    U = torch.randn(n, r)
    V = torch.randn(n, r)

    # Reference PyTorch J^T v
    w_ref = v * d + (v @ U) @ V.T

    # Native CPU
    w_cpu = native.structured_pullback_tensor(v, d, U, V, prefer_metal=False)
    assert torch.allclose(w_cpu, w_ref, atol=1e-4, rtol=1e-4)

    # Native Metal GPU
    w_metal = native.structured_pullback_tensor(v[0:1], d, U, V, prefer_metal=True)
    assert torch.allclose(w_metal, w_ref[0:1], atol=1e-4, rtol=1e-4)


def test_halley_inversion_silu():
    test_z = torch.tensor([-3.0, -1.2784645, 0.0, 1.0, 3.0], dtype=torch.float32)
    y_silu = test_z * torch.sigmoid(test_z)
    branch = test_z >= -1.2784645427610738

    # CPU
    z_cpu = native.halley_invert_activation_tensor(y_silu, branch, act="silu", iters=3, prefer_metal=False)
    assert torch.allclose(z_cpu, test_z, atol=1e-4)

    # Metal GPU
    z_metal = native.halley_invert_activation_tensor(y_silu, branch, act="silu", iters=3, prefer_metal=True)
    assert torch.allclose(z_metal, test_z, atol=1e-4)


def test_halley_inversion_gelu():
    test_z = torch.tensor([-2.5, -0.75179, 0.0, 1.0, 2.5], dtype=torch.float32)
    c = 0.7978845608028654
    u = c * (test_z + 0.044715 * test_z.pow(3))
    y_gelu = 0.5 * test_z * (1.0 + torch.tanh(u))
    branch = test_z >= -0.751791524138

    # CPU
    z_cpu = native.halley_invert_activation_tensor(y_gelu, branch, act="gelu", iters=3, prefer_metal=False)
    assert torch.allclose(z_cpu, test_z, atol=1e-3, rtol=1e-3)

    # Metal GPU
    z_metal = native.halley_invert_activation_tensor(y_gelu, branch, act="gelu", iters=3, prefer_metal=True)
    assert torch.allclose(z_metal, test_z, atol=1e-3, rtol=1e-3)
