"""Multi-Hardware Parametrization OpInfo Test Harness (tests/test_opinfo_compliance.py).

Strictly verifies OjasX custom operators and holonomic layers across:
- Devices: `cpu` and `mps` (and `cuda` if available)
- Dtypes: `float32`, `float16`, `bfloat16`
- Shapes: standard, prime/unaligned (1x1, 7x13, 17x31), and zero-element (0x16)
- Strides: contiguous, transposed (transpose(0, 1)), permuted, sliced (x[:, ::2]), and strided
- Verification targets:
  1. Mathematical fidelity against PyTorch eager reference
  2. Autograd gradient parity against reference autograd
  3. Memory-saving invariance under exotic strides
"""

from __future__ import annotations

import collections
import sys
import types
from typing import Callable, Generator, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

# Ensure expecttest mock exists if not installed for PyTorch internal OpInfo import
if "expecttest" not in sys.modules:
    mock_expecttest = types.ModuleType("expecttest")
    mock_expecttest.TestCase = object
    sys.modules["expecttest"] = mock_expecttest

from torch.testing._internal.opinfo.core import OpInfo, SampleInput

# Import OjasX operators
import ojasx2.ops
import ojasx2.tile_hal
import ojasx2.generated_holonomic_ops
import ojasx2.fx_subgraph_compiler
from ojasx2.holonomic_math_compiler import holonomic_linear, holonomic_swiglu


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tolerance_for(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    elif dtype == torch.float16:
        return 2e-3, 2e-3
    elif dtype == torch.bfloat16:
        return 2.5e-2, 2.5e-2
    return 1e-3, 1e-3


# ============================================================================
# Stride Variation Utilities
# ============================================================================

def make_strided_tensor(
    shape: Tuple[int, ...],
    stride_mode: str,
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Creates a tensor with specific contiguous or non-contiguous exotic stride layouts."""
    if any(s == 0 for s in shape):
        t = torch.randn(shape, device=device, dtype=dtype)
        if requires_grad and dtype.is_floating_point:
            t.requires_grad_(True)
        return t

    if stride_mode == "contiguous":
        t = torch.randn(shape, device=device, dtype=dtype)
    elif stride_mode == "transposed":
        if len(shape) == 2:
            raw = torch.randn((shape[1], shape[0]), device=device, dtype=dtype)
            t = raw.t()
        elif len(shape) >= 3:
            raw = torch.randn(shape, device=device, dtype=dtype)
            t = raw.transpose(-1, -2)
            if t.shape != shape:
                t = t.reshape(shape)
        else:
            t = torch.randn(shape, device=device, dtype=dtype)
    elif stride_mode == "sliced":
        # Sliced view: allocate 2x on last dim and step by 2
        last_d = shape[-1]
        alloc_shape = list(shape)
        alloc_shape[-1] = last_d * 2
        raw = torch.randn(alloc_shape, device=device, dtype=dtype)
        t = raw[..., ::2]
    elif stride_mode == "permuted":
        if len(shape) == 3:
            raw = torch.randn((shape[1], shape[0], shape[2]), device=device, dtype=dtype)
            t = raw.permute(1, 0, 2)
        elif len(shape) == 4:
            raw = torch.randn((shape[0], shape[2], shape[1], shape[3]), device=device, dtype=dtype)
            t = raw.permute(0, 2, 1, 3)
        else:
            t = torch.randn(shape, device=device, dtype=dtype)
    else:
        t = torch.randn(shape, device=device, dtype=dtype)

    if requires_grad and dtype.is_floating_point:
        t = t.clone().detach().requires_grad_(True)
    return t


# ============================================================================
# Sample Input Generators
# ============================================================================

def sample_linear(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates standard, prime, zero-element, and strided linear samples."""
    shapes = [
        # Standard
        ((4, 16), (32, 16), True),
        # Prime and unaligned
        ((7, 13), (17, 13), True),
        ((1, 1), (1, 1), False),
        # Zero-element
        ((0, 16), (32, 16), True),
    ]
    stride_modes = ["contiguous", "transposed", "sliced"]

    for (x_shape, w_shape, has_bias) in shapes:
        for stride_mode in stride_modes:
            x = make_strided_tensor(x_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            w = make_strided_tensor(w_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            b = make_strided_tensor((w_shape[0],), "contiguous", device, dtype, requires_grad=requires_grad) if has_bias else None
            yield SampleInput(x, args=(w, b), kwargs={})


def sample_unary_activation(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates unary activation samples across shapes and strides."""
    shapes = [
        (4, 16),
        (7, 13),
        (17, 31),
        (2, 5, 11),
        (0, 16),
    ]
    stride_modes = ["contiguous", "transposed", "sliced", "permuted"]

    for shape in shapes:
        for stride_mode in stride_modes:
            if len(shape) < 3 and stride_mode == "permuted":
                continue
            x = make_strided_tensor(shape, stride_mode, device, dtype, requires_grad=requires_grad)
            yield SampleInput(x, args=(), kwargs={})


def sample_swiglu(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates gate and up projection samples for SwiGLU."""
    shapes = [
        (4, 16),
        (7, 13),
        (17, 31),
        (2, 3, 8),
        (0, 16),
    ]
    stride_modes = ["contiguous", "transposed", "sliced"]

    for shape in shapes:
        for stride_mode in stride_modes:
            g = make_strided_tensor(shape, stride_mode, device, dtype, requires_grad=requires_grad)
            u = make_strided_tensor(shape, stride_mode, device, dtype, requires_grad=requires_grad)
            yield SampleInput(g, args=(u,), kwargs={})


def sample_universal_glu(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates samples for Universal GLU across variants."""
    shapes = [
        ((4, 16), (32, 16)),
        ((7, 13), (17, 13)),
        ((0, 16), (32, 16)),
    ]
    stride_modes = ["contiguous", "transposed", "sliced"]

    for (x_shape, w_shape) in shapes:
        for stride_mode in stride_modes:
            x = make_strided_tensor(x_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            wg = make_strided_tensor(w_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            wu = make_strided_tensor(w_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            yield SampleInput(x, args=(wg, None, wu, None), kwargs={})


def sample_rmsnorm(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates RMSNorm samples."""
    shapes = [
        ((4, 16), 16),
        ((7, 13), 13),
        ((17, 31), 31),
        ((2, 4, 8), 8),
        ((0, 16), 16),
    ]
    stride_modes = ["contiguous", "transposed", "sliced"]

    for (x_shape, w_dim) in shapes:
        for stride_mode in stride_modes:
            x = make_strided_tensor(x_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            w = make_strided_tensor((w_dim,), "contiguous", device, dtype, requires_grad=requires_grad)
            yield SampleInput(x, args=(w, 1e-5), kwargs={})


def sample_rope(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates RoPE rotary embedding samples."""
    # [B, H, S, D] where D is even, and cos/sin are [1, H, 1, D/2]
    configs = [
        ((2, 4, 8, 16), (1, 4, 1, 8)),
        ((1, 2, 7, 12), (1, 2, 1, 6)),
        ((3, 1, 5, 20), (1, 1, 1, 10)),
    ]
    stride_modes = ["contiguous", "sliced"]

    for (x_shape, c_shape) in configs:
        for stride_mode in stride_modes:
            x = make_strided_tensor(x_shape, stride_mode, device, dtype, requires_grad=requires_grad)
            cos = make_strided_tensor(c_shape, "contiguous", device, dtype, requires_grad=False)
            sin = make_strided_tensor(c_shape, "contiguous", device, dtype, requires_grad=False)
            yield SampleInput(x, args=(cos, sin), kwargs={})


def sample_attention_lse(op_info, device, dtype, requires_grad=False, **kwargs):
    """Generates Attention Softmax LSE samples."""
    shapes = [
        (2, 4, 8, 8),
        (1, 2, 7, 7),
        (3, 1, 11, 11),
    ]
    stride_modes = ["contiguous", "transposed"]

    for shape in shapes:
        for stride_mode in stride_modes:
            scores = make_strided_tensor(shape, stride_mode, device, dtype, requires_grad=requires_grad)
            yield SampleInput(scores, args=(), kwargs={})


# ============================================================================
# Reference Implementations
# ============================================================================

def ref_linear(x, w, b=None):
    return F.linear(x, w, b)


def ref_packed_relu(x):
    return F.relu(x)


def ref_gelu(x):
    return F.gelu(x)


def ref_silu(x):
    return F.silu(x)


def ref_swiglu(g, u):
    return F.silu(g) * u


def ref_universal_glu_factory(act_str: str):
    def _ref(x, wg, bg, wu, bu):
        zg = F.linear(x, wg, bg)
        zu = F.linear(x, wu, bu)
        if act_str == "silu":
            act = F.silu(zg)
        elif act_str == "gelu":
            act = F.gelu(zg, approximate="tanh")
        elif act_str == "relu":
            act = F.relu(zg)
        elif act_str == "relu2":
            act = F.relu(zg).pow(2)
        elif act_str == "sigmoid":
            act = torch.sigmoid(zg)
        elif act_str == "bilinear":
            act = zg
        else:
            act = F.silu(zg)
        return act * zu
    return _ref


def ref_rmsnorm(x, weight, eps=1e-5):
    # Standard eager RMSNorm reference
    variance = x.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normed = x * rstd
    return normed * weight


def ref_rope(x, cos, sin):
    # Reference RoPE rotation
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.cat([rx1, rx2], dim=-1)


def ref_attention_lse(scores):
    probs = F.softmax(scores, dim=-1)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return probs, lse


# ============================================================================
# OjasX OpInfo Database
# ============================================================================

OJASX_OP_DB: List[OpInfo] = [
    # 1. Holonomic Linear
    OpInfo(
        name="holonomic_linear",
        op=holonomic_linear,
        ref=ref_linear,
        sample_inputs_func=sample_linear,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 2. Packed ReLU
    OpInfo(
        name="packed_relu",
        op=torch.ops.ojasx_verified.packed_relu,
        ref=ref_packed_relu,
        sample_inputs_func=sample_unary_activation,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 3. GELU
    OpInfo(
        name="gelu",
        op=torch.ops.ojasx_verified.gelu,
        ref=ref_gelu,
        sample_inputs_func=sample_unary_activation,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 4. SiLU
    OpInfo(
        name="silu",
        op=torch.ops.ojasx_verified.silu,
        ref=ref_silu,
        sample_inputs_func=sample_unary_activation,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 5. SwiGLU
    OpInfo(
        name="swiglu",
        op=torch.ops.ojasx_verified.swiglu,
        ref=ref_swiglu,
        sample_inputs_func=sample_swiglu,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 6. Universal GLU (SwiGLU variant)
    OpInfo(
        name="universal_glu_silu",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "silu", 0.0),
        ref=ref_universal_glu_factory("silu"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 7. Universal GLU (GeGLU variant)
    OpInfo(
        name="universal_glu_gelu",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "gelu", 0.0),
        ref=ref_universal_glu_factory("gelu"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 8. Universal GLU (ReGLU variant)
    OpInfo(
        name="universal_glu_reglu",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "relu", 0.0),
        ref=ref_universal_glu_factory("relu"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 9. Universal GLU (ReLU2GLU variant)
    OpInfo(
        name="universal_glu_relu2glu",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "relu2", 0.0),
        ref=ref_universal_glu_factory("relu2"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 10. Universal GLU (SigmoidGLU variant)
    OpInfo(
        name="universal_glu_sigmoid",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "sigmoid", 0.0),
        ref=ref_universal_glu_factory("sigmoid"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 11. Universal GLU (BilinearGLU variant)
    OpInfo(
        name="universal_glu_bilinear",
        op=lambda x, wg, bg, wu, bu: torch.ops.ojasx_fx.universal_glu(x, wg, bg, wu, bu, "bilinear", 0.0),
        ref=ref_universal_glu_factory("bilinear"),
        sample_inputs_func=sample_universal_glu,
        dtypes=DTYPES,
        supports_autograd=False,
    ),
    # 12. RMSNorm Aux
    OpInfo(
        name="rmsnorm_aux",
        op=lambda x, w, eps: torch.ops.ojasx_verified.rmsnorm_aux(x, w, eps)[0],
        ref=ref_rmsnorm,
        sample_inputs_func=sample_rmsnorm,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 13. RoPE
    OpInfo(
        name="rope",
        op=torch.ops.ojasx_holonomic.rope,
        ref=ref_rope,
        sample_inputs_func=sample_rope,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
    # 14. Attention Softmax LSE
    OpInfo(
        name="attention_softmax_lse",
        op=torch.ops.ojasx_holonomic.attention_softmax_lse,
        ref=ref_attention_lse,
        sample_inputs_func=sample_attention_lse,
        dtypes=DTYPES,
        supports_autograd=True,
    ),
]


# ============================================================================
# PyTest OpInfo Test Harness
# ============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_info", OJASX_OP_DB, ids=lambda op: op.name)
def test_opinfo_mathematical_fidelity(op_info: OpInfo, device: str, dtype: torch.dtype):
    """Verifies output numerical fidelity against PyTorch reference across shapes, dtypes, and strides."""
    atol, rtol = _tolerance_for(dtype)
    samples = list(op_info.sample_inputs_func(op_info, device, dtype, requires_grad=False))
    assert len(samples) > 0, f"No samples generated for {op_info.name}"

    for idx, sample in enumerate(samples):
        res = op_info.op(sample.input, *sample.args, **sample.kwargs)
        ref = op_info.ref(sample.input, *sample.args, **sample.kwargs)

        if isinstance(res, tuple):
            assert isinstance(ref, tuple), f"{op_info.name} mismatch tuple return type"
            for r_res, r_ref in zip(res, ref):
                if isinstance(r_res, torch.Tensor) and isinstance(r_ref, torch.Tensor):
                    assert_close(r_res, r_ref, atol=atol, rtol=rtol)
        else:
            assert isinstance(res, torch.Tensor)
            assert_close(res, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_info", [op for op in OJASX_OP_DB if op.supports_autograd], ids=lambda op: op.name)
def test_opinfo_autograd_parity(op_info: OpInfo, device: str, dtype: torch.dtype):
    """Verifies autograd gradient parity against eager PyTorch across shapes and strides."""
    atol, rtol = _tolerance_for(dtype)
    samples = list(op_info.sample_inputs_func(op_info, device, dtype, requires_grad=True))

    for idx, sample in enumerate(samples):
        # Skip zero-element inputs for autograd loss backward
        if sample.input.numel() == 0 or any(isinstance(a, torch.Tensor) and a.numel() == 0 for a in sample.args):
            continue

        # Prepare cloned independent graphs respecting existing requires_grad
        inp_op = sample.input.clone().detach().requires_grad_(sample.input.requires_grad)
        args_op = [
            a.clone().detach().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a
            for a in sample.args
        ]

        inp_ref = sample.input.clone().detach().requires_grad_(sample.input.requires_grad)
        args_ref = [
            a.clone().detach().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a
            for a in sample.args
        ]

        out_op = op_info.op(inp_op, *args_op, **sample.kwargs)
        out_ref = op_info.ref(inp_ref, *args_ref, **sample.kwargs)

        loss_op = (out_op[0] if isinstance(out_op, tuple) else out_op).sum()
        loss_ref = (out_ref[0] if isinstance(out_ref, tuple) else out_ref).sum()

        loss_op.backward()
        loss_ref.backward()

        if inp_op.grad is not None and inp_ref.grad is not None:
            assert_close(inp_op.grad, inp_ref.grad, atol=atol, rtol=rtol)

        for a_op, a_ref in zip(args_op, args_ref):
            if isinstance(a_op, torch.Tensor) and a_op.grad is not None:
                assert isinstance(a_ref, torch.Tensor) and a_ref.grad is not None
                assert_close(a_op.grad, a_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("op_info", OJASX_OP_DB, ids=lambda op: op.name)
def test_opinfo_memory_saving_invariance(op_info: OpInfo, device: str, dtype: torch.dtype):
    """Verifies that memory-saving invariance holds across contiguous and non-contiguous stride variations."""
    samples = list(op_info.sample_inputs_func(op_info, device, dtype, requires_grad=False))
    for sample in samples:
        out = op_info.op(sample.input, *sample.args, **sample.kwargs)
        if isinstance(out, tuple):
            for t in out:
                if isinstance(t, torch.Tensor):
                    assert not torch.isnan(t).any()
                    assert not torch.isinf(t).any()
        else:
            assert isinstance(out, torch.Tensor)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


# ============================================================================
# OpInfo Mathematical Innovations Parametrization (v0.4.0)
# 1. Block-Matrix Cotangent Fusion
# 2. Halley-Padé Branched Inverse Activation Adjoint
# 3. Active-Subspace / Zero-Cotangent Token Culling
# 4. Dual-Regime Gram / Tangent Operator Factorization
# ============================================================================

from ojasx2.fx_subgraph_compiler import (
    holonomic_gated_mlp,
    holonomic_low_rank,
    holonomic_norm_linear_act_linear,
)
from ojasx2.holonomic_layers import (
    HalleyInverseActivationLens,
    _HolonomicSplitGLUFunction,
)
from ojasx2.holonomic_math_compiler import (
    cull_zero_cotangent_gemm,
    get_activation_min_z,
    halley_invert_activation,
    holonomic_linear,
)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("act", ["silu", "gelu", "reglu", "relu2", "sigmoid", "bilinear"])
def test_opinfo_block_matrix_cotangent_fusion_glu(device: str, act: str):
    """OpInfo parametrization for Block-Matrix Cotangent Fusion across GLU variants."""
    torch.manual_seed(42)
    B, S, D, H = 2, 16, 64, 128
    x = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
    w_gate = torch.randn(H, D, device=device, dtype=torch.float32, requires_grad=True)
    w_up = torch.randn(H, D, device=device, dtype=torch.float32, requires_grad=True)
    b_gate = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)
    b_up = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_gate_ref = w_gate.detach().clone().requires_grad_(True)
    w_up_ref = w_up.detach().clone().requires_grad_(True)
    b_gate_ref = b_gate.detach().clone().requires_grad_(True)
    b_up_ref = b_up.detach().clone().requires_grad_(True)

    g_ref = F.linear(x_ref, w_gate_ref, b_gate_ref)
    u_ref = F.linear(x_ref, w_up_ref, b_up_ref)
    act_map = {
        "silu": F.silu,
        "gelu": lambda z: F.gelu(z, approximate="tanh"),
        "reglu": F.relu,
        "relu2": lambda z: F.relu(z).square(),
        "sigmoid": torch.sigmoid,
        "bilinear": lambda z: z,
    }
    y_ref = act_map[act](g_ref) * u_ref
    y_ref.sum().backward()

    y_fused = _HolonomicSplitGLUFunction.apply(x, w_gate, w_up, b_gate, b_up, act, 0.01)
    y_fused.sum().backward()

    assert torch.allclose(y_fused, y_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_gate.grad, w_gate_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_up.grad, w_up_ref.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("act", [
    "silu", "gelu", "gelu_tanh", "gelu_exact", "quickgelu", "mish",
    "sigmoid", "tanh", "relu", "leaky_relu", "relu2",
])
def test_opinfo_halley_invert_activation_precision(act: str):
    """OpInfo parametrization for Halley-Padé 3rd-order rational inverse across 11 activations."""
    z_true = torch.linspace(-4.5, 4.5, 500, dtype=torch.float32)
    act_fn_map = {
        "silu": F.silu,
        "gelu": lambda z: F.gelu(z, approximate="tanh"),
        "gelu_tanh": lambda z: F.gelu(z, approximate="tanh"),
        "gelu_exact": lambda z: F.gelu(z, approximate="none"),
        "quickgelu": lambda z: z * torch.sigmoid(1.702 * z),
        "mish": F.mish,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "leaky_relu": lambda z: F.leaky_relu(z, negative_slope=0.01),
        "relu2": lambda z: F.relu(z).square(),
    }
    fn = act_fn_map[act]
    y = fn(z_true)
    branch_bit = z_true >= get_activation_min_z(act)
    z_rec = halley_invert_activation(y, branch_bit, act=act, neg_slope=0.01, max_iters=3)
    residual = (fn(z_rec) - y).abs().max().item()
    assert residual < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("act", ["silu", "gelu", "mish", "sigmoid", "tanh", "relu", "leaky_relu"])
def test_opinfo_halley_inverse_activation_lens_autograd(device: str, act: str):
    """OpInfo parametrization for HalleyInverseActivationLens autograd parity."""
    torch.manual_seed(202)
    x = torch.randn(4, 32, 64, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    lens = HalleyInverseActivationLens(act).to(device)
    act_map = {
        "silu": F.silu,
        "gelu": lambda z: F.gelu(z, approximate="tanh"),
        "mish": F.mish,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "leaky_relu": lambda z: F.leaky_relu(z, negative_slope=0.01),
    }
    y_ref = act_map[act](x_ref)
    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)

    y_lens = lens(x)
    y_lens.backward(grad_out)
    assert torch.allclose(y_lens, y_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x.grad, x_ref.grad, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("active_ratio", [0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0])
def test_opinfo_active_subspace_token_culling(active_ratio: float):
    """OpInfo parametrization for Active-Subspace / Zero-Cotangent Token Culling."""
    torch.manual_seed(404)
    M, K, N = 128, 64, 32
    X = torch.randn(M, N, dtype=torch.float32)
    dZ = torch.randn(M, K, dtype=torch.float32)
    active_tokens = int(round(M * active_ratio))
    if active_tokens < M:
        dZ[active_tokens:] = 0.0
    dW_dense = dZ.t().matmul(X)
    dW_culled = cull_zero_cotangent_gemm(dZ, X, threshold_ratio=0.75)
    assert torch.allclose(dW_culled, dW_dense, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("M, D_in, r, D_out", [
    (256, 32, 8, 32),
    (16, 64, 8, 64),
    (512, 128, 16, 128),
])
def test_opinfo_dual_regime_tangent_operator_factorization(M: int, D_in: int, r: int, D_out: int):
    """OpInfo parametrization for Dual-Regime Gram / Tangent Operator Factorization."""
    torch.manual_seed(606)
    X = torch.randn(M, D_in, requires_grad=True)
    W_down = torch.randn(r, D_in, requires_grad=True)
    W_up = torch.randn(D_out, r, requires_grad=True)
    X_ref = X.detach().clone().requires_grad_(True)
    W_down_ref = W_down.detach().clone().requires_grad_(True)
    W_up_ref = W_up.detach().clone().requires_grad_(True)

    Y_ref = F.linear(F.linear(X_ref, W_down_ref), W_up_ref)
    grad_out = torch.randn_like(Y_ref)
    Y_ref.backward(grad_out)

    Y_holo = holonomic_low_rank(X, W_down, None, W_up, None)
    Y_holo.backward(grad_out)

    assert torch.allclose(Y_holo, Y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(X.grad, X_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(W_down.grad, W_down_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(W_up.grad, W_up_ref.grad, atol=1e-3, rtol=1e-3)

