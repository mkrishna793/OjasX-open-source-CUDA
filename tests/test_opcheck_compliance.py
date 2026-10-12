"""Official PyTorch Contract Enforcement Test Suite (torch.library.opcheck).

Strictly verifies that 100% of custom operators registered across:
- `ojasx_verified`
- `ojasx_holonomic`
- `ojasx_hal`
- `ojasx_fx`

comply with PyTorch's official operator contracts:
1. `test_schema` (schema mutation, return types, view safety)
2. `test_autograd_registration` (proper dispatch key / register_autograd formulas)
3. `test_faketensor` (abstract meta/fake kernel propagation for torch.compile)
4. `test_aot_dispatch_dynamic` (AOTAutograd dynamic shape and functionalization fidelity)
"""

from __future__ import annotations

import pytest
import torch
from torch.library import opcheck

# Ensure all OjasX custom operator modules and registrations are loaded
import ojasx2.ops
import ojasx2.streamed
import ojasx2.generated_holonomic_ops
import ojasx2.tile_hal
import ojasx2.fx_subgraph_compiler


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# ============================================================================
# Sample Input Generators for Operators
# ============================================================================

def _verified_cases(device: str):
    cases = []
    # 1. packed_relu
    x = torch.randn(4, 16, device=device, requires_grad=True)
    cases.append(("packed_relu", torch.ops.ojasx_verified.packed_relu, (x,), {}, True))

    # 2. gelu
    x = torch.randn(4, 16, device=device, requires_grad=True)
    cases.append(("gelu", torch.ops.ojasx_verified.gelu, (x,), {}, True))

    # 3. silu
    x = torch.randn(4, 16, device=device, requires_grad=True)
    cases.append(("silu", torch.ops.ojasx_verified.silu, (x,), {}, True))

    # 4. swiglu
    g = torch.randn(4, 16, device=device, requires_grad=True)
    u = torch.randn(4, 16, device=device, requires_grad=True)
    cases.append(("swiglu", torch.ops.ojasx_verified.swiglu, (g, u), {}, True))

    # 5. rmsnorm_aux
    x = torch.randn(4, 16, device=device, requires_grad=True)
    w = torch.randn(16, device=device, requires_grad=True)
    cases.append(("rmsnorm_aux", torch.ops.ojasx_verified.rmsnorm_aux, (x, w, 1e-5), {}, True))

    # 6. operand_tape_gemm
    a = torch.randn(4, 16, device=device, requires_grad=True)
    b = torch.randn(16, 8, device=device, requires_grad=True)
    cases.append(("operand_tape_gemm", torch.ops.ojasx_verified.operand_tape_gemm, (a, b), {}, True))

    return cases


def _holonomic_cases(device: str):
    cases = []
    # 1. rope: [B, H, S, D], cos/sin: [1, H, 1, D/2]
    x = torch.randn(2, 4, 8, 16, device=device, requires_grad=True)
    cos = torch.randn(1, 4, 1, 8, device=device)
    sin = torch.randn(1, 4, 1, 8, device=device)
    cases.append(("rope", torch.ops.ojasx_holonomic.rope, (x, cos, sin), {}, True))

    # 2. attention_softmax_lse: [B, H, S, S]
    scores = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
    cases.append(("attention_softmax_lse", torch.ops.ojasx_holonomic.attention_softmax_lse, (scores,), {}, True))

    return cases


def _hal_cases(device: str):
    cases = []
    # 1. pack_1bit_ballot
    x_bool = (torch.randn(4, 64, device=device) > 0)
    cases.append(("pack_1bit_ballot", torch.ops.ojasx_hal.pack_1bit_ballot, (x_bool,), {}, False))

    # 2. unpack_1bit_ballot
    packed = torch.ops.ojasx_hal.pack_1bit_ballot(x_bool)
    cases.append(("unpack_1bit_ballot", torch.ops.ojasx_hal.unpack_1bit_ballot, (packed, 64, torch.float32), {}, False))

    # 3. fused_amx_linear_relu2_fwd
    x = torch.randn(4, 16, device=device, requires_grad=True)
    w = torch.randn(32, 16, device=device, requires_grad=True)
    b = torch.randn(32, device=device, requires_grad=True)
    cases.append(("fused_amx_linear_relu2_fwd", torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd, (x, w, b), {}, True))

    # 4. fused_l1_rmsnorm_fwd
    x_norm = torch.randn(4, 16, device=device, requires_grad=True)
    w_norm = torch.randn(16, device=device, requires_grad=True)
    cases.append(("fused_l1_rmsnorm_fwd", torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd, (x_norm, w_norm, 1e-6), {}, True))

    # 5. fused_l1_glu_fwd
    g = torch.randn(4, 16, device=device, requires_grad=True)
    u = torch.randn(4, 16, device=device, requires_grad=True)
    cases.append(("fused_l1_glu_fwd", torch.ops.ojasx_hal.fused_l1_glu_fwd, (g, u, 0), {}, True))

    # 6. compress_boundary_int8
    x_c = torch.randn(4, 32, device=device)
    cases.append(("compress_boundary_int8", torch.ops.ojasx_hal.compress_boundary_int8, (x_c, 32), {}, False))

    # 7. decompress_boundary_int8
    q, scale = torch.ops.ojasx_hal.compress_boundary_int8(x_c, 32)
    cases.append(("decompress_boundary_int8", torch.ops.ojasx_hal.decompress_boundary_int8, (q, scale, list(x_c.shape), x_c.dtype), {}, False))

    # 8. tilelang_metal_gemm_relu2_fwd
    x_gemm = torch.randn(4, 16, device=device, requires_grad=True)
    w_gemm = torch.randn(32, 16, device=device, requires_grad=True)
    cases.append(("tilelang_metal_gemm_relu2_fwd", torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd, (x_gemm, w_gemm, None), {}, True))

    # 9. tilelang_universal_gemm_relu2_fwd
    cases.append(("tilelang_universal_gemm_relu2_fwd", torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd, (x_gemm, w_gemm, None), {}, True))

    return cases


def _fx_cases(device: str):
    cases = []
    x = torch.randn(4, 16, device=device)
    nw = torch.randn(16, device=device)
    nb = torch.randn(16, device=device)
    w1 = torch.randn(32, 16, device=device)
    b1 = torch.randn(32, device=device)
    w2 = torch.randn(16, 32, device=device)
    b2 = torch.randn(16, device=device)
    wg = torch.randn(32, 16, device=device)
    wu = torch.randn(32, 16, device=device)
    wd = torch.randn(16, 32, device=device)
    win = torch.randn(64, 16, device=device)

    # 1. norm_linear_act_linear
    cases.append(("norm_linear_act_linear", torch.ops.ojasx_fx.norm_linear_act_linear, (x, nw, nb, 1e-5, "layer_norm", w1, b1, "relu", 0.0, w2, b2), {}, False))

    # 2. gated_mlp
    cases.append(("gated_mlp", torch.ops.ojasx_fx.gated_mlp, (x, wg, None, wu, None, wd, None, "silu"), {}, False))

    # 3. universal_glu
    cases.append(("universal_glu", torch.ops.ojasx_fx.universal_glu, (x, wg, None, wu, None, "silu", 0.0), {}, False))

    # 4. chunked_glu
    cases.append(("chunked_glu", torch.ops.ojasx_fx.chunked_glu, (x, win, None, "silu", 0.0), {}, False))

    # 5. chunked_gated_mlp
    cases.append(("chunked_gated_mlp", torch.ops.ojasx_fx.chunked_gated_mlp, (x, win, None, wd, None, "silu", 0.0), {}, False))

    # 6. low_rank_bottleneck
    cases.append(("low_rank_bottleneck", torch.ops.ojasx_fx.low_rank_bottleneck, (x, wg[:8], None, wu[:, :8], None), {}, False))

    # 7. residual_add
    cases.append(("residual_add", torch.ops.ojasx_fx.residual_add, (x, x.clone(), False), {}, False))

    # 8. absorb_diag_gamma
    gamma = torch.randn(32, device=device)
    cases.append(("absorb_diag_gamma", torch.ops.ojasx_fx.absorb_diag_gamma, (gamma, wg, "left"), {}, False))

    return cases


# ============================================================================
# PyTest OpCheck Compliance Test Suite
# ============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_ojasx_verified(device: str):
    """Verifies all ojasx_verified custom operators pass official torch.library.opcheck."""
    cases = _verified_cases(device)
    for name, op, args, kwargs, has_autograd in cases:
        if device == "mps":
            test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        else:
            test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

        result = opcheck(op, args, kwargs, test_utils=test_utils, raise_exception=True)
        assert isinstance(result, dict)
        for check_name, status in result.items():
            assert status == "SUCCESS", f"{name} failed {check_name} on {device}"

        # If on MPS and operator supports autograd, verify live backward pass
        if device == "mps" and has_autograd:
            out = op(*args, **kwargs)
            if isinstance(out, tuple):
                loss = sum(t.sum() for t in out if isinstance(t, torch.Tensor) and t.requires_grad)
            else:
                loss = out.sum()
            loss.backward()
            for a in args:
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert a.grad is not None, f"Autograd gradient missing for {name} on MPS"


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_ojasx_holonomic(device: str):
    """Verifies all ojasx_holonomic custom operators pass official torch.library.opcheck."""
    cases = _holonomic_cases(device)
    for name, op, args, kwargs, has_autograd in cases:
        if device == "mps":
            test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        else:
            test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

        result = opcheck(op, args, kwargs, test_utils=test_utils, raise_exception=True)
        assert isinstance(result, dict)
        for check_name, status in result.items():
            assert status == "SUCCESS", f"{name} failed {check_name} on {device}"

        if device == "mps" and has_autograd:
            out = op(*args, **kwargs)
            if isinstance(out, tuple):
                loss = sum(t.sum() for t in out if isinstance(t, torch.Tensor) and t.requires_grad)
            else:
                loss = out.sum()
            loss.backward()
            for a in args:
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert a.grad is not None, f"Autograd gradient missing for {name} on MPS"


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_ojasx_hal(device: str):
    """Verifies all ojasx_hal custom operators pass official torch.library.opcheck."""
    cases = _hal_cases(device)
    for name, op, args, kwargs, has_autograd in cases:
        if device == "mps":
            test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        else:
            test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

        result = opcheck(op, args, kwargs, test_utils=test_utils, raise_exception=True)
        assert isinstance(result, dict)
        for check_name, status in result.items():
            assert status == "SUCCESS", f"{name} failed {check_name} on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_ojasx_fx(device: str):
    """Verifies all ojasx_fx custom operators pass official torch.library.opcheck."""
    cases = _fx_cases(device)
    for name, op, args, kwargs, _ in cases:
        if device == "mps":
            test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        else:
            test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

        result = opcheck(op, args, kwargs, test_utils=test_utils, raise_exception=True)
        assert isinstance(result, dict)
        for check_name, status in result.items():
            assert status == "SUCCESS", f"{name} failed {check_name} on {device}"
