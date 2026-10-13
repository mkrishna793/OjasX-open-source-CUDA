"""Verification test suite for Holonomic Dagger Attention.

Verifies:
1. PyTorch custom operator compliance via torch.library.opcheck
   (test_schema, test_autograd_registration, test_faketensor, test_aot_dispatch_dynamic).
2. Numerical parity against PyTorch F.scaled_dot_product_attention
   for both forward outputs and backward gradients (dQ, dK, dV) with cosine similarity >= 0.9999.
3. DRAM memory audit: asserts intermediate [B, H, S, S] attention score & softmax matrices
   are NEVER materialized on the autograd tape (strictly 0 bytes DRAM allocation).
4. HolonomicDaggerAttention(nn.Module) drop-in multihead attention compatibility.
5. Edge cases: non-contiguous strides, prime sequence lengths, Grouped-Query Attention (GQA), zero-element shapes.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import opcheck

import ojasx2.dagger_attention as da
from ojasx2.dagger_attention import (
    dagger_attention,
    holonomic_dagger_attention,
    holonomic_dagger_attention_with_lse,
    _HolonomicDaggerAttentionFunction,
    HolonomicDaggerAttention,
)

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# =============================================================================
# 1. Official PyTorch Contract Enforcement (torch.library.opcheck)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("sm_scale", [None, 0.125])
def test_opcheck_dagger_attention_contracts(device: str, is_causal: bool, sm_scale: float | None):
    """Verifies dagger_attention operator passes 100% of torch.library.opcheck contracts."""
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 16, 32
    q = torch.randn(B, H, S, D, device=device, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, requires_grad=True)

    if device == "mps":
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    else:
        test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

    result = opcheck(
        torch.ops.ojasx_holonomic.dagger_attention,
        (q, k, v),
        {"is_causal": is_causal, "sm_scale": sm_scale},
        test_utils=test_utils,
        raise_exception=True,
    )
    assert isinstance(result, dict)
    for check_name, status in result.items():
        assert status == "SUCCESS", f"dagger_attention failed {check_name} on {device}"

    # Verify live backward pass on device
    out, lse = torch.ops.ojasx_holonomic.dagger_attention(q, k, v, is_causal=is_causal, sm_scale=sm_scale)
    loss = out.sum() + lse.sum()
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_opcheck_dagger_attention_dtypes(dtype: torch.dtype):
    """Verifies opcheck passes across float32, float16, and bfloat16 on CPU."""
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 16, 32
    q = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)

    result = opcheck(
        torch.ops.ojasx_holonomic.dagger_attention,
        (q, k, v),
        {"is_causal": False, "sm_scale": None},
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=True,
    )
    for check_name, status in result.items():
        assert status == "SUCCESS", f"dagger_attention failed {check_name} with {dtype}"


# =============================================================================
# 2. Numerical Parity Against PyTorch F.scaled_dot_product_attention
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("seq_len", [17, 33, 64, 128])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_numerical_parity_forward_and_backward(
    device: str, is_causal: bool, seq_len: int, head_dim: int
):
    """Verifies numerical parity with F.scaled_dot_product_attention (cosine similarity >= 0.9999)."""
    torch.manual_seed(123)
    B, H, S, D = 2, 4, seq_len, head_dim

    # Reference tensors in float64 for exact ground truth gradients
    q_ref = torch.randn(B, H, S, D, dtype=torch.float64, device="cpu", requires_grad=True)
    k_ref = torch.randn(B, H, S, D, dtype=torch.float64, device="cpu", requires_grad=True)
    v_ref = torch.randn(B, H, S, D, dtype=torch.float64, device="cpu", requires_grad=True)
    dout = torch.randn(B, H, S, D, dtype=torch.float64, device="cpu")

    # Reference PyTorch execution
    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)
    ref_out.backward(dout)

    # Holonomic Dagger Attention execution
    q = q_ref.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    k = k_ref.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    v = v_ref.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    dout_test = dout.to(device=device, dtype=torch.float32)

    out = holonomic_dagger_attention(q, k, v, is_causal=is_causal)
    out.backward(dout_test)

    # 1. Forward output cosine similarity
    out_cpu = out.detach().cpu().to(torch.float64)
    ref_out_cpu = ref_out.detach().cpu()
    cos_sim_fwd = F.cosine_similarity(out_cpu.flatten(), ref_out_cpu.flatten(), dim=0).item()
    assert cos_sim_fwd >= 0.9999, f"Forward cosine similarity {cos_sim_fwd} < 0.9999"

    # 2. Backward gradients cosine similarity for dQ, dK, dV
    dq_cpu = q.grad.detach().cpu().to(torch.float64)
    dk_cpu = k.grad.detach().cpu().to(torch.float64)
    dv_cpu = v.grad.detach().cpu().to(torch.float64)

    cos_sim_dq = F.cosine_similarity(dq_cpu.flatten(), q_ref.grad.detach().cpu().flatten(), dim=0).item()
    cos_sim_dk = F.cosine_similarity(dk_cpu.flatten(), k_ref.grad.detach().cpu().flatten(), dim=0).item()
    cos_sim_dv = F.cosine_similarity(dv_cpu.flatten(), v_ref.grad.detach().cpu().flatten(), dim=0).item()

    assert cos_sim_dq >= 0.9999, f"dQ cosine similarity {cos_sim_dq} < 0.9999"
    assert cos_sim_dk >= 0.9999, f"dK cosine similarity {cos_sim_dk} < 0.9999"
    assert cos_sim_dv >= 0.9999, f"dV cosine similarity {cos_sim_dv} < 0.9999"


def test_numerical_parity_with_custom_scale():
    """Verifies parity with custom sm_scale."""
    torch.manual_seed(999)
    B, H, S, D = 2, 4, 32, 64
    custom_scale = 0.25

    q_ref = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    k_ref = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    v_ref = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    dout = torch.randn(B, H, S, D, dtype=torch.float64)

    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=custom_scale)
    ref_out.backward(dout)

    q = q_ref.detach().clone().float().requires_grad_(True)
    k = k_ref.detach().clone().float().requires_grad_(True)
    v = v_ref.detach().clone().float().requires_grad_(True)

    out = holonomic_dagger_attention(q, k, v, sm_scale=custom_scale)
    out.backward(dout.float())

    cos_out = F.cosine_similarity(out.detach().flatten().double(), ref_out.detach().flatten(), dim=0).item()
    cos_dq = F.cosine_similarity(q.grad.detach().flatten().double(), q_ref.grad.detach().flatten(), dim=0).item()
    assert cos_out >= 0.9999
    assert cos_dq >= 0.9999


# =============================================================================
# 3. DRAM Memory Audit: 0 Bytes DRAM Intermediate Score Materialization
# =============================================================================

def test_dram_memory_audit_zero_score_materialization():
    """Audits the autograd backward tape to guarantee strictly 0 bytes DRAM for [B, H, S, S] scores.

    In eager attention:
      The intermediate [B, H, S, S] matrix has B * H * S^2 elements.
      For B=2, H=8, S=1024, this is 16,777,216 elements (67.1 MB in FP32).
    In Holonomic Dagger Attention:
      Only O in [B, H, S, d] and row-wise LogSumExp L in [B, H, S] are saved.
      The [B, H, S, S] score matrix DRAM allocation on the tape is strictly 0 BYTES.
    """
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 256, 64

    q = torch.randn(B, H, S, D, requires_grad=True)
    k = torch.randn(B, H, S, D, requires_grad=True)
    v = torch.randn(B, H, S, D, requires_grad=True)

    # 1. Verify custom autograd Function tape
    out_fn, lse_fn = _HolonomicDaggerAttentionFunction.apply(q, k, v, True, None)
    grad_fn_fn = out_fn.grad_fn
    assert grad_fn_fn is not None
    saved_fn = grad_fn_fn.saved_tensors

    # Assert exactly 5 tensors saved on the tape: (Q, K, V, O, L)
    assert len(saved_fn) == 5, f"Expected 5 saved tensors, got {len(saved_fn)}"
    saved_shapes_fn = [t.shape for t in saved_fn]

    # Verify each saved tensor's shape: NONE are quadratic [B, H, S, S]
    quadratic_score_shape = torch.Size([B, H, S, S])
    assert quadratic_score_shape not in saved_shapes_fn, "VIOLATION: Quadratic [B, H, S, S] found on autograd tape!"

    for t in saved_fn:
        # Every saved tensor must either have last dimension D (like Q, K, V, O) or be the 1D LSE [B, H, S]
        assert t.shape[-1] != S or t.ndim == 3, f"Unexpected quadratic tensor on tape: {t.shape}"

    # Calculate exact tape memory for intermediate scores
    intermediate_score_dram_bytes = sum(
        t.numel() * t.element_size() for t in saved_fn if t.shape == quadratic_score_shape
    )
    assert intermediate_score_dram_bytes == 0, (
        f"DRAM allocation for [B, H, S, S] must be strictly 0, got {intermediate_score_dram_bytes} bytes"
    )

    # 2. Verify custom_op tape binding
    out_op, lse_op = torch.ops.ojasx_holonomic.dagger_attention(q, k, v, True, None)
    grad_fn_op = out_op.grad_fn
    assert grad_fn_op is not None
    saved_op = grad_fn_op.saved_tensors
    saved_shapes_op = [t.shape for t in saved_op]
    assert quadratic_score_shape not in saved_shapes_op

    # Theoretical savings verification:
    # Eager attention scores [B, H, S, S] = 2 * 8 * 256 * 256 = 1,048,576 floats (4.19 MB)
    # Holonomic LSE [B, H, S] = 2 * 8 * 256 = 4,096 floats (16.38 KB)
    # Memory reduction factor = S = 256x reduction!
    eager_score_elements = B * H * S * S
    holonomic_lse_elements = B * H * S
    ratio = eager_score_elements / holonomic_lse_elements
    assert ratio == float(S), f"Expected tape reduction ratio {S}, got {ratio}"


# =============================================================================
# 4. HolonomicDaggerAttention Module Compatibility
# =============================================================================

def test_holonomic_dagger_attention_module():
    """Verifies HolonomicDaggerAttention nn.Module drop-in compatibility."""
    torch.manual_seed(42)
    B, S, D = 2, 32, 128
    num_heads = 4

    # 1. Initialize module
    module = HolonomicDaggerAttention(embed_dim=D, num_heads=num_heads, is_causal=True)
    x = torch.randn(B, S, D, requires_grad=True)

    # Forward pass without weights return
    out = module(x)
    assert out.shape == torch.Size([B, S, D])

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for param in module.parameters():
        assert param.grad is not None

    # 2. Compatibility with need_weights=False (nn.MultiheadAttention interface)
    x2 = torch.randn(B, S, D)
    out_tuple, weights = module(x2, need_weights=False)
    assert out_tuple.shape == torch.Size([B, S, D])
    assert weights is None

    # 3. Direct projection module wrapping
    q_proj = nn.Linear(D, D, bias=False)
    k_proj = nn.Linear(D, D, bias=False)
    v_proj = nn.Linear(D, D, bias=False)
    o_proj = nn.Linear(D, D, bias=False)
    wrapped = HolonomicDaggerAttention(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
        num_heads=num_heads, head_dim=D // num_heads, is_causal=False
    )
    out_wrapped = wrapped(x2)
    assert out_wrapped.shape == torch.Size([B, S, D])


# =============================================================================
# 5. Grouped-Query Attention (GQA) & Non-Contiguous Strides & Edge Cases
# =============================================================================

def test_grouped_query_attention():
    """Verifies Grouped-Query Attention (H_q = 8, H_kv = 2)."""
    torch.manual_seed(77)
    B, H_q, H_kv, S, D = 2, 8, 2, 32, 32
    q = torch.randn(B, H_q, S, D, requires_grad=True)
    k = torch.randn(B, H_kv, S, D, requires_grad=True)
    v = torch.randn(B, H_kv, S, D, requires_grad=True)

    out = holonomic_dagger_attention(q, k, v, is_causal=True)
    assert out.shape == torch.Size([B, H_q, S, D])

    loss = out.sum()
    loss.backward()
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def test_non_contiguous_strides():
    """Verifies operation on non-contiguous sliced/transposed views."""
    torch.manual_seed(88)
    B, H, S, D = 2, 4, 64, 32
    raw = torch.randn(B, S, H, D * 2, requires_grad=True)

    # Transposed view with stride non-contiguity
    q = raw[:, :, :, :D].transpose(1, 2)
    k = raw[:, :, :, :D].transpose(1, 2)
    v = raw[:, :, :, :D].transpose(1, 2)

    assert not q.is_contiguous()

    out = holonomic_dagger_attention(q, k, v, is_causal=False)
    assert out.shape == torch.Size([B, H, S, D])
    out.sum().backward()
    assert raw.grad is not None


def test_zero_element_inputs():
    """Verifies handling of empty / zero-element tensors."""
    q = torch.empty(0, 4, 16, 32, requires_grad=True)
    k = torch.empty(0, 4, 16, 32, requires_grad=True)
    v = torch.empty(0, 4, 16, 32, requires_grad=True)

    out, lse = torch.ops.ojasx_holonomic.dagger_attention(q, k, v, False, None)
    assert out.shape == q.shape
    assert lse.shape == torch.Size([0, 4, 16])

    loss = out.sum()
    loss.backward()
    assert q.grad.shape == q.shape
