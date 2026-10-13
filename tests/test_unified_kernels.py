"""Comprehensive Test Suite for Unified Fused Forward-Backward Execution Kernels.

Validates Requirement R4 and Acceptance Criteria for OjasX v0.5.0:
1. `torch.library.opcheck` compliance (schema, autograd registration, faketensor, dynamic shapes).
2. Autograd dispatch node count reduction from 6-8 nodes to 1 single node per block.
3. Numerical output and backward gradient equivalence against standard PyTorch (cosine similarity >= 0.9999).
4. DRAM memory audit: confirms intermediate activations (X_norm, G, U, silu(G), H) are never materialized on the autograd tape.
5. End-to-end SmartWorkloadRouter optimization across Attention, Unified MLP, and LM Head.
6. FX Subgraph Compiler pattern rewriting for attention and loss subgraphs.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import opcheck

import ojasx2.tile_hal as tile_hal
from ojasx2.tile_hal import (
    HolonomicUnifiedTransformerMLP,
    HolonomicUnifiedTransformerMLPFunction,
)
from ojasx2.smart_router import optimize_model
from ojasx2.fx_subgraph_compiler import transform_subgraph_module


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# =============================================================================
# Reference PyTorch Transformer Block Implementation
# =============================================================================

class EagerReferenceTransformerMLP(nn.Module):
    """Standard eager PyTorch reference implementing RMSNorm -> SwiGLU -> Linear."""

    def __init__(self, dim: int, intermediate_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. RMSNorm
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        x_norm = ((x.float() * rstd) * self.norm_weight.float()).to(x.dtype)

        # 2. Gate & Up projections
        g = self.gate_proj(x_norm)
        u = self.up_proj(x_norm)

        # 3. SwiGLU activation
        h = F.silu(g) * u

        # 4. Down projection
        y = self.down_proj(h)
        return y


def _compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().reshape(-1).float()
    b_flat = b.detach().reshape(-1).float()
    if a_flat.norm() == 0 and b_flat.norm() == 0:
        return 1.0
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())


def _count_autograd_op_nodes(root_grad_fn) -> int:
    """Traverses PyTorch autograd graph and counts operation nodes on backward tape (excluding leaf AccumulateGrad)."""
    visited = set()
    stack = [root_grad_fn]
    op_count = 0
    while stack:
        fn = stack.pop()
        if fn is None or fn in visited:
            continue
        visited.add(fn)
        if "AccumulateGrad" not in fn.name():
            op_count += 1
        for next_fn, _ in fn.next_functions:
            if next_fn is not None and next_fn not in visited:
                stack.append(next_fn)
    return op_count


# =============================================================================
# 1. PyTorch Contract Enforcement: torch.library.opcheck
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_unified_kernel_opcheck(device: str):
    """Enforces official PyTorch opcheck across schema, autograd, faketensor, dynamic."""
    d = 16
    h = 32
    m = 4

    x = torch.randn(m, d, device=device, requires_grad=True)
    norm_w = torch.randn(d, device=device, requires_grad=True)
    gate_w = torch.randn(h, d, device=device, requires_grad=True)
    up_w = torch.randn(h, d, device=device, requires_grad=True)
    down_w = torch.randn(d, h, device=device, requires_grad=True)
    eps = 1e-6

    # 1. Check forward operator across contracts
    if device == "mps":
        test_utils = ["test_schema", "test_faketensor", "test_aot_dispatch_dynamic"]
    else:
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]

    opcheck(
        torch.ops.ojasx_hal.fused_transformer_mlp_fwd,
        (x, norm_w, gate_w, up_w, down_w, eps),
        test_utils=test_utils,
    )

    # 2. Check backward operator
    dy = torch.randn(m, d, device=device)
    opcheck(
        torch.ops.ojasx_hal.fused_transformer_mlp_bwd,
        (dy, x, norm_w, gate_w, up_w, down_w, eps),
        test_utils=["test_schema", "test_faketensor"],
    )


# =============================================================================
# 2. Autograd Tape Node Count: Bypassing Dispatch Overhead
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_autograd_node_count_reduction(device: str):
    """Verifies that unified block reduces autograd tape operation nodes from 6-8 down to strictly 1."""
    d = 64
    h = 128
    m = 8

    # Eager PyTorch reference block
    ref_block = EagerReferenceTransformerMLP(d, h).to(device)
    x_ref = torch.randn(m, d, device=device, requires_grad=True)
    y_ref = ref_block(x_ref)
    eager_node_count = _count_autograd_op_nodes(y_ref.grad_fn)

    # Eager PyTorch creates 6-14 operation nodes (MmBackward, MulBackward, SiluBackward, AddBackward, etc.)
    assert eager_node_count >= 6, f"Expected >= 6 eager autograd op nodes, got {eager_node_count}"

    # Unified block
    unified_block = HolonomicUnifiedTransformerMLP(
        dim=d,
        intermediate_dim=h,
        norm_weight=ref_block.norm_weight.data.clone(),
        gate_weight=ref_block.gate_proj.weight.data.clone(),
        up_weight=ref_block.up_proj.weight.data.clone(),
        down_weight=ref_block.down_proj.weight.data.clone(),
    ).to(device)

    x_uni = torch.randn(m, d, device=device, requires_grad=True)
    y_uni = unified_block(x_uni)

    # The backward tape should have strictly 1 single operation node
    unified_node_count = _count_autograd_op_nodes(y_uni.grad_fn)
    assert unified_node_count == 1, (
        f"Expected strictly 1 autograd op node for unified block, got {unified_node_count}"
    )


# =============================================================================
# 3. Numerical Output and Gradient Parity (Cosine Similarity >= 0.9999)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 64, 128), (2, 8, 32, 64)])
def test_numerical_and_gradient_parity(device: str, dtype: torch.dtype, shape: tuple):
    """Verifies numerical forward and gradient parity with cosine similarity >= 0.9999."""
    if device == "cpu" and dtype == torch.float16:
        # PyTorch CPU doesn't support fp16 matmul natively on older systems
        return

    dim = shape[-1]
    h_dim = 128
    eps = 1e-6

    # Inputs
    torch.manual_seed(42)
    x = torch.randn(*shape, device=device, dtype=dtype)
    norm_w = torch.randn(dim, device=device, dtype=dtype)
    gate_w = torch.randn(h_dim, dim, device=device, dtype=dtype)
    up_w = torch.randn(h_dim, dim, device=device, dtype=dtype)
    down_w = torch.randn(dim, h_dim, device=device, dtype=dtype)

    # Eager copy
    x_ref = x.clone().detach().requires_grad_(True)
    nw_ref = norm_w.clone().detach().requires_grad_(True)
    gw_ref = gate_w.clone().detach().requires_grad_(True)
    uw_ref = up_w.clone().detach().requires_grad_(True)
    dw_ref = down_w.clone().detach().requires_grad_(True)

    # Unified copy
    x_uni = x.clone().detach().requires_grad_(True)
    nw_uni = norm_w.clone().detach().requires_grad_(True)
    gw_uni = gate_w.clone().detach().requires_grad_(True)
    uw_uni = up_w.clone().detach().requires_grad_(True)
    dw_uni = down_w.clone().detach().requires_grad_(True)

    # Eager forward
    var = x_ref.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_norm = ((x_ref.float() * rstd) * nw_ref.float()).to(dtype)
    g = F.linear(x_norm, gw_ref)
    u = F.linear(x_norm, uw_ref)
    h = F.silu(g) * u
    y_ref = F.linear(h, dw_ref)

    # Unified forward
    y_uni = HolonomicUnifiedTransformerMLPFunction.apply(x_uni, nw_uni, gw_uni, uw_uni, dw_uni, eps)

    # Forward cosine similarity
    fwd_sim = _compute_cosine_similarity(y_ref, y_uni)
    assert fwd_sim >= 0.9999, f"Forward cosine similarity {fwd_sim:.6f} < 0.9999"

    # Backward pass
    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_uni.backward(grad_out)

    # Gradient cosine similarities
    dx_sim = _compute_cosine_similarity(x_ref.grad, x_uni.grad)
    dnw_sim = _compute_cosine_similarity(nw_ref.grad, nw_uni.grad)
    dgw_sim = _compute_cosine_similarity(gw_ref.grad, gw_uni.grad)
    duw_sim = _compute_cosine_similarity(uw_ref.grad, uw_uni.grad)
    ddw_sim = _compute_cosine_similarity(dw_ref.grad, dw_uni.grad)

    min_tol = 0.9995 if dtype == torch.float16 else 0.9999
    assert dx_sim >= min_tol, f"dx cosine similarity {dx_sim:.6f} < {min_tol}"
    assert dnw_sim >= min_tol, f"dnorm_w cosine similarity {dnw_sim:.6f} < {min_tol}"
    assert dgw_sim >= min_tol, f"dgate_w cosine similarity {dgw_sim:.6f} < {min_tol}"
    assert duw_sim >= min_tol, f"dup_w cosine similarity {duw_sim:.6f} < {min_tol}"
    assert ddw_sim >= min_tol, f"ddown_w cosine similarity {ddw_sim:.6f} < {min_tol}"


# =============================================================================
# 4. DRAM Memory Audit: Zero Intermediate Activation Tape Retention
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_dram_memory_audit_intermediate_activations(device: str):
    """Verifies that intermediate activations (X_norm, G, U, silu(G), H) are NEVER materialized in DRAM tape."""
    d = 128
    h = 256
    m = 16

    x = torch.randn(m, d, device=device, requires_grad=True)
    norm_w = torch.randn(d, device=device, requires_grad=True)
    gate_w = torch.randn(h, d, device=device, requires_grad=True)
    up_w = torch.randn(h, d, device=device, requires_grad=True)
    down_w = torch.randn(d, h, device=device, requires_grad=True)

    saved_tensors_captured = []

    def pack_hook(tensor):
        saved_tensors_captured.append(tensor)
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = HolonomicUnifiedTransformerMLPFunction.apply(x, norm_w, gate_w, up_w, down_w, 1e-6)

    # In eager PyTorch, G, U, silu(G), H all have shape [m, h] = [16, 256]
    # Check that NO saved tensor has shape [m, h]
    for saved in saved_tensors_captured:
        assert saved.shape != (m, h), (
            f"VIOLATION: Intermediate activation tensor of shape {saved.shape} was materialized on autograd tape!"
        )

    # The saved tensors must strictly be the boundary inputs and weights: x, norm_w, gate_w, up_w, down_w
    assert len(saved_tensors_captured) <= 5, (
        f"Expected at most 5 saved boundary tensors, got {len(saved_tensors_captured)}"
    )


# =============================================================================
# 5. SmartWorkloadRouter: Full Transformer Model End-to-End Optimization
# =============================================================================

class MockLlamaAttention(nn.Module):
    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(x)


class MockLlamaMLP(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gate_proj = nn.Linear(d, h)
        self.up_proj = nn.Linear(d, h)
        self.down_proj = nn.Linear(h, d)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockFullTransformerLLM(nn.Module):
    def __init__(self, d: int = 64, h: int = 128, vocab_size: int = 10000):
        super().__init__()
        self.attn = MockLlamaAttention(d)
        self.mlp = MockLlamaMLP(d, h)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return self.lm_head(x)


@pytest.mark.parametrize("device", DEVICES)
def test_smart_router_transformer_integration(device: str):
    """Verifies that SmartWorkloadRouter automatically rewrites Attention, MLP, and LM head."""
    model = MockFullTransformerLLM().to(device)
    opt_model, report = optimize_model(model, prefer_unified_mlp=True)

    actions = report.action_counts
    assert "HOLONOMIC_DAGGER_ATTENTION" in actions, "Router failed to detect & rewrite attention!"
    assert "OJASX_UNIFIED_TRANSFORMER_MLP" in actions, "Router failed to detect & rewrite unified MLP!"
    assert "FUSED_LINEAR_CROSS_ENTROPY" in actions, "Router failed to detect & rewrite LM head!"

    # Test forward pass with training loss
    x = torch.randn(2, 16, 64, device=device, requires_grad=True)
    targets = torch.randint(0, 10000, (2, 16), device=device)

    # 1. Unsupervised / logits forward
    logits = opt_model(x)
    assert logits.shape == (2, 16, 10000)

    # 2. Supervised fused cross-entropy loss forward
    h = opt_model.attn(x)
    loss = opt_model.lm_head(h, targets)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    loss.backward()
    assert x.grad is not None


# =============================================================================
# 6. FX Subgraph Compiler: Attention and Loss Rewrites
# =============================================================================

class FXSDPAModel(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class FXLinearCEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 1000)

    def forward(self, h, targets):
        logits = self.head(h)
        return F.cross_entropy(logits, targets)


def test_fx_subgraph_compiler_attention_and_loss_passes():
    """Verifies that HolonomicSubgraphTransformer rewrites attention and cross-entropy subgraphs."""
    # 1. Attention pass
    gm_attn = torch.fx.symbolic_trace(FXSDPAModel())
    opt_gm_attn, stats_attn = transform_subgraph_module(gm_attn)
    assert stats_attn.attention_count == 1, (
        f"Expected 1 rewritten attention subgraph, got {stats_attn.attention_count}"
    )

    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out_orig = FXSDPAModel()(q, k, v)
    out_opt = opt_gm_attn(q, k, v)
    assert torch.allclose(out_orig, out_opt, atol=1e-4, rtol=1e-4)

    # 2. Linear + Cross Entropy pass
    model_loss = FXLinearCEModel()
    gm_loss = torch.fx.symbolic_trace(model_loss)
    opt_gm_loss, stats_loss = transform_subgraph_module(gm_loss)
    assert stats_loss.linear_cross_entropy_count == 1, (
        f"Expected 1 rewritten linear cross entropy subgraph, got {stats_loss.linear_cross_entropy_count}"
    )

    h = torch.randn(4, 32)
    targets = torch.randint(0, 1000, (4,))
    loss_orig = model_loss(h, targets)
    loss_opt = opt_gm_loss(h, targets)
    assert torch.allclose(loss_orig, loss_opt, atol=1e-3, rtol=1e-3)
