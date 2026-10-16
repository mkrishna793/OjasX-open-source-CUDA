"""
Audit Fixes Workstream C: Subgraph Compiler, Smart Router & MLIR Dialects.
Validates:
1. GQA/MQA Support in HolonomicSymplecticSubgraphFunction & HolonomicResidualSubgraphFunction.
2. Zero-allocation gated MLP backward (elimination of w_comb = torch.cat([w_gate, w_up])).
3. DaggerAttention non-double-wrapping in smart router & zero-alloc LM head weight reuse.
4. MLIR HardwareCodeEmitter '?' dynamic symbolic dimension handling & FX shape/view SSA propagation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from ojasx2.fx_subgraph_compiler import (
    holonomic_gated_mlp,
    holonomic_glu,
    holonomic_symplectic_subgraph,
    holonomic_residual_subgraph,
    HolonomicGatedMLPFunction,
    HolonomicUniversalGLUFunction,
    HolonomicFusedGLUMLPFunction,
    HolonomicSymplecticSubgraphFunction,
    HolonomicResidualSubgraphFunction,
)
from ojasx2.smart_router import (
    SmartWorkloadRouter,
    HolonomicFusedLinearCrossEntropyHead,
)
from ojasx2.fused_cross_entropy import FusedLinearCrossEntropy
from ojasx2.dagger_attention import DaggerAttention
from ojasx2.mlir_dialects import (
    MLIRModule,
    MLIRFunction,
    MLIRBlock,
    MLIRValue,
    TensorType,
    FloatType,
    HolonomicDialect,
    HardwareCodeEmitter,
    fx_to_holonomic_mlir,
)


# =============================================================================
# 1. Zero-Allocation Gated MLP Backward Tests
# =============================================================================

def test_zero_allocation_gated_mlp_backward():
    """Verifies HolonomicGatedMLPFunction backward does not allocate w_comb via torch.cat."""
    b_sz, s_len, dim, hidden = 2, 4, 16, 32
    x = torch.randn(b_sz, s_len, dim, requires_grad=True)
    w_gate = torch.randn(hidden, dim, requires_grad=True)
    b_gate = torch.randn(hidden, requires_grad=True)
    w_up = torch.randn(hidden, dim, requires_grad=True)
    b_up = torch.randn(hidden, requires_grad=True)
    w_down = torch.randn(dim, hidden, requires_grad=True)
    b_down = torch.randn(dim, requires_grad=True)

    # Intercept torch.cat during backward to confirm w_gate and w_up are never concatenated
    orig_cat = torch.cat
    cat_w_comb_calls = []

    def tracking_cat(tensors, *args, **kwargs):
        for t in tensors:
            if (t is w_gate) or (t is w_up):
                cat_w_comb_calls.append(True)
        return orig_cat(tensors, *args, **kwargs)

    out = holonomic_gated_mlp(x, w_gate, b_gate, w_up, b_up, w_down, b_down, act="silu")
    loss = out.sum()

    torch.cat = tracking_cat
    try:
        loss.backward()
    finally:
        torch.cat = orig_cat

    assert len(cat_w_comb_calls) == 0, "w_gate and w_up were concatenated during backward!"
    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert w_gate.grad is not None and not torch.isnan(w_gate.grad).any()
    assert w_up.grad is not None and not torch.isnan(w_up.grad).any()
    assert w_down.grad is not None and not torch.isnan(w_down.grad).any()


def test_zero_allocation_universal_glu_backward():
    """Verifies HolonomicUniversalGLUFunction (and HolonomicFusedGLUMLPFunction) backward does not concatenate w_comb."""
    b_sz, s_len, dim, hidden = 2, 4, 16, 32
    x = torch.randn(b_sz, s_len, dim, requires_grad=True)
    w_gate = torch.randn(hidden, dim, requires_grad=True)
    w_up = torch.randn(hidden, dim, requires_grad=True)

    orig_cat = torch.cat
    cat_w_comb_calls = []

    def tracking_cat(tensors, *args, **kwargs):
        for t in tensors:
            if (t is w_gate) or (t is w_up):
                cat_w_comb_calls.append(True)
        return orig_cat(tensors, *args, **kwargs)

    out = holonomic_glu(x, w_gate, None, w_up, None, act="silu")
    loss = out.sum()

    torch.cat = tracking_cat
    try:
        loss.backward()
    finally:
        torch.cat = orig_cat

    assert len(cat_w_comb_calls) == 0, "w_gate and w_up were concatenated in GLU backward!"
    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert w_gate.grad is not None and not torch.isnan(w_gate.grad).any()
    assert w_up.grad is not None and not torch.isnan(w_up.grad).any()
    assert HolonomicFusedGLUMLPFunction is HolonomicUniversalGLUFunction


def test_gated_mlp_gradient_numerical_accuracy():
    """Verifies mathematical equivalence of dz_gate @ w_gate + dup @ w_up against autograd."""
    dim, hidden = 8, 16
    x = torch.randn(2, 3, dim, dtype=torch.float64, requires_grad=True)
    w_gate = torch.randn(hidden, dim, dtype=torch.float64, requires_grad=True)
    w_up = torch.randn(hidden, dim, dtype=torch.float64, requires_grad=True)
    w_down = torch.randn(dim, hidden, dtype=torch.float64, requires_grad=True)

    # Reference computation using native autograd
    x_ref = x.clone().detach().requires_grad_(True)
    w_gate_ref = w_gate.clone().detach().requires_grad_(True)
    w_up_ref = w_up.clone().detach().requires_grad_(True)
    w_down_ref = w_down.clone().detach().requires_grad_(True)

    h_gate = F.linear(x_ref, w_gate_ref)
    h_up = F.linear(x_ref, w_up_ref)
    ref_out = F.linear(F.silu(h_gate) * h_up, w_down_ref)
    ref_loss = (ref_out ** 2).sum()
    ref_loss.backward()

    # Holonomic gated MLP
    holo_out = holonomic_gated_mlp(x, w_gate, None, w_up, None, w_down, None, act="silu")
    holo_loss = (holo_out ** 2).sum()
    holo_loss.backward()

    torch.testing.assert_close(holo_out, ref_out)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(w_gate.grad, w_gate_ref.grad)
    torch.testing.assert_close(w_up.grad, w_up_ref.grad)
    torch.testing.assert_close(w_down.grad, w_down_ref.grad)


# =============================================================================
# 2. Grouped-Query Attention (GQA/MQA) in Symplectic & Residual Subgraphs
# =============================================================================

def test_symplectic_subgraph_gqa_forward_and_backward():
    """Verifies HolonomicSymplecticSubgraphFunction correctly handles GQA (num_heads=4, num_kv_heads=2)."""
    b_sz, s_len, dim = 2, 4, 32
    num_heads = 4
    num_kv_heads = 2
    hd = dim // num_heads  # 8
    kv_dim = num_kv_heads * hd  # 16
    w_qkv_len = dim + 2 * kv_dim  # 32 + 32 = 64

    q1 = torch.randn(b_sz, s_len, dim, requires_grad=True)
    q2 = torch.randn(b_sz, s_len, dim, requires_grad=True)
    w_attn_qkv = torch.randn(w_qkv_len, dim, requires_grad=True)
    w_attn_out = torch.randn(dim, dim, requires_grad=True)
    w_mlp_gate = torch.randn(64, dim, requires_grad=True)
    w_mlp_up = torch.randn(64, dim, requires_grad=True)
    w_mlp_down = torch.randn(dim, 64, requires_grad=True)

    q1_next, q2_next = holonomic_symplectic_subgraph(
        q1, q2,
        w_attn_qkv, None,
        w_attn_out, None,
        w_mlp_gate, None,
        w_mlp_up, None,
        w_mlp_down, None,
        act="silu",
        num_heads=num_heads,
    )

    assert q1_next.shape == (b_sz, s_len, dim)
    assert q2_next.shape == (b_sz, s_len, dim)

    loss = q1_next.sum() + q2_next.sum()
    loss.backward()

    assert q1.grad is not None and not torch.isnan(q1.grad).any()
    assert q2.grad is not None and not torch.isnan(q2.grad).any()
    assert w_attn_qkv.grad is not None and not torch.isnan(w_attn_qkv.grad).any()
    assert w_attn_out.grad is not None and not torch.isnan(w_attn_out.grad).any()


def test_symplectic_subgraph_mqa_forward_and_backward():
    """Verifies HolonomicSymplecticSubgraphFunction correctly handles MQA (num_heads=4, num_kv_heads=1)."""
    b_sz, s_len, dim = 2, 4, 32
    num_heads = 4
    num_kv_heads = 1
    hd = dim // num_heads  # 8
    kv_dim = num_kv_heads * hd  # 8
    w_qkv_len = dim + 2 * kv_dim  # 32 + 16 = 48

    q1 = torch.randn(b_sz, s_len, dim, requires_grad=True)
    q2 = torch.randn(b_sz, s_len, dim, requires_grad=True)
    w_attn_qkv = torch.randn(w_qkv_len, dim, requires_grad=True)
    w_attn_out = torch.randn(dim, dim, requires_grad=True)
    w_mlp_gate = torch.randn(64, dim, requires_grad=True)
    w_mlp_up = torch.randn(64, dim, requires_grad=True)
    w_mlp_down = torch.randn(dim, 64, requires_grad=True)

    q1_next, q2_next = holonomic_symplectic_subgraph(
        q1, q2,
        w_attn_qkv, None,
        w_attn_out, None,
        w_mlp_gate, None,
        w_mlp_up, None,
        w_mlp_down, None,
        act="silu",
        num_heads=num_heads,
    )

    assert q1_next.shape == (b_sz, s_len, dim)
    assert q2_next.shape == (b_sz, s_len, dim)

    loss = q1_next.sum() + q2_next.sum()
    loss.backward()

    assert q1.grad is not None and not torch.isnan(q1.grad).any()
    assert q2.grad is not None and not torch.isnan(q2.grad).any()
    assert w_attn_qkv.grad is not None and not torch.isnan(w_attn_qkv.grad).any()


def test_residual_subgraph_gqa_forward_and_backward():
    """Verifies HolonomicResidualSubgraphFunction correctly handles GQA with RMSNorm and LayerNorm."""
    b_sz, s_len, dim = 2, 4, 32
    num_heads = 4
    num_kv_heads = 2
    hd = dim // num_heads  # 8
    kv_dim = num_kv_heads * hd  # 16
    w_qkv_len = dim + 2 * kv_dim  # 64

    for norm_type in ("rms", "layer"):
        x = torch.randn(b_sz, s_len, dim, requires_grad=True)
        norm1_w = torch.ones(dim, requires_grad=True)
        norm1_b = torch.zeros(dim, requires_grad=True) if norm_type == "layer" else None
        w_attn_qkv = torch.randn(w_qkv_len, dim, requires_grad=True)
        w_attn_out = torch.randn(dim, dim, requires_grad=True)
        norm2_w = torch.ones(dim, requires_grad=True)
        norm2_b = torch.zeros(dim, requires_grad=True) if norm_type == "layer" else None
        w_mlp_gate = torch.randn(64, dim, requires_grad=True)
        w_mlp_up = torch.randn(64, dim, requires_grad=True)
        w_mlp_down = torch.randn(dim, 64, requires_grad=True)

        out = holonomic_residual_subgraph(
            x,
            norm1_w, norm1_b, 1e-5,
            w_attn_qkv, None,
            w_attn_out, None,
            num_heads,
            norm2_w, norm2_b, 1e-5,
            w_mlp_gate, None,
            w_mlp_up, None,
            w_mlp_down, None,
            act="silu",
            norm_type=norm_type,
        )

        assert out.shape == (b_sz, s_len, dim)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and not torch.isnan(x.grad).any()
        assert w_attn_qkv.grad is not None and not torch.isnan(w_attn_qkv.grad).any()
        assert w_attn_out.grad is not None and not torch.isnan(w_attn_out.grad).any()
        assert w_mlp_gate.grad is not None and not torch.isnan(w_mlp_gate.grad).any()


# =============================================================================
# 3. Smart Router: DaggerAttention Non-Double-Wrapping & Zero-Alloc LM Head
# =============================================================================

def test_dagger_attention_not_double_wrapped():
    """Verifies DaggerAttention in a model is never double-wrapped during router optimization."""
    class TransformerBlockWithDagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DaggerAttention(embed_dim=128, num_heads=4)
            self.lm_head = nn.Linear(128, 32000)

        def forward(self, x):
            return self.lm_head(self.attn(x))

    model = TransformerBlockWithDagger()
    orig_attn = model.attn
    orig_lm_weight = model.lm_head.weight

    router = SmartWorkloadRouter()
    optimized_model, report = router.optimize_model(model, batch_tokens=1024)

    # 1. DaggerAttention is unchanged and not wrapped into another attention class
    assert optimized_model.attn is orig_attn
    assert isinstance(optimized_model.attn, DaggerAttention)
    assert not hasattr(optimized_model.attn, "attn")

    # 2. LM Head is rewritten to HolonomicFusedLinearCrossEntropyHead reusing original weight tensor
    assert isinstance(optimized_model.lm_head, HolonomicFusedLinearCrossEntropyHead)
    assert optimized_model.lm_head.weight is orig_lm_weight


def test_fused_linear_cross_entropy_head_init_empty():
    """Verifies HolonomicFusedLinearCrossEntropyHead(_init_empty=True) skips weight allocation and sets out_features."""
    head = HolonomicFusedLinearCrossEntropyHead(in_features=128, vocab_size=32000, _init_empty=True)
    assert head.weight is None
    assert head.in_features == 128
    assert head.vocab_size == 32000
    assert head.out_features == 32000


def test_fused_linear_cross_entropy_out_features_property():
    """Verifies FusedLinearCrossEntropy exposes out_features matching vocab_size."""
    fce = FusedLinearCrossEntropy(in_features=64, vocab_size=5000)
    assert fce.out_features == 5000
    assert fce.vocab_size == 5000

    fce.out_features = 6000
    assert fce.vocab_size == 6000
    assert fce.out_features == 6000


# =============================================================================
# 4. MLIR '?' Symbolic Dimension & FX SSA Shape Propagation
# =============================================================================

def test_mlir_hardware_emitter_symbolic_dimension():
    """Verifies HardwareCodeEmitter._extract_attention_dims handles '?' in s_val without crashing on int('?')."""
    class ShapeHolder:
        def __init__(self, value):
            self.value = value

    mod = MLIRModule(name="sym_test")
    fn = MLIRFunction(name="forward", args=[], return_types=[], body=MLIRBlock())
    mod.functions.append(fn)

    q = MLIRValue("%q", TensorType(("?", 8, 128, 64), FloatType(32)))
    k = MLIRValue("%k", TensorType(("?", 8, 128, 64), FloatType(32)))
    v = MLIRValue("%v", TensorType(("?", 8, 128, 64), FloatType(32)))
    op = HolonomicDialect.dagger_attention(q, k, v, TensorType(("?", 8, 128, 64), FloatType(32)))
    op.attributes["shape"] = ShapeHolder(["?", 8, 128, 64])
    fn.body.operations.append(op)

    # Should not raise ValueError: invalid literal for int() with base 10: '?'
    dims = HardwareCodeEmitter._extract_attention_dims(mod)
    assert dims == (1, 8, 128, 64)

    # Test with multiple symbolic dimensions in s_val
    op.attributes["shape"] = ShapeHolder(["?", "?", "?", 64])
    dims_multi_sym = HardwareCodeEmitter._extract_attention_dims(mod)
    assert dims_multi_sym == (1, 1, 1, 64)


def test_fx_to_holonomic_mlir_shape_view_ssa_propagation():
    """Verifies fx_to_holonomic_mlir propagates SSA values through view/reshape/contiguous nodes."""
    class MLPWithViews(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            h = self.fc1(x)
            h_v = h.view(-1, 256)
            h_c = h_v.contiguous()
            return self.fc2(h_c)

    model = MLPWithViews()
    gm = fx.symbolic_trace(model)
    mlir_mod = fx_to_holonomic_mlir(gm)

    fwd = mlir_mod.functions[0]
    ops = fwd.body.operations
    assert len(ops) == 2, f"Expected 2 morphism ops, got {len(ops)}"

    op_fc1, op_fc2 = ops[0], ops[1]
    # Crucial check: op_fc2 operand 0 must consume op_fc1's result, NOT the fallback %0 / input
    assert op_fc2.operands[0] == op_fc1.results[0], (
        f"SSA broken: op_fc2 takes {op_fc2.operands[0].name} instead of op_fc1 result {op_fc1.results[0].name}"
    )

    # Verify SSA invariants
    assert mlir_mod.verify(), "MLIR module failed SSA verification!"
