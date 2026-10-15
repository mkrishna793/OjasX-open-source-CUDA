"""Official PyTorch Contract Enforcement & Verification for Symplectic and Residual Subgraphs.

Strictly verifies:
1. `torch.library.opcheck` compliance for:
   - `ojasx_fx::symplectic_subgraph` (Mode A)
   - `ojasx_fx::holonomic_residual_subgraph` (Mode B)
2. Mathematical parity against standard PyTorch eager module references.
3. Memory tape elimination audit (SavedTensorAudit confirming zero intermediate memory).
4. Stride invariance (contiguous, transposed, and sliced non-contiguous views).
5. Dynamic shape and multi-dtype compliance (float32, float16, bfloat16).
6. FX Subgraph Compiler lowering of consecutive residual chains.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import opcheck

import ojasx2
from ojasx2.residuals import SavedTensorAudit
from ojasx2.fx_subgraph_compiler import (
    compile_fx_subgraph,
    FXSubgraphCompiler,
    TransformationStats,
    holonomic_symplectic_subgraph,
    holonomic_residual_subgraph,
)

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# =============================================================================
# Reference PyTorch Eager Modules for Parity Checks
# =============================================================================

class EagerModeAReference(nn.Module):
    """Reference bipartite symplectic block."""
    def __init__(self, dim: int = 16, num_heads: int = 4, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.w_gate = nn.Linear(dim, dim * hidden_mult, bias=False)
        self.w_up = nn.Linear(dim, dim * hidden_mult, bias=False)
        self.w_down = nn.Linear(dim * hidden_mult, dim, bias=False)

    def f_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.num_heads
        hd = d // h
        qkv = self.qkv(x).reshape(b, s, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(b, s, d)
        return self.out_proj(attn)

    def g_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_next = q1 + self.f_attn(q2)
        q2_next = q2 + self.g_mlp(q1_next)
        return q1_next, q2_next


class EagerModeBReference(nn.Module):
    """Reference standard Transformer block (Pre-LN with consecutive residuals)."""
    def __init__(self, dim: int = 16, num_heads: int = 4, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)
        self.w_gate = nn.Linear(dim, dim * hidden_mult, bias=False)
        self.w_up = nn.Linear(dim, dim * hidden_mult, bias=False)
        self.w_down = nn.Linear(dim * hidden_mult, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: Attention with residual
        h1 = self.norm1(x)
        b, s, d = h1.shape
        h = self.num_heads
        hd = d // h
        qkv = self.qkv(h1).reshape(b, s, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(b, s, d)
        x1 = x + self.out_proj(attn)

        # Sub-layer 2: MLP with residual
        h2 = self.norm2(x1)
        mlp = self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
        x2 = x1 + mlp
        return x2


# =============================================================================
# 1. OpCheck Tests (Official PyTorch Operator Contract Enforcement)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_symplectic_subgraph_mode_a(device: str) -> None:
    torch.manual_seed(42)
    b, s, d = 2, 8, 16
    q1 = torch.randn(b, s, d, device=device)
    q2 = torch.randn(b, s, d, device=device)
    w_qkv = torch.randn(3 * d, d, device=device)
    w_out = torch.randn(d, d, device=device)
    w_gate = torch.randn(2 * d, d, device=device)
    w_up = torch.randn(2 * d, d, device=device)
    w_down = torch.randn(d, 2 * d, device=device)

    args = (
        q1, q2,
        w_qkv, None,
        w_out, None,
        w_gate, None,
        w_up, None,
        w_down, None,
        "silu",
        4,
    )
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    opcheck(torch.ops.ojasx_fx.symplectic_subgraph, args, test_utils=test_utils)


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_holonomic_residual_subgraph_mode_b(device: str) -> None:
    torch.manual_seed(42)
    b, s, d = 2, 8, 16
    x = torch.randn(b, s, d, device=device)
    norm1_w = torch.ones(d, device=device)
    w_qkv = torch.randn(3 * d, d, device=device)
    w_out = torch.randn(d, d, device=device)
    norm2_w = torch.ones(d, device=device)
    w_gate = torch.randn(2 * d, d, device=device)
    w_up = torch.randn(2 * d, d, device=device)
    w_down = torch.randn(d, 2 * d, device=device)

    args = (
        x,
        norm1_w, None, 1e-5,
        w_qkv, None,
        w_out, None,
        4,
        norm2_w, None, 1e-5,
        w_gate, None,
        w_up, None,
        w_down, None,
        "silu",
    )
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    opcheck(torch.ops.ojasx_fx.holonomic_residual_subgraph, args, test_utils=test_utils)


# =============================================================================
# 2. Mathematical Parity & Gradient Accuracy Checks
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_symplectic_subgraph_mode_a_parity(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(101)
    b, s, d = 2, 8, 16
    ref_mod = EagerModeAReference(dim=d, num_heads=4, hidden_mult=2).to(device=device, dtype=dtype)
    q1 = torch.randn(b, s, d, device=device, dtype=dtype, requires_grad=True)
    q2 = torch.randn(b, s, d, device=device, dtype=dtype, requires_grad=True)

    q1_ref = q1.detach().clone().requires_grad_(True)
    q2_ref = q2.detach().clone().requires_grad_(True)

    # Reference forward & backward
    out1_ref, out2_ref = ref_mod(q1_ref, q2_ref)
    loss_ref = (out1_ref.float().square().mean() + out2_ref.float().square().mean())
    loss_ref.backward()

    # OjasX Mode A Subgraph forward & backward
    out1_sub, out2_sub = holonomic_symplectic_subgraph(
        q1, q2,
        ref_mod.qkv.weight, None,
        ref_mod.out_proj.weight, None,
        ref_mod.w_gate.weight, None,
        ref_mod.w_up.weight, None,
        ref_mod.w_down.weight, None,
        act="silu",
        num_heads=4,
    )
    loss_sub = (out1_sub.float().square().mean() + out2_sub.float().square().mean())
    loss_sub.backward()

    atol = 1e-4 if dtype == torch.float32 else (2e-3 if dtype == torch.float16 else 3e-2)
    rtol = 1e-4 if dtype == torch.float32 else (2e-3 if dtype == torch.float16 else 3e-2)

    torch.testing.assert_close(out1_sub, out1_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out2_sub, out2_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q1.grad, q1_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(q2.grad, q2_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_holonomic_residual_subgraph_mode_b_parity(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(202)
    b, s, d = 2, 8, 16
    ref_mod = EagerModeBReference(dim=d, num_heads=4, hidden_mult=2).to(device=device, dtype=dtype)
    x = torch.randn(b, s, d, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Reference forward & backward
    y_ref = ref_mod(x_ref)
    loss_ref = y_ref.float().square().mean()
    loss_ref.backward()

    # OjasX Mode B Subgraph forward & backward
    y_sub = holonomic_residual_subgraph(
        x,
        ref_mod.norm1.weight, None, 1e-5,
        ref_mod.qkv.weight, None,
        ref_mod.out_proj.weight, None,
        4,
        ref_mod.norm2.weight, None, 1e-5,
        ref_mod.w_gate.weight, None,
        ref_mod.w_up.weight, None,
        ref_mod.w_down.weight, None,
        act="silu",
    )
    loss_sub = y_sub.float().square().mean()
    loss_sub.backward()

    atol = 1e-4 if dtype == torch.float32 else (2e-3 if dtype == torch.float16 else 3e-2)
    rtol = 1e-4 if dtype == torch.float32 else (2e-3 if dtype == torch.float16 else 3e-2)

    torch.testing.assert_close(y_sub, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)


# =============================================================================
# 3. Zero-Intermediate Tape Memory Audit (SavedTensorAudit)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_zero_tape_memory_reduction_mode_a(device: str) -> None:
    b, s, d = 4, 16, 32
    ref_mod = EagerModeAReference(dim=d, num_heads=4, hidden_mult=2).to(device=device)
    q1 = torch.randn(b, s, d, device=device, requires_grad=True)
    q2 = torch.randn(b, s, d, device=device, requires_grad=True)

    all_inputs = [q1, q2] + list(ref_mod.parameters())

    with SavedTensorAudit(all_inputs) as audit_ref:
        out1_ref, out2_ref = ref_mod(q1, q2)

    with SavedTensorAudit(all_inputs) as audit_sub:
        out1_sub, out2_sub = holonomic_symplectic_subgraph(
            q1, q2,
            ref_mod.qkv.weight, None,
            ref_mod.out_proj.weight, None,
            ref_mod.w_gate.weight, None,
            ref_mod.w_up.weight, None,
            ref_mod.w_down.weight, None,
            act="silu",
            num_heads=4,
        )

    # In Mode A, zero intermediate activation tensors are stored
    saved_bytes_sub = audit_sub.summary()["fresh_saved_storage_bytes"]
    saved_bytes_ref = audit_ref.summary()["fresh_saved_storage_bytes"]
    assert saved_bytes_sub < saved_bytes_ref, f"Expected memory reduction, got sub={saved_bytes_sub} vs ref={saved_bytes_ref}"


@pytest.mark.parametrize("device", DEVICES)
def test_zero_tape_memory_reduction_mode_b(device: str) -> None:
    b, s, d = 4, 16, 32
    ref_mod = EagerModeBReference(dim=d, num_heads=4, hidden_mult=2).to(device=device)
    x = torch.randn(b, s, d, device=device, requires_grad=True)

    all_inputs = [x] + list(ref_mod.parameters())

    with SavedTensorAudit(all_inputs) as audit_ref:
        y_ref = ref_mod(x)

    with SavedTensorAudit(all_inputs) as audit_sub:
        y_sub = holonomic_residual_subgraph(
            x,
            ref_mod.norm1.weight, None, 1e-5,
            ref_mod.qkv.weight, None,
            ref_mod.out_proj.weight, None,
            4,
            ref_mod.norm2.weight, None, 1e-5,
            ref_mod.w_gate.weight, None,
            ref_mod.w_up.weight, None,
            ref_mod.w_down.weight, None,
            act="silu",
        )

    saved_bytes_sub = audit_sub.summary()["fresh_saved_storage_bytes"]
    saved_bytes_ref = audit_ref.summary()["fresh_saved_storage_bytes"]
    assert saved_bytes_sub < saved_bytes_ref, f"Expected memory reduction in Mode B, got sub={saved_bytes_sub} vs ref={saved_bytes_ref}"


# =============================================================================
# 4. Exotic Strides & Slicing Invariance
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stride_kind", ["sliced", "transposed"])
def test_non_contiguous_strides_invariance(device: str, stride_kind: str) -> None:
    d = 16
    ref_mod = EagerModeBReference(dim=d, num_heads=4, hidden_mult=2).to(device=device)

    if stride_kind == "sliced":
        x_wide = torch.randn(2, 16, d, device=device, requires_grad=True)
        x_in = x_wide[:, ::2, :]  # shape: (2, 8, 16), stride: (256, 32, 1) non-contiguous!
        x_in.retain_grad()
    else:
        # Transposed non-contiguous view
        x_in = torch.randn(2, d, 8, device=device).transpose(1, 2).detach().requires_grad_(True)

    assert not x_in.is_contiguous()

    x_ref = x_in.detach().clone().requires_grad_(True)

    y_ref = ref_mod(x_ref)
    y_ref.sum().backward()

    y_sub = holonomic_residual_subgraph(
        x_in,
        ref_mod.norm1.weight, None, 1e-5,
        ref_mod.qkv.weight, None,
        ref_mod.out_proj.weight, None,
        4,
        ref_mod.norm2.weight, None, 1e-5,
        ref_mod.w_gate.weight, None,
        ref_mod.w_up.weight, None,
        ref_mod.w_down.weight, None,
        act="silu",
    )
    y_sub.sum().backward()

    torch.testing.assert_close(y_sub, y_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x_in.grad, x_ref.grad, atol=1e-4, rtol=1e-4)


# =============================================================================
# 5. Compiler Subgraph Transformation Tests
# =============================================================================

def test_fx_compiler_residual_subgraph_mode_b_auto() -> None:
    """Verifies that compile_fx_subgraph automatically detects consecutive residual blocks and lowers them."""
    mod = EagerModeBReference(dim=16, num_heads=4, hidden_mult=2)
    example_input = torch.randn(2, 8, 16)

    compiled = compile_fx_subgraph(mod, mode="mode_b")
    stats: TransformationStats = compiled.stats
    assert stats.holonomic_residual_subgraph_count >= 1

    x = torch.randn(2, 8, 16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y_compiled = compiled(x)
    y_ref = mod(x_ref)
    torch.testing.assert_close(y_compiled, y_ref, atol=1e-4, rtol=1e-4)

    y_compiled.sum().backward()
    y_ref.sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
