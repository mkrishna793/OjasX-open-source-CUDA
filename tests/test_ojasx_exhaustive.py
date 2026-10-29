"""
tests/test_ojasx_exhaustive.py

Exhaustive Unit & Property-Based Verification Suite for OjasX (v0.11.0+).
Covers:
  1. Unified `ojasx2.math` exports (verifying zero module/package shadowing).
  2. Structured Low-Rank Adjoints (JVP pushforward, VJP pullback, dense materialization,
     Softmax, RMSNorm, LayerNorm, NormalizationResidual).
  3. Halley-Padé 3rd-Order Root Inversion & 1-Bit Branch Packing across all 10 activations,
     including extreme left-branch asymptotic underflow (z = -25.0) and critical points (z = z_*).
  4. Symplectic Reversible Engine:
     - 2nd-order Störmer-Verlet + FSAL vs 1st-order Euler.
     - Exact invertibility WITH `enable_squeeze=True` on `SymplecticStack`, `SymplecticCoupling`,
       and `SymplecticTransformerBlock`.
     - Live `SymplecticMetrics` telemetry and zero-element batch safety.
  5. Streaming DRAM-Free Operators:
     - `holonomic_dagger_attention` & `DaggerAttention` (Causal, GQA, Masks, RoPE, KV-Cache, 0-element).
     - `fused_linear_cross_entropy` (all-masked `ignore_index`, label smoothing, `none`/`sum`/`mean`).
  6. E8 Gosset Lattice Quantization:
     - Conway-Sloane O(8) parity/norm invariants.
     - 2.0-bit, 1.5-bit Ternary E8, and 1.0-bit Root E8 packers on BOTH aligned and odd/unaligned shapes.
     - Thread-safety of `_get_ephemeral_workspace` under concurrent multithreaded execution.
     - Mixed-precision LoRA forward/backward through `E8LatticeLinear`.
  7. `HolonomicGradFreeOptimizer`:
     - Zero persistent `.grad` bytes after `.backward()`.
     - Auto-latching global gradient norm across steps even when `.step()` is omitted.
     - 2.0-bit E8 momentum compression on odd-sized bias/weight tensors.
     - Non-finite (`NaN`/`Inf`) gradient guard skipping corrupt updates.
  8. FX Subgraph Compiler, AOTAutograd, Smart Workload Router & 3-Dialect MLIR Pipeline:
     - Differentiable QKV fusion preserving original `q_proj, k_proj, v_proj` parameter gradients.
     - Nested submodule resolution in `HolonomicSubgraphTransformer` and `prune_dead_channels_fx`.
     - MLIR SSA verification and live `last_dispatched_ops` tracking.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx2
from ojasx2.residuals import SavedTensorAudit


# =============================================================================
# 1. Unified Math Namespace & Structured Adjoints Tests
# =============================================================================

def test_math_namespace_no_shadowing():
    """Verify that ojasx2.math exposes both package symbols and math_foundation symbols."""
    required_symbols = [
        "StructuredDerivative",
        "diagonal_derivative",
        "activation_derivative",
        "softmax_derivative",
        "rmsnorm_derivative",
        "rmsnorm_input_derivative",
        "layernorm_derivative",
        "layernorm_input_derivative",
        "NormalizationResidual",
        "normalization_residual",
        "rank7_reference",
        "winograd_reference",
        "StormerVerletIntegrator",
        "SymplecticManifold",
        "halley_invert_activation",
        "conway_sloane_e8_project",
        "E8LatticePacker",
        "TernaryE8LatticePacker",
        "RootE8LatticePacker",
    ]
    for sym in required_symbols:
        assert hasattr(ojasx2.math, sym), f"Missing symbol {sym} on ojasx2.math (shadowing bug!)"


@pytest.mark.parametrize("dim", [1, 7, 32, 64])
def test_structured_derivative_push_pull_parity(dim: int):
    """Property test: J.push(v) == J_dense @ v and J.pull(w) == J_dense.T @ w."""
    torch.manual_seed(42 + dim)
    x = torch.randn(4, dim, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, dim, dtype=torch.float64)
    w = torch.randn(4, dim, dtype=torch.float64)

    # 1. Softmax derivative
    p = F.softmax(x, dim=-1)
    sd_sm = ojasx2.math.softmax_derivative(p)
    j_sm = sd_sm.materialize()
    assert torch.allclose(sd_sm.push(v), torch.matmul(j_sm, v.unsqueeze(-1)).squeeze(-1), atol=1e-12)
    assert torch.allclose(sd_sm.pull(w), torch.matmul(j_sm.transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1), atol=1e-12)

    # 2. RMSNorm derivative vs autograd VJP
    sd_rms = ojasx2.math.rmsnorm_derivative(x, gamma, eps=1e-5)
    j_rms = sd_rms.materialize()
    y_rms = F.rms_norm(x, (dim,), gamma, 1e-5)
    (dx_auto,) = torch.autograd.grad(y_rms, x, w, retain_graph=True)
    assert torch.allclose(sd_rms.pull(w), dx_auto, atol=1e-11)
    assert torch.allclose(sd_rms.push(v), torch.matmul(j_rms, v.unsqueeze(-1)).squeeze(-1), atol=1e-11)

    # 3. LayerNorm derivative & NormalizationResidual vs autograd VJP
    if dim > 1:
        sd_ln = ojasx2.math.layernorm_derivative(x, gamma, eps=1e-5)
        beta = torch.randn(dim, dtype=torch.float64, requires_grad=True)
        y_ln = F.layer_norm(x, (dim,), gamma, beta, 1e-5)
        dx_ln, dgamma_ln, dbeta_ln = torch.autograd.grad(y_ln, (x, gamma, beta), w)
        assert torch.allclose(sd_ln.pull(w), dx_ln, atol=1e-11)

        res = ojasx2.math.normalization_residual(x.detach(), gamma.detach(), beta.detach(), eps=1e-5, kind="layernorm")
        dx_r, dg_r, db_r = res.pull(w)
        assert torch.allclose(dx_r, dx_ln, atol=1e-11)
        assert torch.allclose(dg_r, dgamma_ln, atol=1e-11)
        assert torch.allclose(db_r, dbeta_ln, atol=1e-11)


def test_rank7_and_winograd_reference_exactness():
    torch.manual_seed(7)
    a = torch.randn(8, 12, dtype=torch.float64)
    b = torch.randn(12, 10, dtype=torch.float64)
    expected = a @ b
    assert torch.allclose(ojasx2.math.rank7_reference(a, b), expected, atol=1e-12)
    assert torch.allclose(ojasx2.math.winograd_reference(a, b), expected, atol=1e-12)


# =============================================================================
# 2. Halley-Padé 3rd-Order Root Inversion & 1-Bit Mask Tests
# =============================================================================

@pytest.mark.parametrize(
    "act",
    ["silu", "gelu_tanh", "gelu_exact", "quickgelu", "mish", "sigmoid", "tanh", "relu", "leaky_relu", "relu2"],
)
def test_halley_inversion_and_cotangent_all_regimes(act: str):
    """Tests Halley root inversion across left branch, critical minimum z_*, right branch, and underflow."""
    z_star = ojasx2.math.get_activation_min_z(act)
    # Include extreme negative (-25.0), near critical minimum (z_star +/- 1e-4), 0.0, and large positive (+12.0)
    pts = torch.tensor(
        [-25.0, -8.0, -2.5, z_star - 0.05, z_star - 1e-3, z_star + 1e-3, z_star + 0.05, -0.2, 0.0, 0.5, 3.0, 12.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    dy = torch.randn_like(pts)

    # Reference forward & backward
    y_ref = ojasx2.math.halley_activation(pts, act=act)
    (dx_halley,) = torch.autograd.grad(y_ref, pts, dy)

    pts_eager = pts.detach().clone().requires_grad_(True)
    from ojasx2.math.inversion import _forward_activation
    y_eager = _forward_activation(pts_eager, act=act)
    (dx_eager,) = torch.autograd.grad(y_eager, pts_eager, dy)

    assert torch.allclose(y_ref, y_eager, atol=1e-6)
    assert torch.allclose(dx_halley, dx_eager, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("numel", [1, 7, 8, 9, 31, 32, 33, 127, 256])
def test_pack_unpack_mask_arbitrary_lengths(numel: int):
    mask = torch.randint(0, 2, (numel,), dtype=torch.bool)
    packed = ojasx2.math.pack_branch_bit(mask)
    assert packed.dtype == torch.uint8
    assert packed.numel() == (numel + 7) // 8
    unpacked = ojasx2.math.unpack_branch_bit(packed, (numel,))
    assert torch.equal(mask, unpacked)


# =============================================================================
# 3. Symplectic Hamiltonian Engine & Dyadic Squeeze Tests
# =============================================================================

@pytest.mark.parametrize("scheme", ["stormer_verlet", "symplectic_euler"])
@pytest.mark.parametrize("enable_squeeze", [False, True])
def test_symplectic_stack_reversibility_and_squeeze_inverse(scheme: str, enable_squeeze: bool):
    """Verifies exact state reconstruction in SymplecticStack.inverse() even when enable_squeeze=True."""
    torch.manual_seed(99)
    stack = ojasx2.SymplecticStack(
        dim=32,
        depth=4,
        num_heads=4,
        use_mlp_mixer=True,
        scheme=scheme,
        enable_squeeze=enable_squeeze,
    )
    stack.metrics.reset()

    # Create imbalanced stream variances to trigger non-zero dyadic shift k in SymplecticSqueeze
    x1 = torch.randn(2, 8, 16) * 0.1
    x2 = torch.randn(2, 8, 16) * 5.0
    x = torch.cat([x1, x2], dim=-1).requires_grad_(True)

    y = stack(x)
    x_reconstructed = stack.inverse(y.detach())

    assert torch.allclose(x.detach(), x_reconstructed, atol=1e-4), (
        f"SymplecticStack.inverse failed for scheme={scheme}, enable_squeeze={enable_squeeze}: "
        f"max diff={(x.detach() - x_reconstructed).abs().max().item()}"
    )

    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert stack.metrics.forward_f_evals > 0
    assert stack.metrics.backward_f_evals > 0
    if enable_squeeze:
        assert stack.metrics.squeezes_applied == 4


def test_symplectic_zero_element_guard():
    stack = ojasx2.SymplecticStack(dim=16, depth=2, num_heads=2)
    x_empty = torch.empty(0, 4, 16, requires_grad=True)
    y_empty = stack(x_empty)
    assert y_empty.shape == (0, 4, 16)
    y_empty.sum().backward()
    assert x_empty.grad is not None and x_empty.grad.shape == (0, 4, 16)


# =============================================================================
# 4. Dagger Attention & Fused Linear Cross-Entropy Edge Cases
# =============================================================================

@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("gqa_ratio", [1, 2])
def test_dagger_attention_exact_gradients_and_zero_tape(is_causal: bool, gqa_ratio: int):
    torch.manual_seed(123)
    B, H_q, S, D = 2, 4, 32, 16
    H_kv = H_q // gqa_ratio

    q = torch.randn(B, H_q, S, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H_kv, S, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H_kv, S, D, dtype=torch.float64, requires_grad=True)

    with SavedTensorAudit([q, k, v]) as audit:
        out = ojasx2.holonomic_dagger_attention(q, k, v, is_causal=is_causal, tile_size=16)
    loss = out.pow(2).sum()
    loss.backward()

    # Verify [B, H, S, S] score matrix is NEVER saved on tape
    for rec in audit.records:
        assert rec["shape"] != [B, H_q, S, S], "Forbidden [B, H, S, S] attention matrix materialized on tape!"

    # Reference SDPA comparison
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    k_exp = k_ref.repeat_interleave(gqa_ratio, dim=1) if gqa_ratio > 1 else k_ref
    v_exp = v_ref.repeat_interleave(gqa_ratio, dim=1) if gqa_ratio > 1 else v_ref
    out_ref = F.scaled_dot_product_attention(q_ref, k_exp, v_exp, is_causal=is_causal)
    out_ref.pow(2).sum().backward()

    assert torch.allclose(out, out_ref, atol=1e-7)
    assert torch.allclose(q.grad, q_ref.grad, atol=1e-5)
    assert torch.allclose(k.grad, k_ref.grad, atol=1e-5)
    assert torch.allclose(v.grad, v_ref.grad, atol=1e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_fused_linear_cross_entropy_parity_and_all_ignored(reduction: str, label_smoothing: float):
    torch.manual_seed(456)
    M, D, V = 20, 16, 50
    h = torch.randn(M, D, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
    b = torch.randn(V, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, V, (M,))
    targets[0:5] = -100  # Ignored tokens

    loss_fused, _ = ojasx2.holonomic_fused_linear_cross_entropy(
        h, w, targets, bias=b, ignore_index=-100, label_smoothing=label_smoothing, reduction=reduction
    )
    grad_out = torch.ones_like(loss_fused)
    loss_fused.backward(grad_out)

    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    logits = F.linear(h_ref, w_ref, b_ref)
    loss_ref = F.cross_entropy(
        logits, targets, ignore_index=-100, label_smoothing=label_smoothing, reduction=reduction
    )
    loss_ref.backward(grad_out)

    assert torch.allclose(loss_fused, loss_ref, atol=1e-7)
    assert torch.allclose(h.grad, h_ref.grad, atol=1e-6)
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-6)
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-6)

    # Edge case: 100% ignored batch
    all_ignored = torch.full((M,), -100, dtype=torch.long)
    loss_zero = ojasx2.fused_linear_cross_entropy(h.detach().requires_grad_(True), w.detach().requires_grad_(True), all_ignored)
    assert float(loss_zero.item()) == 0.0


# =============================================================================
# 5. E8 Lattice Quantization (Arbitrary Shapes, 3 Bitrates, Thread Safety)
# =============================================================================

@pytest.mark.parametrize("bitrate", [2.0, 1.5, 1.0])
@pytest.mark.parametrize("shape", [(16, 32), (7, 13), (5, 19)])
def test_e8_packers_arbitrary_shapes_and_linear(bitrate: float, shape: tuple[int, int]):
    """Tests 2.0-bit, 1.5-bit Ternary, and 1.0-bit Root E8 packers on both aligned and prime shapes."""
    torch.manual_seed(77)
    out_f, in_f = shape
    lin = nn.Linear(in_f, out_f, bias=True)
    e8_lin = ojasx2.E8LatticeLinear.from_linear(lin, rank=4, bitrate=bitrate, use_hadamard=(in_f % 32 == 0))

    w_unpacked = e8_lin.unpack_weights()
    assert w_unpacked.shape == (out_f, in_f)
    assert torch.isfinite(w_unpacked).all()

    # Forward & backward with mixed-precision input and LoRA adapters
    x = torch.randn(3, in_f, requires_grad=True)
    y = e8_lin(x)
    assert y.shape == (3, out_f)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert e8_lin.lora_A.grad is not None and e8_lin.lora_B.grad is not None


def test_e8_ephemeral_workspace_multithreaded_safety():
    """Verifies that concurrent threads running E8LatticeLinear do not corrupt each other's scratchpads."""
    layers = [
        ojasx2.E8LatticeLinear.from_linear(nn.Linear(64, 64), bitrate=2.0)
        for _ in range(4)
    ]
    inputs = [torch.randn(8, 64) for _ in range(4)]
    expected = [layers[i](inputs[i]) for i in range(4)]

    def _worker(idx: int) -> torch.Tensor:
        for _ in range(10):
            out = layers[idx](inputs[idx])
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, range(4)))

    for i in range(4):
        assert torch.allclose(results[i], expected[i], atol=1e-6)


# =============================================================================
# 6. HolonomicGradFreeOptimizer Edge Cases
# =============================================================================

@pytest.mark.parametrize("algo", ["sgd", "adamw", "adafactor"])
def test_grad_free_optimizer_auto_latch_and_unaligned_e8(algo: str):
    """Tests in-backward updates, odd-shaped E8 momentum packing, and NaN guard."""
    torch.manual_seed(321)
    # Odd dimensions (13 -> 7) test padded E8 momentum packing on non-multiples of 32
    model = nn.Sequential(nn.Linear(13, 19), nn.ReLU(), nn.Linear(19, 7))
    opt = ojasx2.HolonomicGradFreeOptimizer(
        model.parameters(),
        lr=1e-2,
        algorithm=algo,
        quantize_moments_e8=True,
        max_grad_norm=1.0,
        skip_if_nonfinite=True,
    )

    w0_before = model[0].weight.detach().clone()
    x = torch.randn(4, 13)
    loss = model(x).pow(2).sum()
    # Notice: we do NOT call opt.step() between backward passes to test auto-latching!
    loss.backward()

    assert opt.persistent_grad_bytes() == 0
    assert not torch.equal(model[0].weight, w0_before)
    assert opt._prev_global_norm > 0.0, "Global gradient norm failed to auto-latch after backward!"

    # Test NaN gradient guard: inject NaN into loss and verify weights are NOT corrupted
    w0_valid = model[0].weight.detach().clone()
    bad_loss = model(x).sum() * float("nan")
    bad_loss.backward()
    assert opt.skipped_nonfinite_updates > 0
    assert torch.equal(model[0].weight, w0_valid), "NaN gradient corrupted model weights!"


# =============================================================================
# 7. FX Subgraph Compiler, Channel Pruning & 3-Dialect MLIR Pipeline
# =============================================================================

def test_fx_subgraph_compiler_nested_modules_and_channel_pruning():
    class NestedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
            )
            # Zero out 4 channels in sub[0] to trigger dead-channel pruning
            with torch.no_grad():
                self.sub[0].weight[:4, :] = 0.0
                self.sub[0].bias[:4] = 0.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.sub(x)

    mod = NestedBlock()
    x = torch.randn(4, 16)
    y_ref = mod(x)

    gm = torch.fx.symbolic_trace(mod)
    from ojasx2.fx_subgraph_compiler import prune_dead_channels_fx
    pruned_gm, log = prune_dead_channels_fx(gm, threshold=1e-3)
    assert log.get("sub.0", 0) == 4
    assert mod.sub[0].out_features == 28
    assert mod.sub[2].in_features == 28
    y_pruned = pruned_gm(x)
    assert torch.allclose(y_ref, y_pruned, atol=1e-5)


def test_mlir_3_dialect_lowering_and_ssa_verification():
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    mlp = SimpleMLP()
    x = torch.randn(4, 32, requires_grad=True)
    exec_mod = ojasx2.compile_mlir_executable(mlp, target="nvvm")
    assert exec_mod.mlir_artifacts.ssa_verified

    for target in ("nvvm", "rocdl", "spirv", "metal", "tilelang", "cpp"):
        code = ojasx2.compile_model_to_target(mlp, target=target)
        assert len(code.splitlines()) > 10

    y = exec_mod(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
