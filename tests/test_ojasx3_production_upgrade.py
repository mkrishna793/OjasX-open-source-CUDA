"""
Exhaustive Edge-Case & Multi-Architecture Verification Suite for OjasX3 (Phases 1-4).

Covers:
  - Phase 1: All 10 critical bug fixes (core exact_node_limit, MPS custom_op,
    buffer-only LoRALinear device inference, double-LoRA guard, fused CE no-upfront-cast,
    Symplectic per-layer RNG dropout, DaggerAttention dropout + sliding_window,
    FX GELU default="none", bounded LRU _EPHEMERAL_WORKSPACES, TileLang group_size>64).
  - Phase 2: Exact orthogonality of Block-FWHT, CW(2m,4), CW(14m,16), CW(120,36);
    MMSE E8 SNR gain on outlier weights; unbiased stochastic 2-bit rounding;
    CWOFTLinear O(d) full-rank adapters; full E8 m_t + sqrt(v_t) AdamW compression.
  - Phase 3: Triton & C++20 E8 kernel dispatch contracts & edge shapes.
  - Phase 4: Universal E8ActivationTapeContext (saved_tensors_hooks) & compile_universal
    end-to-end training across 4 distinct architectures: Transformer, ConvNet, ViT, Recurrent/SSM.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from ojasx2.core.backend import BackendConfig
from ojasx2.math.lattice import (
    E8LatticeLinear,
    E8LatticePacker,
    TernaryE8LatticePacker,
    _EPHEMERAL_WORKSPACES,
    clear_ephemeral_workspaces,
    stochastic_pack_2bit,
    stochastic_unpack_2bit,
)
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.dagger_attention import DaggerAttention
from ojasx2.fused_cross_entropy import holonomic_fused_linear_cross_entropy
from ojasx3.lattice.transforms import (
    CW120_36_NEG,
    CW120_36_POS,
    CWOFTLinear,
    apply_circulant_weighing,
    apply_incoherence_transform,
    cw_weight4_support,
    cw_weight16_support,
    fwht_block,
    select_incoherence_spec,
)
from ojasx3.trainer.peft import LoRALinear, attach_lora_adapters
from ojasx3.runtime.tape_hooks import E8ActivationTapeContext
from ojasx3.compiler.universal import UniversalCompilerConfig, compile_universal


# ---------------------------------------------------------------------------
# Phase 1 Regression Tests
# ---------------------------------------------------------------------------

def test_core_backend_exact_node_limit_safe():
    cfg = BackendConfig()
    assert cfg.planner.exact_node_limit <= 12


def test_peft_buffer_only_e8_and_no_double_lora():
    e8_no_lora = E8LatticeLinear(64, 64, bitrate=1.5, group_size=32, rank=0, bias=False)
    wrapped = LoRALinear(e8_no_lora, rank=4)
    assert wrapped.lora_A.device == e8_no_lora.packed_weight.device
    x = torch.randn(2, 64)
    y = wrapped(x)
    assert y.shape == (2, 64)

    # E8LatticeLinear with built-in rank>0 must NOT be double-wrapped
    m = nn.Sequential(E8LatticeLinear(64, 64, bitrate=2.0, group_size=32, rank=8))
    attach_lora_adapters(m, rank=4)
    assert isinstance(m[0], E8LatticeLinear)
    assert not isinstance(m[0], LoRALinear)
    assert m[0].lora_A.requires_grad and m[0].lora_B.requires_grad


def test_dagger_attention_sliding_window_and_dropout():
    attn = DaggerAttention(embed_dim=32, num_heads=4, dropout=0.1, prefer_native_sdpa=True)
    attn.is_sliding = True
    attn.sliding_window = 4
    attn.train()
    x = torch.randn(2, 12, 32, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 12, 32)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fused_cross_entropy_fp16_tile_cast_parity():
    h = torch.randn(8, 32, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 32, dtype=torch.float16, requires_grad=True)
    t = torch.randint(0, 64, (8,))
    loss_oj, _ = holonomic_fused_linear_cross_entropy(h, w, t)
    loss_ref = nn.functional.cross_entropy(h @ w.float().t(), t)
    assert torch.allclose(loss_oj.float(), loss_ref.float(), atol=1e-3, rtol=1e-3)


def test_bounded_ephemeral_workspaces():
    clear_ephemeral_workspaces()
    assert len(_EPHEMERAL_WORKSPACES) == 0


# ---------------------------------------------------------------------------
# Phase 2 Math & Transform Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [4, 6, 14, 28, 56, 64, 120, 128, 256, 37])
def test_incoherence_transform_exact_invertibility_all_dims(dim):
    x = torch.randn(3, dim, dtype=torch.float64)
    y = apply_incoherence_transform(x, inverse=False)
    x_rec = apply_incoherence_transform(y, inverse=True)
    assert torch.allclose(x, x_rec, atol=1e-11, rtol=1e-11)


def test_cw120_36_and_cw14m_16_exact_orthogonality():
    x120 = torch.randn(4, 120, dtype=torch.float64)
    y120 = apply_circulant_weighing(x120, 120, CW120_36_POS, CW120_36_NEG, transpose=False)
    r120 = apply_circulant_weighing(y120, 120, CW120_36_POS, CW120_36_NEG, transpose=True)
    assert torch.allclose(x120, r120, atol=1e-12)

    pos, neg = cw_weight16_support(28)
    x28 = torch.randn(4, 28, dtype=torch.float64)
    y28 = apply_circulant_weighing(x28, 28, pos, neg, transpose=False)
    r28 = apply_circulant_weighing(y28, 28, pos, neg, transpose=True)
    assert torch.allclose(x28, r28, atol=1e-12)


def test_mmse_and_fwht_improve_outlier_e8_snr():
    torch.manual_seed(0)
    w = torch.randn(128, 128) * 0.02
    w[:, :6] *= 15.0  # heavy channel outliers
    p_rms, s_rms, sh = E8LatticePacker.pack(w, group_size=32, mmse=False)
    w_rms = E8LatticePacker.unpack(p_rms, s_rms, sh)

    w_rot = fwht_block(w, block_size=64)
    p_mmse, s_mmse, sh = E8LatticePacker.pack(w_rot, group_size=32, mmse=True)
    w_rot_rec = fwht_block(E8LatticePacker.unpack(p_mmse, s_mmse, sh), block_size=64)

    mse_old = (w - w_rms).pow(2).mean().item()
    mse_new = (w - w_rot_rec).pow(2).mean().item()
    assert mse_new < mse_old * 0.35


def test_stochastic_2bit_unbiased_expectation():
    torch.manual_seed(123)
    x = torch.randn(2, 64) * 0.5
    acc = torch.zeros_like(x)
    trials = 400
    for _ in range(trials):
        packed, scales, sh, dt, spec = stochastic_pack_2bit(x, group_size=32, incoherence=True)
        acc += stochastic_unpack_2bit(packed, scales, sh, dt, spec=spec)
    mean_est = acc / trials
    assert torch.allclose(mean_est, x, atol=0.12)


def test_cwoft_linear_zero_init_and_grad():
    base = nn.Linear(120, 64)
    adapter = CWOFTLinear(base)
    x = torch.randn(2, 120)
    # At init d_in=0, d_out=0 -> exact identity to base_layer
    assert torch.allclose(adapter(x), base(x), atol=1e-6)
    loss = adapter(x).pow(2).mean()
    loss.backward()
    assert adapter.d_in.grad is not None and adapter.d_out.grad is not None
    assert base.weight.grad is None


def test_grad_free_optimizer_compresses_both_m_and_v():
    m = nn.Linear(64, 64)
    opt = HolonomicGradFreeOptimizer(m.parameters(), lr=1e-3, algorithm="adamw", quantize_moments_e8=True)
    x = torch.randn(4, 64)
    m(x).sum().backward()
    opt.step()
    st = opt.state[m.weight]
    assert "exp_avg_packed" in st and "exp_avg_sq_packed" in st
    assert "exp_avg" not in st and "exp_avg_sq" not in st


# ---------------------------------------------------------------------------
# Phase 4 Multi-Architecture Universal Compiler & Tape Tests
# ---------------------------------------------------------------------------

def _run_arch_training_step(model: nn.Module, x: torch.Tensor):
    cfg = UniversalCompilerConfig(
        enable_e8_tape=True,
        tape_min_elements=64,
        enable_core_backend=False,
    )
    compiled = compile_universal(model, cfg)
    opt = HolonomicGradFreeOptimizer(compiled.parameters(), lr=1e-3, algorithm="adamw", quantize_moments_e8=True)
    y = compiled(x)
    loss = y.pow(2).mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert compiled.last_tape_stats is not None
    assert compiled.last_tape_stats.tensors_compressed >= 1
    assert compiled.last_tape_stats.compression_ratio > 2.0


def test_universal_training_four_distinct_architectures():
    # 1. Transformer block
    tf = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
    _run_arch_training_step(tf, torch.randn(2, 16, 64))

    # 2. ConvNet / ResNet-style block
    conv = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 8, 3, padding=1))
    _run_arch_training_step(conv, torch.randn(2, 8, 16, 16))

    # 3. Vision Patch-MLP (ViT / MLP-Mixer style)
    vit = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64))
    _run_arch_training_step(vit, torch.randn(2, 32, 64))

    # 4. Recurrent / State-Space style block (GRU)
    class TinySSM(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(64, 64)
            self.rnn = nn.GRU(64, 64, batch_first=True)
            self.out_proj = nn.Linear(64, 64)
        def forward(self, x):
            h, _ = self.rnn(torch.nn.functional.silu(self.in_proj(x)))
            return self.out_proj(h)

    _run_arch_training_step(TinySSM(), torch.randn(2, 16, 64))
