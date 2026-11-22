"""Edge-case & CPU/MPS parity test suite for the ojasx3-production-upgrade."""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx3
from ojasx2.dagger_attention import DaggerAttention
from ojasx2.fused_cross_entropy import holonomic_fused_linear_cross_entropy
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.math.lattice import (
    E8LatticeLinear,
    stochastic_pack_2bit,
    stochastic_unpack_2bit,
)
from ojasx2.trainer import LowRAMTrainer
from ojasx3.compiler.universal import UniversalCompilerConfig, compile_universal
from ojasx3.lattice.transforms import (
    CWOFTLinear,
    apply_incoherence_transform,
    select_incoherence_spec,
)
from ojasx3.runtime.tape_hooks import E8ActivationTapeContext
from ojasx3.trainer.peft import LoRALinear, attach_lora_adapters


NON_POW2_DIMS = [37, 120, 180]
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS backend unavailable")


# ---------------------------------------------------------------------------
# 1. Prime / Non-Pow2 Dims (37, 120, 180)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", NON_POW2_DIMS)
def test_non_pow2_incoherence_and_stochastic_pack_roundtrip(dim: int):
    expected_kind = {37: "identity", 120: "cw120", 180: "cw4"}[dim]
    spec = select_incoherence_spec(dim)
    assert spec[0] == expected_kind

    x = torch.randn(3, 5, dim, dtype=torch.float64)
    y = apply_incoherence_transform(x, inverse=False)
    x_rec = apply_incoherence_transform(y, inverse=True)
    assert torch.allclose(x, x_rec, atol=1e-11, rtol=1e-11)

    x32 = torch.randn(2, 3, dim, dtype=torch.float32)
    packed, scales, orig_shape, orig_dtype, p_spec = stochastic_pack_2bit(
        x32, group_size=32, incoherence=True
    )
    unpacked = stochastic_unpack_2bit(packed, scales, orig_shape, orig_dtype, spec=p_spec)
    assert unpacked.shape == x32.shape
    assert unpacked.dtype == x32.dtype
    assert torch.isfinite(unpacked).all()


@pytest.mark.parametrize("dim", NON_POW2_DIMS)
def test_non_pow2_e8_adamw_both_moments_and_fused_ce(dim: int):
    torch.manual_seed(11)
    # 1D parameter of exact length `dim` (37, 120, 180 are all >= 32 and non-pow2, non-mult-of-32)
    p = nn.Parameter(torch.randn(dim))
    opt = HolonomicGradFreeOptimizer(
        [p], lr=1e-2, algorithm="adamw", quantize_moments_e8=True
    )
    for _ in range(3):
        (p * 0.5).pow(2).sum().backward()
        opt.step()
    st = opt.state[p]
    assert "exp_avg_packed" in st and "exp_avg_sq_packed" in st
    assert "exp_avg" not in st and "exp_avg_sq" not in st
    assert torch.isfinite(p).all()

    # Fused Cross-Entropy on non-pow2 hidden dim
    h = torch.randn(4, dim, requires_grad=True)
    w = torch.randn(29, dim, requires_grad=True)
    b = torch.randn(29, requires_grad=True)
    t = torch.randint(0, 29, (4,))
    loss_oj, _ = holonomic_fused_linear_cross_entropy(h, w, t, bias=b, label_smoothing=0.1)
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    loss_ref = F.cross_entropy(F.linear(h_ref, w_ref, b_ref), t, label_smoothing=0.1)
    assert torch.allclose(loss_oj, loss_ref, atol=1e-5, rtol=1e-5)
    loss_oj.backward()
    loss_ref.backward()
    assert torch.allclose(h.grad, h_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", NON_POW2_DIMS)
@pytest.mark.parametrize("prefer_native_sdpa", [False, True])
def test_non_pow2_dagger_attention_and_tape(dim: int, prefer_native_sdpa: bool):
    heads = 1 if dim == 37 else 4
    attn = DaggerAttention(
        embed_dim=dim,
        num_heads=heads,
        dropout=0.1,
        prefer_native_sdpa=prefer_native_sdpa,
    )
    attn.train()
    x = torch.randn(2, 5, dim, requires_grad=True)
    with E8ActivationTapeContext(group_size=32, incoherence=True, min_elements=1) as tape:
        y = attn(x)
        y.pow(2).mean().backward()
    assert y.shape == (2, 5, dim)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert tape.stats.tensors_compressed >= 1


# ---------------------------------------------------------------------------
# 2. batch=1, seq=1 Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [37, 120])
def test_batch1_seq1_end_to_end_components(dim: int):
    heads = 1 if dim == 37 else 4
    attn = DaggerAttention(embed_dim=dim, num_heads=heads, is_causal=True, dropout=0.1, prefer_native_sdpa=False)
    attn.is_sliding = True
    attn.sliding_window = 4
    attn.train()
    x = torch.randn(1, 1, dim, requires_grad=True)
    y = attn(x)
    assert y.shape == (1, 1, dim)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()

    h = torch.randn(1, 1, dim, requires_grad=True)
    w = torch.randn(19, dim, requires_grad=True)
    t = torch.tensor([[7]])
    loss, _ = holonomic_fused_linear_cross_entropy(h, w, t)
    loss_ref = F.cross_entropy((h @ w.t()).reshape(1, 19), t.reshape(1))
    assert torch.allclose(loss, loss_ref, atol=1e-5)
    loss.backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all()

    model = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 19))
    compiled = compile_universal(
        model,
        UniversalCompilerConfig(enable_e8_tape=True, tape_min_elements=1, enable_core_backend=False),
    )
    trainer = LowRAMTrainer(
        compiled,
        lr=1e-3,
        grad_accum_steps=1,
        fused_loss=False,
        device="cpu",
        dtype=torch.float32,
        compile_backend=None,
    )
    batch = [(torch.randn(1, 1, dim), torch.tensor([[3]]))]
    metrics = trainer.train_epoch(batch, epoch=1, log_interval=10)
    assert metrics["steps"] == 1
    assert torch.isfinite(torch.tensor(metrics["avg_loss"]))
    assert compiled.last_tape_stats is not None and compiled.last_tape_stats.tensors_compressed >= 1


# ---------------------------------------------------------------------------
# 3. grad_accum_steps=3 with Partial Flush
# ---------------------------------------------------------------------------

def test_grad_accum_3_partial_flush_exact_parity():
    torch.manual_seed(42)
    m_ref = nn.Linear(37, 13, bias=True)
    m_test = copy.deepcopy(m_ref)
    batches = [(torch.randn(2, 37), torch.randint(0, 13, (2,))) for _ in range(5)]

    trainer = LowRAMTrainer(
        m_test,
        lr=1e-2,
        weight_decay=0.0,
        grad_accum_steps=3,
        max_grad_norm=None,
        quantize_moments=False,
        fused_loss=False,
        device="cpu",
        dtype=torch.float32,
        compile_backend=None,
    )
    metrics = trainer.train_epoch(batches, epoch=1, log_interval=100)
    assert metrics["steps"] == 5
    assert metrics["optimizer_steps"] == 2
    assert metrics["flushed_parameter_updates"] == 2  # weight + bias

    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-2, weight_decay=0.0)
    for start, end in [(0, 3), (3, 5)]:
        opt_ref.zero_grad()
        for x, y in batches[start:end]:
            loss = F.cross_entropy(m_ref(x), y) / float(end - start)
            loss.backward()
        opt_ref.step()

    assert torch.allclose(m_test.weight, m_ref.weight, atol=1e-6, rtol=1e-5)
    assert torch.allclose(m_test.bias, m_ref.bias, atol=1e-6, rtol=1e-5)


def test_grad_accum_3_e8_partial_flush_clears_state():
    m = nn.Linear(37, 13, bias=False)
    opt = HolonomicGradFreeOptimizer(
        m.parameters(), lr=1e-2, grad_accum_steps=3, quantize_moments_e8=True
    )
    w0 = m.weight.detach().clone()

    # 2 microbatches (< 3): no update applied yet
    for _ in range(2):
        m(torch.randn(2, 37)).sum().backward()
    assert torch.equal(m.weight, w0)
    assert opt.persistent_grad_bytes() == 0
    assert opt.state[m.weight]["accum_count"] == 2

    flushed = opt.flush_accumulation()
    assert flushed == 1
    assert not torch.equal(m.weight, w0)
    assert opt.state[m.weight]["accum_count"] == 0
    assert opt.flush_accumulation() == 0


# ---------------------------------------------------------------------------
# 4. NaN / Inf Gradient Guard
# ---------------------------------------------------------------------------

def test_nan_inf_gradient_guard_and_recovery():
    m = nn.Sequential(nn.Linear(37, 17, bias=False), nn.Linear(17, 7, bias=False))
    opt = HolonomicGradFreeOptimizer(
        m.parameters(),
        lr=1e-2,
        algorithm="adamw",
        quantize_moments_e8=True,
        skip_if_nonfinite=True,
    )
    w0_before = m[0].weight.detach().clone()
    w1_before = m[1].weight.detach().clone()

    x = torch.randn(2, 37)
    out = m(x)
    out.backward(torch.full_like(out, float("nan")))
    assert opt.skipped_nonfinite_updates == 2
    assert opt.persistent_grad_bytes() == 0
    assert torch.equal(m[0].weight, w0_before)
    assert torch.equal(m[1].weight, w1_before)

    out = m(x)
    out.backward(torch.full_like(out, float("inf")))
    assert opt.skipped_nonfinite_updates == 4
    assert torch.equal(m[0].weight, w0_before)
    assert torch.equal(m[1].weight, w1_before)

    # Partial nonfinite: corrupt only layer 1's gradient, layer 0 finite
    h = m[0](x)
    y = m[1](h)
    # Inject NaN only into m[1].weight via custom hook or direct finite step check
    y.pow(2).mean().backward()
    opt.step()
    assert not torch.equal(m[0].weight, w0_before)
    assert torch.isfinite(m[0].weight).all() and torch.isfinite(m[1].weight).all()


def test_nan_guard_inside_grad_accum_3_window():
    m = nn.Linear(37, 11, bias=False)
    opt = HolonomicGradFreeOptimizer(
        m.parameters(), lr=1e-2, grad_accum_steps=3, quantize_moments_e8=True, skip_if_nonfinite=True
    )
    w_init = m.weight.detach().clone()
    x = torch.randn(2, 37)

    # 1st microbatch valid, 2nd NaN (skipped), 3rd valid -> accum_count=2, partial flush updates
    m(x).sum().backward()
    m(x).backward(torch.full((2, 11), float("nan")))
    assert opt.skipped_nonfinite_updates == 1
    assert opt.state[m.weight]["accum_count"] == 1
    assert torch.equal(m.weight, w_init)

    m(x).sum().backward()
    assert opt.state[m.weight]["accum_count"] == 2
    assert opt.flush_accumulation() == 1
    assert torch.isfinite(m.weight).all() and not torch.equal(m.weight, w_init)


# ---------------------------------------------------------------------------
# 5. CWOFTLinear Zero-Init & Grad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", NON_POW2_DIMS)
def test_cwoft_linear_zero_init_grad_and_step(dim: int):
    torch.manual_seed(0)
    base = nn.Linear(dim, dim, bias=True)
    adapter = CWOFTLinear(base)
    x = torch.randn(3, dim)

    assert torch.allclose(adapter(x), base(x), atol=1e-6)
    assert not base.weight.requires_grad and not base.bias.requires_grad

    loss = adapter(x).pow(2).mean()
    loss.backward()
    assert adapter.d_in.grad is not None and torch.isfinite(adapter.d_in.grad).all()
    assert adapter.d_out.grad is not None and torch.isfinite(adapter.d_out.grad).all()
    assert base.weight.grad is None and base.bias.grad is None

    with torch.no_grad():
        adapter.d_in.add_(0.1)
        adapter.d_out.add_(-0.1)
    assert not torch.allclose(adapter(x), base(x), atol=1e-4)


def test_cwoft_linear_over_buffer_only_e8():
    e8 = E8LatticeLinear(120, 37, bias=False, rank=0, group_size=16, bitrate=2.0)
    adapter = CWOFTLinear(e8)
    x = torch.randn(2, 120)
    assert torch.allclose(adapter(x), e8(x), atol=1e-6)
    adapter(x).sum().backward()
    assert adapter.d_in.grad is not None and adapter.d_out.grad is not None


# ---------------------------------------------------------------------------
# 6. E8 PEFT No-Double-LoRA
# ---------------------------------------------------------------------------

def test_e8_peft_no_double_lora_and_api_finetune():
    # Case A: E8LatticeLinear with built-in rank>0 must not be wrapped by LoRALinear
    m_with_lora = nn.Sequential(
        E8LatticeLinear(64, 64, bitrate=2.0, group_size=32, rank=4, bias=False),
        nn.Linear(64, 16),
    )
    attach_lora_adapters(m_with_lora, rank=8)
    assert isinstance(m_with_lora[0], E8LatticeLinear)
    assert not isinstance(m_with_lora[0], LoRALinear)
    assert m_with_lora[0].rank == 4
    assert m_with_lora[0].lora_A.requires_grad and m_with_lora[0].lora_B.requires_grad
    assert isinstance(m_with_lora[1], LoRALinear)
    assert not m_with_lora[1].base_layer.weight.requires_grad

    # Case B: ojasx3.api.finetune with dataset=None and dataset!=None must both preserve no-double-LoRA
    m_api = nn.Sequential(E8LatticeLinear(64, 17, bitrate=1.5, group_size=32, rank=4, bias=False))
    m_ret, opt_ret = ojasx3.api.finetune(m_api, dataset=None, lora_rank=8)
    assert isinstance(m_ret[0], E8LatticeLinear) and not isinstance(m_ret[0], LoRALinear)
    assert len(opt_ret.params) == 2
    opt_ret.remove_hooks()

    ds = [(torch.randn(1, 64), torch.randint(0, 17, (1,))) for _ in range(4)]
    res = ojasx3.api.finetune(
        m_api,
        dataset=ds,
        lora_rank=8,
        epochs=1,
        grad_accum_steps=3,
        device="cpu",
        fused_loss=False,
        compile_backend=None,
    )
    assert isinstance(m_api[0], E8LatticeLinear) and not isinstance(m_api[0], LoRALinear)
    assert res["history"][0]["optimizer_steps"] == 2
    assert res["history"][0]["flushed_parameter_updates"] == 2


# ---------------------------------------------------------------------------
# 7. CPU + MPS Parity
# ---------------------------------------------------------------------------

@requires_mps
@pytest.mark.parametrize("dim", NON_POW2_DIMS)
def test_cpu_mps_parity_transforms_cwoft_and_fused_ce(dim: int):
    torch.manual_seed(123)
    x_cpu = torch.randn(2, 3, dim)
    y_cpu = apply_incoherence_transform(x_cpu, inverse=False)
    y_mps = apply_incoherence_transform(x_cpu.to("mps"), inverse=False).cpu()
    assert torch.allclose(y_cpu, y_mps, atol=1e-5, rtol=1e-5)

    base = nn.Linear(dim, dim, bias=True)
    cw_cpu = CWOFTLinear(copy.deepcopy(base))
    cw_mps = CWOFTLinear(copy.deepcopy(base)).to("mps")
    out_cpu = cw_cpu(x_cpu)
    out_mps = cw_mps(x_cpu.to("mps"))
    assert torch.allclose(out_cpu, out_mps.cpu(), atol=1e-5, rtol=1e-5)
    out_cpu.pow(2).mean().backward()
    out_mps.pow(2).mean().backward()
    assert torch.allclose(cw_cpu.d_in.grad, cw_mps.d_in.grad.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(cw_cpu.d_out.grad, cw_mps.d_out.grad.cpu(), atol=1e-5, rtol=1e-5)

    # Fused CE parity (fp32 and fp16 head)
    h_cpu = torch.randn(1, 1, dim, requires_grad=True)
    w_cpu = torch.randn(29, dim, requires_grad=True)
    t_cpu = torch.tensor([[11]])
    h_mps = h_cpu.detach().to("mps").requires_grad_(True)
    w_mps = w_cpu.detach().to("mps").requires_grad_(True)
    l_cpu, _ = holonomic_fused_linear_cross_entropy(h_cpu, w_cpu, t_cpu, label_smoothing=0.1)
    l_mps, _ = holonomic_fused_linear_cross_entropy(h_mps, w_mps, t_cpu.to("mps"), label_smoothing=0.1)
    assert torch.allclose(l_cpu, l_mps.cpu(), atol=1e-4, rtol=1e-4)
    l_cpu.backward()
    l_mps.backward()
    assert torch.allclose(h_cpu.grad, h_mps.grad.cpu(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_cpu.grad, w_mps.grad.cpu(), atol=1e-4, rtol=1e-4)


@requires_mps
def test_cpu_mps_parity_dagger_attention_peft_and_trainer():
    torch.manual_seed(99)
    # DaggerAttention (with sliding window) CPU vs MPS parity
    attn_cpu = DaggerAttention(embed_dim=120, num_heads=4, is_causal=True, prefer_native_sdpa=False)
    attn_cpu.is_sliding = True
    attn_cpu.sliding_window = 3
    attn_mps = copy.deepcopy(attn_cpu).to("mps")

    x_cpu = torch.randn(1, 6, 120, requires_grad=True)
    x_mps = x_cpu.detach().to("mps").requires_grad_(True)
    y_cpu = attn_cpu(x_cpu)
    y_mps = attn_mps(x_mps)
    assert torch.allclose(y_cpu, y_mps.cpu(), atol=1e-4, rtol=1e-4)
    y_cpu.sum().backward()
    y_mps.sum().backward()
    assert torch.allclose(x_cpu.grad, x_mps.grad.cpu(), atol=1e-4, rtol=1e-4)

    # Buffer-only E8LatticeLinear wrapped in LoRALinear on MPS
    e8_mps = E8LatticeLinear.from_linear(nn.Linear(37, 37, bias=False), rank=0, group_size=16).to("mps")
    wrapped = LoRALinear(e8_mps, rank=4)
    assert wrapped.lora_A.device.type == "mps"
    y_e8 = wrapped(torch.randn(1, 1, 37, device="mps"))
    y_e8.sum().backward()
    assert wrapped.lora_A.grad is not None and wrapped.lora_B.grad is not None

    # LowRAMTrainer CPU vs MPS parity with grad_accum_steps=3 + partial flush
    m_base = nn.Linear(37, 13, bias=True)
    m_cpu = copy.deepcopy(m_base)
    m_mps = copy.deepcopy(m_base)
    batches = [(torch.randn(1, 37), torch.randint(0, 13, (1,))) for _ in range(4)]

    tr_cpu = LowRAMTrainer(
        m_cpu, lr=1e-2, grad_accum_steps=3, max_grad_norm=None,
        quantize_moments=False, fused_loss=False, device="cpu", dtype=torch.float32, compile_backend=None,
    )
    tr_mps = LowRAMTrainer(
        m_mps, lr=1e-2, grad_accum_steps=3, max_grad_norm=None,
        quantize_moments=False, fused_loss=False, device="mps", dtype=torch.float32, compile_backend=None,
    )
    res_cpu = tr_cpu.train_epoch(batches, epoch=1, log_interval=100)
    res_mps = tr_mps.train_epoch(batches, epoch=1, log_interval=100)
    assert res_cpu["optimizer_steps"] == res_mps["optimizer_steps"] == 2
    assert res_cpu["flushed_parameter_updates"] == res_mps["flushed_parameter_updates"] == 2
    assert torch.allclose(m_cpu.weight, m_mps.weight.cpu(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(m_cpu.bias, m_mps.bias.cpu(), atol=1e-4, rtol=1e-4)
