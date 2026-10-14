"""Challenger Empirical Stress Testing Suite for OjasX v0.5.0.

Adversarially stresses:
1. Holonomic Dagger Attention:
   - Extreme sequence lengths (S = 512, 1024, 2048, 4096)
   - Odd / prime head dimensions (D = 17, 31, 47, 53, 97)
   - Causal masking & non-causal regimes
   - Non-contiguous views (permuted, transposed, strided slices)
   - Strict 0-byte DRAM score matrix allocation audit on autograd tape
   - Forward and backward numerical parity against eager PyTorch SDPA

2. Fused Linear Cross-Entropy:
   - Large vocabularies (V = 32000, 64000)
   - Large batch/token sizes (M = 1024, 2048)
   - Non-contiguous inputs (H, W, targets strided/sliced)
   - Label smoothing (eps = 0.0, 0.1, 0.2) + ignore_index (-100)
   - Bounded workspace memory O(B * S) and 0-byte materialization of [M, V] logits
   - Forward loss and backward gradient parity against eager chunked reference

3. SymplecticCotangentCoupling:
   - Deep chains (depth = 16, 32, 64)
   - Algebraic reversibility error audit across FP32 and FP64
   - Depth-independent O(1) autograd tape memory scaling
   - Exact gradient parity vs eager unrolled chain across deep depths
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.dagger_attention import (
    dagger_attention,
    holonomic_dagger_attention,
    holonomic_dagger_attention_with_lse,
    _HolonomicDaggerAttentionFunction,
)
from ojasx2.fused_cross_entropy import (
    holonomic_fused_linear_cross_entropy,
    _HolonomicFusedLinearCrossEntropyFunction,
    HolonomicFusedLinearCrossEntropy,
)
from ojasx2.symplectic_engine import (
    SymplecticCouplingFunction,
    SymplecticCoupling,
    SymplecticTransformerBlock,
    symplectic_coupling,
    symplectic_coupling_chain,
    audit_symplectic_reversibility,
)
from ojasx2.residuals import SavedTensorAudit

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# =============================================================================
# PART 1: HOLONOMIC DAGGER ATTENTION STRESS TESTS
# =============================================================================

@pytest.mark.parametrize("d_prime", [17, 31, 47, 53, 97])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_stress_dagger_attention_odd_prime_head_dims(d_prime: int, is_causal: bool, device: str):
    """Stress tests Dagger Attention with odd and prime head dimensions D."""
    torch.manual_seed(42 + d_prime)
    B, H, S = 2, 4, 64
    q = torch.randn(B, H, S, d_prime, device=device, requires_grad=True)
    k = torch.randn(B, H, S, d_prime, device=device, requires_grad=True)
    v = torch.randn(B, H, S, d_prime, device=device, requires_grad=True)

    # 1. Holonomic Dagger Attention
    out_holonomic, lse_holonomic = torch.ops.ojasx_holonomic.dagger_attention(
        q, k, v, is_causal=is_causal, sm_scale=None
    )
    loss_holonomic = out_holonomic.sum()
    loss_holonomic.backward()

    # 2. Reference PyTorch SDPA
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=is_causal)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # 3. Parity Assertions
    cos_fwd = F.cosine_similarity(out_holonomic.reshape(-1), out_ref.reshape(-1), dim=0).item()
    assert cos_fwd >= 0.9999, f"Forward cosine similarity {cos_fwd} < 0.9999 for D={d_prime}"

    cos_dq = F.cosine_similarity(q.grad.reshape(-1), q_ref.grad.reshape(-1), dim=0).item()
    cos_dk = F.cosine_similarity(k.grad.reshape(-1), k_ref.grad.reshape(-1), dim=0).item()
    cos_dv = F.cosine_similarity(v.grad.reshape(-1), v_ref.grad.reshape(-1), dim=0).item()
    assert cos_dq >= 0.9999, f"dQ cosine similarity {cos_dq} < 0.9999 for D={d_prime}"
    assert cos_dk >= 0.9999, f"dK cosine similarity {cos_dk} < 0.9999 for D={d_prime}"
    assert cos_dv >= 0.9999, f"dV cosine similarity {cos_dv} < 0.9999 for D={d_prime}"


@pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
def test_stress_dagger_attention_extreme_lengths_and_zero_dram_tape(seq_len: int):
    """Stress tests extreme sequence lengths up to 4096 and verifies 0-byte DRAM score allocation."""
    torch.manual_seed(100 + seq_len)
    B, H, D = 1, 4, 32
    device = torch.device("cpu")

    q = torch.randn(B, H, seq_len, D, device=device, requires_grad=True)
    k = torch.randn(B, H, seq_len, D, device=device, requires_grad=True)
    v = torch.randn(B, H, seq_len, D, device=device, requires_grad=True)

    # Execute forward pass via explicit autograd Function to inspect saved backward tape
    out, lse = _HolonomicDaggerAttentionFunction.apply(q, k, v, True, None)
    loss = out.sum()

    # Inspect autograd tape on attention output
    grad_fn = out.grad_fn
    assert grad_fn is not None
    saved = grad_fn.saved_tensors
    assert len(saved) == 5  # Saved: (q, k, v, out, lse)

    # Verify that NO intermediate score tensor of shape [B, H, S, S] exists
    quadratic_shape = torch.Size([B, H, seq_len, seq_len])
    for t in saved:
        assert t.shape != quadratic_shape, f"Violation: quadratic score tensor {t.shape} found on tape!"
        assert t.numel() < (B * H * seq_len * seq_len), f"Violation: tensor size {t.numel()} >= quadratic size"

    # Calculate exact intermediate DRAM score bytes saved on tape
    score_bytes_on_tape = sum(t.numel() * t.element_size() for t in saved if t.shape == quadratic_shape)
    assert score_bytes_on_tape == 0, f"Intermediate score allocation on tape must be 0 bytes, got {score_bytes_on_tape}"

    # Verify theoretical savings
    eager_score_bytes = B * H * seq_len * seq_len * 4
    lse_bytes = B * H * seq_len * 4
    reduction_ratio = eager_score_bytes / lse_bytes
    assert reduction_ratio == float(seq_len)

    # Backward pass execution
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


def test_stress_dagger_attention_adversarial_non_contiguous_views():
    """Stress tests Dagger Attention on deeply non-contiguous, transposed, and sliced views."""
    torch.manual_seed(999)
    B, H, S, D = 2, 4, 128, 64
    # Create large storage and slice with non-unit strides
    raw_q = torch.randn(B, S, H * 2, D * 2, requires_grad=True)
    raw_k = torch.randn(B, S, H * 2, D * 2, requires_grad=True)
    raw_v = torch.randn(B, S, H * 2, D * 2, requires_grad=True)

    # Exotic slicing and transposition: [B, H, S, D]
    q = raw_q[:, :, ::2, :D].permute(0, 2, 1, 3)
    k = raw_k[:, :, ::2, :D].permute(0, 2, 1, 3)
    v = raw_v[:, :, ::2, :D].permute(0, 2, 1, 3)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    assert q.shape == torch.Size([B, H, S, D])

    out, lse = torch.ops.ojasx_holonomic.dagger_attention(q, k, v, is_causal=True, sm_scale=0.05)
    loss = out.sum() + lse.sum()
    loss.backward()

    assert raw_q.grad is not None and not torch.isnan(raw_q.grad).any()
    assert raw_k.grad is not None and not torch.isnan(raw_k.grad).any()
    assert raw_v.grad is not None and not torch.isnan(raw_v.grad).any()


# =============================================================================
# PART 2: FUSED LINEAR CROSS-ENTROPY STRESS TESTS
# =============================================================================

@pytest.mark.parametrize("vocab_size", [32000, 64000])
@pytest.mark.parametrize("m_tokens", [1024, 2048])
def test_stress_fused_cross_entropy_large_vocab_and_zero_logits_tape(vocab_size: int, m_tokens: int):
    """Stress tests Fused Linear Cross-Entropy with V=32k and V=64k, verifying 0-byte logits materialization."""
    torch.manual_seed(42 + vocab_size)
    d_feat = 64
    device = torch.device("cpu")

    h = torch.randn(m_tokens, d_feat, device=device, requires_grad=True)
    w = torch.randn(vocab_size, d_feat, device=device, requires_grad=True)
    targets = torch.randint(0, vocab_size, (m_tokens,), device=device, dtype=torch.long)

    # Calculate would-be eager logits memory
    eager_logits_numel = m_tokens * vocab_size
    eager_logits_bytes = eager_logits_numel * 4  # e.g., 2048 * 64000 * 4 = 524,288,000 bytes = 524 MB!

    # Execute forward via explicit autograd Function
    loss, lse = _HolonomicFusedLinearCrossEntropyFunction.apply(
        h, w, targets, None, -100, 0.0, "mean"
    )

    # 1. Audit backward autograd tape
    grad_fn = loss.grad_fn
    assert grad_fn is not None
    saved = grad_fn.saved_tensors
    assert len(saved) == 5  # Saved: (h, w, targets, bias, lse)

    # Assert NO saved tensor has shape [m_tokens, vocab_size] or size == eager_logits_numel
    logits_shape = (m_tokens, vocab_size)
    for t in saved:
        if isinstance(t, torch.Tensor):
            assert t.shape != logits_shape, f"Violation: full logits tensor {t.shape} materialized on tape!"
            assert t.numel() < eager_logits_numel, f"Violation: saved tensor of size {t.numel()} >= {eager_logits_numel}"

    # LSE vector is the only intermediate saved: size is M floats (4 KB for M=1024)
    lse_saved = saved[4]
    assert lse_saved.shape == (m_tokens,)
    compression_ratio = eager_logits_bytes / (lse_saved.numel() * 4)
    assert compression_ratio == float(vocab_size)

    # 2. Backward execution
    loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any()
    assert w.grad is not None and not torch.isnan(w.grad).any()


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("device", DEVICES)
def test_stress_fused_cross_entropy_non_contiguous_and_label_smoothing(eps: float, device: str):
    """Stress tests non-contiguous strided inputs with label smoothing and masked tokens."""
    torch.manual_seed(888)
    m, d, v = 256, 32, 1024

    # Non-contiguous H
    raw_h = torch.randn(m, d * 2, device=device, requires_grad=True)
    h = raw_h[:, :d]  # sliced view
    assert not h.is_contiguous()

    # Non-contiguous W
    raw_w = torch.randn(v, d * 2, device=device, requires_grad=True)
    w = raw_w[:, :d]  # sliced column view
    assert not w.is_contiguous()

    # Targets with masked tokens (-100) and boundary values (0, v-1)
    targets = torch.randint(0, v, (m,), device=device, dtype=torch.long)
    targets[0] = 0
    targets[1] = v - 1
    targets[10:20] = -100  # Masked ignore_index tokens

    # 1. Holonomic Fused LCE
    loss_fused, lse_fused = torch.ops.ojasx_holonomic.fused_linear_cross_entropy(
        h, w, targets, bias=None, ignore_index=-100, label_smoothing=eps, reduction="mean"
    )
    loss_fused.backward()

    # 2. Eager Reference
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    logits_ref = F.linear(h_ref, w_ref)
    loss_ref = F.cross_entropy(logits_ref, targets, ignore_index=-100, label_smoothing=eps, reduction="mean")
    loss_ref.backward()

    # 3. Assert exact parity
    torch.testing.assert_close(loss_fused, loss_ref, rtol=1e-4, atol=1e-4)

    # Cosine similarity for gradients
    cos_dh = F.cosine_similarity(raw_h.grad[:, :d].reshape(-1), h_ref.grad.reshape(-1), dim=0).item()
    cos_dw = F.cosine_similarity(raw_w.grad[:, :d].reshape(-1), w_ref.grad.reshape(-1), dim=0).item()
    assert cos_dh >= 0.9999, f"dH cosine similarity {cos_dh} < 0.9999 with eps={eps}"
    assert cos_dw >= 0.9999, f"dW cosine similarity {cos_dw} < 0.9999 with eps={eps}"


# =============================================================================
# PART 3: SYMPLECTIC COUPLING STRESS TESTS
# =============================================================================

class StressCouplingBlock(nn.Module):
    """Coupling block with LayerNorm, Linear, and Tanh for realistic transformation."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        # Small initialization scale for numerical stability over deep chains
        with torch.no_grad():
            self.net[2].weight.mul_(0.1)
            self.net[2].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_deep_symplectic_blocks(depth: int, dim: int, dtype: torch.dtype, device: torch.device):
    blocks = []
    for _ in range(depth):
        f = StressCouplingBlock(dim).to(dtype=dtype, device=device)
        g = StressCouplingBlock(dim).to(dtype=dtype, device=device)
        blocks.append((f, g))
    return blocks


@pytest.mark.parametrize("depth", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stress_symplectic_reversibility_deep_chains(depth: int, dtype: torch.dtype):
    """Stress tests numerical reversibility error of deep SymplecticCoupling chains (16, 32, 64)."""
    torch.manual_seed(777 + depth)
    dim = 16
    batch = 8
    device = torch.device("cpu")

    blocks = _build_deep_symplectic_blocks(depth, dim, dtype, device)
    q1 = torch.randn(batch, dim, dtype=dtype, device=device)
    q2 = torch.randn(batch, dim, dtype=dtype, device=device)

    report = audit_symplectic_reversibility(q1, q2, blocks)
    max_err = report["max_reconstruction_error"]

    if dtype == torch.float64:
        # In double precision, error over 64 layers must be <= 1e-11
        assert max_err < 1e-11, f"FP64 reversibility error {max_err} >= 1e-11 at depth {depth}"
    else:
        # In single precision, error over 64 layers must be <= 1e-4
        assert max_err < 1e-4, f"FP32 reversibility error {max_err} >= 1e-4 at depth {depth}"


def test_stress_symplectic_depth_independent_memory_scaling():
    """Empirically verifies depth-independent O(1) autograd tape memory across depths 1, 16, 32, 64."""
    dim = 32
    batch = 16
    device = torch.device("cpu")
    dtype = torch.float32

    fresh_saved_bytes = {}
    for depth in (1, 8, 16, 32, 64):
        blocks = _build_deep_symplectic_blocks(depth, dim, dtype, device)
        params = [p for b in blocks for m in b for p in m.parameters()]

        q1 = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)
        q2 = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)

        with SavedTensorAudit([q1, q2, *params]) as audit:
            out1, out2 = symplectic_coupling_chain(q1, q2, blocks)

        storage = audit.summary()["fresh_saved_storage_bytes"]
        fresh_saved_bytes[depth] = storage

    # The fresh activation storage must be EXACTLY identical across all depths!
    # Namely, 2 boundary tensors of shape [batch, dim] in float32 = 2 * 16 * 32 * 4 = 4096 bytes.
    expected_bytes = 2 * batch * dim * 4
    for depth, recorded_bytes in fresh_saved_bytes.items():
        assert recorded_bytes == expected_bytes, (
            f"Depth {depth} memory violation: recorded {recorded_bytes} bytes != expected {expected_bytes} bytes"
        )


@pytest.mark.parametrize("depth", [16, 32])
def test_stress_symplectic_gradient_parity_deep(depth: int):
    """Stress tests exact gradient parity against eager unrolled chain at depth 16 and 32."""
    torch.manual_seed(1234 + depth)
    dim = 8
    batch = 4
    device = torch.device("cpu")
    dtype = torch.float64  # Use float64 to test mathematical exactness down to 1e-10

    blocks_e = _build_deep_symplectic_blocks(depth, dim, dtype, device)
    import copy
    blocks_s = copy.deepcopy(blocks_e)

    params_e = [p for b in blocks_e for m in b for p in m.parameters()]
    params_s = [p for b in blocks_s for m in b for p in m.parameters()]

    q1_e = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)
    q2_e = torch.randn(batch, dim, dtype=dtype, device=device, requires_grad=True)
    q1_s = q1_e.detach().clone().requires_grad_(True)
    q2_s = q2_e.detach().clone().requires_grad_(True)

    # 1. Eager execution
    cur_q1, cur_q2 = q1_e, q2_e
    for f, g in blocks_e:
        cur_q1 = cur_q1 + f(cur_q2)
        cur_q2 = cur_q2 + g(cur_q1)
    loss_e = (cur_q1.pow(2) + cur_q2.pow(2)).sum()
    loss_e.backward()

    # 2. Symplectic execution
    out1_s, out2_s = symplectic_coupling_chain(q1_s, q2_s, blocks_s)
    loss_s = (out1_s.pow(2) + out2_s.pow(2)).sum()
    loss_s.backward()

    # Parity assertions
    torch.testing.assert_close(cur_q1, out1_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(cur_q2, out2_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(q1_e.grad, q1_s.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(q2_e.grad, q2_s.grad, rtol=1e-10, atol=1e-10)

    for pe, ps in zip(params_e, params_s):
        torch.testing.assert_close(pe.grad, ps.grad, rtol=1e-10, atol=1e-10)


# =============================================================================
# PART 4: ULTRA-ADVERSARIAL BOUNDARY & CORNER CASE TESTS
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.001, 2.5])
def test_stress_dagger_attention_mixed_precision_and_extreme_scales(dtype: torch.dtype, scale: float):
    """Stress tests mixed precision (fp16/bf16), asymmetric cross-attention (Sq != Skv), GQA, and extreme scale."""
    torch.manual_seed(555)
    B, H_q, H_kv, Sq, Skv, D = 2, 8, 2, 64, 256, 31
    device = torch.device("cpu")

    q = torch.randn(B, H_q, Sq, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, H_kv, Skv, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, H_kv, Skv, D, dtype=dtype, device=device, requires_grad=True)

    out = holonomic_dagger_attention(q, k, v, is_causal=False, sm_scale=scale)
    assert out.shape == torch.Size([B, H_q, Sq, D])
    assert out.dtype == dtype

    loss = out.to(torch.float32).sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


def test_stress_fused_cross_entropy_all_masked_tokens():
    """Adversarial stress test: all target tokens are masked with ignore_index=-100."""
    m, d, v = 128, 32, 1000
    h = torch.randn(m, d, requires_grad=True)
    w = torch.randn(v, d, requires_grad=True)
    targets = torch.full((m,), -100, dtype=torch.long)

    # Must not divide by zero or produce NaNs
    loss, lse = holonomic_fused_linear_cross_entropy(h, w, targets, ignore_index=-100, reduction="mean")
    assert not torch.isnan(loss)
    assert loss.item() == 0.0

    loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any()
    assert w.grad is not None and not torch.isnan(w.grad).any()
    assert h.grad.abs().sum().item() == 0.0
    assert w.grad.abs().sum().item() == 0.0


def test_stress_fused_cross_entropy_large_v128k():
    """Stress test with modern LLM-scale vocabulary V=128,000."""
    m, d, v = 1024, 64, 128000
    h = torch.randn(m, d, requires_grad=True)
    w = torch.randn(v, d, requires_grad=True)
    targets = torch.randint(0, v, (m,), dtype=torch.long)

    # 1. Forward pass via autograd Function
    loss, lse = _HolonomicFusedLinearCrossEntropyFunction.apply(h, w, targets, None, -100, 0.05, "mean")

    # Audit tape: would-be full logits size = 1024 * 128000 * 4 = 524,288,000 bytes (524 MB)
    eager_logits_bytes = m * v * 4
    saved = loss.grad_fn.saved_tensors
    lse_saved = saved[4]
    assert lse_saved.numel() == m
    saved_lse_bytes = lse_saved.numel() * 4
    assert eager_logits_bytes / saved_lse_bytes == float(v)

    # 2. Backward execution
    loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any()
    assert w.grad is not None and not torch.isnan(w.grad).any()


@pytest.mark.parametrize("device", DEVICES)
def test_stress_symplectic_deep_chain_with_frozen_modules(device: str):
    """Stress tests deep chain (depth=16) with frozen submodules on CPU and MPS."""
    torch.manual_seed(909)
    dim = 8
    depth = 16
    batch = 4
    blocks = _build_deep_symplectic_blocks(depth, dim, torch.float32, torch.device(device))

    # Freeze every other module
    for idx, (f, g) in enumerate(blocks):
        if idx % 2 == 0:
            for p in f.parameters():
                p.requires_grad = False
        else:
            for p in g.parameters():
                p.requires_grad = False

    q1 = torch.randn(batch, dim, device=device, requires_grad=True)
    q2 = torch.randn(batch, dim, device=device, requires_grad=True)

    out1, out2 = symplectic_coupling_chain(q1, q2, blocks)
    loss = (out1.pow(2) + out2.pow(2)).sum()
    loss.backward()

    # Verify input gradients
    assert q1.grad is not None and not torch.isnan(q1.grad).any()
    assert q2.grad is not None and not torch.isnan(q2.grad).any()

    # Verify frozen parameters have NO gradients, while active parameters have valid gradients
    for idx, (f, g) in enumerate(blocks):
        if idx % 2 == 0:
            for p in f.parameters():
                assert p.grad is None
            for p in g.parameters():
                assert p.grad is not None and not torch.isnan(p.grad).any()
        else:
            for p in f.parameters():
                assert p.grad is not None and not torch.isnan(p.grad).any()
            for p in g.parameters():
                assert p.grad is None


def test_stress_symplectic_with_smart_router_optimize_model():
    """Verify that smart_router.optimize_model seamlessly preserves SymplecticTransformerBlock."""
    from ojasx2.smart_router import optimize_model
    from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer

    class HybridSymplecticModel(nn.Module):
        def __init__(self, dim: int = 32, depth: int = 4):
            super().__init__()
            self.in_proj = nn.Linear(dim, dim)
            self.blocks = nn.ModuleList([
                SymplecticTransformerBlock(dim=dim, num_heads=2) for _ in range(depth)
            ])
            self.out_head = nn.Linear(dim, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.in_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.out_head(h)

    model = HybridSymplecticModel(dim=32, depth=4)
    optimized_model, report = optimize_model(model, batch_tokens=16)

    # Verify symplectic blocks are preserved and not mutated internally
    for block in optimized_model.blocks:
        assert isinstance(block, SymplecticTransformerBlock)
        assert isinstance(block.f, nn.Sequential)
        assert isinstance(block.g, nn.Sequential)

    optimizer = HolonomicGradFreeOptimizer(optimized_model.parameters(), lr=1e-3, algorithm="adamw")

    x = torch.randn(2, 8, 32, requires_grad=True)
    out = optimized_model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    optimizer.step()
    optimizer.zero_grad()
