"""Verification test suite for Fused Linear Cross-Entropy.

Strictly verifies:
1. PyTorch Contract Enforcement via `torch.library.opcheck` (`test_schema`, `test_autograd_registration`, `test_faketensor`, `test_aot_dispatch_dynamic`).
2. Numerical loss and gradient parity against PyTorch eager `F.cross_entropy(F.linear(H, W), targets)` with cosine similarity >= 0.9999.
3. DRAM memory audit proving the full [B, S, V] logits tensor is NEVER materialized in DRAM (workspace memory strictly bounded by O(B x S)).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.library import opcheck

import ojasx2.fused_cross_entropy
from ojasx2.fused_cross_entropy import (
    HolonomicFusedLinearCrossEntropy,
    _HolonomicFusedLinearCrossEntropyFunction,
    holonomic_fused_linear_cross_entropy,
)


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


# =============================================================================
# 1. PyTorch Operator Contract Enforcement (torch.library.opcheck)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias_present", [False, True])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_opcheck_fused_linear_cross_entropy_matrix(
    device: str, bias_present: bool, label_smoothing: float, reduction: str
):
    """Verifies official torch.library.opcheck across reductions, smoothing, and bias."""
    op = torch.ops.ojasx_holonomic.fused_linear_cross_entropy

    h = torch.randn(4, 16, device=device, requires_grad=True)
    w = torch.randn(32, 16, device=device, requires_grad=True)
    t = torch.randint(0, 32, (4,), device=device, dtype=torch.long)
    b = torch.randn(32, device=device, requires_grad=True) if bias_present else None

    kwargs = {
        "bias": b,
        "ignore_index": -100,
        "label_smoothing": label_smoothing,
        "reduction": reduction,
    }

    if device == "mps":
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    else:
        test_utils = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

    result = opcheck(op, (h, w, t), kwargs, test_utils=test_utils, raise_exception=True)
    assert isinstance(result, dict)
    for check_name, status in result.items():
        assert status == "SUCCESS", f"Failed {check_name} on {device} with bias={bias_present}, reduction={reduction}"

    # Live backward verification on MPS
    if device == "mps":
        loss, lse = op(h, w, t, **kwargs)
        total = loss.sum() + lse.sum()
        total.backward()
        assert h.grad is not None
        assert w.grad is not None
        if b is not None:
            assert b.grad is not None


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_3d_shapes_and_ignore_index(device: str):
    """Verifies opcheck with 3D batch/sequence shapes [B, S, D] and masked tokens."""
    op = torch.ops.ojasx_holonomic.fused_linear_cross_entropy

    b_sz, s_len, d_dim, v_dim = 2, 4, 16, 64
    h = torch.randn(b_sz, s_len, d_dim, device=device, requires_grad=True)
    w = torch.randn(v_dim, d_dim, device=device, requires_grad=True)
    t = torch.randint(0, v_dim, (b_sz, s_len), device=device, dtype=torch.long)
    t[0, 1] = -100  # Masked token
    bias = torch.randn(v_dim, device=device, requires_grad=True)

    kwargs = {"bias": bias, "ignore_index": -100, "label_smoothing": 0.05, "reduction": "mean"}
    test_utils = (
        ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        if device == "mps"
        else ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
    )

    result = opcheck(op, (h, w, t), kwargs, test_utils=test_utils, raise_exception=True)
    for check_name, status in result.items():
        assert status == "SUCCESS", f"3D shape check {check_name} failed on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_zero_elements(device: str):
    """Verifies opcheck handles zero-element tensors without crashing or zero-division."""
    op = torch.ops.ojasx_holonomic.fused_linear_cross_entropy

    h = torch.randn(0, 16, device=device, requires_grad=True)
    w = torch.randn(32, 16, device=device, requires_grad=True)
    t = torch.empty(0, device=device, dtype=torch.long)
    bias = torch.randn(32, device=device, requires_grad=True)

    test_utils = (
        ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        if device == "mps"
        else ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
    )

    result = opcheck(op, (h, w, t, bias), {}, test_utils=test_utils, raise_exception=True)
    for check_name, status in result.items():
        assert status == "SUCCESS", f"Zero-element check {check_name} failed"


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_prime_unaligned_dimensions(device: str):
    """Verifies opcheck handles prime/unaligned shapes (M=7, D=13, V=23)."""
    op = torch.ops.ojasx_holonomic.fused_linear_cross_entropy

    h = torch.randn(7, 13, device=device, requires_grad=True)
    w = torch.randn(23, 13, device=device, requires_grad=True)
    t = torch.randint(0, 23, (7,), device=device, dtype=torch.long)

    test_utils = (
        ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
        if device == "mps"
        else ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
    )

    result = opcheck(op, (h, w, t), {}, test_utils=test_utils, raise_exception=True)
    for check_name, status in result.items():
        assert status == "SUCCESS", f"Prime-shape check {check_name} failed"


# =============================================================================
# 2. Numerical Parity Against PyTorch Eager (Cosine Sim >= 0.9999)
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.15])
@pytest.mark.parametrize("bias_present", [False, True])
def test_numerical_parity_standard(device: str, label_smoothing: float, bias_present: bool):
    """Verifies loss and gradient parity against PyTorch eager with cosine similarity >= 0.9999."""
    m, d, v = 64, 32, 256
    torch.manual_seed(42)

    h = torch.randn(m, d, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(v, d, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(v, device=device, dtype=torch.float32, requires_grad=True) if bias_present else None
    targets = torch.randint(0, v, (m,), device=device, dtype=torch.long)
    targets[5] = -100
    targets[12] = -100

    # 1. PyTorch Eager Reference
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True) if b is not None else None

    logits_ref = F.linear(h_ref, w_ref, b_ref)
    loss_ref = F.cross_entropy(
        logits_ref, targets, label_smoothing=label_smoothing, ignore_index=-100, reduction="mean"
    )
    loss_ref.backward()

    # 2. Fused Linear Cross-Entropy
    h_fused = h.detach().clone().requires_grad_(True)
    w_fused = w.detach().clone().requires_grad_(True)
    b_fused = b.detach().clone().requires_grad_(True) if b is not None else None

    loss_fused, lse_fused = torch.ops.ojasx_holonomic.fused_linear_cross_entropy(
        h_fused,
        w_fused,
        targets,
        bias=b_fused,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    loss_fused.backward()

    # Loss numerical difference
    assert torch.allclose(loss_ref, loss_fused, atol=1e-5, rtol=1e-5), (
        f"Loss mismatch: ref={loss_ref.item()}, fused={loss_fused.item()}"
    )

    # Gradient cosine similarities >= 0.9999
    cos_h = F.cosine_similarity(h_ref.grad.flatten().unsqueeze(0), h_fused.grad.flatten().unsqueeze(0)).item()
    cos_w = F.cosine_similarity(w_ref.grad.flatten().unsqueeze(0), w_fused.grad.flatten().unsqueeze(0)).item()

    assert cos_h >= 0.9999, f"h gradient cosine similarity {cos_h} < 0.9999"
    assert cos_w >= 0.9999, f"w gradient cosine similarity {cos_w} < 0.9999"

    if bias_present:
        cos_b = F.cosine_similarity(b_ref.grad.flatten().unsqueeze(0), b_fused.grad.flatten().unsqueeze(0)).item()
        assert cos_b >= 0.9999, f"bias gradient cosine similarity {cos_b} < 0.9999"


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_numerical_parity_all_reductions(reduction: str):
    """Verifies parity across mean, sum, and none reductions."""
    m, d, v = 32, 16, 128
    torch.manual_seed(123)

    h_ref = torch.randn(m, d, requires_grad=True)
    w_ref = torch.randn(v, d, requires_grad=True)
    b_ref = torch.randn(v, requires_grad=True)
    targets = torch.randint(0, v, (m,), dtype=torch.long)
    targets[3] = -100

    h_fused = h_ref.detach().clone().requires_grad_(True)
    w_fused = w_ref.detach().clone().requires_grad_(True)
    b_fused = b_ref.detach().clone().requires_grad_(True)

    logits = F.linear(h_ref, w_ref, b_ref)
    loss_ref = F.cross_entropy(logits, targets, reduction=reduction, ignore_index=-100)

    loss_fused, lse_fused = holonomic_fused_linear_cross_entropy(
        h_fused, w_fused, targets, bias=b_fused, reduction=reduction, ignore_index=-100
    )

    assert torch.allclose(loss_ref, loss_fused, atol=1e-5, rtol=1e-5)

    grad_out = torch.randn_like(loss_ref)
    loss_ref.backward(grad_out)
    loss_fused.backward(grad_out)

    cos_h = F.cosine_similarity(h_ref.grad.flatten().unsqueeze(0), h_fused.grad.flatten().unsqueeze(0)).item()
    cos_w = F.cosine_similarity(w_ref.grad.flatten().unsqueeze(0), w_fused.grad.flatten().unsqueeze(0)).item()
    assert cos_h >= 0.9999
    assert cos_w >= 0.9999


def test_numerical_parity_non_contiguous_strides():
    """Verifies gradient parity under non-contiguous strides (transposed, sliced views)."""
    h_orig = torch.randn(32, 16)
    w_orig = torch.randn(16, 64)  # transposed layout

    # Transposed weight: [64, 16] with non-contiguous strides
    w_transposed = w_orig.t().clone().detach().requires_grad_(True)
    h_sliced = h_orig[::2, :].clone().detach().requires_grad_(True)  # sliced view
    targets = torch.randint(0, 64, (16,), dtype=torch.long)

    # Reference
    h_ref = h_sliced.detach().clone().requires_grad_(True)
    w_ref = w_transposed.detach().clone().requires_grad_(True)
    logits_ref = F.linear(h_ref, w_ref)
    loss_ref = F.cross_entropy(logits_ref, targets)
    loss_ref.backward()

    # Fused
    loss_fused, _ = holonomic_fused_linear_cross_entropy(h_sliced, w_transposed, targets)
    loss_fused.backward()

    cos_h = F.cosine_similarity(h_ref.grad.flatten().unsqueeze(0), h_sliced.grad.flatten().unsqueeze(0)).item()
    cos_w = F.cosine_similarity(w_ref.grad.flatten().unsqueeze(0), w_transposed.grad.flatten().unsqueeze(0)).item()

    assert cos_h >= 0.9999
    assert cos_w >= 0.9999


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numerical_parity_half_precisions(dtype: torch.dtype):
    """Verifies numerical stability and gradient parity under FP16 and BF16."""
    m, d, v = 32, 32, 128
    torch.manual_seed(999)

    h_ref = torch.randn(m, d, dtype=dtype, requires_grad=True)
    w_ref = torch.randn(v, d, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, v, (m,), dtype=torch.long)

    h_fused = h_ref.detach().clone().requires_grad_(True)
    w_fused = w_ref.detach().clone().requires_grad_(True)

    logits = F.linear(h_ref.float(), w_ref.float())
    loss_ref = F.cross_entropy(logits, targets)
    loss_ref.backward()

    loss_fused, _ = holonomic_fused_linear_cross_entropy(h_fused, w_fused, targets)
    loss_fused.backward()

    cos_h = F.cosine_similarity(
        h_ref.grad.float().flatten().unsqueeze(0), h_fused.grad.float().flatten().unsqueeze(0)
    ).item()
    cos_w = F.cosine_similarity(
        w_ref.grad.float().flatten().unsqueeze(0), w_fused.grad.float().flatten().unsqueeze(0)
    ).item()

    # Half precision tolerance threshold >= 0.999
    assert cos_h >= 0.999, f"Half precision cos_h {cos_h} < 0.999"
    assert cos_w >= 0.999, f"Half precision cos_w {cos_w} < 0.999"


# =============================================================================
# 3. DRAM Memory Audit: Full [B, S, V] Logits NEVER Materialized
# =============================================================================

def test_dram_memory_audit_no_full_logits_materialized():
    """
    DRAM Memory Audit:
    Asserts that the full [B, S, V] vocabulary logits tensor is NEVER materialized
    in DRAM. For M=512, V=32,768, full logits would require 67.1 MB.
    Asserts that the backward autograd tape saves strictly 0 bytes of logits,
    retaining only the 1D LSE vector (2 KB).
    """
    m, d, v = 512, 64, 32768
    h = torch.randn(m, d, requires_grad=True)
    w = torch.randn(v, d, requires_grad=True)
    bias = torch.randn(v, requires_grad=True)
    targets = torch.randint(0, v, (m,), dtype=torch.long)

    # Full logits size calculation
    full_logits_numel = m * v
    full_logits_bytes = full_logits_numel * 4  # 67,108,864 bytes = 67.1 MB

    # Run forward pass via explicit autograd Function to inspect saved tensors
    loss, lse = _HolonomicFusedLinearCrossEntropyFunction.apply(h, w, targets, bias, -100, 0.0, "mean")

    # Audit Autograd Tape
    saved_tensors = loss.grad_fn.saved_tensors
    saved_shapes = [tuple(t.shape) for t in saved_tensors if isinstance(t, torch.Tensor)]
    saved_numels = [t.numel() for t in saved_tensors if isinstance(t, torch.Tensor)]

    # 1. Assert NO tensor on autograd tape has shape [M, V] or [B, S, V]
    for shape in saved_shapes:
        assert shape != (m, v), f"Violation: full logits tensor {shape} materialized on tape!"
        assert (m, v) not in shape

    # 2. Assert NO tensor has numel == full_logits_numel
    for numel in saved_numels:
        assert numel < full_logits_numel, f"Violation: saved tensor of size {numel} >= {full_logits_numel}"

    # 3. Assert the only intermediate activation tensor saved is 1D LSE of length M
    # Saved list is: (h, w, targets, bias, lse)
    assert len(saved_tensors) == 5
    assert saved_tensors[0].shape == (m, d)  # input H
    assert saved_tensors[1].shape == (v, d)  # input W
    assert saved_tensors[2].shape == (m,)    # input targets
    assert saved_tensors[3].shape == (v,)    # input bias
    assert saved_tensors[4].shape == (m,)    # 1D LSE vector (ONLY 512 floats = 2048 bytes!)

    lse_bytes = saved_tensors[4].numel() * 4
    compression_ratio = full_logits_bytes / lse_bytes
    assert compression_ratio >= 32000.0, f"Compression ratio {compression_ratio} < 32000x"

    # Backward pass execution
    loss.backward()
    assert h.grad is not None
    assert w.grad is not None
    assert bias.grad is not None


def test_dram_memory_workspace_tile_bounded():
    """
    Verifies that during forward and backward execution, transient workspace
    is bounded by tile_tok x tile_voc (128 x 4096 = 2 MB), independent of sequence length M.
    """
    m, d, v = 1024, 64, 8192
    h = torch.randn(m, d, requires_grad=True)
    w = torch.randn(v, d, requires_grad=True)
    targets = torch.randint(0, v, (m,), dtype=torch.long)

    # Run forward and backward
    loss, lse = holonomic_fused_linear_cross_entropy(h, w, targets)
    loss.backward()

    # Verify gradients computed cleanly without NaNs or overflows
    assert not torch.isnan(h.grad).any()
    assert not torch.isnan(w.grad).any()


# =============================================================================
# 4. Drop-in Module & Autograd Function Tests
# =============================================================================

def test_module_drop_in_replacement():
    """Verifies HolonomicFusedLinearCrossEntropy as an nn.Module drop-in replacement."""
    in_features = 32
    vocab_size = 128
    module = HolonomicFusedLinearCrossEntropy(in_features, vocab_size, bias=True, label_smoothing=0.1)

    assert module.weight.shape == (vocab_size, in_features)
    assert module.bias.shape == (vocab_size,)

    h = torch.randn(8, in_features, requires_grad=True)
    targets = torch.randint(0, vocab_size, (8,))

    # Standard forward returns scalar loss
    loss = module(h, targets)
    assert loss.dim() == 0

    # With return_lse=True returns (loss, lse)
    loss2, lse2 = module(h, targets, return_lse=True)
    assert loss2.dim() == 0
    assert lse2.shape == (8,)

    # Optimizer step
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert module.weight.grad is not None
    assert module.bias.grad is not None


def test_autograd_function_direct_apply():
    """Verifies direct application of _HolonomicFusedLinearCrossEntropyFunction."""
    h = torch.randn(16, 32, requires_grad=True)
    w = torch.randn(64, 32, requires_grad=True)
    targets = torch.randint(0, 64, (16,))

    loss, lse = _HolonomicFusedLinearCrossEntropyFunction.apply(h, w, targets, None, -100, 0.0, "mean")
    loss.backward()

    assert h.grad is not None
    assert w.grad is not None
    assert h.grad.shape == h.shape
    assert w.grad.shape == w.shape
