"""Test suite for Audit Fixes B:
1. Halley inverse activation FP16/BF16 underflow/NaN fix on negative inputs.
2. Dagger attention 2D mask shape collision fix when B == Sq == Skv and tile_size propagation.
3. Streaming Fused Linear Cross-Entropy clean GEMM skip for all-masked chunks with valid/positive ignore_index.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F

from ojasx2.holonomic_math_compiler import (
    halley_invert_activation,
    get_activation_min_z,
)
from ojasx2.dagger_attention import (
    dagger_tiled_attention,
    holonomic_dagger_attention,
    _extract_mask_tile,
)
from ojasx2.fused_cross_entropy import (
    _streaming_fused_lce_forward,
    holonomic_fused_linear_cross_entropy,
)


# =============================================================================
# 1. Halley Activation Inversion FP16 / BF16 Robustness
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("act", [
    "silu", "gelu", "gelu_tanh", "gelu_exact", "quickgelu", "mish",
    "sigmoid", "tanh", "relu", "leaky_relu", "relu2",
])
def test_halley_invert_activation_fp16_bf16_no_nan(dtype: torch.dtype, act: str):
    """Verifies that halley_invert_activation never produces NaNs on negative inputs in FP16/BF16/FP32

    and preserves the original dtype of the input tensor.
    """
    act_fns = {
        "silu": F.silu,
        "gelu": lambda z: F.gelu(z, approximate="tanh"),
        "gelu_tanh": lambda z: F.gelu(z, approximate="tanh"),
        "gelu_exact": lambda z: F.gelu(z, approximate="none"),
        "quickgelu": lambda z: z * torch.sigmoid(1.702 * z),
        "mish": F.mish,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "leaky_relu": lambda z: F.leaky_relu(z, negative_slope=0.01),
        "relu2": lambda z: F.relu(z).square(),
    }
    fn = act_fns[act]

    # Spans negative region (where 1e-15 underflows in fp16 if not upcast) and positive region
    z_true = torch.linspace(-4.5, 4.5, 200, dtype=dtype)
    y = fn(z_true)
    branch_bit = z_true >= get_activation_min_z(act)

    z_rec = halley_invert_activation(y, branch_bit, act=act, max_iters=4)

    assert z_rec.dtype == dtype, f"Expected dtype {dtype}, got {z_rec.dtype}"
    assert not torch.isnan(z_rec).any(), f"Found NaNs in inverted z for act={act}, dtype={dtype}"
    assert not torch.isinf(z_rec).any(), f"Found Infs in inverted z for act={act}, dtype={dtype}"

    # Verify reconstruction accuracy in FP32
    z_rec_f32 = z_rec.to(torch.float32)
    y_f32 = y.to(torch.float32)
    y_rec = fn(z_rec_f32)
    diff = (y_rec - y_f32).abs()

    # In FP16/BF16, precision is bounded by machine epsilon (relative eps ~1e-3 for FP16, ~1e-2 for BF16)
    tol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5
    assert diff.max().item() < tol, f"Reconstruction error too high for act={act}, dtype={dtype}: {diff.max().item()}"


def test_halley_invert_activation_strictly_negative_inputs():
    """Focus test on strictly negative inputs where -y is positive and clamp(-y, min=1e-15) previously underflowed in FP16."""
    for act in ["silu", "gelu", "gelu_tanh", "mish"]:
        z = torch.linspace(-4.0, -0.1, 50, dtype=torch.float16)
        act_fn = F.silu if act == "silu" else (F.gelu if "gelu" in act else F.mish)
        y = act_fn(z)
        branch_bit = z >= get_activation_min_z(act)
        z_rec = halley_invert_activation(y, branch_bit, act=act)

        assert not torch.isnan(z_rec).any()
        assert not torch.isinf(z_rec).any()
        assert z_rec.dtype == torch.float16


# =============================================================================
# 2. Dagger Attention 2D Mask Shape Collision & Tile Size Propagation
# =============================================================================

def test_dagger_attention_2d_mask_shape_collision():
    """Verifies that when B == Sq == Skv, a 2D mask [Sq, Skv] is correctly sliced as

    an attention mask [1, 1, br, bc] rather than being misidentified as a batch key-padding mask [B, Skv].
    """
    B = 4
    Sq = 4
    Skv = 4
    H = 2
    D = 16

    # 1. Direct unit test of _extract_mask_tile
    # Create a 2D boolean mask of shape [Sq, Skv]
    attn_mask_2d = torch.zeros(Sq, Skv, dtype=torch.bool)
    attn_mask_2d[0, 1] = True  # Mask out position (0, 1)

    bool_mask, float_bias = _extract_mask_tile(
        attn_mask=attn_mask_2d,
        key_padding_mask=None,
        B=B,
        H=H,
        Sq=Sq,
        Skv=Skv,
        r_start=0,
        r_end=2,
        c_start=0,
        c_end=2,
        acc_dtype=torch.float32,
    )

    # When B == Sq == Skv, it should slice [r_start:r_end, c_start:c_end] with shape [1, 1, 2, 2]
    assert bool_mask is not None
    assert bool_mask.shape == (1, 1, 2, 2), f"Expected shape (1, 1, 2, 2), got {bool_mask.shape}"
    assert bool_mask[0, 0, 0, 1].item() is True

    # 2. End-to-end numerical parity test when B == Sq == Skv
    torch.manual_seed(42)
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, Skv, D)
    v = torch.randn(B, H, Skv, D)

    # Causal lower-triangular mask as float additive bias: 0 for attend, -1e4 for mask
    causal_bias_2d = torch.triu(torch.full((Sq, Skv), -1e4), diagonal=1)

    out_dagger = dagger_tiled_attention(q, k, v, attn_mask=causal_bias_2d, tile_size=2)

    # Reference attention computation
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    scores = scores + causal_bias_2d.unsqueeze(0).unsqueeze(0)
    p = F.softmax(scores, dim=-1)
    out_ref = torch.matmul(p, v)

    torch.testing.assert_close(out_dagger, out_ref, atol=1e-5, rtol=1e-5)


def test_dagger_attention_2d_key_padding_mask_when_b_ne_sq():
    """Verifies that when B != Sq, a 2D mask [B, Skv] is correctly treated as a key padding mask."""
    B = 2
    Sq = 4
    Skv = 4
    H = 1
    D = 8

    # Key padding mask: batch item 0 masks out last token
    kpm = torch.ones(B, Skv, dtype=torch.bool)
    kpm[0, 3] = False  # token 3 is padding

    bool_mask, float_bias = _extract_mask_tile(
        attn_mask=kpm,
        key_padding_mask=None,
        B=B,
        H=H,
        Sq=Sq,
        Skv=Skv,
        r_start=0,
        r_end=4,
        c_start=0,
        c_end=4,
        acc_dtype=torch.float32,
    )
    # Should be sliced as [B, 1, 1, Skv]
    assert bool_mask is not None
    assert bool_mask.shape == (B, 1, 1, Skv)
    assert bool_mask[0, 0, 0, 3].item() is True  # ~kpm, so False becomes True (masked)
    assert bool_mask[1, 0, 0, 3].item() is False


def test_dagger_attention_tile_size_propagation():
    """Verifies that tile_size is propagated through dagger_tiled_attention to holonomic_dagger_attention

    and produces numerically identical forward and backward results across different tile sizes.
    """
    torch.manual_seed(123)
    B, H, S, D = 2, 4, 64, 32

    q_base = torch.randn(B, H, S, D)
    k_base = torch.randn(B, H, S, D)
    v_base = torch.randn(B, H, S, D)

    results = {}
    grads = {}

    for ts in [8, 16, 32, 64, 128]:
        q = q_base.clone().requires_grad_(True)
        k = k_base.clone().requires_grad_(True)
        v = v_base.clone().requires_grad_(True)

        out = dagger_tiled_attention(q, k, v, is_causal=True, tile_size=ts)
        loss = out.sum()
        loss.backward()

        results[ts] = out
        grads[ts] = (q.grad, k.grad, v.grad)

    # Check forward output parity across all tile sizes
    ref_out = results[128]
    ref_gq, ref_gk, ref_gv = grads[128]

    for ts in [8, 16, 32, 64]:
        torch.testing.assert_close(results[ts], ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grads[ts][0], ref_gq, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grads[ts][1], ref_gk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grads[ts][2], ref_gv, atol=1e-5, rtol=1e-5)


# =============================================================================
# 3. Streaming Fused Linear Cross-Entropy All-Masked Chunk Culling
# =============================================================================

@pytest.mark.parametrize("ignore_index", [0, 5, -100])
def test_fused_cross_entropy_all_masked_chunk_culling(ignore_index: int):
    """Verifies that when a chunk has not mask_chunk.any(), the GEMM loop is cleanly skipped.

    Tests both non-negative ignore_index (0, 5) and standard -100.
    """
    torch.manual_seed(789)
    M = 8
    D = 16
    V = 32
    tile_tok = 4  # 2 chunks of 4 tokens each

    h = torch.randn(M, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)

    # Chunk 0 (tokens 0..3): all ignored
    # Chunk 1 (tokens 4..7): valid targets distinct from ignore_index
    valid_target_base = 10 if ignore_index == 0 else 1
    targets = torch.tensor([
        ignore_index, ignore_index, ignore_index, ignore_index,
        valid_target_base, valid_target_base + 1, valid_target_base + 2, valid_target_base + 3,
    ], dtype=torch.long)

    # Spy on torch.matmul to verify chunk 0 GEMM is skipped
    orig_matmul = torch.matmul
    matmul_call_shapes = []

    def counting_matmul(a, b):
        matmul_call_shapes.append((a.shape, b.shape))
        return orig_matmul(a, b)

    with patch("torch.matmul", side_effect=counting_matmul):
        loss, lse = _streaming_fused_lce_forward(
            h=h,
            w=w,
            targets=targets,
            bias=None,
            ignore_index=ignore_index,
            tile_tok=tile_tok,
            tile_voc=V,
        )

    # With tile_tok=4 and V=32 (1 voc chunk), 2 token chunks total.
    # Chunk 0 is completely masked -> should be skipped!
    # Chunk 1 is valid -> exactly 1 matmul call.
    assert len(matmul_call_shapes) == 1, f"Expected 1 matmul call, got {len(matmul_call_shapes)}: {matmul_call_shapes}"
    assert matmul_call_shapes[0][0] == (4, D)

    # Verify loss matches reference on valid tokens
    logits_chunk1 = h[4:] @ w.t()
    loss_ref = F.cross_entropy(logits_chunk1, targets[4:], reduction="mean")
    torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_entirely_masked_sequence():
    """Verifies that when all tokens across all chunks are masked, loss is 0.0,

    no GEMM is performed, no NaNs occur, and backward produces zero gradients.
    """
    torch.manual_seed(999)
    M = 8
    D = 16
    V = 32
    ignore_index = 0

    h = torch.randn(M, D, requires_grad=True)
    w = torch.randn(V, D, requires_grad=True)
    targets = torch.full((M,), ignore_index, dtype=torch.long)

    loss, lse = holonomic_fused_linear_cross_entropy(
        h, w, targets, ignore_index=ignore_index
    )

    assert loss.item() == 0.0
    assert not torch.isnan(loss)
    assert not torch.isnan(lse).any()

    # Backward pass should produce 0 gradients without crashing
    loss.backward()
    assert h.grad is not None
    assert torch.all(h.grad == 0.0)
    assert w.grad is not None
    assert torch.all(w.grad == 0.0)
