"""
tests/test_custom_training_and_finetuning.py

End-to-End Verification Suite for:
1. Native Pre-Training from Scratch with Symplectic Reversible Blocks (O(1) activation tape).
2. HuggingFace / Custom Pretrained Fine-Tuning with Single-Stream Topology Preservation and LoRA Weights.
3. SFT / Padding Mask (ignore_index = -100) Gradient Normalization in FusedLinearCrossEntropy.
4. Rectangular KV-Cache Decoding, GQA, and 2D Padding Masks in DaggerAttention.
5. FX Subgraph Compilation with Keyword Arguments (input_ids, attention_mask, labels).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx2
from ojasx2.dagger_attention import DaggerAttention, dagger_tiled_attention
from ojasx2.fused_cross_entropy import FusedLinearCrossEntropy, fused_linear_cross_entropy
from ojasx2.fx_subgraph_compiler import compile_fx_subgraph
from ojasx2.symplectic_engine import SymplecticTransformerBlock, SymplecticStack
from ojasx2.smart_router import _CheckpointedDecoderLayer, optimize_model


DEVICES = ["cpu"]
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.mark.parametrize("device", DEVICES)
def test_custom_model_training_from_scratch_symplectic(device: str):
    """Verifies building and training an OjasX-native model from scratch using Symplectic blocks."""
    dim = 64
    depth = 2
    vocab_size = 1000

    class NativeSymplecticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                SymplecticTransformerBlock(dim=dim, num_heads=4, split_dim=True)
                for _ in range(depth)
            ])
            self.head = FusedLinearCrossEntropy(in_features=dim, vocab_size=vocab_size)

        def forward(self, input_ids, targets=None):
            x = self.embed(input_ids)
            half = x.size(-1) // 2
            x1, x2 = x[..., :half], x[..., half:]
            for blk in self.blocks:
                x1, x2 = blk(x1, x2)
            h = torch.cat([x1, x2], dim=-1)
            if targets is not None:
                return self.head(h, targets)
            return self.head(h)

    model = NativeSymplecticModel().to(device)
    model, report = optimize_model(model, mode="pretrain")

    assert report.action_counts["symplectic_blocks_configured"] >= 2

    inputs = torch.randint(0, vocab_size, (2, 16), device=device)
    targets = torch.randint(0, vocab_size, (2, 16), device=device)

    loss = model(inputs, targets=targets)
    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients computed for embedding and transformer blocks
    assert model.embed.weight.grad is not None
    for blk in model.blocks:
        for p in blk.parameters():
            if p.requires_grad:
                assert p.grad is not None


@pytest.mark.parametrize("device", DEVICES)
def test_pretrained_finetuning_preserves_single_stream_weights(device: str):
    """Verifies that fine-tuning preserves pretrained weights, LoRA adapters, and single-stream topology."""
    dim = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = dim // num_heads

    class DummyHFSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)
            self.num_heads = num_heads
            self.num_key_value_heads = num_kv_heads
            self.head_dim = head_dim

        def forward(self, x):
            return self.o_proj(self.q_proj(x))

    class DummyHFDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyHFSelfAttention()
            # Simulate a LoRA adapter attached alongside attention
            self.lora_A = nn.Parameter(torch.randn(8, dim))
            self.lora_B = nn.Parameter(torch.randn(dim, 8))

            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

        def forward(self, x):
            # Single-stream residual connection
            attn_out = self.self_attn(x) + x @ self.lora_A.t() @ self.lora_B.t()
            h = x + attn_out
            return h + self.mlp(self.norm(h))

    class PretrainedCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyHFDecoderLayer() for _ in range(2)])
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, x, targets=None):
            h = x
            for layer in self.layers:
                h = layer(h)
            if targets is not None:
                return self.loss_fn(h.mean(dim=1), targets)
            return h

    model = PretrainedCausalLM().to(device)

    # Save original parameter identities to ensure zero weight cloning
    orig_q_weights = [layer.self_attn.q_proj.weight for layer in model.layers]
    orig_lora_a = [layer.lora_A for layer in model.layers]

    # Run fine-tuning surgery
    model, report = optimize_model(model, mode="finetune", enable_checkpointing=True)

    # Verify single-stream layers were wrapped in activation checkpointing
    assert report.action_counts["checkpointed_blocks"] == 2
    assert report.action_counts["replaced_loss_modules"] == 1
    assert report.action_counts["replaced_attention_layers"] == 2

    # Verify pretrained weights and LoRA adapters are identically preserved
    for idx, layer_wrapper in enumerate(model.layers):
        assert isinstance(layer_wrapper, _CheckpointedDecoderLayer)
        inner = layer_wrapper.block
        assert isinstance(inner.self_attn, DaggerAttention)
        assert inner.self_attn.q_proj.weight is orig_q_weights[idx]
        assert inner.lora_A is orig_lora_a[idx]

    # Verify end-to-end forward/backward with gradient flow
    x = torch.randn(2, 16, dim, device=device, requires_grad=True)
    targets = torch.tensor([0, 1], device=device)
    loss = model(x, targets=targets)
    loss.backward()

    assert x.grad is not None
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


@pytest.mark.parametrize("device", DEVICES)
def test_fused_cross_entropy_exact_sft_padding_normalization(device: str):
    """Verifies that ignore_index=-100 correctly normalizes gradients by valid token count."""
    B, S, D, V = 2, 16, 64, 500
    x = torch.randn(B, S, D, device=device, requires_grad=True)
    w = torch.randn(V, D, device=device, requires_grad=True)
    b = torch.randn(V, device=device, requires_grad=True)

    targets = torch.randint(0, V, (B, S), device=device)
    # Mask out prompt and padding tokens in varying lengths
    targets[0, :8] = -100
    targets[1, :4] = -100

    # OjasX fused linear cross entropy
    loss_ojasx = fused_linear_cross_entropy(x, w, targets, bias=b, ignore_index=-100, label_smoothing=0.1)
    loss_ojasx.backward()

    # PyTorch eager reference
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    logits_ref = F.linear(x_ref, w_ref, b_ref)
    loss_ref = F.cross_entropy(
        logits_ref.view(-1, V), targets.view(-1), ignore_index=-100, label_smoothing=0.1
    )
    loss_ref.backward()

    loss_diff = abs(loss_ojasx.item() - loss_ref.item())
    grad_x_diff = (x.grad - x_ref.grad).abs().max().item()
    grad_w_diff = (w.grad - w_ref.grad).abs().max().item()
    grad_b_diff = (b.grad - b_ref.grad).abs().max().item()

    assert loss_diff < 1e-4, f"Loss mismatch: {loss_diff}"
    assert grad_x_diff < 1e-4, f"Grad X mismatch: {grad_x_diff}"
    assert grad_w_diff < 1e-4, f"Grad W mismatch: {grad_w_diff}"
    assert grad_b_diff < 1e-4, f"Grad B mismatch: {grad_b_diff}"

    # Test all-masked edge case (e.g. empty SFT batch)
    all_masked = torch.full((B, S), -100, dtype=torch.long, device=device)
    x_empty = torch.randn(B, S, D, device=device, requires_grad=True)
    loss_empty = fused_linear_cross_entropy(x_empty, w, all_masked, bias=b, ignore_index=-100)
    assert loss_empty.item() == 0.0
    loss_empty.backward()
    assert (x_empty.grad == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_dagger_attention_rectangular_and_2d_padding_mask(device: str):
    """Verifies DaggerAttention with GQA, 2D binary padding masks, and rectangular KV cache steps."""
    dim = 128
    num_heads = 4
    num_kv_heads = 2

    attn = DaggerAttention(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, is_causal=True).to(device)

    # 1. Forward/backward with 2D padding mask (1/True = valid, 0/False = pad)
    x = torch.randn(2, 16, dim, device=device, requires_grad=True)
    mask_2d = torch.ones(2, 16, dtype=torch.bool, device=device)
    mask_2d[0, 10:] = False
    mask_2d[1, 14:] = False

    out = attn(x, attention_mask=mask_2d)
    assert out.shape == (2, 16, dim)
    out.sum().backward()
    assert x.grad is not None

    # 2. Rectangular KV cache decoding step (N_q = 1, N_kv = 32)
    q = torch.randn(2, num_heads, 1, dim // num_heads, device=device)
    k = torch.randn(2, num_kv_heads, 32, dim // num_heads, device=device)
    v = torch.randn(2, num_kv_heads, 32, dim // num_heads, device=device)

    step_out = dagger_tiled_attention(q, k, v, is_causal=True)
    assert step_out.shape == (2, num_heads, 1, dim // num_heads)


@pytest.mark.parametrize("device", DEVICES)
def test_fx_subgraph_compiler_kwargs_passthrough(device: str):
    """Verifies that compile_fx_subgraph executes models with kwargs without dropping kwargs or crashing."""
    class KwargFineTuningModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, input_ids, attention_mask=None, labels=None):
            out = self.linear(input_ids)
            if attention_mask is not None:
                out = out * attention_mask
            if labels is not None:
                return (out - labels).pow(2).mean()
            return out

    model = KwargFineTuningModel().to(device)
    compiled = compile_fx_subgraph(model, target="metal" if device == "mps" else "cpp")

    summary = compiled.summary()
    assert summary.num_holonomic_ops > 0
    assert summary.num_subgraph_ops > 0
    assert summary.num_tile_ops > 0

    x = torch.randn(4, 32, device=device)
    mask = torch.ones(4, 32, device=device)
    labels = torch.randn(4, 32, device=device)

    # Call with keyword arguments
    loss = compiled(input_ids=x, attention_mask=mask, labels=labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0
