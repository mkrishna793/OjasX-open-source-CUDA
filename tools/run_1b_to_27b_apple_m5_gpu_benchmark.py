#!/usr/bin/env python3
"""Genuine 1B to 27B Full-Model Multi-Layer Training Benchmark on Apple Silicon GPU (MPS).

Zero extrapolation, zero layer reuse:
- Every single Transformer block (0 .. L-1) owns distinct physical packed_weight,
  block_scales, RMSNorm, and trainable LoRA (lora_A, lora_B) tensors on MPS GPU.
- Verifies unique data_ptr() across all 7*L E8LatticeLinear weight matrices.
- Executes 3 full forward + backward + HolonomicGradFreeOptimizer steps per model scale,
  recording hardware-reported torch.mps.current_allocated_memory(), driver_allocated_memory(),
  loss trajectory (Step 1 -> Step 2 -> Step 3), gradient norms, and weight evolution ||W_t - W_0||.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ojasx2
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.math.lattice import E8LatticeLinear


def init_valid_ternary_e8(layer: E8LatticeLinear) -> None:
    """Initialize packed_weight with valid 12-bit Ternary E8 rank codes in [0, 3281)."""
    triplets = layer.packed_weight.view(-1, 3)
    n = triplets.shape[0]
    device = layer.packed_weight.device
    # Construct in chunks if very large to keep temporary RNG buffer tiny
    chunk = 4_194_304
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        m = end - start
        c0 = torch.randint(0, 3281, (m,), device=device, dtype=torch.int32)
        c1 = torch.randint(0, 3281, (m,), device=device, dtype=torch.int32)
        triplets[start:end, 0] = (c0 & 0xFF).to(torch.uint8)
        triplets[start:end, 1] = (((c0 >> 8) & 0x0F) | ((c1 & 0x0F) << 4)).to(torch.uint8)
        triplets[start:end, 2] = ((c1 >> 4) & 0xFF).to(torch.uint8)
    layer.block_scales.fill_(0.45 / math.sqrt(layer.in_features))


class FullE8TransformerBlock(nn.Module):
    """One physically distinct Transformer decoder block with 7 E8LatticeLinear projections."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        *,
        bitrate: float = 1.5,
        group_size: int = 64,
        lora_rank: int = 8,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.q_proj = E8LatticeLinear(
            d_model, d_model, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.k_proj = E8LatticeLinear(
            d_model, d_model, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.v_proj = E8LatticeLinear(
            d_model, d_model, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.o_proj = E8LatticeLinear(
            d_model, d_model, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.norm2 = nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.gate_proj = E8LatticeLinear(
            d_model, d_ffn, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.up_proj = E8LatticeLinear(
            d_model, d_ffn, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        self.down_proj = E8LatticeLinear(
            d_ffn, d_model, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        for proj in (
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.gate_proj, self.up_proj, self.down_proj,
        ):
            init_valid_ternary_e8(proj)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o_proj(attn)
        h2 = self.norm2(x)
        mlp = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x + mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward_impl, x, use_reentrant=False)


class FullE8CausalLM(nn.Module):
    """Complete multi-layer Causal Language Model with L distinct E8 Transformer blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ffn: int,
        n_layers: int,
        n_heads: int,
        *,
        bitrate: float = 1.5,
        group_size: int = 64,
        lora_rank: int = 8,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList([
            FullE8TransformerBlock(
                d_model, d_ffn, n_heads,
                bitrate=bitrate, group_size=group_size,
                lora_rank=lora_rank, device=device, dtype=dtype,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = E8LatticeLinear(
            d_model, vocab_size, bias=False, rank=lora_rank,
            group_size=group_size, bitrate=bitrate, device=device, dtype=dtype,
        )
        init_valid_ternary_e8(self.lm_head)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


MODEL_LADDER: List[Tuple[str, int, int, int, int, int]] = [
    # (label, n_layers, d_model, d_ffn, n_heads, vocab_size)
    ("1B",  20, 2048,  5632, 16, 16384),
    ("4B",  34, 3072,  8192, 24, 16384),
    ("8B",  32, 4096, 14336, 32, 16384),
    ("14B", 44, 5120, 13824, 40, 16384),
    ("27B", 60, 6144, 16384, 48, 16384),
]


def count_logical_and_physical(model: FullE8CausalLM) -> Dict[str, int]:
    e8_logical_params = 0
    e8_packed_bytes = 0
    e8_scale_bytes = 0
    unique_weight_ptrs = set()
    e8_layer_count = 0

    for m in model.modules():
        if isinstance(m, E8LatticeLinear):
            e8_layer_count += 1
            e8_logical_params += m.in_features * m.out_features
            e8_packed_bytes += m.packed_weight.nbytes
            e8_scale_bytes += m.block_scales.nbytes
            unique_weight_ptrs.add(m.packed_weight.data_ptr())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_bytes = sum(p.nbytes for p in model.parameters() if p.requires_grad)
    frozen_float_params = model.embed.weight.numel()
    frozen_float_bytes = model.embed.weight.nbytes

    total_logical_params = e8_logical_params + trainable_params + frozen_float_params
    total_persistent_bytes = e8_packed_bytes + e8_scale_bytes + trainable_bytes + frozen_float_bytes

    assert len(unique_weight_ptrs) == e8_layer_count, (
        f"Expected {e8_layer_count} unique weight buffers, got {len(unique_weight_ptrs)}"
    )

    return {
        "total_logical_parameters": total_logical_params,
        "e8_backbone_parameters": e8_logical_params,
        "trainable_lora_and_norm_parameters": trainable_params,
        "embedding_parameters": frozen_float_params,
        "e8_linear_layer_count": e8_layer_count,
        "unique_packed_weight_data_ptrs": len(unique_weight_ptrs),
        "e8_packed_bytes": e8_packed_bytes,
        "e8_scale_bytes": e8_scale_bytes,
        "trainable_bytes": trainable_bytes,
        "embedding_bytes": frozen_float_bytes,
        "total_persistent_bytes": total_persistent_bytes,
        "fp16_equivalent_bytes": total_logical_params * 2,
        "fp32_equivalent_bytes": total_logical_params * 4,
    }


def run_single_scale(
    label: str,
    n_layers: int,
    d_model: int,
    d_ffn: int,
    n_heads: int,
    vocab_size: int,
    *,
    steps: int = 3,
    seq_len: int = 16,
    device: str = "mps",
) -> Dict[str, object]:
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    baseline_alloc = torch.mps.current_allocated_memory() if device == "mps" else 0
    t_init_start = time.perf_counter()

    model = FullE8CausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ffn=d_ffn,
        n_layers=n_layers,
        n_heads=n_heads,
        bitrate=1.5,
        group_size=64,
        lora_rank=8,
        device=device,
        dtype=torch.float16,
    )
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    init_seconds = time.perf_counter() - t_init_start

    ledger = count_logical_and_physical(model)
    model_alloc_bytes = (
        torch.mps.current_allocated_memory() - baseline_alloc if device == "mps" else ledger["total_persistent_bytes"]
    )
    model_driver_bytes = torch.mps.driver_allocated_memory() if device == "mps" else model_alloc_bytes

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    initial_snapshot = [p.detach().clone() for p in trainable_params[:16]]

    optimizer = HolonomicGradFreeOptimizer(
        trainable_params,
        lr=5e-4,
        algorithm="adamw",
        weight_decay=0.0,
        quantize_moments_e8=False,
    )

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    step_records = []
    peak_alloc_bytes = model_alloc_bytes
    peak_driver_bytes = model_driver_bytes

    for step_idx in range(1, steps + 1):
        t_step = time.perf_counter()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size).float(), target_ids.view(-1))
        loss.backward()

        grad_sq = sum(
            float(p.grad.float().norm().item() ** 2)
            for p in trainable_params
            if p.grad is not None
        )
        grad_norm = math.sqrt(grad_sq) if grad_sq > 0 else float(getattr(optimizer, "last_grad_norm", 0.0))

        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
            cur_alloc = torch.mps.current_allocated_memory()
            cur_driver = torch.mps.driver_allocated_memory()
            peak_alloc_bytes = max(peak_alloc_bytes, cur_alloc)
            peak_driver_bytes = max(peak_driver_bytes, cur_driver)

        step_time = time.perf_counter() - t_step
        step_records.append({
            "step": step_idx,
            "loss": round(float(loss.item()), 6),
            "grad_norm": round(grad_norm, 6),
            "step_seconds": round(step_time, 3),
        })

    weight_delta_sq = sum(
        float((p.detach() - p0).float().norm().item() ** 2)
        for p, p0 in zip(trainable_params[:16], initial_snapshot)
    )
    weight_delta = math.sqrt(weight_delta_sq)

    result = {
        "scale_label": label,
        "architecture": {
            "n_layers": n_layers,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "n_heads": n_heads,
            "vocab_size": vocab_size,
            "lora_rank": 8,
            "bitrate": 1.5,
            "group_size": 64,
        },
        "parameters": ledger,
        "memory_gb": {
            "fp32_uncompressed_gb": round(ledger["fp32_equivalent_bytes"] / (1024 ** 3), 2),
            "fp16_uncompressed_gb": round(ledger["fp16_equivalent_bytes"] / (1024 ** 3), 2),
            "ojasx_persistent_weights_gb": round(ledger["total_persistent_bytes"] / (1024 ** 3), 2),
            "mps_model_allocated_gb": round(model_alloc_bytes / (1024 ** 3), 2),
            "mps_peak_training_allocated_gb": round(peak_alloc_bytes / (1024 ** 3), 2),
            "mps_peak_driver_gb": round(peak_driver_bytes / (1024 ** 3), 2),
            "compression_ratio_vs_fp16": round(ledger["fp16_equivalent_bytes"] / max(1, ledger["total_persistent_bytes"]), 2),
        },
        "training": {
            "init_seconds": round(init_seconds, 2),
            "steps": step_records,
            "initial_loss": step_records[0]["loss"],
            "final_loss": step_records[-1]["loss"],
            "loss_decreased": step_records[-1]["loss"] < step_records[0]["loss"],
            "weight_evolution_l2": round(weight_delta, 6),
        },
    }

    del logits, loss, optimizer, trainable_params, initial_snapshot, model
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/apple_m5_1b_to_27b_gpu_benchmark.json")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    total_ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())

    print("=" * 88)
    print(f"OJASX GENUINE 1B -> 27B FULL-MODEL TRAINING BENCHMARK ON {cpu_brand} GPU (MPS)")
    print(f"System Unified Memory: {total_ram_bytes / (1024**3):.2f} GB | PyTorch: {torch.__version__} | Device: mps")
    print("=" * 88)

    results = []
    for label, n_layers, d_model, d_ffn, n_heads, vocab_size in MODEL_LADDER:
        print(f"\n>>> Instantiating & Training Full {label} Model ({n_layers} layers, D={d_model}, FFN={d_ffn})...")
        res = run_single_scale(
            label, n_layers, d_model, d_ffn, n_heads, vocab_size,
            steps=args.steps, device="mps",
        )
        results.append(res)
        p = res["parameters"]
        m = res["memory_gb"]
        tr = res["training"]
        print(
            f"  [VERIFIED {label}] Params: {p['total_logical_parameters']:,} "
            f"({p['unique_packed_weight_data_ptrs']} unique E8 weight tensors)"
        )
        print(
            f"  [MEMORY] FP16 Eq: {m['fp16_uncompressed_gb']:.2f} GB -> "
            f"OjasX 1.5b Model RAM: {m['mps_model_allocated_gb']:.2f} GB | "
            f"Peak Training GPU RAM: {m['mps_peak_training_allocated_gb']:.2f} GB "
            f"(Driver: {m['mps_peak_driver_gb']:.2f} GB)"
        )
        print(
            f"  [TRAINING] Loss: {tr['initial_loss']:.4f} -> {tr['final_loss']:.4f} "
            f"(Decreased={tr['loss_decreased']}) | ||dW||={tr['weight_evolution_l2']:.5f}"
        )

    payload = {
        "benchmark": "OjasX Genuine 1B to 27B Full-Model Multi-Layer GPU Training Benchmark",
        "zero_extrapolation": True,
        "zero_layer_reuse": True,
        "hardware": {
            "chip": cpu_brand,
            "unified_memory_gb": round(total_ram_bytes / (1024 ** 3), 2),
            "device": "mps (Apple Metal GPU)",
            "pytorch_version": torch.__version__,
            "ojasx_version": ojasx2.__version__,
            "platform": platform.platform(),
        },
        "models": results,
    }

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark telemetry to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
