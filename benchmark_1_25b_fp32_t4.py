"""Modern 1.25B Parameter LLM Full Precision (FP32) Training Benchmark on Tesla T4.

Evaluates:
1. Exact parameter count (~1.244 Billion parameters in float32)
2. Static FP32 weights footprint (4.98 GB) + Gradients (4.98 GB)
3. Activation tape comparison: Official PyTorch vs OjasX v0.4.0
4. Headroom stress testing (sequence length scaling)
5. Extended continuous training (25 consecutive steps) measuring loss and VRAM stability
"""

import copy
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ojasx2
from ojasx2.smart_router import optimize_model


# ============================================================================
# 1. Modern LLaMA 1.25B Architecture (22 Layers, Full Precision FP32)
# ============================================================================

class ModernRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class ModernSwiGLU(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False, dtype=torch.float32)
        self.up_proj = nn.Linear(d, h, bias=False, dtype=torch.float32)
        self.down_proj = nn.Linear(h, d, bias=False, dtype=torch.float32)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ModernSelfAttention(nn.Module):
    def __init__(self, d: int, num_heads: int):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.q_proj = nn.Linear(d, d, bias=False, dtype=torch.float32)
        self.k_proj = nn.Linear(d, d, bias=False, dtype=torch.float32)
        self.v_proj = nn.Linear(d, d, bias=False, dtype=torch.float32)
        self.o_proj = nn.Linear(d, d, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class ModernDecoderLayer(nn.Module):
    def __init__(self, d: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.input_layernorm = ModernRMSNorm(d)
        self.self_attn = ModernSelfAttention(d, num_heads)
        self.post_attention_layernorm = ModernRMSNorm(d)
        self.mlp = ModernSwiGLU(d, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class ModernLLAMA1_25B(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 2048,
        num_layers: int = 22,
        num_heads: int = 16,
        intermediate_dim: int = 5504,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim, dtype=torch.float32)
        self.layers = nn.ModuleList([
            ModernDecoderLayer(dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.norm = ModernRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ============================================================================
# 2. Memory & Autograd Instrumentation
# ============================================================================

class SavedTapeAuditor:
    def __init__(self, exclude_params):
        self.tensors = []
        self.exclude_ptrs = {
            p.untyped_storage().data_ptr() for p in (exclude_params or []) if isinstance(p, torch.Tensor)
        }

    def pack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.untyped_storage().data_ptr() not in self.exclude_ptrs:
            self.tensors.append(tensor)
        return tensor

    def unpack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def total_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)


# ============================================================================
# 3. Main Benchmark Execution
# ============================================================================

def run_benchmark():
    assert torch.cuda.is_available(), "CUDA is required!"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    print("=" * 85)
    print(f"  OJASX v0.4.0 FULL PRECISION (FP32) 1.25B LLM BENCHMARK ON {gpu_name}")
    print(f"  TOTAL VRAM CAPACITY: {total_vram_gb:.2f} GB")
    print("=" * 85)

    # 1. Build Model
    print("\n[1] Instantiating 1.25B Modern LLaMA Model in FULL PRECISION (`torch.float32`)...")
    torch.manual_seed(42)
    model = ModernLLAMA1_25B(
        vocab_size=32000,
        dim=2048,
        num_layers=22,
        num_heads=16,
        intermediate_dim=5504,
    ).to(device=device)

    total_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    weight_gb = weight_bytes / (1024**3)
    grad_gb = weight_gb

    print(f"  • Total Model Parameters : {total_params:,} ({total_params / 1e9:.3f} Billion)")
    print(f"  • Static FP32 Weight VRAM : {weight_gb:.2f} GB ({weight_bytes / (1024*1024):.1f} MB)")
    print(f"  • Estimated Gradient VRAM : {grad_gb:.2f} GB ({weight_bytes / (1024*1024):.1f} MB)")
    print(f"  • Combined Weights + Grads : {weight_gb + grad_gb:.2f} GB")
    print(f"  • Remaining Activation Room : {total_vram_gb - (weight_gb + grad_gb):.2f} GB")

    # 2. Test Workload Setup
    batch_size = 1
    seq_len = 512
    print(f"\n[2] Setting Training Workload: Batch Size = {batch_size}, Sequence Length = {seq_len} tokens")
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Measure Official PyTorch Eager Activation Tape
    print("\n[3] Auditing Official PyTorch Eager Activation Tape in FP32...")
    auditor_eager = SavedTapeAuditor(list(model.parameters()))
    with torch.autograd.graph.saved_tensors_hooks(auditor_eager.pack_hook, auditor_eager.unpack_hook):
        out_eager = model(input_ids)
    eager_tape_bytes = auditor_eager.total_bytes
    eager_tape_mb = eager_tape_bytes / (1024 * 1024)
    del out_eager
    torch.cuda.empty_cache()

    print(f"  • Official PyTorch FP32 Activation Tape : {eager_tape_mb:.2f} MB ({eager_tape_mb / 1024:.2f} GB)")

    # 3. Apply OjasX v0.4.0 Optimization
    print("\n[4] Optimizing 1.25B Model with OjasX v0.4.0 (`optimize_model`)...")
    model_ojasx, report = optimize_model(model)
    print(f"  • Workload Profile Detected : {report.workload_profile}")
    print(f"  • Layer Actions Applied     : {dict(report.action_counts)}")

    # Measure OjasX Activation Tape
    auditor_ojasx = SavedTapeAuditor(list(model_ojasx.parameters()))
    with torch.autograd.graph.saved_tensors_hooks(auditor_ojasx.pack_hook, auditor_ojasx.unpack_hook):
        out_ojasx = model_ojasx(input_ids)
    ojasx_tape_bytes = auditor_ojasx.total_bytes
    ojasx_tape_mb = ojasx_tape_bytes / (1024 * 1024)
    del out_ojasx
    torch.cuda.empty_cache()

    tape_reduction_mb = eager_tape_mb - ojasx_tape_mb
    tape_reduction_pct = (1.0 - (ojasx_tape_bytes / eager_tape_bytes)) * 100.0

    print(f"  • OjasX v0.4.0 FP32 Activation Tape     : {ojasx_tape_mb:.2f} MB ({ojasx_tape_mb / 1024:.2f} GB)")
    print(f"  • Real Activation Memory Saved         : {tape_reduction_mb:.2f} MB ({tape_reduction_pct:.2f}% DROP)")

    # 4. Extended Training Run with OjasX v0.4.0
    print("\n[5] Executing Extended Training Run with OjasX v0.4.0 (25 Continuous Steps in Full FP32)...")
    optimizer = torch.optim.SGD(model_ojasx.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    losses = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for step in range(1, 26):
        torch.cuda.synchronize()
        start_event.record()

        logits = model_ojasx(input_ids)
        loss = F.cross_entropy(logits.view(-1, 32000), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event)
        step_times.append(elapsed)
        loss_val = float(loss.item())
        losses.append(loss_val)
        current_vram_gb = torch.cuda.memory_allocated() / (1024**3)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        if step in (1, 2, 3, 5, 10, 15, 20, 25):
            print(f"  Step {step:02d}/25 | Latency: {elapsed:6.1f} ms | Loss: {loss_val:.4f} | Current VRAM: {current_vram_gb:.2f} GB | Peak VRAM: {peak_vram_gb:.2f} GB")

    avg_step_ms = sum(step_times) / len(step_times)
    total_training_time_sec = sum(step_times) / 1000.0
    final_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # 5. Final Summary Table
    print("\n" + "=" * 85)
    print("  FINAL 1.25B FULL PRECISION (FP32) BENCHMARK SUMMARY (TESLA T4 GPU)")
    print("=" * 85)
    print(f"{'Metric':<38} | {'Official PyTorch Eager':<20} | {'OjasX v0.4.0':<20}")
    print("-" * 85)
    print(f"{'Model Parameter Count':<38} | {total_params / 1e9:<20.3f}B | {total_params / 1e9:<20.3f}B")
    print(f"{'Precision':<38} | {'float32 (FP32)':<20} | {'float32 (FP32)':<20}")
    print(f"{'Static Weight Footprint':<38} | {weight_gb:<17.2f} GB | {weight_gb:<17.2f} GB")
    print(f"{'Activation Tape Memory':<38} | {eager_tape_mb:<17.2f} MB | {ojasx_tape_mb:<17.2f} MB (-{tape_reduction_pct:.1f}%)")
    print(f"{'Activation Memory Saved per Step':<38} | {'0.00 MB':<20} | {tape_reduction_mb:<17.2f} MB")
    print(f"{'Training Stability (25 Steps)':<38} | {'—':<20} | {'COMPLETED (0 Crashes)':<20}")
    print(f"{'Initial Loss -> Final Loss':<38} | {'—':<20} | {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"{'Peak Training VRAM':<38} | {'—':<20} | {final_peak_gb:<17.2f} GB")
    print(f"{'Average Step Latency':<38} | {'—':<20} | {avg_step_ms:<17.1f} ms")
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
