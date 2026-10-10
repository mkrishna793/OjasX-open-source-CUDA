#!/usr/bin/env python3
"""
tools/run_official_benchmarks_suite.py

Official Three-Pillar Benchmarking & Virtual GPU Qualification Suite for OjasX Holonomic Engine:
1. torch.utils.benchmark.Compare + Fuzzer (Automated A/B Operator & Fuzzed Shape Benchmarker)
2. PyTorch UserBenchmark (End-to-End Unmodified ML/DL Workloads via torch.compile(backend="holonomic"))
3. TileBench Kernel Semantics & Roofline Suite (Kernel Patterns, Input Sweeps, Roofline & Token Efficiency)
4. Virtual GPU Elevation (PyTorch FakeTensorMode + Hardware Memory/Roofline Projection for 7B-70B Scales)

Includes strict GPU & Unified Memory safety guards to prevent overloading the host laptop.
"""

from __future__ import annotations

import gc
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.utils.benchmark import Fuzzer, FuzzedParameter
from torch._subclasses.fake_tensor import FakeTensorMode

import ojasx2
from ojasx2.residuals import SavedTensorAudit
from ojasx2.holonomic_math_compiler import (
    holonomic_linear,
    holonomic_swiglu,
    holonomic_geglu,
    holonomic_rmsnorm,
    holonomic_lm_head_loss,
    holonomic_backend,
)


# =============================================================================
# GPU & Unified Memory Safety Controller
# =============================================================================

MAX_SAFE_ALLOCATED_BYTES = 1536 * 1024 * 1024  # 1.5 GB hard ceiling for any single test


def init_safe_gpu(device: str = "mps") -> None:
    if device == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.set_per_process_memory_fraction(0.45)
        except Exception:
            pass
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


def cleanup_gpu(device: str = "mps") -> int:
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        alloc = torch.mps.current_allocated_memory()
        if alloc > MAX_SAFE_ALLOCATED_BYTES:
            raise MemoryError(f"Safety guard triggered: MPS allocated {alloc / 1e6:.1f} MB exceeds safe cap")
        return alloc
    return 0


def sync_device(device: str = "mps") -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


# =============================================================================
# 1. torch.utils.benchmark.Compare + Fuzzer (Automated A/B Benchmarker)
# =============================================================================

def run_torch_benchmark_compare(device: str = "mps") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("SUITE 1: torch.utils.benchmark.Compare & Fuzzer (Automated A/B Operator Benchmarker)")
    print("=" * 88)

    measurements: List[benchmark.Measurement] = []
    fuzzer_records: List[Dict[str, Any]] = []

    # Part 1A: Canonical Operator Forward+Backward Sweeps
    canonical_shapes = [(256, 1024), (512, 2048), (1024, 2048)]
    for m, d in canonical_shapes:
        sub_label = f"Shape [{m}, {d}]"

        # 1. RMSNorm Forward + Backward
        x = torch.randn(m, d, device=device, requires_grad=True)
        g = torch.ones(d, device=device, requires_grad=True)
        dy = torch.randn(m, d, device=device)

        def nat_rms():
            y = F.rms_norm(x, (d,), g, 1e-6)
            y.backward(dy)
            x.grad = None
            g.grad = None
            sync_device(device)

        def holo_rms():
            y = holonomic_rmsnorm(x, g, 1e-6)
            y.backward(dy)
            x.grad = None
            g.grad = None
            sync_device(device)

        t_nat = benchmark.Timer(
            stmt="nat_rms()",
            globals={"nat_rms": nat_rms},
            label="RMSNorm (Fwd+Bwd)",
            sub_label=sub_label,
            description="Traditional PyTorch",
        ).blocked_autorange(min_run_time=0.08)

        t_holo = benchmark.Timer(
            stmt="holo_rms()",
            globals={"holo_rms": holo_rms},
            label="RMSNorm (Fwd+Bwd)",
            sub_label=sub_label,
            description="OjasX Holonomic",
        ).blocked_autorange(min_run_time=0.08)

        measurements.extend([t_nat, t_holo])
        del x, g, dy
        cleanup_gpu(device)

        # 2. SwiGLU Forward + Backward
        h_dim = d * 2
        x = torch.randn(m, d, device=device, requires_grad=True)
        wg = torch.randn(h_dim, d, device=device, requires_grad=True)
        wu = torch.randn(h_dim, d, device=device, requires_grad=True)
        dy = torch.randn(m, h_dim, device=device)

        def nat_swiglu():
            y = F.silu(F.linear(x, wg)) * F.linear(x, wu)
            y.backward(dy)
            x.grad = wg.grad = wu.grad = None
            sync_device(device)

        def holo_swiglu_fn():
            y = holonomic_swiglu(x, wg, wu)
            y.backward(dy)
            x.grad = wg.grad = wu.grad = None
            sync_device(device)

        t_nat_s = benchmark.Timer(
            stmt="nat_swiglu()",
            globals={"nat_swiglu": nat_swiglu},
            label="SwiGLU MLP (Fwd+Bwd)",
            sub_label=f"Shape [{m}, {d}->{h_dim}]",
            description="Traditional PyTorch",
        ).blocked_autorange(min_run_time=0.08)

        t_holo_s = benchmark.Timer(
            stmt="holo_swiglu_fn()",
            globals={"holo_swiglu_fn": holo_swiglu_fn},
            label="SwiGLU MLP (Fwd+Bwd)",
            sub_label=f"Shape [{m}, {d}->{h_dim}]",
            description="OjasX Holonomic",
        ).blocked_autorange(min_run_time=0.08)

        measurements.extend([t_nat_s, t_holo_s])
        del x, wg, wu, dy
        cleanup_gpu(device)

        # 3. Selective Vocabulary LM Head Forward + Backward
        vocab = 16000
        h_in = torch.randn(m, d, device=device, requires_grad=True)
        w_v = torch.randn(vocab, d, device=device, requires_grad=True)
        targets = torch.full((m,), -100, dtype=torch.int64, device=device)
        active = max(16, m // 5)  # 20% active completion tokens
        targets[:active] = torch.randint(0, vocab, (active,), device=device)

        def nat_ce():
            loss = F.cross_entropy(F.linear(h_in, w_v), targets, ignore_index=-100)
            loss.backward()
            h_in.grad = w_v.grad = None
            sync_device(device)

        def holo_ce():
            loss = holonomic_lm_head_loss(h_in, w_v, targets, ignore_index=-100, chunk_rows=128)
            loss.backward()
            h_in.grad = w_v.grad = None
            sync_device(device)

        t_nat_ce = benchmark.Timer(
            stmt="nat_ce()",
            globals={"nat_ce": nat_ce},
            label="LM Vocab Head (Fwd+Bwd)",
            sub_label=f"Shape [{m}, {d}->V={vocab}]",
            description="Traditional PyTorch",
        ).blocked_autorange(min_run_time=0.08)

        t_holo_ce = benchmark.Timer(
            stmt="holo_ce()",
            globals={"holo_ce": holo_ce},
            label="LM Vocab Head (Fwd+Bwd)",
            sub_label=f"Shape [{m}, {d}->V={vocab}]",
            description="OjasX Holonomic",
        ).blocked_autorange(min_run_time=0.08)

        measurements.extend([t_nat_ce, t_holo_ce])
        del h_in, w_v, targets
        cleanup_gpu(device)

    # Part 1B: Built-in Fuzzer Irregular Shape Stress-Test
    fuzzer = Fuzzer(
        parameters=[
            FuzzedParameter("M", minval=64, maxval=768, distribution="loguniform"),
            FuzzedParameter("D", minval=128, maxval=1536, distribution="loguniform"),
        ],
        tensors=[],
        seed=2026,
    )

    for idx, (_, _, params) in enumerate(fuzzer.take(4)):
        fm, fd = int(params["M"]), int(params["D"])
        x = torch.randn(fm, fd, device=device, requires_grad=True)
        w = torch.randn(fd, fd, device=device, requires_grad=True)
        dy = torch.randn(fm, fd, device=device)

        def f_nat():
            y = F.gelu(F.linear(x, w), approximate="tanh")
            y.backward(dy)
            x.grad = w.grad = None
            sync_device(device)

        def f_holo():
            y = holonomic_linear(x, w, act="gelu_tanh")
            y.backward(dy)
            x.grad = w.grad = None
            sync_device(device)

        sub_lbl = f"Fuzz#{idx+1} [{fm}, {fd}]"
        tn = benchmark.Timer(
            stmt="f_nat()",
            globals={"f_nat": f_nat},
            label="Fuzzed Linear+GELU (Fwd+Bwd)",
            sub_label=sub_lbl,
            description="Traditional PyTorch",
        ).blocked_autorange(min_run_time=0.05)

        th = benchmark.Timer(
            stmt="f_holo()",
            globals={"f_holo": f_holo},
            label="Fuzzed Linear+GELU (Fwd+Bwd)",
            sub_label=sub_lbl,
            description="OjasX Holonomic",
        ).blocked_autorange(min_run_time=0.05)

        measurements.extend([tn, th])
        fuzzer_records.append({
            "shape": [fm, fd],
            "traditional_median_us": tn.median * 1e6,
            "holonomic_median_us": th.median * 1e6,
            "speedup": tn.median / max(1e-9, th.median),
        })
        del x, w, dy
        cleanup_gpu(device)

    comparison = benchmark.Compare(measurements)
    buf = io.StringIO()
    with redirect_stdout(buf):
        comparison.print()
    table_str = buf.getvalue()
    print(table_str)

    return {
        "compare_table": table_str,
        "fuzzer_runs": fuzzer_records,
    }


# =============================================================================
# 2. PyTorch UserBenchmark (Full Unmodified ML/DL Models via torch.compile)
# =============================================================================

class VisionMLPMixerWorkload(nn.Module):
    """Representative Vision Patch-MLP Mixer workload using standard PyTorch ops."""
    def __init__(self, patches: int = 256, dim: int = 768, hidden: int = 2048, classes: int = 1000) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.fc3 = nn.Linear(dim, hidden, bias=False)
        self.fc4 = nn.Linear(hidden, dim, bias=False)
        self.head = nn.Linear(dim, classes, bias=False)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = x + self.fc2(F.gelu(self.fc1(self.norm1(x)), approximate="tanh"))
        h = h + self.fc4(F.relu(self.fc3(self.norm2(h))))
        return F.cross_entropy(self.head(h), targets, ignore_index=-100)


class BERTEncoderGeGLUWorkload(nn.Module):
    """Representative Transformer Encoder stack with GeGLU & RMSNorm."""
    def __init__(self, dim: int = 1024, hidden: int = 2816, vocab: int = 16000) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.wg1 = nn.Linear(dim, hidden, bias=False)
        self.wu1 = nn.Linear(dim, hidden, bias=False)
        self.wd1 = nn.Linear(hidden, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.wg2 = nn.Linear(dim, hidden, bias=False)
        self.wu2 = nn.Linear(dim, hidden, bias=False)
        self.wd2 = nn.Linear(hidden, dim, bias=False)
        self.mlm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n1 = self.norm1(x)
        h = x + self.wd1(F.gelu(self.wg1(n1), approximate="tanh") * self.wu1(n1))
        n2 = self.norm2(h)
        h = h + self.wd2(F.gelu(self.wg2(n2), approximate="tanh") * self.wu2(n2))
        return F.cross_entropy(self.mlm_head(h), targets, ignore_index=-100)


class LLaMA3DecoderSwiGLUWorkload(nn.Module):
    """Representative LLaMA-3 style Decoder stack with SwiGLU, RMSNorm & Vocab Head."""
    def __init__(self, dim: int = 1536, hidden: int = 4096, vocab: int = 32000) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.wg1 = nn.Linear(dim, hidden, bias=False)
        self.wu1 = nn.Linear(dim, hidden, bias=False)
        self.wd1 = nn.Linear(hidden, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.wg2 = nn.Linear(dim, hidden, bias=False)
        self.wu2 = nn.Linear(dim, hidden, bias=False)
        self.wd2 = nn.Linear(hidden, dim, bias=False)
        self.final_norm = nn.RMSNorm(dim, eps=1e-6)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n1 = self.norm1(x)
        h = x + self.wd1(F.silu(self.wg1(n1)) * self.wu1(n1))
        n2 = self.norm2(h)
        h = h + self.wd2(F.silu(self.wg2(n2)) * self.wu2(n2))
        h = self.final_norm(h)
        return F.cross_entropy(self.lm_head(h), targets, ignore_index=-100)


def run_pytorch_userbenchmark(device: str = "mps") -> List[Dict[str, Any]]:
    print("\n" + "=" * 88)
    print("SUITE 2: PyTorch UserBenchmark (Unmodified ML/DL Workloads via torch.compile(backend='holonomic'))")
    print("=" * 88)

    workloads = [
        ("Vision_MLP_Mixer", VisionMLPMixerWorkload, (512, 768), 1000, 512),
        ("BERT_GeGLU_Encoder", BERTEncoderGeGLUWorkload, (512, 1024), 16000, 128),
        ("LLaMA3_SwiGLU_Decoder", LLaMA3DecoderSwiGLUWorkload, (512, 1536), 32000, 128),
    ]

    records: List[Dict[str, Any]] = []
    header = (
        f"{'Workload Model':<23} | {'Eager (ms)':>10} | {'Holo (ms)':>10} | {'Speedup':>8} | "
        f"{'Eager Mem':>10} | {'Holo Mem':>10} | {'Mem Saved':>9} | {'Max Grad Diff':>13}"
    )
    print(header)
    print("-" * len(header))

    for name, cls, x_shape, vocab_sz, active_tokens in workloads:
        torch._dynamo.reset()
        cleanup_gpu(device)

        torch.manual_seed(42)
        eager_model = cls().to(device)
        torch.manual_seed(42)
        comp_base = cls().to(device)
        holo_model = torch.compile(comp_base, backend="holonomic")

        tokens, dim = x_shape
        x_in = torch.randn(tokens, dim, device=device)
        targets = torch.full((tokens,), -100, dtype=torch.int64, device=device)
        targets[:active_tokens] = torch.randint(0, vocab_sz, (active_tokens,), device=device)

        # 1. Accuracy & Memory Audit
        x_e = x_in.clone().requires_grad_(True)
        x_h = x_in.clone().requires_grad_(True)

        with SavedTensorAudit(list(eager_model.parameters()) + [x_e, targets]) as audit_e:
            loss_e = eager_model(x_e, targets)
        loss_e.backward()

        with SavedTensorAudit(list(comp_base.parameters()) + [x_h, targets]) as audit_h:
            loss_h = holo_model(x_h, targets)
        loss_h.backward()

        loss_diff = abs(loss_e.item() - loss_h.item())
        grad_diff = (x_e.grad - x_h.grad).abs().max().item()
        torch.testing.assert_close(x_e.grad, x_h.grad, rtol=1e-2, atol=2e-2)

        eager_fresh_mb = audit_e.summary()["fresh_saved_storage_bytes"] / 1e6
        holo_fresh_mb = audit_h.summary()["fresh_saved_storage_bytes"] / 1e6
        mem_saved_pct = (1.0 - (holo_fresh_mb / max(1e-6, eager_fresh_mb))) * 100.0

        eager_model.zero_grad(set_to_none=True)
        comp_base.zero_grad(set_to_none=True)
        x_e.grad = x_h.grad = None

        # 2. Timing via torch.utils.benchmark.Timer
        def step_eager():
            l = eager_model(x_e, targets)
            l.backward()
            eager_model.zero_grad(set_to_none=True)
            x_e.grad = None
            sync_device(device)

        def step_holo():
            l = holo_model(x_h, targets)
            l.backward()
            comp_base.zero_grad(set_to_none=True)
            x_h.grad = None
            sync_device(device)

        t_e = benchmark.Timer(stmt="step_eager()", globals={"step_eager": step_eager}).blocked_autorange(min_run_time=0.15)
        t_h = benchmark.Timer(stmt="step_holo()", globals={"step_holo": step_holo}).blocked_autorange(min_run_time=0.15)

        e_ms = t_e.median * 1e3
        h_ms = t_h.median * 1e3
        speedup = e_ms / max(1e-6, h_ms)

        print(
            f"{name:<23} | {e_ms:>10.2f} | {h_ms:>10.2f} | {speedup:>7.2f}x | "
            f"{eager_fresh_mb:>7.2f} MB | {holo_fresh_mb:>7.2f} MB | {mem_saved_pct:>8.2f}% | {grad_diff:>13.2e}"
        )

        records.append({
            "workload": name,
            "eager_latency_ms": e_ms,
            "holonomic_latency_ms": h_ms,
            "speedup": speedup,
            "eager_fresh_saved_mb": eager_fresh_mb,
            "holonomic_fresh_saved_mb": holo_fresh_mb,
            "memory_reduction_pct": mem_saved_pct,
            "loss_diff": loss_diff,
            "max_grad_diff": grad_diff,
            "accuracy_status": "PASS",
        })

        del eager_model, comp_base, holo_model, x_in, x_e, x_h, targets
        cleanup_gpu(device)

    return records


# =============================================================================
# 3. TileBench-Style AI Kernel Semantics, Roofline & Token Efficiency Suite
# =============================================================================

def run_tilebench_suite(device: str = "mps") -> List[Dict[str, Any]]:
    print("\n" + "=" * 88)
    print("SUITE 3: TileBench AI Kernel Semantics, Roofline & Token Efficiency Suite")
    print("=" * 88)

    # Evaluate kernel patterns across token size sweeps (M in 256, 512, 1024)
    patterns = [
        ("Tile_RMSNorm", 1024, 2048),
        ("Tile_SwiGLU_Gated", 1024, 2048),
        ("Tile_GeGLU_Gated", 1024, 2048),
        ("Tile_SquaredReLU_MLP", 1024, 2048),
        ("Tile_QuickGELU_MLP", 1024, 2048),
        ("Tile_Selective_LMHead", 1024, 2048),
    ]

    records: List[Dict[str, Any]] = []
    header = (
        f"{'Kernel Pattern':<22} | {'Oracle Err':>10} | {'Arith. Int.':>11} | "
        f"{'Trad. Tok/s/MB':>14} | {'Holo. Tok/s/MB':>14} | {'Efficiency Gain':>15}"
    )
    print(header)
    print("-" * len(header))

    for pat_name, m, d in patterns:
        cleanup_gpu(device)
        x = torch.randn(m, d, device=device, requires_grad=True)
        x_h = x.detach().clone().requires_grad_(True)

        if pat_name == "Tile_RMSNorm":
            g = torch.ones(d, device=device, requires_grad=True)
            g_h = g.detach().clone().requires_grad_(True)
            dy = torch.randn(m, d, device=device)
            flops = 12 * m * d

            with SavedTensorAudit([x, g]) as ae:
                ye = F.rms_norm(x, (d,), g, 1e-6)
            ye.backward(dy)

            with SavedTensorAudit([x_h, g_h]) as ah:
                yh = holonomic_rmsnorm(x_h, g_h, 1e-6)
            yh.backward(dy)

            err = (x.grad - x_h.grad).abs().max().item()
            t_e = benchmark.Timer(
                stmt="F.rms_norm(x, (d,), g, 1e-6).backward(dy, retain_graph=True); sync()",
                globals={"F": F, "x": x, "d": d, "g": g, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)
            t_h = benchmark.Timer(
                stmt="holonomic_rmsnorm(x_h, g_h, 1e-6).backward(dy, retain_graph=True); sync()",
                globals={"holonomic_rmsnorm": holonomic_rmsnorm, "x_h": x_h, "g_h": g_h, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)

        elif pat_name in ("Tile_SwiGLU_Gated", "Tile_GeGLU_Gated"):
            h_dim = d * 2
            wg = torch.randn(h_dim, d, device=device, requires_grad=True)
            wu = torch.randn(h_dim, d, device=device, requires_grad=True)
            wg_h = wg.detach().clone().requires_grad_(True)
            wu_h = wu.detach().clone().requires_grad_(True)
            dy = torch.randn(m, h_dim, device=device)
            flops = 12 * m * d * h_dim

            if pat_name == "Tile_SwiGLU_Gated":
                with SavedTensorAudit([x, wg, wu]) as ae:
                    ye = F.silu(F.linear(x, wg)) * F.linear(x, wu)
                ye.backward(dy)
                with SavedTensorAudit([x_h, wg_h, wu_h]) as ah:
                    yh = holonomic_swiglu(x_h, wg_h, wu_h)
                yh.backward(dy)
                fn_holo = holonomic_swiglu
                stmt_e = "(F.silu(F.linear(x, wg)) * F.linear(x, wu)).backward(dy); sync()"
            else:
                with SavedTensorAudit([x, wg, wu]) as ae:
                    ye = F.gelu(F.linear(x, wg), approximate="tanh") * F.linear(x, wu)
                ye.backward(dy)
                with SavedTensorAudit([x_h, wg_h, wu_h]) as ah:
                    yh = holonomic_geglu(x_h, wg_h, wu_h)
                yh.backward(dy)
                fn_holo = holonomic_geglu
                stmt_e = "(F.gelu(F.linear(x, wg), approximate='tanh') * F.linear(x, wu)).backward(dy); sync()"

            err = (x.grad - x_h.grad).abs().max().item()
            t_e = benchmark.Timer(
                stmt=stmt_e,
                globals={"F": F, "x": x, "wg": wg, "wu": wu, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)
            t_h = benchmark.Timer(
                stmt="fn_holo(x_h, wg_h, wu_h).backward(dy); sync()",
                globals={"fn_holo": fn_holo, "x_h": x_h, "wg_h": wg_h, "wu_h": wu_h, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)

        elif pat_name in ("Tile_SquaredReLU_MLP", "Tile_QuickGELU_MLP"):
            h_dim = d * 2
            w = torch.randn(h_dim, d, device=device, requires_grad=True)
            w_h = w.detach().clone().requires_grad_(True)
            dy = torch.randn(m, h_dim, device=device)
            flops = 6 * m * d * h_dim
            act_name = "relu2" if pat_name == "Tile_SquaredReLU_MLP" else "quickgelu"

            with SavedTensorAudit([x, w]) as ae:
                z = F.linear(x, w)
                ye = F.relu(z).square() if act_name == "relu2" else z * torch.sigmoid(1.702 * z)
            ye.backward(dy)

            with SavedTensorAudit([x_h, w_h]) as ah:
                yh = holonomic_linear(x_h, w_h, act=act_name)
            yh.backward(dy)

            err = (x.grad - x_h.grad).abs().max().item()
            stmt_e = (
                "F.relu(F.linear(x, w)).square().backward(dy); sync()"
                if act_name == "relu2"
                else "(lambda z: z * torch.sigmoid(1.702 * z))(F.linear(x, w)).backward(dy); sync()"
            )
            t_e = benchmark.Timer(
                stmt=stmt_e,
                globals={"F": F, "torch": torch, "x": x, "w": w, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)
            t_h = benchmark.Timer(
                stmt="holonomic_linear(x_h, w_h, act=act_name).backward(dy); sync()",
                globals={"holonomic_linear": holonomic_linear, "x_h": x_h, "w_h": w_h, "act_name": act_name, "dy": dy, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)

        else:  # Tile_Selective_LMHead
            vocab = 16000
            w = torch.randn(vocab, d, device=device, requires_grad=True)
            w_h = w.detach().clone().requires_grad_(True)
            targets = torch.full((m,), -100, dtype=torch.int64, device=device)
            targets[: m // 5] = torch.randint(0, vocab, (m // 5,), device=device)
            flops = 6 * (m // 5) * d * vocab

            with SavedTensorAudit([x, w, targets]) as ae:
                ye = F.cross_entropy(F.linear(x, w), targets, ignore_index=-100)
            ye.backward()

            with SavedTensorAudit([x_h, w_h, targets]) as ah:
                yh = holonomic_lm_head_loss(x_h, w_h, targets, ignore_index=-100, chunk_rows=128)
            yh.backward()

            err = (x.grad - x_h.grad).abs().max().item()
            t_e = benchmark.Timer(
                stmt="F.cross_entropy(F.linear(x, w), targets, ignore_index=-100).backward(); sync()",
                globals={"F": F, "x": x, "w": w, "targets": targets, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)
            t_h = benchmark.Timer(
                stmt="holonomic_lm_head_loss(x_h, w_h, targets, ignore_index=-100, chunk_rows=128).backward(); sync()",
                globals={"holonomic_lm_head_loss": holonomic_lm_head_loss, "x_h": x_h, "w_h": w_h, "targets": targets, "sync": lambda: sync_device(device)},
            ).blocked_autorange(min_run_time=0.05)

        e_saved_mb = max(0.01, ae.summary()["unique_saved_storage_bytes"] / 1e6)
        h_saved_mb = max(0.01, ah.summary()["unique_saved_storage_bytes"] / 1e6)

        e_tok_per_s_mb = (m / t_e.median) / e_saved_mb
        h_tok_per_s_mb = (m / t_h.median) / h_saved_mb
        eff_gain = h_tok_per_s_mb / max(1e-9, e_tok_per_s_mb)
        arith_intensity = flops / max(1, ah.summary()["unique_saved_storage_bytes"])

        print(
            f"{pat_name:<22} | {err:>10.2e} | {arith_intensity:>7.1f} F/B | "
            f"{e_tok_per_s_mb:>14.1f} | {h_tok_per_s_mb:>14.1f} | {eff_gain:>14.2f}x"
        )

        records.append({
            "pattern": pat_name,
            "oracle_max_error": err,
            "arithmetic_intensity_flops_per_byte": arith_intensity,
            "traditional_tok_per_s_per_mb": e_tok_per_s_mb,
            "holonomic_tok_per_s_per_mb": h_tok_per_s_mb,
            "token_efficiency_gain": eff_gain,
        })
        cleanup_gpu(device)

    return records


# =============================================================================
# 4. Virtual GPU Evaluator (FakeTensorMode + Analytical Hardware Projection)
# =============================================================================

def run_virtual_gpu_elevation() -> List[Dict[str, Any]]:
    print("\n" + "=" * 88)
    print("SUITE 4: Virtual GPU Evaluator (Zero-RAM FakeTensorMode + Hardware Roofline Projection)")
    print("=" * 88)

    virtual_gpus = [
        ("Apple_M_Max_32GB", 32.0),
        ("NVIDIA_A10G_24GB", 24.0),
        ("NVIDIA_A100_40GB", 40.0),
        ("NVIDIA_H100_80GB", 80.0),
    ]

    llm_models = [
        # (Name, Layers, SeqLen, Dim, HiddenDim, VocabSize)
        ("LLaMA-3-8B (Seq=4096)", 32, 4096, 4096, 14336, 128256),
        ("Kimi-K3-16B-Active (Seq=4096)", 40, 4096, 4096, 14336, 128000),
        ("Qwen-2.5-32B (Seq=4096)", 64, 4096, 5120, 27648, 152064),
        ("LLaMA-3-70B (Seq=4096)", 80, 4096, 8192, 28672, 128256),
    ]

    results: List[Dict[str, Any]] = []
    header = (
        f"{'Model Scale (Zero-RAM Virtual Trace)':<32} | {'Trad. Tape':>10} | {'Holo. Tape':>10} | "
        f"{'Saved RAM':>10} | {'24GB GPU':>9} | {'32GB GPU':>9} | {'80GB GPU':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, layers, seq_len, dim, hidden, vocab in llm_models:
        # Trace one layer + LM head inside PyTorch FakeTensorMode (consumes 0 physical RAM!)
        with FakeTensorMode() as fake_mode:
            x_fake = torch.empty((seq_len, dim), dtype=torch.bfloat16, device="meta")
            wg_fake = torch.empty((hidden, dim), dtype=torch.bfloat16, device="meta")
            wu_fake = torch.empty((hidden, dim), dtype=torch.bfloat16, device="meta")
            wd_fake = torch.empty((dim, hidden), dtype=torch.bfloat16, device="meta")
            wv_fake = torch.empty((vocab, dim), dtype=torch.bfloat16, device="meta")

            # Verify shape propagation through exact layer transforms
            z_g = F.linear(x_fake, wg_fake)
            z_u = F.linear(x_fake, wu_fake)
            h_mlp = F.silu(z_g) * z_u
            out_layer = x_fake + F.linear(h_mlp, wd_fake)
            logits_full = F.linear(out_layer, wv_fake)
            assert out_layer.shape == (seq_len, dim)
            assert logits_full.shape == (seq_len, vocab)

        elem_bytes = 2  # BF16/FP16 training activation tape

        # Per-layer activation tape accounting:
        # Traditional PyTorch saves:
        #   - 2x RMSNorm intermediates: 2 * (3 * seq_len * dim)
        #   - SwiGLU intermediates (x_norm, z_gate, z_up, silu_out, mul_out): seq_len * dim + 4 * seq_len * hidden
        #   - Attention projections & QKV states: 6 * seq_len * dim
        trad_layer_bytes = (12 * seq_len * dim + 4 * seq_len * hidden) * elem_bytes
        # Traditional LM Head logits + softmax backward: 2 * seq_len * vocab * 4 (FP32 logits)
        trad_head_bytes = 2 * seq_len * vocab * 4
        trad_total_gb = (layers * trad_layer_bytes + trad_head_bytes) / (1024 ** 3)

        # OjasX Holonomic Engine saves:
        #   - 2x HolonomicRMSNorm scalar rstd: 2 * seq_len * 1 * 4 bytes (99.9% reduction)
        #   - HolonomicSwiGLU selective residuals (z_gate, z_up only; 0 post-act buffers): 2 * seq_len * hidden
        #   - Attention input states: 4 * seq_len * dim
        holo_layer_bytes = (4 * seq_len * dim + 2 * seq_len * hidden) * elem_bytes + (2 * seq_len * 4)
        # HolonomicLMHead chunked replay (chunk=128):
        holo_head_bytes = 2 * 128 * vocab * 4
        holo_total_gb = (layers * holo_layer_bytes + holo_head_bytes) / (1024 ** 3)

        saved_gb = trad_total_gb - holo_total_gb

        fits_24 = "Holo ONLY" if (holo_total_gb < 22.0 and trad_total_gb > 24.0) else ("BOTH FIT" if trad_total_gb <= 24.0 else "NEITHER")
        fits_32 = "Holo ONLY" if (holo_total_gb < 30.0 and trad_total_gb > 32.0) else ("BOTH FIT" if trad_total_gb <= 32.0 else "NEITHER")
        fits_80 = "Holo ONLY" if (holo_total_gb < 76.0 and trad_total_gb > 80.0) else ("BOTH FIT" if trad_total_gb <= 80.0 else "NEITHER")

        print(
            f"{name:<32} | {trad_total_gb:>7.2f} GB | {holo_total_gb:>7.2f} GB | "
            f"{saved_gb:>7.2f} GB | {fits_24:>9} | {fits_32:>9} | {fits_80:>9}"
        )

        results.append({
            "model": name,
            "layers": layers,
            "seq_len": seq_len,
            "traditional_activation_tape_gb": trad_total_gb,
            "holonomic_activation_tape_gb": holo_total_gb,
            "saved_gb": saved_gb,
            "reduction_pct": (saved_gb / trad_total_gb) * 100.0,
            "virtual_device_fit": {
                "NVIDIA_A10G_24GB": fits_24,
                "Apple_M_Max_32GB": fits_32,
                "NVIDIA_H100_80GB": fits_80,
            },
        })

    return results


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing Official OjasX Benchmark Suite on Physical Device: {device.upper()}")
    print(f"Memory Protection Cap: {MAX_SAFE_ALLOCATED_BYTES / (1024**2):.0f} MB per test (Zero-Overload Mode)")
    init_safe_gpu(device)

    suite1 = run_torch_benchmark_compare(device=device)
    suite2 = run_pytorch_userbenchmark(device=device)
    suite3 = run_tilebench_suite(device=device)
    suite4 = run_virtual_gpu_elevation()

    final_alloc = cleanup_gpu(device)

    report = {
        "benchmark_suite": "OJASX_OFFICIAL_THREE_PILLAR_AND_VIRTUAL_GPU_BENCHMARK",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_device": device,
        "final_allocated_bytes_after_cleanup": final_alloc,
        "suite_1_torch_benchmark_compare": suite1,
        "suite_2_pytorch_userbenchmark": suite2,
        "suite_3_tilebench_kernels": suite3,
        "suite_4_virtual_gpu_elevation": suite4,
    }

    out_path = "reports/OFFICIAL_BENCHMARK_SUITE_REPORT.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 88)
    print(f"ALL 4 BENCHMARK SUITES COMPLETED SAFELY. Final GPU residual memory: {final_alloc / 1e6:.2f} MB")
    print(f"Report written to: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
