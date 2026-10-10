#!/usr/bin/env python3
"""
tools/run_industrial_all_level_benchmark.py

OjasX five-level local qualification and benchmark sample suite:
  - LEVEL 1 (native CPU / Metal operator checks):
      Official `torch.library.opcheck` + `Z3 SMT BitVec32` + `torch.utils.benchmark.Compare`
  - LEVEL 2 (Official PyTorch `OpInfo` `op_db` Conformance Level):
      Directly tests against `torch.testing._internal.common_methods_invocations.op_db`
  - LEVEL 3 (Meta `BackendBench` Production Parquet Trace Level):
      Directly loads `third_party/backend_bench_problems.parquet` (19,171 real production rows)
      selects 25 traces and tests capped, contiguous FP32 linear/ReLU2 workloads
  - LEVEL 4 (Industry Model Training & LoRA Smart-Router Level):
      Small LLaMA-style RMSNorm/SwiGLU/residual/loss block (no attention or RoPE)
      and LoRA Fine-Tuning via `SmartWorkloadRouter`
  - LEVEL 5 (Multi-Laptop P2P Heterogeneous Virtual Unified GPU Mesh Level):
      Same-process local MPS/CPU or CPU/CPU loopback, with estimated capacity descriptors
"""

from __future__ import annotations

import ast
import copy
import gc
import json
import hashlib
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import ojasx2
from ojasx2 import (
    HolonomicBitPackedLinear,
    HolonomicGradFreeOptimizer,
    HolonomicLMHead,
    HolonomicLinear,
    HolonomicNativeRMSNorm,
    HolonomicRMSNorm,
    HolonomicRoPE,
    HolonomicSwiGLU,
    HolonomicVirtualMeshEngine,
    MeshNodeSpec,
    SavedTensorAudit,
    SmartWorkloadRouter,
    probe_hardware_profile,
    tilelang_emit_universal_source,
    tilelang_query_chip_schedule,
    tilelang_universal_gemm_relu2,
    verify_xor_swizzle_and_ballot_smt,
)
from ojasx2.tile_hal import tile_kernel_status


def init_safe_gpu() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.45)
        torch.mps.synchronize()
        torch.mps.empty_cache()
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_gpu() -> float:
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def parse_backendbench_shape(arg_str: str) -> Tuple[List[int], torch.dtype, List[int]]:
    """Parses a Meta BackendBench tensor descriptor like `((64, 512), torch.float32, (512, 1))`."""
    safe_str = (
        arg_str.replace("torch.float32", "'float32'")
        .replace("torch.float16", "'float16'")
        .replace("torch.bfloat16", "'bfloat16'")
    )
    parsed = ast.literal_eval(safe_str)
    first = parsed[0] if isinstance(parsed, tuple) and isinstance(parsed[0], tuple) and isinstance(parsed[0][0], tuple) else parsed
    shape = list(first[0])
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(first[1], torch.float32)
    stride = list(first[2])
    return shape, dtype, stride


def qualify_backendbench_case(y_ref, y_actual, x_ref_grad, x_actual_grad, w_ref_grad, w_actual_grad):
    """Require output and both operand gradients to agree before counting a pass."""
    for expected, actual in ((y_ref, y_actual), (x_ref_grad, x_actual_grad), (w_ref_grad, w_actual_grad)):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    if torch.count_nonzero(x_ref_grad) == 0 and torch.count_nonzero(x_actual_grad) == 0:
        return 1.0
    similarity = F.cosine_similarity(x_ref_grad.flatten(), x_actual_grad.flatten(), dim=0).item()
    if not torch.isfinite(torch.tensor(similarity)) or similarity < 0.999:
        raise AssertionError(f"Input gradient cosine similarity below qualification threshold: {similarity}")
    return similarity


def main() -> None:
    import pandas as pd
    from torch.testing._internal.common_methods_invocations import op_db
    if not tile_kernel_status()["native_tilelang_loaded"]:
        raise RuntimeError("This native-kernel benchmark requires the optional native TileLang library; reference fallback timings are not native measurements")
    torch.manual_seed(20261010)
    device = init_safe_gpu()
    silicon = probe_hardware_profile()

    print("=" * 100)
    print("OJASX v0.3.0 — FIVE-LEVEL LOCAL SAMPLE QUALIFICATION & BENCHMARK")
    print(f"Silicon Probed: {silicon.chip_class} | {silicon.physical_cpu_cores} Cores | L1d={silicon.l1d_cache_bytes // 1024}KB | L2={silicon.l2_cache_bytes // 1024}KB | Device={device}")
    print("=" * 100)

    # =========================================================================
    # LEVEL 1: C++17 / TileLang 6-Backend Kernel Level (opcheck + Z3 + Compare)
    # =========================================================================
    print("\n[LEVEL 1/5] Native CPU operators and CPU/Metal mask comparison...")
    smt_res = verify_xor_swizzle_and_ballot_smt()
    assert smt_res["zero_bank_conflicts_proven"] is True

    x_op = torch.randn(32, 64, requires_grad=True)
    w_op = torch.randn(64, 64, requires_grad=True)
    b_op = torch.randn(64, requires_grad=True)
    rw_op = torch.randn(64, requires_grad=True)

    torch.library.opcheck(torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd.default, (x_op, w_op, b_op))
    torch.library.opcheck(torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd.default, (x_op, rw_op, 1e-6))
    torch.library.opcheck(torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd.default, (x_op, w_op, b_op, "cpu", 4))
    torch.library.opcheck(torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd.default, (x_op, w_op, b_op, "metal", 4))

    # Cross-Backend Bit-Exact Ballot Mask Check (CPU L1d/L2 NEON vs Metal SIMDGroup GPU)
    out_c, mask_c = tilelang_universal_gemm_relu2(x_op.detach(), w_op.detach(), b_op.detach(), target_arch="cpu")
    out_m, mask_m = tilelang_universal_gemm_relu2(x_op.detach(), w_op.detach(), b_op.detach(), target_arch="metal")
    cross_backend_bit_exact = bool(torch.equal(mask_c, mask_m))
    assert cross_backend_bit_exact is True

    # Official `torch.utils.benchmark.Compare` on C++ L1 RMSNorm (Fwd + Bwd)
    x_rms = torch.randn(4, 256, 512, requires_grad=True)
    w_rms = torch.ones(512, requires_grad=True)
    dy_rms = torch.randn(4, 256, 512)

    def run_pytorch_rms():
        x_rms.grad = None
        w_rms.grad = None
        y = F.rms_norm(x_rms, (512,), w_rms, 1e-6)
        y.backward(dy_rms)

    mod_cpp_rms = HolonomicNativeRMSNorm(512)
    with torch.no_grad():
        mod_cpp_rms.weight.copy_(w_rms)

    def run_ojasx_cpp_rms():
        x_rms.grad = None
        mod_cpp_rms.weight.grad = None
        y = mod_cpp_rms(x_rms)
        y.backward(dy_rms)

    m_py_rms = benchmark.Timer(
        stmt="run_pytorch_rms()",
        globals={"run_pytorch_rms": run_pytorch_rms},
        label="Level 1 C++ Kernel",
        sub_label="RMSNorm Fwd+Bwd [4, 256, 512]",
        description="PyTorch Standard",
    ).blocked_autorange(min_run_time=0.15)

    m_oj_rms = benchmark.Timer(
        stmt="run_ojasx_cpp_rms()",
        globals={"run_ojasx_cpp_rms": run_ojasx_cpp_rms},
        label="Level 1 C++ Kernel",
        sub_label="RMSNorm Fwd+Bwd [4, 256, 512]",
        description="OjasX C++ L1 vDSP",
    ).blocked_autorange(min_run_time=0.15)

    benchmark.Compare([m_py_rms, m_oj_rms]).print()
    rmsnorm_speedup = m_py_rms.median / m_oj_rms.median
    print(f"  -> C++ L1 RMSNorm Speedup         : {rmsnorm_speedup:.2f}x faster ({m_py_rms.median*1e6:.1f} us -> {m_oj_rms.median*1e6:.1f} us)")
    print(f"  -> CPU/Metal mask comparison     : {cross_backend_bit_exact} (Metal GPU availability={silicon.has_mps_gpu})")

    # =========================================================================
    # LEVEL 2: Official PyTorch `OpInfo` (`op_db`) Conformance Suite
    # =========================================================================
    print("\n[LEVEL 2/5] Sampled FP32 linear conformance using PyTorch OpInfo inputs...")
    linear_op = next(op for op in op_db if op.name == "nn.functional.linear")
    samples = list(linear_op.sample_inputs(str(device), torch.float32, requires_grad=True))
    opinfo_passed = 0
    opinfo_total = 0
    for sample in samples:
        if sample.input.numel() == 0 or sample.args[0].numel() == 0:
            continue
        for act in ("none", "relu", "silu", "gelu", "relu2"):
            opinfo_total += 1
            x1 = sample.input.detach().clone().requires_grad_(True)
            w1 = sample.args[0].detach().clone().requires_grad_(True)
            b1 = sample.args[1].detach().clone().requires_grad_(True) if len(sample.args) > 1 and sample.args[1] is not None else None

            x2 = x1.detach().clone().requires_grad_(True)
            w2 = w1.detach().clone().requires_grad_(True)
            b2 = b1.detach().clone().requires_grad_(True) if b1 is not None else None

            z_ref = F.linear(x1, w1, b1)
            if act == "relu":
                y_ref = F.relu(z_ref)
            elif act == "silu":
                y_ref = F.silu(z_ref)
            elif act == "gelu":
                y_ref = F.gelu(z_ref, approximate="tanh")
            elif act == "relu2":
                y_ref = F.relu(z_ref).square()
            else:
                y_ref = z_ref

            y_holo = ojasx2.holonomic_linear(x2, w2, b2, act=act)
            g_out = torch.ones_like(y_ref)
            y_ref.backward(g_out)
            y_holo.backward(g_out)

            torch.testing.assert_close(y_ref, y_holo, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-3, atol=1e-3)
            if b1 is not None:
                torch.testing.assert_close(b1.grad, b2.grad, rtol=1e-3, atol=1e-3)
            opinfo_passed += 1

    cleanup_gpu()
    assert opinfo_total > 0 and opinfo_passed == opinfo_total
    print(f"  -> Linear FP32 first-order cases : {opinfo_passed}/{opinfo_total} PASSED (sampled numerical conformance)")

    # =========================================================================
    # LEVEL 3: Meta `BackendBench` Real Production Trace (`19,171` Parquet Rows)
    # =========================================================================
    print("\n[LEVEL 3/5] Capped synthetic workloads derived from BackendBench traces...")
    parquet_path = REPO_ROOT / "third_party" / "backend_bench_problems.parquet"
    df = pd.read_parquet(parquet_path)
    total_parquet_rows = len(df)
    mm_df = df[
        (df["op_name"].isin(["aten.mm.default", "aten.addmm.default"])) & (df["runnable"] == True)
    ].head(25)

    class _TSpec:
        def __init__(self, shape, dtype=torch.float32, stride=None):
            self.shape = list(shape)
            self.dtype = dtype
            self.stride = stride

    eval_env = {
        "T": _TSpec,
        "f16": torch.float16,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "i32": torch.int32,
        "i64": torch.int64,
        "b8": torch.bool,
        "torch": torch,
    }

    bb_passed = 0
    bb_total = 0
    bb_cos_sims: List[float] = []
    bb_saved_ratio_list: List[float] = []

    for _, row in mm_df.iterrows():
        raw_args = eval(row["args"], {"__builtins__": {}}, eval_env)
        args_tuple = raw_args[0] if isinstance(raw_args, tuple) and len(raw_args) == 2 and isinstance(raw_args[1], dict) else raw_args
        if row["op_name"] == "aten.mm.default" and len(args_tuple) >= 2:
            M, K = args_tuple[0].shape[0], args_tuple[0].shape[1]
            N = args_tuple[1].shape[1]
        elif row["op_name"] == "aten.addmm.default" and len(args_tuple) >= 3:
            M, K = args_tuple[1].shape[0], args_tuple[1].shape[1]
            N = args_tuple[2].shape[1]
        else:
            continue

        # Independent caps change aspect ratios; this is a synthetic trace-derived workload.
        M = min(int(M), 512)
        K = min(int(K), 1024)
        N = min(int(N), 1024)
        if M <= 0 or K <= 0 or N <= 0:
            continue
        bb_total += 1

        x_bb = torch.randn(M, K, device="cpu", dtype=torch.float32, requires_grad=True)
        w_bb = torch.randn(N, K, device="cpu", dtype=torch.float32, requires_grad=True)
        x_ref = x_bb.detach().clone().requires_grad_(True)
        w_ref = w_bb.detach().clone().requires_grad_(True)

        with SavedTensorAudit([x_ref, w_ref]) as aud_ref:
            y_ref = torch.relu(F.linear(x_ref, w_ref)).square()
        y_ref.sum().backward()

        with SavedTensorAudit([x_bb, w_bb]) as aud_oj:
            y_oj, _ = torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x_bb, w_bb, None)
        y_oj.sum().backward()

        cos_x = qualify_backendbench_case(y_ref, y_oj, x_ref.grad, x_bb.grad, w_ref.grad, w_bb.grad)
        bb_cos_sims.append(cos_x)
        ref_bytes = max(1, aud_ref.summary()["fresh_saved_storage_bytes"])
        oj_bytes = max(1, aud_oj.summary()["fresh_saved_storage_bytes"])
        bb_saved_ratio_list.append(ref_bytes / oj_bytes)
        bb_passed += 1

    assert bb_total > 0 and bb_passed == bb_total
    min_bb_cos = min(bb_cos_sims)
    avg_bb_compression = sum(bb_saved_ratio_list) / len(bb_saved_ratio_list)
    print(f"  -> Meta BackendBench Parquet Rows : {total_parquet_rows:,} production traces loaded")
    print(f"  -> Capped FP32 trace-derived cases: {bb_passed}/{bb_total} PASSED (min grad cosine sim = {min_bb_cos:.6f})")
    print(f"  -> Avg Activation Mask Compression: {avg_bb_compression:.1f}x smaller saved state vs PyTorch")

    # =========================================================================
    # LEVEL 4: End-to-End Industry Model Training (Llama-3 Decoder + LoRA Router)
    # =========================================================================
    print("\n[LEVEL 4/5] Small LLaMA-style block and LoRA-style model...")

    # 4A: Llama-3 Decoder Block with Holonomic RMSNorm + SwiGLU + Selective LMHead + 0B .grad
    torch.manual_seed(42)
    B, S, D, H, V = 4, 128, 256, 512, 4096

    class StandardLlamaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(D, eps=1e-6)
            self.wg = nn.Linear(D, H, bias=False)
            self.wu = nn.Linear(D, H, bias=False)
            self.wd = nn.Linear(H, D, bias=False)
            self.lm_head = nn.Linear(D, V, bias=False)

        def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            mlp = self.wd(F.silu(self.wg(h)) * self.wu(h))
            out = x + mlp
            logits = self.lm_head(out.view(-1, D))
            return F.cross_entropy(logits, targets.view(-1), ignore_index=-100)

    class OjasXLlamaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = HolonomicRMSNorm(D, eps=1e-6)
            self.swiglu = HolonomicSwiGLU(D, H, bias=False)
            self.wd = HolonomicLinear(H, D, bias=False, act="none")
            self.lm_head = HolonomicLMHead(D, V, chunk_rows=128)

        def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            mlp = self.wd(self.swiglu(h))
            out = x + mlp
            return self.lm_head(out.view(-1, D), targets.view(-1), ignore_index=-100)

    std_llama = StandardLlamaBlock().to(device)
    oj_llama = OjasXLlamaBlock().to(device)
    with torch.no_grad():
        oj_llama.norm.weight.copy_(std_llama.norm.weight)
        oj_llama.swiglu.w_gate.copy_(std_llama.wg.weight)
        oj_llama.swiglu.w_up.copy_(std_llama.wu.weight)
        oj_llama.wd.weight.copy_(std_llama.wd.weight)
        oj_llama.lm_head.weight.copy_(std_llama.lm_head.weight)

    x_tok = torch.randn(B, S, D, device=device, requires_grad=True)
    x_tok_oj = x_tok.detach().clone().requires_grad_(True)
    targets = torch.full((B, S), -100, dtype=torch.int64, device=device)
    targets[:, -32:] = torch.randint(0, V, (B, 32), device=device)

    with SavedTensorAudit(list(std_llama.parameters()) + [x_tok]) as aud_std_llama:
        loss_std = std_llama(x_tok, targets)
    loss_std.backward()
    std_llama_tape_mb = aud_std_llama.summary()["fresh_saved_storage_bytes"] / (1024 * 1024)

    opt_gf = HolonomicGradFreeOptimizer(oj_llama.parameters(), lr=1e-3, algorithm="adamw")
    with SavedTensorAudit(list(oj_llama.parameters()) + [x_tok_oj]) as aud_oj_llama:
        loss_oj = oj_llama(x_tok_oj, targets)
    loss_oj.backward()
    torch.testing.assert_close(loss_oj, loss_std, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x_tok_oj.grad, x_tok.grad, rtol=1e-3, atol=1e-3)
    oj_llama_tape_mb = aud_oj_llama.summary()["fresh_saved_storage_bytes"] / (1024 * 1024)
    persistent_grad_mb = opt_gf.persistent_grad_bytes() / (1024 * 1024)
    opt_gf.remove_hooks()

    llama_tape_reduction_pct = (1.0 - oj_llama_tape_mb / max(1e-6, std_llama_tape_mb)) * 100.0
    print(f"  -> Llama-3 Block Loss Parity      : Std={loss_std.item():.6f} vs OjasX={loss_oj.item():.6f}")
    print(f"  -> Llama-3 Saved Activation Tape  : {std_llama_tape_mb:.2f} MB -> {oj_llama_tape_mb:.2f} MB ({llama_tape_reduction_pct:.1f}% saved)")
    print(f"  -> Persistent `.grad` Memory      : {persistent_grad_mb:.2f} MB (via HolonomicGradFreeOptimizer)")

    # 4B: LoRA Fine-Tuning via SmartWorkloadRouter
    class LoRAFineTuneModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(512, eps=1e-6)
            self.frozen_qkv = nn.Linear(512, 1536, bias=False)
            self.frozen_qkv.weight.requires_grad_(False)
            self.lora_a = nn.Linear(512, 16, bias=False)
            self.lora_b = nn.Linear(16, 1536, bias=False)
            self.head = nn.Sequential(nn.Linear(1536, 512), nn.GELU(approximate="tanh"))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            return self.head(self.frozen_qkv(h) + self.lora_b(self.lora_a(h)))

    lora_std = LoRAFineTuneModel().to(device)
    lora_smart = copy.deepcopy(lora_std)
    router = SmartWorkloadRouter(tiny_param_threshold=1024)
    lora_smart, router_report = router.optimize_model(lora_smart, batch_tokens=1024)

    x_l1 = torch.randn(4, 256, 512, device=device, requires_grad=True)
    x_l2 = x_l1.detach().clone().requires_grad_(True)
    with SavedTensorAudit(list(lora_std.parameters()) + [x_l1]) as aud_l1:
        lora_std(x_l1).sum().backward()
    with SavedTensorAudit(list(lora_smart.parameters()) + [x_l2]) as aud_l2:
        lora_smart(x_l2).sum().backward()
    torch.testing.assert_close(x_l2.grad, x_l1.grad, rtol=1e-3, atol=1e-3)

    lora_std_mb = aud_l1.summary()["fresh_saved_storage_bytes"] / (1024 * 1024)
    lora_smart_mb = aud_l2.summary()["fresh_saved_storage_bytes"] / (1024 * 1024)
    lora_saved_pct = (1.0 - lora_smart_mb / max(1e-6, lora_std_mb)) * 100.0
    print(f"  -> LoRA Smart Router Tape Memory  : {lora_std_mb:.2f} MB -> {lora_smart_mb:.2f} MB ({lora_saved_pct:.1f}% saved)")

    cleanup_gpu()

    # =========================================================================
    # LEVEL 5: Multi-Laptop P2P Heterogeneous Virtual Unified GPU Mesh Level
    # =========================================================================
    print("\n[LEVEL 5/5] Same-process local-device TCP loopback...")
    nodes = [
        MeshNodeSpec("Laptop_A_Primary", "Apple_Silicon_GPU", 18.0, 14.0, str(device)),
        MeshNodeSpec("Laptop_B_Partner", "Partner_GPU_Node", 8.0, 6.5, "cpu"),
    ]
    mesh_net = nn.Sequential(
        HolonomicLinear(512, 1024, act="silu"),
        HolonomicLinear(1024, 1024, act="gelu_tanh"),
        HolonomicLinear(1024, 512, act="silu"),
        HolonomicLinear(512, 256, act="none"),
    )
    ref_net = copy.deepcopy(mesh_net).to("cpu")
    mesh_engine = HolonomicVirtualMeshEngine(mesh_net, nodes=nodes, compress_boundary=True)

    x_m = torch.randn(4, 128, 512, requires_grad=True)
    x_r = x_m.detach().clone().requires_grad_(True)

    mesh_engine(x_m).square().mean().backward()
    ref_net(x_r).square().mean().backward()

    mesh_cos = F.cosine_similarity(x_m.grad.flatten(), x_r.grad.flatten(), dim=0).item()
    assert mesh_cos > 0.999, f"Loopback gradient qualification failed: {mesh_cos}"
    mesh_comp = mesh_engine.telemetry.compression_ratio
    mesh_pooled_gb = mesh_engine.plan.pooled_memory_gb
    mesh_engine.close()

    loss_std_val = loss_std.item()
    loss_oj_val = loss_oj.item()
    compressed_kb = mesh_engine.telemetry.compressed_wire_bytes / 1024.0
    uncompressed_kb = mesh_engine.telemetry.uncompressed_fp32_bytes / 1024.0
    del std_llama, oj_llama, lora_std, lora_smart, mesh_engine, mesh_net, ref_net
    del x_tok, x_tok_oj, targets, x_l1, x_l2, x_m, x_r

    residual_gpu_mb = cleanup_gpu()
    print(f"  -> Caller-supplied capacity estimate: {mesh_pooled_gb:.1f} GB (local loopback, not measured GPU pooling)")
    print(f"  -> TCP Boundary Wire Compression  : {mesh_comp:.2f}x smaller ({compressed_kb:.1f} KB vs {uncompressed_kb:.1f} KB)")
    print(f"  -> Loopback Grad Cosine Sim       : {mesh_cos:.6f}")
    print(f"  -> Post-Run Residual GPU Memory   : {residual_gpu_mb:.2f} MB")

    scorecard = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local sampled numerical qualification; no full-model, multi-machine or six-vendor hardware certification",
        "device": str(device),
        "source_sha256": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in (
            "tools/run_industrial_all_level_benchmark.py", "ojasx2/holonomic_math_compiler.py",
            "ojasx2/mesh_hal.py", "ojasx2/tile_hal.py")},
        "version": "0.3.0",
        "silicon": silicon.__dict__,
        "level_1_cpp_tilelang_kernels": {
            "opcheck_passed": True,
            "z3_smt_zero_bank_conflicts": True,
            "cross_backend_cpu_metal_bit_exact_mask": cross_backend_bit_exact,
            "metal_gpu_available": silicon.has_mps_gpu,
            "cpp_l1_rmsnorm_speedup_x": round(rmsnorm_speedup, 3),
            "pytorch_rmsnorm_us": round(m_py_rms.median * 1e6, 2),
            "ojasx_cpp_rmsnorm_us": round(m_oj_rms.median * 1e6, 2),
        },
        "level_2_pytorch_official_opinfo": {
            "passed": opinfo_passed,
            "total": opinfo_total,
            "pass_rate_pct": 100.0 * opinfo_passed / opinfo_total,
            "scope": "nn.functional.linear nonempty FP32 samples, five activations, forward and first-order input/weight/bias gradients",
        },
        "level_3_meta_backendbench_parquet": {
            "total_parquet_rows_loaded": total_parquet_rows,
            "sampled_capped_cases_verified": bb_passed,
            "sampled_capped_cases_total": bb_total,
            "scope": "First 25 runnable mm/addmm traces; independent caps M<=512 K,N<=1024; contiguous FP32 synthetic linear/ReLU2",
            "min_gradient_cosine_similarity": round(min_bb_cos, 6),
            "avg_activation_mask_compression_x": round(avg_bb_compression, 2),
        },
        "level_4_industry_model_training": {
            "workload": "Custom LLaMA-style RMSNorm/SwiGLU/residual/loss block without attention or RoPE",
            "dimensions": {"batch": B, "sequence": S, "width": D, "hidden": H, "vocabulary": V},
            "memory_scope": "Fresh autograd-saved storage, excluding caller-supplied inputs and parameters; not peak training memory",
            "llama3_loss_std": round(loss_std_val, 6),
            "llama3_loss_ojasx": round(loss_oj_val, 6),
            "llama3_saved_tape_std_mb": round(std_llama_tape_mb, 3),
            "llama3_saved_tape_ojasx_mb": round(oj_llama_tape_mb, 3),
            "llama3_tape_reduction_pct": round(llama_tape_reduction_pct, 2),
            "persistent_grad_mb": round(persistent_grad_mb, 3),
            "lora_smart_router_saved_pct": round(lora_saved_pct, 2),
        },
        "level_5_multi_laptop_p2p_mesh": {
            "topology": "same-process local loopback",
            "declared_capacity_gb": mesh_pooled_gb,
            "tcp_wire_compression_x": round(mesh_comp, 2),
            "gradient_cosine_similarity": round(mesh_cos, 6),
        },
        "residual_gpu_memory_mb": round(residual_gpu_mb, 3),
    }

    report_path = REPO_ROOT / "reports" / "INDUSTRIAL_ALL_LEVEL_BENCHMARK_SCORE.json"
    report_path.write_text(json.dumps(scorecard, indent=2))
    print("\n" + "=" * 100)
    print(f"SAVED LOCAL SAMPLE QUALIFICATION SCORECARD TO: {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    report = REPO_ROOT / "reports" / "INDUSTRIAL_ALL_LEVEL_BENCHMARK_SCORE.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"status": "running", "generated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    try:
        main()
    except Exception as exc:
        report.write_text(json.dumps({"status": "failed", "error": str(exc), "generated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
        raise
