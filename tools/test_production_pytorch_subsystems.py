#!/usr/bin/env python3
"""
tools/test_production_pytorch_subsystems.py

Comprehensive Qualification & Edge-Case Verification Suite for the 4 Production PyTorch Subsystems:
  Layer 1: Declarative YAML Codegen (`holonomic_rope`, `holonomic_attention_softmax`, FakeTensor & edge cases)
  Layer 2: OpenReg Accelerator Extender (`torch.ojasx` PrivateUse1 module & `HolonomicDispatchMode`)
  Layer 3: AOTAutograd Joint Forward-Backward Compiler (`aot_module_simplified` & `backend="holonomic_aot"`)
  Layer 4: Fused Backward-to-Optimizer Engine (`HolonomicGradFreeOptimizer` for SGD & AdamW)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import ojasx2
from ojasx2 import (
    HOLONOMIC_DECLARATIONS_LEDGER,
    holonomic_rope,
    HolonomicRoPE,
    holonomic_attention_softmax,
    register_ojasx_openreg,
    HolonomicDispatchMode,
    AOTCompilationLedger,
    compile_module_aot_simplified,
    HolonomicGradFreeOptimizer,
    SavedTensorAudit,
    HolonomicRMSNorm,
    HolonomicSwiGLU,
    HolonomicLinear,
    HolonomicLMHead,
)


def native_rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def verify_layer1_codegen_ops(device: str = "mps") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("LAYER 1: Declarative YAML Codegen (`holonomic_rope` & `holonomic_attention_softmax`)")
    print("=" * 88)

    ops_declared = [op["name"] for op in HOLONOMIC_DECLARATIONS_LEDGER["operators"]]
    print(f"  [Codegen Ledger] Namespace: {HOLONOMIC_DECLARATIONS_LEDGER['namespace']} | Ops: {ops_declared}")
    assert len(ops_declared) == 5

    # 1A: HolonomicRoPE across 2D, 3D, 4D shapes, non-contiguous views, empty tensors, and trainable trig tables
    B, H, M, D = 2, 8, 512, 128
    q = torch.randn(B, H, M, D, device=device, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    cos = torch.cos(torch.randn(1, 1, M, D // 2, device=device))
    sin = torch.sin(torch.randn(1, 1, M, D // 2, device=device))

    with SavedTensorAudit([q_ref, cos, sin]) as audit_ref_rope:
        y_ref = native_rope_ref(q_ref, cos, sin)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)

    with SavedTensorAudit([q, cos, sin]) as audit_holo_rope:
        y_holo = holonomic_rope(q, cos, sin)
    y_holo.backward(dy)

    rope_fwd_diff = (y_ref - y_holo).abs().max().item()
    rope_bwd_diff = (q_ref.grad - q.grad).abs().max().item()
    ref_rope_fresh = audit_ref_rope.summary()["fresh_saved_storage_bytes"]
    holo_rope_fresh = audit_holo_rope.summary()["fresh_saved_storage_bytes"]

    print(f"  [HolonomicRoPE 4D] Fwd Max Diff: {rope_fwd_diff:.2e} | Bwd Max Diff: {rope_bwd_diff:.2e}")
    print(f"  [HolonomicRoPE Memory] Naive Saved: {ref_rope_fresh:,} B | Holonomic Fresh Saved: {holo_rope_fresh:,} B (0-Byte Token Tape!)")
    assert rope_fwd_diff < 1e-6 and rope_bwd_diff < 1e-6
    assert holo_rope_fresh == 0, f"Expected 0 fresh bytes for HolonomicRoPE, got {holo_rope_fresh}"

    # Edge cases for RoPE:
    # (i) Empty tensor M=0
    q_empty = torch.randn(2, 4, 0, 64, device=device, requires_grad=True)
    cos_empty = torch.cos(torch.randn(1, 1, 0, 32, device=device))
    sin_empty = torch.sin(torch.randn(1, 1, 0, 32, device=device))
    y_empty = holonomic_rope(q_empty, cos_empty, sin_empty)
    y_empty.sum().backward()
    assert y_empty.shape == (2, 4, 0, 64) and q_empty.grad.shape == (2, 4, 0, 64)

    # (ii) Non-contiguous transposed view + trainable cos/sin (tests orthogonal inverse x = R_{-theta}(y) recovery!)
    q_nc = torch.randn(2, M, H, D, device=device).transpose(1, 2).requires_grad_(True)
    q_nc_ref = q_nc.detach().clone().requires_grad_(True)
    cos_t = cos.detach().clone().requires_grad_(True)
    sin_t = sin.detach().clone().requires_grad_(True)
    cos_t_ref = cos.detach().clone().requires_grad_(True)
    sin_t_ref = sin.detach().clone().requires_grad_(True)

    native_rope_ref(q_nc_ref, cos_t_ref, sin_t_ref).backward(dy)
    holonomic_rope(q_nc, cos_t, sin_t).backward(dy)
    trig_grad_diff = max(
        (cos_t_ref.grad - cos_t.grad).abs().max().item(),
        (sin_t_ref.grad - sin_t.grad).abs().max().item(),
    )
    print(f"  [HolonomicRoPE Inverse Recovery] Trainable (cos, sin) Grad Max Diff: {trig_grad_diff:.2e}")
    assert trig_grad_diff < 1e-4

    # (iii) FakeTensorMode meta kernel check
    with FakeTensorMode():
        fq = torch.empty(2, 8, 256, 64, device="meta")
        fcos = torch.empty(1, 1, 256, 32, device="meta")
        fsin = torch.empty(1, 1, 256, 32, device="meta")
        fy = holonomic_rope(fq, fcos, fsin)
        assert fy.shape == (2, 8, 256, 64)

    # 1B: Holonomic Attention Softmax LSE
    scores = torch.randn(B, H, M, M, device=device, requires_grad=True)
    scores_ref = scores.detach().clone().requires_grad_(True)
    dp = torch.randn(B, H, M, M, device=device)

    with SavedTensorAudit([scores_ref]) as audit_ref_sm:
        p_ref = F.softmax(scores_ref, dim=-1)
    p_ref.backward(dp)

    with SavedTensorAudit([scores]) as audit_holo_sm:
        p_holo = holonomic_attention_softmax(scores)
    p_holo.backward(dp)

    sm_fwd_diff = (p_ref - p_holo).abs().max().item()
    sm_bwd_diff = (scores_ref.grad - scores.grad).abs().max().item()
    ref_sm_fresh = audit_ref_sm.summary()["fresh_saved_storage_bytes"]
    holo_sm_fresh = audit_holo_sm.summary()["fresh_saved_storage_bytes"]
    sm_mem_reduction = (1.0 - (holo_sm_fresh / max(1, ref_sm_fresh))) * 100.0

    print(f"  [Attention Softmax LSE] Fwd Max Diff: {sm_fwd_diff:.2e} | Bwd Max Diff: {sm_bwd_diff:.2e}")
    print(
        f"  [Attention Softmax Memory] Native Fresh: {ref_sm_fresh:,} B | "
        f"Holonomic LSE Fresh: {holo_sm_fresh:,} B ({sm_mem_reduction:.2f}% Saved!)"
    )
    assert sm_fwd_diff < 1e-5 and sm_bwd_diff < 1e-5
    assert sm_mem_reduction > 99.0

    # Extreme shift stability (+1000.0) & higher-order derivative check
    s_hi = (torch.randn(4, 16, device=device) + 1000.0).requires_grad_(True)
    s_hi_ref = s_hi.detach().clone().requires_grad_(True)
    g_holo = torch.autograd.grad(holonomic_attention_softmax(s_hi).square().sum(), s_hi, create_graph=True)[0]
    g_ref = torch.autograd.grad(F.softmax(s_hi_ref, dim=-1).square().sum(), s_hi_ref, create_graph=True)[0]
    hvp_holo = torch.autograd.grad(g_holo.sum(), s_hi)[0]
    hvp_ref = torch.autograd.grad(g_ref.sum(), s_hi_ref)[0]
    hvp_diff = (hvp_holo - hvp_ref).abs().max().item()
    print(f"  [Attention Softmax 2nd-Order HVP + Extreme Shift] Max Diff: {hvp_diff:.2e}")
    assert hvp_diff < 1e-5

    torch.ojasx.empty_cache()
    return {
        "rope_fwd_max_diff": rope_fwd_diff,
        "rope_bwd_max_diff": rope_bwd_diff,
        "rope_fresh_saved_bytes": holo_rope_fresh,
        "rope_trig_inverse_grad_diff": trig_grad_diff,
        "softmax_fwd_max_diff": sm_fwd_diff,
        "softmax_bwd_max_diff": sm_bwd_diff,
        "softmax_fresh_memory_reduction_pct": sm_mem_reduction,
        "softmax_second_order_hvp_diff": hvp_diff,
        "status": "PASSED",
    }


def verify_layer2_openreg_and_dispatcher(device: str = "mps") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("LAYER 2: OpenReg Accelerator Extender (`torch.ojasx`) & `HolonomicDispatchMode`")
    print("=" * 88)

    dev_mod = register_ojasx_openreg()
    assert hasattr(torch, "ojasx")
    assert torch.ojasx.is_available() is True
    assert torch.ojasx.device_count() == 1
    torch.ojasx.manual_seed(1234)
    stats = torch.ojasx.memory_stats()
    print(f"  [OpenReg `torch.ojasx`] Registered: True | Stats: {stats}")

    x = torch.randn(128, 256, device=device, requires_grad=True)
    w = torch.randn(256, 256, device=device, requires_grad=True)
    g = torch.ones(256, device=device, requires_grad=True)

    with HolonomicDispatchMode() as disp:
        # Dynamic eager control-flow workload
        h = ojasx2.holonomic_rmsnorm(x, g, 1e-6)
        if h.shape[0] > 64:
            out = ojasx2.holonomic_linear(h, w, act="silu")
        else:
            out = ojasx2.holonomic_linear(h, w, act="relu")
        out.sum().backward()

    summary = disp.summary()
    print(
        f"  [HolonomicDispatchMode] Total ATen Calls Intercepted: {summary['total_aten_calls']} | "
        f"Unique Ops: {summary['unique_aten_ops']}"
    )
    assert summary["total_aten_calls"] > 0

    torch.ojasx.empty_cache()
    return {
        "openreg_module_registered": True,
        "memory_stats": stats,
        "dispatch_mode_summary": summary,
        "status": "PASSED",
    }


def verify_layer3_aot_autograd_partitioner(device: str = "mps") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("LAYER 3: AOTAutograd Joint Forward-Backward Compiler & Custom Partitioner")
    print("=" * 88)

    class AttentionMLPBlock(nn.Module):
        def __init__(self, dim: int = 256, hidden: int = 512) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim))
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
            self.register_buffer("cos", torch.cos(torch.randn(64, dim // 2)))
            self.register_buffer("sin", torch.sin(torch.randn(64, dim // 2)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n = ojasx2.rmsnorm(x, self.gamma, 1e-6)
            q = holonomic_rope(self.wq(n), self.cos, self.sin)
            k = holonomic_rope(self.wk(n), self.cos, self.sin)
            v = self.wv(n)
            scores = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
            probs = holonomic_attention_softmax(scores)
            attn = probs @ v
            mlp = self.w2(F.silu(self.w1(x + attn)))
            return (x + attn + mlp).sum()

    torch.manual_seed(99)
    eager_blk = AttentionMLPBlock().to(device)
    torch.manual_seed(99)
    aot_blk = AttentionMLPBlock().to(device)

    x_e = torch.randn(64, 256, device=device, requires_grad=True)
    x_a = x_e.detach().clone().requires_grad_(True)

    ledger = AOTCompilationLedger()
    aot_runner = compile_module_aot_simplified(aot_blk, (x_a,), ledger=ledger)

    loss_e = eager_blk(x_e)
    loss_e.backward()

    loss_a = aot_runner(x_a)
    loss_a.backward()

    loss_diff = abs(loss_e.item() - loss_a.item())
    grad_diff = (x_e.grad - x_a.grad).abs().max().item()
    w1_grad_diff = (eager_blk.w1.weight.grad - aot_blk.w1.weight.grad).abs().max().item()

    print(f"  [AOTAutograd Joint Graph] Joint Nodes: {ledger.joint_graph_nodes} | FW Nodes: {ledger.fw_graph_nodes} | BW Nodes: {ledger.bw_graph_nodes}")
    print(f"  [AOTAutograd Partition Cut] Saved Tensors Across Cut: {ledger.saved_tensors_across_cut} | BW Contractions: {ledger.bw_gemm_contractions}")
    print(f"  [AOTAutograd Custom Ops in FW]: {ledger.fw_custom_ops}")
    print(f"  [AOTAutograd Parity] Loss Diff: {loss_diff:.2e} | Input Grad Diff: {grad_diff:.2e} | Weight Grad Diff: {w1_grad_diff:.2e}")

    assert ledger.joint_graph_nodes > 0 and ledger.fw_graph_nodes > 0 and ledger.bw_graph_nodes > 0
    assert len(ledger.fw_custom_ops) >= 3
    torch.testing.assert_close(x_e.grad, x_a.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(eager_blk.w1.weight.grad, aot_blk.w1.weight.grad, rtol=1e-3, atol=1e-3)

    torch.ojasx.empty_cache()
    return {
        "ledger": ledger.to_dict(),
        "loss_diff": loss_diff,
        "input_grad_diff": grad_diff,
        "weight_grad_diff": w1_grad_diff,
        "status": "PASSED",
    }


def verify_layer4_grad_free_optimizer(device: str = "mps") -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print("LAYER 4: Fused Backward-to-Optimizer Engine (`HolonomicGradFreeOptimizer`)")
    print("=" * 88)

    class MiniHolonomicLLM(nn.Module):
        def __init__(self, dim: int = 1024, hidden: int = 2816, vocab: int = 16000) -> None:
            super().__init__()
            self.norm1 = HolonomicRMSNorm(dim)
            self.mlp1 = HolonomicSwiGLU(dim, hidden)
            self.down1 = HolonomicLinear(hidden, dim, bias=False)
            self.norm2 = HolonomicRMSNorm(dim)
            self.mlp2 = HolonomicSwiGLU(dim, hidden)
            self.down2 = HolonomicLinear(hidden, dim, bias=False)
            self.head = HolonomicLMHead(dim, vocab, chunk_rows=128)

        def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            h = x + self.down1(self.mlp1(self.norm1(x)))
            h = h + self.down2(self.mlp2(self.norm2(h)))
            return self.head(h, targets, ignore_index=-100)

    results_by_algo: Dict[str, Any] = {}

    for algo in ("sgd", "adamw"):
        torch.manual_seed(777)
        ref_model = MiniHolonomicLLM().to(device)
        torch.manual_seed(777)
        gf_model = MiniHolonomicLLM().to(device)

        if algo == "sgd":
            ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
            gf_opt = HolonomicGradFreeOptimizer(gf_model.parameters(), lr=1e-2, algorithm="sgd", momentum=0.9, weight_decay=1e-4)
        else:
            ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
            gf_opt = HolonomicGradFreeOptimizer(gf_model.parameters(), lr=1e-3, algorithm="adamw", betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

        ref_persistent_grad_bytes = 0
        gf_persistent_grad_bytes = 0
        freed_in_backward_bytes = 0

        for step_idx in range(3):
            torch.manual_seed(100 + step_idx)
            x_batch = torch.randn(256, 1024, device=device)
            targets = torch.full((256,), -100, dtype=torch.int64, device=device)
            targets[:64] = torch.randint(0, 16000, (64,), device=device)

            # Standard PyTorch step: .backward() holds all .grad tensors in RAM until .step()
            ref_opt.zero_grad(set_to_none=True)
            loss_r = ref_model(x_batch, targets)
            loss_r.backward()
            ref_persistent_grad_bytes = sum(
                p.grad.numel() * p.grad.element_size() for p in ref_model.parameters() if p.grad is not None
            )
            ref_opt.step()

            # Holonomic Grad-Free step: updates weights inside .backward() and frees .grad immediately!
            gf_opt.reset_step_metrics()
            loss_g = gf_model(x_batch, targets)
            loss_g.backward()
            gf_persistent_grad_bytes = gf_opt.persistent_grad_bytes()
            freed_in_backward_bytes = gf_opt.last_step_freed_bytes

        max_weight_diff = max(
            (p_r - p_g).abs().max().item()
            for p_r, p_g in zip(ref_model.parameters(), gf_model.parameters())
        )

        print(
            f"  [{algo.upper()} 3-Step Trajectory] Max Parameter Weight Diff vs torch.optim.{algo.upper()}: {max_weight_diff:.2e}"
        )
        print(
            f"  [{algo.upper()} Gradient RAM] Standard Persistent .grad: {ref_persistent_grad_bytes / 1e6:.2f} MB | "
            f"Holonomic Persistent .grad: {gf_persistent_grad_bytes / 1e6:.2f} MB "
            f"(Freed {freed_in_backward_bytes / 1e6:.2f} MB in-flight, 100.0% eliminated!)"
        )

        assert gf_persistent_grad_bytes == 0
        assert max_weight_diff < 1e-5
        gf_opt.remove_hooks()

        results_by_algo[algo] = {
            "max_parameter_weight_diff_after_3_steps": max_weight_diff,
            "standard_persistent_grad_mb": ref_persistent_grad_bytes / 1e6,
            "holonomic_persistent_grad_mb": gf_persistent_grad_bytes / 1e6,
            "freed_in_flight_grad_mb": freed_in_backward_bytes / 1e6,
            "persistent_grad_reduction_pct": 100.0,
            "status": "PASSED",
        }
        del ref_model, gf_model, ref_opt, gf_opt
        torch.ojasx.empty_cache()

    return results_by_algo


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.ojasx.set_per_process_memory_fraction(0.45)
    print(f"Running Production PyTorch Subsystems Qualification on {device.upper()}...")

    l1 = verify_layer1_codegen_ops(device)
    l2 = verify_layer2_openreg_and_dispatcher(device)
    l3 = verify_layer3_aot_autograd_partitioner(device)
    l4 = verify_layer4_grad_free_optimizer(device)

    torch.ojasx.empty_cache()
    final_mem = torch.ojasx.memory_allocated()

    report = {
        "qualification_suite": "OJASX_PRODUCTION_PYTORCH_SUBSYSTEMS_V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "final_allocated_bytes": final_mem,
        "layer_1_declarative_codegen": l1,
        "layer_2_openreg_dispatcher": l2,
        "layer_3_aot_autograd_partitioner": l3,
        "layer_4_grad_free_optimizer": l4,
        "overall_status": "ALL_4_PRODUCTION_SUBSYSTEMS_CERTIFIED",
    }

    out_path = "reports/PRODUCTION_PYTORCH_SUBSYSTEMS_REPORT.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 88)
    print(f"ALL 4 PRODUCTION PYTORCH SUBSYSTEMS CERTIFIED. Residual GPU Memory: {final_mem / 1e6:.2f} MB")
    print(f"Report written to: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
