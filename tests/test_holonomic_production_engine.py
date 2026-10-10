"""
tests/test_holonomic_production_engine.py

Pytest Qualification Suite for the Production OjasX Holonomic Engine (v0.2.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx2
from ojasx2 import (
    HolonomicLinear,
    HolonomicSwiGLU,
    HolonomicGeGLU,
    HolonomicRMSNorm,
    HolonomicLMHead,
    HolonomicRoPE,
    holonomic_rope,
    holonomic_attention_softmax,
     holonomic_rmsnorm,
    HolonomicDispatchMode,
    HolonomicGradFreeOptimizer,
    SavedTensorAudit,
    compile_module_aot_simplified,
    AOTCompilationLedger,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def test_rmsnorm_memory_and_parity() -> None:
    x = torch.randn(128, 512, device=DEVICE, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    g = torch.ones(512, device=DEVICE, requires_grad=True)
    g_ref = g.detach().clone().requires_grad_(True)
    dy = torch.randn(128, 512, device=DEVICE)

    with SavedTensorAudit([x_ref, g_ref]) as a_ref:
        y_ref = F.rms_norm(x_ref, (512,), g_ref, 1e-6)
    y_ref.backward(dy)

    with SavedTensorAudit([x, g]) as a_holo:
        y_holo = holonomic_rmsnorm(x, g, 1e-6)
    y_holo.backward(dy)

    torch.testing.assert_close(y_ref, y_holo, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x_ref.grad, x.grad, rtol=1e-4, atol=1e-5)
    assert a_holo.summary()["fresh_saved_storage_bytes"] < a_ref.summary()["fresh_saved_storage_bytes"] // 100


def test_swiglu_and_geglu_parity() -> None:
    for cls, act_fn in ((HolonomicSwiGLU, F.silu), (HolonomicGeGLU, lambda z: F.gelu(z, approximate="tanh"))):
        torch.manual_seed(12)
        mod = cls(256, 512, bias=False, device=DEVICE)
        x = torch.randn(64, 256, device=DEVICE, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        wg_ref = mod.w_gate.detach().clone().requires_grad_(True)
        wu_ref = mod.w_up.detach().clone().requires_grad_(True)
        dy = torch.randn(64, 512, device=DEVICE)

        y_ref = act_fn(F.linear(x_ref, wg_ref)) * F.linear(x_ref, wu_ref)
        y_ref.backward(dy)

        y_holo = mod(x)
        y_holo.backward(dy)

        torch.testing.assert_close(y_ref, y_holo, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x_ref.grad, x.grad, rtol=1e-3, atol=1e-2)


def test_selective_lm_head() -> None:
    torch.manual_seed(21)
    head = HolonomicLMHead(256, 2000, chunk_rows=64, device=DEVICE)
    h = torch.randn(128, 256, device=DEVICE, requires_grad=True)
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = head.weight.detach().clone().requires_grad_(True)
    targets = torch.full((128,), -100, dtype=torch.int64, device=DEVICE)
    targets[:32] = torch.randint(0, 2000, (32,), device=DEVICE)

    loss_ref = F.cross_entropy(F.linear(h_ref, w_ref), targets, ignore_index=-100)
    loss_ref.backward()

    loss_holo = head(h, targets, ignore_index=-100)
    loss_holo.backward()

    torch.testing.assert_close(loss_ref, loss_holo, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(h_ref.grad, h.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w_ref.grad, head.weight.grad, rtol=1e-4, atol=1e-4)


def test_rope_and_attention_softmax() -> None:
    rope = HolonomicRoPE(head_dim=64, max_seq_len=128, device=DEVICE)
    q = torch.randn(2, 4, 64, 64, device=DEVICE, requires_grad=True)
    with SavedTensorAudit([q, rope.cos_cached, rope.sin_cached]) as audit_rope:
        y = rope(q, seq_len=64)
    y.sum().backward()
    assert audit_rope.summary()["fresh_saved_storage_bytes"] == 0
    assert q.grad is not None

    scores = torch.randn(2, 4, 64, 64, device=DEVICE, requires_grad=True)
    scores_ref = scores.detach().clone().requires_grad_(True)
    dp = torch.randn_like(scores)
    F.softmax(scores_ref, dim=-1).backward(dp)
    holonomic_attention_softmax(scores).backward(dp)
    torch.testing.assert_close(scores_ref.grad, scores.grad, rtol=1e-4, atol=1e-5)


def test_aot_and_dynamo_compiler() -> None:
    class TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(128, eps=1e-6)
            self.wg = nn.Linear(128, 256, bias=False)
            self.wu = nn.Linear(128, 256, bias=False)
            self.wd = nn.Linear(256, 128, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n = self.norm(x)
            return (x + self.wd(F.silu(self.wg(n)) * self.wu(n))).sum()

    torch.manual_seed(55)
    m_e = TinyBlock().to(DEVICE)
    torch.manual_seed(55)
    m_c = TinyBlock().to(DEVICE)
    opt_m = torch.compile(m_c, backend="holonomic")

    x1 = torch.randn(32, 128, device=DEVICE, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    m_e(x1).backward()
    opt_m(x2).backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-3, atol=1e-3)


def test_grad_free_optimizer_and_dispatch_mode() -> None:
    torch.manual_seed(99)
    m_ref = HolonomicLinear(128, 128, act="silu", device=DEVICE)
    torch.manual_seed(99)
    m_gf = HolonomicLinear(128, 128, act="silu", device=DEVICE)

    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3)
    opt_gf = HolonomicGradFreeOptimizer(m_gf.parameters(), lr=1e-3, algorithm="adamw")

    x = torch.randn(32, 128, device=DEVICE)
    with HolonomicDispatchMode() as disp:
        m_ref(x).sum().backward()
        opt_ref.step()
        m_gf(x).sum().backward()

    assert disp.total_dispatched > 0
    assert opt_gf.persistent_grad_bytes() == 0
    torch.testing.assert_close(m_ref.weight, m_gf.weight, rtol=1e-5, atol=1e-5)
    opt_gf.remove_hooks()
    torch.ojasx.empty_cache()


def test_cpp_tile_hal_opcheck_and_z3_smt() -> None:
    smt = ojasx2.verify_xor_swizzle_and_ballot_smt()
    assert smt["zero_bank_conflicts_proven"] is True
    assert smt["cpp_swizzle_bijection_verified"] is True

    x = torch.randn(4, 32, requires_grad=True)
    w = torch.randn(64, 32, requires_grad=True)
    b = torch.randn(64, requires_grad=True)
    rw = torch.randn(32, requires_grad=True)

    torch.library.opcheck(torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd.default, (x, w, b))
    torch.library.opcheck(torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd.default, (x, rw, 1e-6))
    torch.library.opcheck(torch.ops.ojasx_hal.pack_1bit_ballot.default, (x,))


def test_smart_workload_router_lora_frozen_bypass() -> None:
    import copy

    class LoRAModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen_proj = nn.Linear(256, 512, bias=False)
            self.frozen_proj.weight.requires_grad_(False)
            self.tiny = nn.Linear(8, 8, bias=False)
            self.head = nn.Sequential(nn.Linear(512, 64), nn.GELU(approximate="tanh"))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.frozen_proj(x))

    m_eager = LoRAModel().to(DEVICE)
    m_smart = copy.deepcopy(m_eager)
    router = ojasx2.SmartWorkloadRouter(tiny_param_threshold=512)
    m_smart, report = router.optimize_model(m_smart, batch_tokens=128)

    assert report.workload_profile == "LORA_FROZEN_BACKBONE"
    assert report.action_counts.get("SKIP_BACKWARD_FROZEN_WEIGHTS", 0) == 1
    assert report.action_counts.get("NATIVE_PYTORCH_FAST_BYPASS", 0) == 1
    assert report.action_counts.get("OJASX_HOLONOMIC_FUSED", 0) == 1

    x1 = torch.randn(4, 32, 256, device=DEVICE, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    m_eager(x1).sum().backward()
    m_smart(x2).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-4)
    torch.ojasx.empty_cache()


def test_multi_laptop_p2p_virtual_mesh_tcp() -> None:
    import copy

    nodes = [
        ojasx2.MeshNodeSpec("Laptop_A", "Apple_GPU_AMX", 18.0, 14.0, str(DEVICE)),
        ojasx2.MeshNodeSpec("Laptop_B", "Partner_GPU", 8.0, 6.5, "cpu"),
    ]
    model = nn.Sequential(
        ojasx2.HolonomicLinear(128, 256, act="silu"),
        ojasx2.HolonomicLinear(256, 128, act="gelu_tanh"),
        ojasx2.HolonomicLinear(128, 128, act="silu"),
        ojasx2.HolonomicLinear(128, 64, act="none"),
    )
    ref_model = copy.deepcopy(model).to("cpu")
    mesh = ojasx2.HolonomicVirtualMeshEngine(model, nodes=nodes, compress_boundary=True)

    x_m = torch.randn(4, 32, 128, requires_grad=True)
    x_r = x_m.detach().clone().requires_grad_(True)

    mesh(x_m).square().mean().backward()
    ref_model(x_r).square().mean().backward()

    cos = F.cosine_similarity(x_m.grad.flatten(), x_r.grad.flatten(), dim=0).item()
    assert mesh.plan.pooled_memory_gb == 26.0
    assert mesh.telemetry.compression_ratio > 3.5
    assert cos > 0.999
    mesh.close()
    torch.ojasx.empty_cache()


def test_tilelang_metal_simdgroup_and_ballot_opcheck() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "third_party" / "tilelang" / "src" / "metal" / "codegen" / "codegen_metal.cc").exists()

    sched = torch.ops.ojasx_hal.tilelang_query_schedule(1024, 1024, 1024, 4, 3)
    assert list(sched) == [2, 2, 1, 128, 1024, 1, 2]

    msl = torch.ops.ojasx_hal.tilelang_emit_metal_gemm_source(32, 32, 8, 4)
    assert "simdgroup_float8x8" in msl
    assert "simd_ballot" in msl
    assert "rasterization2DMLX" in msl

    x = torch.randn(32, 64, requires_grad=True)
    w = torch.randn(64, 64, requires_grad=True)
    b = torch.randn(64, requires_grad=True)
    torch.library.opcheck(torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd.default, (x, w, b, 4))

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out_tl, _ = torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(x, w, b, 4)
    out_ref = torch.relu(F.linear(x_ref, w_ref, b_ref)).pow(2)

    out_tl.sum().backward()
    out_ref.sum().backward()

    torch.testing.assert_close(out_tl, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w.grad, w_ref.grad, rtol=1e-3, atol=1e-3)


def test_tilelang_universal_all_6_hardware_backends() -> None:
    # 1. Verify chip-adaptive schedules across all 6 hardware families (including iGPU vs dGPU SLM adaptation)
    cuda_sched = ojasx2.tilelang_query_chip_schedule("cuda", 1024, 1024, 1024, smem_kb=164, l2_kb=51200)
    rocm_sched = ojasx2.tilelang_query_chip_schedule("rocm", 1024, 1024, 1024, smem_kb=64, l2_kb=8192)
    sycl_igpu = ojasx2.tilelang_query_chip_schedule("sycl", 1024, 1024, 1024, smem_kb=64, l2_kb=4096)
    sycl_dgpu = ojasx2.tilelang_query_chip_schedule("sycl", 1024, 1024, 1024, smem_kb=128, l2_kb=16384)
    ascend_sched = ojasx2.tilelang_query_chip_schedule("ascend", 1024, 1024, 1024, smem_kb=128, l2_kb=8192)
    cpu_sched = ojasx2.tilelang_query_chip_schedule("cpu", 1024, 1024, 1024, smem_kb=32, l2_kb=4096)
    metal_sched = ojasx2.tilelang_query_chip_schedule("metal", 1024, 1024, 1024, smem_kb=32, l2_kb=4096)

    assert cuda_sched["simd_lanes"] == 32 and cuda_sched["swizzle_bytes"] == 128 and cuda_sched["pipeline_stages"] == 3
    assert rocm_sched["simd_lanes"] == 64 and rocm_sched["smem_bank_bytes"] == 128
    assert sycl_igpu["block_m"] == 32 and sycl_dgpu["block_m"] == 64
    assert ascend_sched["block_m"] == 64 and ascend_sched["pipeline_stages"] == 2
    assert cpu_sched["block_k"] == 32
    assert metal_sched["smem_bank_bytes"] == 16

    # 2. Verify Universal Codegen across all 6 targets
    expected_markers = {
        "cuda": ["tl_templates/cuda/threadblock_swizzle.h", "__ballot_sync(0xffffffffu, is_pos)"],
        "rocm": ["tl_templates/hip/gemm.h", "__ballot(is_pos)"],
        "sycl": ["sycl::ext::oneapi::group_ballot", "sycl::local_accessor"],
        "ascend": ["tl_templates/ascend/gemm.h", "ascend_gemm_l1<"],
        "cpu": ["tl_templates/cpu/gemm.h", "GemmHolonomicRelu2BallotTile<"],
        "metal": ["simdgroup_float8x8", "simd_ballot(is_pos)"],
    }
    for arch, markers in expected_markers.items():
        b_k = 8 if arch == "metal" else 32
        src = ojasx2.tilelang_emit_universal_source(arch, 32, 32, b_k, 4)
        for marker in markers:
            assert marker in src, f"Missing {marker} in {arch} kernel source"

    # 3. Official PyTorch C++ opcheck + Cross-Backend (CPU L1d/L2 NEON vs Metal SIMDGroup GPU) Bit-Exact Mask Parity
    torch.manual_seed(77)
    x_cpu = torch.randn(32, 64, requires_grad=True)
    w_cpu = torch.randn(64, 64, requires_grad=True)
    b_cpu = torch.randn(64, requires_grad=True)

    torch.library.opcheck(
        torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd.default,
        (x_cpu, w_cpu, b_cpu, "cpu", 4),
    )
    torch.library.opcheck(
        torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd.default,
        (x_cpu, w_cpu, b_cpu, "metal", 4),
    )

    x_mtl = x_cpu.detach().clone().requires_grad_(True)
    w_mtl = w_cpu.detach().clone().requires_grad_(True)
    b_mtl = b_cpu.detach().clone().requires_grad_(True)

    out_c, mask_c = ojasx2.tilelang_universal_gemm_relu2(x_cpu, w_cpu, b_cpu, target_arch="cpu")
    out_m, mask_m = ojasx2.tilelang_universal_gemm_relu2(x_mtl, w_mtl, b_mtl, target_arch="metal")

    assert torch.equal(mask_c, mask_m), "1-bit packed ballot masks must match bit-for-bit across CPU and Metal!"
    torch.testing.assert_close(out_c, out_m, rtol=1e-3, atol=1e-3)

    out_c.sum().backward()
    out_m.sum().backward()
    torch.testing.assert_close(x_cpu.grad, x_mtl.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w_cpu.grad, w_mtl.grad, rtol=1e-3, atol=1e-3)

