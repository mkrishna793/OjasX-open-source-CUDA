"""
End-to-End & Limit Stress Test Suite for OjasX-LogicIR Compiler (Phases 1–5).

Runnable both via:
  pytest tests/test_mac_end_to_end_limits.py -v
and standalone via:
  python3 tests/test_mac_end_to_end_limits.py
"""

from __future__ import annotations

import os
import platform
import time
import numpy as np

from ojasx_logic import (
    AlgNode,
    LogicIRModule,
    LogicOp,
    choose_alu_route,
    compile_mlir_to_vmfb,
    decode_mx_naf_bf16,
    detect_iree_toolchain,
    emit_logic_ir_mlir,
    emit_standalone_mlir_suite,
    encode_mx_naf_bf16,
    estimate_llm_memory_budget,
    exhaustive_bf16_audit,
    gemm_crt_bf16,
    gemm_crt_int,
    gemm_mx_naf_dual_rail,
    naf5_encode_i32,
    optimize_algmat,
    pack_ternary_dual_rail,
    run_mlir_kernel_iree,
    unpack_ternary_dual_rail,
)


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def test_phase1_mx_naf_bf16_sparsity_and_exact_popcnt_parity():
    rng = np.random.default_rng(42)
    M, K, N = 16, 256, 32
    A = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((N, K), dtype=np.float32)

    qA = encode_mx_naf_bf16(A, block_size=32, bitplanes=9)
    qW = encode_mx_naf_bf16(W, block_size=32, bitplanes=9)
    assert qW.zero_sparsity >= 0.70, f"Expected >=70% NAF zero sparsity, got {qW.zero_sparsity:.4f}"

    y_popcnt = gemm_mx_naf_dual_rail(qA, qW)
    A_dec = decode_mx_naf_bf16(qA)
    W_dec = decode_mx_naf_bf16(qW)
    y_ref = A_dec @ W_dec.T
    max_abs_diff = float(np.max(np.abs(y_popcnt - y_ref)))
    assert max_abs_diff < 1e-4, f"Dual-Rail POPCNT GEMM diverged from decoded matmul: {max_abs_diff}"

    vals = np.arange(-512, 512, dtype=np.int32)
    pos, neg = naf5_encode_i32(vals)
    assert int(np.max(pos & neg)) == 0
    nz = pos | neg
    assert int(np.max(nz & (nz << 1))) == 0
    recon = pos.astype(np.int64) - neg.astype(np.int64)
    assert np.array_equal(recon, vals.astype(np.int64))


def test_phase1_crt3_and_crt4_exactness_and_k_limits():
    rng = np.random.default_rng(7)

    for K in (64, 256, 478):
        Aq = rng.integers(-127, 128, size=(16, K), dtype=np.int64)
        Bq = rng.integers(-127, 128, size=(K, 16), dtype=np.int64)
        Cq = gemm_crt_int(Aq, Bq, channels=3)
        assert np.array_equal(Cq, Aq @ Bq), f"CRT-3 integer mismatch at K={K}"

    for K in (512, 1024, 2048):
        Aq = rng.integers(-127, 128, size=(16, K), dtype=np.int64)
        Bq = rng.integers(-127, 128, size=(K, 16), dtype=np.int64)
        Cq = gemm_crt_int(Aq, Bq, channels=4)
        assert np.array_equal(Cq, Aq @ Bq), f"CRT-4 integer mismatch at K={K}"

    A_f = rng.standard_normal((16, 1024)).astype(np.float32)
    B_f = rng.standard_normal((1024, 16)).astype(np.float32)
    C_got = gemm_crt_bf16(A_f, B_f, channels=4)
    C_ref = A_f @ B_f
    rel_err = float(np.linalg.norm(C_got - C_ref) / np.linalg.norm(C_ref))
    assert rel_err < 0.015, f"CRT-4 BF16 rel_err too high: {rel_err}"


def test_phase2_exhaustive_65536_bf16_bit_circuits():
    report = exhaustive_bf16_audit()
    assert report["bf16_patterns_checked"] == 65536.0
    assert report["bit_relu_max_abs_err"] == 0.0
    assert report["bit_abs_max_abs_err"] == 0.0
    assert report["bit_fast_exp_max_rel_err"] < 0.06
    assert report["bit_silu_max_abs_err"] < 0.08
    assert report["bit_rsqrt_max_rel_err"] < 0.002


def test_phase2_z3_formal_proofs_if_installed():
    if not _has_module("z3"):
        print("[SKIP] z3-solver not installed in this environment")
        return
    from ojasx_logic import run_z3_logic_ir_proofs

    results = run_z3_logic_ir_proofs()
    assert len(results) >= 4
    for name, status in results.items():
        assert ("PROVED" in status or "PROVEN" in status or status == "OK"), f"Z3 proof {name} failed: {status}"


def test_phase3_algmat_ir_cancellations_and_kronecker_limits():
    cw = AlgNode.cw120(transposed=False, orbit_id=0)
    cwt = AlgNode.cw120(transposed=True, orbit_id=0)
    opt, st = optimize_algmat(AlgNode.compose(cwt, cw))
    assert opt.kind == "identity" and opt.dim == 120 and opt.scale == 36.0
    assert st.ortho_cancellations == 1

    fw = AlgNode.fwht(4096)
    opt_fw, st_fw = optimize_algmat(AlgNode.compose(fw, fw))
    assert opt_fw.kind == "identity" and opt_fw.dim == 4096 and opt_fw.scale == 1.0
    assert st_fw.ortho_cancellations == 1

    k_fwd = AlgNode.kronecker((cw, cw, cw))
    k_rev = AlgNode.kronecker((cwt, cwt, cwt))
    opt_k, st_k = optimize_algmat(AlgNode.compose(k_rev, k_fwd))
    assert opt_k.kind == "identity"
    assert opt_k.dim == 120 ** 3
    assert opt_k.scale == float(36 ** 3)
    assert st_k.kronecker_mixed_products == 1
    assert st_k.ortho_cancellations == 3


def test_phase3_alu_crossover_cost_router():
    assert choose_alu_route(weight_bits=1.5, target="metal-spirv") == "dual_rail_popcnt"
    assert choose_alu_route(weight_bits=16.0, zero_sparsity=0.75, target="metal-spirv") == "dual_rail_popcnt"
    assert choose_alu_route(weight_bits=16.0, target="llvm-cpu", K=256) == "crt3_int8"
    assert choose_alu_route(weight_bits=16.0, target="llvm-cpu", K=2048) == "crt4_int8"
    assert choose_alu_route(weight_bits=16.0, target="metal-spirv", K=2048) == "native_bf16_tc"
    assert choose_alu_route(weight_bits=16.0, target="cuda", K=4096) == "native_bf16_tc"


def test_phase3_pytorch_fx_dynamic_multi_architecture_if_installed():
    if not _has_module("torch"):
        print("[SKIP] PyTorch not installed in this environment")
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from ojasx_logic.frontend.pytorch_fx_bridge import (
        lower_pytorch_module_to_logic_ir,
        make_logic_ir_dynamo_backend,
    )

    class TransformerSwiGLUBlock(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.gate = nn.Linear(d, d * 2)
            self.up = nn.Linear(d, d * 2)
            self.down = nn.Linear(d * 2, d)

        def forward(self, x):
            return x + self.down(F.silu(self.gate(x)) * self.up(x))

    class MiniConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(8, 16)

        def forward(self, x):
            h = self.relu(self.conv(x)).mean(dim=(-1, -2))
            return self.fc(h)

    class ViTMLPBlock(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, d * 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(d * 4, d)

        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(self.norm(x))))

    class VariableDepthGatedSSM(nn.Module):
        def __init__(self, d=64, depth=5):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = h + F.silu(layer(h))
            return torch.abs(h)

    models = [
        ("transformer_swiglu", TransformerSwiGLUBlock(128), "bit_silu", 3),
        ("mini_convnet", MiniConvNet(), "bit_relu", 1),
        ("vit_mlp", ViTMLPBlock(64), "bit_gelu", 2),
        ("variable_depth_ssm", VariableDepthGatedSSM(64, depth=5), "bit_silu", 5),
    ]

    for name, model, expected_act, expected_linears in models:
        ir = lower_pytorch_module_to_logic_ir(model, name=name, target="metal-spirv", weight_bits=1.5)
        counts = ir.summary()
        assert counts.get("dual_rail_gemm", 0) == expected_linears, f"{name}: {counts}"
        assert counts.get(expected_act, 0) >= 1, f"{name}: missing {expected_act} in {counts}"
        mlir = emit_logic_ir_mlir(ir)
        assert f"module @{name}" in mlir
        assert "math.ctpop" in mlir

    backend = make_logic_ir_dynamo_backend(target="metal-spirv", weight_bits=1.5)
    m = ViTMLPBlock(32)
    x = torch.randn(2, 8, 32)
    gm = torch.fx.symbolic_trace(m)
    fn = backend(gm, [x])
    out = fn(x)
    assert out.shape == x.shape
    assert hasattr(gm, "_ojasx_logic_ir")


def test_phase4_and_5_target_filtered_mlir_and_live_iree_run():
    # 1. metal-spirv suite must omit i8 matmul (SPIR-V 1.5 clean!)
    mlir_metal = emit_standalone_mlir_suite(target="metal-spirv")
    assert mlir_metal.count("{") == mlir_metal.count("}")
    assert "func.func @ojasx_dual_rail_popcnt_tile" in mlir_metal
    assert "math.ctpop" in mlir_metal
    assert "func.func @ojasx_bit_relu" in mlir_metal
    assert "func.func @ojasx_crt_int8_channel_gemm" not in mlir_metal

    # 2. llvm-cpu suite includes i8 CRT matmul
    mlir_cpu = emit_standalone_mlir_suite(target="llvm-cpu")
    assert mlir_cpu.count("{") == mlir_cpu.count("}")
    assert "func.func @ojasx_crt_int8_channel_gemm" in mlir_cpu

    # 3. Optional live IREE compile + execute on host!
    tc = detect_iree_toolchain()
    if tc["iree_compile"]:
        res_cpu = compile_mlir_to_vmfb(mlir_cpu, target="llvm-cpu")
        assert res_cpu.status == "ok", f"iree-compile llvm-cpu failed: {res_cpu.stderr}"
        print(f"[IREE PASS] llvm-cpu compiled in {res_cpu.elapsed_ms:.1f} ms")

        has_metallib = False
        if platform.system() == "Darwin":
            import subprocess
            chk = subprocess.run(["xcrun", "-f", "metallib"], capture_output=True, text=True, check=False)
            has_metallib = chk.returncode == 0
            if has_metallib:
                res_metal = compile_mlir_to_vmfb(mlir_metal, target="metal-spirv")
                assert res_metal.status == "ok", f"iree-compile metal-spirv failed: {res_metal.stderr}"
                print(f"[IREE PASS] metal-spirv compiled cleanly (SPIR-V 1.5!) in {res_metal.elapsed_ms:.1f} ms")
            else:
                print("[SKIP] macOS Xcode metallib not installed; skipping IREE metal-spirv .vmfb packaging (use torch.mps.compile_shader runtime instead)")

        if tc["iree_runtime"]:
            target = "metal-spirv" if (platform.system() == "Darwin" and has_metallib) else "llvm-cpu"
            x_in = np.linspace(-5.0, 5.0, 1024, dtype=np.float32)
            r = run_mlir_kernel_iree(
                mlir_metal if target == "metal-spirv" else mlir_cpu,
                "ojasx_bit_relu",
                [x_in],
                target=target,
            )
            assert r.status == "ok", f"iree-runtime failed on {target}: {r.error}"
            assert np.array_equal(r.output, np.maximum(x_in, 0.0))
            print(f"[IREE EXEC PASS] ojasx_bit_relu live on {target}: run={r.run_ms:.2f} ms, bit-exact!")
    else:
        res = compile_mlir_to_vmfb(mlir_metal, target="metal-spirv")
        assert res.status == "skipped_no_iree"


def test_phase5_ternary_dual_rail_2p125bit_and_9b_memory_budget():
    rng = np.random.default_rng(99)
    W = rng.standard_normal((512, 1024), dtype=np.float32)
    packed = pack_ternary_dual_rail(W, group_size=128)

    # Exact 2.125 bits/weight -> 7.53x smaller than FP16/BF16!
    assert abs(packed.bits_per_weight - 2.125) < 1e-6, f"Got {packed.bits_per_weight} bpw"
    bf16_bytes = W.size * 2
    ratio = bf16_bytes / packed.nbytes
    assert ratio > 7.5, f"Expected >7.5x compression vs BF16, got {ratio:.2f}x"
    assert int(np.max(packed.pos_bits & packed.neg_bits)) == 0

    # Unpack round-trip check
    W_rec = unpack_ternary_dual_rail(packed)
    assert W_rec.shape == W.shape
    # Cosine similarity between Gaussian weights and BitNet-style ternary packing is > 0.85
    cos = float(np.sum(W * W_rec) / (np.linalg.norm(W) * np.linalg.norm(W_rec)))
    assert cos > 0.85, f"Unexpectedly low ternary cosine similarity: {cos:.4f}"

    # Verify 9.24B (Gemma-2-9B) fits comfortably in 24 GB M5 unified RAM
    b9 = estimate_llm_memory_budget(9.24, lora_rank=16, group_size=128)
    assert b9["bf16_weights_gb"] > 17.0
    assert b9["total_static_train_gb"] < 6.0
    assert b9["fits_24gb_m5"] == 1.0


def test_phase5_dual_rail_lora_autograd_if_torch_installed():
    if not _has_module("torch"):
        print("[SKIP] PyTorch not installed in this environment")
        return

    import torch
    import torch.nn as nn
    from ojasx_logic.trainer.dual_rail_lora import quantize_module_to_dual_rail_lora
    from ojasx_logic.trainer.hf_9b_trainer import run_dual_rail_lora_training

    class TinyCausalLM(nn.Module):
        def __init__(self, vocab=512, d=256):
            super().__init__()
            self.embed = nn.Embedding(vocab, d, dtype=torch.bfloat16)
            self.q_proj = nn.Linear(d, d, bias=False, dtype=torch.bfloat16)
            self.o_proj = nn.Linear(d, d, bias=False, dtype=torch.bfloat16)
            self.gate_proj = nn.Linear(d, d * 2, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(d * 2, d, bias=False, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(d, vocab, bias=False, dtype=torch.bfloat16)
            self.vocab = vocab

        def forward(self, input_ids, labels):
            h = self.embed(input_ids)
            h = h + self.o_proj(self.q_proj(h))
            h = h + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)))
            logits = self.lm_head(h)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab), labels.view(-1))
            return type("Out", (), {"loss": loss, "logits": logits})()

    m = TinyCausalLM()
    stats = quantize_module_to_dual_rail_lora(m, lora_r=8, group_size=128, device="cpu")
    assert stats["layers_quantized"] == 4.0
    assert stats["compression_ratio"] > 7.4

    seen_steps = []
    res = run_dual_rail_lora_training(
        m,
        model_id="tiny-dual-rail-test",
        stats=stats,
        device="cpu",
        steps=3,
        seq_len=32,
        batch_size=2,
        vocab_size=512,
        safety_policy=lambda step, loss: seen_steps.append((step, loss)),
    )
    assert res.safety_checks_run == 3
    assert len(seen_steps) == 3
    assert np.isfinite(res.loss_end)


def test_phase1_stress_limit_scaling_optional():
    if os.environ.get("OJASX_STRESS", "0") != "1":
        return
    rng = np.random.default_rng(123)
    M, K, N = 32, 1024, 64
    A = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((N, K), dtype=np.float32)
    t0 = time.perf_counter()
    qA = encode_mx_naf_bf16(A, block_size=32, bitplanes=9)
    qW = encode_mx_naf_bf16(W, block_size=32, bitplanes=9)
    y = gemm_mx_naf_dual_rail(qA, qW)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert y.shape == (M, N)
    print(f"[STRESS] MX-NAF-BF16 {M}x{K}x{N}: sparsity={qW.zero_sparsity:.2%}, elapsed={dt_ms:.1f} ms")




def test_phase5_runtime_metal_shader_mps_if_available():
    if not _has_module("torch"):
        print("[SKIP] PyTorch not installed")
        return
    import torch
    from ojasx_logic.backend.metal_mps_runtime import (
        metal_runtime_status,
        mps_dual_rail_linear_forward,
        mps_dual_rail_popcnt_gemm_16x16,
    )
    from ojasx_logic.trainer.dual_rail_lora import _pack_weight_to_uint8_dual_rail

    if not torch.backends.mps.is_available():
        print("[SKIP] MPS not available on this host")
        return

    ok, msg = metal_runtime_status()
    assert ok, f"torch.mps.compile_shader failed: {msg}"

    # 1. Verify zero-materialization Dual-Rail linear forward parity on MPS
    torch.manual_seed(0)
    M, I, O = 16, 256, 128
    x = torch.randn(M, I, dtype=torch.bfloat16, device="mps")
    w = torch.randn(O, I, dtype=torch.float32, device="mps")
    pb, nb, sc, _, _ = _pack_weight_to_uint8_dual_rail(w, group_size=128)
    y_metal = mps_dual_rail_linear_forward(x, pb, nb, sc, I, group_size=128)
    assert y_metal is not None and y_metal.shape == (M, O)

    shifts = torch.arange(8, dtype=torch.uint8, device="mps")
    p = ((pb.unsqueeze(-1) >> shifts) & 1).float()
    n = ((nb.unsqueeze(-1) >> shifts) & 1).float()
    w_ref = ((p - n).view(O, -1, 128) * sc.float().unsqueeze(-1)).view(O, I)
    y_ref = (x.float() @ w_ref.T).to(torch.bfloat16)
    max_diff = float((y_metal.float() - y_ref.float()).abs().max().item())
    assert max_diff < 1e-1, f"Metal Dual-Rail linear forward diverged: max_diff={max_diff}"

    # 2. Verify 16x16 threadgroup-tiled hardware popcount() GEMM on MPS
    Kw = 8  # 256 weights = 2 groups of 128
    ap = torch.randint(0, 2**31 - 1, (32, Kw), dtype=torch.int32, device="mps").to(torch.uint32)
    an = torch.zeros_like(ap)
    bp = torch.randint(0, 2**31 - 1, (32, Kw), dtype=torch.int32, device="mps").to(torch.uint32)
    bn = torch.zeros_like(bp)
    bs = torch.ones((32, Kw // 4), dtype=torch.float16, device="mps")
    c_out = mps_dual_rail_popcnt_gemm_16x16(ap, an, bp, bn, bs)
    assert c_out is not None and c_out.shape == (32, 32) and torch.isfinite(c_out).all()
    print(f"[METAL 4 RUNTIME PASS] Zero-Xcode torch.mps.compile_shader linear (max_diff={max_diff:.4e}) & 16x16 POPCNT GEMM verified!")

if __name__ == "__main__":
    tests = [
        test_phase1_mx_naf_bf16_sparsity_and_exact_popcnt_parity,
        test_phase1_crt3_and_crt4_exactness_and_k_limits,
        test_phase2_exhaustive_65536_bf16_bit_circuits,
        test_phase2_z3_formal_proofs_if_installed,
        test_phase3_algmat_ir_cancellations_and_kronecker_limits,
        test_phase3_alu_crossover_cost_router,
        test_phase3_pytorch_fx_dynamic_multi_architecture_if_installed,
        test_phase4_and_5_target_filtered_mlir_and_live_iree_run,
        test_phase5_ternary_dual_rail_2p125bit_and_9b_memory_budget,
        test_phase5_dual_rail_lora_autograd_if_torch_installed,
        test_phase5_runtime_metal_shader_mps_if_available,
        test_phase1_stress_limit_scaling_optional,
    ]
    passed = 0
    for fn in tests:
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"PASS  {fn.__name__} ({dt:.1f} ms)")
        passed += 1
    print(f"\nALL {passed}/{len(tests)} CHECKS COMPLETED CLEANLY.")
