<h1 align="center">⚡ O J A S X (v0.3.0)</h1>

<p align="center">
  <strong>The Universal AI Compute, Holonomic Mathematical & Multi-Backend C++ Tile Compiler</strong><br/>
  <em>"Ojas" (ओजस्) — The vital energy that powers all creation.<br/>We channel that energy through every GPU, NPU, and CPU on Earth.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge" alt="AGPL-3.0"/>
  <img src="https://img.shields.io/badge/Version-0.3.0-orange?style=for-the-badge" alt="v0.3.0"/>
  <img src="https://img.shields.io/badge/Tests-369%2F369_Passing-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/Hardware-CUDA_%7C_ROCm_%7C_SYCL_%7C_Ascend_%7C_Metal_%7C_CPU-purple?style=for-the-badge" alt="Any GPU"/>
  <img src="https://img.shields.io/badge/PyTorch-2.8%2B-red?style=for-the-badge" alt="PyTorch"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA_CUDA-sm__80%2Fsm__90-76b900?style=flat-square&logo=nvidia" alt="NVIDIA"/>
  <img src="https://img.shields.io/badge/AMD_ROCm-gfx90a%2Fgfx942-ed1c24?style=flat-square&logo=amd" alt="AMD"/>
  <img src="https://img.shields.io/badge/Intel_oneAPI_SYCL-iGPU_%26_Arc_XPU-0071c5?style=flat-square&logo=intel" alt="Intel"/>
  <img src="https://img.shields.io/badge/Apple_Silicon-Metal_4_%2B_AMX-000000?style=flat-square&logo=apple" alt="Apple"/>
  <img src="https://img.shields.io/badge/Huawei_Ascend-910B_Cube%2BUB-cf0a2c?style=flat-square" alt="Ascend"/>
  <img src="https://img.shields.io/badge/Formal_Proofs-Lean_4_%2B_Rocq_%2B_Z3_SMT-blueviolet?style=flat-square" alt="Formal Proofs"/>
</p>

<p align="center">
  <img src="assets/breaking_monopoly.png" alt="Breaking the Monopoly" width="600"/>
</p>

---

## 💡 Overview

**OjasX (`ojasx2`)** is a formally verified **Holonomic Mathematical & Hardware-Aware C++ Tile Compiler** for PyTorch 2.x designed to break proprietary hardware lock-in and enable training and deploying large AI models on consumer GPUs, integrated GPUs (`iGPU`), unified-memory laptops, and heterogeneous multi-laptop peer-to-peer clusters.

Unlike simple wrapper libraries, **OjasX** combines four deep innovations in a single drop-in stack:

1. **Holonomic Reverse-Mode Mathematical Compiler (`ojasx2.holonomic_math_compiler`, `ojasx2.aot_holonomic_compiler`)**:
   - Eliminates intermediate activation tensor storage in DRAM/VRAM via category-theoretic Dagger / Lens pullbacks, radial invariance ($O(1)$ 4-byte scalar $\mu_2$ for `RMSNorm`), `1-bit` hardware ballot sign packing (`32×` compression vs FP32), and selective vocabulary streaming (`HolonomicLMHead`).
   - Hooks directly into `torch.compile(model, backend="holonomic")` / `backend="holonomic_aot"` and `HolonomicGradFreeOptimizer` (`0.00 MB` persistent `.grad` buffers).
2. **Native C++17 / Objective-C++ & Universal 6-Backend TileLang Engine (`ojasx2.tile_hal`, `ojasx2/csrc/`)**:
   - Integrates and extends [`tile-ai/tilelang`](third_party/tilelang) (`SwizzleMode`, `ComputeSquareWarpPartition`, `CanUseCooperativeTensor`) into native `TORCH_LIBRARY(ojasx_hal, m)` C++ operators (`libojasx_tile_kernels.dylib`).
   - Adapts tile schedules (`block_m`, `block_n`, `block_k`), shared-memory XOR bank-conflict swizzling, and hardware 1-cycle ballot instructions across **all 6 hardware targets**:
     - **`"cuda"`**: NVIDIA `sm_80`/`sm_90`/`sm_100` (`tl::rasterization2DRow`, `SWIZZLE_128B`, Tensor Core `mma.sync`/`wgmma`, `__ballot_sync`).
     - **`"rocm"`**: AMD Instinct & Radeon `gfx90a`/`gfx942`/`gfx1100` (`tl::MfmaTraits`, `128B` LDS swizzling, 64-lane wavefront `__ballot` split into 32-bit words).
     - **`"sycl"`**: UXL Foundation Intel oneAPI DPC++ & Vulkan SPIR-V for `iGPU` (Intel Iris Xe 64KB SLM) and `dGPU` (Intel Arc XPU) (`joint_matrix` + `group_ballot`).
     - **`"ascend"`**: Huawei Ascend 910B NPU (`ascend_gemm_l1` double-buffered `L1 -> L0A/L0B -> L0C` Cube + `UB` Vector unit).
     - **`"cpu"`**: Multi-Core x86_64 `AVX-512`/`AVX2` & ARM64 `NEON` + Apple `AMX`/`vDSP` (`tl::cpu::GemmHolonomicRelu2BallotTile<32,32,32>`).
     - **`"metal"`**: Apple Silicon `Metal 4` GPU (`rasterization2DMLX`, 16B padded `threadgroup` SRAM, `simdgroup_float8x8` hardware matrix units, 1-cycle `simd_ballot`).
3. **Smart Workload-Aware Cost Router (`ojasx2.smart_router`)**:
   - Inspects any `torch.nn.Module` and decides **what NOT to execute** and which backend to route each layer to:
     - Automatically detects frozen LoRA / PEFT backbones and routes them to `FrozenLinearZeroTape` (`SKIP_BACKWARD_FROZEN_WEIGHTS`), saving **66.59% of autograd activation memory** with `0.00e+00` gradient drift.
     - Bypasses sub-threshold layers to native PyTorch fast-paths (`NATIVE_PYTORCH_FAST_BYPASS`) so kernel launch overhead is never wasted.
4. **Multi-Laptop P2P Heterogeneous Virtual Unified GPU Mesh (`ojasx2.mesh_hal`)**:
   - Pools heterogeneous laptops (e.g., an `18 GB` laptop + an `8 GB` partner laptop = **`26 GB` Virtual Unified GPU**) over real OS TCP sockets.
   - Uses C++ `bfloat16` scale + `int8` block boundary compression (`torch.ops.ojasx_hal.compress_boundary_int8`) to shrink inter-laptop wire traffic by **`3.76×` (`73.4%` bandwidth saved)** while maintaining **`0.999992` gradient cosine similarity**.

---

## 📁 Repository Structure

```text
OjasX/
├── ojasx2/                         # Production Python + C++ Library (v0.3.0)
│   ├── csrc/                       # Native C++17 & Objective-C++ Kernels (TORCH_LIBRARY)
│   │   ├── ojasx_tile_kernels.cpp  # AMX/vDSP/OpenMP L1 RMSNorm, 1-Bit Ballot, INT8 Mesh Codec
│   │   ├── ojasx_tilelang_metal.mm # Universal 6-Backend TileLang Engine + Metal simdgroup_float8x8
│   │   ├── ojasx_openreg_guard.cpp # C++ PrivateUse1 (torch.ojasx) DeviceGuard & Allocator
│   │   ├── libojasx_tile_kernels.dylib
│   │   └── libojasx_openreg.dylib
│   ├── tile_hal.py                 # Hardware Abstraction Layer, FakeTensor, Autograd & Z3 SMT
│   ├── smart_router.py             # Smart Workload Cost Model & Frozen-Backbone Zero-Tape Router
│   ├── mesh_hal.py                 # Multi-Laptop P2P Heterogeneous Virtual Unified GPU Engine
│   ├── holonomic_math_compiler.py  # Holonomic Linear, SwiGLU, GeGLU, RMSNorm, Selective LMHead
│   ├── aot_holonomic_compiler.py   # AOTAutograd Joint Forward-Backward Graph Compiler
│   ├── grad_free_optimizer.py      # 0-Byte Persistent .grad Optimizer (SGD / AdamW)
│   └── ...                         # Dagger lenses, radial core, spectral & cotangent compilers
├── third_party/tilelang/           # Integrated TileLang C++ Templates (CUDA, HIP, Ascend, Metal, CPU)
├── lean/                           # Lean 4 Formal Proofs (0 sorry)
├── rocq/                           # Rocq (Coq) 9.0 Formal Proofs (108 verified .v files, 0 Admitted)
├── tests/                          # 370-Test Qualification Suite (opcheck, gradcheck, Z3 SMT, MPS/CPU)
├── tools/                          # Hardware Benchmarks, Conformance & Formal Verification Runners
├── docs/                           # 23 Mathematical & Hardware Architecture Monographs
├── reports/                        # Certified Hardware, Mesh & Formal Verification JSON Reports
└── pyproject.toml
```

---

## 🚀 Quickstart

### 1. Drop-In Smart Workload Router (LoRA / Transformer / Vision)
```python
import torch
import ojasx2

router = ojasx2.SmartWorkloadRouter()
model, report = router.optimize_model(your_pytorch_model, batch_tokens=2048)
print(report.workload_profile, report.action_counts)
```

### 2. Universal 6-Backend TileLang Hardware Codegen & Execution
```python
import torch
import ojasx2

# Query chip-adaptive tile schedule for any architecture ("cuda", "rocm", "sycl", "ascend", "cpu", "metal")
sched = ojasx2.tilelang_query_chip_schedule("sycl", M=1024, N=1024, K=1024, smem_kb=64, l2_kb=4096)

# Emit hardware-native kernel source for any target
cuda_src = ojasx2.tilelang_emit_universal_source("cuda", block_m=64, block_n=64, block_k=32, panel_size=4)

# Execute fused GEMM + Holonomic ReLU2 + 1-Cycle Hardware Ballot (32x smaller saved mask!)
x = torch.randn(32, 64, requires_grad=True)
w = torch.randn(64, 64, requires_grad=True)
out, packed_1bit_mask = ojasx2.tilelang_universal_gemm_relu2(x, w, target_arch="auto")
out.sum().backward()
```

### 3. Multi-Laptop P2P Heterogeneous Virtual Unified GPU Mesh
```python
import ojasx2

nodes = [
    ojasx2.MeshNodeSpec("Laptop_A", "Apple_M_Series_GPU", total_vram_gb=18.0, free_vram_gb=14.0, device_str="mps"),
    ojasx2.MeshNodeSpec("Laptop_B", "Partner_Laptop_GPU", total_vram_gb=8.0,  free_vram_gb=6.5,  device_str="cpu"),
]
mesh_engine = ojasx2.HolonomicVirtualMeshEngine(model, nodes=nodes, compress_boundary=True)
loss = mesh_engine(x).square().mean()
loss.backward()
mesh_engine.close()
```

---

## 📊 Certified Real-World Benchmarks & Verification

All C++ operators and mathematical passes are verified with **`torch.library.opcheck`**, **`torch.autograd.gradcheck`**, **`Z3 SMT BitVec32`**, **`Lean 4`**, and **`Rocq 9.0`**:

| Subsystem | Metric | Baseline PyTorch | OjasX v0.3.0 | Improvement / Verification |
| :--- | :--- | :--- | :--- | :--- |
| **C++ L1-Resident `RMSNorm`** | Forward + Backward Latency (`[4, 256, 512]`) | `873.5 us` | **`379.5 us`** | **`2.30×` Faster** (`4 B/row` saved vs `2048 B/row`) |
| **1-Bit Hardware Ballot Packing** | Saved Activation Mask Memory (`ReLU2`) | `32 bits / elem` | **`1 bit / elem`** | **`32.0×` Smaller** (`simd_ballot` / `__ballot_sync`) |
| **Smart Workload Router (LoRA)** | Saved Autograd Tape Memory | `84.09 MB` | **`28.10 MB`** | **`66.59%` Memory Saved** (`0.00e+00` max grad diff) |
| **Multi-Laptop P2P TCP Mesh** | Inter-Laptop Wire Bandwidth (`18GB + 8GB`) | `1048.6 KB` | **`278.5 KB`** | **`3.76×` Compression** (`0.999992` grad cosine sim) |
| **Z3 SMT & Formal Proofs** | Shared-Memory Bank Conflicts & Theorems | — | **`unsat` (0 conflicts)** | **`369 passed` in `pytest`** |

Run the full qualification suite:
```bash
pytest -q
```
