Coming soon the backend building is almost completing new math+c++ hardware complier is in perp the sooner OJASX foundation work complete the project will be completely new a ture openCUDA not just the cuda breaking we are coming with the new innovation where you can build models in your locally hardware where tradtionall it takes a hug gpu memory 
<h1 align="center">⚡ O J A S X</h1>

<p align="center">
  <strong>The Universal AI Compute Engine</strong><br/>
  <em>"Ojas" (ओजस्) — The vital energy that powers all creation.<br/>We channel that energy through every GPU on Earth.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge" alt="AGPL-3.0"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-orange?style=for-the-badge" alt="v1.0.0"/>
  <img src="https://img.shields.io/badge/Tests-38%2F38_Passing-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/Hardware-ANY_GPU-purple?style=for-the-badge" alt="Any GPU"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge" alt="PyTorch"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Supported-ed1c24?style=flat-square&logo=amd" alt="AMD"/>
  <img src="https://img.shields.io/badge/Intel-Supported-0071c5?style=flat-square&logo=intel" alt="Intel"/>
  <img src="https://img.shields.io/badge/Qualcomm-Supported-3253dc?style=flat-square&logo=qualcomm" alt="Qualcomm"/>
  <img src="https://img.shields.io/badge/ARM_Mali-Supported-0091bd?style=flat-square&logo=arm" alt="ARM"/>
  <img src="https://img.shields.io/badge/NVIDIA-Supported-76b900?style=flat-square&logo=nvidia" alt="NVIDIA"/>
</p>

---
note:- this readme isn't updated so plz go with the every file in this project yet this project is in the cooking we are devloping it and stay tune for the future update for the OJASX which is coming with more deeply developed and also more mature C++ backends with the more robust the improvments fo this next version is gone be crazy so stay tune 

## 💡 The Problem

Today, if you want to train an AI model, you have **one choice**: buy an NVIDIA GPU and use CUDA. One company controls the entire AI revolution. Students can't afford it. Startups can't compete. Developing nations are locked out.

**That ends now.**

<p align="center">
  <img src="assets/breaking_monopoly.png" alt="Breaking the Monopoly" width="600"/>
</p>

## ⚡ The Solution: OjasX

**OjasX** is a PyTorch extension that makes AI run on **ANY GPU on Earth** — AMD, Intel, Qualcomm, ARM Mali, or NVIDIA — through the universal OpenCL standard.

```python
import torch
import torchcl  # ← This is OjasX

# Create tensors on ANY GPU (not just NVIDIA!)
x = torchcl.to_opencl(torch.randn(512, 512))
y = torchcl.to_opencl(torch.randn(512, 512))

# Matrix math on YOUR GPU
z = torchcl.matmul(x, y)
z = torchcl.relu(z)

# Train neural networks — no CUDA needed
result = torchcl.to_cpu(z)
```

> *"In Ayurveda, Ojas is the supreme energy that sustains life itself. OjasX channels that energy through silicon — giving the power of AI to every human being, on every chip."*

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="OjasX Architecture" width="600"/>
</p>

OjasX is built as a layered engine. Each layer has a single responsibility:

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR PYTHON CODE                     │
│              x = torchcl.matmul(a, b)                   │
├─────────────────────────────────────────────────────────┤
│                    PUBLIC API (api.py)                  │
│         40+ operations: add, matmul, relu, softmax...   │
├─────────────────────────────────────────────────────────┤
│                 COMPUTE ENGINE (engine.py)              │
│       Converts PyTorch tensors ↔ OpenCL GPU buffers     │
│              Launches kernels with optimal sizing       │
├──────────────────────┬──────────────────────────────────┤
│   JIT COMPILER       │      KERNEL REGISTRY             │
│  Fuses multiple ops  │   Loads & caches compiled        │
│  into single kernels │   .cl kernel programs            │
├──────────────────────┴──────────────────────────────────┤
│              OPENCL RUNTIME (context.py + memory.py)    │
│    Platform discovery │ Buffer pool │ CPU↔GPU transfers │
├─────────────────────────────────────────────────────────┤
│                   ANY OPENCL DEVICE                     │
│        AMD │ Intel │ Qualcomm │ ARM │ NVIDIA │ FPGA     │
└─────────────────────────────────────────────────────────┘
```

### How It Works — The 5 Layers

**Layer 1: Runtime Foundation** — Discovers your GPU, creates an OpenCL context and command queue, and manages a **caching buffer pool** that reuses GPU memory allocations for zero-overhead tensor creation.

**Layer 2: GPU Kernels** — Hand-written OpenCL C kernels optimized for each operation class:
- `elementwise.cl` — 16 kernels: arithmetic, math, comparisons
- `activation.cl` — 9 kernels with backward passes for autograd
- `matmul.cl` — Tiled matrix multiply using GPU local memory
- `reduction.cl` — Parallel tree reduction + softmax

**Layer 3: Compute Engine** — The bridge. Converts PyTorch tensors to OpenCL buffers, calculates optimal workgroup sizes, and launches kernels. This layer ensures **any** PyTorch operation maps to the correct GPU kernel.

**Layer 4: JIT Compiler** — The brain. Instead of launching 3 separate kernels for `relu(bias + matmul(A, B))`, it **fuses** them into a **single** GPU kernel at runtime. One launch instead of three = 3x less overhead.

**Layer 5: Auto-Tuner** — Queries your specific GPU (compute units, local memory, max workgroup) and selects **optimal** parameters. OjasX runs well on a $50 Intel integrated GPU AND a $10,000 data center GPU — it adapts automatically.

---

## 📊 V1 Capabilities

### Supported Operations (40+)

| Category | Operations | GPU Kernels |
|:---:|:---|:---:|
| **Arithmetic** | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `exp`, `log`, `sqrt` | 16 |
| **Activations** | `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `leaky_relu`, `softmax` | 9 |
| **Matrix** | `matmul` (tiled), `transpose` | 4 |
| **Reductions** | `sum`, `mean`, `max`, `min` | 4 |
| **Creation** | `zeros`, `ones`, `full`, `randn` | 2 |
| **Movement** | `to_opencl`, `to_cpu`, `synchronize` | 2 |
| **JIT Fusion** | Any chain of element-wise ops fused into 1 kernel | ∞ |

### Test Results

```
============================================================
  OjasX V1 — Test Results
============================================================

  Smoke Tests:     30/30 PASS  ✓
  JIT Tests:        8/8  PASS  ✓
  MNIST Training:  Converges   ✓
  ─────────────────────────────
  TOTAL:           38/38 PASS

  Device: Intel(R) Iris(R) Xe Graphics
  OpenCL: 3.0 | 96 CUs | 6,466 MB
============================================================
```

### Performance (Intel Iris Xe — $0 integrated GPU)

| Operation | Size | Time |
|:---:|:---:|:---:|
| Matrix Multiply | 512×512 | 3.6 ms |
| ReLU | 65,536 elements | 0.1 ms |
| Full MLP Forward | 32×784→10 | 9.0 ms |
| MNIST Train Step | batch=64 | 10 ms |

---

## 🚀 Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.x
- **Any** GPU with OpenCL drivers (Intel, AMD, NVIDIA, ARM...)

### Install

```bash
git clone https://github.com/mkrishna793/-OjasX.git
cd -OjasX
pip install -e .
```

### Verify Your GPU

```python
import torchcl
print(torchcl.get_device_info())
# {'name': 'Intel(R) Iris(R) Xe Graphics', 'version': 'OpenCL 3.0 NEO', ...}
```

### Run Tests

```bash
python tests/test_smoke.py    # 30 operation tests
python tests/test_jit.py      # JIT compiler tests
```

### Train a Neural Network

```bash
python examples/train_mnist.py
```

---

## 🧠 JIT Kernel Fusion — The Secret Weapon

Most AI frameworks launch **one GPU kernel per operation**. Three operations = three kernel launches = three rounds of overhead.

OjasX's JIT compiler **fuses** chains of operations into **one kernel**:

```
WITHOUT FUSION (3 kernel launches):
  GPU Launch 1: temp1 = A + B          ← overhead
  GPU Launch 2: temp2 = relu(temp1)    ← overhead
  GPU Launch 3: result = sigmoid(temp2)← overhead

WITH OJASX JIT (1 kernel launch):
  GPU Launch 1: result = sigmoid(relu(A + B))  ← ONE launch, ZERO overhead
```

The JIT compiler generates this fused OpenCL C code **at runtime**:

```c
__kernel void fused_binary_unary(__global const float* a,
                                 __global const float* b,
                                 __global float* output,
                                 const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float a_val = a[gid];
        float b_val = b[gid];
        // sigmoid(relu(a + b)) — ALL in one kernel!
        output[gid] = (1.0f / (1.0f + exp(-(fmax((a_val + b_val), 0.0f)))));
    }
}
```

---

## 🌍 Vision & Mission

### Why This Matters

Today, 3 billion people own devices with GPUs that can run AI. But only those with NVIDIA cards can actually do it. That's not a technical limitation — it's a **business decision** by one company.

**OjasX changes this:**

- 🎓 **Students** in developing countries can train models on their Intel/AMD laptops
- 🏢 **Startups** can build AI products without $50K NVIDIA hardware budgets
- 🔬 **Researchers** can run experiments on whatever hardware their lab has
- 🌐 **Governments** can build sovereign AI infrastructure without vendor lock-in
- 🤖 **Edge devices** with ARM/Qualcomm chips can run local AI
  
  ### 🌱 For the Earth ###

AI has an energy crisis no one talks about. A single NVIDIA A100 GPU consumes **400 watts** — training one large model can emit as much CO₂ as **five cars in their entire lifetime**.

But here's what they don't tell you: most of that energy is wasted on hardware that was designed to render video game pixels, not multiply matrices. Meanwhile, chips built for efficiency already exist:

- **ARM Mali GPUs** consume **2-5 watts** — running in billions of phones worldwide
- **Intel Iris Xe** uses **15-25 watts** — already in your laptop
- **Qualcomm Adreno** draws **3-8 watts** — powers every Android flagship

These chips aren't as fast as a $40,000 NVIDIA H100. But they don't need to be. Not every AI task requires a nuclear reactor. A student training a model, a doctor running diagnostics, a farmer using crop analysis — they need **good enough** AI that runs on hardware that **doesn't burn the planet**.

OjasX makes this possible. By unlocking AI on energy-efficient silicon, we're not just breaking a monopoly — **we're reducing the carbon cost of intelligence itself**.

> *"The greenest GPU is the one already in your pocket."*


### The Roadmap

| Version | Milestone |
|:---:|:---|
| **V1** ✅ | Core engine: 40+ ops, JIT fusion, auto-tuner, MNIST training |
| **V2** | Full autograd on GPU, conv2d, batch_norm, embedding |
| **V3** | `device="opencl"` syntax via PrivateUse1 C++ integration |
| **V4** | Performance parity with CUDA on equivalent hardware |
| **V5** | torch.compile full graph execution, multi-GPU support |

---

## ⚖️ License — The Shield of the People

> **This section is the most important part of this entire project.**

OjasX is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### What This Means

#### ✅ For the People (Students, Researchers, Indies, Startups)

**Use it. Modify it. Build on it. Ship products. Completely free. Forever.**

You can:
- Use OjasX in your research, your startup, your school project
- Modify the source code however you want
- Distribute your modifications
- Use it commercially in your own products

**One simple rule:** If you modify OjasX and distribute your version (or run it as a cloud service), you must release your modifications under the same AGPL-3.0 license. This ensures your improvements flow back to humanity.

#### 🛡️ Against Monopolists

The AGPL-3.0 is a **"copyleft" or "viral" license**. Here's what that means for companies who want to lock this technology away:

> **If a corporation takes OjasX, modifies it, and uses it to build a proprietary product or cloud service — they are LEGALLY REQUIRED to publish ALL of their modifications as open source under the same license.**

This makes it **mathematically impossible** to build a secret monopoly on top of this technology. The license acts as an unbreakable shield that keeps the fire of compute in the hands of the public **forever**.

The AGPL specifically closes the "cloud loophole" that exists in the regular GPL — so companies can't hide modified code behind SaaS servers.

#### 💼 Commercial Licensing (Dual License)

If your company wants to use OjasX **without** open-sourcing your proprietary code, a **commercial license** is available. Contact: [mkrishna.16july@gmail.com](mailto:mkrishna.16july@gmail.com) and [mohanakrishnanannuru@gmail.com]
This dual-licensing model ensures:
1. **The technology stays free** for individuals and the open-source community
2. **Corporations must contribute back** if they use the free version
3. **Companies who want privacy** can pay for it — funding further development

> *"The technology belongs to everyone. If you want to keep secrets, pay the toll."*

---

## 📁 Project Structure

```
OjasX/
├── torchcl/                    # Core Python package
│   ├── __init__.py             # Entry point — auto-initializes OpenCL
│   ├── api.py                  # Public API — 40+ tensor operations
│   ├── _backend.py             # torch.compile backend integration
│   ├── runtime/                # OpenCL runtime layer
│   │   ├── context.py          # Platform/device/context management
│   │   └── memory.py           # Caching buffer pool allocator
│   ├── kernels/                # GPU kernel source code
│   │   ├── elementwise.cl      # Arithmetic & math (16 kernels)
│   │   ├── activation.cl       # Activation functions (9 kernels)
│   │   ├── matmul.cl           # Matrix multiply — naive + tiled
│   │   ├── reduction.cl        # Parallel reductions + softmax
│   │   └── registry.py         # Kernel loader & compiler cache
│   ├── ops/                    # Operation dispatch
│   │   └── engine.py           # Central compute engine
│   └── jit/                    # JIT compilation engine
│       ├── compiler.py         # Fused kernel code generator
│       ├── cache.py            # LRU kernel cache
│       └── tuner.py            # Hardware-adaptive auto-tuner
├── tests/                      # Test suite (38 tests)
│   ├── test_smoke.py           # 30 operation correctness tests
│   └── test_jit.py             # 8 JIT compiler tests
├── examples/                   # Usage examples
│   ├── basic_usage.py          # Arithmetic + matmul + MLP demo
│   └── train_mnist.py          # Full MNIST training on OpenCL
├── assets/                     # Images and media
├── LICENSE                     # AGPL-3.0
├── pyproject.toml              # Build configuration
└── README.md                   # This file
```

---

## 🛠️ Built With

| Technology | Role |
|:---:|:---|
| **PyTorch 2.x** | Tensor framework — we extend, not fork |
| **OpenCL 3.0** | Universal GPU compute standard (Khronos Group) |
| **PyOpenCL** | Python↔OpenCL bridge for GPU kernel management |
| **NumPy** | CPU↔GPU data marshaling |

---

## 🤝 Contributing

This project is open to everyone. Whether you're a first-time contributor or a GPU programming veteran — your help makes AI accessible to the world.

```bash
# Clone
git clone https://github.com/mkrishna793/-OjasX.git
cd -OjasX

# Install in dev mode
pip install -e ".[dev]"

# Run tests
python tests/test_smoke.py
python tests/test_jit.py
```

**Areas where contributions are needed:**
- **More operators** — conv2d, batch_norm, embedding, gather/scatter
- **Performance** — Optimized kernels for specific GPU architectures
- **Testing** — Test on AMD, Qualcomm, ARM Mali hardware
- **Documentation** — Tutorials, benchmarks, comparisons

---

## 👤 Author

Created by **mkrishna793** — with the conviction that AI compute is a human right, not a corporate privilege.

---
