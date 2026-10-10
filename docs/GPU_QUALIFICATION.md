# Apple M5 GPU qualification — 7 October 2026

Subsequent speed-focused extension: [cached nonlinear reverse](CACHED_REVERSE_THEOREM.md) adds thirteen Lean statements and a native Metal SiLU reverse step. It removes projection replay in cached blocks, trades depth-independent tape for roughly half of native retained activation storage, and has separate GPU timing/training evidence. The measurements below describe the preceding GPU qualification run and its original implementation.

OjasX now executes selected algebraic backpropagation schedules on the laptop's
Apple M5 through PyTorch MPS. The final experiment passed its declared output,
gradient and synthetic training comparisons with CPU fallback disabled.
The gains depend strongly on the graph and shape. This prototype has a useful
GPU linear-chain result and substantial activation-tape savings; several paths
remain slower than native GPU PyTorch.

Evidence: [final measurements](reports/mps-qualification.json),
[initial measurements](reports/mps-qualification-initial.json),
[GPU regression tests](tests/test_mps_qualification.py), and
[full verification](reports/verification.json).

## Measured results

Timings include forward and every first-order input/parameter gradient, excluding
the optimizer. Values are medians of nine shuffled rounds with three executions
per sample, after three warmups, using GPU synchronization. Native is eager
PyTorch on the same MPS device and FP32 inputs. These small measurements are
shape-specific and do not establish sustained large-training throughput.

| GPU workload | Native → OjasX joint time | Native/OjasX speed ratio | Fresh saved activation storage reduction |
|---|---:|---:|---:|
| Linear pair, batch 1024, widths 32→4096→64 | 1.404 → 0.333 ms | **4.22×** | **99.95%**, 16 MiB → 8 KiB |
| Same pair, native forward with shared backward | 1.404 → 0.848 ms | **1.66×** | **100%** of fresh tape; supplied operands remain |
| Linear–RMSNorm–Linear, same dimensions, learned scale/bias | 7.225 → 4.347 ms | **1.66×** | **97.10%**, 50,335,744 → 1,458,176 bytes |
| Four-leaf matrix chain | 0.344 → 0.351 ms | 0.98× | 93.84% |
| Rank-7 GEMM, 256×256 operands | 0.183 → 1.035 ms | 0.18× | Both retain supplied operands; zero fresh tape |
| Packed ReLU, 1024×1024 | 0.269 → 1.384 ms | 0.19× | 96.875%, 4 MiB → 128 KiB |
| Reversible coupling, depth 64, batch 128, split width 32 | 20.586 → 32.609 ms | 0.63× | **99.22%**, 4,177,920 → 32,768 bytes |
| Compiled three-block residual model, 8×32×16 tokens | 1.477 → 10.598 ms | **0.14×** | **86.44%** |

The wide-output linear control rejects reassociation as unprofitable and retains
native execution. Its timing variation is ordinary measurement noise, not an
optimization gain. GELU and SiLU retained native output/first-gradient values in
the sample but their registered wrappers added overhead. SwiGLU removed its
fresh intermediate tape while measuring about 0.95× native speed. The RMSNorm
primitive reduced fresh tape by 99.90% while measuring about 0.71× native speed.

“Fresh saved storage” excludes supplied inputs and parameter storage, deduplicates
aliases, and measures forward autograd tape. A 100% reduction means no additional
fresh tensor was saved; training still requires operands, outputs, gradients and
workspaces. The report also contains synchronized live MPS allocator readings
before forward, after forward, and after backward. None of these is a peak-memory
measurement. Parameters, optimizer states and unified-memory system pressure
prevent interpreting these percentages as total training-memory savings.

## What mathematics produced the gain

For two consecutive linear maps, let

```text
X[m,k], A[n,k], B[p,n]
H = X Aᵀ
Y = H Bᵀ
```

The contracted representation forms `P = B A` and evaluates `Y = X Pᵀ`.
For incoming gradient `G[m,p]`, the shared matrix `T = Gᵀ X` gives

```text
dX = G P
dA = Bᵀ T
dB = T Aᵀ
```

The original weights remain trainable. When the hidden width `n` greatly exceeds
the input/output widths, the graph avoids forming and retaining a large hidden
activation. The real-number equality follows from associativity and adjoint
composition. The same reassociation changes FP32 rounding; it has no universal
bitwise-equivalence guarantee.

RMSNorm has an additional sufficient statistic. With learned hidden scale `γ`,

```text
Q = Aᵀ A
row_moment = sum((X Q) * X, last_axis) / n
P = (B * γ) A
Y = (X Pᵀ) / sqrt(row_moment + ε) + final_bias
```

This represents the normalized hidden vector through its row moment and projected
output. Native autograd differentiates this smaller graph, including `γ` and both
original weights. The favorable measured shape reduces the modeled main-GEMM
scalar operations from 2,407,038,976 to 93,546,496: **96.11%**. That cost model
excludes pointwise work, guards, copies, launches and hardware utilization, which
explains why its arithmetic saving is much larger than its measured speedup.

In the first GPU experiment RMSNorm was slightly slower despite reducing tape.
Batching input endpoint reductions into one host check, and combining the moment
checks, reduced repeated GPU waits without dropping a guard. The final path still
checks finite inputs, finite moments above `10 ε`, and finite outputs. These are
heuristic conditioning checks; they do not certify all floating-point errors.
The initial report is preserved so the performance improvement is reviewable.

Arbitrary GELU/SwiGLU between dense maps still has an input-dependent derivative.
This work does not prove that such a layer can always collapse into a small fixed
matrix. Reversible coupling instead reconstructs earlier states from later states
and reevaluates branches during backward. Its memory saving comes with extra
arithmetic, which explains its measured slowdown. Packed ReLU stores exact branch
information in one bit per element, but this ATen implementation spends extra
launches packing/unpacking those bits.

## Numerical and training qualification

The GPU regression suite covers 58 sampled MPS tests, plus the existing packed
ReLU hardware check. Coverage includes all linear/RMS gradient-demand patterns,
zero learned scales, fallback conditions, nonfinite inputs, second-derivative
directions, GELU/SiLU/SwiGLU, matrix-chain planning, rank-7, reversible GELU/SiLU
replay, and eight compiled graph families with gradient accumulation and AdamW.
Unsupported hardware skips are explicit.

The benchmark contract is declared before execution: entrywise `rtol=2e-4`,
`atol=2e-4`, plus relative-L2 error below `2e-4`. It compares each output and each
gradient with both MPS native PyTorch and independently evaluated CPU FP64 native
PyTorch on the same rounded inputs. All sampled benchmark comparisons passed.
The JSON preserves individual absolute and norm errors. This contract is separate
from the earlier MLX probe's tighter contract, which failed and remains failed.

Two 30-step AdamW runs, each accumulating two microbatches per update, passed:

| Model | Graph rewrite sites | Maximum parameter absolute difference |
|---|---:|---:|
| Three RMS-core residual blocks | 3 | 2.63e-6 |
| Standard attention/SwiGLU Transformer control | 0 | 0 |

The Transformer control demonstrates preserved behavior with no applicable
contraction; it provides no evidence of a Transformer speed or memory gain.
These are synthetic tests, not real-data convergence studies. The existing CPU
BatchNorm-bias/AdamW negative qualification remains relevant: small local gradient
differences can be amplified by an optimizer.

The 139 existing Lean statements and 41 scoped Z3 models support the stated
mathematics. No new theorem was needed to transfer these tensor schedules to
MPS. Differential GPU tests provide the implementation evidence; there is no
formal refinement proof from Lean to GPU machine code.

## Use the qualified paths

The public default remains `accuracy='strict'`. Algebraic mode is explicit and
requires numerical qualification for the actual model and shapes. Newly enabled
device guards admit CPU FP32/FP64 and MPS FP32 with autocast disabled; other
accelerators retain native fallback or the reversible API's explicit rejection.
The shared-mask ReLU and LayerNorm contraction APIs still have CPU-only guards.

```python
import torch
from ojasx2 import linear_chain, make_backend

# x, w1 and w2 must already be MPS FP32 tensors.
y = linear_chain(x, w1, w2, accuracy='algebraic')
y.square().mean().backward()

# Recognizes eligible Linear(no first bias)–RMSNorm–Linear subgraphs.
# Standard parameter objects and ordinary PyTorch optimizers are preserved.
model = model.to(device='mps', dtype=torch.float32)
backend = make_backend(accuracy='algebraic', contraction_policy='rms_core')
compiled_model = torch.compile(model, backend=backend, fullgraph=True)
```

Native fallback evaluates ordinary PyTorch on the tensors' existing GPU; it does
not transfer the workload to CPU. The scalar cost guard predicts arithmetic
profitability. It does not predict GPU time, so the small residual model above
should use strict execution when speed is the objective.

Reproduce from the rebuild directory with the recorded environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B -m pytest tests/test_mps_qualification.py tests/test_hardware.py -q
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/qualify_mps.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_all.py
```

## Hardware implementation implications

Today the schedules use MPS-backed ATen tensor operations, not generated custom
Metal kernels. The largest measured remaining bottleneck is launch/wait overhead
on small graphs. A next backend implementation should combine RMS row-statistic,
scaling and finite-flag reductions where legal, retain explicit conditioning and
strict fallbacks, and tune tile sizes for the measured widths. Packed-mask
construction and decoding also need a native kernel before their storage benefit
can plausibly become a speed benefit. Rank-7 requires hardware-aware crossover
selection; this GPU sample should use native GEMM.

Whole-model scheduling must budget parameter/gradient/optimizer residency and
transient scratch in addition to saved activations. Validate each native lowering
against these MPS/reference results, then measure true peak and sustained training
throughput. The goal of fitting a workload previously requiring several GPUs on
one remains an engineering objective; this qualification does not establish it.
Neural Engine training and GPU targets beyond this M5 remain unqualified.
