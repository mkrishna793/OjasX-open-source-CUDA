# OjasX findings: mathematics, developer use and hardware implementation

This guide specifies how to use the measured contraction optimization and how a hardware backend could implement it. The delivered implementation executes qualified PyTorch CPU graphs. The GPU schedules below are an implementation design; no native CUDA, Metal or Neural Engine kernel is certified by this project.

## 1. What the latest finding actually changes

A model segment with two consecutive Linear layers, no intervening nonlinearity and matching shapes can keep its original trainable parameters while changing the order of multiplication. Let `X` be `m×k`, `W1` be `n×k`, and `W2` be `p×n`. For cotangent `G` of shape `m×p`, the joint schedule is:

```text
P   = W2 @ W1          # p×k
Y   = X @ P.T         # m×p
T   = G.T @ X         # p×k, shared by both weight pullbacks
 dX  = G @ P          # m×k
 dW1 = W2.T @ T       # n×k
 dW2 = T @ W1.T       # p×n
```

This avoids the conventional wide hidden activation `H=X@W1.T` and hidden cotangent `G@W2`. It still returns both full-size weight gradients. Associativity, transpose laws and the derivative's adjoint establish the real-algebra identities. See [Contraction.lean](lean/OjasXMath/Contraction.lean) and [the detailed mathematical/measurement guide](CONTRACTION_OPTIMIZATION.md).

The initial recorded CPU sample, batch 256 and widths 1024→2048→32, took 5.6216 ms for native forward/input/both-weight gradients and 1.0633 ms for the guarded joint schedule. It removed 86.42% of modeled scalar arithmetic, 93.75% of fresh saved storage and 16.62% of profiler incremental tensor peak. These are measurements of this segment, not percentages to apply to an entire model. The [initial benchmark](reports/contraction-benchmark-initial.json) is preserved; the current runner also produces a fresh [benchmark](reports/contraction-benchmark.json).

Ordinary PyTorch autograd on the reassociated graph already realizes the same sharing and was slightly faster than the adapter. The mathematical optimization is the graph rewrite; a special gradient engine is not required to obtain it.

## 2. Use it without changing the training interface

```python
import torch
from ojasx2 import make_backend

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048, bias=False),
    torch.nn.Linear(2048, 32, bias=False),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
compiled = torch.compile(model, backend=make_backend(
    accuracy="algebraic", contraction_policy="linear_chain"))

x = torch.randn(256, 1024)
target = torch.randn(256, 32)
optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(compiled(x), target)
loss.backward()
optimizer.step()
```

Keep the optimizer attached to the original model parameters. The rewrite does not replace `W1/W2` with a single trainable parameter: doing that would change the parameterization and optimization dynamics. Input-gradient computation is omitted when unnecessary; the cost guard accounts for differentiation demand. Bias in joint mode uses `W2@b1+b2`, differentiating through both original biases.

`accuracy="strict"` keeps native arithmetic order. Algebraic execution is opt-in because rounding changes. `contraction_policy="linear_chain_backward"` retains native forward evaluation and changes only the backward/tape; it currently falls back for biased chains. The current qualified fast paths are positive matching 2D CPU FP32/FP64 with no CPU autocast. Unsupported devices, shapes and types retain native execution. These helpers do not move a GPU tensor to CPU.

The current subsequent run passes 121 Lean statements, 33 scoped Z3 checks and 209 CPU tests, with two unavailable-device skips. Its favorable centered-weight Gram LayerNorm sample (batch 512, widths 32→2048→16) takes 1.7405→0.7623 ms (2.28×), with 92.23% fewer modeled main-GEMM operations and 89.85% lower profiler incremental tensor peak. This changes floating output order; the research guide documents tolerance samples, short training qualification and the limits of heuristic conditioning guards. These results are additional patterns and cannot be multiplied by the original segment speedup.

The new `contraction_policy="matrix_chain"` searches a longer bias-free sequence, counting forward and all demanded first-order pullbacks. It preserves nonlinear and fanout boundaries. See [the research extension](NONLINEAR_AND_CONTRACTION_RESEARCH.md) for the recurrence and measurements. For an explicit sequence use `linear_sequence(x, weights, accuracy="algebraic")`; for arbitrary compatible 2D matrix leaves use `matrix_chain(leaves, accuracy="algebraic")`.

Two explicit LayerNorm adapters are also available: `layernorm_linear_chain` retains native forward evaluation; `gram_layernorm_linear_chain` is an experimental algebraic forward that avoids the wide activation through centered-weight moments. Both require a bias-free first layer and last-axis normalization; use the nonlinear guide for formulas and numerical limits.

The ReLU experiment is a separate explicit adapter, `relu_linear_chain(x,w1,w2,bias1,bias2,accuracy="algebraic")`. It selects a changed backward only when every actual batch row has the identical ReLU mask and the scalar model predicts a saving. Random differing masks keep their ordinary graph. It is not an automatically enabled nonlinear compiler pass.

## 3. Compile the semantics before choosing a kernel

Capture a graph with shapes, original parameter identities, dtype, accumulation dtype, strides, device, alias/version state, RNG/effects and gradient demand. A legal candidate must preserve all observable consumers and requested parameter gradients. Two Linear calls separated by ReLU, SiLU, GELU, dropout or an unknown operation are not a consecutive linear segment. If an inner value also feeds a loss, residual or another consumer, do not discard that value.

Attach three different obligations to each candidate: exact real function/pullback identity, sufficient residual information with safe lifetimes, and a target-specific floating execution contract. A Lean theorem name attached to IR does not mechanically verify the emitted program. The project currently checks mathematics plus independent Python conformance; formal IR refinement is future work.

For a longer sequence, the implemented planner minimizes a scalar model over binary multiplication trees. Hardware selection needs more information: predicted library time, dispatch cost, copies, forward and backward tape peaks, temporary workspace and numerical sensitivity. Retain more than one candidate on a time/memory/error frontier when possible. Scalar optimality alone cannot guarantee lowest hardware time.

## 4. A concrete device GEMM schedule

Start with six ordinary device-library GEMMs and a small residual descriptor. Allocate `P` and `T` as `p×k` tensors, not as packed copies of the complete weights. All strides/transposes below are logical; use the library's layout descriptors instead of materializing a transpose unless measurements justify a copy.

| Phase | Operation | Output | Required operands/lifetime |
|---|---|---|---|
| Forward | `P = W2 W1` | p×k | Both weights at the current parameter version |
| Forward | `Y = X Pᵀ` | m×p | X and P; add composed bias if supported |
| Backward | `T = Gᵀ X` | p×k | Needed when either weight gradient is requested |
| Backward | `dX = G P` | m×k | Needed only if X cotangent is requested |
| Backward | `dW1 = W2ᵀ T` | n×k | Skip when W1 gradient is not requested |
| Backward | `dW2 = T W1ᵀ` | p×n | Skip when W2 gradient is not requested |

`T` must remain live until both consumers finish. `P` must remain live until `dX` finishes, or be recomputed with an explicit additional-product budget. The backward-only policy makes that recomputation explicit. For higher derivatives, intermediates recomputed from original trainable tensors must remain connected to the derivative graph; detached caches are insufficient.

A minimal scheduling sketch is:

```text
require semantic guards and selected accuracy contract
P = device_gemm(W2, W1)
Y = device_gemm(X, transpose(P))
save references/version tokens for demanded X/weights; retain P if dX requested

when G becomes available:
    if dW1 or dW2: T = device_gemm(transpose(G), X)
    if dX:        dX = device_gemm(G, P)
    if dW1:       dW1 = device_gemm(transpose(W2), T)
    if dW2:       dW2 = device_gemm(T, transpose(W1))
    accumulate shared-parameter contributions using the chosen reduction order
    release T/P only after their final dependent command completes
```

This is backend pseudocode, not an installed GPU implementation. Preserve the optimizer's separate original weight updates. Reusing `P` across an optimizer step without checking both weight versions is wrong. Captured execution must update pointers/versions or recompute P each step; cache layouts and algorithm choices separately from numerical tensor contents.

### Structured normalization schedules

The explicit `layernorm_linear_chain` helper retains native forward/row statistics and rewrites its first-order pullback. Its input width k must be small enough that `W1.T@W1` and `X.T@diag(α)@X`, both k×k, cost less than the wide original cotangents. The nonlinear research guide gives every formula and includes gradients for γ, β and final bias. Emit those corrections/reductions alongside T/K; omitting normalization parameter gradients changes training.

The further `gram_layernorm_linear_chain` variant forms centered weights `B=W1−mean_rows(W1)` and `S=B.T@B`, computes row means/variances via X/S, and evaluates the normalized narrow output directly. This removes the m×n primal activation in real algebra. For a device implementation, schedule B/S once for the current weight version, process X in tiles, and retain the chosen smaller residuals. S is k×k and is not free; B consumes n×k storage unless centering is integrated into the product. Current native autograd retains the reduced graph's required tensors; a custom liveness planner must account for them explicitly.

Do not reuse S across a first-weight update. Avoid computing variance as `mean(z²)−mean(z)²`; centered-weight Gram still needs conditioning analysis. The experimental CPU helper has conservative native-fallback checks, but these are not certified floating error bounds. A device lowering must qualify them independently and preserve or explicitly change the accuracy contract. Higher-order derivatives on the native-forward custom schedule replay the native graph; the Gram variant uses native autograd on its reduced graph. Neither first-order memory figure applies automatically to all higher-order workloads.

## 5. Memory hierarchy and arithmetic units

The rewrite removes large logical intermediates. Tiling then determines actual device traffic. Load operand tiles into on-chip storage, perform the tile products, and write the requested output tiles. Reuse small P/T tiles where practical, but do not assume the entire P or T fits in registers. At `p=32,k=1024`, each FP32 tensor is 128 KiB. A target must choose tile dimensions from its real register/shared-memory limits, occupancy and library behavior.

GEMM dimensions and tile occupancy affect efficiency; a much narrower product can perform fewer operations while using the hardware less efficiently. The [NVIDIA GEMM guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the compute/memory and tiling tradeoffs. The project's conditional roofline theorem states `time ≥ max(F/P,Q/B)` when supplied work and transferred bytes obey the stated capacity inequalities. It supplies a lower bound, not a measured latency predictor.

Required dense `dW1` and `dW2` outputs still occupy their full sizes. Optimizer states, parameter masters and other model activations also remain. Report saved tape, temporary peak, actual allocator peak and process/device resident memory separately. “Move GEMM from memory into ALUs” means reducing materialization/traffic while reusing operand tiles; arithmetic units still need operand values.

After a correct library-GEMM implementation is qualified, custom kernels can combine tile consumers or integrate mask checks. Any launch reduction is an additional hardware optimization with its own rounding and race proofs; the mathematical reassociation itself works without fusion. Handle edge tiles and synchronization using actual target properties, not the project's example 32-bank model.

## 6. NVIDIA, Apple and other targets

For NVIDIA, a possible first backend is cuBLASLt with explicitly selected layouts, compute types, workspace and algorithms; its [official documentation](https://docs.nvidia.com/cuda/cublas/index.html) describes those controls. Keep tensors on the device, express transposes through descriptors, and measure the entire forward/backward sequence with device timing. Pin math mode and qualify any TF32 or reduced-precision accumulation separately. No cuBLAS backend is implemented here.

For Apple, the first engineering route is qualified Metal/PyTorch MPS matrix products; [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) is a candidate lower-level matrix primitive. Device access is now restored and selected MPS FP32 schedules have actual numerical, training, timing and retained-storage measurements in [GPU qualification](GPU_QUALIFICATION.md). The [cached nonlinear reverse extension](CACHED_REVERSE_THEOREM.md) also supplies a native Metal SiLU reconstruction/VJP kernel. Full native compiler lowering, global peak-memory planning and other GPU targets remain pending. Core ML ML Programs converted the explicit forward/VJP graphs, but the earlier probe failed runtime compilation; no Neural Engine training or speedup is claimed. Core ML prediction graphs containing manual VJPs are not by themselves an on-device training engine or optimizer integration.

AMD or additional GPU targets need independent qualified library primitives and layout/synchronization rules. Portability applies to the tensor identities and contracts; it does not imply identical kernel code, performance or floating bits on every device.

## 7. Accuracy and deployment qualification

Use strict native execution when reference arithmetic order is required. Algebraic mode must be qualified against a pinned runtime/target with representative data scales, cancellation, first and higher gradients, exceptional values, autocast, aliases/mutations and complete optimizer trajectories. The current finite samples are not universally bounded rounding certificates. The [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) discuss execution-dependent differences.

For the ReLU shared-mask path, branch masks must be checked from the actual current forward preactivations. Never reuse a mask merely because the previous step used it. Exact all-row equality permits a common diagonal; near-equality does not. Integrating mask/finite checks into a target kernel may reduce guard traffic, but that implementation and its costs remain to be measured.

A production qualification should compare original eager execution, native compiled execution, reassociated native autograd, the guarded backend and any custom kernel. Measure warm and cold runs separately, all requested gradients, optimizer steps, peak live allocation and real-device residency. Reject candidates that lose after copies/guards/recomputation. Treat relative/absolute gradient error and training behavior as separate checks.

For whole-model speed, account for the fraction f spent in the optimized segment. If its speedup is s and everything else costs the same, the total speedup is `1/((1−f)+f/s)`. Segment acceleration cannot be directly applied to optimizer time, data loading or unrelated layers. This is a timing decomposition, not a claim about a measured whole model.

## 8. Files and reproducibility

[CONTRACTION_OPTIMIZATION.md](CONTRACTION_OPTIMIZATION.md) records the initial result and assumptions. [NONLINEAR_AND_CONTRACTION_RESEARCH.md](NONLINEAR_AND_CONTRACTION_RESEARCH.md) records subsequent generalization and nonlinear experiments. [COMPILER_PLAN.md](COMPILER_PLAN.md) covers remaining production milestones. Sources are in `ojasx2/contractions.py`, `planner.py`, `nonlinear.py` and `compiler.py`; proofs are in `lean/OjasXMath/Contraction.lean` and `Nonlinear.lean`.

Run `python -B tools/verify_all.py` from the rebuild with the pinned Python dependencies and local Lean toolchain. The runner sequentially builds/audits Lean, checks Z3, runs tests, and records independent CPU benchmarks/training qualification. [verification.json](reports/verification.json) identifies checked source hashes. See [the verification report](reports/VERIFICATION.md) for trust boundaries and optional device failures. The source archive excludes local toolchains, environments and generated Core ML packages.
