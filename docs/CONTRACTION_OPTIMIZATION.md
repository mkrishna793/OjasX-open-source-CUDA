# Joint contraction optimization: proved mathematics and measured CPU improvement

For a particular linear segment, we now have a substantial reduction in arithmetic and a measured speedup. For two consecutive Linear layers, batch 256 and widths 1024 → 2048 → 32, forward plus input and both original weight gradients took **5.6216 ms natively and 1.0633 ms through the guarded joint schedule: 5.29× faster** on the recorded CPU. The conventional scalar-operation model removes **86.42%** of arithmetic. Fresh autograd-saved storage falls from 2 MiB to 128 KiB (**93.75%**); profiler incremental tensor peak falls from 11.28125 to 9.40625 MiB (**16.62%**).

These are separate quantities. This does not mean 93.75% less total training memory or a 5.29× faster whole model. Both original weight matrices remain trainable and all requested gradients are returned. There is no truncated rank, frozen parameter, changed model width or kernel-fusion requirement. This is known matrix associativity applied to the joint forward/pullback graph, rather than a claim of a historically new universal backpropagation algorithm.

The initial measurements below are preserved in [the initial benchmark snapshot](reports/contraction-benchmark-initial.json). The subsequent [nonlinear/contraction extension](NONLINEAR_AND_CONTRACTION_RESEARCH.md) and [hardware implementation guide](HARDWARE_IMPLEMENTATION_GUIDE.md) generalize the implementation; current project counts below include that extension.

## What was found inside tensor multiplication

For an ordinary real matrix product `Y = A B`, the pullback is `dA = G Bᵀ`, `dB = Aᵀ G`. The derivative's adjoint, composed in reverse, is the relevant categorical operation. For a two-product segment, conventional reverse mode creates a wide hidden activation and a wide hidden cotangent. Changing the contraction order can avoid both intermediates without discarding either weight gradient.

The inspected [PyTorch 2.8 Linear implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Linear.cpp) uses matrix multiplication and, for eligible biased inputs, `addmm`. The new backend therefore operates on actual captured Linear nodes rather than replacing arbitrary ML operators with a speculative derivative.

## One joint identity, arbitrary finite real matrix dimensions

Let `A` have shape `m × k`, `B` shape `k × n`, `C` shape `n × p`, and upstream cotangent `G` shape `m × p`. Write `Y = (A B) C`. Compute just two shared contractions:

```text
P = B C                 # k × p
T = Aᵀ G                # k × p

Y  = A P
 dA = G Pᵀ
 dB = T Cᵀ
 dC = Bᵀ T
```

The ordinary chain rule gives `(G Cᵀ) Bᵀ`, `Aᵀ(G Cᵀ)`, and `(A B)ᵀ G`. Associativity and the transpose product law establish equality to the three expressions above. The existing derivative/pairing and composition proofs establish why these are cotangents; the new [Contraction.lean](lean/OjasXMath/Contraction.lean) proves the schedule identities for arbitrary finite index types. It adds fifteen statements, all kernel-checked with audited standard Lean dependencies and no unfinished proofs.

For PyTorch's row-oriented weights this becomes:

```text
P = W2 @ W1
T = G.T @ X
Y   = X @ P.T
 dX  = G @ P
 dW1 = W2.T @ T
 dW2 = T @ W1.T
```

The implementation computes six GEMMs with smaller dimensions instead of six GEMMs with the original dimensions. It reduces the arithmetic inside these products; it does not claim six independent dense products become one hardware multiplication. Plain PyTorch autograd on the reassociated forward graph already realizes this sharing. It took **0.9762 ms** on the largest sample, slightly faster than the guarded adapter. The public joint path therefore uses native autograd; the explicit custom dagger remains a qualified reference and supplies the backward-only alternative.

## Conditional cost theorem

Counting one scalar multiply or add as one operation, an `a × b` by `b × c` product costs `2abc − ac`. With the input and both weight gradients demanded, the native joint cost is:

```text
6mkn + 6mnp − (2mn + mp + mk + nk + pn)
```

The reassociated joint cost is:

```text
6pk(m+n) − (2pk + mp + mk + nk + pn)
```

Lean checks the exact count identities and difference. It also proves that, for positive dimensions with `4p ≤ m` and `4p ≤ n`, the reassociated **leading arithmetic term is less than half** of the native leading term. This is a shape-dependent arithmetic result, not a latency theorem. The Python selector includes actual subtraction terms and requested-gradient demand. Bias work, copies, memory traffic and dispatch are excluded from that selector's scalar model.

If `p` is wide, reassociation can lose. The control sample below rejects the optimization and uses the native graph, incurring small dispatch/guard overhead. Forcing reassociation on that control was slower. Model cost must eventually be supplemented with target measurements and numerical conditioning checks.

## Recorded CPU measurements

FP32, PyTorch 2.8.0, arm64 CPU, two threads. Each row uses the same original inputs and weights, three warmups and eleven randomized-order paired samples. The timed work includes forward and input/both-weight gradients, excludes optimizer steps, and runs separately from the proof/training jobs.

| Shape m × k × n × p | Native ms | Joint ms | Joint speedup | Native-forward / changed-backward ms | Speedup |
|---|---:|---:|---:|---:|---:|
| 128 × 256 × 512 × 16 | 0.2151 | 0.0984 | 2.19× | 0.1739 | 1.24× |
| 256 × 512 × 1024 × 32 | 1.1623 | 0.3089 | 3.76× | 0.7053 | 1.65× |
| 256 × 1024 × 2048 × 32 | 5.6216 | 1.0633 | 5.29× | 3.7626 | 1.49× |
| 128 × 256 × 512 × 256 | 0.3269 | 0.3337 | 0.98× | 0.3340 | 0.98× |

The wide-output control's guarded joint/backward paths fall back to native. The [benchmark JSON](reports/contraction-benchmark-initial.json) includes all timing samples, errors, saved-storage accounting and profiler event counts. Raw tensor-memory timelines are included beside it. Fresh storage excludes supplied inputs and parameters; profiler incremental tensor peak excludes initially live supplied tensors and does not measure process resident memory, allocator caches, Core ML allocations or GPU memory.

In the largest sample the backward-only path took **3.7626 ms (1.49× faster)**. It keeps the native forward evaluation and its sampled forward bits, but discards the wide hidden residual and changes the backward contraction order. It retains required inputs/weights, recomputes the effective weight for the input gradient, and uses seven GEMMs. Fresh saved temporary storage was zero and profiler incremental tensor peak was 15.51% lower. Zero fresh tape does not mean zero storage or zero recomputation.

## Accuracy and training qualification

Real-algebra equality is exact. Floating-point reassociation changes rounding, so the joint path is **not bit-identical** to the original evaluation. On the largest FP32 sample, relative L2 differences were `1.41e-6` for output, `1.16e-6` for input gradient, `4.33e-7` for first weight gradient and `9.08e-7` for second weight gradient. All sampled comparisons passed `rtol=3e-4, atol=3e-5`. These are observations, not exhaustive error bounds for cancellation, overflow, underflow, exceptional inputs or arbitrary scales.

The [training qualification](reports/contraction-training.json) uses the same two dense parameter matrices, deterministic batches, SGD with momentum and 80 FP32 steps on a synthetic eight-class teacher task. All three paths achieved **79.296875% held-out accuracy**, with **100% agreement on predicted classes**. Joint final loss was `0.1144675612` versus native `0.1144675389`; maximum sampled step-gradient relative L2 discrepancy was `2.30e-6`. This establishes one short trajectory, not general real-dataset convergence or unchanged training quality for LLMs.

The new twenty-eight CPU tests also cover FP64 numerical first/second derivatives, all gradient-demand patterns, original parameter identity, a 20-step Adam trajectory, affine biases, mutation checks, noncontiguous cotangents, strict/native fallback, autocast fallback and graph boundary guards. Full project verification is **121 Lean statements, 33 scoped Z3 checks, 209 CPU tests passed, two device tests skipped**. See [verification.json](reports/verification.json). Lean does not mechanically certify the Python interpreter or emitted kernels.

## Backend use with an ordinary model and optimizer

```python
import torch
from ojasx2 import make_backend

compiled = torch.compile(model, backend=make_backend(
    accuracy="algebraic",
    contraction_policy="linear_chain",
))
# Existing model parameters, loss, optimizer and backward calls remain usable.
```

`contraction_policy="linear_chain_backward"` selects native-forward/reassociated-backward behavior. Strict default execution keeps native operations. The opt-in algebraic label permits rounding differences; it is not a completed universally bounded-error mode.

The graph rule recognizes adjacent single-consumer Linear nodes, including bias in joint mode. It never crosses an activation, dropout or an intervening operation, and rejects an inner activation with additional consumers. Runtime qualification is positive matching 2D CPU real FP32/FP64 with no CPU autocast; unsupported devices/shapes/dtypes fall back natively. Joint bias uses `W2 @ b1 + b2`, preserving ordinary bias parameters and native differentiation. Backward-only biased chains currently fall back to native. The affine forward/second-weight identities are proved; every bias-gradient case relies on native autograd plus differential tests, not a separate complete Lean proof of broadcasting gradients.

## Exact factors, nonlinear rank growth and rejected approximation

For a single GEMM, if its cotangent is **already supplied** as an exact factorization `G = L Rᵀ` of inner dimension `r`, then:

```text
dA = L (B R)ᵀ
dB = (Aᵀ L) Rᵀ
```

Lean proves these identities and the leading backward arithmetic `4kr(m+n)` versus dense `4mkn`. The helper `factored_gemm_pullback` exposes this known-factor case. Discovering factors via SVD is additional work; it is not implicitly free. A narrow output supplies useful structure in a consecutive linear chain, but a nonlinear mask can destroy that structure: a rank-one all-ones matrix multiplied elementwise by the 2×2 identity mask has rank two. The suite tests this obstruction.

The spectral experiment tried retaining eight singular components of a generic dense cotangent and produced **87–89% relative gradient error**. Accordingly, no rank truncation is enabled. If `E = G − G_r`, the bounds `‖E Bᵀ‖F ≤ ‖E‖F ‖B‖2` and `‖Aᵀ E‖F ≤ ‖A‖2 ‖E‖F` explain how singular-value tails can amplify into gradient errors; these matrix-specific bounds are numerically checked here, not newly formalized in Lean. The existing general Lean backward-error theorem handles local error amplification under supplied constants.

[GaLore](https://arxiv.org/abs/2403.03507) targets optimizer-state memory using projected gradients, and [PowerSGD](https://arxiv.org/abs/1905.13727) targets gradient communication. Their benefits do not supply a proof that our arbitrary full-precision cotangents can be deleted without approximation. Manifold constraints likewise need an explicitly constrained model, metric and update rule.

## What physics contributes

The useful hardware constraint is compute capacity versus data movement. If actual work `F ≤ tP` and actual transferred bytes `Q ≤ tB` for positive compute capacity `P` and bandwidth `B`, Lean proves `t ≥ max(F/P, Q/B)`. This conditional roofline bound explains why fewer FLOPs alone need not produce lower latency; all supplied traffic/capacity estimates still require hardware evidence. The [GEMM performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) describes how arithmetic intensity, memory movement and matrix dimensions affect execution.

The optimization changes which tensors must be materialized and retained. Hardware ultimately loads operands into registers and performs arithmetic, but calling that “moving GEMM from memory into ALUs” is not itself an algorithm. Dense input/weight-gradient outputs still require their full output storage. Tiling, avoiding large intermediates, and reusing small contractions can reduce traffic; the next backend must measure actual traffic and total live storage. A universal thermodynamic or quantum reduction in classical exact training work has not been established.

## Core ML and the next work

Both explicit native and joint forward/all-three-VJP graphs converted successfully to Core ML ML Programs with dynamic weights, six MIL matmuls and FP16 at shape `64 × 128 × 256 × 8`. Execution of this new probe was blocked because Apple's compiler could not create a runtime working directory, including an attempt using writable `/private/tmp`. See [Core ML contraction report](reports/coreml-contraction.json). Its process exited normally because it records per-model failures; that is **not** a passing numerical execution check. The prior smaller Core ML probe ran on CPU. No GPU or Neural Engine execution or speedup is confirmed by either result.

The next useful generalization is a joint contraction planner that searches forward and pullback paths together, preserves all requested gradients and treats nonlinear residuals as explicit boundaries. Longer purely linear segments, factored cotangents where graph structure provides exact factors, and repeated projections are candidates. Attention, convolutions and nonlinear FFNs require their own identities, rounding contracts and liveness measurements. Reversible architectures and checkpointing are additional memory strategies with architecture or recomputation tradeoffs, rather than consequences of a free dagger operation. The current [compiler plan](COMPILER_PLAN.md) records the required graph, numerical, device and refinement work. Whole-model GPU training, real-dataset qualification and certified machine-code lowering remain pending.
