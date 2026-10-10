# GPU-independent compiler plan

This document specifies the remaining production implementation. The delivered package now includes a checked mathematical core and a guarded FX backend prototype for GEMM and pointwise-memory rewrites. Native GPU code generation and full compiler certification remain pending. See STREAMED_DAGGER_THEOREM.md for the implemented prototype and measurements.

## 1. Portability contract

The portable part is the meaning of a graph, its cotangent flow, its residual requirements and its optimization obligations. Scheduling, instruction selection, layouts and floating accumulation are target-specific.

“Any GPU” should mean an extensible target interface: a device with a qualified backend can execute supported graphs, and unsupported features are reported clearly. It cannot mean one kernel runs efficiently on every GPU, every dtype is available, or bits match between all devices.

Use a tensor semantic IR above a hardware IR. MLIR is a candidate for later lowering: its [GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) models GPU execution, while [target-specific dialects](https://mlir.llvm.org/docs/Dialects/) expose hardware operations. Choosing MLIR is a proposed engineering direction, not a dependency installed or evaluated in this rebuild.

## 2. Semantic IR

Represent each value with shape constraints, element and accumulation dtype, layout/strides, device, alias set, version, ownership, and differentiation demand. Represent each primitive with:

| Field | Meaning |
|---|---|
| `primal` | Mathematical function and chosen floating operation sequence |
| `pullback` | Cotangent rule and higher-order differentiation policy |
| `residual` | Sufficient information, alternative encodings, decode and lifetime |
| `effects` | Mutation, aliasing, RNG, synchronization and external calls |
| `accuracy` | Strict reference, reproducible, or bounded-error contract |
| `guards` | Shape/dtype/device/domain conditions required by the implementation |
| `evidence` | Lean theorem reference, machine-model proof and conformance tests |

Initial primitives: linear/bilinear products, add and cotangent fanout, reshape/transpose/split/concat, reductions, ReLU, exact/tanh GELU, SiLU, SwiGLU, RMSNorm and explicit residual store/load. Keep trainable weights and requested gradients explicit. Attention/softmax, dropout, convolution, loss functions and optimizer fusion require additional rules; they are not covered merely because MLP rules work.

An example semantic operation is:

```text
linear(X, W) -> Y
  floating: PyTorch-compatible selected GEMM
  residual: references to X and W, guarded by alias/version rules
  cotangents_requested: {X, W}
  pullback: dX = dY @ W; dW = dY.T @ X
  proof: OjasXMath.matrix_product_pullback
```

If only `dX` is requested, retaining `X` solely for `dW` is unnecessary. If other consumers need it, storage remains live. This is the reason a global liveness planner is required.

## 3. Accuracy modes

| Mode | Allowed behavior | Acceptance requirement |
|---|---|---|
| Strict reference | Preserve qualified PyTorch operation order/dtypes/kernels; only machine-equivalent replacements | Raw-byte conformance against a pinned backend and runtime; exceptional-value behavior documented |
| Reproducible | Fixed implementation, reduction tree, RNG mapping and launch policy | Repeated runs match on the qualified target; no cross-vendor bit promise |
| Bounded error | Reassociation, packed projections, Winograd or mixed precision when qualified | Real proof plus target-specific numerical/error tests within the declared budget |

Strict is the default. The current Python `shared_projections` has `strict` and opt-in `algebraic` behavior; that experimental label is **not** a completed bounded-error compiler implementation. It must acquire quantitative error guards and fallback rules before becoming one.

Do not assume fusion preserves bits: FMA contraction, casts, intermediate precision, subnormal flushing, approximate transcendental instructions and changed reductions can all change results. PyTorch describes execution-dependent differences in its [numerical accuracy documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

Strict activation rewrites must also specify aliasing, saved-tensor version checks and behavior under mutation. The current tests cover immutable training workloads; they do not certify every user mutation pattern.

## 4. Pass sequence and invariants

1. **Capture and normalize.** Capture a guarded graph with PyTorch export/FX/AOT facilities. Preserve module parameters, training mode, RNG, mutation and unsupported graph boundaries. Required evidence: input/parameter/output equivalence before optimization.
2. **Annotate semantics.** Assign each primitive a real denotation, floating contract, gradient demand and residual options. An unknown operator keeps its qualified PyTorch execution boundary or fails capture; it must not receive a guessed derivative.
3. **Generate reverse flow.** Reverse composition, sum cotangents at fanout, return every requested parameter gradient, and attach residual uses. Bind derivative rules to operator mode: tanh GELU and exact GELU are distinct primitives.
4. **Eliminate dead work.** Use gradient demand, branch reachability and proven zero cotangents. Do not drop weight products based on a static dagger argument.
5. **Plan residuals globally.** Deduplicate aliases and account for tensors already needed by other consumers. Choose existing live outputs before adding flags, and flags before complete inputs when sufficient. Retaining the output of ReLU may already cost zero additional storage if a subsequent Linear needs it. Handle version checks and views before reusing storage.
6. **Apply proved shared circuits.** Use the four-add decoder dagger and seven-add encoder daggers when the numeric mode permits their order. Group projections only with matching inputs, dtype, layout, effects and compatible cotangent demand. Preserve branch accumulation.
7. **Choose product algorithms.** Evaluate dense GEMM versus rank-7 using actual dimensions, workspace, numeric mode and target measurements. Thin matrices and tiny products can reject rank-7 despite a square asymptotic saving. Include tape costs and all fourteen backward products.
8. **Lower and schedule.** Select backend operations, layouts, tiles, streams, synchronization and memory reuse from capability data. Prove/index-check bounded addresses before emitting a layout optimization.
9. **Validate and cache.** Run a target qualification suite and measured benchmarks. Cache only with exact graph, source, compiler, device, driver, dtype, accuracy and guard identities. Unqualified artifacts cannot silently count as optimized execution.

The original project's passthrough compile callback is not evidence that any of these stages exist. The rebuilt package intentionally makes no production compiler registration claim.

## 5. Residual planning objective

For each candidate representation `s_i`, require sufficiency for the requested pullback. Then optimize a weighted model of retained storage, forward temporary peak, backward temporary peak, recomputation FLOPs, additional GEMMs, dispatch count and numeric error. Treat them as separate quantities; lowering logical saved bytes can increase unique live storage.

Forward and backward liveness must be solved together. A seven-quadrant plus one-child recursive forward workspace bound does not constrain the entire training tape. Checkpointing is an explicit optional policy with a measured recomputation budget. The SwiGLU reference trades one pointwise recomputation for one fewer retained pointwise output; it introduces no projection recomputation.

Start with a deterministic greedy planner and exact storage accounting. Add a bounded optimization solver only after the model correctly predicts measured behavior. Z3 is appropriate for discrete feasibility/legality, not as a black-box proof that arbitrary runtime workloads are faster.

## 6. Target interface and backend qualification

Each target supplies capability discovery, tensor allocation/copy, library-GEMM access, pointwise/reduction lowering, subgroup/layout rules, barriers, launch limits, and timing/memory measurements. The interface must report unsupported operations explicitly.

| Target family | Proposed lowering route | First qualification |
|---|---|---|
| CPU | Existing PyTorch reference, then native lowering if worthwhile | Current reference tests and profiler/storage measurements |
| NVIDIA CUDA | Qualified library GEMM plus target-specific kernels; evaluate Triton/MLIR | Device activation conformance, linear VJPs, projection grouping, full training |
| AMD HIP/ROCm | Qualified library GEMM plus AMD-specific lowering | Same semantic suite with independent layout/reduction checks |
| Apple Metal | Dedicated Metal lowering or qualified existing PyTorch operations | Packed-mask integer operations, reductions, dtype limits, stream behavior |
| Vulkan/SPIR-V and additional targets | Backend-specific compute lowering | Feature discovery, scalar reference, synchronization and supported dtype checks |

These are planned routes. No listed GPU backend was built in this phase. Do not treat a layout proved for a 32-bank model as valid for every target or instruction. Check subgroup sizes, bank mappings, access width and alignment against the actual device.

A CPU fallback can be an explicitly selected diagnostic mode. Silent GPU-to-CPU transfers would obscure execution costs and should not be used to claim GPU support.

## 7. Proof and testing pipeline

Every rewrite needs four separate forms of evidence:

- **Semantic theorem:** exact real/ring identity and all domain assumptions, checked in Lean.
- **Residual obligation:** information sufficiency, aliases/versioning and lifetime safety; global schedule when memory is claimed.
- **Machine obligation:** floating and integer semantics, bounded addressing, race freedom and barriers under the specific backend model. Z3 can discharge bounded subproblems; layout examples are not a race-freedom proof for full kernels.
- **Implementation qualification:** raw-bit or tolerance differential tests, numerical derivative checks, branched gradients, optimizer trajectories, real device profiling and memory observations.

The current manifest hashes checked sources and records results. Future proof-carrying IR must mechanically establish that the emitted operation sequence instantiates the theorem and guards. Textually attaching a theorem name or source hash is insufficient. Neither Lean, Z3, nor testing alone provides every item above.

## 8. Delivery milestones and completion gates

| Milestone | Status | Completion gate |
|---|---|---|
| M0: mathematical/reference core | Delivered | Lean + Z3 + CPU tests + honest saved-storage measurements |
| M1: guarded strict graph backend | Prototype delivered | Actual FX GEMM/SwiGLU rewrites pass the scoped CPU suite; full semantics coverage and native lowering pending |
| M2: global residual planner | Pending | Unique live-storage estimates agree with measured whole-model peaks and alias/version safety |
| M3: one physical GPU backend | Pending | Complete device suite, workloads, profiling and numerics; explicit unsupported paths |
| M4: qualified bounded-error optimization | Pending | Shape-aware packed/rank-7 decisions; backward-inclusive cost; error and memory budgets |
| M5: additional GPU families | Pending | Independent device capability and conformance evidence for each family |
| M6: certified IR refinement | Pending | Mechanized link from IR semantics and guards through transformation to emitted programs |

The physical laptop has an Apple M5 Metal-capable GPU. This tool process reports zero Metal devices and MPS inaccessible. Core ML advertises a Neural Engine but the tested gradient graph selects CPU and reports IOSurface/ANE compilation errors. Apple hardware lowering is therefore the natural next target once a usable device context is available; no Apple acceleration is certified by these CPU results.

## Mathematical qualification extension (7 October)

The [certified backpropagation structure](BACKPROP_MATH_STRUCTURE.md) adds actual derivative and adjoint obligations to primitive contracts, closure under composition, cotangent uniqueness/accumulation and certified residual re-encoding. The joint finite multilinear theorem supplies a candidate cotangent rule for tensor contractions. Schedule costs must include reconstruction and compare against ordinary AD with its retained forward intermediates. No new runtime rewrite is enabled by these proof additions. Connecting the general encoded circuit to a flattened-coordinate certified-stage constructor, formal graph fanout/parameter-sharing nodes and compiler refinement remains pending.


## Implemented joint contraction extension (7 October)

The new [contraction policy](CONTRACTION_OPTIMIZATION.md) recognizes adjacent single-consumer captured Linear nodes. `contraction_policy="linear_chain"` with `accuracy="algebraic"` uses native autograd on the reassociated graph; `"linear_chain_backward"` retains native forward evaluation and changes the joint pullback. Joint mode supports affine bias; backward-only bias falls back. Shape/dtype/device/autocast and gradient-demand guards preserve native fallback. No rewrite crosses activation/dropout or inner fanout. The qualified runtime is 2D CPU FP32/FP64; other targets remain native. Lean supplies arbitrary-finite-dimension schedule identities; Z3 checks a concrete symbolic matrix instance and conditional cost/capacity models. CPU timing, tensor-memory accounting and short synthetic training establish a conditional improvement, not a whole-model or GPU claim. Numerical conditioning guards, hardware cost fitting and global contraction/liveness planning remain required for production deployment.


## Further implemented contraction/nonlinear research

`contraction_policy="matrix_chain"` recognizes maximal adjacent bias-free single-consumer Linear chains and dispatches a CPU/dtype/autocast-guarded scalar-cost DP plan including demanded first-order gradients. It changes no original trainable leaves and preserves nonlinear/fanout boundaries. Hardware cost and global liveness/refinement remain pending. The new explicit `relu_linear_chain`, `layernorm_linear_chain` and experimental `gram_layernorm_linear_chain` adapters qualify different nonlinear structures; they are not automatic FX nonlinear passes. The Gram variant changes floating output order and uses heuristic conditioning/native-fallback guards, not certified error bounds. See [the hardware guide](HARDWARE_IMPLEMENTATION_GUIDE.md) and [research report](NONLINEAR_AND_CONTRACTION_RESEARCH.md).
