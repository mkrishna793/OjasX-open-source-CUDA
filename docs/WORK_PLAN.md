The reversible-optic extension corrects the attached symplectic/domain/inversion claims, proves arbitrary-function coupling inverses and certified reconstruction contracts, and adds a first-order CPU chain prototype. Current totals: 139 Lean statements, 33 core plus eight reversible SMT identities, 274 CPU tests and two device skips. Three new SMT counterexamples refute universal symplecticity, stale-weight reconstruction and FP32 reversibility. Memory benefit and replay slowdown are both recorded. See [REVERSIBLE_OPTIC_THEOREMS.md](REVERSIBLE_OPTIC_THEOREMS.md).

The workload-scaling extension adds contiguous leading batch/token-axis support to the RMS core, 35 CPU checks across graph/shape families, seven scaled block measurements and two whole-graph measurements. The verification total is 128 Lean statements, 33 scoped SMT checks, 257 passing CPU tests and two device skips. A passing regression reproduces a negative FP32 BatchNorm-bias/AdamW optimizer qualification; universal numerical equivalence is not established. See [WORKLOAD_SCALE_MATH_TOOLS.md](WORKLOAD_SCALE_MATH_TOOLS.md).

The RMSNorm sufficient-core extension adds seven Lean statements and thirteen CPU tests, bringing verification to 128 Lean statements, 33 scoped Z3 checks, 222 CPU tests and two device skips. It delivers an actual supported PyTorch compiler rewrite, all-gradient and second-derivative tests, favorable/unprofitable CPU controls, and a hardware lowering design. The attached paired-dagger idea is evaluated separately from arithmetic compression. See [DAGGER_RADIAL_CORE.md](DAGGER_RADIAL_CORE.md). Native GPU lowering and bitwise equivalence remain unproved.

The primary product objective is reducing the arithmetic, data movement and live training state enough to run selected much larger workloads on one accelerator. CUDA independence and additional hardware support serve that objective. The updated [single-GPU implementation plan](OJASX_SINGLE_GPU_PLAN.md) defines separate capacity/throughput challenges, a joint mathematical and global residual planner, TileLang/Metal lowering, whole-model optimizer-state accounting, and evidence gates. It is a plan; no additional hardware execution or performance improvement is certified by it.

The subsequent contraction/nonlinear extension is complete within its stated scope: 121 Lean statements, 33 scoped Z3 checks, 209 CPU tests and two device skips. It adds demanded-gradient-aware long-chain planning, an exact-mask conditional ReLU backward, a native-forward structured LayerNorm backward and an experimental centered-weight Gram normalization graph. All original trainable gradients are preserved in qualified cases; timing, saved tape, tensor peak, controls and short optimizer trajectories are recorded. Developer/hardware and research guides are delivered. Universal floating error bounds, real-dataset qualification, automatic nonlinear compiler rules and physical GPU lowering remain pending.

See [HARDWARE_IMPLEMENTATION_GUIDE.md](HARDWARE_IMPLEMENTATION_GUIDE.md) and [NONLINEAR_AND_CONTRACTION_RESEARCH.md](NONLINEAR_AND_CONTRACTION_RESEARCH.md).

Earlier joint contraction phase (historical counts):

The joint contraction extension is complete within its stated scope: 96 Lean statements, 27 scoped Z3 checks, 164 CPU tests and two device skips. Consecutive linear layers with a narrow output now have a guarded algebraic rewrite, measured CPU speedup, saved-storage and profiler measurements, and an 80-step synthetic FP32 training qualification. Core ML conversion passed; this new graph could not execute because runtime compilation failed. See [CONTRACTION_OPTIMIZATION.md](CONTRACTION_OPTIMIZATION.md). Native GPU lowering, full nonlinear graph planning, real-dataset qualification and universally bounded floating error remain pending.

Foundational mathematical phase (historical counts):

The initial 7 October mathematical extension is complete within its stated scope: 81 Lean theorems, 24 Z3 checks and 136 CPU tests, plus two device skips. It supplies certified derivative/adjoint composition, general finite multilinear cotangents, zero-safe scalar sharing schedules, error propagation and real geometric tangent projections. This does not certify all ML primitives, native GPU lowering or a universal half-cost backward algorithm. See BACKPROP_MATH_STRUCTURE.md.

Earlier backend phase:

The strengthened math/backend prototype is complete: 47 Lean theorems, twelve Z3 checks, 103 passing CPU tests and two inaccessible-device skips, deep FFN and rank-7 storage/timing measurements, guarded FX rewrites, and Apple hardware probes. The original production directory is preserved. See STREAMED_DAGGER_THEOREM.md for the complete proof and current results; native GPU lowering and certification remain pending in COMPILER_PLAN.md.

1. Install pinned Lean 4 and mathlib locally; compile complete theorem sources and inspect their axioms.
2. Prove linear dagger laws, compositional pullbacks, matrix-gradient identities, Winograd equivalence and shared decoder cotangents, information-sufficiency constraints, and corrected cost/workspace bounds.
3. Use Z3 for concrete machine models: mask packing, layout domains, FP32 counterexamples, and optimizer legality with explicit assumptions. Keep real algebra and machine arithmetic claims separate.
4. Implement explicit residual-carrying lenses and correct PyTorch activations. Reduce exact ReLU residuals to a bit mask; retain sufficient GELU/SiLU inputs; use a documented RMSNorm statistic strategy.
5. Test forward/backward accuracy, numerical gradients, edge values, branched ML workloads, optimizer trajectories, actual GEMM events, and saved-tensor bytes. Distinguish bit equality from tolerance comparisons.
6. Explore shared multi-head projections and rank-7 products as algebraically valid experimental optimizations. Measure costs; do not assert universal bit equivalence or speedup.
7. Specify GPU-independent semantic IR, accuracy policies, residual planning, and backend contracts for CUDA, HIP, Metal, Vulkan/SPIR-V and CPU, with target-specific validation gates.
8. Publish a reproducible verification report that lists proved, tested, conditional and pending claims separately.
