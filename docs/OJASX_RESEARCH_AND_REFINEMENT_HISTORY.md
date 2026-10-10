# OjasX research and refinement history

Consolidated on 8 October 2026; updated through version 2.4 on 9 October 2026. This is the research record. The companion [OjasX mathematical foundation](OJASX_MATHEMATICAL_FOUNDATION.md) is the current specification to use when extending the mathematics or implementing the backend.

## What the work has produced

OjasX has a checked abstract backpropagation framework, conditional algorithms that compute all original gradients through smaller representations, sufficient-residual memory rules, bounded automated discovery, and scoped Apple GPU implementations. Its mathematics combines derivative adjoints, multilinear algebra, contraction planning, invariant statistics, polynomial feature spaces, finite Koopman representations, and numerical error contracts.

The strongest exact-real nonlinear results concern existing radial blocks and a structured triangular recurrence. The latest coordinate-sensitive results concern bounded polynomial approximations to GELU/SiLU with very narrow inputs. Some qualified examples have substantial speed and saved-activation reductions. Most-model acceleration, a universal compact nonlinear representation, bitwise equivalence after reassociation, minimum total memory, and one GPU replacing three or four have not been established.

The original version 1.8 consolidation improved specification/evidence organization. Version 1.9 adds the numerical repairs described below. No new foundational axioms or speed measurements are claimed.

## How to read the evidence

| Evidence level | What it establishes | What remains separate |
|---|---|---|
| Lean kernel-checked theorem | Its exact statement under its explicit hypotheses | Executability, Python/Metal implementation correctness, floating-point error and speed |
| Rocq theorem and independent kernel recheck | Its rational/algebraic contract and closed assumptions | An efficient GPU implementation or correctness of every extracted deployment |
| Z3 counterexample or refutation | The property encoded in the solver's stated domain | Unencoded analytic, numerical, hardware or workload properties |
| Symbolic synthesis | A candidate expression and bounded search outcome | Kernel proof, numerical qualification and measured benefit |
| GPU comparison | Sampled values, gradients, optimizer behavior or timing on the recorded device | Every input, every model, long training or another GPU |
| Saved-storage measurement | Retained tensor storage under the report's accounting rules | Total peak RAM, allocator workspace, weights, dense gradients and optimizer states |

The current whole-project audit reports **547 Lean theorem declarations and 266 Rocq theorem/lemma declarations**. These are statement counts, not 813 independent inventions. Some generated statements specialize the same underlying identity. Lean uses the standard audited axioms propext, Classical.choice and Quot.sound; the audit finds no added optimization axioms. Rocq assumptions are closed and independently rechecked.

The authoritative current bundle is [formal verification](reports/spectral-formal-verification.json), [bounded discovery](reports/spectral-discovery.json), [GPU observations](reports/spectral-gpu.json), and [source-matched qualification](reports/spectral-qualification.json). Earlier reports retain their original scopes and historical counts. They should not be added together or treated as current global source-hash certifications.

## 1. Architectural audit and the first mathematical core

The original project and supplied architectural/monograph notes motivated a separate verified reference rebuild. The original OjasX implementation was preserved. This reference core does not replace its entire allocator, GPU driver or production compiler. The [combined project audit](../OjasX-combined-analysis.md) records the earlier architecture analysis.

The first useful clarification was that reverse mode uses the **adjoint of the derivative at the actual forward state**. For a nonlinear function f, this is Df(x)*, not an inverse activation and not a fixed matrix independent of x.

For a matrix layer Y = XW + b and incoming cotangent Δ, the original requested gradients remain

$$
\bar X=\Delta W^T,\qquad \bar W=X^T\Delta,\qquad
\bar b=\sum_i\Delta_i.
$$

Category theory supplies composition and accumulation laws. Arithmetic savings require an additional representational or contraction identity. Renaming these products as daggers does not remove them.

The core bilinear circuit y = D((Ua) ⊙ (Vb)) has shared reverse factor d = Dᵀg and joint pullback

$$
\bar a=U^T(d\odot Vb),\qquad
\bar b=V^T(d\odot Ua).
$$

Streaming channels can avoid retaining every encoded intermediate. It does not prove that two requested gradients have become one cheaper dense multiplication. Sum/difference packing of two outputs remains an invertible representation of both outputs and has floating overflow/rounding counterexamples.

Proofs: [Algebra](lean/OjasXMath/Algebra.lean), [Composition](lean/OjasXMath/Composition.lean), [StreamedDagger](lean/OjasXMath/StreamedDagger.lean). Runtime: [bilinear](ojasx2/bilinear.py), [streamed](ojasx2/streamed.py).

## 2. Certified composition, residual information and multilinear gradients

The foundation was strengthened from an adjoint formula to stages carrying an **actual derivative proof** and a residual that reconstructs that derivative. Inputs may be product spaces containing the data and every original parameter.

The residual theorem says a representation is sufficient precisely when inputs with the same representation have the same pullback for every cotangent. This gives a correctness criterion, not an automatically efficient encoder/decoder.

Counterexamples drove important restrictions:

- Squaring loses sign: equal forward outputs can require different backward slopes.
- A constant-output affine layer can still have input-dependent weight gradients.
- An arbitrary independent Boolean ReLU mask needs enough information to distinguish its possible masks; fixed-width lossless encoding cannot universally shrink n independent bits below n bits.
- Tiny residuals do not erase dense parameter-gradient output requirements.

Multilinear product rules were made division-free, including at zeros. A prefix/rolling circuit beats independently recomputing every product of the other factors. Its attractive count against that naive baseline does not imply the same improvement against an already efficient retained-prefix reverse-mode implementation.

Sphere and Stiefel projection identities add geometry for explicitly constrained models. They establish selected tangent/projection properties; they do not invisibly replace Euclidean training or prove general constrained-optimizer convergence.

Proofs: [CertifiedBackprop](lean/OjasXMath/CertifiedBackprop.lean), [Residuals](lean/OjasXMath/Residuals.lean), [Multilinear](lean/OjasXMath/Multilinear.lean), [Geometry](lean/OjasXMath/Geometry.lean).

## 3. Joint linear contraction planning

An adjacent linear pair can be evaluated through its narrow projection core:

$$
H=(XW+b)Q=X(WQ)+bQ.
$$

The optimized expression is differentiated with respect to X, W, b and Q. WQ remains derived data, recomputed after parameter changes. Training it as an independent parameter would change the model and lose original gradients.

This work extended to longer matrix chains and demand-aware contraction plans. Savings depend on dimensions, output requirements and the need to recover each original weight gradient. The compiler must account for the backward schedule as well as the forward reassociation.

Recorded favorable CPU experiments supplied early algorithmic evidence; later MPS qualification supplied GPU evidence. Neither establishes a gain for every shape. Current repeated large linear-block results are summarized below.

Proofs: [Contraction](lean/OjasXMath/Contraction.lean), [GradientFactors](lean/OjasXMath/GradientFactors.lean). Runtime: [contractions](ojasx2/contractions.py), [planner](ojasx2/planner.py), [gradient factor compiler](ojasx2/gradient_factor_compiler.py).

## 4. Pointwise memory and reversible coupling

Packed ReLU derivative masks and activation residual policies demonstrated that saved backward information can be smaller than an ordinary full-value tape in selected graphs. Alias overlap matters: packing an output already retained by a following Linear can fail to lower unique storage.

Reversible coupling reconstructs earlier states for an existing reversible architecture. The depth-64 historical CPU example reduced retained activation tape by 99.22% but performed extra replay and was slower. It is evidence of a memory/computation tradeoff, not a speed breakthrough for arbitrary architectures.

The cached reverse refinement retains selected preactivations, removes redundant projection replay, and implements a native Metal SiLU reverse step. Two recorded M5 runs measured **1.06–1.11× native PyTorch joint throughput**, **1.60–1.70× the earlier replay prototype**, and **46–49% less fresh activation tape**. These scopes must not be confused with the earlier 99.22% replay memory result.

Floating reconstruction drift, large coefficients and activation derivatives refined the original exact-real coupling argument. Numerical reconstruction and differentiation need their own contract.

Proofs: [ReversibleOptic](lean/OjasXMath/ReversibleOptic.lean), [CachedReverse](lean/OjasXMath/CachedReverse.lean). Runtime: [reversible](ojasx2/reversible.py), [cached reverse](ojasx2/cached_reverse.py). Evidence: [cached GPU report](reports/reversible-cache-mps.json).

## 5. Rocq, extraction, structured Jacobians and broader compatibility

An isolated Rocq 9.2 / OCaml 4.14.2 toolchain was installed beside Lean 4.34 / Mathlib 4.34. Exact rational reference algorithms were extracted and exercised historically. Extraction preserves the proved source computation within its assumptions; it is not a blanket claim that every FFI, compiler, hardware or deployment is free of runtime bugs.

Rocq added matrix derivative/adjoint contracts, composition, branching, finite repetition, structured diagonal-plus-low-rank pullbacks, indexing, optimizer preservation and conditional planner costs. Gather's pullback is scatter-add; duplicate indices must accumulate.

Softmax and normalization have structured Jacobian contractions already useful in ordinary reverse-mode implementations. Recognizing that structure avoids a full n×n Jacobian. It does not by itself establish a new improvement over PyTorch's existing vector formulas.

Normalization refinements preserve epsilon corrections and original affine gain/bias gradients. A logical normalized-value residual can use n values plus one inverse scale per row. Graph references, aliases, transient tensors and allocator peak remain separate.

Historical CPU compatibility tests cover linear chains, GELU MLPs, SwiGLU, convolution, softmax attention, LayerNorm/RMSNorm, shared-weight recurrence, repeated-index embeddings, fixed-dropout residuals, polynomial research gates and an unrolled ODE. They check selected derivative behavior; they are not formal proofs or GPU speed qualification of those complete models.

A significant negative result is retained: a Conv2d-bias → BatchNorm composition trained with AdamW had tiny local gradient differences but about **0.00104** maximum bias-parameter deviation after three steps, beyond the 0.0002 parameter contract. Local gradient tolerance alone is insufficient for optimizer trajectory qualification.

Sources: [Rocq foundation](ROCQ_MATH_FOUNDATION.md), [refinement](FOUNDATION_REFINEMENT.md), [broad CPU reference report](reports/broad-foundation.json), [edge report](reports/foundation-edges.json), [workload report](reports/workload-scaling.json).

## 6. Matrix multiplication candidates

Known Strassen and Strassen–Winograd schedules were formalized for joint forward/input-gradient/weight-gradient computation. In a single 2×2 block decomposition the modeled counts are:

| Schedule | Block products | Block additions |
|---|---:|---:|
| Conventional three matrix products | 24 | 12 |
| Strassen | 21 | 54 |
| Winograd | 21 | 45 |

With product cost M and addition cost A, the modeled non-increase conditions are M ≥ 14A and M ≥ 11A respectively. More additions, scratch and dispatch can defeat the multiplication reduction. Historical CPU references were slower in all eight measured cases; automatic dispatch is disabled.

This is verification and adaptation of known algorithms. It is not evidence of a newly invented universally faster GEMM or a current qualified GPU replacement.

Proof: [RankSeven.v](rocq/RankSeven.v). Reference: [math foundation](ojasx2/math_foundation.py). Evidence: [broad report](reports/broad-foundation.json).

## 7. Abstract engine and automated polynomial discovery

Mathlib supplies real complete inner-product spaces, continuous linear derivatives and adjoints. The abstract stage contract proves composition, cotangent accumulation, identical-primal gradient preservation, sufficient-residual substitution and exact-gradient optimizer preservation.

Bounded discovery introduced polynomial basis refinement: reject insufficient bases, add missing monomials, derive coefficients, refute the negated identity, generate proof sources, and let the kernels check them. Homogeneous polynomial tensor expansion is general over arbitrary finite dimensions and degree. Efficient symmetric grouping and runtime qualification have narrower scopes.

The central limitation became explicit: a polynomial basis can grow faster than the computation it was intended to compress. A mathematical representation is useful only after construction, reverse contraction, original parameter-gradient recovery, memory and numerical costs are counted.

Proof: [AbstractEngine](lean/OjasXMath/AbstractEngine.lean). Earlier circuit: [feature core](ojasx2/feature_core.py). Discovery: [discover_backprop.py](tools/discover_backprop.py). Guide: [abstract engine](ABSTRACT_BACKPROP_ENGINE.md).

## 8. Early GPU nonlinear approximation

The first GPU-only GELU/SiLU/SwiGLU experiments used small-domain Taylor surrogates, typically with gate radius at most 0.05 and input widths 1, 2 or 4. The forward polynomial and backward polynomial derivative were matched, with all original operand gradients.

The retained activation storage improved, but the final guarded nonlinear references remained slower and were not speed-admitted. A normalized exact-real linear control passed sampled GPU checks; an unscaled stress control failed its numerical tolerance. Both outcomes remain recorded.

These experiments motivated the later larger-domain degree-8 refinement. They do not show that an approximate derivative can be substituted into an unchanged activation forward pass without changing the gradient semantics.

Sources: [nonlinear research](GPU_NONLINEAR_DISCOVERY.md), [NonlinearBounds](lean/OjasXMath/NonlinearBounds.lean), [TaylorDiscovered.v](rocq/TaylorDiscovered.v), [nonlinear GPU search](reports/nonlinear-gpu-search.json).

## 9. Closed nonlinear coordinates and Koopman representations

The compiler searched identity, log, reciprocal and logit charts in a bounded operation grammar. A log representation of positive power/scale chains has a valid positive-domain real identity, but FP32 error forced exponent guards and the qualified active candidates remained slower.

The successful structured recurrence is

$$
f_{a,b,c}(x,y)=(ax,by+cx^2),\qquad E(x,y)=(x,y,x^2).
$$

It has a closed three-observable action

$$
E(f(x,y))=
\begin{pmatrix}a&0&0\\0&b&c\\0&0&a^2\end{pmatrix}E(x,y).
$$

Original distinct stage parameters or shared parameters remain trainable. Ordered operator composition can propagate the large batch once instead of once per stage. Repeated separate-parameter GPU examples measured **2.40–2.61× at depth 32** and **5.20–6.07× at depth 64**, with **95.63% and 97.25% less fresh retained activation storage**. The reported state shape was [512,32,2].

Repeated squaring rejects the tested fixed monomial bases because the highest degree doubles. This refutes those bases, not every imaginable representation. Ordinary GELU/SiLU was not shown to possess an equally cheap exact closed basis.

Sources: [coordinate compiler](MATHEMATICAL_COMPILER.md), [coordinate runtime](ojasx2/coordinate_compiler.py), [CoordinateCompiler proofs](lean/OjasXMath/CoordinateCompiler.lean), [coordinate qualification](reports/coordinate-qualification.json).

## 10. Joint activation and all-gradient factor refinement

The refinement treats activation and reverse contraction as one problem:

$$
Z=XW+b,\quad H=\phi(Z)Q,\quad
G=PQ^T,\quad \Delta=\phi'(Z)\odot G.
$$

Linear, square and certified shared-mask ReLU cases have exact-real factor rules. A Hadamard product of factors requires the product feature index; a rank-s factor and rank-t factor can need up to st represented channels. Truncating that index back to s loses information without a separate approximation contract.

The GPU suite recorded **83 checks with zero failures** across correctness, size scaling, edges, higher derivatives, composed graphs and primitive slopes. At [8192,128,4096,2], repeated linear full-step gains were **2.23–2.62×**, and square gains were **1.36–1.61×**. Square reduced fresh saved storage by 50%; the linear block reduced 134,217,728 bytes to 1,024 bytes.

Small shapes remained slower. ReLU did not pass repeated speed admission, and heterogeneous masks retained native execution. Ordinary dense GELU/SiLU had no admitted exact factor rule.

Sources: [activation/backprop refinement](ACTIVATION_BACKPROP_REFINEMENT.md), [GradientFactors](lean/OjasXMath/GradientFactors.lean), [gradient GPU report](reports/gradient-factor-gpu.json), [gradient ledger](reports/gradient-qualification.json).

## 11. Invariant radial compilation

For an existing radial block, φ(z) = f(mean(z²))z, the compiler uses a small input-space Gram core and projection core. Bias is included; every original X/W/b/Q gradient and trainable sigmoid-gate gradient is recovered. It does not train the collapsed cores independently.

With A = [X,1], V = [W;b], C = VVᵀ and T = VQ:

$$
s_i=\frac{(AC)_i\cdot A_i}{N},\quad u=AT,\quad H_i=f(s_i)u_i.
$$

This avoids wide batch-by-hidden activations for eligible shapes. It builds a (k+1)×(k+1) Gram matrix, not a batch-by-batch matrix.

Counterexamples rejected several proposed generalizations:

- A radial Jacobian is generally full rank. Rank-one incoming rows across different forward states can become rank three; the recorded cubic witness has determinant 12.
- A fixed commutator argument for linear maps does not commute arbitrary nonlinear activations.
- Thin sketches are noninjective; an exact trace identity does not recover missing gradient information.
- An isotropic average correction estimate is not a uniform bound for aligned gradients.
- Equivariance alone needs explicit representation/multiplicity assumptions and is not a license to discard arbitrary learned weights.

At [8192,128,4096,2], repeated cubic, reciprocal and trainable sigmoid radial blocks measured **4.62–6.09×** full-step gains with about **96% less fresh saved storage**. A complete two-block radial cross-entropy research model measured **5.87–6.49×** with **98.52% less fresh saved storage**; its storage hook excluded the loss tape although timing included loss.

Three radial–LayerNorm–radial cases failed the relative-gradient criterion near cancellation and retain native execution. A stable joint normalization identity was proved at this phase. Version 1.9 implements it, while sensitive full-model relative qualification remains unsuccessful. Certified chamber ReLU cases were correct but slower, with no saved-storage gain.

Sources: [invariant refinement](INVARIANT_FOUNDATION_REFINEMENT.md), [InvariantRadial proofs](lean/OjasXMath/InvariantRadial.lean), [radial runtime](ojasx2/invariant_radial_compiler.py), [composition GPU report](reports/invariant-composition-gpu.json), [invariant ledger](reports/invariant-qualification.json).

## 12. Latest coordinate-sensitive spectral refinement

The latest supplied idea combined Hermite expansion, symmetric polynomial features and Khatri–Rao contractions. Its valid polynomial factor identity was retained, while unsupported symmetry, distribution, accuracy and rank assumptions were removed.

| Initial proposal | Refined result |
|---|---|
| Normalization makes activations Gaussian | Mean/variance constraints do not establish a distribution; a two-point witness refutes the implication |
| Unscaled probabilists' Hermite basis is orthonormal | Squared norm is k!; coefficients require division by k!, or basis normalization |
| Degree 3–4 gives slope error below 1e-4 | Recorded degree-4 Gaussian weighted L2 slope errors were about 0.07796 GELU and 0.02594 SiLU |
| GELU/SiLU preserve coordinate reflections | These activations are not odd; shared coordinate-wise maps have permutation equivariance |
| The largest homogeneous degree bounds the entire feature space | All retained degrees must be accounted for |
| A low-rank adapter compresses every original dense gradient | A dense base remains; changing trainable parameterization changes the optimizer problem |

The implementation expands a selected polynomial p(XW+b) into input monomials Ψ and coordinate-specific translated coefficients Θ:

$$
p(XW+b)=\Psi\Theta^T,\qquad
H_p=\Psi(\Theta^TQ).
$$

Bias translation is shared across monomials of the same degree. Zero-safe derivatives recover every original X/W/b/Q gradient. No radial replacement is needed, but the selected polynomial changes the native activation.

A degree-12 Taylor derivative was converted to Chebyshev form, truncated through degree seven, then integrated into a matched degree-8 primal. Kernel-checked omitted-tail bounds versus the polynomial reference are about 3.40e-6 for GELU and 5.81e-7 for SiLU on |z| ≤ 1. The true-activation analytic remainder and IEEE/GPU arithmetic are separate unresolved formal obligations.

The 32,769-point MPS grid on [-1,1] measured:

| Activation | Maximum sampled value error | Maximum sampled slope error |
|---|---:|---:|
| GELU | 1.0133e-6 | 1.2040e-5 |
| SiLU | 3.5763e-7 | 3.3379e-6 |

Final repeated full-step results include original gradients, MSE loss, SGD and guards, against eager native GELU/SiLU:

| Shape [batch,input,hidden,output] | Activation | Three-seed speed range | Fresh saved storage reduction | Speed admission |
|---|---|---:|---:|---|
| [8192,2,4096,2] | GELU | 1.19–1.50× | 98.52% | Passed |
| [8192,2,4096,2] | SiLU | 0.92–1.34× | 98.52% | Failed |
| [8192,1,8192,2] | GELU | 2.81–3.43× | 99.67% | Passed |
| [8192,1,8192,2] | SiLU | 1.69–3.54× | 99.67% | Passed |

Four narrow composed classification models passed output/all-gradient comparison and four-step SGD/AdamW checks, including one with input LayerNorm. There is no recorded complete-model speed result or long-term quality/convergence claim for these models.

The latest suite has two scalar grids, twelve polynomial cases, twelve repeated large cases, eleven selected edges, two polynomial higher-derivative comparisons and four composed models. Four degree-4 raw approximations failed and remain rejected. Final polynomial implementation and admission checks pass. Speed admission requires three distinct seeds each above 1.05×; correctness and memory admission are separate.

The full feature count is S = binomial(k+D,D). Degree eight has 9 features for k=1, 45 for k=2 and 495 for k=4. For a conventional large dense input width it is prohibitive. The runtime grammar allows k=1–4 and at most 512 features; actual qualification is narrower.

Sources: [spectral refinement](SPECTRAL_COORDINATE_FOUNDATION.md), [spectral runtime](ojasx2/spectral_compiler.py), [SpectralFoundation](lean/OjasXMath/SpectralFoundation.lean), [SpectralChebyshev](lean/OjasXMath/SpectralChebyshev.lean), [current qualification](reports/spectral-qualification.json).

## 13. Numerical reliability repair, version 1.9

A stable scale–LayerNorm pullback now uses epsilon/D times inverse sqrt(D), where D=a² variance+epsilon, instead of subtracting nearly equal contributions or first forming an inverse cube. Lean proves the actual scalar derivative and equivalent evaluation form. Rocq checks the cotangent identities; three nonvacuous Z3 checks cover cancellation, ratio evaluation and zero scale.

The new [GPU-only implementation](ojasx2/numerical_repairs.py) retains original operand dependencies for higher derivatives. A joint radial–LayerNorm prototype preserves original X/W/b/Q/gate parameters. Conservative projection routing upgrades eligible explicit degree-4 approximation requests to degree eight, avoids inadmissible wide feature construction, retains native two-input SiLU, and uses native execution for requested higher-order original-activation semantics.

A checked scalar-loss/all-gradient interface evaluates both pure deterministic programs and selects the native results on mismatched gradients, nonfinite candidates or candidate errors. This adds verification work and is not an acceleration.

The [new GPU report](reports/numerical-repairs-gpu.json) records seven normalization/overflow cases, three composed models with three native-compared SGD steps, five policy cases, three invalid-epsilon cases and three rejection cases. All final reliability checks pass. Two raw relative normalization comparisons and the three sensitive algebraic model candidates still fail the original criterion; native routing resolves their execution rather than qualifying those optimizations. No tolerance was relaxed.

The [repair verifier](tools/verify_numerical_repairs.py) rebuilds the complete proof audit, checks the new SMT identities and validates current GPU source hashes. This phase adds two Lean and two Rocq statements. Its [verification report](reports/numerical-repairs-verification.json) records the limits explicitly.

## What is new within this project

The verified algebra mostly uses existing mathematics. This record does not assert historical novelty of adjoints, Gram methods, reversible coupling, Strassen, polynomial features, Koopman lifts or Chebyshev approximation.

The project contributions are the specific checked joint schedules; recovery of every original parameter gradient; bias-folded coordinate feature implementation; counterexample-driven restrictions; matched forward/backward approximation; concrete GPU guards and rejection ledgers; and the combined proof/discovery/execution workflow. Establishing publishable mathematical novelty would require a separate literature comparison.

## Why the broad objective is still difficult

In an ordinary dense coordinate activation, every element of the slope matrix can depend on its own forward value. That destroys many small representations that were valid before the activation. Factoring the incoming gradient alone does not guarantee that the product with those slopes remains small.

Three costs must be controlled together: the representation width, every original parameter-gradient recovery, and the information retained or recomputed for backward. Nonlinear polynomial features can grow combinatorially; reversible reconstruction adds computation; lossy sketches alter gradients. Floating cancellation and sensitive optimizer updates impose additional constraints.

These are identified obstacles and rejected candidates, not a proof that all future improvements are impossible. The current broad framework can express and certify new rules. It has not yet found a cheap rule covering most ordinary dense nonlinear training.

## Complete guide and source map

The two consolidated files are the primary reading path. Earlier guides below preserve phase details, equations, observations and limitations.

| Existing guide | Material preserved |
|---|---|
| [MATHEMATICAL_SPEC](MATHEMATICAL_SPEC.md) | Original categorical and matrix core |
| [STREAMED_DAGGER_THEOREM](STREAMED_DAGGER_THEOREM.md) | Joint bilinear circuit, streaming and original proof chain |
| [BACKPROP_MATH_STRUCTURE](BACKPROP_MATH_STRUCTURE.md) | Certified differentiation, multilinear and geometry extension |
| [CONTRACTION_OPTIMIZATION](CONTRACTION_OPTIMIZATION.md) | Linear reassociation, all-gradient schedules and CPU observations |
| [NONLINEAR_AND_CONTRACTION_RESEARCH](NONLINEAR_AND_CONTRACTION_RESEARCH.md) | Longer chains, mask and normalization experiments |
| [WORKLOAD_SCALE_MATH_TOOLS](WORKLOAD_SCALE_MATH_TOOLS.md) | Representative workload compatibility and optimizer negative result |
| [REVERSIBLE_OPTIC_THEOREMS](REVERSIBLE_OPTIC_THEOREMS.md) | Reversible architecture and replay tradeoff |
| [CACHED_REVERSE_THEOREM](CACHED_REVERSE_THEOREM.md) | Cached coupling theorem and native Metal observation |
| [DAGGER_RADIAL_CORE](DAGGER_RADIAL_CORE.md) | Earlier RMSNorm Gram/projection specialization |
| [ROCQ_MATH_FOUNDATION](ROCQ_MATH_FOUNDATION.md) | Toolchain, rational extraction and exact contracts |
| [BACKEND_MATH_FOUNDATION_V1](BACKEND_MATH_FOUNDATION_V1.md) | Broad families, modeled cost and known rank-seven algorithms |
| [FOUNDATION_REFINEMENT](FOUNDATION_REFINEMENT.md) | Normalization residuals, epsilon and indexing edges |
| [ABSTRACT_BACKPROP_ENGINE](ABSTRACT_BACKPROP_ENGINE.md) | Mathlib engine, primitive obligations and polynomial discovery |
| [GPU_NONLINEAR_DISCOVERY](GPU_NONLINEAR_DISCOVERY.md) | Small-domain nonlinear approximation and rejected speed candidates |
| [MATHEMATICAL_COMPILER](MATHEMATICAL_COMPILER.md) | Coordinate changes, finite closure and Koopman GPU results |
| [ACTIVATION_BACKPROP_REFINEMENT](ACTIVATION_BACKPROP_REFINEMENT.md) | Joint factor rules, size scaling and all-original-gradient tests |
| [INVARIANT_FOUNDATION_REFINEMENT](INVARIANT_FOUNDATION_REFINEMENT.md) | Radial rules, counterexamples and rejected normalized compositions |
| [SPECTRAL_COORDINATE_FOUNDATION](SPECTRAL_COORDINATE_FOUNDATION.md) | Latest coordinate-sensitive approximation, proofs and GPU ledger |
| [GPU_QUALIFICATION](GPU_QUALIFICATION.md) | Earlier actual MPS qualification and hardware restrictions |
| [HARDWARE_IMPLEMENTATION_GUIDE](HARDWARE_IMPLEMENTATION_GUIDE.md) | Developer interface, schedule and lowering requirements |
| [COMPILER_PLAN](COMPILER_PLAN.md) | Planned semantic IR and target lowering |
| [OJASX_SINGLE_GPU_PLAN](OJASX_SINGLE_GPU_PLAN.md) | Capacity/throughput objective and whole-model qualification |
| [WORK_PLAN](WORK_PLAN.md) | Initial task plan |
| [Verification overview](reports/VERIFICATION.md) | Historical combined checks and measurement accounting |

All 25 current Lean modules are in [lean/OjasXMath](lean/OjasXMath); the complete audit imports them through [OjasXMath.lean](lean/OjasXMath.lean). Rocq modules are enumerated by [_RocqProject](rocq/_RocqProject), with assumptions printed by [Audit.v](rocq/Audit.v). Reference engines are in [ojasx2](ojasx2), search/verification/qualification tools in [tools](tools), raw evidence in [reports](reports), and older spectral prototypes in [experiments](experiments).

The supplied notes remain original proposals, not proof certificates: [architecture](</Users/mohana/.codex/attachments/2d98fb04-5c83-4d32-946e-8c9d4da5d99b/Pasted text.txt>), [initial monograph](</Users/mohana/.codex/attachments/5756547f-8f02-413c-acc6-cf836cfb1384/Pasted text.txt>), [directional-arrow proposal](</Users/mohana/.codex/attachments/2fb26243-01a6-4cb3-9dc9-f8e9ff3c7815/Pasted text.txt>), [reversible proposal](</Users/mohana/.codex/attachments/820f08cd-44bc-4cdb-9117-cfa70a1c1b25/Pasted text.txt>), [Rocq workflow](</Users/mohana/.codex/attachments/95a9b6db-d558-4c4e-a305-c8a383dc1e3a/Pasted text.txt>), [alternative algebra](</Users/mohana/.codex/attachments/62d05f6e-f977-4d79-8054-3a8ae234a189/Pasted text.txt>), and [Hermite/Veronese proposal](</Users/mohana/.codex/attachments/050d1579-2cef-4630-ac14-4ac395260822/Pasted text.txt>).

## Rechecking this record

Use the formal-only current runner and existing GPU evidence:

~~~sh
cd /Users/mohana/Documents/Codex/2026-10-06/v-2/outputs/OjasX_rebuilt
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/refine_spectral_foundation.py --qualify-existing-gpu
~~~

It performs bounded symbolic discovery, rebuilds/audits the full formal foundation, independently rechecks Rocq, refreshes source hashes and rejects stale spectral GPU evidence. It runs no numerical CPU ML. A GPU rerun is a separate opt-in command described in the companion foundation.

The current evidence supports conditional algorithms worth building upon. The next mathematical priority is a cheaper representation for wide coordinate nonlinearities with matched primal/derivative, controlled feature growth, all original gradients and whole-model error. Total-memory planning and portable production lowering must then preserve those verified contracts.


## 9 October 2026: objective-aware foundation version 2.0

This refinement broadened the joint mathematical unit from selected projection blocks to complete scalar training objectives. It added a squared-error Gram schedule and an explicitly separable streaming program, preserving ordinary nonlinear activation functions and original parameter objects. The algebra uses established finite-sum differentiation and Gram/trace identities; the contribution here is their executable joint schedule, contracts, original-gradient recovery, proof integration and measured rejection policy.

The new sources are [objective_compiler.py](ojasx2/objective_compiler.py), [ObjectiveCompiler.lean](lean/OjasXMath/ObjectiveCompiler.lean), [ObjectiveCompiler.v](rocq/ObjectiveCompiler.v), [bounded discovery](tools/discover_objective_rules.py), [GPU qualification](tools/test_objective_gpu.py) and [combined runner](tools/refine_objective_foundation.py). The current [foundation specification](OJASX_MATHEMATICAL_FOUNDATION.md) gives the equations and API.

Lean adds 23 theorems, including actual derivatives rather than adjoint algebra alone. Rocq adds 16 rational theorem/lemma declarations. Z3 checks all entries of a finite quadratic example against SymPy differentiation and falsifies plausible omissions; the formal tensor identities are dimension-general. The final search includes 24 checks/counterexamples and 27 shape decisions. No new mathematical axiom asserts cheapness or numerical correctness.

The initial GPU phase passed full loss/gradient and short SGD/AdamW comparisons for six synthetic objective families. It showed substantial retained-activation savings but rejected speed admission for both the Gram and sparse-weight reference schedules. Larger layers were measured rather than assumed faster. The runtime then gained a distinct gradient-cancellation guard: a loss-energy check alone admitted a stationary-weight example whose raw rearranged gradient differed from a native zero gradient. That raw failure is retained beside the repaired native backward result. Bias and requested-gradient costs use the original dense baseline, and the encoded extra feature is charged only to the Gram candidate. The fallback preserves the original bias placement and residual scaling order.

The source-matched final results are [objective-gpu.json](reports/objective-gpu.json) and [objective-qualification.json](reports/objective-qualification.json). [objective-gpu-pre-gradient-guard.json](reports/objective-gpu-pre-gradient-guard.json) preserves an earlier run, with its old source hashes and narrower guard scope. It is historical evidence only. No accuracy tolerance was relaxed to admit the cancellation example.

Separable streaming reduces peak retained chunk tape as batch grows, with no forward replay. It does not reduce all arithmetic for unmasked examples. Removing fixed zero-weight independent examples can reduce model arithmetic, but requires uniform parameter usage and finite per-example loss/gradient obligations; finite operands alone cannot rule out overflow. Batch-coupled objectives, training BatchNorm, hidden mutation, fresh randomness, and arbitrary higher-order differentiation are outside this first-order contract. Callback declarations are not automatically kernel-checked.

All new numerical work ran sequentially on MPS with CPU fallback disabled. Proof checking, SymPy, Z3 and process orchestration ran on the host. The complete formal audit and previous source-matched qualifications are preserved by the combined runner. The current toolchain is a broader reference mathematical compiler, with explicit unproved Python/IEEE/backend boundaries, not a completed universal optimizer.

## 9 October 2026: loss-directed foundation version 2.1

The work added finite weighted log-partition derivatives, smoothed cotangents, conditional numerical error transport, selected loss-row elimination and bounded head replay. Lean adds 17 checked statements; Rocq adds nine. The total audit now covers 492 Lean and 247 Rocq statements, without new axioms. Counters count formal declarations, not independent mathematical discoveries.

The exact-real rules use existing mathematics. `tools/discover_loss_rules.py` differentiates three components with SymPy, performs 20 nonvacuous Z3 checks/counterexamples and explores 108 shape schedules. Counterexamples require the weighted probability mass, one global chunk denominator, retained ignored-prefix attention tokens and an overflow guard. Additional GPU checks required native handling for autocast and higher derivatives. The cost planner was corrected to charge projection replay during bias-only training; its demand-aware counts are rechecked in the GPU cases.

**Version 2.1 measured outcomes (latest source-matched run).** All 48 loss/model/edge/timing cases, seven FX graph cases, 36 TorchCTS OpInfo cases and 16 BackendBench profiles pass their recorded comparisons. Two tied-weight decoder architectures preserve every original parameter gradient through three SGD and three AdamW updates; the largest relative L2 discrepancy in those update/gradient/state checks is 5.21e-07.

| Experiment | Latest result | Scope |
|---|---:|---|
| Selected output head, N=2048, K=256, V=8192, 1/16 active labels | 2.28–3.43× native/candidate throughput ratio over three seeds | Head forward, loss, both gradients, bias and zero-lr SGD; not complete-model speed |
| Leading dense matrix arithmetic in that head | 93.75% fewer FLOPs | Real-arithmetic count; guards and pointwise/indexing costs excluded |
| Already preselected PyTorch/candidate speed ratio | 0.34–0.42× | Already optimized PyTorch is faster; no superior GEMM claim |
| All-active replay, N=1024, K=32, V=1024, chunk=32 | 4,194,308 → 8,196 fresh saved bytes (99.80% fewer) | Autograd-retained loss storage; inputs, parameters and optimizer excluded |
| Sampled MPS peak increment, N=512, K=128, V=4096, chunk=32 | 25,166,592 → 8,668,928 bytes (65.6% lower) | Operation-boundary allocator samples, not a certified total peak |
| Complete small tied SwiGLU decoder, 4×128 tokens, width64, vocabulary4096, 1/16 active labels | 0.54–0.88× | Correctness passes; speed rejected; same forward/all-gradients/zero-lr SGD accounting |

Timings use two warmups and five shuffled, synchronized rounds per seed, with pauses. The head baseline is direct native PyTorch, not a guarded wrapper. The stronger preselected comparator receives precomputed active indices. A previous smaller-head run failed consistent speed admission and remains in `reports/loss-gpu-before-guard-refinement.json` as historical evidence, with its old hashes. Consolidating validation reduced repeated scans/synchronizations; warmups and removing diagnostic Python dispatch wrappers from timing avoid conflating instrumentation with execution. Neither that refinement nor a fast head converts the slower decoder into a qualified speed win.

The graph integration (`ojasx2/loss_fx.py`) recognizes a private Linear→cross_entropy pattern while retaining shared logits and hooks. Seven GPU graph tests include module/functional forms, observed logits, soft targets, unreduced loss, higher derivatives and hooks. This is a conservative FX prototype; an unrestricted dynamic PyTorch compiler is not implemented.

TorchCTS 0.4.1 and BackendBench 0.1.0 were installed in a new Python 3.11 / Torch 2.8 environment. Official upstream sources are pinned in the external report. The stock CTS CPU-reference and BackendBench CUDA/CPU-fallback harnesses were replaced at the integration boundary by explicit GPU-only adapters. The BackendBench profile parser uses a syntax allowlist; malicious calls and negative strides are rejected. It preserves actual shapes/strides and records why every other profile is filtered. The GPU suite corrected an initial arbitrary-axis softmax adapter signature error; final OpInfo cases have no filters or mismatches. No stock-suite certification is claimed.

`torch.profiler(profile_memory=True)` exports two traces but provides host dispatch activity on this MPS build. The report separates those events from synchronized timings, autograd saved storage and sampled MPS allocator increments. No CPU numerical ML was used, and no massive model allocation or persistent automation was attempted.

The 200B/32GB target now has an explicit resource ledger: 400 GB decimal of independent 16-bit parameter data, and 3.2 TB for the specified dense mixed-precision Adam state layout, before activations. These head-level optimizations do not change those numbers. Exact external-state storage and traffic, or a justified alternative representation/state layout, remain separate research and system obligations.

Current files:

- `ojasx2/loss_compiler.py`: loss schedules, shape/work plan and resident-state ledger.
- `ojasx2/loss_fx.py`: conservative ordinary-model graph rewrite.
- `lean/OjasXMath/LossCompiler.lean` and `rocq/LossCompiler.v`: real calculus and rational algebra.
- `tools/discover_loss_rules.py`: bounded symbolic discovery and counterexample refinement.
- `tools/test_loss_gpu.py`, `tools/test_loss_fx_gpu.py`, `tools/test_external_conformance_gpu.py`: numerical/model/provenance tests.
- `tools/refine_loss_foundation.py`: repeatable sequential audit and admission.
- `reports/loss-qualification.json`, `reports/loss-gpu.json`, `reports/loss-fx-gpu.json`, `reports/external-conformance-gpu.json`: current evidence.
- `reports/traces/loss-strict.json` and `reports/traces/loss-algebraic.json`: exported host traces.

The complete OjasX foundation remains a framework for guarded, conditional optimizations with native coverage paths. Version 2.1 establishes additional useful loss arithmetic and storage results. It does not establish 100× most-model training, all future edge cases, or 200B training entirely within 32GB.

## Observable cotangents and exact-support nonlinear defects, version 2.2

The research target changed from compressing the entire hidden gradient to preserving all observations used by original input, weight and bias gradients. An exact symbolic nullspace calculation found 18 invisible degrees of freedom in a 48-entry example. The dimension-general Lean construction D*=PD+DQ−PDQ preserves both matrix contractions; including the constant column preserves bias. Its factored form uses two inner spaces. Forming the factors can still cost dense work. No low-rank slope assumption is needed for representation, but cheap construction remains conditional.

The new runtime retains the native nonlinear forward and shares affine gradient moments, with separate value and slope defects. It accepts sparse entry or exceptional-column support only after inspecting both certificates. Interval bounds avoid full elementwise slope discovery for eligible piecewise-affine columns. Smooth native saturation is treated as bounded floating-point evidence, not an exact real identity. The solver refuted ignoring nonzero defects, checking only primal values, preserving only one gradient, and moving an invisible component through a coordinate-dependent derivative.

A separate streamed MLP/SwiGLU rule saves only original operands and replays row chunks. This covers centered dense nonlinear activations for saved-memory reduction, while increasing recomputation. Small attention/residual/normalization examples check all original gradients and three SGD/AdamW updates. No full large-model speedup is inferred.

The first 60-case GPU sweep passed correctness but every timed new candidate ran slower. Its raw report remains in `reports/cotangent-quotient-gpu-initial.json`, with the matching prototype in `experiments/cotangent_quotient_initial.py`. Guard consolidation and interval certification reduced overhead. A larger narrow-output ReLU shape was then added as a separate speed candidate; narrow output is a structural restriction, not a representative claim for all Transformer MLPs.

The expanded 80-case sweep exposed hardtanh boundary mismatches at −1 and +1: on this installed MPS backend, native backward returned slope one at the endpoints, while the initial interval rule assumed zero. Input and bias gradients failed with relative error 1. The raw failure remains in `reports/cotangent-quotient-gpu-boundary-failure.json` and the source in `experiments/cotangent_quotient_boundary_failure.py`. Strict exterior/interior guards repaired the rule, with native slope inspection for endpoints. Lean interval hypotheses were strengthened accordingly. This demonstrates why an exact-real piecewise convention cannot be silently equated to a backend's selected backward.

The added proof modules contain 23 Lean and seven Rocq declarations. Lean checks observation equivalence, kernel and contraction identities, actual derivative subtraction, finite-support elimination, cost conditions and interval conventions. Rocq checks rational joint-gradient and streamed-sum contracts. These are scoped mathematical statements, not 30 new independent inventions. The bounded discovery records 12 checks/counterexamples, four scalar gradient identities and 144 shape/support plans.

Current sources: `ojasx2/cotangent_quotient.py`, `lean/OjasXMath/CotangentQuotient.lean`, `rocq/CotangentQuotient.v`, `tools/discover_cotangent_quotient.py`, `tools/test_cotangent_quotient_gpu.py`, and `tools/refine_cotangent_foundation.py`. Current outcomes are in `reports/cotangent-qualification.json`; the two canonical documents and machine-readable manifest retain all earlier scopes.

The final 80-case rerun passed all comparisons, with maximum recorded relative L2 error 1.42e-6. Eligible small GELU/SiLU blocks retained 4,112 rather than 262,144 fresh bytes (98.43% less). The corresponding joint arithmetic model, including its scalar certificate estimate, fell by 52.04%; hardware overhead is separate. Dense centered SwiGLU streaming removed 16,777,216 fresh saved activation bytes in the largest row-scaling case, retaining original operands and paying backward recomputation. ReLU at the larger narrow-output timing shape ranged 0.995–1.394× over three seeds; this failed the repeated >1.05× threshold. Smaller defect and streamed candidates were slower. Thus no new broad speed claim is admitted.

## Two-sided demand algebra, version 2.3

After the request to develop from existing mathematics without further literature searching, a new OjasX rule was derived from exact cotangent support. It moves beyond requiring near-affine activation slopes: arbitrary native nonlinear derivatives are evaluated only where the actual incoming gradient demands them. Full original forward and every original parameter gradient are retained. Support is discovered after branch accumulation and cannot be inferred across attention from the final label mask.

The scalar/Jacobian pairing gives row and output-column restriction. A further refinement avoids recomputing the final output projection during backward; its input pullback and weight gradient are formed directly. Full leading cost includes hidden projection replay and support scans. The first GPU run passed 70 correctness/model/edge/timing cases but repeated speed measurements were inconsistent. It remains in reports/demand-gpu-before-output-refinement.json. A subsequent refinement removed redundant zero initialization of parameter gradients that are directly computed.

The proof extension contains 12 Lean and four Rocq declarations. Bounded symbolic checks reject tolerance pruning, premature branch deletion, invalid upstream sparsity propagation, higher-derivative erasure and uncharged replay. This is an extension using known sparsity and derivative principles, not a claim of historical mathematical novelty. The three papers' algorithmic cost and memory principles informed the approach; none of their main theorems is being asserted as a general ML optimization.

Current sources: ojasx2/demand_algebra.py, lean/OjasXMath/DemandAlgebra.lean, rocq/DemandAlgebra.v, tools/discover_demand_algebra.py, tools/test_demand_gpu.py and tools/refine_demand_foundation.py. The authoritative outcome is reports/demand-qualification.json.

The final 70-case rerun passed with maximum relative L2 discrepancy 5.00e-7. Small native GELU/SwiGLU blocks eliminated respectively 50,440/100,880 fresh saved activation bytes while retaining their original operands. For the synthetic sparse-demand timing shape, the full leading arithmetic model fell by 63.01%/62.15%. Timings remained inconsistent (GELU 0.51–1.19×, SwiGLU 0.76–1.40×), so neither is speed-admitted. The new foundation therefore contains a broader exact activation-compatible rule, but does not establish dense-all-active arithmetic reduction or full-model acceleration.

## Dense nonlinear residuals and batched matrix candidates, version 2.4

The previous exact-demand rule gave no arithmetic reduction for dense G. This round tested a complementary sufficient-residual construction and the existing rank-seven circuit without requiring sparse slopes, cotangents or saturation. No additional literature was searched.

The new first-order schedule retains Z for ordinary GELU/SiLU and both preactivations for SwiGLU. It reconstructs native pointwise values to obtain every original gradient without replaying the expensive hidden matrix projections. In all-gradient M33/K12/N65/R16 cases, GELU/SiLU tape changes 17,160→8,580 fresh bytes; SwiGLU 34,320→17,160. ReLU/tanh/sigmoid/identity already have a one-intermediate native tape and select native. Frozen/projection-only paths conservatively retain native behavior. This is broader dense-gradient memory coverage, not a new historical mathematical identity or 50% lower total model RAM.

The existing Winograd identity is batched as seven encoded products and differentiated through the transposed shared circuit. Cache and replay are charged separately; the decoder's explicit unary negation is added to the historical cost model. At 1024³, the cached/replayed all-gradient counts are 12.40%/12.37% below conventional scalar counts. Encoded cache storage and packing remain costs. The reference GPU schedule is slower and is not speed-admitted.

Six cancellation examples refute raw relative-accuracy admission: three forward and three gradient cases. A checked path compares against charged native products and falls back on disagreement; all six repaired cases pass. The counterexamples are retained in the raw report. Strict mode remains native. Numerical checking is not a proof of IEEE behavior or a speed reduction.

New source: ojasx2/dense_joint.py, Lean DenseJoint.lean, Rocq DenseJoint.v, discover_dense_joint.py, test_dense_joint_gpu.py and refine_dense_foundation.py. Twenty new Lean declarations and eight Rocq statements include actual affine-activation calculus, arbitrary finite dense adjoints and scalar cost/storage contracts. The finite discovery checks the existing polynomial coefficients, synthesizes transposed encoders, uses eight Z3 checks/counterexamples and enumerates 576 demand-aware plans.

All 80 final GPU cases pass, including SGD/AdamW updates in six small attention/norm/residual runs with losses over every output coordinate. The canonical foundation records final three-seed timing ratios, source hashes, exclusions and backend use. Timings fail the repeated >1.05× speed threshold. The host-dispatch profiler confirms six GEMMs in both GELU paths, with one additional activation evaluation in the residual path. It does not expose MPS peak memory.

This round produces a useful broader memory foundation and stronger numerical/cost guards. Broad complete-model speedups and substantially cheaper ordinary dense nonlinear arithmetic remain research targets; they are not inferred from the memory saving.
