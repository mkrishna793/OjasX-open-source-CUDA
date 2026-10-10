# OjasX invariant-based activation and backpropagation foundation

This refinement turns the useful part of the supplied geometric proposal into an executable mathematical compiler rule: an existing radial activation followed by a projection can be evaluated and differentiated through small Gram statistics. It preserves the original input, dense weights, bias, projection weights and trainable scalar gate parameters. It does not require training a low-rank replacement for the original weights.

The largest isolated blocks measured 4.62–6.09× faster complete training steps and about 96% less fresh saved activation storage. A complete two-block radial classification model measured 5.87–6.49× faster and 98.52% less saved activation storage across three seeds. These are MPS FP32 measurements against eager PyTorch execution of the **same radial architecture**. They do not establish corresponding gains for existing GELU/SiLU Transformers or most ML models.

## 1. What the tools actually discovered and checked

The supplied Rocq workflow is useful for organizing definitions and proofs. Its example `identity_proof` simply restates the declared `add_zero_law`. Declaring an optimization as an axiom would similarly assume the result we need to establish. Rocq explicitly treats such declarations as assumptions; see its [official assumptions documentation](https://rocq-prover.org/doc/V9.2.0/refman/language/core/assumptions.html).

Instead, `rocq/InvariantRadial.v` introduces an `InvariantFoundation` module and an `ExactRewrite` record. A rewrite must contain an implementation, a reverse implementation, proofs against the original primal and supplied original pullback, and a modeled-work domination proof. The original pullback still needs its primitive derivative contract. No optimization axioms are introduced. This is a foundation contract, not proof that every Python implementation satisfies the contract.

`tools/discover_invariant_alternatives.py` performs bounded counterexample-guided synthesis. For the energy `(x+y)^2`, it first tries the basis `[x^2,y^2]`. Z3 supplies counterexamples until that basis becomes inconsistent. The tool refines the basis with `xy`, synthesizes coefficients `[1,1,2]`, and universally checks the polynomial identity. These cross terms are precisely why a full small Gram matrix is needed; diagonal statistics alone lose information.

The tool also derives the radial Jacobian and gate derivatives with SymPy, checks 21 SMT claims/counterexamples, evaluates 144 size decisions, and generates proof source. The generated cubic example includes a bias, two input coordinates, two hidden coordinates and two projection coordinates: the primal equality, all twelve original-variable gradient equalities, and twelve genuine Lean `HasDerivAt` derivative proofs. This bounded search is not an unrestricted theorem inventor.

The new files add 47 Lean statements and 30 Rocq statements. The final all-project counts and trust audit are recorded in [the formal verification report](reports/invariant-formal-verification.json). Lean checks actual radial directional calculus over arbitrary finite feature sets, including zero, reciprocal gates on nonnegative energy, sigmoid energy derivatives and both trainable sigmoid gate derivatives. Rocq checks arbitrary finite matrix contractions over canonical rationals, including all input, weight and projection gradients. Its scalar coefficients are supplied; Rocq does not independently prove exponential calculus here.

## 2. The repaired mathematical construction

Let `m` be the batch/token count, `k` the input width, `N` the hidden width, and `r` the output projection width. The original block is

\[
Z=XW+b,\quad s_i=\|Z_i\|^2/N,\quad Y_i=f(s_i)Z_i,\quad H=YQ.
\]

This is a nonlinear **vector** activation. Each vector is scaled by a function of its energy. Current executable gates are cubic `f(s)=s`, reciprocal `f(s)=1/(1+s)`, and a trainable sigmoid `f(s)=sigmoid(a s+b_g)`. The abstract proof applies to any gate whose actual derivative is supplied and proved. Adding an executable gate requires its numerical and derivative contracts.

Introduce the augmented input and weights:

\[
A=[X,1],\quad V=\begin{bmatrix}W\\b\end{bmatrix},\quad d=k+1,
\quad C=VV^T,\quad T=VQ.
\]

Then compute

\[
s_i=(AC)_i\cdot A_i/N,\quad u=AT,\quad H_i=f(s_i)u_i.
\]

The large `m × N` activation is replaced by the `d × d` Gram matrix `C`, the `d × r` projection `T`, and batch values of width `d` or `r`. Orthogonal changes of the hidden feature coordinates preserve these statistics when the projection is transformed consistently. This connects symmetry, sufficient statistics, matrix contraction and adjoint calculus. It does not depend on a false batch-rank bound or on random directions being orthogonal.

For upstream gradient `P`, define

\[
R_i=f(s_i)P_i,\quad e_i=f'(s_i)(P_i\cdot u_i)/N,
\quad B=A^TR,\quad D=A^T\operatorname{diag}(e)A.
\]

All original gradients are recovered by

\[
\nabla_A H=RT^T+2\operatorname{diag}(e)AC,
\quad \nabla_V H=BQ^T+2DV,
\quad \nabla_Q H=V^TB.
\]

Discard the appended constant coordinate from `∇A`; split `∇V` into the original `∇W` and `∇b`. For the sigmoid gate, with `t_i=f(s_i)(1-f(s_i))(P_i·u_i)`, its original gradients are

\[
\nabla_a H=\sum_i t_i s_i,\qquad \nabla_{b_g}H=\sum_i t_i.
\]

**Do not train C and T as independent parameters.** They are derived statistics. Updating the original W, b and Q using the recovered gradients preserves their optimization semantics over exact reals. Independently updating C/T or low-rank U/V generally changes the optimizer trajectory.

Squared energy avoids dividing by the radius at zero. For a differentiable gate, the activation derivative is

\[
D\phi_z[dz]=f(s)dz+\frac{2}{N}f'(s)z\langle z,dz\rangle.
\]

Its adjoint has the same scalar-plus-rank-one structure. That describes a **single vector's Jacobian**, which is generally full rank. Different samples may have different Jacobians and their batch gradient can still have full rank. The Gram rewrite remains valid despite that rank growth.

## 3. Corrections to the supplied alternative formulations

| Proposal | Checked conclusion |
|---|---|
| Radial Jacobians make batch rank at most r+1 | False for different sample states. Cubic activation sends rank-one incoming rows `(1,0,0)` at states `(1,0,0),(1,1,0),(1,0,1)` to `[[3,0,0],[4,2,0],[4,0,2]]`, determinant 12 and rank three. Shared right multiplication preserves rank ≤r, but samples generally do not share that matrix. |
| Apply ReLU to the radius to obtain cheap ordinary ReLU | For positive radius, radius-ReLU gives the identity vector map. Coordinate ReLU changes negative coordinates. These are different models. |
| Fixed commutator `[K,Z]` is a nonlinear rank-preserving activation | It is linear in Z. Its derivation satisfies Leibniz but can double rank: the safe elementary bound is 2r, not r. If K is trainable its gradient must also be retained. The discovery source and GPU tests include these checks. |
| Chamber grouping avoids coordinate ReLU rank growth | Exact masks with C distinct chambers admit factors of width rC. If every sample has a different mask, C can grow with m. The prototype requires supplied labels and masks, validates them on GPU, and does not implement free automatic clustering. |
| Trace duality makes a thin random sketch exact | Trace duality is an identity; a noninjective sketch loses directions. A two-coordinate counterexample has exact contraction 1 and sketched contraction 0. An exact subspace certificate or a quantified approximation bound is required. |
| High dimension makes all radial interactions negligible | Independent isotropic directions have a useful statistical moment. Gradients and features need not be independent. An aligned cubic example has relative error 2/3 if the correction is dropped, even at dimension 1024. |
| Schur's lemma reduces all dense weights to scalars | Requires an equivariant model and the appropriate representation assumptions. Multiplicity spaces allow nontrivial mixing between repeated irreducible copies. It does not constrain arbitrary PyTorch weights. |
| Factorized U/V training preserves dense-W SGD | Generally false, including zero-factor cases where factor gradients vanish but the dense-W gradient does not. The implemented Gram rule keeps the original dense W. |

These corrections preserve the useful symmetry idea. Radial architectures and equivariant vector networks already exist; see [Vector Neurons](https://arxiv.org/abs/2104.12229) and [radial neural networks](https://arxiv.org/abs/2107.02550). Representation assumptions and Schur's lemma are treated in the [MIT representation theory text](https://math.mit.edu/~etingof/representationtheorybook.pdf). This work is a verified optimization of that structural family, not a claim to have invented those mathematics.

## 4. Arithmetic, scaling and memory

The conservative leading arithmetic comparison for forward plus all requested gradients is

\[
\text{native}\approx6mN(d+r),\qquad
\text{core}\approx6d(d+r)(m+N)+12md+12mr.
\]

Ignoring the explicit smaller terms, the condition is `d(m+N) < mN`. The runtime evaluates the full modeled condition, rejects nonpositive shapes, and does not treat this as a latency guarantee. Shared optimizer work can be added to both sides without invalidating domination; the formal common-cost lemma states that elementary fact. GPU timings include plain SGD, not AdamW state or model convergence.

For narrow input and wide hidden layers this removes much of the batch-by-hidden work. If input width grows toward hidden width, the Gram construction can become expensive and should be rejected. Dense parameter storage, dense parameter-gradient outputs and their updates still scale with kN; this is not total training cost sublinear in hidden width. A scalar-plus-rank-one Jacobian alone cannot remove the cost of writing every dense gradient.

Fresh saved activation storage excludes original inputs/parameters. It also excludes total allocator peak, transient workspaces, dense output gradients and optimizer state. The percentage reported here cannot be applied to total laptop RAM or used to claim one GPU replaces four GPUs for arbitrary models.

## 5. GPU qualification and the failure that refined admission

All numerical ML executed on the laptop's Apple MPS GPU with `PYTORCH_ENABLE_MPS_FALLBACK=0`. There was no CPU tensor oracle. Host work was limited to symbolic algebra, proof checking, orchestration and report processing.

The main suite has 45 checks: twelve activation/shape cases, nine repeated large cases, fourteen edges, three higher-derivative comparisons and seven alternative-claim checks. All passed. Edges include zero, negative and noncontiguous inputs, nonfinite values, cancellation, near cancellation, frozen parameters, large/nonfinite upstream cotangents, strict policy, empty dimensions and cost rejection. Higher derivatives use a differentiable original-program recomputation. This is sampled qualification, not coverage of every possible FP32 input.

The largest block `[m,k,N,r]=[8192,128,4096,2]` produced:

| Gate | Three independent seed speed ratios | Fresh saved storage, native → core |
|---|---|---|
| Cubic | 4.620×, 5.263×, 5.579× | 268,468,224 → 10,766,348 bytes |
| Reciprocal | 6.086×, 5.430×, 5.052× | 268,500,992 → 10,799,116 bytes |
| Trainable sigmoid | 4.776×, 4.895×, 4.723× | 268,500,992 → 10,799,116 bytes |

Small shapes were substantially slower despite modeled arithmetic eligibility. Repeated speed admission therefore requires at least three distinct seeds, all exceeding 1.05×. Cost eligibility alone does not enable automatic speed dispatch.

The composition suite retains six model candidates, seven chamber cases and two sketch/alignment witnesses. Three complete two-block radial models with cross-entropy, shape `[8192,32,4096,32,8]`, passed all input/original-parameter gradients and four plain-SGD steps. They measured 6.020×, 6.494× and 5.867× faster. Fresh saved block activation storage fell from 538,050,560 to 7,976,616 bytes, or 98.52%. Loss tape is outside that memory-hook window; loss computation is included in timing. No model-quality or long-term training-convergence improvement is established.

Three radial–LayerNorm–radial candidates **failed** the unchanged relative-gradient criterion. The first gate's gradient nearly cancels through LayerNorm: absolute differences were below 1e-8, but relative errors were around 1–7%. This matters for near-zero gradients. The original failed report is preserved in `reports/invariant-composition-gpu-initial.json`, and the final report still records all three raw candidate rejections. Whole-program preflight selects the original calculation for those models. Zero admission failures does **not** mean those optimized candidates passed.

The cancellation is structurally understood. If `r^2(a^2 variance+eps)=1`, then

\[
r-a^2\,variance\,r^3=eps\,r^3.
\]

The new Lean/Rocq identity shows why subtracting two large terms to obtain an epsilon-sized scale sensitivity is numerically fragile. A future joint radial–normalization rule should evaluate this sensitivity stably. Such a rule is not implemented or qualified in this refinement; current policy retains the original calculation.

Exact chamber ReLU cases passed, including an invalid-certificate fallback. Large repeated timings were 0.822×, 0.722× and 0.683×, and saved storage did not improve. This candidate is **not speed-admitted**. The mask certificate is supplied; its validation is timed, automatic discovery is not included.

Numerical comparison requires `torch.allclose` with absolute/relative tolerance 2e-4 **and** relative L2 error ≤2e-4. Tolerances were not loosened to hide the normalized failures. Exact-real proofs do not establish FP32 bitwise equality. The conditioning guard rejects negative/nonfinite or cancellation-dominated Gram energies and never clips a negative estimate to make it pass. It is a heuristic safeguard, not a certified IEEE error bound.

## 6. Using the implementation and lowering it to hardware

`ojasx2/invariant_radial_compiler.py` provides the opt-in API:

```python
from ojasx2.invariant_radial_compiler import invariant_radial

# Original tensors are FP32 on MPS; W, bias and Q remain original parameters.
h = invariant_radial(x, W, bias, Q, "reciprocal", accuracy="algebraic")
# Trainable sigmoid additionally accepts gate=[a,b_g], a two-element parameter.
```

Default `accuracy="strict"` runs the original radial expression. Algebraic mode is a research option. Unsupported dtypes, autocast, poor modeled cost, nonfinite/extreme inputs and poor Gram conditioning retain the original calculation. Higher derivatives and unsafe cotangents recompute the original differentiable program. Whole-model admission still requires preflight; the runtime does not infer sensitive LayerNorm compositions automatically.

A future compiler can recognize an **existing** `linear → norm-dependent vector scale → projection` subgraph and attach a verified rewrite descriptor. Developers keep the original tensors, optimizer and autograd interface. Replacing scalar GELU with radial GELU would instead require an explicit architecture change and accuracy/convergence study. Separate geometric fibers require separate local statistics and proofs; the present prototype handles one radial group per row, not every vector bundle architecture.

Hardware lowering should compute C and T in GPU tiles, reuse the resulting cores across the batch, compute AC/u/energy/scale in tiles, and reduce AᵀR and Aᵀdiag(e)A during reverse execution. Finally it emits the original dense W/b/Q gradients. C/T must be refreshed after original parameter updates. Avoid materializing a full m×m Gram matrix; only its diagonal energies are needed. Tile scratch sizes, cache lifetime, aliasing, launch cost, deterministic reductions and allocator peak need their own backend qualification. The current executable uses PyTorch MPS operations, not a newly implemented native Metal or portable GPU compiler.

## 7. Reproducible discovery, proof and test loop

From this directory, using the installed Python environment with SymPy, Z3 and PyTorch:

```sh
# Symbolic discovery, all Lean/Rocq proofs and assumptions, independent Rocq check:
python -B tools/refine_invariant_foundation.py

# Also execute both numerical suites on MPS; fails if GPU access is unavailable:
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B tools/refine_invariant_foundation.py --gpu

# Validate the current source hashes against existing GPU evidence:
python -B tools/refine_invariant_foundation.py --qualify-existing-gpu
```

The loop regenerates bounded discoveries, compiles proofs, checks assumptions, independently kernel-rechecks Rocq, optionally runs the GPU tests, and writes an admission/rejection ledger. It does not run legacy CPU numerical verification scripts or install a background scheduler. A stale source hash or failed proof/test prevents qualification. Final proof and execution are separate obligations: neither one alone certifies a complete backend.

Authoritative reports: [symbolic search](reports/invariant-discovery.json), [formal audit](reports/invariant-formal-verification.json), [main GPU results](reports/invariant-gpu.json), [composition results and raw rejections](reports/invariant-composition-gpu.json), and [qualification ledger](reports/invariant-qualification.json).

The resulting foundation adds a nonlinear invariant compiler family to the earlier linear, polynomial and coordinate rules. It supplies real scoped improvements and explicit rejection conditions. Universal compression, global minimum memory, automatic coverage of arbitrary future primitives, and dramatic savings across most existing models remain unproved.
