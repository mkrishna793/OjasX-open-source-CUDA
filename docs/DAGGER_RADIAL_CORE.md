# OjasX: combine mathematical compression with dagger-aware execution

This extension implements a smaller computation graph for an existing nonlinear block and connects it to the PyTorch compiler backend. It does not replace backpropagation with local targets or claim a universal reduction for every network. The mathematical ingredients are existing linear algebra, adjoints, quadratic forms and differentiation; the engineering contribution is choosing a cheaper joint forward/gradient graph.

GPU update: [MPS qualification](GPU_QUALIFICATION.md) now includes actual M5 FP32 execution. A wide RMS core measured 1.66× speed and 97.10% less fresh activation tape; the small compiled residual remained slower. Earlier CPU measurements below retain their original scope.

## 1. What your attached idea gets right, and what needs correction

For row-oriented activations, use Z=XW and Y=σ(Z). If the loss sends G=dL/dY, define E=G⊙σ′(Z). The two backward products are dX=EWᵀ and dW=XᵀE. The attachment's formulas are correct for a linear layer; an activation requires E, not raw G.

A derivative is a linear map at the current input. In Euclidean coordinates, its dagger is its transpose and applying that transpose is the pullback. For a nonlinear function, this is the dagger of its Jacobian, not a global inverse of the nonlinear function. Parameter gradients must also be included. Category theory provides composition laws and an interface for legal rewrites; it does not itself cut the scalar multiplication count.

Your paired stream (Xnext W, Eold Wᵀ) is a legitimate scheduling target when both streams are ready and use the same version of W. It still contains two products. A tile can serve both products while resident in fast memory; keeping an entire large W resident is generally impossible. Traffic savings depend on tile traversal, layouts, occupancy and cache behavior. Extra accumulators can reduce occupancy and make the paired kernel slower.

The displayed rectangular block multiplication needs repair. A dimensionally valid identity is

    blockdiag(Xnext, Eold) · blockdiag(W, Wᵀ)
      = blockdiag(Xnext W, Eold Wᵀ).

This is a symbolic specification. Do not allocate its zero blocks or compute them. dW=XoldᵀEold remains a third contraction: it consumes Eold, not the already-computed dX. Old activations, masks and normalization statistics must remain available or be recomputed correctly.

Pipelining consecutive microbatches inside one gradient-accumulation window can preserve synchronous training. Overlapping an optimizer update requires explicit weight versions: both forward and backward must use the intended old weights. Updating a shared parameter before its final use can silently change the algorithm.

Predictive coding is a separate research direction. Exact reverse differentiation has been demonstrated for a prescribed zero-divergence inference method on computational graphs; that does not establish the attachment's claim that ordinary training becomes one free, simultaneous local relaxation. Inference iterations and scheduling must be counted. See [Reverse Differentiation via Predictive Coding](https://arxiv.org/abs/2103.04689).

## 2. The new implemented rule: an RMSNorm sufficient core

Consider an existing bias-free linear layer followed by RMSNorm and another linear layer:

- X is m×k; A is k×n; B is n×p.
- γ has n entries; Γ=diag(γ); b has p entries.
- Z=XA.
- r_i=(sum_j Z_ij²/n + ε)^(-1/2), with ε>0.
- Y=diag(r) Z Γ B + b.

Ordinary evaluation builds the wide m×n activation Z and differentiates through it. Instead, compute

    Q = AAᵀ                 (k×k)
    P = AΓB                 (k×p)
    q_i = x_i Q x_iᵀ/n + ε  (one number per row)
    r_i = q_i^(-1/2)
    U = XP                  (m×p)
    Y = diag(r) U + b.

Why does this work? Expanding a sum of squares gives

    ||x_i A||² = x_i (AAᵀ) x_iᵀ.

Matrix associativity gives

    (XA)ΓB = X(AΓB).

RMSNorm divides each row by one scalar. That scalar can be computed from Q, so the wide activation is not needed by the rest of this block. This is an exact identity over real numbers, without approximating ranks, pruning weights or dividing by γ. Zero γ entries are allowed.

The smaller graph still depends on every entry of A, B and γ. Compute Q and P again for the current parameter version; treating them as detached constants would produce wrong gradients.

This applies to the stated block. Many Transformer RMSNorm layers do not have a preceding bias-free dense expansion and following contraction in this arrangement; a compiler cannot manufacture the arrangement across residual branches or nonlinear gates.

## 3. Full gradients, step by step

For an incoming m×p cotangent G, let

    H = diag(r) G
    T = Xᵀ H
    s_i = -(r_i³/2) · dot(G_i, U_i)
    S = Xᵀ diag(s/n) X.

Here T is k×p and S is k×k. Applying the chain rule to the small graph gives

    dP = T
    dQ = S
    dX = H Pᵀ + diag(s/n) X(Q+Qᵀ)
    dA = (S+Sᵀ)A + T Bᵀ Γ
    dB = Γ Aᵀ T
    dγ = diagonal(Aᵀ T Bᵀ)
    db = sum_rows G.

The diagonal formula specifies a contraction: implement it as a reduction, without constructing an n×n matrix. For example, sum elementwise (AᵀT)⊙B along the p dimension.

Q and S are symmetric in exact arithmetic. This also suggests a future custom pullback that reuses XQ and combines transpose contributions. The delivered Python path uses ordinary autograd on the reduced graph; these extra symmetry optimizations are not implemented or benchmarked here. Rounding may make a computed Q slightly asymmetric, so replacing two rounded terms by twice one term requires the algebraic accuracy contract.

The normalization derivative is not omitted. For a row z and direction v,

    d(RMSNorm without γ)[v]
      = r v - (r³/n) z dot(z,v).

This is a scalar-scaled identity plus a rank-one correction. Its transpose has the same form. That is the useful connection between nonlinear normalization, numerical linear algebra and dagger composition.

## 4. When arithmetic actually falls

For all input and weight gradients, the dominant dense-product model is

    native: 6mn(k+p)
    core:   6k(k+p)(n+m).

The core wins in this leading model exactly when

    k(n+m) < mn, or equivalently k < mn/(m+n).

This count includes recomputing Q and P and differentiating through them. It excludes pointwise normalization, γ operations, copies, guards, dispatch and low-level GEMM workspace. The runtime's cost function includes multiply/add subtraction terms and demanded-gradient flags; positive modeled saving is necessary, not sufficient for real speed.

For m=512, k=32, n=2048, p=16, the leading ratio is about 7.81%, or about 92.19% fewer main-product operations. This refers to this isolated block's forward plus gradients. Dense parameter/gradient storage and optimizer state remain. It is not a 92% reduction of a whole model or proof that one GPU replaces four.

The smaller computation need not use fewer GEMM calls. More small products can perform much less total arithmetic than fewer huge products. A custom kernel should optimize this smaller graph, rather than bundle the original huge products and call that arithmetic elimination.

## 5. What Lean verifies, and what testing verifies

`lean/OjasXMath/RadialCore.lean` adds seven checked statements:

1. Arbitrary finite-index RMS quadratic-moment reassociation.
2. Matrix forward reassociation with arbitrary row scaling and γ.
3. An actual `HasDerivAt` theorem for h(t)/sqrt(q(t)), given derivatives of h and q and q(t)>0. This supplies the analytic scalar normalization rule; it is not by itself a full multivariate RMSNorm certificate.
4. Arbitrary finite-index adjoint pairing for a scaled identity minus rank-one derivative.
5. The joint leading-product cost identity.
6. A sufficient strict profitability inequality under positive dimensions.
7. The paired-stream output specification, a definitional statement, not a bandwidth theorem.

Every theorem is included in the root axiom audit. There are no admitted proofs or new custom axioms. The existing certified-composition framework explains how primitive derivative/adjoint contracts compose; a complete Lean refinement proof of this Python RMSNorm implementation is not delivered.

CPU differential tests compare native PyTorch outputs, all original parameter gradients, Hessian-vector products and numerical first/second derivatives. Additional tests cover all gradient-demand combinations, zero γ entries, changed weights, strided operands, optional scale/bias, AMP/strict/native fallbacks and actual `torch.compile` graph rewrites. The backend keeps parameter objects and the usual optimizer interface.

Floating-point reassociation is not bitwise equivalence. The benchmark records both elementwise tolerance and relative L2 errors. Some FP32 gradients in the favorable large block fail the pre-existing elementwise rtol=3e-4, atol=3e-5 comparison near zero, despite relative L2 errors around 10^-6. The experiment's separately stated relative-L2 criterion is 10^-5; those elementwise failures remain visible in the report. Strict mode is the default, and algebraic mode requires accepting its rounding contract.

Current conditioning and finite-value guards are heuristics. They reject nonfinite operands/intermediates and very small moments, but do not guarantee an error bound for every finite input or backward cotangent. An ill-conditioned or overflowing backward can differ from the native path. Production integration needs a declared numerical contract, mixed-precision validation and target-specific fallbacks.

## 6. Use without changing ordinary model code

For a supported model containing the pattern:

```python
from ojasx2 import make_backend
backend = make_backend(accuracy='algebraic', contraction_policy='rms_core')
compiled_model = torch.compile(model, backend=backend, fullgraph=True)
# Existing loss, loss.backward(), and optimizer.step() remain the same.
```

This is an opt-in backend policy, not automatic silent deployment. The FX pass recognizes captured functional RMSNorm between two linear calls, requires a bias-free first linear and single-use intermediates, and keeps unsupported graphs native. Runtime guards admit CPU FP32/FP64 and sampled MPS FP32 (including contiguous leading batch/token axes), explicit positive finite epsilon, compatible shapes and no autocast. Default RMSNorm eps=None retains native evaluation. The MPS path uses ATen tensor operations; native GPU machine-code generation is not delivered.

For isolated experiments, `rms_core_linear_chain(x,w1,w2,gamma,bias2,eps,accuracy='algebraic')` uses PyTorch Linear layouts: w1[n,k], w2[p,n]. An optional evidence list records the selected path and modeled cost. It never caches detached contractions across optimizer updates.

Run `python -B tools/verify_all.py` in the project's dependency environment. `reports/radial-core.json` contains randomized-order CPU timings, output/gradient errors, main-product costs, saved-storage audit and profiler incremental tensor peak for favorable and unprofitable shapes. Profiler tensor storage is not process RSS, physical GPU memory or an all-model capacity measurement.

## 7. Custom GEMM / hardware implementation contract

The following is a design for future lowering, not an already-tested GPU kernel.

1. Capture the whole forward/gradient subgraph and validate shape, gradient demand, aliases, consumer count, dtype, epsilon and parameter version. Keep strict graphs on the native implementation.
2. Tile A to form Q=AAᵀ and P=AΓB. Choose traversal to reuse A tiles where capacity allows, without forcing incompatible reductions into one oversized kernel. Keep Q/P on-chip only if resource accounting permits.
3. Stream X tiles. Compute xQ, reduce dot(xQ,x)/n, apply rsqrt, and compute xP. Store small row statistics and the residuals needed by the chosen pullback. Avoid allocating m×n Z; do not replace it by an equivalent hidden workspace.
4. In backward, form T and S with tiled reductions and sufficient accumulators. Emit dA, dB and dγ in tiles. Dense gradient destinations still require storage or an explicitly legal streaming optimizer schedule.
5. Where a next-microbatch forward tile and current-microbatch backward tile are both ready, consider your paired dagger stream to reuse a weight tile. Benchmark it against independently optimized GEMMs; it complements the algebraic reduction.
6. For Apple GPU, CUDA or HIP, choose target-specific tile sizes, layouts, matrix instructions, barriers and reduction order. Never assume a transpose costs nothing or that on-chip capacity is shared identically across vendors. Avoid assuming unsupported simultaneous row/column hardware instructions.
7. Qualify forward, every cotangent, accumulation, optimizer updates and numerical contract on the actual device, including unfavourable shapes. Selected MPS execution is now tested in [GPU qualification](GPU_QUALIFICATION.md); complete memory peak and sustained large-model performance against optimized baselines remain pending.

## 8. What remains difficult

An arbitrary GELU or SwiGLU gate depends on each wide hidden entry, not just its row norm. In general, neither its value nor its derivative can be recovered from Q and P alone. Exact common masks, exact separable derivative factors or another proved sufficient statistic can enable selected rewrites; arbitrary numerical low rank is an approximation and must be labeled accordingly.

Residual branches and other consumers may require the wide tensor even when this block's output does not. Weights, dense gradients and optimizer moments may dominate memory after activation storage falls. Strong OjasX results therefore need global residual planning, shape-aware mathematics, numerical qualification and low-level execution together.

The practical direction is to search for sufficient small statistics of an existing operator, prove that its output and pullback can be reconstructed from them, and compile that smaller calculation. That changes how familiar mathematics is executed while preserving the developer's model and objective.


## 9. Latest measured CPU results

PyTorch 2.8.0, two CPU threads; forward plus all input/parameter gradients. Timings vary between runs; these values come from the current report.

| Shape m,k,n,p | Selected path | Native ms | Selected ms | Speedup | Incremental tensor peak reduction |
|---|---|---:|---:|---:|---:|
| 512-32-2048-16 | rms_core | 2.472 | 0.548 | 4.51× | 95.94% |
| 128-256-512-128 | unprofitable | 0.624 | 0.642 | 0.97× | 0.00% |

A separate 40-step synthetic AdamW trajectory selected the reduced path for 40/40 steps. Maximum gradient relative L2 difference: 2.35e-06; maximum parameter absolute difference: 5.96e-08. This checks one constructed trajectory, not general convergence.

Verification at the original RMS extension run: 128 Lean statements, 33 scoped Z3 checks, 222 passing CPU tests and two device skips. The newly added RMSNorm identities use Lean; the existing SMT models were rerun but do not constitute a new RMSNorm proof.

The subsequent [workload-scaling guide](WORKLOAD_SCALE_MATH_TOOLS.md) records broader tests, multi-axis support and an important optimizer-sensitive FP32 counterexample.
