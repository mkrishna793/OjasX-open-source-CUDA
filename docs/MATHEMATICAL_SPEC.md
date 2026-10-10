# Mathematical foundation and improvements

## 1. The central categorical statement

For a differentiable real map `f : A → B`, reverse-mode differentiation computes

\[
R[f](x,\bar y)=Df(x)^\dagger\bar y.
\]

The derivative is linear in the tangent argument; its adjoint acts on cotangents. With standard Euclidean coordinates this is `J_f(x)ᵀ · dy`. More generally the canonical object is the dual pullback; an identification with an adjoint depends on the chosen inner products. For metric matrices `G_A,G_B`, the adjoint coordinates are `G_A⁻¹ Jᵀ G_B`, rather than a bare transpose.

For a fixed linear map, the derivative is constant and a static dagger suffices. For GELU, SiLU, normalization, multiplication and other nonlinear maps, the derivative depends on primal values. Their reverse maps therefore carry residuals. This distinction agrees with [Reverse Derivative Categories](https://arxiv.org/abs/1910.07065). The category-theoretic foundation predates OjasX; this rebuild verifies specific constructions, not a claim of historical novelty.

The implementation represents a stage as

\[
F_f(x)=(s,y),\qquad B_f(s,\bar y)=\bar x.
\]

Composition stores the paired residual `(s,t)` and reverses the pullbacks:

\[
B_{g\circ f}((s,t),\bar z)=B_f(s,B_g(t,\bar z)).
\]

`Composition.lean` checks forward composition, reverse composition, reverse associativity under tape reassociation, and left/right reverse identities. These are laws of the composition mechanism. They do not automatically certify an arbitrary primitive's derivative or every axiom of an abstract Cartesian reverse differential category.

## 2. A general theorem for activation-memory elimination

Let `encode(x)` be everything the backward pass retains: an output, a flag, a statistic, or a more general tape. Let `needed(x)` be the complete pullback function, or any coefficient sufficient to construct it.

There exists a decoder satisfying `decode(encode(x)) = needed(x)` for every input **if and only if**

\[
\operatorname{encode}(x)=\operatorname{encode}(y)
\Longrightarrow
\operatorname{needed}(x)=\operatorname{needed}(y).
\]

`residual_sufficiency_iff` proves this for arbitrary types, with a nonempty result type to supply a default on unreachable encodings. The reverse direction uses classical choice; it establishes existence, not an efficient computable reconstruction algorithm.

This generalizes the output-only question to any proposed residual compression. If two inputs collide in the retained representation while requiring different pullbacks, exact decoding is impossible. The Lean library also proves concrete impossibility results for the derivative of `x²` and the left derivative of a product. The old GELU/SiLU output-recovery approaches were rejected by numerical counterexamples in the earlier audit; this rebuild does **not** claim a complete Lean formalization of those transcendental activations.

`exact_boolean_encoding_requires_bits` proves that an injective encoding of all `n`-flag vectors into `b`-bit vectors requires `n ≤ b`. This is a worst-case, fixed-width lower bound for independent flags. It does not prohibit compression on restricted inputs, nor force storing flags when already-live tensors determine them.

## 3. Concrete residual strategies

| Operation | Retained information | Backward policy | Guarantee here |
|---|---|---|---|
| ReLU | `ceil(numel/8)` bytes of threshold flags | Select `dy` or zero | Observed raw-bit equality against PyTorch, including selected NaN/zero cases |
| Exact/tanh GELU | Forward input | Matching `aten.gelu_backward` mode | Observed first-order forward/backward bit equality |
| SiLU | Forward input | Native first-order backward; differentiable formula for higher orders | Observed first-order bits; double-precision numerical higher-derivative checks |
| SwiGLU | Gate and up inputs | Recompute pointwise SiLU only | Observed first-order bits; avoids one saved SiLU output |
| RMSNorm | Input, gamma, one reciprocal-RMS scalar per row | Direct VJP; recompute statistic for higher derivatives | Forward bits of the stated expression; gradients validated with tolerance |

ReLU's threshold flag is `not (x <= 0)` to match the tested PyTorch backward behavior for NaN inputs. This is a runtime convention, distinct from classical real differentiation at zero. No classical derivative is asserted at ReLU's kink.

For RMSNorm with last dimension `n`, positive epsilon, `r = (mean(x²)+epsilon)^(-1/2)`, `y = (x*r)*gamma`, and `g = dy*gamma`,

\[
dx=r g-\frac{x r^3}{n}\sum_j g_jx_j,
\qquad d\gamma=\sum_{\text{rows}}dy\,(x r).
\]

Zero and negative gamma are valid. No division by gamma or recovery of input norms from the output is attempted. Lean checks the algebraic row-statistic rewrite; numerical gradcheck checks the implementation, including the statistic as an auxiliary differentiable output. This is not a Lean proof of the complete RMSNorm analytic derivative.

## 4. Parameter gradients and shared projections

For `C = A B`, the real directional derivative is `dC = dA B + A dB`. The Lean Frobenius pairing theorem identifies both cotangents:

\[
\bar A=\bar C B^T,\qquad \bar B=A^T\bar C.
\]

Thus a trainable linear layer generally needs both matrix products. Merely reversing a graph cannot eliminate the weight gradient. Fixed/frozen parameters and graph structure may make individual products unnecessary; the IR must express which cotangents are requested.

For shared-input projections `Y_i=X W_iᵀ`, concatenate the weights and outputs to use one larger projection. Autograd then computes one larger input-gradient product and one weight-gradient product. The measured three-head example changes full forward/backward ATen GEMM events from `9` to `3`. The leading dense arithmetic is unchanged. Reduced dispatch and opportunities for reuse are potential performance benefits, not a measured speedup.

This transformation changes input-cotangent accumulation and GEMM shapes. It is disabled in strict mode and tolerance-tested in algebraic mode. Fanout always requires adding cotangents from all consumers; sharing a primal is not permission to drop a branch gradient.

## 5. Rank-7 block multiplication and its dagger

`winograd_block_equivalence` proves the supplied Winograd scheme for any noncommutative ring. Multiplication order is preserved, so the theorem specializes to matrix blocks as well as scalar rings. The implementation supports compatible even rectangular 2D matrices, rather than only square matrices. Odd sizes and mismatched operands are rejected in the experiment.

The algorithm uses four A-encoding additions, four B-encoding additions, seven smaller products, and seven decoder additions. The adjoint decoder builds seven channels from four additions and one negation, and its pairing identity is proved. The rebuild also reverses each shared encoder circuit using seven additions/subtractions rather than ten independent transpose-column operations; Lean proves both optimized encoder daggers. Explicit backward reuses the encoded operands and uses fourteen smaller products. Its forward/backward product count is `7 + 14 = 21`, compared with `1 + 2 = 3` large products for ordinary multiplication. Fewer scalar multiplications therefore do not mean fewer GEMM calls.

For positive even rectangular dimensions `(m,k,n)`, counting scalar multiplication/addition/subtraction as one operation and using conventional leaves (excluding unary sign changes, data movement and launch overhead):

\[
T_\mathrm{dense}=2mkn-mn,\qquad
T_\mathrm{Winograd}=\tfrac74mkn+mk+kn.
\]

The saving is `mkn/4 - mn - mk - kn`. It is positive for square `n > 12`, zero at `n=12`, and can be negative for thin matrices. At square `n=512`, this model saves about `12.22%` forward scalar operations. It says nothing about tensor-core utilization, launch overhead, temporary traffic, or latency.

With the shared decoder and both seven-add encoder daggers, the corresponding backward and joint formulas are:

\[
T_{\mathrm{backward}}=\tfrac72mkn+mn,\qquad
T_{\mathrm{joint}}=\tfrac{21}{4}mkn+mk+kn+mn.
\]

Lean proves these identities and the square joint crossover `n > 8`, compared with dense joint cost `6n³-3n²`. At `n=512`, the joint scalar-operation model decreases from `804519936` to `705429504` operations, about `12.32%`. This improvement assumes the encoded tape is retained; it is not a certificate of lower training memory or measured runtime.

For two recursive square levels, the corrected formula is

\[
\tfrac{49}{32}n^3+\tfrac{29}{4}n^2.
\]

At `n=512` the modeled saving is about `22.65%`. This is a proved cost identity for the stipulated schedule. The reference code implements one level only; two-level training execution has not been certified.

The conditional depth-first workspace recurrence is

\[
W_0(n)=0,\quad
W_{d+1}(n)=\tfrac74n^2+W_d(n/2)
\leq\tfrac73n^2.
\]

Lean proves it for every depth. It assumes seven quadrant buffers and one live child, measured in scalar storage units. It does not bound the training tape retained across all recursive products, GPU caches, external buffers, or the actual Python implementation's peak allocation. A training-memory theorem requires a separate joint forward/backward liveness schedule.

## 6. Machine arithmetic and exactness

The real S/A reconstruction `(w+wt)/2 + (w-wt)/2 = w` is proved. Z3 also proves that its rounded binary32 execution can fail: `w=1`, `wt=100000000` reconstructs zero under round-to-nearest ties-to-even. Such real rewrites require a floating-point contract.

The Lean real identities describe the derivatives of real operations. PyTorch backpropagation is a particular floating implementation of those derivative formulas; it is not the classical derivative of each discrete rounding instruction. Matching PyTorch at the bit level therefore requires preserving its operational conventions, accumulation order, intermediate precision, casts, exceptional values and selected kernels.

Strict mode must retain those operations or supply a machine-level equivalence proof for the specific replacement. Reproducibility means repeatability for a pinned execution contract. Bounded-error mode admits qualified reordering. Neither category theory nor a real proof makes arbitrary-GPU output bits identical; [PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) explain why execution arrangements matter.

## 7. What became stronger

The mathematical claims now have explicit types, assumptions, complete proofs, and kernel-reported dependencies. The Winograd theorem is ring-general; its implementation is rectangular. Residual sufficiency is an iff for arbitrary encodings, rather than an activation-specific heuristic. A binary-storage lower bound prevents unsupported promises of universal zero-state nonlinear backward. Reverse composition supports distinct primal/cotangent/residual types. Cost and workspace statements have corrected constants and explicit scheduler assumptions.

The next mathematical target is a verified tensor-IR refinement relation: real denotation, chosen floating operational semantics, residual sufficiency, and alias/liveness invariants attached to each transformation. That is the bridge from these mathematical proofs to a certified compiler, and remains future work.

## 8. Unified circuit gradient and operand-only execution

The follow-up implementation adds a complete exact expansion and actual `HasDerivAt` proof for arbitrary directions through a finite bilinear circuit, then characterizes both cotangents by a shared decoder adjoint. It also proves the operand-only cost formulas, correct sequential accumulation and conditional scratch/tape statements. See [the full theorem and backend guide](STREAMED_DAGGER_THEOREM.md) and [StreamedDagger.lean](lean/OjasXMath/StreamedDagger.lean).

The operand-only rank-7 scheme trades pointwise reconstruction for smaller retained tape; its joint scalar count is `21mkn/4 + 11(mk+kn)/4 + mn`. This differs from the cached formula above. The guarded FX backend also supports a pointwise-memory rewrite that preserves native GEMMs, with measured bit-equal CPU FFN gradients and a deeper-stack tensor-peak reduction. Performance, memory accounting and Apple hardware qualification are reported separately in the new guide.

## 9. Certified multilinear extension (7 October)

See [BACKPROP_MATH_STRUCTURE.md](BACKPROP_MATH_STRUCTURE.md) for the new complete proof structure: actual Fréchet derivatives for certified primitives, composition and adjoint uniqueness, arbitrary-finite-operand analytic cotangent rules, residual re-encoding, zero-safe scalar circuits, local error amplification and sphere/Stiefel tangent projections. Its arithmetic comparison explicitly includes ordinary retained-prefix reverse mode; it does not infer a GEMM or whole-training speedup from a smaller independent-formula count.


## Joint linear-chain contraction and conditional physical bound

The new [Contraction.lean](lean/OjasXMath/Contraction.lean) proves `Y=(AB)C=A(BC)`, the three reassociated joint pullbacks using shared `P=BC` and `T=AᵀG`, affine forward/second-weight identities, known-factor cotangent formulas, exact scalar-count identities and a conditional more-than-half leading-arithmetic saving for positive dimensions with `4p≤m,n`. It also proves a residual-fraction identity and a conditional capacity/bandwidth lower bound. These are real-algebra and supplied-rate results, not IEEE bit or device timing theorems. [The implementation and measured qualification](CONTRACTION_OPTIMIZATION.md) explain original parameter preservation, native-autograd sharing, nonlinear boundaries and accuracy/memory scope.


## Nonlinear structure and centered-weight moments

[Nonlinear.lean](lean/OjasXMath/Nonlinear.lean) adds 25 statements: joint node counts/longer associativity; exact ReLU values and analytic derivatives on open branches; runtime-premised common-mask forward/three pullbacks; separable derivative-factor expansion; diagonal-minus-rank-one pairing, softmax zero-sum, attention row-scalar and LayerNorm shift-null identities; dimension-general Gram/shared-cotangent reassociations; conditional more-than-half leading LayerNorm backward saving; linear means/centering, quadratic Gram moments and normalized forward/joint cost. The [research guide](NONLINEAR_AND_CONTRACTION_RESEARCH.md) derives the complete schedules, describes finite Z3/SymPy derivative checks, and distinguishes them from unprovided universal analytic normalization/softmax Lean constructors and machine-level rounding certificates.
