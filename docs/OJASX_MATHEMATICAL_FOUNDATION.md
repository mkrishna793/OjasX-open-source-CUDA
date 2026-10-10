# OjasX mathematical foundation

Canonical specification, version 2.4, refined 9 October 2026.

OjasX's mathematical foundation is a **verified framework for conditional joint forward/backward optimization**. It can express broad differentiable programs and admit cheaper representations when correctness, numerical error, all original gradients, and cost conditions are satisfied. Its current implementations cover selected families. The framework's generality does not establish substantial savings for every family.

This file is the primary mathematical and backend specification. The companion [research and refinement history](OJASX_RESEARCH_AND_REFINEMENT_HISTORY.md) preserves the full progression, counterexamples, measurements and earlier file map. Version 1.8 consolidated the contracts and coverage ledger. Version 1.9 adds the scoped numerical repairs below, two Lean statements, two Rocq statements and three nonvacuous Z3 checks. It adds no foundational axioms or speed measurements.


### Dense nonlinear residual and dense arithmetic refinement, version 2.4

This version removes the sparse-gradient requirement **for a new memory optimization**. Ordinary centered GELU, SiLU, square, leaky ReLU, hardtanh and SwiGLU blocks can retain a smaller first-order activation tape with a fully dense incoming gradient. It also adds a GPU execution candidate for the existing rank-seven dense matrix identity. These are complementary algorithms; their individual benefits cannot be added together into an unmeasured whole-model result.

**Mathematical compiler rule.** A primitive's residual is sufficient when it reconstructs both the values needed by downstream parameter gradients and the primitive's selected pullback. [DenseJoint.lean](lean/OjasXMath/DenseJoint.lean) makes those two obligations explicit in a typed residual contract. It does not assume that nonlinear derivatives are constant, low rank or sparse. A future research primitive can supply a different residual and prove the same obligations. Effects, mutation, RNG state, kink conventions and higher derivatives require their own contracts.

For the ordinary block

$$Z=XW+b,\qquad H=\phi(Z),\qquad Y=HQ,$$

the forward is unchanged. Retain Z and reconstruct H by the original native activation during backward:

$$H=\phi(Z),\quad \bar Q=H^TG,\quad
D=\phi'(Z)\odot(GQ^T),$$
$$\bar X=DW^T,\quad \bar W=X^TD,\quad \bar b=\mathbf1^TD.$$

Every coordinate-dependent slope and every requested original gradient remains. No loss mask or sparse G is needed. Recomputing H costs one pointwise activation evaluation; it does **not** recompute XW or HQ. In the GELU profiler probe, native and candidate each execute six matrix multiplications; activation evaluation changes from once to twice. This is a sufficient-residual/selective-recomputation construction using existing mathematics, not a historically new activation identity.

For SwiGLU, retain Z and U=XW_u+b_u, reconstruct S=SiLU(Z) and H=S⊙U, and use

$$D_Z=\operatorname{SiLU}'(Z)\odot(GQ^T)\odot U,
\qquad D_U=(GQ^T)\odot S.$$

Then compute both original projection gradients, both biases, Q's gradient and the input gradient. Both forward hidden projections are preserved without first-order matrix replay.

**Dense memory measurements.** At M33/K12/N65/R16, GELU and SiLU fresh saved activation storage changes from **17,160 to 8,580 bytes**, and SwiGLU from **34,320 to 17,160 bytes**: **50% less** in each case. Square, leaky ReLU and hardtanh also save 50% in these all-gradient cases. These ratios concern fresh retained activations only. Original inputs/weights, gradient outputs, optimizer state, live backward tensors and allocator peak remain. With element width B, the models are 2MNB→MNB for the pointwise cases and 4MNB→2MNB for the gated case. The absolute saving grows with M×N; the fraction remains 50%. It is not a new asymptotic bound on total model memory.

Native ReLU, tanh, sigmoid and identity already retain a single relevant intermediate in these blocks. They route to native rather than claim an additional reduction. Projection-only training and frozen output projection also route to native conservatively. Strict accuracy, FP16, empty tensors, autocast, repeated operands and declared higher derivatives use native execution. Unexpected higher-order requests on the residual path rebuild the native differentiable graph. Saved original operands retain PyTorch mutation/version checks. This reconstruction does not erase zero first derivatives or their potentially nonzero Hessians.

**Dense arithmetic candidate.** The existing Winograd seven-product identity and its transposed shared circuit now execute as a batched GPU product: one batched call for forward and up to two for the matrix gradients. Logical subproducts number 7+7+7 instead of 8+8+8, although this is still forward followed by backward. It is not a single computation that knows the future loss cotangent. Dense matrices, dense gradients and arbitrary nonlinear neighbors do not invalidate the exact-real matrix identity.

For all matrix gradients, with positive even dimensions, the complete scalar models are

$$C_{native}=6MKN-MN-MK-KN,$$
$$C_{cached}=\tfrac{21}{4}MKN+MK+KN+\tfrac54MN,$$
$$C_{replayed}=\tfrac{21}{4}MKN+2MK+2KN+\tfrac54MN.$$

The extra MN/4 charges the decoder's unary negation. Replay charges both encoders. The runtime also charges only the actually requested gradient products for frozen operands/inference. At 1024³ this models **12.40% fewer scalar operations with cached encodings** or **12.37% fewer with encoding replay**. The leading reduction approaches 12.5%; this one-level rule does not imply 100× reduction. Packing, copies, kernel dispatch, bandwidth and extra buffers are outside these scalar counts and were measured through GPU timing separately.

Caching retains an additional 7(MK+KN)B/4 encoded bytes. Replay avoids retaining those encodings but reconstructs them. The planner uses a conservative logical workspace estimate and rejects a candidate above its caller's budget; this is not an allocator-peak certificate. The nonlinear memory rule currently uses native matrix products: no combined rank-seven-plus-residual speed or memory result is claimed.

**Counterexamples and repair.** Six adversarial MPS cancellation cases fail raw rank-seven relative-error comparisons: three in forward values and three in a parameter gradient. The exact-real identity still holds. The repaired checked path compares forward and all demanded gradients with native products, rejects any disagreement beyond elementwise atol/rtol=2e-4 and relative L2=2e-4, and returns native results on rejection. All six checked cases pass. Reference products add arithmetic and must be charged; checked mode is numerical qualification, not a free speed optimization. Strict mode is native. Algebraic mode is an explicitly selected research candidate without a uniform IEEE error guarantee. Checked local budgets do not by themselves prove long-training or whole-graph error bounds.

**Measured timings.** Ratios below are native time divided by candidate time; below one means the candidate is slower. Each seed uses two warmups and five shuffled synchronized measurements. GELU/SwiGLU include the entire block, dense squared loss, all original gradients and zero-lr SGD, at M1024/K256/N2048/R256. Matrix candidates use 1024³.

| Candidate | Three seed ratios | Repeated >1.05× speed admission |
|---|---|---|
| gelu | 0.891×, 1.035×, 0.990× | Rejected |
| rank7_cache | 0.451×, 0.435×, 0.496× | Rejected |
| rank7_replay | 0.416×, 0.488×, 0.338× | Rejected |
| swiglu | 1.026×, 1.027×, 0.926× | Rejected |

Thus this version establishes broader dense-nonlinear memory coverage, a dimension-general dense arithmetic candidate, and repaired numerical qualification. **It does not establish the requested broad training-speed breakthrough.** The rank-seven reference path is excluded from automatic speed dispatch. The residual schedule is a memory tradeoff, not a speed-admitted default.

**Proof and test inventory.** This refinement adds 20 Lean theorem declarations and eight Rocq statements. They include actual affine-activation differentiation, arbitrary finite coordinate-plus-matrix adjoints, bias and both SwiGLU projection gradients, sufficient-residual obligations, and complete scalar/tape identities. Exact-real calculus requires actual derivative hypotheses; chosen native slopes at kinks are separate conventions. The existing ring-generic Winograd equivalence and arbitrary-dimensional matrix-product pullback remain the basis of the dense arithmetic proof. The whole-project kernel audit now checks 547 Lean and 266 Rocq declarations, with no added optimization axioms and an independent Rocq recheck.

The tool loop synthesizes the transposed shared encoders from U/V/W, checks the exact primal polynomial and all eight scalar block-gradient coordinates, uses eight nonvacuous Z3 checks/counterexamples, and enumerates 576 shape/cache/budget/gradient-demand plans. It searches this specified finite grammar; it does not autonomously invent unrestricted mathematics. All **80 new bounded GPU cases** pass after repairs. They include ten activation families, dense/noncontiguous/nonfinite/large/zero inputs or cotangents, frozen operands, empty/FP16/native fallbacks, mutation checks, nonzero Hessians at zero first gradient, and three SGD plus three AdamW steps in six small causal-attention/normalization/residual model runs. This is sampled qualification, not every-model conformance or long training.

**Use and backend integration.**

~~~python
from ojasx2.dense_joint import residual_mlp, dense_matmul

# Original MPS FP32 tensors; GELU/SiLU/other supported native activations.
y = residual_mlp(x, w, bias, q, kind="gelu", accuracy="algebraic")

# Original two SwiGLU projections and biases are preserved.
y = residual_mlp(x, w, bias, q, up_weight, up_bias,
                 kind="swiglu", accuracy="algebraic")

# Numerical verification with charged native products and fallback.
c = dense_matmul(a, b, accuracy="checked", cache=False,
                  workspace_budget=256*1024*1024)
~~~

The eventual compiler should match private deterministic activation/projection blocks, choose a proven sufficient residual, keep branch cotangents accumulated before backward, and preserve every requested original parameter. Backend implementation should place only required residuals on the persistent tape and use bounded temporary storage for reconstruction. Batch the seven encoded subproducts only after shape, workspace, accuracy and whole-step performance qualification. Packing and encoded cache storage must be included in the hardware planner. These APIs remain an explicit MPS research prototype; automatic arbitrary PyTorch graph rewriting, custom GPU kernel qualification and other-device execution are not completed by this module.

Run the finite proof/evidence workflow with:

~~~sh
.tools/conformance-venv/bin/python -B tools/refine_dense_foundation.py
~~~

It runs symbolic discovery, refreshes the complete Lean/Rocq audit and validates source-matched older and new GPU evidence. Add --gpu to execute the new GPU suite after proofs finish. All numerical ML runs are MPS-only with CPU fallback disabled. Host proof and symbolic processing are permitted. No recurring background task is installed.

Authoritative new evidence: [GPU results](reports/dense-gpu.json), [discovery](reports/dense-discovery.json), [qualification](reports/dense-qualification.json), [whole-project proof audit](reports/spectral-formal-verification.json). Profiler traces are [native](reports/dense-strict-trace.json) and [residual](reports/dense-algebraic-trace.json); they record host dispatch of GPU work, not a GPU timeline or device memory peak.

### Two-sided backward demand algebra, version 2.3

This refinement develops a new OjasX computation rule using existing support algebra and derivative linearity. It requires no further literature search and does not claim a historically new mathematical theory. It transfers the earlier papers' useful design principles—charge construction, differentiate the complete computation, account for live storage—without importing their Fourier, scheduling or quantum-state conclusions into general ML.

For a private deterministic rowwise block, write

$$Z=XW+b,\qquad H=\phi(Z),\qquad Y=HQ.$$

The original full forward remains necessary. Let G be the incoming gradient after **all downstream branches** have accumulated. Define S as the rows of G with any nonzero entry, and T as its columns with any nonzero entry. Then G is exactly reconstructed from G[S,T] with zeros elsewhere. The new schedule first finds this support, then reconstructs only H[S,:]. It never reconstructs the output projection Y[S,T] during backward.

For a pointwise activation, define

$$D_S=\phi'(Z_S)\odot\bigl(G_{S,T}Q_{:,T}^{T}\bigr).$$

The complete requested gradients are

$$\bar X_{S,:}=D_SW^T,\quad \bar X_{\bar S,:}=0,$$
$$\bar W=X_S^TD_S,\quad \bar b=\mathbf1^TD_S,$$
$$\bar Q_{:,T}=H_S^TG_{S,T},\quad \bar Q_{:,\bar T}=0.$$

This works with arbitrary finite slopes on the selected rows: no near-constant GELU slope, low-rank activation matrix, or saturation certificate is needed. SwiGLU uses the native derivative of its two-projection gated map on the same selected rows. All original weight gradients remain dense where mathematically demanded.

The construction follows a different sufficient-information condition from v2.2: discard coordinates annihilated by the actual incoming cotangent rather than seek sparse deviations in the activation derivative. It is an exact-zero optimization; small nonzero cotangents are never pruned. With a dense G, it gives no arithmetic reduction. An intermediate block before attention may receive a dense gradient even when the final loss selects a few tokens. The support of the actual block cotangent must be measured; a label mask cannot be propagated through arbitrary attention.

**Complete leading arithmetic.** With shapes X[M,K], W[K,N], Q[N,R], A active rows and C active output columns:

- Native forward plus backward: 6MN(K+R).
- Candidate: 2MN(K+R) + 6ANK + 4ANC + 2MR.
- The 6ANK includes recomputing the first projection and its two backward contractions; 4ANC includes both projection gradients. The last term is a scalar scan estimate.
- For SwiGLU, replace K by 2K in these expressions to account for both hidden projections.
- Only admit the arithmetic candidate when its entire replay/scan cost is below the native backward cost. Pointwise, allocation, synchronization, gather/scatter and memory traffic still require timing.

The implementation saves the original block operands, avoiding fresh forward activation retention. It still computes the full forward and may need substantial transient memory. Original weights, dense gradient outputs and optimizer state remain. Dense-demand fallback recomputes the native block, so it can trade retained memory for extra arithmetic; it does not promise a free fallback. The refined backward initializes only scattered gradient outputs to zero and produces the other parameter gradients directly.

**Formal scope.** [DemandAlgebra.lean](lean/OjasXMath/DemandAlgebra.lean) adds 12 declarations, including the actual derivative of a fixed-cotangent weighted objective, nonlinear zero-row propagation, support-restricted parameter sums, matrix restriction and complete cost identities. [DemandAlgebra.v](rocq/DemandAlgebra.v) adds four rational algebra/partition statements. They introduce no optimization axioms. The weighted derivative treats G as a fixed first-order cotangent; this does not justify erasing higher derivatives at points where G happens to be zero.

**Runtime boundaries.** [demand_algebra.py](ojasx2/demand_algebra.py) accepts private rowwise native activation/MLP/SwiGLU blocks on MPS. Strict mode is native; algebraic mode is explicitly selected. Non-FP32, empty, repeated-operand and autocast cases select native execution. A conservative input magnitude/dimension envelope screens overflow before zero elision; it is not a universal IEEE error proof. Nonfinite upstream and higher-order backward requests recompute the original differentiable block. Mutation checks use PyTorch's saved-tensor version tracking. Stochastic/stateful user functions, arbitrary custom operators and automatic graph rewriting are outside this prototype.

**Tool workflow.** The bounded symbolic tool checks eight identities/counterexamples, an arbitrary-Jacobian support calculation and 192 shape/support plans. Counterexamples reject tolerance pruning, deleting a branch before cotangent accumulation, treating sparse output gradients as sparse upstream gradients, and inferring a zero Hessian from a zero first derivative. These are finite specified searches, not an automated solution of unrestricted mathematical discovery.

The GPU suite covers ten activation families, five demand patterns, frozen weights, projection-only training, absent biases, noncontiguous inputs, FP16 fallback, nonfinite/large operands, nonfinite upstream, higher derivatives and mutation. Three small causal-attention/residual/normalization models include a shared-weight loss branch and three SGD updates. Timings include full native forward, selected-output loss, all original gradients and zero-lr updates on a synthetic support pattern. No whole-model speed is inferred from these timings.

**Latest scoped outcomes.** All 70 new GPU cases pass after the allocation refinement; maximum recorded relative L2 discrepancy is 5.00e-7. In the M65/K12/N97/R16 correctness cases, fresh saved activation bytes fall from 50,440 to zero for GELU, and 100,880 to zero for SwiGLU. Original operands still occupy 14,372 and 19,416 unique saved bytes respectively; backward scratch and gradient outputs remain. This is not zero total GPU memory.

At M4096/K512/N2048/R512, with 1/16 incoming-gradient rows and 1/4 output columns active, the complete leading arithmetic model estimates 63.01% less work for GELU and 62.15% less for SwiGLU. The original full forward is included. These shapes use a synthetic selected-output objective; they do not establish the same support inside an ordinary complete model.

Repeated full-block speed ratios were 0.51–1.19× for GELU and 0.76–1.40× for SwiGLU. Both fail the repeated >1.05× admission rule. Raw samples show substantial timing variability; no reliable new speedup is claimed. Exact demand restriction broadens the mathematical applicability beyond near-affine activations, but its arithmetic benefit still requires sufficiently sparse incoming demand.

Use the explicit primitive:

~~~python
from ojasx2.demand_algebra import demand_mlp
y = demand_mlp(x, w, b, q, kind="gelu", accuracy="algebraic")
loss = downstream(y)
loss.backward()  # support is obtained from the actual incoming gradient
~~~

Recheck with:

~~~sh
.tools/conformance-venv/bin/python -B tools/refine_demand_foundation.py
~~~

The runner searches the finite symbolic family, rechecks the full Lean/Rocq foundation, validates prior source-matched GPU evidence and admits new correctness/memory/speed separately. Add --gpu to repeat the bounded MPS suite after formal compilation. The current evidence is [demand qualification](reports/demand-qualification.json), [raw GPU tests](reports/demand-gpu.json), and [symbolic checks](reports/demand-discovery.json). Host tools perform proofs and symbolic algebra; numerical ML uses only MPS with CPU fallback disabled.

### Observable cotangent and nonlinear-defect foundation, version 2.2

This revision adds a **quotient-space formulation of requested gradients**, a joint nonlinear correction algorithm and a streamed nonlinear-memory schedule. It uses existing linear algebra, differential pullbacks and exact-support decompositions. No foundational axiom is invented or assumed to make an optimization true. This is a new component of OjasX, not a claim of historically new quotient mathematics or a universally faster backpropagation algorithm.

**What changes in the mathematical target.** For a private linear stage with input X[M,K], weights W[K,N] and hidden cotangent D[M,N], the requested gradients observe D through

$$\mathcal O(D)=(X^T D,\;D W^T,\;\mathbf1^T D).$$

Two cotangents are equivalent when they give the same observations. The useful object is therefore a class in the quotient by the kernel of this joint linear map. A complicated or full-rank D can still have a simpler representative. The local derivative is linear even when the primal network is nonlinear; the quotient is defined at that particular forward state and for the actual requested observations. Category-theoretic composition supplies the context in which the observations are used. It does not supply a free way to construct them.

Choose P and Q satisfying XᵀP=Xᵀ and QWᵀ=Wᵀ. Include the constant column in X, or independently require 1ᵀP=1ᵀ, to preserve the bias. Then

$$D_* = PD + DQ - PDQ$$

preserves the requested input, weight and bias gradients. If P=UA and Q=BV, then

$$D_* = U(AD)+(DB-U(ADB))V.$$

This gives two factor terms with total inner width a+b. The Lean statement proves this factorization and the width count; it is not a separate matrix-rank theorem or a cheap-construction theorem. If P and Q are full identity maps there need be no useful compression. A hidden-gradient hook that requests D itself changes the observation map and invalidates discarding its unseen components.

The symbolic tool constructed an exact example with 48 hidden entries, 30 independent observed degrees of freedom and an 18-dimensional joint nullspace, including bias observations. Z3 refuted moving this quotient across an arbitrary coordinate-dependent nonlinear derivative. The rewrite must occur after that derivative, or discharge a new observation-preservation proof. **The open algorithmic problem is constructing the sufficient factors without first paying the original dense contraction cost.** Merely materializing D* is not an acceleration.

**Executable joint nonlinear rule.** Consider the private block Y=φ(XW+b)H, with H[N,R] and upstream G[M,R]. Define Z=XW+b, choose per-column c and o, and retain both defects:

$$E_f=\phi(Z)-(Z\operatorname{diag}(c)+\mathbf1o^T),\qquad
E_s=\phi'(Z)-\mathbf1c^T.$$

Let C=diag(c), T=XᵀG, s=1ᵀG and Dₑ=Eₛ⊙(GHᵀ). Then all original gradients are

$$\bar X=G(WCH)^T+D_eW^T,$$
$$\bar W=(TH^T)C+X^TD_e,\qquad
\bar b=CHs^T+D_e^T\mathbf1,$$
$$\bar H=CW^TT+(Cb+o)s+E_f^TG.$$

Bias vectors in these equations use column orientation; s is a row. The implementation shares T and the smaller contracted weights. It preserves the native forward. Sparse entry defects use bounded scatter corrections; small exceptional column sets use ordinary native VJPs on those columns. No nonzero defect is removed by a tolerance threshold. In the dense case, the identities remain valid but their representation need not be cheaper.

For ReLU/leaky-ReLU/hardtanh, strict activation intervals certify common affine pieces. Lean checks six interval/convention lemmas. Hardtanh endpoints are excluded from the exterior/interior interval certificate: a GPU counterexample showed that this installed MPS backward returns a unit slope at ±1. The refined code delegates those entries to native slope inspection or the native column calculation. Classical differentiability at a kink is not asserted.

For smooth GELU/SiLU/tanh/sigmoid, the prototype compares **rounded native forward values and unit-cotangent slopes**. Saturation can make the measured support small. This is not a proof that real smooth derivatives become exactly affine, and bounded upstream guards do not establish an all-input floating-point error theorem. Reassociation, cancellation and native operation order still require numerical qualification. Centered ordinary wide GELU/SiLU generally yields dense defects and receives no arithmetic-reduction certificate.

**Cost and applicability.** Native leading joint matrix work is 6MN(K+R). Both schedules retain the original 2MN(K+R) forward. The candidate backward estimate is 4MKR+6KNR+4E(K+R), plus scalar terms and a visible certificate estimate. The arithmetic inequality is checked before admission. Certificate scans, nonlinear evaluation, reductions, indexing and synchronization are also timed; a favorable FLOP estimate alone does not admit speed. The strongest current test shapes have a narrow downstream width R compared with K and N. A usual Transformer MLP with R≈K may not satisfy this inequality.

The implementation is [cotangent_quotient.py](ojasx2/cotangent_quotient.py). All X/W/H gradients are required for this initial compact route; frozen operands are conservatively native. Strict accuracy, non-FP32, empty shapes, autocast, declared higher derivatives and invalid numerical domains select native computation. Unexpected higher-order requests recompute the original differentiable block. A large/nonfinite upstream cotangent also recomputes the native block. A failed support search can still incur scan overhead. The API is explicit; automatic graph discovery, arbitrary hooks, stochastic blocks and every future primitive are not implemented by this module.

**Broad memory schedule for ordinary nonlinear blocks.** `streamed_mlp` supports ReLU, leaky-ReLU, hardtanh, exact native GELU, SiLU, tanh, sigmoid, square, identity and SwiGLU. It retains the original operands and recomputes small row chunks to accumulate every original parameter gradient. The full native forward is still evaluated. The row chunk bounds backward activation workspace, not the full forward peak or parameter state. This schedule does not depend on sparse nonlinear defects, but spends recomputation and can be slower. It is a checkpointing/streaming construction integrated with the all-gradient contracts, not a claim that nonlinear arithmetic disappeared.

**Proof and qualification boundaries.** [Lean CotangentQuotient](lean/OjasXMath/CotangentQuotient.lean) adds 23 theorem declarations: observation equivalence, kernel characterization, dimension-general quotient contractions/factors, actual derivative subtraction, joint defect identities, finite-support elision, costs and activation intervals. [Rocq CotangentQuotient](rocq/CotangentQuotient.v) adds seven rational algebra/partition declarations. Lean's existing real derivative foundation supplies calculus; Rocq does not newly prove transcendental GELU derivatives. The full audit checks every existing module and independently rechecks Rocq. The Python/Metal implementation is not extracted from these proofs.

The bounded discovery uses 12 nonvacuous Z3 checks/counterexamples, an exact symbolic nullspace, four scalar gradient identities and 144 shape/support plans. GPU qualification includes all original gradients, ten activation families across the two APIs, empty/noncontiguous/frozen/nonfinite/half cases, clipping endpoints, mutation checks, shared branches, higher derivatives, and small attention/residual/normalization models through three SGD and three AdamW updates. These are bounded tests, not coverage of every PyTorch operator or future model. Correctness checks require both elementwise atol=rtol=2e-4 and relative L2≤2e-4. Saved-storage and repeated speed admissions are independent.

**Version 2.2 measured results.** All **80** new bounded GPU cases pass after the boundary repair. The largest recorded relative L2 discrepancy is **1.42e-6**, below the stated 2e-4 threshold. The latest report supersedes the initial and failed intermediate runs.

| Case | Result | Interpretation |
|---|---:|---|
| Biased GELU/SiLU block, M128 K8 N256 R4, two exceptional columns | 262,144 → 4,112 fresh saved bytes, **98.43% less** | Original native nonlinear forward; rounded-slope numerical scope |
| ReLU on the same shape/support | 131,072 → 4,112 fresh saved bytes, **96.86% less** | Piecewise-affine interval certificate |
| Same eligible block's joint arithmetic estimate | 2,359,296 → 1,131,520, **52.04% lower** | Matrix/support arithmetic plus scalar certificate estimate; not total instruction count or latency |
| Centered SwiGLU, M1024 K64 N1024 R64, chunk32 | 16,777,216 → 0 **fresh activation** saved bytes | Original operands still retain 1,056,768 unique bytes; backward recomputes chunks; no zero-memory claim |
| ReLU timing, M2048 K128 N2048 R16 | **0.47–0.62×** native/candidate | Slower; speed rejected |
| Larger narrow-output ReLU, M4096 K2048 N2048 R16 | **0.995–1.394×** over three seeds | Does not meet repeated >1.05× admission |
| Biased GELU / SiLU on smaller timing shape | **0.24–0.32× / 0.17–0.28×** | Certificate/construction costs dominate; speed rejected |
| Centered streamed GELU | **0.945×**, one timing seed | No repeated speed qualification |

The two stored-memory percentages concern the small correctness shape, not the larger timing shapes. No new method is speed-admitted by this run. The broader nonlinear memory route works in the sampled cases; simultaneous broad arithmetic and speed savings remain a research target.

**Use and repeat.** From the project root, with tensors already on MPS:

```python
from ojasx2.cotangent_quotient import defect_projection, streamed_mlp
# Original X, W, b, H; explicit acceptance of changed floating-point order.
y = defect_projection(x, w, b, h, "relu", accuracy="algebraic")
# General supported nonlinear memory route, with additional recomputation.
y = streamed_mlp(x, w, b, h, kind="gelu",
                 accuracy="algebraic", chunk_rows=32)
# SwiGLU additionally supplies its original up_weight and up_bias.
```

`accuracy="strict"` is the default. The public block boundary must remain private; caller-visible intermediate gradients are not reconstructed. No existing developer model is silently substituted with a radial or polynomial network.

```sh
.tools/conformance-venv/bin/python -B tools/refine_cotangent_foundation.py
```

This runs finite discovery, the complete Lean/Rocq audit and prior source-matched qualification, then validates the new GPU evidence. Add `--gpu` to rerun the bounded GPU suite after proofs finish. Numerical ML uses MPS with CPU fallback disabled; symbolic/proof tools run on the host. The authoritative new ledger is [cotangent qualification](reports/cotangent-qualification.json), with [raw GPU cases](reports/cotangent-quotient-gpu.json) and [discovery](reports/cotangent-quotient-discovery.json). Initial slow results and the hardtanh counterexample are preserved. No perpetual discovery service or recurring automation is installed.

### Loss-directed compiler and GPU conformance, version 2.1

This revision extends the foundation to ordinary **linear classification and vocabulary cross-entropy losses**, with the original GELU, SiLU/SwiGLU, attention, normalization, trainable parameters and tied weights retained. Its useful structure comes from the objective's support and finite-sum decomposition. This is classical loss algebra and recomputation, now integrated with the existing derivative-adjoint framework; it is not a new universal matrix multiplication algorithm.

The implementation is [loss_compiler.py](ojasx2/loss_compiler.py). The conservative [FX pass](ojasx2/loss_fx.py) recognizes a private, single-use `Linear → cross_entropy` output in a traced model. It keeps shared/observed logits, custom modules and module hooks on the native route. Runtime guards retain native handling for strict accuracy, higher derivatives, autocast, non-FP32 inputs, soft targets through the FX adapter, non-scalar reductions, empty/all-ignored inputs and invalid numerical domains. Invalid labels are rejected before GPU indexing. The direct primitive accepts index targets and sum/mean losses; it is not a replacement for every PyTorch loss signature. Arbitrary Python tracing effects and a production `torch.compile` backend are outside this pass.

**Mathematical rule.** For feature rows H, classifier W, logits z=HW+b, fixed class weights c, smoothing ε and target t, set

$$
a_j=((1-\varepsilon)\mathbf{1}_{j=t}+\varepsilon/V)c_j,\quad
m=\sum_j a_j,\quad
L=m\log\sum_j e^{z_j}-\sum_j a_jz_j.
$$

The exact-real logit cotangent is

$$g_j=m\,\operatorname{softmax}(z)_j-a_j.$$

Lean proves the actual derivative of the finite log-partition function, this weighted derivative and cotangent pairing, the zero-sum cotangent, smoothed weight mass, joint matrix dagger, arithmetic conditions, storage ordering and conditional numerical error propagation. Rocq checks the rational algebra and finite partition contracts. The implementation reuses **native local cross-entropy VJPs** rather than assuming that an independently rounded closed formula is bitwise identical. Mean reduction uses the global sum of target class weights (or active count without weights), including under smoothing. It never averages unequal chunk means.

**Two schedules.** Let N be the total output rows, A the non-ignored rows, K feature width and V vocabulary size.

- **Selected rows:** project only A active loss rows. Leading matrix work for the forward pass and both matrix gradients falls from 6NKV to 6AKV. This is an exact-real zero-cotangent elimination. Ignored prefix tokens remain in upstream attention and can still receive gradients through active tokens.
- **Replayed chunks:** evaluate at most B active rows at a time and replay only that head chunk during backward. Leading matrix work is 8AKV when both matrix gradients are demanded; live logit size is bounded by B×V. Without ignored labels, recomputation adds matrix work. All original dense parameter gradients and optimizer states remain necessary.

For replay, the leading arithmetic improves when 4A<3N, and is at most half the dense baseline when 8A≤3N. These bounds exclude validation, indexing, loss arithmetic, accumulation, the upstream model and optimizer. The runtime's `head_plan` adjusts for demanded gradients, including projection replay during bias-only training. Its six-logit working estimate and head budget are planning estimates, not a certified bound on total Metal peak memory. The direct primitive requires enough budget for one vocabulary row; vocabulary-axis tiling remains a separate extension.

**Numerical contract.** Exact-real equivalence is distinct from floating-point equivalence. Chunk summation and selected-row GEMMs can change rounding. The checked FP32 sample criterion remains both `atol=rtol=2e-4` and relative L2 error ≤2e-4; matching NaN/Inf semantics are checked on native fallback cases. An input/weight magnitude envelope screens head overflow risk. It does not prove every future operation or gradient finite, nor certify arbitrary class weights, training lengths, hardware reductions or cancellation patterns. Default accuracy is `strict`; algebraic rewrites are explicitly requested and only the recorded domains are GPU-qualified.

The new source proofs are [Lean LossCompiler](lean/OjasXMath/LossCompiler.lean), with 17 theorem declarations, and [Rocq LossCompiler](rocq/LossCompiler.v), with nine declarations. They add no axioms. Conditional probability errors propagate as |δg|≤|m|δp, and a dense pullback's component error is bounded by δg times the sum of absolute weights. Rounding-error inputs to those bounds still require numerical analysis; the Python and Metal implementations are not extracted verified floating-point code.

**Repeatable evidence workflow.** [refine_loss_foundation.py](tools/refine_loss_foundation.py) runs bounded discovery, the entire existing Lean/Rocq kernel audit, prior source-matched qualifications, optional new GPU checks, provenance/hash validation and separate correctness/memory/speed admission. Discovery covers 20 nonvacuous Z3 checks/counterexamples, three SymPy derivatives and 108 shape decisions. It refutes incorrect weight mass, chunk means, attention-token deletion and finite-input-only overflow assumptions. It searches stated candidates and conditions; it does not invent arbitrary foundational axioms.

The isolated environment `.tools/conformance-venv` contains Python 3.11, PyTorch 2.8.0, TorchCTS 0.4.1 and BackendBench 0.1.0. [Pinned dependencies](configs/conformance-requirements.lock), upstream commit hashes and a pinned 19,171-row profile corpus are recorded. The original numerical environment remains unchanged.

- **TorchCTS:** its real OpInfo registry supplies 36 bounded FP32 cases over nine operator families. Samples are generated directly on MPS; the stock CPU-reference runner is not used. OjasX GELU/SiLU backward and packed-ReLU residuals are compared to native MPS, alongside both-operand matrix daggers and arbitrary-axis softmax VJPs. Other tested operators use native fallback. [The manifest](configs/torchcts_mps.py) passes the official checker.
- **BackendBench:** an allowlisted AST parser loads 16 original profiles across eight operator families, preserving shapes, dtypes and strides. All 19,171 profile selection decisions are recorded; oversized, synthetic and out-of-scope records are filtered before allocation. No profile is silently shrunk or counted as a complete model. The stock CUDA runner and its CUDA-to-CPU fallback are not invoked. Numerical profile values and original model context are not present in the corpus.
- **Profiler:** `torch.profiler(profile_memory=True, record_shapes=True)` exports [native](reports/traces/loss-strict.json) and [candidate](reports/traces/loss-algebraic.json) Chrome traces. In this installed PyTorch/MPS build, these contain host dispatch events, not Metal kernel timing or exact GPU memory allocation. Synchronized timings, saved-tensor accounting and MPS allocator samples provide separate evidence. Open the trace JSON with Perfetto; no hosted upload was made.

The latest numerical outcomes and speed rejection decisions are authoritative in [loss qualification](reports/loss-qualification.json), [raw GPU results](reports/loss-gpu.json), [graph-rewrite results](reports/loss-fx-gpu.json), and [external conformance](reports/external-conformance-gpu.json). A passing fallback test establishes coverage, not cheaper arithmetic. These are adapters with bounded evidence, not full TorchCTS or BackendBench certification.

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

**Use without changing a model's mathematical definition.** Run from the project root in the new environment:

```python
import torch
from ojasx2.loss_fx import rewrite_linear_cross_entropy

graph = torch.fx.symbolic_trace(model)  # traceable ordinary PyTorch model
ledger = rewrite_linear_cross_entropy(graph, accuracy="algebraic")
loss = graph(inputs_on_mps, labels_on_mps)
loss.backward()  # original parameters and upstream activations remain
```

For explicit use, the primitive expects H[...,K], W[K,V] and integer targets with shape H.shape[:-1]:

```python
from ojasx2.loss_compiler import linear_cross_entropy
loss = linear_cross_entropy(h, weight.T, targets, bias,
                            accuracy="algebraic", schedule="auto",
                            chunk_rows=128, logit_budget_bytes=32*1024*1024)
```

Higher derivatives should be declared with `derivative_order=2` before forward; unexpected `create_graph=True` on the first-order replay route is rejected. Native fallback retains its original memory demand and is not guaranteed to fit the candidate's head budget. Frozen inputs/weights, bias-only training, noncontiguous views, weighted/smoothed losses, tied decoder weights, SGD/AdamW updates, mutation/version checks and nonfinite native routing have dedicated tests.

Recheck the current source evidence without numerical execution:

```sh
.tools/conformance-venv/bin/python -B tools/refine_loss_foundation.py
```

Add `--gpu` for the bounded MPS suites. Proof tools, metadata parsing and orchestration use the host. Numerical tensors run on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`; the conformance and correctness phases additionally reject CPU tensor execution through a dispatch guard. Timings exclude that diagnostic guard and use the same MPS-only numerical paths. No recurring/background automation was installed.

**200B/32GB resource contract.** `resident_state_ledger(200_000_000_000)` performs accounting only. At 16 bits, independent dense weights occupy **400 GB decimal** before activations. A stated layout with BF16 weights/gradients, two FP32 moments and an FP32 master copy totals **3.2 TB decimal**. These are explicit storage assumptions, not requirements of every optimizer. The new head schedules do not remove parameter state, and no 200B model was allocated or trained. A 32GB-device implementation must separately account for exact parameter representation and external storage/transfer or a different state layout. No 100× whole-model reduction or 200B-in-32GB result is claimed by this revision.

Prior art matters: [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) already addresses vocabulary-loss memory through tiled kernels, with optional gradient filtering. OjasX's current primitive is an exact-real row-selection/replay reference; it does not implement that kernel, approximate softmax gradients or establish superiority over CCE or an already preselected PyTorch head. The tool integration uses [official TorchCTS](https://github.com/TorchCTS/TorchCTS), [official BackendBench](https://github.com/meta-pytorch/BackendBench), and the [PyTorch cross-entropy contract](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html).

### Objective-aware compiler refinement, version 2.0

The new [objective compiler](ojasx2/objective_compiler.py) optimizes a complete scalar training objective and its requested gradients. This adds two exact-real rule families which can surround **ordinary native GELU, SiLU, convolution, normalization and attention computations**. It changes neither those activations nor the original trainable parameters.

The formal sources are [Lean ObjectiveCompiler](lean/OjasXMath/ObjectiveCompiler.lean) and [Rocq ObjectiveCompiler](rocq/ObjectiveCompiler.v): 23 new Lean theorems and 16 Rocq theorem/lemma declarations. These include actual finite-objective derivatives, the affine residual derivative, the joint quadratic dagger, a dimension-general Gram identity, streamed pullback sums, arithmetic crossover conditions and an explicit residual-storage model. No desired optimization property was inserted as an axiom.

**A. Quadratic output elimination.** Let H be the features produced by any differentiable network, W its final linear weight, and Y a fixed dense target. For the sum of squared errors, define G=HᵀH, C=HᵀY and S=WWᵀ. Then

$$
L=\|HW-Y\|_F^2=\langle G,S\rangle_F-2\langle W,C\rangle_F+\|Y\|_F^2,
$$

$$
\nabla_H L=2(HS-YW^T),\qquad \nabla_W L=2(GW-C).
$$

The mean loss divides all three expressions by the original number of target elements. H can contain the outputs of a deep nonlinear feature network: its gradient is propagated through the original graph. A trainable bias is included by appending a constant feature, with its original gradient recovered. The runtime planner charges this extra feature while retaining the original feature width in the native baseline.

For an unbiased m×k feature matrix and k×n weight matrix, the leading matrix arithmetic changes from 6mkn to 4mkn+4k²(m+n), when both feature and weight gradients are requested. It is cheaper exactly when 2k(m+n)<mn. Lean also proves at least a quarter reduction in this leading matrix count when m,n≥16k. This count includes both requested matrix gradients; scalar energy reductions, finite/cancellation guards, bias encoding, the feature network and optimizer are additional. A returned wide prediction is not part of this scalar-loss API. Dense targets and weight gradients still occupy memory.

The formulation is a classical Gram/trace identity, now integrated with the joint gradient contract. It is not a new universal multiplication algorithm. The runtime retains the ordinary route for target gradients, explicitly requested higher derivatives, unsuitable shapes and strict accuracy mode. Loss and each demanded gradient have cancellation checks; a rejected backward recomputes the original residual and pays the additional work. Finite inputs and a positive loss do not certify well-conditioned gradients. The tests preserve a concrete stationary-weight counterexample and check its repaired route. These checks are empirical safeguards, not complete IEEE error bounds.

**B. Separable objective streaming.** For fixed weights aᵢ, fixed positive denominator D, independent per-example losses ℓᵢ(θ) and one parameter state θ,

$$
L(\theta)=D^{-1}\sum_i a_i\ell_i(\theta),\qquad
\nabla L(\theta)=D^{-1}\sum_i a_i\nabla\ell_i(\theta).
$$

The compiler evaluates one chunk's forward and backward, releases that chunk's tape, and accumulates every original parameter gradient. It returns requested input gradients as well. The optimizer and gradient clipping belong **after the complete accumulation**. The same denominator is used for every chunk, including an uneven final chunk. This is a formalized implementation of gradient accumulation, not a newly discovered identity.

Under a simple schedule with P persistent storage and A retained activation bytes per independent example, the residual bound changes from P+BA to P+bA for batch B and chunk b≤B. This theorem models retained residuals; it does not prove total GPU peak. Full inputs/targets, parameter and optimizer state, gradient accumulators and transient workspaces must still be counted. With no skipped examples, arithmetic is largely retained and gradient accumulation adds work. It is a memory strategy.

Zero-weight examples can be eliminated before executing the model only when parameter usage is uniform and all per-example losses and derivatives are finite. This can remove their forward and backward arithmetic. The callback contract explicitly records these obligations; finite input tensors alone are insufficient because intermediate overflow can make 0×∞ a NaN. Skipping masked tokens inside attention is not implied: other tokens may depend on them. The supplied implementations are reference schedules, and small GPU operations/guards can outweigh the arithmetic savings.

**Scope and executable obligations.** `SeparableContract` is a caller declaration, not an automatically generated proof of a Python callback. The callback must return one FP32 MPS scalar loss per independent example and preserve parameter, buffer and random state. Batch-dependent normalization, contrastive losses coupling examples, fresh dropout, recurrent state crossing chunks and hidden mutation require additional treatment and are not admitted by this contract. Shared parameters are supplied once and receive all contributions. Frozen/unused operands, all-zero weights, sum and weighted-mean reductions have explicit tests. Input tensors are independent differentiation variables at this API boundary; returned gradients must be propagated by the caller if it placed additional computation before that boundary. First-order streaming does not construct a higher-order derivative graph.

The new bounded GPU suite covers six complete synthetic model objectives: GELU and SiLU MLPs, shared residual layers, a custom sine activation, a SiLU CNN, and a causal-attention block. It compares losses, all original parameter/input gradients, and three full SGD and AdamW updates including moment state. It also tests scaling, quadratic heads, gradient cancellation, target derivatives, higher derivatives through the native route, noncontiguous/frozen operands and rejection contracts. This is broader training-objective coverage; it is not a dataset-convergence study or a full large-model deployment.

The final new suite passes **34 bounded GPU cases**. The MLP scaling run retained 38,280 fresh saved bytes per chunk at batch sizes 128, 512 and 2,048; the full-batch baseline grew from 135,172 to 2,162,692 bytes. At batch 2,048 that is a **98.23% reduction in this retained-activation metric**. Across the three quadratic shapes, the actual bias-aware planner reports **18.41%, 27.40% and 29.07% less leading matrix arithmetic**. Measured fresh saved activation reductions are **81.39%, 94.35% and 89.18%**, respectively. These values exclude persistent inputs/targets, weights, gradients, optimizer state and unmeasured transient workspace. The added gradient-cancellation checks preserve accuracy in the stationary counterexample but add runtime overhead. New quadratic timing ratios were **0.22–0.39× native speed** in the timed shapes; zero-weight streaming ratios were **0.22–0.28×**. Both speed admissions are rejected.

**Admission is separated by outcome.** GPU correctness, retained memory, leading arithmetic and repeated speed are distinct fields in [the objective qualification ledger](reports/objective-qualification.json). The new reference schedules have not earned speed admission on this laptop; their slower timings remain visible. Earlier speed-qualified linear/radial/polynomial families retain their own scopes. No speed result is inferred from the new operation-count theorem.

Use the two explicit APIs:

~~~python
from ojasx2.objective_compiler import (
    quadratic_loss, compile_separable_objective, SeparableContract,
)

# Original nonlinear feature graph; original h, weight and bias get gradients.
loss = quadratic_loss(h, weight, target, bias, accuracy="algebraic")
loss.backward()

# After auditing a deterministic per-example callback:
contract = SeparableContract(
    independent_examples=True, deterministic=True, no_mutation=True,
    justification="Describe the audited graph and its effect/domain conditions",
)
program = compile_separable_objective(per_example_loss, contract=contract, chunk_size=32)
result = program.value_and_grad(tuple(model.parameters()), (inputs, targets))
for parameter, gradient in zip(model.parameters(), result.parameter_grads):
    parameter.grad = gradient
optimizer.step()  # Once after the complete objective; do not step per chunk.
~~~

Zero-weight elimination requires the additional contract fields `uniform_parameter_usage=True` and `finite_per_example_loss_and_gradients=True`; leave them false unless discharged. These APIs are not a complete transparent `torch.compile` implementation.

The bounded refinement command reruns discovery, compiles/audits the entire Lean/Rocq project, independently rechecks Rocq and validates source-matched GPU evidence:

~~~sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/refine_objective_foundation.py
~~~

Add `--gpu` to refresh GPU results. Numeric ML executes on MPS with CPU fallback disabled; proof, symbolic search and orchestration are host tasks. The suite is sequential, includes pauses and installs no background loop. [Discovery](reports/objective-discovery.json) records 24 Z3 checks/counterexamples and 27 finite shape decisions, including failures of local-mean averaging, early optimizer updates, batch centering, variable-weight omission, omitted target terms, unconditional Gram admission and unsafe numerical assumptions. These finite searches do not establish globally optimal costs.

Related primary background: [PyTorch functional module execution](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html) describes an interface for later module integration; the present callback interface is explicit. [FlashAttention](https://arxiv.org/abs/2205.14135) is relevant existing work on exact attention memory/IO savings. This addition does not reimplement FlashAttention or claim its arithmetic/latency properties.

### Numerical reliability refinement, version 1.9

Implemented in [numerical_repairs.py](ojasx2/numerical_repairs.py). For a row u, v=u−mean(u), variance=mean(v²), scale a and D=a² variance+epsilon, the normalized output is a v/sqrt(D). Its scale derivative has coefficient

$$
\frac{1}{\sqrt D}-\frac{a^2\operatorname{variance}}{D^{3/2}}
=\frac{\epsilon}{D^{3/2}}
=\frac{\epsilon/D}{\sqrt D}.
$$

The last form removes near-equal subtraction and avoids computing an overflowing inverse cube first. Lean proves the actual scalar derivative for variance≥0 and epsilon>0, including a=0; Rocq checks the corresponding cotangent algebra. The GPU vector implementation and original-parameter/higher-order behavior are empirically checked; a complete IEEE implementation certificate is not claimed.

| Identified issue | Implemented reliability treatment |
|---|---|
| Degree-4 native approximation rejection | Explicit approximation requests can select the existing qualified degree-8 family, subject to domain and whole-program qualification |
| Radial–LayerNorm cancellation | Stable joint scale/normalization derivative and overflow-avoiding evaluation implemented |
| Two-input SiLU speed rejection | Conservative adapter retains native execution; no new speed repair asserted |
| Wide polynomial feature explosion | Reject feature/cost-ineligible expansion before building the basis and retain native |
| Higher-order derivative behavior | Stable normalization retains original differentiable operands; conservative projection adapter uses the original native function when higher order is requested |
| Remaining near-zero numerical disagreement | Checked scalar-loss/all-original-gradient API validates each invocation and returns native results when the original criteria fail |

Seven normalization/overflow cases, three previously sensitive composed models, five routing cases, three invalid-epsilon cases and three explicit candidate-rejection cases pass the scoped GPU repair checks. The composed models also undergo three native-compared SGD steps. Two raw relative normalization comparisons and all three raw algebraic normalized models remain rejected; their tolerances were not relaxed. They pass through correct native routing, not by pretending the optimized candidate passed.

The checked API evaluates both callbacks and is intentionally costly. Callbacks must be pure/deterministic and scalar-valued; random masks must be fixed and mutation excluded. It does not certify arbitrary optimizers, global error, every graph or speed. The whole-program-qualified argument is a caller obligation, not an automatic certificate.

Reports: [GPU checks and raw rejections](reports/numerical-repairs-gpu.json), [combined proof/Z3/hash verification](reports/numerical-repairs-verification.json). Recheck with:

~~~sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_numerical_repairs.py
~~~


## 1. Objective and current scope

The product objective is to train the developer's original model with less arithmetic, less retained activation information and less total GPU memory, while preserving its parameterization and required accuracy. Moving work into a different representation should pay for encoding, decoding, derivatives and original gradient recovery.

| Delivered now | Remaining objective |
|---|---|
| Kernel-checked real/rational mathematical contracts | Formal refinement of deployed Python/Metal/GPU arithmetic |
| All-original-gradient rules for selected linear and nonlinear representations | Cheap representations for ordinary wide nonlinear networks |
| Residual sufficiency, composition, accumulation and conditional costs | A global minimum-memory or globally optimal arithmetic planner |
| Scoped MPS FP32 correctness and speed/memory evidence | Most-model quality, speed and total peak-memory qualification |
| Explicit research APIs and limited earlier graph adapters | Transparent production integration across general PyTorch graphs and GPU vendors |

The current whole-project formal audit contains **492 Lean theorems and 247 Rocq theorem/lemma declarations**. The spectral discovery retains **32 Z3 checks and 28 shape/degree decisions**; version 2.1 adds the loss-directed checks listed above. These counts measure audited work, not a guarantee of workload coverage.

Numerical ML in the current research loop is GPU-only MPS with CPU fallback disabled. Symbolic algebra and proof checking run on the host. Historical CPU evidence is identified in the companion record. Neural Engine training and cross-vendor production execution are unqualified.

## 2. The broad mathematical engine

Let E contain every differentiable input and original parameter, and F contain the requested outputs. For a differentiable primal f : E → F, reverse mode at x with output cotangent g is

$$
\operatorname{back}_f(x,g)=Df(x)^*g,
\qquad
\langle Df(x)v,g\rangle=\langle v,\operatorname{back}_f(x,g)\rangle.
$$

The implemented abstract proof works in real complete inner-product spaces using continuous linear derivatives. Finite tensors are concrete instances; a product input space includes weights, biases, gain parameters and other differentiable operands.

An abstract stage carries:

- The original primal function f.
- A derivative Df with an actual HasFDerivAt proof.
- A residual encoder R and a map J with J(R(x)) = Df(x).
- The reverse operation J(R(x))*g.

For stages f then h:

$$
D(h\circ f)(x)=Dh(f(x))\,Df(x),
\qquad
\operatorname{back}_{h\circ f}(x,g)
=\operatorname{back}_f(x,\operatorname{back}_h(f(x),g)).
$$

Cotangents from branches add. Shared parameters accumulate contributions from every use. A finite composition is correct when its leaves satisfy the contracts and its effects are represented faithfully. This does not prove a derivative for a new leaf automatically.

If two differentiable primals are identical on the needed neighborhood, derivative uniqueness gives identical pullbacks for all requested operands. A rewrite valid only at isolated points does not suffice. At nondifferentiable points, PyTorch's selected backward convention needs a separate contract; it is not a classical derivative proof.

Sources: [CertifiedBackprop](lean/OjasXMath/CertifiedBackprop.lean), [AbstractEngine](lean/OjasXMath/AbstractEngine.lean), [Rocq Contracts](rocq/Contracts.v), [Composition](lean/OjasXMath/Composition.lean).

### Residual sufficiency

For a proposed residual R, exact first-order sufficiency means

$$
R(x)=R(y)\Longrightarrow
\operatorname{back}_f(x,g)=\operatorname{back}_f(y,g)
\quad\text{for every }g.
$$

The proof establishes existence of a decoder under this condition. Efficient computability and small storage still require a construction. Higher derivatives additionally require the proper dependence of the residual on x; detached sufficient first-order statistics do not automatically produce correct Hessians.

Source: [Residuals](lean/OjasXMath/Residuals.lean).

## 3. A single admission contract for every optimization

The following is the canonical specification for a rule. It combines obligations already represented in different proof/runtime modules. There is no fully implemented universal rule registry containing every field below.

| Contract field | Required evidence |
|---|---|
| Original primal and operands | Exact operator semantics, all original parameter objects, demanded outputs and cotangents |
| Domain | Shapes, index ranges, dtype/device, finite-value assumptions, epsilon, layout and branch conditions |
| Representation | Encoder, decoder, structured coefficients and, for sequences, closure/domain preservation |
| Derivative | Actual derivative or explicit selected-backward convention; encoder/decoder and all parameter dependencies included |
| Residual | Enough information for every demanded cotangent; separate first-order/higher-order policy |
| Numerical mode | Strict, real-algebraic with numerical qualification, or explicitly approximate |
| Error | Primal, every original gradient, loss-cotangent transport, update/state and conditioning requirements |
| Effects | Aliases, mutation/versioning, random state, shared parameters, branch accumulation and update barriers |
| Arithmetic | Encoding, forward, reverse, original-gradient recovery, guards, recomputation, loss and optimizer costs |
| Storage | Retained residuals, dense outputs, parameters/state, transient workspace, aliases and live peak |
| Qualification | Proof identifiers, source hashes, raw failures, GPU/device evidence and repeated timing scope |
| Rejection | Correct native or same-surrogate fallback; explicit higher-order and boundary behavior |

A proof assumption is a condition to discharge. It must not assert the desired low rank, accuracy or speed without evidence.

The acceptance pipeline is:

~~~mermaid
flowchart LR
  A["Original joint graph"] --> B["Candidate representation"]
  B --> C["Derivative and residual proofs"]
  C --> D["Domain, error and cost checks"]
  D --> E["GPU correctness and full-step timing"]
  E --> F["Scoped admission"]
  B --> R["Record counterexample and refine"]
  C --> R
  D --> R
  E --> R
~~~

The diagram is the specification. Existing tools implement bounded versions for selected rule families, not general autonomous theorem invention.

## 4. Numerical modes and error transport

| Mode | Meaning | Present policy |
|---|---|---|
| Strict | Preserve the original selected computation | Default native path for the newest experimental APIs |
| Algebraic | Same real primal and gradients through reordered operations | Explicit opt-in, numerical/domain guards and scoped GPU qualification |
| Approximate | A different primal with its own matched derivative | Explicit opt-in, bounded domain, model/optimizer preflight required |

Real equality is not bitwise floating equality. Guards are conservative research safeguards, not complete IEEE certificates. Narrow historical bit-equality tests do not qualify arbitrary reassociation.

For a coordinate approximation p to φ on its proven domain, require separate value and slope bounds

$$
|p(z)-\phi(z)|\le\epsilon_0,\qquad
|p'(z)-\phi'(z)|\le\epsilon_1.
$$

For fixed original operands and fixed upstream G, an input-gradient component satisfies

$$
|\bar X^{p}_{ia}-\bar X^{\phi}_{ia}|
\le\epsilon_1\sum_j|G_{ij}W_{aj}|.
$$

Weight and bias components have corresponding bounds with X or 1 in the sum. Projection gradients use the value error ε₀. These are absolute bounds; cancellation can make a small absolute difference a large relative error.

If the changed primal changes the loss cotangent, the slope-product error has an additional term:

$$
|p'(z)\widehat G-\phi'(z)G|
\le\epsilon_1|G|+|p'(z)|\,|\widehat G-G|.
$$

Changes in later forward states and parameter trajectories require further propagation. A local scalar approximation bound does not certify a whole training run.

For plain SGD with the same parameter and step size, update error is at most |η| times gradient error. Exact equal gradients preserve any deterministic update fed identical state. Approximate gradients need optimizer-specific qualification; AdamW's behavior near small moments cannot be inferred from the SGD bound.

Sources: [AbstractEngine error/update statements](lean/OjasXMath/AbstractEngine.lean), [SpectralFoundation](lean/OjasXMath/SpectralFoundation.lean), [Optimizer.v](rocq/Optimizer.v).

## 5. The joint block that identifies the main bottleneck

Use dimensions X ∈ ℝ^(m×k), W ∈ ℝ^(k×N), b ∈ ℝ^N and Q ∈ ℝ^(N×r). For

$$
Z=XW+b,\qquad Y=\phi(Z),\qquad H=YQ,
$$

and incoming P ∈ ℝ^(m×r):

$$
G=PQ^T,\quad \Delta=\phi'(Z)\odot G,\quad
\bar X=\Delta W^T,\quad
\bar W=X^T\Delta,\quad
\bar b=\sum_i\Delta_i,\quad
\bar Q=Y^TP.
$$

The expensive wide objects are Z, Y, G and Δ. A joint rule should compute H and every requested pullback without constructing them when a cheaper representation exists.

The leading conventional forward/all-gradient matrix arithmetic is approximately 6mN(k+r), before pointwise work, loss and updates. This is a multiply/add model, not a hardware instruction count.

G is factored by P and Q, but an arbitrary elementwise φ'(Z) can destroy that small representation. This is the central wide nonlinear obstacle. Nondense structure must be justified at the derivative state actually reached.

## 6. Current exact-real optimization families

### Linear projection

For φ(z)=z, form C=WQ and B=XᵀP, and t=ΣᵢPᵢ:

$$
H=XC+bQ,\quad
\bar X=PC^T,\quad
\bar W=BQ^T,\quad
\bar b=tQ^T,\quad
\bar Q=W^TB+b^Tt.
$$

The leading joint arithmetic becomes 6kr(m+N), plus bias/pointwise work. Savings depend on dimensions and requested gradients. The original W/b/Q are retained and C is refreshed after updates.

Source: [gradient_factor_compiler.py](ojasx2/gradient_factor_compiler.py). Repeated full-step GPU gains exist for [8192,128,4096,2]; small cases are slower. Longer-chain planning has separate backward recovery costs.

### Exact derivative factors: square and certified masks

If G=UVᵀ and a slope matrix S=ABᵀ, then

$$
G\odot S=(U\star A)(V\star B)^T,
$$

where ⋆ is the row-wise product over all pairs of feature indices. Width can grow from s and t to st; no exact truncation to the old width is implied.

For square, φ'(Z)=2Z has width k+1 including bias; the combined slope/cotangent representation has width r(k+1). The present implementation still materializes a wide forward activation and retains it for the Q-gradient, so it has a smaller memory improvement than the fully lifted spectral family.

ReLU supports a batch-uniform per-feature mask or a supplied verified chamber grouping. Arbitrary masks need their own representation. Current certified mask routes have not passed a reliable repeated speed gain.

Sources: [GradientFactors](lean/OjasXMath/GradientFactors.lean), [GradientFactors.v](rocq/GradientFactors.v), [chamber compiler](ojasx2/chamber_compiler.py).

### Existing radial activation with a small invariant core

For φ(z)=f(mean(z²))z, let A=[X,1], V=[W;b], d=k+1, C=VVᵀ and T=VQ. Evaluate

$$
AC=A C,\quad s_i=(AC)_i\cdot A_i/N,\quad u=AT,\quad H_i=f(s_i)u_i.
$$

For upstream P define Rᵢ=f(sᵢ)Pᵢ and eᵢ=f'(sᵢ)(Pᵢ·uᵢ)/N. Then

$$
B=A^TR,\quad D=A^T\operatorname{diag}(e)A,
$$

$$
\bar A=RT^T+2\operatorname{diag}(e)AC,\quad
\bar V=BQ^T+2DV,\quad
\bar Q=V^TB.
$$

The first k columns/rows recover X/W gradients; the final V row recovers b. For a trainable gate, also differentiate f with respect to its original parameters.

The real identity holds for gates with the stated derivative contract. Runtime examples are cubic f(s)=s, reciprocal 1/(1+s), and a trainable sigmoid gate. Replacing an ordinary GELU with a radial gate would change its model and is outside this exact rewrite.

The current conservative leading model compares 6mN(d+r) with 6d(d+r)(m+N)+12md+12mr. Actual comparisons must also include gates, guards, allocations, loss, updates and hardware behavior. The main sufficient shape advantage is d(m+N) substantially below mN.

The runtime tests finite/magnitude/Gram conditioning and retains native calculation on rejected cases. Its higher-order path recomputes the original radial block. Sensitive normalized compositions remain rejected.

Sources: [InvariantRadial](lean/OjasXMath/InvariantRadial.lean), [InvariantRadial.v](rocq/InvariantRadial.v), [invariant_radial_compiler.py](ojasx2/invariant_radial_compiler.py).

### Finite closed representations across depth

An encoding E, decoding D and parameterized operator K qualify when

$$
E(f_\theta(x))=K_\theta E(x),\qquad D(E(x))=x
$$

on a preserved domain. Ordered composition then gives

$$
f_{\theta_{L-1}}\circ\cdots\circ f_{\theta_0}(x)
=D(K_{\theta_{L-1}}\cdots K_{\theta_0}E(x)).
$$

Differentiate the encoder, every original K dependency, and the decoder. Constant representation width is crucial; closure alone with an enormous width gives no saving.

The current successful example is the triangular recurrence (ax,by+cx²) with E=(x,y,x²). Tests cover shared or separate original stage parameters and selected higher derivatives. Ordinary dense activations do not inherit this finite closure.

Source: [coordinate_compiler.py](ojasx2/coordinate_compiler.py), [CoordinateCompiler](lean/OjasXMath/CoordinateCompiler.lean).

### Other conditional tools

Reversible coupling and cached preactivation schedules can trade residual storage against replay. Normalization can use sufficient normalized-value residuals with epsilon correction. Gather/scatter uses exact repeated-index accumulation. Rank-seven GEMM candidates have known product/addition tradeoffs and no current admitted GPU replacement. Sphere/Stiefel projections are tools for explicitly constrained models.

These are separate rules. Their speed ratios cannot be multiplied together without an actual jointly qualified graph.

## 7. Current coordinate nonlinear approximation family

Choose one polynomial p(z)=Σₙ₌₀ᴰ cₙzⁿ. For a multi-index α with s=|α|≤D:

$$
\Psi_{i\alpha}=X_i^\alpha,\qquad
T_s(b)=\sum_{n=s}^D c_n{n\choose s}b^{n-s},
$$

$$
\Theta_{j\alpha}=
\frac{s!}{\prod_a\alpha_a!}
\left(\prod_a W_{aj}^{\alpha_a}\right)T_s(b_j).
$$

Then p(XW+b)=ΨΘᵀ and Hₚ=ΨC with C=ΘᵀQ. For incoming P:

$$
\bar C=\Psi^TP,\quad \bar\Psi=PC^T,\quad
\bar\Theta=Q\bar C^T,\quad \bar Q=\Theta\bar C.
$$

Zero-safe monomial differentiation recovers X/W gradients. Shared translated coefficients satisfy

$$
T_s'(b)=(s+1)T_{s+1}(b),\qquad T_D'(b)=0,
$$

recovering original b gradients without dividing by X or W.

This is an exact identity for the selected polynomial. It is an approximation when the original activation is GELU or SiLU. Forward and backward must use the same polynomial.

### Domain, feature growth and evidence

The latest degree-8 family uses a truncated Chebyshev derivative of a degree-12 reference, integrated into a matched primal. Its formal tail bounds compare polynomials; true-activation remainder and GPU floating error remain separate formal gaps.

The feature count is

$$
S=\binom{k+D}{D}.
$$

Degree eight has S=9,45,165,495 for k=1,2,3,4. This growth prevents extrapolation to a usual wide dense layer. An exact low-rank factorization of X could in principle lower an effective width, but discovering it, differentiating its dependencies, handling degeneracy and paying its costs are not implemented or qualified.

The current runtime grammar permits k=1–4 and S≤512; the strongest measured cases use k=1 or 2. Eligible execution also requires MPS FP32, no autocast, finite bounded values and a profitable leading-cost model. For |z|≤1 it uses the conservative box check

$$
\max_i|Z_{ij}|
\le\sum_a(\max_i|X_{ia}|)|W_{aj}|+|b_j|.
$$

Current bounds include |X/W/b|≤1 and |Q|≤8. No Gaussianity is inferred from normalization. No clipping manufactures an accuracy pass.

Strict mode uses native GELU/SiLU. Approximate mode admits the bounded research degree-8/12 families; polynomial mode examines polynomial candidates subject to guards. Domain or cost rejection returns native execution. Consequently the guarded API is not one global fixed polynomial, and branch-boundary derivative behavior is not universally proved.

Higher-order or unsafe-cotangent reverse execution recomputes the **same polynomial** on an admitted branch. It does not replace that derivative with a native GELU Hessian, but its wide recomputation can remove first-order memory/work savings.

Source: [spectral_compiler.py](ojasx2/spectral_compiler.py), [coefficients](ojasx2/spectral_coefficients.py), [SpectralFoundation](lean/OjasXMath/SpectralFoundation.lean), [SpectralChebyshev](lean/OjasXMath/SpectralChebyshev.lean), [PolynomialContract](rocq/SpectralFoundation.v).

## 8. Workload coverage and extension ledger

This is a map of major present-day workload families and extension obligations. It is not an exhaustive proof of every model or PyTorch operator. A future model is covered semantically only after its primitives, domains and effects satisfy the contracts. The integration policy below describes how a backend should treat unsupported cases; the general automatic backend is still incomplete.

| Workload/operator family | Foundation support and actual evidence | Cheaper execution established? | Missing condition or integration policy |
|---|---|---|---|
| Dense affine layers and linear chains | Matrix derivatives, composition, original gradients; historical and MPS examples | Yes for favorable chains/projection shapes | Joint backward cost, numerical range, every weight gradient; retain native elsewhere |
| Ordinary coordinate GELU/SiLU MLPs | Native derivative compatibility historically; matched polynomial GPU research | Conditional approximations for narrow inputs; no broad exact reduction | Wide feature growth, uniform analytic remainder, whole-model error and quality |
| Square/polynomial research layers | Tensor expansions and zero-safe joint factors; GPU cases | Yes for selected square/narrow-feature blocks | Degree/width/cost; retain every original operand |
| ReLU networks | Selected derivative masks and residual information; certified mask/chamber examples | Memory policies in selected graphs; no robust new mask speed win | Heterogeneous mask complexity, zero convention and alias accounting |
| SwiGLU and gated MLPs | Product rules, native compatibility, small-domain surrogate research | No reliable admitted broad nonlinear speed improvement | Two branch values/derivatives, feature interaction growth and original gate/projection gradients |
| Radial/RMS-style projected research blocks | Exact-real invariant core and MPS full-step qualification | Yes for eligible existing radial blocks | Gram conditioning, original gate/affine parameters and composition sensitivity |
| LayerNorm/RMSNorm | Structured pullback and sufficient normalized-value residuals; CPU edges and scoped GPU compositions | Selected core paths; no blanket standalone/new full-model gain | Epsilon, affine gradients, reduction precision, cancellation, axis and optimizer tests |
| BatchNorm and running statistics | Historical model tests and AdamW negative result | No general new qualification | Training/eval modes, running state, batch dependence; sensitive cases native |
| Softmax/cross-entropy | Structured diagonal-minus-rank-one contractions; historical compatibility, narrow GPU losses | Avoiding explicit Jacobians is established baseline behavior; no broad new gain | Stable reductions, masks, empty rows, underflow, loss cotangent changes |
| Attention/Transformer blocks | Matmul/softmax/composition ingredients and historical model examples | No qualified general cheaper attention or ordinary Transformer training | Sequence quadratic structure, masks, dropout, optimized baseline and memory/workspace |
| Convolutions/CNNs | Bilinear/repeated composition ingredients; historical derivative tests | No dedicated new GPU convolution algorithm | Spatial indexing, groups, stride/padding/dilation, kernel parameters and native optimized baseline |
| RNNs/unrolled shared-weight programs | Repetition and cotangent accumulation; historical recurrence tests | No general nonlinear recurrent reduction | Shared-gradient accumulation, state/residual cost and long-depth numerical behavior |
| Triangular Koopman recurrence | Closed observable proofs and MPS original-stage-gradient tests | Yes for qualified 32/64-stage cases | Must satisfy actual closure and numerical guards; no arbitrary recurrence substitution |
| Embeddings/gather/scatter | Finite in-range indexing and repeated-index adjoint; exact rational/CPU examples | Correct accumulation, not a universal arithmetic reduction | Duplicate/empty indices, sparse gradient formats, bounds and GPU implementation |
| Residual branches/shared parameters | Composition and cotangent addition; representative graph tests | Only gains of individually eligible combined subgraphs | Join accumulation, alias/version lifetimes and original shared parameters |
| Dropout/stochastic layers | Fixed-mask linear pullback examples | No blanket stochastic memory/speed claim | RNG state, exact mask replay and distribution semantics |
| Reversible architectures | Coupling and cached reverse rules; scoped CPU/MPS/Metal evidence | Conditional memory saving and modest cached speed gain | Existing reversible model, reconstruction error, replay cost and higher-order behavior |
| Manifold-constrained research | Sphere/Stiefel projection identities | No general ML throughput/convergence qualification | Chosen metric, constraint, retraction/update and every manifold parameter |
| Scientific ML/unrolled ODEs | Historical derivative compatibility; abstract composition applies conditionally | No qualified general solver-training compression | Solver steps/events/tolerances, adjoint choice, custom derivatives and state |
| Custom/new differentiable primitives | Abstract engine is extensible | No saving until a rule is constructed | Supply derivative, residual, domain/effects, error/cost proofs and GPU tests |
| Higher-order/meta-learning workloads | Some selected HVP checks; differentiable recomputation policies | First-order savings may disappear | Residual dependence, nesting, derivative order and optimizer differentiation |
| Dynamic shapes/control flow/mutation | Mathematical contracts can state cases; no comprehensive captured executor | No global qualification | Runtime domains, selected branches, alias versions and fallback boundaries |
| Sparse/ragged/quantized/complex/distributed training | Outside the newest FP32 real dense runtime qualification | Not established | Dedicated semantics, conjugate adjoints where needed, discrete conventions, communications/state |

Compatibility, formal semantic ingredients, admitted arithmetic savings and GPU speed are different columns. They must remain separate when expanding this ledger.

## 9. Arithmetic, memory and scaling requirements

A candidate cost must include:

$$
C_{\rm step}
=C_{\rm encode}+C_{\rm forward}+C_{\rm loss}
+C_{\rm reverse}+C_{\rm original\ gradients}
+C_{\rm guards/replay}+C_{\rm optimizer}.
$$

Current models are leading scalar arithmetic or scoped counts. Copy traffic, launches, synchronization, transcendental instructions and real GPU tile behavior require separate measurement. Common optimizer work cancels in an algebraic comparison only when both paths use the same update schedule and state.

For dense gradients returned to a conventional optimizer, their output elements still need storage and production. Factorizing an intermediate does not factorize those outputs automatically. A low-rank parameterization changes which variables are trained and needs explicit model consent.

Define peak memory at each instant with aliases counted once:

$$
M_{\rm peak}=\max_t
\big(M_{\rm weights}(t)+M_{\rm gradients}(t)+M_{\rm optimizer}(t)
+M_{\rm residuals}(t)+M_{\rm temporaries}(t)+M_{\rm workspace}(t)
+M_{\rm allocator}(t)\big).
$$

Sum live quantities at the same instant; do not sum independently measured component maxima. Plain FP32 Adam with weights, gradients and two moment arrays is approximately 16 bytes per parameter before other storage, depending on implementation and precision. Large activation reductions can leave this persistent storage dominant.

The recorded 96–99.67% savings concern fresh retained activation storage, with exclusions stated per report. They do not establish the same percentage reduction in peak GPU RAM.

For a graph where a fraction f of original step time is improved by factor s, the unchanged remainder limits total speed to approximately 1/((1-f)+f/s), before new overheads. Local speed ratios cannot be multiplied to predict whole-model throughput.

The one-GPU capacity/throughput goal requires measured complete steps with the same model, trainable parameters, batch/context, data, precision, optimizer and quality targets. No present result certifies that goal for ordinary large models.

## 10. Current measured admission summary

All rows are scoped recorded observations, principally against eager native execution. They do not compare every optimized compiler or vendor kernel.

| Family and representative case | Recorded result | Admission boundary |
|---|---|---|
| Linear [8192,128,4096,2] | 2.23–2.62× full-step; 134,217,728 → 1,024 fresh saved bytes | Original gradients retained; small shapes slower |
| Square, same shape | 1.36–1.61×; 50% fewer fresh saved bytes | Wide forward/Q-gradient still present |
| Radial, same shape | 4.62–6.09× across three gate families; about 96% less fresh storage | Existing radial semantics; sensitive normalized compositions rejected |
| Two-block radial research classifier | 5.87–6.49× full-step; 98.52% less fresh storage | Narrow supplied model; memory hook excluded loss tape |
| Koopman distinct weights, 64 stages [512,32,2] | 5.20–6.07×; 97.25% less fresh storage | Triangular closed recurrence only |
| Degree-8 GELU [8192,2,4096,2] | 1.19–1.50×; 98.52% less fresh storage | Approximate, bounded, two-input case |
| Degree-8 SiLU, same shape | 0.92–1.34×; 98.52% less fresh storage | Repeated speed admission failed |
| Degree-8 GELU/SiLU [8192,1,8192,2] | 2.81–3.43× / 1.69–3.54×; 99.67% less fresh storage | Approximate, bounded, one-input cases |
| Four composed narrow GELU/SiLU models | All-gradient, four-step SGD/AdamW comparisons passed | No complete-model speed, long-term convergence or quality claim |

Latest exact byte counts, seed ratios and raw errors are in [spectral qualification](reports/spectral-qualification.json). Earlier family evidence is in [gradient qualification](reports/gradient-qualification.json), [invariant qualification](reports/invariant-qualification.json) and [coordinate qualification](reports/coordinate-qualification.json). Earlier hash bundles are historical snapshots; the latest whole-project formal audit is authoritative for current proof sources.

## 11. Source organization

| Responsibility | Main files |
|---|---|
| Real differentiation and adjoint composition | [CertifiedBackprop](lean/OjasXMath/CertifiedBackprop.lean), [AbstractEngine](lean/OjasXMath/AbstractEngine.lean), [Composition](lean/OjasXMath/Composition.lean) |
| Residual information and multilinear/geometry rules | [Residuals](lean/OjasXMath/Residuals.lean), [Multilinear](lean/OjasXMath/Multilinear.lean), [Geometry](lean/OjasXMath/Geometry.lean) |
| Matrix/rational contracts, sufficient normalization and indexing | [Matrix.v](rocq/Matrix.v), [Contracts.v](rocq/Contracts.v), [Structured.v](rocq/Structured.v), [Compact.v](rocq/Compact.v), [Indexed.v](rocq/Indexed.v) |
| Cost/optimizer and known GEMM candidates | [Planner.v](rocq/Planner.v), [Optimizer.v](rocq/Optimizer.v), [RankSeven.v](rocq/RankSeven.v), [Contraction](lean/OjasXMath/Contraction.lean) |
| Exact factor, invariant and representation rules | [GradientFactors](lean/OjasXMath/GradientFactors.lean), [InvariantRadial](lean/OjasXMath/InvariantRadial.lean), [CoordinateCompiler](lean/OjasXMath/CoordinateCompiler.lean) |
| Polynomial/error and generated identities | [SpectralFoundation](lean/OjasXMath/SpectralFoundation.lean), [SpectralDiscovered](lean/OjasXMath/SpectralDiscovered.lean), [SpectralChebyshev](lean/OjasXMath/SpectralChebyshev.lean), [SpectralFoundation.v](rocq/SpectralFoundation.v), [SpectralDiscovered.v](rocq/SpectralDiscovered.v) |
| Current GPU reference implementations | [gradient_factor_compiler.py](ojasx2/gradient_factor_compiler.py), [invariant_radial_compiler.py](ojasx2/invariant_radial_compiler.py), [coordinate_compiler.py](ojasx2/coordinate_compiler.py), [spectral_compiler.py](ojasx2/spectral_compiler.py) |
| Discovery, formal audit, GPU execution, admission | [discover_spectral_foundation.py](tools/discover_spectral_foundation.py), [verify_spectral_foundation.py](tools/verify_spectral_foundation.py), [test_spectral_gpu.py](tools/test_spectral_gpu.py), [qualify_spectral_foundation.py](tools/qualify_spectral_foundation.py), [refine_spectral_foundation.py](tools/refine_spectral_foundation.py) |
| Machine-readable current specification | [foundation_manifest.json](foundation_manifest.json) |

Lean/Mathlib are pinned by [lean-toolchain](lean/lean-toolchain) and [lake-manifest.json](lean/lake-manifest.json). Rocq installation and extraction details are in [rocq/README.md](rocq/README.md).

Proof sources are the authority for theorems; generated discovery reports and GPU records qualify their respective scopes. Source hashes bind observations to files and detect staleness. They do not prove semantics of Python or generated machine instructions.

## 12. Reproducible discovery and refinement

The tools do bounded candidate search, coefficient synthesis, negated-identity checks and proof generation. Counterexamples refine a representation or reject it; no desired improvement is installed as an axiom.

Run the current formal-only rebuild and validate existing GPU evidence:

~~~sh
cd /Users/mohana/Documents/Codex/2026-10-06/v-2/outputs/OjasX_rebuilt
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/refine_spectral_foundation.py --qualify-existing-gpu
~~~

For a fresh numerical GPU run after a relevant change:

~~~sh
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/refine_spectral_foundation.py --gpu
~~~

The GPU command must see a usable Metal/MPS device and fails instead of running numerical CPU fallback. Run benchmarks after formal compilation finishes and sequentially with other numerical benchmarks stopped. The runner installs no recurring/background automation.

Admission requires proof checks, matching source hashes, sampled polynomial/all-gradient accuracy, applicable model/update checks and policy correctness. Repeated speed admission requires three distinct seeds with each recorded ratio above 1.05. Memory can pass when speed fails. Raw candidate rejections remain in the reports.

The current suite is finite and deliberately scoped: four degree-4 native approximations are rejected; higher derivatives are compared with the selected polynomial. Older general verification commands execute historical CPU ML suites and are outside the current GPU-only numerical workflow.

### Existing explicit APIs

The following uses the original GPU parameter tensors; it is an integration example, not a new benchmark or a general graph compiler:

~~~python
from ojasx2.gradient_factor_compiler import projection_block
from ojasx2.invariant_radial_compiler import invariant_radial
from ojasx2.spectral_compiler import spectral_projection

# x, W, bias and Q must be compatible original FP32 MPS tensors.
# Choose a path only after its whole-program/domain qualification.
linear_output = projection_block(x, W, bias, Q,
                                 kind="identity", accuracy="algebraic")
radial_output = invariant_radial(x, W, bias, Q,
                                 kind="reciprocal", accuracy="algebraic")
approximate_output = spectral_projection(x, W, bias, Q,
                                        kind="gelu", degree=8,
                                        accuracy="approximate")
~~~

These are alternative examples with different primal semantics. A radial output must match an existing radial model. Approximate output requires accepting the bounded surrogate. Loss/backward/optimizer use the original tensors; metadata and derived coefficients must not become stale across updates.

## 13. Backend contract and research-model extension

A production backend should capture the joint graph, keep original parameters, select only discharged mathematical rules, plan live residuals/workspace, and lower the resulting contractions and feature construction to target GPU tiles. The planned user experience is ordinary model/loss/optimizer code, but a general portable implementation has not been delivered.

Hardware-specific lowering must preserve reduction precision, index bounds, alias/version behavior, barriers and update order. Updating a parameter early is valid only after every consumer of its old value and required accumulation has finished. Gradient clipping, shared uses and optimizer state create additional barriers.

Target-independent algebra does not certify a Metal, CUDA, AMD or Ascend kernel. Each backend needs separate execution, error and peak-memory qualification. GEMM itself is arithmetic performed by GPU units; removing intermediate memory traffic and reducing the mathematical contractions are distinct optimizations to measure.

For a research-defined operation:

1. State the primal, all trainable operands, domains and discrete conventions.
2. Prove its actual derivative or selected-backward rule and branch accumulation.
3. Construct sufficient residuals or a domain-preserving closed representation.
4. Include encoder/decoder and original parameter dependencies in reverse mode.
5. Prove or bound width growth, construction/recovery work and residual requirements.
6. Supply floating and whole-program error conditions, including optimizer/higher-order behavior.
7. Test original outputs and all gradients on GPU, then complete-step timing and live peak memory.
8. Admit only the supported region and preserve rejection evidence.

A new primitive can extend the framework without assuming all future primitives are already proved or efficient.

## 14. The remaining mathematics to pursue

The largest unresolved requirement is **wide coordinate nonlinear compression that remains cheaper after all original gradient recovery**. The current polynomial width is combinatorial; radial invariants change ordinary coordinate activations; arbitrary masks and derivative states do not automatically remain low rank.

Priorities are a representation with slower width growth or certified exploitable structure, uniform activation-and-derivative remainder bounds with realistic domains, qualification of the new stable joint normalization near cancellation, and whole-model optimizer error transport. These are research obligations; this document does not claim their solutions.

The foundation is ready as a precise reference and extension framework. Its current useful algorithms should be evaluated inside realistic optimized baseline models before claiming most-model speed, least possible memory or the multi-GPU-to-one-GPU product goal.
