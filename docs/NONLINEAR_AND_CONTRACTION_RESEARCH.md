# Further contraction and nonlinear-backward research

This extension investigates two questions: can the earlier linear-segment saving generalize, and can actual nonlinear backward computations use smaller contractions while preserving every original parameter gradient? It delivers a longer-chain planner, a guarded shared-mask ReLU schedule, a structured LayerNorm segment, proofs/checks and measured positive and negative controls. No gradient rank truncation is enabled.

## 1. Longer linear chains: optimize the backward as well

For compatible matrix leaves `A0,…,As`, associativity permits many binary trees. The old adjacent-pair rule can miss a useful narrow matrix farther down the chain. The new planner searches all binary parenthesizations by dynamic programming, minimizing forward plus demanded first-order pullback arithmetic. It uses ordinary PyTorch autograd on the selected tree, so all original trainable leaves and shared-leaf accumulation remain present.

For a node with dimensions `a×b` and `b×c`, scalar multiply/add counts are:

```text
forward:        2abc − ac
left pullback:  2abc − ab    if the left subtree contains a requested leaf
right pullback: 2abc − bc    if the right subtree contains a requested leaf
all demanded:   6abc − (ac + ab + bc)
```

Let `C[i,j]` be the best modeled cost for leaves i through j. Its recurrence is the minimum, over every split s, of `C[i,s]+C[s+1,j]+node_cost(d_i,d_(s+1),d_(j+1),left_need,right_need)`. A subtree needs a cotangent exactly when one of its leaves is differentiated. Base costs are zero for a single leaf. Optimal substructure supplies a mathematical induction proof of scalar-model optimality among these trees; this DP induction/implementation refinement is not mechanized in Lean. The tests independently enumerate every binary tree and differentiation pattern for several short chains and check the recurrence's result.

Lean checks the all-demand node cost and arbitrary-dimension associativity. Native autograd supplies differentiation of the selected graph; tests check all leaves and higher derivatives. This is not an optimizer over arbitrary DAG common-subexpression sharing, fanout, dispatch, numerical conditioning or tape/temporary memory.

### Measurements and use

| Matrix dimensions | Native ms | Planned ms | Speedup | Earlier pair rule ms |
|---|---:|---:|---:|---:|
| 128 × 256 × 512 × 1024 × 16 | 0.7818 | 0.2875 | 2.72× | 0.8000 |
| 256 × 512 × 1024 × 2048 × 32 | 5.5713 | 1.2171 | 4.58× | 5.5780 |
| 128 × 256 × 512 × 256 × 512 | 0.4650 | 0.4739 | 0.98× | 0.4839 |

The large segment removes 87.25% of modeled scalar arithmetic and 93.75% of fresh saved storage, with 11.20% lower profiler incremental tensor peak. The large three-weight segment has dimensions batch 256, widths 512→1024→2048→32. The selected tree is `X @ (W1.T @ (W2.T @ W3.T))`. The earlier pair rule falls back and misses this opportunity. Nine GEMMs remain; their dimensions change. Modeled arithmetic and memory savings are in [research-extensions.json](reports/research-extensions.json), along with eleven randomized paired timing samples per variant. No optimized tree is selected for the wide-output control when its modeled saving is zero.

```python
from ojasx2 import make_backend, linear_sequence, matrix_chain_plan
compiled = torch.compile(model, backend=make_backend(
    accuracy="algebraic", contraction_policy="matrix_chain"))
# Explicit bias-free sequence, retaining the original weight objects:
y = linear_sequence(x, (w1, w2, w3), accuracy="algebraic")
plan = matrix_chain_plan((256, 512, 1024, 2048, 32))
```

The compiler recognizes maximal adjacent bias-free single-consumer Linear chains. It preserves activations, dropout and fanout boundaries. Runtime eligibility is positive compatible 2D CPU FP32/FP64 with no CPU autocast. The strict default uses the original left fold. Cached plans contain only immutable shape/demand data; numerical parameter products are recomputed after updates. The public plan result is a fresh dictionary.

## 2. Why ordinary nonlinearities obstruct arbitrary collapse

For `Z=XW1.T`, `H=φ(Z)`, `Y=HW2.T`, define the derivative grid `D_ij=φ′(Z_ij)`. The ordinary backward is:

```text
Q   = G @ W2
 dZ = Q ⊙ D
 dX = dZ @ W1
 dW1 = dZ.T @ X
 dW2 = G.T @ H
```

The derivative is the adjoint of the actual input-dependent Jacobian. Even if Q has rank at most the narrow output width, elementwise multiplication by a general D can increase rank. A rank-one all-ones Q multiplied by a 2×2 identity mask has rank two. A categorical dagger licenses correct composition; it does not prove D can be replaced by a constant or remove the independent weight cotangents.

If an exact supplied factorization `D_ij=Σ_q U_iq V_jq` exists and `Q_ij=Σ_a G_ia C_ja`, then:

```text
(Q ⊙ D)_ij = Σ_q Σ_a (U_iq G_ia) (V_jq C_ja)
```

[Nonlinear.lean](lean/OjasXMath/Nonlinear.lean) proves this arbitrary-finite-index expansion. It supplies factors of width `r*p`; the conventional rank upper bound follows from the factor width. No decomposition algorithm or free low-rank assumption is supplied. If the width becomes large or discovering factors costs too much, the representation can lose. Smooth activations also leave the second weight gradient dependent on their actual H; a factored derivative alone does not reconstruct H.

## 3. ReLU: a proved conditional branch-region schedule

ReLU is piecewise linear. If **all rows of the actual batch have the same exact Boolean branch mask**, define a diagonal matrix `D` for that mask. Then `H=Z D`. The pullback is a linear-chain pullback with that same D inserted. Lean proves the ReLU gate value, the actual scalar derivative on positive and negative open regions, and all three shared-diagonal matrix pullback identities. At zero the classical derivative is absent; runtime tests use PyTorch's selected zero cotangent convention.

In row-oriented code, set `C=W2 D`, `T=G.T X`, `P=C W1`. Then:

```text
 dX  = G P
 dW1 = C.T T
 dW2 = (T W1.T + sum_rows(G).T b1.T) D
 db1 = (sum_rows(G) W2) D
 db2 = sum_rows(G)
```

The first-bias correction is omitted if b1 is absent. This computes full dense gradients, including zeros for inactive channels. Native forward evaluation is retained, so the sampled forward bytes match exactly. Backward reassociation changes rounded values.

The implementation checks the current batch mask, cost and finite inputs; it never assumes a previous mask remains valid. Candidate finite checks use scalar min/max endpoints, with explicit tests for NaN and both infinities, avoiding full-size pointwise validation temporaries. Original input/weight/bias references and one Boolean per channel are retained. There is no wide activation tape on the selected path.

### Positive and negative controls

| ReLU case / shape 512 × 1024 × 2048 × 32 | Native ms | Guarded ms | Speedup | Fresh saved bytes: native → guarded | Tensor peak reduction |
|---|---:|---:|---:|---:|---:|
| common_mask | 7.7934 | 5.8602 | 1.33× | 4,194,304 → 2,048 | 21.86% |
| random_distinct_masks | 7.5894 | 8.7254 | 0.87× | 4,194,304 → 4,194,304 | 0.00% |

The common-mask case is deliberately constructed using first biases ±3 and small random inputs; it is not a claim that random ReLU batches commonly satisfy the premise. The varying-mask control had 512 unique mask rows and a full-rank sampled 64×128 mask. It uses the native backward and still pays detection overhead. The first implementation lowered tape but increased peak and slowed execution; [the preserved earlier report](reports/research-extensions-before-guard-optimization.json) records that negative finding. Optimizing finite-validation temporaries addressed the regression on the favorable case; it did not make the condition universal.

A forty-step favorable synthetic SGD test keeps both dense weights and biases trainable. Its output/gradient/parameter comparisons are recorded in the research report. First and second derivatives, mutation checks, exact zero-boundary selection, autocast fallback and exception cases are tested. First-order raw-bit gradient equality is not claimed.

```python
from ojasx2 import relu_linear_chain
# Original module parameters, loss and optimizer remain usable.
y = relu_linear_chain(x, first.weight, second.weight,
                      first.bias, second.bias, accuracy="algebraic")
```

This explicit adapter is not a general automatic nonlinear graph pass. A future exact grouped-mask rule could partition rows into q distinct masks and accumulate each group's full weight contributions. Ignoring pointwise/grouping work, its leading changed backward cost is `6qpnk+4mpk`, versus dense `4mn(k+p)`. For many distinct masks q grows and the advantage disappears; grouping/scatter and dense-gradient accumulation must also be charged. That grouped runtime is not implemented here.

## 4. LayerNorm: a nonlinear reduction without identical masks

LayerNorm has a structured Jacobian even when every row differs. For one row with n coordinates, centered values `c=z−μ`, `μ=mean(z)` and `r=(mean(c²)+ε)^(-1/2)`, differential calculus gives:

```text
δμ = mean(δz)
δvar = 2 mean(c ⊙ δz)       because sum(c)=0
δr = −r³ mean(c ⊙ δz)
δ(c r) = r(δz−mean(δz)) − r³ c mean(c ⊙ δz)
```

Thus the input Jacobian is `rI − (r/n)11.T − (r³/n)c c.T`: a diagonal plus two rank-one corrections. It is self-adjoint before the affine scale. Epsilon remains in r; the centered direction is generally not a null direction when epsilon is positive. Removing it as if this were an exact tangent projection would be incorrect.

For a segment `Z=XW1.T`, `H=(Z−μ)r ⊙ γ+β`, `Y=HW2.T+b2`, all original weights, scale γ, shift β and final bias b2 can remain independently trainable. The current first Linear must be bias-free. This implementation keeps the native forward and its native returned row statistics. It does not substitute another model architecture.

### A complete shared first-order schedule

Rows scale by the m×1 vectors r and μ; γ and β have n entries. Let `s=sum_rows(G)` and compute:

```text
C = W2 ⊙ γ                       # p×n
T = G.T @ (r ⊙ X)                # p×k; shared
V = G.T @ (r ⊙ μ)                # p×1
K = T @ W1.T − V                 # p×n; broadcast V across n
R = W1.T @ W1                    # k×k; needed only for dX

α = (r²/n) ⊙ sum_columns(G ⊙ (Y−b2−W2@β))
u = α ⊙ μ − (r/n) ⊙ (G @ sum_columns(C))
Q = X.T @ (α ⊙ X)               # k×k; needed only for dW1

 dX  = (r ⊙ G) @ (C @ W1)
       + u ⊙ sum_rows(W1)
       − (α ⊙ X) @ R
 dW1 = C.T @ T + broadcast_rows(u.T @ X) − W1 @ Q
 dW2 = K ⊙ γ + s.T ⊙ β
 dγ  = sum_rows(W2 ⊙ K)
 dβ  = s @ W2
 db2 = s
```

Absent scale/shift/bias use their identity/zero conventions. The scalar identity behind α is `sum_j dH_ij γ_j zhat_ij = sum_a G_ia (Y_ia−b2_a−(W2β)_a)`. It reuses the already available narrow output rather than materializing a wide normalization cotangent solely for that dot product. It is the same finite-pairing principle used in the attention row-scalar identity.

To see the reduction explicitly, expand the normalization cotangent as:

```text
dZ = diag(r) G C + u 1.T − diag(α) X W1.T
```

Multiplying by W1 and by X.T gives the two Gram formulas above through associativity and distributivity. The T/K expressions expand the second weight and scale gradients directly from the affine normalized forward. The new Lean matrix statements prove these dimension-general reassociations and shared cotangents. Z3 independently checks the complete six-gradient symbolic segment at input shape 2×2, hidden width 3 and output width 2. The analytic LayerNorm Jacobian is derived above, independently differentiated with SymPy including positive epsilon, and tested against native gradients/higher derivatives; a universal analytic LayerNorm `HasFDerivAt` constructor is not yet supplied in Lean. This distinction is part of the proof boundary.

### Cost, memory and measurements

The native leading backward GEMM work is `4mn(k+p)`. With every input/weight/scale gradient requested, the new main-GEMM leading work is:

```text
6pnk + 4mpk + 4nk² + 4mk²
```

Lean proves the count identity and that **more than half of this leading backward arithmetic is removed when positive dimensions satisfy `m≥8k` and `n≥8k`**. The native forward products are still performed; normalization, vector products, finite checks, dispatch and transfers are additional work. The selector uses requested-gradient-dependent scalar counts for the main GEMMs. It does not select the k-large negative control.

| Shape m × k × n × p | Native ms | Native-forward structured ms | Speedup | Fresh saved bytes: native → structured | Tensor peak reduction |
|---|---:|---:|---:|---:|---:|
| 512 × 32 × 2048 × 16 | 1.7405 | 1.3587 | 1.28× | 8,392,704 → 36,864 | 34.18% |
| 512 × 64 × 2048 × 16 | 1.9652 | 1.5844 | 1.24× | 8,392,704 → 36,864 | 34.18% |
| 128 × 256 × 512 × 16 | 0.3442 | 0.3590 | 0.96× | 525,312 → 525,312 | 0.00% |

The selected path saves the narrow native output plus two native row statistics and references to original inputs/parameters, rather than full pre-normalization and normalized activations. Full input and parameter gradients remain dense. Higher derivatives deliberately replay the native differentiable segment to reconnect statistics and output; their arithmetic/memory does not follow the first-order savings. Nonfinite upstream also uses native replay. The benchmark includes forward, guards and all original gradients. It excludes optimizer time and reports profiler tensor peak rather than process/GPU residency.

The forty-step random-input synthetic AdamW qualification keeps all original weights, γ, β and the final bias trainable. It is not a real-dataset convergence or accuracy certificate. The report records every selected step, maximum sampled gradient discrepancy and final output/parameter differences. Reassociation and Gram expansions can be sensitive to cancellation; finite checks alone do not establish a global error budget. Ill-conditioned/scaled cases need separate qualification and future conditioning guards. No rank truncation or changed precision is used.

```python
from ojasx2 import layernorm_linear_chain
# First layer must be bias-free; norm normalizes only the last hidden axis.
y = layernorm_linear_chain(
    x, first.weight, second.weight, norm.weight, norm.bias, second.bias,
    eps=norm.eps, accuracy="algebraic")
```

This explicit helper is qualified on matching 2D CPU FP32/FP64 and positive epsilon. Strict mode, unprofitable dimensions, CPU autocast and unsupported contexts retain native execution. It is not yet an automatic FX LayerNorm pattern pass. RMSNorm drops the mean correction and has a related one-rank-one Jacobian, suggesting a simpler extension; this phase does not enable a new structured RMSNorm segment runtime.

### Further reduction: remove the wide forward activation through centered weights

There is an additional exact-real identity for a bias-free first Linear. Define `a=mean_rows(W1)`, `B=W1−a` and `S=B.T@B`. The row moments can be computed without forming Z:

```text
μ = X @ a
variance = sum_columns((X @ S) ⊙ X) / n
r = rsqrt(variance + ε)
C = W2 ⊙ γ
Y = (X @ (C @ W1).T − μ ⊙ sum_columns(C)) ⊙ r + W2 @ β + b2
```

Centering W1 first avoids forming the variance as a subtraction of two large quadratic moments. In real arithmetic `Z−μ = X B.T`, so `variance_i = (1/n) X_i S X_i.T`. Lean proves mean interchange, centering, the arbitrary-finite quadratic Gram moment identity, the affine normalized forward reassociation and the main leading joint cost. Z3 checks complete mean/variance/forward identities in a finite symbolic instance. Since the row variances agree and positive epsilon supplies the same positive denominator, the real normalized forward functions agree; smooth derivative equality then follows. Ordinary native autograd differentiates this reduced graph, preserving all original parameters and higher derivatives.

For all leaves demanded, its leading main-GEMM forward/backward work is `6(nk²+mk²+npk+mpk)` versus `6mn(k+p)` for the native segment. Additional vector products, means, pointwise reductions, guards and dispatch remain. This variant can remove the wide m×n forward activation as well as its tape; it retains smaller Gram/intermediate tensors, including centered weights, rather than pretending no tensors are saved.

| Shape m × k × n × p | Native ms | Gram joint ms | Speedup | Candidate main-GEMM modeled saving | Fresh saved bytes: native → Gram | Tensor peak reduction |
|---|---:|---:|---:|---:|---:|---:|
| 512 × 32 × 2048 × 16 | 1.7405 | 0.7623 | 2.28× | 92.23% | 8,392,704 → 501,952 | 89.85% |
| 512 × 64 × 2048 × 16 | 1.9652 | 1.0110 | 1.94× | 84.41% | 8,392,704 → 844,096 | 81.05% |
| 128 × 256 × 512 × 16 | 0.3442 | 0.3645 | 0.94× | -150.11% | 525,312 → 525,312 | 0.00% |

The negative modeled-saving row describes the rejected candidate; that control executes the native graph and pays guard overhead. The native-forward structured option saves only 36,864 fresh bytes on the first sample, versus 501,952 for Gram, but its profiler incremental peak is 8,425,499 bytes versus 1,298,624 for Gram. Gram trades a larger small-tensor tape for removal of the wide forward temporaries. Select for the actual workload's tape/peak objective rather than treating these measures as interchangeable.

For the first Gram sample, maximum sampled gradient relative L2 discrepancy was 1.19e-06; output relative L2 discrepancy was 1.16e-06. All sampled comparisons passed `rtol=3e-4, atol=3e-5`. Fresh saved storage fell 94.02%, while the larger profiler peak fell 89.85%; these percentages describe different measurements.

This variant changes output rounding as well as gradient rounding. It is an experimental explicit helper, `gram_layernorm_linear_chain`, with strict native default. It checks finite inputs and rejects nonpositive/tiny variance (`variance≤10ε`), unprofitable shapes and an excessive common weight offset. Those conservative guards are **heuristics, not a universally certified numerical-error or condition-number bound**. Gram matrices can square conditioning, and near-null directions/cancellation can still matter. The samples and short training test qualify a finite set of inputs; they do not justify deploying on untested arbitrary scales. No precision reduction, clipping, rank truncation or permanent parameter replacement is performed.

```python
from ojasx2 import gram_layernorm_linear_chain
# Experimental algebraic forward; retain the same ordinary module parameters.
y = gram_layernorm_linear_chain(
    x, first.weight, second.weight, norm.weight, norm.bias, second.bias,
    eps=norm.eps, accuracy="algebraic")
```

The Gram training path selected all 40 steps, with maximum gradient relative L2 discrepancy 1.55e-06, final output relative L2 discrepancy 2.48e-06, and maximum final parameter difference 1.19e-07. These are finite trajectory observations. The same forty-step synthetic AdamW test compares this variant to both the original and native-forward structured schedules. The report records selection counts, gradients and final parameter/output differences. Its first and second derivatives are also checked numerically in double precision, and risky small-variance/large-offset controls use native execution. This is not an automatic compiler pass or a qualified GPU implementation.

## 5. Other nonlinear activations: what can and cannot be reduced

| Primitive | Exact-real derivative/VJP information | Useful reduction | Current boundary |
|---|---|---|---|
| Sigmoid | `y(1−y)` from y | Reuse an already live output | Native code already uses output-based backward; no new general arithmetic speedup claimed |
| Tanh | `1−y²` from y | Reuse an already live output | Same caveat; rounded execution and higher derivatives must be qualified |
| ReLU/leaky ReLU | Boolean branch, or two fixed slopes | Pack branch information; exploit exact common masks or sparse blocks | Activation/gradient outputs may already be retained by adjacent GEMMs; alias-aware memory matters |
| SiLU | `s+x s(1−s)`, `s=sigmoid(x)` | Shared sigmoid/pointwise recomputation; supplied low-rank derivative structure if present | The derivative varies per entry; output alone is not generally invertible or sufficient |
| Exact GELU | `Φ(x)+x exp(−x²/2)/sqrt(2π)` | Reuse sufficient pointwise information; tiled recomputation | Tanh-approximate GELU is a different primitive; no silent substitution |
| Softmax | `p ⊙ (g−sum(p⊙g))` | Diagonal minus rank-one VJP, no dense Jacobian | Native softmax already has an efficient VJP; this formula is not a new PyTorch speedup |
| LayerNorm | Diagonal plus two row corrections | Shared T/K and smaller Gram products in favorable surrounding shapes | First-order segment qualified; real-dataset and device performance pending |
| RMSNorm | Diagonal plus one row correction | Candidate extension of the Gram schedule | Existing pointwise/statistic reference remains; new structured segment not implemented |

The installed [PyTorch activation source](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/Activation.cpp) is the appropriate baseline for which information it already reuses. A new claim must beat that implementation or an equivalent native reformulation, not an artificially materialized full Jacobian. Low-order polynomial approximations or approximate mask grouping remain disabled without a declared error contract.

## 6. Attention and the softmax row scalar

For `O=P V`, `dP=dO V.T`, associativity of finite sums gives `sum_j P_ij dP_ij = sum_d O_id dO_id`. Lean proves the arbitrary finite row-pairing identity. The softmax diagonal-minus-rank-one pairing and zero-sum cotangent are also checked. Its analytic directional derivative is independently checked with SymPy; the full exponential softmax primitive is not newly certified in Lean.

The isolated benchmark compares constructing dP solely to obtain this scalar with reusing an already supplied O. It can be much faster and avoids a large intermediate, but it does not time a complete attention backward. This is a known ingredient of IO-aware exact attention; [FlashAttention](https://arxiv.org/abs/2205.14135) applies such structure with tiling and recomputation. Its published hardware results are not OjasX measurements. Native optimized attention is the real comparison target.

## 7. What physics changes in this design

Compute/bandwidth limits constrain schedules, not algebraic correctness. Reducing saved tape can increase recomputation or temporary peak. The ReLU guard experiment demonstrated this directly. LayerNorm's Gram schedule can replace wide matrix products with smaller ones but introduces new products and reductions; a hardware cost model must include all of them. Neither scheme implies zero dense-gradient output storage.

The practical goal is to avoid materializing large intermediates, reuse sufficient statistics/shared products, and choose tiles that fit the actual device hierarchy. The [hardware implementation guide](HARDWARE_IMPLEMENTATION_GUIDE.md) specifies tensor lifetimes, library-GEMM lowering, cache invalidation, target qualification and future tiling. Checkpointing explicitly trades recomputation for memory; [PyTorch's checkpointing discussion](https://pytorch.org/blog/activation-checkpointing-techniques/) describes those tradeoffs. [Reversible residual networks](https://arxiv.org/abs/1707.04585) require a suitable architecture; generic activation inversion is not free.

## 8. Verification and remaining research

The full runner passes **121 Lean statements, 33 scoped Z3 checks and 209 CPU tests**, with two unavailable-device skips. This phase adds 25 Lean statements, six Z3 models and 45 CPU tests to the preceding 96/27/164 baseline. The [source manifest](reports/verification.json) hashes all 52 checked sources. Detailed output is in [VERIFICATION.md](reports/VERIFICATION.md).

These checks establish exact stated real algebra and scoped implementation conformance. They do not certify every Python instruction, all floating inputs, universal ML accuracy, all optimizers or any physical GPU kernel. Hardware output/tape/peak distinctions and strict-versus-algebraic use remain explicit.

The next production steps are a global time/memory/error-aware contraction planner, automatic guarded LayerNorm/RMSNorm graph rules, quantified conditioning/error guards, real training datasets and one physically accessible GPU backend. For ReLU, grouped masks or exact zero blocks need a target cost model before broad use. For attention, compare a complete tiled backward to native optimized SDPA. Higher-order specialized schedules and formal primitive/IR refinement remain separate obligations. These are bounded research targets with positive and negative controls, not a promise to halve arbitrary backpropagation.
