# OjasX: GPU nonlinear discovery and verified scope

The new work provides an automated, checked mathematical candidate for reducing activation storage and modeled arithmetic in small-input, wide-hidden GELU/SiLU/SwiGLU blocks. It does **not** establish faster nonlinear training: the final guarded GPU implementation remained slower. This is an opt-in research module, with automatic backend dispatch disabled.

All new ML tensor computations, reference comparisons, gradient tests and timings ran on the Apple GPU through PyTorch MPS, with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Python orchestrated the runs and read scalar results. SymPy, Z3, Lean and Rocq ran as host proof processes. No CPU ML reference was substituted. The Neural Engine was not used.

The machine-readable results are [qualification](reports/nonlinear-qualification.json), [GPU cases](reports/nonlinear-gpu-search.json), [repeat timings](reports/nonlinear-gpu-repeat.json), [edge cases](reports/nonlinear-gpu-edges.json), [formal audit](reports/nonlinear-formal-verification.json) and [discovery trace](reports/nonlinear-discovery.json). Qualification checks the recorded source hashes against the current files.

## What changed mathematically

Consider a dense nonlinear block with all weights trainable:

\[
X\in\mathbb R^{m\times k},\quad A\in\mathbb R^{k\times n},\quad
B\in\mathbb R^{n\times p},\qquad Y=\phi(XA)B.
\]

Ordinary execution constructs the wide intermediate `X @ A`. Ordinary backpropagation needs input, first-weight and second-weight gradients. Optimizing only the input gradient would omit a large part of training.

The discovery tool finds the local quadratic Taylor candidates

\[
\operatorname{SiLU}(z)\approx z/2+z^2/4,
\qquad
\operatorname{GELU}(z)\approx z/2+z^2/\sqrt{2\pi}.
\]

These approximate the native activations; they are not exact identities for SiLU or GELU. The runtime restricts the gate preactivation to `|X @ A| <= delta`, where `0 < delta <= 0.05`. The gate is checked on every call and its matrix multiplication is charged to the candidate.

For the **polynomial itself**, multilinear algebra gives an exact identity. Let

\[
Q(X)_{r,(i,j)}=X_{ri}X_{rj},\qquad
T(X)=[X,Q(X)],\qquad
H(A)=\begin{bmatrix}A/2\\c\,(A_i\odot A_j)_{i,j}\end{bmatrix}.
\]

Here each row of the second block of `H` contains products of two rows of `A`, and `c` is the quadratic coefficient. Then

\[
\widehat Y=T(X)C,\qquad C=H(A)B.
\]

The expensive batch-by-hidden contraction has been replaced by an input-feature-by-hidden contraction and a batch-by-feature contraction. Its feature width is `s = k + k²`, independent of hidden width `n`.

For SwiGLU the gate and up-projection have distinct trainable weights:

\[
\widehat Y=((XA)/2+(XA)^2/4)\odot(XV)\;B.
\]

Its input features have degree two and three, with full tensor width `s = k² + k³`. Both `A` and `V` receive gradients. This larger feature space is a material limitation.

This combines Taylor approximation, tensor-product algebra and adjoint composition. The abstract engine supplies the chain and branch rules; the candidate supplies a specific primal and its derivatives. Category theory organizes composition and correctness obligations. The arithmetic saving here comes from the low-width polynomial feature representation.

## All gradients use the same primal

For an arbitrary upstream cotangent `G`, the contracted circuit computes

```
dC = T.T @ G       dT = G @ C.T
dH = dC @ B.T     dB = H.T @ dC
```

The feature and coefficient pullbacks recover `dX`, `dA`, and, for SwiGLU, `dV`. For example, the quadratic coefficient pullback contains the sum of both index orders, `dH[i,j] + dH[j,i]`. Repeated indices have multiplicity two; cubic input features have three contributions. Omitting these terms gives incorrect gradients.

SymPy differentiates the coefficient maps; Z3 checks their identities; Lean and Rocq check generated proofs. The final runtime uses full tensor contractions instead of executing each symbolic term separately. Higher derivatives recompute the feature statistics when building the derivative graph. GPU Hessian-vector tests compare derivatives of the same surrogate. They do not establish equality to higher derivatives of the native activation.

## Automated tools and exact proof boundaries

The formal-only checker passes:

- **283 Lean project theorems**, with only the standard audited Lean axioms `propext`, `Classical.choice` and `Quot.sound`; no added project axioms or unfinished proofs.
- **133 Rocq statements**, including 33 new exact-rational polynomial/pullback identities, closed under the global context and independently checked by `rocq check`.
- **37 new Z3 checks**, including polynomial identities, tensor pullbacks, sigmoid factor bounds and a counterexample to an identity with a generic bounded sigmoid value.

The new Lean module has 59 generated statements. It includes actual `HasDerivAt` proofs for the first four derivatives of SiLU, using the previously proved derivative of the actual sigmoid function. The separate bound module adds six statements: a general Taylor remainder bound, quadratic primal/slope specializations, weighted residual error, two sigmoid factor bounds packaged in one theorem, and a fourth-derivative polynomial bound.

The Z3 witness involving a bounded sigmoid variable is a counterexample to that encoded polynomial identity. It is not a proof that this variable equals `sigmoid(x)` at the witness. Actual native activation comparisons occur on the GPU; actual SiLU differentiation occurs in Lean.

The search is reproducible and bounded: quadratic/cubic polynomials, generated input widths 1, 2 and 4, and explicit shape/domain gates. It does not search every mathematical theory, establish global optimality, or prove future ML models are covered.

Rocq's new statements use exact canonical rational arithmetic. The GELU coefficient is a free coefficient in these algebraic identities. Identifying its analytic value belongs to the real-analysis layer. Exact rational specifications and program extraction do not certify the Python implementation, Metal GEMMs, compiler, runtime, or floating-point hardware.

## Error bounds and remaining analytic obligations

The Lean Taylor theorem proves, under its stated smoothness, derivative and Taylor-coefficient premises,

\[
|\phi(z)-\widehat\phi(z)|\le M\delta^4/24,
\qquad
|\phi'(z)-\widehat\phi'(z)|\le M\delta^3/6.
\]

The specializations require `z != 0`. Zero inputs are tested separately on GPU; no unstated extension of these particular theorem statements is claimed.

The proposed analytic constants are `M = 1 + delta/2` for SiLU and `M = (delta⁴ + 7 delta² + 4)/2` for GELU. At `delta = 0.05`, the corresponding **conditional real-arithmetic** local bounds are approximately:

| Activation | Primal absolute error | Slope absolute error |
|---|---:|---:|
| SiLU | 2.67e-7 | 2.14e-5 |
| GELU | 5.23e-7 | 4.18e-5 |

The actual SiLU derivative chain and the algebraic sigmoid bounds are checked. The complete activation-specific connection to the smoothness/coefficient premises of the Taylor theorem, the GELU analytic remainder constant, and FP32 GEMM roundoff are **not** all discharged. These constants are not a certified end-to-end GPU error budget.

Local errors also grow with surrounding weights and upstream gradients. The checked weighted-sum theorem gives

\[
\left|\sum_j e_jw_j\right|\le\epsilon\sum_j|w_j|
\quad\text{when }|e_j|\le\epsilon.
\]

For SwiGLU, gate-slope errors are additionally multiplied by the up-projection. For a loss-dependent upstream gradient, differences in forward values introduce another error term requiring a loss/graph regularity bound. Long training trajectories and model convergence remain unproved.

## GPU results

Twenty-one nonlinear cases passed the recorded FP32 forward/all-gradient checks and four-step plain-SGD comparisons. Fifteen entered the approximate feature path; six exercised domain or width fallback. The contract combines `rtol = atol = 2e-4` with relative L2 error at most `2e-4`. Random upstream VJPs check every original operand gradient. Three Hessian-vector cases for the surrogate passed. Forty-five additional GPU checks cover generated widths, near-boundary values, rejected domains, NaN/infinity, coefficient overflow, strong cancellation, invalid radii, FP16 and autocast rejection.

These are bounded synthetic blocks. The four-step loss is mean squared output with a small SGD learning rate; some weight changes round below FP32 resolution. Agreement of short trajectories is not evidence of full-model training convergence.

For the large tested block `(m,k,n,p) = (512,2,4096,8)`:

| Candidate | Native fresh retained activation storage | Candidate fresh retained storage | Modeled scalar arithmetic reduction |
|---|---:|---:|---:|
| GELU quadratic | 16 MiB | 0.106 MiB | 15.68× |
| SiLU quadratic | 16 MiB | 0.106 MiB | 15.68× |
| SwiGLU polynomial | 32 MiB | 0.211 MiB | 15.20× |

Fresh retained storage excludes input and parameter storage. It is about 99.34% lower here. The sampled allocator high-water observations also decrease, but the domain guard still constructs a wide preactivation. The implementation does not eliminate all batch-by-hidden temporary storage.

The final candidates have **zero jointly accepted nonlinear speed/memory wins**. Repeated timing uses three independent seeds, 12 warmups, 11 shuffled rounds and 20 complete forward/loss/all-gradient/SGD steps per sample. Those repeats found every large nonlinear candidate slower. Timing variance between short runs and longer repeats is recorded rather than converted into a speed claim.

The normalized linear control `(512,16,2048,16)` passed the accuracy contract and measured a speed gain in the recorded run, with forward and all gradients timed. Its separate unscaled stress control failed pointwise tolerance despite small relative L2 error. Linear reassociation is exact over real arithmetic, but it changes FP32 rounding. Neither control includes optimizer timing. Earlier scoped cached-reverse and RMS experiments remain documented in [their original GPU guide](GPU_QUALIFICATION.md).

Memory measurements use saved-tensor storage accounting and allocator samples at ATen operation returns. The [PyTorch MPS API](https://docs.pytorch.org/docs/stable/mps.html) distinguishes tensor allocations from Metal driver allocations. Samples exclude hidden workspace and can miss transient allocations, so this report does not certify total GPU peak memory. The [MPS environment documentation](https://docs.pytorch.org/docs/stable/mps_environment_variables.html) describes the CPU fallback setting used in these runs.

## Why fewer scalar operations did not yield faster training

The first generated prototype executed hundreds of scalar-derived tensor operations. Replacing it with full tensor contractions reduced the number of ATen dispatches substantially, while increasing feature width slightly. Nevertheless, small GEMMs, broadcasts, data checks, domain reductions, CPU/GPU scalar synchronization and launch overhead outweighed the arithmetic savings in the final guarded tests.

The cost model charges all six feature-circuit GEMMs, feature construction, coefficient construction, their pullbacks, a full gate projection, and shared plain-SGD updates. Native transcendental and pointwise activation work is assigned zero, making the native reference a conservative lower model. The candidate's explicit tensor-arithmetic bounds are upper bounds for that straight-line model. Domain reductions, indexing/reshape/copies, kernel launches, synchronization and hidden backend workspace are outside that scalar model; measured runtime includes execution overhead.

These operation counts are not measured Metal instructions, FLOPs from a profiler, energy savings, or a proof of speed. The runtime guard itself adds scalar arithmetic/reductions beyond the contraction model. Momentum, Adam, master weights and optimizer state are not reduced by these identities.

## Using and reproducing the research module

The module is [ojasx2/nonlinear_taylor_gpu.py](ojasx2/nonlinear_taylor_gpu.py). Supply MPS FP32 tensors with shapes `X[m,k]`, `A[k,n]`, `B[n,p]`, and optionally `V[k,n]` for SwiGLU. The tensors may require gradients normally:

```python
from ojasx2.nonlinear_taylor_gpu import taylor_block

evidence = []
y = taylor_block(x, a, b, kind="silu", delta=0.05, evidence=evidence)
loss = y.square().mean()
loss.backward()
```

This call explicitly opts into the surrogate. It accepts generated widths 1/2/4 only when the arithmetic/storage shape gate passes, the preactivation gate passes, inputs are finite and bounded in magnitude, and the conservative gate conditioning indicator passes. The hidden width is bounded at one million for this research contract. Other supported shapes/domains fall back to the ordinary MPS activation block. FP16, autocast, invalid shapes and invalid radii raise an error. The magnitude checks avoid the tested overflow/cancellation failures; they are not a complete FP32 correctness certificate.

The domain decision is treated as a discrete choice; differentiation computes the selected branch's derivative, not the derivative of branch selection at its boundary. There is no transparent integration for arbitrary user-defined PyTorch graphs in this module.

From this project directory, with the existing Python environment:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_nonlinear_foundation.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/search_nonlinear_gpu.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/repeat_nonlinear_gpu.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/gpu_nonlinear_edges.py
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/summarize_nonlinear_qualification.py
```

Run GPU timing tools sequentially, without concurrent GPU benchmarks. The first command checks proofs only. The older broad verification commands also execute CPU numerical workloads; they were not used for this GPU-only experiment. In the agent sandbox MPS was unavailable; running the authorized tests outside the sandbox exposed the GPU and succeeded with fallback disabled.

## Implication for the OjasX foundation

The abstract engine now has a more concrete nonlinear candidate, additional actual derivative proofs, bounded error contracts, full original-parameter pullbacks, cost accounting, and reproducible GPU rejection evidence. Those make the foundation more testable and help avoid implementing an invalid optimization.

Broad modern dense ML blocks often have input widths in the hundreds or thousands. Quadratic/cubic feature growth makes this specific construction unsuitable there. Generalizing the algebra to arbitrary dimensions does not make its cost favorable. Ordinary non-small preactivations also reject the quadratic approximation.

A broadly useful arithmetic breakthrough still requires a compact representation that remains cheap through common nonlinear gates, handles all parameter gradients, carries numerical bounds across a complete training step, and survives measured runtime/storage costs. This experiment has not supplied that result. Further mathematical search should address those specific obstructions; the current nonlinear prototype should remain opt-in research.
