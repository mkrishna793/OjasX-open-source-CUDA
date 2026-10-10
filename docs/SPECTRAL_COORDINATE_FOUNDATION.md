# Coordinate-sensitive spectral and Veronese refinement for OjasX

The supplied Hermite–Veronese–tensor-contraction idea contains a valid factor identity. This refinement implements that idea, corrects its accuracy/rank assumptions, and improves the representation with shared bias translation and a degree-8 Chebyshev derivative basis. It preserves the original X, W, bias and projection-Q variables and computes the derivative of the **same polynomial used in forward execution**.

The new result concerns coordinate-wise GELU/SiLU approximations on explicitly bounded preactivation domains and narrow input spaces. It extends the previous radial compiler without requiring radial replacement activations. Finite polynomials do change the original activation; sampled agreement is not an exact identity or a universal training guarantee.

## Corrections that became proof obligations

| Supplied statement | Tool-checked refinement |
|---|---|
| Radial activation collapses activation rank to one | Radial maps preserve the vector direction and are rotation-equivariant. Their scalar scale is rotation-invariant; their full Jacobian is generally full rank. The earlier batch-rank counterexample remains valid. |
| GELU/SiLU have coordinate-reflection symmetry | Neither is odd, so `φ(-z)=-φ(z)` fails. Their shared coordinate-wise application is permutation-equivariant, with permutations consistently applied to the rest of the model. |
| LayerNorm/RMSNorm makes inputs Gaussian | Normalization imposes statistics, not a probability law. The deterministic vector `[-1,1]` is mean-zero/unit-variance, and stays two-point after LayerNorm. The GPU and Z3 witnesses reject a Gaussian assumption inferred from normalization alone. See [the LayerNorm definition](https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html). |
| Probabilists' Hermite polynomials are orthonormal as written | Their Gaussian squared norm is k!. Correct coefficients are `E[g He_k]/k!`, or use `He_k/sqrt(k!)` as an orthonormal basis. See [NIST's orthogonality table](https://dlmf.nist.gov/18.3). |
| Degree 3–4 gives derivative error below 1e-4 | The corrected GPU Gaussian quadrature gives degree-4 weighted L2 error about **0.07796 for GELU** and **0.02594 for SiLU**. Both fail that claim. Smoothness alone does not supply the claimed exponential-decay constants or a universal degree choice. |
| A Gaussian average error controls every gradient | It does not imply a uniform activation error, and small absolute errors can still yield large relative errors when a gradient nearly cancels. Error transport and whole-program qualification remain necessary. |
| `rank(Z) ≤ min(m,k,N)` is automatically small | That upper bound is true for the linear part but can be large. Bias needs accounting. A polynomial derivative's represented width grows combinatorially with input rank and degree. |
| The highest homogeneous-degree width bounds the entire polynomial | Multiple degrees require their combined feature space. A safe full monomial width through degree D is `binomial(k+D,D)` before additional sparsity is established. Incoming factors multiply widths unless contractions can be shared differently. |
| LoRA makes all hidden activations low rank | A low-rank update does not make the dense base activation low rank. Training U/V factors also differs from dense-W SGD. The implementation keeps all original dense parameters. |
| Radial memory necessarily needs an m×m Gram matrix | The earlier implementation uses a small input-space Gram matrix and its diagonal energies; it does not materialize an m×m matrix. |

The definitions of exact coordinate GELU and SiLU used in the reference tests follow [PyTorch's activation definitions](https://docs.pytorch.org/docs/stable/nn.functional). GELU uses `approximate='none'`.

## Exact polynomial identity and full original gradients

Consider the original projected block

\[
Z=XW+b,\quad H=\phi(Z)Q,
\]

with X of shape m×k, W of shape k×N, bias length N and Q of shape N×r. For a selected polynomial `p(z)=sum c_m z^m`, construct a feature basis indexed by multi-indices α with `|α|≤D`:

\[
\Psi_{i\alpha}=X_i^\alpha,\qquad
T_s(b)=\sum_{m=s}^D c_m {m\choose s}b^{m-s},
\]

\[
\Theta_{j\alpha}={s!\over\prod_a\alpha_a!}
 \left(\prod_a W_{aj}^{\alpha_a}\right)T_s(b_j),\qquad s=|\alpha|.
\]

Then, exactly over the reals,

\[
p(XW+b)=\Psi\Theta^T,\qquad H_p=\Psi C,\quad C=\Theta^TQ.
\]

Each hidden coordinate keeps its own W column, bias and coefficient row. No shared radial scale or rotation assumption is imposed. The monomial multiplicities are essential: the tool's counterexample-guided search rejects a square without its mixed term and discovers coefficient 2.

Bias is **folded into coefficient shifts T_s**, rather than introducing a new input feature coordinate and retaining duplicate powers of the appended constant. All T_s are shared across monomials of the same total degree. Their derivatives satisfy `T_s'(b)=(s+1)T_(s+1)(b)`, with `T_(D+1)=0`.

For upstream P, compute

\[
\nabla_C=\Psi^TP,\quad \nabla_\Psi=PC^T,\quad
\nabla_\Theta=Q\nabla_C^T,\quad \nabla_Q=\Theta\nabla_C.
\]

Apply the zero-safe monomial derivative `d(x^n)/dx=n x^(n-1)` to obtain the original ∇X and ∇W. Apply the shared T_s derivatives to obtain ∇b. The implementation does not divide by X/W values and handles zero components without a reconstruction division.

The forward, backward and Q-gradient all reuse the coefficient/projection core. This can avoid explicitly building both the m×N activation and its m×N gradient. Θ still costs N×S storage/work, where `S=binomial(k+D,D)`. Dense ∇W/∇Q outputs remain dense. The corresponding Khatri–Rao identity is already proved for arbitrary finite factor spaces in `GradientFactors`; it does not justify truncating the product index back to its old rank.

## Why the degree-8 refinement is more useful than the proposed Hermite truncation

The tools expand the degree-12 Taylor **derivative** into Chebyshev polynomials, retain terms through degree seven, then integrate the resulting derivative and set the primal constant to zero. Both execution directions therefore use one degree-8 primal. This is a deterministic symbolic basis conversion and checked truncation, not unrestricted automatic mathematics invention.

On `|z|≤1`, Mathlib proves `|T_n(z)|≤1`. The new Lean proofs use that fact and exact polynomial identities to certify the omitted derivative terms:

- GELU derivative difference versus the degree-12 polynomial reference is bounded by `83*|c|/9732096`, with `c=1/sqrt(2*pi)`, about 3.40e-6.
- SiLU derivative difference versus its degree-12 reference is bounded by `4747/8174960640`, about 5.81e-7.

These are **kernel-checked polynomial-reference bounds**. The analytic remainder from degree-12 Taylor to the true activation, and FP32 arithmetic error, are separate obligations. This refinement does not claim a complete formal GELU/SiLU remainder or IEEE proof. A 32,769-point MPS grid on [-1,1] also compares true activations and their actual derivatives:

| Activation | Maximum sampled primal absolute error | Maximum sampled derivative absolute error |
|---|---:|---:|
| GELU | 1.0133e-6 | 1.2040e-5 |
| SiLU | 3.5763e-7 | 3.3379e-6 |

For two input coordinates, reducing degree 12 to degree 8 cuts the feature basis from 91 to 45. The runtime caches shared monomial powers for backward use and folds bias into the coefficient calculation. These changes reduce representation overhead relative to the first degree-12 prototype.

## Formal scope and revised assumptions

No optimization or Gaussian axioms are added. `rocq/SpectralFoundation.v` introduces a typed `PolynomialContract`: the implementation must match its polynomial primal/reverse, while original-function error bounds are explicit obligations. An assumption stored in that contract is not evidence that it holds for every activation or program.

`lean/OjasXMath/SpectralFoundation.lean` proves actual derivatives of arbitrary finite polynomials, tensor-power projection identities, shared projection/Q-gradient contractions, zero-safe monomial differentiation and error transport for input, weight, bias and projection gradients. For example, a coordinate derivative-error bound ε implies

\[
\left|\sum_j (p'(z_j)-\phi'(z_j))G_jW_j\right|
 \le\epsilon\sum_j|G_jW_j|.
\]

This is an absolute error contract for fixed upstream factors. When the loss or upstream cotangent also changes, additional terms appear; `spectral_loss_cotangent_error` makes them explicit. Cancellation can prevent conversion to a small relative error. Whole-model SGD/AdamW tests therefore complement local identities.

Generated degree-2/4/6 biased coordinate examples include primal identities, all original-variable gradients and actual Lean derivative proofs. The generated Chebyshev identities and reference-tail bounds are checked in Lean. Rocq independently checks finite matrix reassociation, bias-folding rules and the generated rational polynomial identities. All proof modules, including earlier work, are compiled, audited for assumptions and independently kernel-rechecked by the final runner. Final counts are in [the formal report](reports/spectral-formal-verification.json).

The valid conditions are now explicit: a matching polynomial primal/slope; retained original parameters; accounted feature and upstream widths; a bounded numerical domain or a separate rigorous remainder contract; acceptable whole-program gradient/optimizer error; and measured cost/memory eligibility. The tools refine these conditions rather than declaring desired savings as axioms.

## GPU results and rejection policy

Numerical ML, coefficient quadrature, reference gradients and optimizer arithmetic all run on MPS with CPU fallback disabled. Formal proof tools and symbolic algebra run on the host. Final benchmarks run after proof compilation finished, so concurrent formal builds do not contaminate those measurements. Earlier observations during compilation remain archived as `spectral-gpu-during-build.json`; the initial degree-12 implementation/report are also preserved.

Repeated full-step timings include forward, MSE loss, all input/original W/b/Q gradients, plain SGD, and numeric/domain guards. Baseline is eager execution of the original GELU/SiLU block. Static GPU metadata is warmed before timing, which is a steady-state measurement.

| Shape [m,k,N,r], degree 8 | GELU: three seed speed ratios | SiLU: three seed speed ratios | Native → candidate fresh saved bytes |
|---|---|---|---|
| [8192,2,4096,2] | 1.502×, 1.265×, 1.191× | 1.105×, 1.345×, 0.921× | 268,435,456 → 3,981,672 |
| [8192,1,8192,2] | 3.431×, 2.814×, 3.319× | 3.540×, 2.782×, 1.686× | 536,870,912 → 1,769,544 |

The two-input SiLU case fails repeated speed admission. One-input large cases pass it for both activations, and the two-input GELU case passes. These particular input widths are strong structural restrictions; they do not represent a normal width-768 Transformer layer.

Saved activation-storage reduction is 98.52% for the two-input block and 99.67% for the one-input block. It excludes original inputs/parameters, optimizer state, dense gradient outputs, transient workspace and total allocator peak. It does not imply equal savings in total GPU RAM.

Three composed GELU→projection→SiLU→projection classification models and one model with LayerNorm before the first block passed output/all-gradient comparisons, four SGD steps and four AdamW steps. AdamW moment, variance and parameter calculations were all GPU operations. These were narrow-input, bounded-domain models of shape `[4096,2,4096,2,8]`; they establish no model-quality, long-term convergence or complete-model speed gain. Memory hooks include the cross-entropy tape in this composition test. Raw candidate results are preserved separately from final admission decisions.

The suite also checks twelve polynomial cases, twelve repeated large cases, eleven policy/domain/numeric/frozen/noncontiguous/zero/cotangent edges, two higher-derivative comparisons and two scalar grids. All polynomial implementation and final admission checks passed. Four degree-4 candidates failed comparisons against the native activation and remain rejected. Hermite degree-4 accuracy claims also remain rejected. Higher derivatives are checked against the **same polynomial surrogate**, not asserted equal to native activation Hessians.

Numerical comparisons retain the earlier criteria: `allclose` with absolute/relative tolerance 2e-4 plus relative L2 error ≤2e-4. No tolerance was relaxed to turn rejected degree-4 candidates into successes. The final [GPU report](reports/spectral-gpu.json) contains individual errors and optimizer checks.

## Cost, broad scaling and implementation

The feature count grows as `binomial(k+D,D)`. Even small polynomial degrees can become impractical at large k. The new runtime rejects unsupported input widths, feature counts above 512, poor modeled full-step costs, autocast and unsupported dtypes. Its bounded executable grammar covers k=1,2,3,4; passing current accuracy/speed evidence is narrower. Models outside that grammar continue using native execution.

A cheap box bound avoids building Z merely to check its domain:

\[
\max_i |Z_{ij}|\le \sum_a (\max_i|X_{ia}|)|W_{aj}|+|b_j|.
\]

The experimental candidate requires this bound ≤1, finite values, |X/W/b|≤1 and |Q|≤8. This can conservatively reject a case whose actual Z would fit. It never infers Gaussianity or clips values to manufacture accuracy. These guards are research safeguards, not an FP32 overflow/rounding certificate. Cost savings are leading arithmetic models, not proven GPU instruction or latency bounds.

The public API is opt-in:

```python
from ojasx2.spectral_compiler import spectral_projection

# All tensors are original FP32 MPS inputs/parameters.
h = spectral_projection(x, W, bias, Q,
                        kind="gelu", degree=8, accuracy="approximate")
```

Default `accuracy="strict"` uses the original activation. `polynomial` mode examines admitted polynomial branches; `approximate` admits only the research degree-8 or degree-12 families. Cost/domain/policy rejection uses native execution. Thus the guarded API is not a globally identical polynomial model across all branch boundaries, and no derivative theorem for those boundaries is claimed. Whole-program preflight remains required before adopting a candidate for a sensitive model or optimizer.

Unsafe upstream cotangents and higher derivatives recompute the same polynomial densely on an admitted branch. This preserves the surrogate's derivative semantics but can restore wide temporary tensors in those fallback paths. The standard first-order admitted branch avoids m×N activation/cotangent materialization.

For backend integration, recognize an existing coordinate activation followed by a projection; retain all original parameter objects; attach the polynomial, domain, error and feature-cost contracts; lower feature/translated-coefficient construction and contractions to GPU tiles; and qualify the complete graph. Refresh Θ/C after every parameter update. A native lowering could fuse monomial construction and reductions to reduce dispatch overhead, but this refinement uses ordinary MPS operations and does not deliver a new Metal/CUDA compiler. Tile liveness, aliasing, precision, total peak memory and general graph dispatch remain separate work.

## Reproducible tool loop

From this directory, using the installed Python with SymPy, Z3 and PyTorch:

```sh
python -B tools/refine_spectral_foundation.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B tools/refine_spectral_foundation.py --gpu
python -B tools/refine_spectral_foundation.py --qualify-existing-gpu
```

The runner performs bounded symbolic discovery and counterexample checks, regenerates proof/coefficients, audits all Lean/Rocq modules, independently rechecks Rocq, optionally runs the GPU suite, and creates a source-hash-matched admission ledger. Numerical failures or stale evidence stop qualification. It installs no background automation and runs no legacy CPU ML test suites.

Reports: [discovery and corrected assumptions](reports/spectral-discovery.json), [formal audit](reports/spectral-formal-verification.json), [GPU tests and raw approximation rejections](reports/spectral-gpu.json), and [qualified shapes/policies](reports/spectral-qualification.json).

The foundation now supports coordinate-sensitive polynomial representation and a bounded, tested approximation family for ordinary GELU/SiLU. The remaining broad bottleneck is feature growth with input dimension, along with whole-model numerical error. This work supplies conditional improvements; it does not establish an exact compact representation or dramatic savings for most existing training workloads.
