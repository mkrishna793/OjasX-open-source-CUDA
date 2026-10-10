# Combined activation and backpropagation refinement

This iteration adds a bounded mathematical rule-discovery and qualification
laboratory. It searches existing calculus, tensor-product and matrix-composition
identities; checks counterexamples; generates proofs; and tests accepted rules
on the Apple GPU. It does not invent assumptions that make an invalid shortcut
true. It has not established a universal cheaper representation of backpropagation.

## The combined object

For a normal dense block and a trainable projection head,

\[
Z=XW+b,\qquad Y=\phi(Z),\qquad H=YQ,
\]

let the upstream cotangent of `H` be `P`. Ordinary reverse mode produces

\[
G=PQ^T,\qquad T=G\odot\phi'(Z),
\]
\[
\nabla_X=T W^T,\quad \nabla_W=X^T T,\quad
\nabla_b=\mathbf 1^T T,\quad \nabla_Q=Y^T P.
\]

Here `X` is batch × input width, `W` is input × hidden width, and `Q` is
hidden × projection width. **W, b and Q remain the original trainable
parameters.** We also return the input gradient. The loss can be any downstream
differentiable computation that supplies `P`, not just the test's squared loss.

The mathematical search operates on the combination `G elementwise φ'(Z)`.
Moving the derivative through an arbitrary matrix multiplication is invalid.
Treating a nonlinear activation as a linear operator is also invalid.

## The refined representation rule

Suppose the upstream gradient and actual activation derivative have explicit
representations

\[
G_{ij}=\sum_{a=1}^{r} P_{ia}Q_{ja},\qquad
D_{ij}=\phi'(Z_{ij})=\sum_{c=1}^{s} S_{ic}U_{jc}.
\]

Then their Hadamard product has the exact representation

\[
T_{ij}=\sum_{a,c}(P_{ia}S_{ic})(Q_{ja}U_{jc})
        =(AB^T)_{ij}.
\]

The factor space is the tensor product of the incoming factor space and the
derivative factor space. Its width is **r × s**, not automatically r. This rule
is proved for arbitrary finite index sets, including arbitrarily chosen batch
and layer sizes. The factors are not obtained by SVD or rank truncation.

All three affine-layer gradients can then be computed as

\[
\nabla_X=A(WB)^T,\qquad
\nabla_W=(X^T A)B^T,\qquad
\nabla_b=(\mathbf1^T A)B^T.
\]

The original projection gradient is still required. A transformation that
only computes the input gradient is not admitted as complete backpropagation.

This is a concrete extension of the earlier abstract adjoint engine: local
derivative contracts compose through the chain rule, while the compiler seeks
smaller representations for the resulting cotangent contractions. Category
theory supplies composition and duality; tensor products describe factor
interaction; finite linear algebra supplies executable contractions. None of
these alone implies a reduction for an arbitrary model.

## Linear and nonlinear specializations

**Linear:** `D=1` preserves the incoming factors. This version also composes the
forward projections: `H=X(WQ)+bQ`. It retains the small `WQ`, shares `XᵀP`
between the original W and Q gradients, and avoids storing the wide hidden
activation. This is checked against eager PyTorch, with every original weight
updated independently.

**Square activation:** `φ(z)=z²`, so `D=2Z=2XW+2b` has explicit factor width
`k+1`, where k is the input width. The combined gradient therefore uses width
`r(k+1)`. This is a genuine nonlinear function with exact polynomial derivatives;
it is not an approximation to GELU or SiLU. The forward square activation is
unchanged. Saving Y suffices for the original Q gradient; the slope factors
come directly from X, W and b, so the preactivation Z need not also be retained.

**ReLU:** if every sample has the same mask for each feature, the derivative
can be folded into Q without factor-width growth. The mask is checked on the
GPU. A varying mask selects dense backpropagation. PyTorch's chosen derivative
at zero is zero; this is a nonsmooth convention, not a theorem that ReLU is
classically differentiable at zero. Correctness tests passed, but its timing
was inconsistent, so it does not qualify as a demonstrated speed win.

**GELU, SiLU, tanh and sigmoid:** tools derive their slope expressions and GPU
tests compare them with native autograd. This search did not find cheap exact
polynomial derivative factors for their general dense case. The public block
uses native PyTorch for these activations. No surrogate silently replaces them.

## What the counterexamples changed

1. The bounded CEGIS loop starts with a candidate coefficient for a combined
   chain-rule contraction. Z3 finds a witness when it is wrong. The witness is
   added to the synthesis constraints; the corrected coefficient is verified
   symbolically. This is a small grammar, not autonomous discovery of all math.
2. A rank-one incoming gradient need not stay rank-one: take G to be the 2×2
   all-ones matrix and a valid ReLU preactivation `[[1,-1],[-1,1]]`. The derivative
   mask is the identity matrix. The gated gradient is full-rank. Z3 rejects any
   rank-one representation of it, and Lean/Rocq prove that factorization cannot
   exist. The refined product-index representation preserves the real width.
3. Output-only state does not determine every original parameter gradient:
   `ReLU(a*x+b)` with a=0, b=1 always outputs 1, but its a-gradient depends on x.
   Input/parameter sufficient information cannot be discarded merely because
   forward outputs agree.
4. Shape growth makes exact factors more expensive. The search evaluates
   **567 activation/shape decisions** and accepts **179 conditional arithmetic
   candidates** in its finite grid. These numbers are not workload coverage
   percentages. In particular, increasing input width increases square-slope
   factor width; increasing depth can repeatedly multiply factor widths.
5. GPU measurements rejected small arithmetic-favorable cases as speed wins.
   A refinement shared forward/backward contractions in the linear path.
   Larger layer sizes were tested rather than extrapolating small timings.
6. Nonfinite inputs, extreme input values and nonfinite/extreme upstream
   cotangents use native computation. Higher-order AD recomputes the native
   differentiable block, because a detached cached residual alone would give
   incorrect second derivatives. The recomputation cost is explicit.
7. Numeric guards run without autograd tracking: guard comparisons are not
   model derivatives and should not create an unused differentiation graph.

No numerical guard establishes a uniform floating-point error bound. All
optimized rules remain exact-real identities with **sampled FP32 qualification**.
Near cancellation, changed reduction order can still change bits or exceed a
tolerance outside the tested cases. `accuracy='strict'` is the default and
retains native PyTorch execution.

## GPU evidence and operation scaling

Current evidence is in [the GPU report](reports/gradient-factor-gpu.json) and
[the admission ledger](reports/gradient-qualification.json). The complete-step
timing includes forward, loss, input/W/b/Q gradients, ordinary SGD updates,
numeric guards and the ReLU mask guard when applicable. CPU ML execution and
MPS CPU fallback are disabled. Proof/search programs run on the host.

The largest repeated shape is **batch 8192, input width 128, hidden width 4096,
projection width 2**. See the ledger for the current measured timing ranges and
retained-storage bytes. The tested baseline is eager PyTorch's original block,
not torch.compile, an optimized fused implementation or a complete training model.

The leading cost of materializing the incoming gradient and computing the two
dense matrix gradients is approximately

\[
2mnr+4mnk,
\]

whereas an exact factor width R permits approximately

\[
4kR(m+n),
\]

plus factor construction, bias reduction and validation work. For square,
R=r(k+1). The large tested square shape reduces these modeled backward
contractions substantially, but the dense forward and original Q gradient
remain. A whole-step arithmetic ratio is consequently less dramatic than the
backward-contraction ratio. The planner's counts are leading scalar arithmetic
models, not proofs of Python/Metal instruction counts. Memory transfers,
comparisons, copies and occupancy require separate target models.

Memory measurements count fresh tensor storage saved by the block for autograd,
excluding the explicitly supplied inputs and parameters and deduplicating
aliases. They are **not total peak GPU RAM** and exclude dense gradient outputs,
optimizer state, allocator workspace and other model layers. Linear stores a
small composition; square retains one wide activation instead of two. ReLU
does not gain retained activation memory in this implementation.

## Breadth, including other modern operations

| Operation family | Current treatment | What is established |
|---|---|---|
| Linear projections | Joint forward/all-gradient contraction | Shape-conditioned exact-real rule; sampled GPU gains at large shapes |
| Square nonlinear projection | Product-index derivative factors | Exact nonlinear rule; all original gradients; scoped GPU gains/storage reduction |
| ReLU projection | Uniform-mask factors or dense fallback | Tested gradients; speed is not consistently improved |
| Dense GELU/SiLU/tanh/sigmoid | Native autograd; symbolic slope catalogue | Gradient support, without a new broad arithmetic reduction |
| Softmax and normalization | Existing structured contracts; native PyTorch in composed GPU graphs | No dense Jacobian needed; no new speed claim from these tests |
| Attention and convolutions | Native operations surrounding the tested blocks | Hybrid-graph gradient tests, not replacement attention/convolution kernels |
| Branches and user research activations | Native PyTorch composition | Tested repeated-use custom activation branch with all surrounding gradients |
| Unseen/custom primitive | Require a valid derivative/residual contract before an optimized rule | The general contract extends; an efficient representation is not automatic |

The mixed tests include convolution → LayerNorm → ReLU projection,
attention/softmax → GELU projection, and square projection → a custom
`y*sin(y)+0.1*y²+y` branch. Native surrounding operations retain their original
autograd implementation. This demonstrates composition on examples; it does
not certify every PyTorch operation or every future research architecture.

## Formal proof and reproducible refinement

Lean modules:

- `GradientFactors.lean`: conditional actual calculus rule, arbitrary finite
  Hadamard-factor closure, input/weight/bias pullbacks, joint linear forward
  and Q-gradient contraction, width growth, cost guard and ReLU rank witness.
- Generated `GradientDiscovered.lean`: actual input, weight and bias derivatives
  and joint dagger statements for linear, square and cube polynomial examples.
- The existing `AbstractEngine.lean` supplies compositional derivative and
  optimizer preservation contracts. Its leaf validity hypotheses remain required.

Rocq independently checks exact-rational versions of the factor algebra and
generated joint polynomial identities. It does not prove the Python/Metal
implementation's floating-point behavior. Auditing the complete project checks
for unfinished proofs and additional assumptions. Lean retains only its usual
`propext`, `Classical.choice`, and `Quot.sound`; Rocq statements are closed under
the global context. Proof counts certify statements, not universal performance.

Run one complete bounded refinement cycle from this directory:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 \
  /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B \
  tools/refine_gradient_foundation.py --gpu
```

That executes mathematical synthesis, all-project kernel/assumption checks,
GPU-only qualification, and a hash-checked speed/memory admission ledger. It
fails when the GPU is unavailable, a solver returns an unexpected result, a
proof fails, numerical checks fail, or evidence hashes are stale. It never
substitutes a CPU numerical workload. Without `--gpu`, it runs formal checks
only; `--qualify-existing-gpu` validates separately recorded current GPU results.

The ledger rejects speed admission unless at least three independent-seed runs
all exceed 1.05×. Memory admission is separate. It is a research decision record,
not an installed automatic graph rewrite. Slower initial reports are preserved
as `gradient-factor-gpu-initial.json` and `gradient-factor-gpu-guard-tracked.json`.
No background scheduler has been installed.

Explicit experimental use:

```python
from ojasx2.gradient_factor_compiler import projection_block

# x, w, bias and q are the original MPS tensors/parameters.
h = projection_block(x, w, bias, q, kind="square", accuracy="algebraic")
loss = downstream_loss(h)
loss.backward()
```

Use `accuracy="strict"` for native execution. The API currently recognizes
seven activations and this dense-projection pattern. It does not automatically
recognize or replace an arbitrary PyTorch model graph.

The measurement approach uses PyTorch's documented
[saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).
Higher-order derivative behavior follows the distinction between saved
residuals and differentiable recomputation in
[PyTorch's autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd).
