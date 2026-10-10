# OjasX abstract backpropagation engine and bounded mathematical discovery

The foundation now has an abstract exact-real engine, an automated polynomial
rewrite search, and a new nonlinear feature-contraction reference. This is
scoped mathematical progress, not a universal cheaper-training theorem.
The generated theorem claims become accepted only when the verification runner
finishes the Lean kernel and dependency audit. Current evidence is in
[abstract-verification.json](reports/abstract-verification.json).

## What changed

The engine extends the existing certified-stage foundation rather than adding
custom axioms. It uses the pinned Lean 4.34.0 and Mathlib v4.34.0 installation.
Additional Mathlib adjoint dependencies were built locally from the pinned
sources. Rocq and Z3 remain complementary checkers. Mathlib serves as both a
source of reusable theorems and an example of layered formalization; it is not
a benchmark claiming that OjasX computes faster than Mathlib.

The abstract source is [AbstractEngine.lean](lean/OjasXMath/AbstractEngine.lean).
The machine-generated polynomial identities are in
[Discovered.lean](lean/OjasXMath/Discovered.lean).
The CPU reference is [feature_core.py](ojasx2/feature_core.py).

The previous [normalization and indexing refinement](FOUNDATION_REFINEMENT.md)
still applies. Its 100 closed Rocq statements are retained. Lean now checks
218 project theorems, including 16 abstract-engine theorems and 50 generated
statements: 18 polynomial identities and 32 actual scalar derivative proofs.
These counts are evidence of checked statements, not
evidence of a new universal complexity bound.

## Abstract engine: actual derivatives, residuals and daggers

An `AbstractStage E F S` has:

1. A primal function from E to F.
2. A continuous linear derivative at each input, with a `HasFDerivAt` proof.
3. An encoder producing the saved residual S.
4. A derivative reconstructed from S, with a proof that reconstruction equals
   the actual derivative at the forward input.

E and F are real complete inner-product spaces for the reverse operation.
This includes suitable finite tensor spaces and can include inputs and every
trainable parameter together. A tensor implementation must supply the proper
space and derivative contracts; ordinary function spaces with a different norm
are not silently assumed to have the desired Euclidean inner product.

The reverse operation is the adjoint of the reconstructed derivative. It is
not the inverse of the activation. At the correct residual it satisfies

```
<dx, reverse(residual, g)> = <D primal(x)[dx], g>
```

Composition follows Mathlib's genuine derivative chain rule. Its adjoint
reverses the order of composition. Branch cotangents add. If two certified
stages have the same entire primal function, uniqueness of derivatives proves
that their entire adjoints, and therefore all operand gradients, are equal.
Including parameters in E avoids proving only the input gradient while missing
the weight gradients. Equal gradients also preserve any deterministic optimizer
update given equal parameters and optimizer state; SGD is a concrete instance.

These are conditional contracts, not proofs that every arbitrary supplied
operation satisfies them. At a ReLU kink, a classical Frechet derivative is
not available; a selected derivative convention such as PyTorch's zero rule
needs a separate nonsmooth contract. Stochastic operations must expose random
state/masks. Side effects, mutation, complex gradients and mixed precision
likewise need their own semantics. The abstract adjoint is noncomputable Lean
mathematics and supplies no efficient matrix representation by itself.

Relevant Mathlib foundations include the
[Frechet derivative chain rule](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/Calculus/FDeriv/Comp.html)
and [continuous-linear adjoints](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/InnerProductSpace/Adjoint.html).
The local pinned source is authoritative for this build, since online Mathlib
documentation can describe a newer revision.

## Concrete nonlinear reduction: polynomial feature contraction

Consider the ordinary no-bias block

```
Z = X @ A                 # X[m,k], A[k,n]
Y = (Z ** degree) @ B     # B[n,p]
```

For degree two, expand the square:

```
Y[row,p] = sum_i sum_l X[row,i]*X[row,l]
                         * sum_j A[i,j]*A[l,j]*B[j,p]
```

Instead of making the wide m-by-n activation, the forward pass can use input
monomial features and contracted coefficients. For k=2 the features are
`x0*x0`, `x0*x1`, `x1*x1`; the middle coefficient has multiplier two.
For degree d, the symmetric feature count is `choose(k+d-1,d)`.
The reference computes

```
T = monomial_features(X)                 # m by s
H = weighted_monomial_features(A.T).T    # s by n
C = H @ B                               # s by p
Y = T @ C
```

All original operands remain trainable. Their pullbacks are

```
dC = T.T @ G
dT = G @ C.T
dB = H.T @ dC
dH = dC @ B.T
dX = monomial_pullback(X, dT)
dA = weighted_monomial_pullback(A.T, dH.T).T
```

The monomial pullback multiplies all factors except the differentiated
occurrence. It uses no division and accumulates repeated-coordinate
contributions, so zero factors do not invalidate the rule.

`square_feature_contraction` proves the forward identity for arbitrary finite
dimensions. `homogeneous_feature_tensor` generalizes the expansion to every
finite degree, using the ordered tensor feature space of size k^d. The Python
reference groups equivalent monomials into the smaller symmetric space.
Generated `ring` proofs certify that grouping and gradient identities for
the two-input/two-hidden instances. Additional generated proofs use Mathlib's
sum, product and power derivative rules to establish the actual derivative of
both native and contracted functions with respect to every scalar operand.
These are not merely comparisons of guessed gradient expressions. Exact rational tests and
PyTorch comparisons qualify the reference at additional dimensions. There is
not yet a general Lean proof of every symmetric grouping adapter or a formal
refinement of the Python executor. The abstract equality theorem describes
how a certified full-primal rewrite preserves all actual derivatives.

This is established polynomial/tensor algebra applied to training. It is not
a claim to have invented a new polynomial identity. Its useful contribution
here is the checked rewrite, full operand-gradient path, and explicit
profitability conditions.

## Cost and storage: conditions that make the reduction useful

The search counts multiplications and additions for both forward GEMMs and
all four gradient GEMMs. It also includes monomial construction, weighted
coefficients, division-free monomial pullbacks and a plain SGD update of A and
B. It does not count only the forward projection or only block products.

For positive dimensions, one ordinary `U[a,b] @ V[b,c]` plus both operand
pullbacks costs, in the specified scalar dot-product model,

```
Multiplications: 3*a*b*c
Additions:       a*c*(b-1) + a*b*(c-1) + b*c*(a-1)
```

The native schedule retains Z and Z^d for backward. Saving only Z would add
recomputation to its second-weight pullback. Its fresh logical residual count
is therefore 2*m*n for d>1. The proposed sufficient residual is T, H and C,
with `s*(m+n+p)` values. Inputs, parameters, parameter gradients, returned
outputs, optimizer state and transient workspace remain additional storage.
Logical counts do not measure allocator peak or the autograd graph.

For degree two and `(m,k,n,p)=(512,2,2048,2)`:

| Quantity | Specified native schedule | Feature circuit |
|---|---:|---:|
| Multiplications | 15,728,640 | 73,216 |
| Additions | 10,475,520 | 48,628 |
| Total including shared SGD update | 26,220,544 | 138,228 |
| Logical fresh retained values | 2,097,152 | 7,686 |

This is approximately 189.7 times fewer modeled scalar arithmetic operations
and 272.9 times fewer logical residual values for this narrow polynomial shape.
These ratios are not GPU speedups, total-RAM ratios or claims about GELU models.
The cost formulas include a particular scalar algorithm, not a lower bound on
the best possible native or feature implementation. GEMM lowering, copies,
dispatch, reductions, hardware instructions and memory traffic are excluded.

`complete_step_cost_guard` proves that adding identical optimizer cost preserves
a proven arithmetic comparison. The cost model must separately establish that
its algorithm counts are accurate. Workspace and allocator peaks need separate
certificates. A passing arithmetic guard alone does not prove a memory guard.

The qualified Python reference accepts degrees one, two and three; the shape
search examines 216 degree-two/three shape candidates. It retains
only candidates with both smaller modeled arithmetic and no larger modeled
residuals, and rejects reference feature expansions above 4,096 features.
The current grid has 73 cost-eligible candidates. Increasing k or degree can
destroy the advantage; it is not a universal nonlinear contraction.

## Numerical guarantees and their boundary

Exact polynomial equality does not imply identical floating bits. Z3 retains
an FP32 intermediate-overflow counterexample. The reference therefore remains
opt-in and tolerance-based, with FP32/FP64 qualification.

New Lean rules propagate local rounded addition and multiplication error to
the expression output. A separate theorem converts a gradient error delta
into an SGD step error bounded by `abs(learning_rate)*delta`. The existing
composition theorem propagates reverse errors using an explicit Lipschitz
bound. These are genuine conditional inequalities, not invented error bounds
for an arbitrary GPU kernel.

[certify_polynomial_roundoff.py](tools/certify_polynomial_roundoff.py) executes
24 bounded scalar traces of the native and feature circuits, including all
operand gradients. Exact rational checks validate the local rounding budget
`u*abs(operation)+eta`, including a subnormal absolute term. It propagates
absolute bounds and checks every node against its exact value. Overflow rejects
the certificate. The recorded traces check 13,824 scalar operations.
This is an offline pointwise oracle, not a uniform-domain certificate, an
interval search over all inputs, or a refinement of PyTorch/Metal GEMM reduction
orders. It is too expensive to be the proposed production implementation.

## Automated clue and theorem search

[discover_backprop.py](tools/discover_backprop.py) implements this bounded loop:

1. Derive degree-two/three polynomial coefficients with SymPy.
   A bounded basis-refinement loop first tries constant and linear feature
   spaces for square/cubic examples. Inconsistent sample constraints reject
   them; the loop enriches the basis until it finds coefficients and then asks
   Z3 to refute the resulting identity over every real scalar input. Sample
   agreement alone never accepts the rule. This is a minimal-degree result
   within that finite univariate grammar, not a global algorithm lower bound.
2. Construct native and contracted expressions and all symbolic operand
   derivatives for a concrete finite instance.
3. Ask Z3 to refute equality. Accept these symbolic checks only on `unsat`;
   timeout or unknown does not count as a proof.
4. Emit 18 Lean `ring` identity statements and 32 actual `HasDerivAt` statements
   using Mathlib calculus rules, with proof terms checked by Lean's kernel.
5. Retain counterexamples to inappropriate linear contraction, output-only
   square differentiation, zero-guarded division and bitwise reassociation.
6. Enumerate a fixed shape grid and reject cost, residual or feature-budget
   failures. SymPy also rejects SiLU, GELU and tanh from this exact polynomial
   grammar; replacing them by polynomials would change the model and needs
   an approximation contract.
7. Qualify retained mathematics with exact and floating tests. No candidate is
   automatically dispatched into a model or accepted as a GPU optimization.

The tools automate a reproducible family of hypotheses and proofs. They do not
search all mathematics, discover arbitrary theorems independently, or establish
global optimality. Reports distinguish symbolic success awaiting Lean from
success accepted by the full verifier.

## Qualification and reproduction

From the project root:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_abstract_foundation.py
```

The runner regenerates the discovered identities and complete Lean audit,
builds all project Lean modules, checks theorem dependencies, tests the new
reference and numerical traces, runs the CPU regression suite, and rechecks
the 100-statement Rocq foundation. It fails closed and records source hashes.
Lean's allowed foundational dependencies are `propext`, `Classical.choice`
and `Quot.sound`; no custom axioms, unfinished proofs or `sorryAx` are accepted.
Kernel validation certifies the stated theorem relative to its definitions,
premises and allowed foundations, not the truth of a performance interpretation.
See [Lean proof validation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

The new reference suite checks 72 exact Fraction cases and 96 floating cases
with all three operand gradients, including zeros, repeated factors,
noncontiguous matrices and small inputs. Two cases compare higher derivatives
with respect to all operands. Four degree/optimizer configurations each run
four SGD or AdamW steps. Ten invalid reference requests are rejected.
Four additional graph cases cover sequential contracted blocks with tanh and
residual addition, and shared-weight branches followed by affine LayerNorm.
They check every input/parameter gradient in FP32/FP64 and higher derivatives
for both graphs in FP64. These are correctness tests, not whole-graph cost or
performance proofs.
Those optimizer tests additionally update the input tensor to exercise its
gradient; the cost example treats only A and B as optimizer parameters.

A single CPU observation at `(64,2,128,2)` measured native fresh saved storage
of 131,072 bytes and reference storage of 18,600 bytes. The reference took
approximately 0.00197 seconds versus 0.000411 seconds for native execution in
that preliminary observation. This is less retained fresh storage but slower
execution, not a statistically qualified benchmark. The verifier records a
fresh observation; consult its report for current values. Native PyTorch often
already has strong optimizations that a Python mathematical oracle lacks.

## How this becomes useful to OjasX

For an eligible polynomial block, a backend can keep the original developer
model and trainable parameters, choose the certified feature representation,
and implement its monomial construction and all VJPs. It must also qualify
rounding, workspace, parameter versions, aliasing and whole-step behavior.
Automatic selection remains disabled here. A generalized compiler path needs
a graph matcher and appropriately costed materialized operators.

For GELU, SiLU, attention and arbitrary future primitives, the abstract engine
still supplies composition and verification obligations. This particular
feature contraction does not reduce their arithmetic. Other supported rules
include linear-chain contraction, compact normalization, activation residuals,
rank-seven multiplication and structured pullbacks, each with its own guards.
Adding an unfamiliar operation requires an actual derivative, sufficient
residual, numerical contract and an independently justified cost comparison.

We have made the foundation more precise and added a useful nonlinear
optimization family. We have not closed the research gap of simultaneously
reducing arithmetic, storage and runtime across most modern models or every
future PyTorch workload. That remains an explicit result to establish, not an
assumption supplied by category theory or a stronger proof assistant.
