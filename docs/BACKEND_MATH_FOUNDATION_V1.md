# OjasX backend mathematics — scoped foundation v1

This document records v1. For the subsequent normalization and indexing fixes,
100-statement audit and new edge qualification, see
[scoped refinement 1.1](FOUNDATION_REFINEMENT.md).

7 October 2026. This is the consolidated mathematical contract for the next
OjasX backend implementation. It extends the existing Lean and Rocq work; it
does not certify a complete compiler, arbitrary future primitives, or large
speedups across most models.

## Result of this extension

The Rocq development now contains **89 audited theorem/lemma statements**, up
from 55, plus executable contract constructors. The kernel recheck passed, and
every audited statement is closed under the global context. The current runner
also passed:

- The original 358 extracted reference-engine cases and 315 sampled PyTorch FP64 comparisons.
- **249 additional exact extracted-engine cases**, using independent Python
  rational arithmetic, including operation-count instrumentation.
- **18 structured-derivative cases**: nine operations in FP32 and FP64, including
  directional derivatives, pullbacks, cotangent accumulation and derivatives of
  pullbacks.
- **12 representative workload families**, including checks of all supplied
  differentiable inputs/parameters, JVP/VJP pairing, finite differences and
  Hessian-vector products.
- Eight matrix shape/dtype cases comparing native, Strassen, Winograd and the
  existing streamed implementation; all forward/input-gradient/weight-gradient
  numerical checks passed. Three additional higher-order matrix checks passed.
- **10 scoped Z3 checks**: five negated identities/guards proved unsatisfiable and
  five expected counterexamples retained.

The reproducible evidence is in [rocq-verification.json](reports/rocq-verification.json)
and [broad-foundation.json](reports/broad-foundation.json). These results establish
the listed coverage, not a percentage of all ML models.

## The common mathematical interface

Most differentiable tensor programs combine linear maps, bilinear maps,
pointwise nonlinearities, reductions, normalization, indexing, branching and
repeated composition. Their local backward operation consumes an upstream
cotangent and applies the adjoint of the local derivative.

The new `Contracts.v` makes that interface explicit:

\[
\langle J_x\,dx,g\rangle=\langle dx,J_x^*g\rangle.
\]

It constructs composed contracts and branching contracts, proves reverse-order
pullback composition, supports finite repetition, and establishes cotangent
accumulation under the stated linearity and nondegeneracy conditions. A residual
replacement can be substituted into a larger backward calculation when it
reproduces the same local pullback.

**An adjoint contract alone does not prove that a supplied map is the derivative
of a primal function.** Each primitive still needs that derivative theorem or an
explicitly unverified/tested status. For matrix multiplication, the Rocq
first-order expansion supplies this connection. For the newly tested analytic
coefficient builders, Rocq proves the supplied-coefficient algebra; the Python
analytic formulas are tested against PyTorch and higher derivatives. This
distinction is part of the backend contract.

This approach fits the compositional JVP/VJP interface documented by
[PyTorch's function transforms](https://docs.pytorch.org/docs/stable/func.api.html).
The installed test runtime is PyTorch **2.8.0**, irrespective of the version
currently served by the online documentation.

## One derivative representation covering several important operations

The new structured family is

\[
J=\operatorname{diag}(d)+UV^T,
\quad
J^Tg=d\odot g+V(U^Tg).
\]

`Structured.v` proves the adjoint identity, equality with the corresponding dense
operator, additivity in the cotangent, the softmax rank-one specialization and
the zero-sum softmax pullback property. `Foundation.v` connects these rules with
the composition contracts and matrix adjoints.

| Operation | Structure of its input derivative | Additional obligation |
|---|---|---|
| ReLU, SiLU, exact GELU, sigmoid, tanh, square | Diagonal: rank zero correction | Compute the actual slope; respect the chosen ReLU convention at zero |
| Softmax | `diag(p) - p p^T` | `p` must be the actual probability vector |
| RMSNorm, including affine gain | Diagonal plus rank one | Use the actual inverse RMS, gain and epsilon |
| LayerNorm, including affine gain | Diagonal plus rank two | Use the actual mean, inverse standard deviation, gain and epsilon |
| Matmul | Separate linear adjoints for each operand | Preserve every required parameter gradient |
| Branches and auxiliary losses | Sum cotangents before a common pullback | Same primal evaluation and the intended summed objective |

The factored derivative can be applied with a modeled `4*n*r + 2*n` scalar
operations, compared with `2*n*n` for a supplied explicit dense Jacobian-vector
product. Rocq proves the cost inequality when `n >= 2*r + 1`.

The cost model assumes that the small projected vector is computed once and
reused. The extracted functional reference may recompute entries; it is a
semantic oracle. The PyTorch reference computes the projection once. Neither the
factor-generation cost nor transcendental function evaluation is included in
that particular model.

**Native PyTorch already exploits compact rules for these standard operations.**
This is a broad verified representation and extension interface, not a claim
that replacing native softmax with our reference suddenly saves quadratic work.
Likewise, literally storing all `d`, `U`, and `V` can consume more memory than a
specialized native residual. A backend should represent constants, broadcasts
and shared statistics implicitly when possible.

## Matrix multiplication investigation and refinement

The new `RankSeven.v` verifies both the standard Strassen construction and the
Winograd rearrangement. It proves scalar correctness, arbitrary finite block
correctness, and both operand gradients. The block proof preserves operand
order; it never assumes that matrices commute.

These are established algorithms. The contribution here is their checked
formulation, extraction, cost conditions and qualification for OjasX. See
[Strassen's original paper](https://courses.grainger.illinois.edu/cs357/su2013/Honors/Strassen.pdf)
and the research on [memory-efficient Strassen–Winograd schedules](https://arxiv.org/abs/0707.2347).

For one two-by-two block level:

| Schedule | Products per forward multiplication | Additions/subtractions per forward multiplication | Forward plus both gradient products |
|---|---:|---:|---|
| Conventional | 8 | 4 | `24 M + 12 A` |
| Strassen | 7 | 18 | `21 M + 54 A` |
| Winograd | 7 | 15 | `21 M + 45 A` |

Here `M` and `A` are supplied uniform block multiplication and addition costs;
the joint comparison assumes equally shaped square blocks. Rocq proves:

\[
21M+54A\leq24M+12A\iff M\geq14A,
\]
\[
21M+45A\leq24M+12A\iff M\geq11A.
\]

Thus the refinement reduces the modeled addition count by **16.7% relative to
the standard Strassen joint schedule**, with the same **12.5% reduction in block
multiplications relative to the conventional schedule**. These percentages
describe arithmetic counts, not measured throughput.

The extracted programs were instrumented at their arithmetic callbacks:
Strassen produced 21 multiplications and 54 additions/subtractions for the
joint scalar case; Winograd produced 21 and 45. Both matched independent exact
matrix and gradient calculations.

### Actual qualification result

All three candidate Python implementations were **slower than native matmul**
in the eight tested CPU shape/dtype cases. In the recorded final run, Winograd's
native-time/candidate-time ratios were about **0.10–0.30**, and the existing
streamed variant's were about **0.10–0.43**. A ratio below one means the candidate
is slower. The report contains exact values; timing can vary by run.

For the FP64 256-by-256 example, additional fresh storage retained by autograd
was:

| Path | Extra retained bytes beyond supplied operands/upstream |
|---|---:|
| Native | 0 |
| Strassen reference | 1,310,720 |
| Winograd reference | 1,048,576 |
| Existing streamed path | 0 |

Zero here means no **additional fresh saved storage** in this audit. It does not
mean zero operands, zero scratch space, or zero training memory. The streamed
path's existing custom backward and higher-order fallback have a different
execution schedule from the extracted three-product reference. They were
numerically checked here; this is not a new Rocq refinement proof of that code.

**Decision: keep these multiplication candidates opt-in.** Their mathematics
saves products; this CPU realization does not qualify as a runtime improvement.
The new budget guard also checks supplied scratch estimates, but deriving those
estimates from a real allocator/schedule remains a backend obligation.

## What the broad workload tests cover

The suite tests these twelve families:

1. Consecutive linear layers.
2. GELU MLP.
3. SwiGLU.
4. Convolution with activation and trainable bias.
5. Softmax attention.
6. Affine LayerNorm.
7. Affine RMSNorm.
8. A recurrent block with shared weights.
9. Embedding lookup with repeated indices.
10. A residual branch with a fixed dropout mask.
11. A research-style polynomial gate.
12. A finite unrolled ODE-style update.

These tests exercise derivative composition and all supplied trainable inputs,
including directional finite differences and second-order directional checks.
They use PyTorch's functional transforms and native primitives. They do not
constitute an extracted Rocq executor for all twelve models, nor do they measure
whole-model acceleration from the new foundation.

The existing conditional linear/RMS-core and reversible results remain relevant
optimization candidates. Standard attention implementations can already use
memory-efficient methods: [FlashAttention](https://arxiv.org/abs/2205.14135)
reduces memory traffic, and [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
selects among available implementations. Those are appropriate baselines; a
comparison against a deliberately materialized quadratic attention matrix alone
would not establish a new advantage over optimized PyTorch.

## Backend obligations and future model support

The contract manifest is [foundation_manifest.json](foundation_manifest.json).
Each backend primitive needs:

1. Shape, dtype, device, aliasing and randomness semantics.
2. A valid primal derivative rule and its adjoint contract.
3. Sufficient residual information, plus ownership/lifetime rules.
4. A numerical contract: exact algebra, measured tolerance or proved error bound.
5. A cost model that includes added arithmetic, recomputation and scratch space.
6. A qualified fallback when an optimization's premises fail.

A new model assembled from supported primitives can reuse the composition
structure. A genuinely new primitive needs a new derivative contract. Discrete
sampling, changed randomness, implicit solvers, dynamic control flow, mixed
precision and distributed execution require their own semantics and
qualification. The fixed-mask test does not certify arbitrary stochastic
programs.

Aggregating several losses can share a pullback when their cotangents belong to
the same primal execution and the desired result is their summed gradient.
It does not produce every individual gradient for free, and it does not merge
successive optimizer steps whose parameters have changed.

Memory and arithmetic are separate objectives. Recomputing a discarded
activation may save storage while increasing arithmetic. A blanket “minimum
memory and minimum work” claim is not established by these local theorems.

## Organization and use

| Artifact | Purpose |
|---|---|
| `rocq/Contracts.v` | Composed, branched and repeated adjoint contracts |
| `rocq/Structured.v` | Diagonal/low-rank derivatives and cost condition |
| `rocq/Foundation.v` | Connections to matrix and structured primitive rules |
| `rocq/RankSeven.v` | Strassen/Winograd correctness, gradients and guarded cost models |
| `rocq/extracted/ojasx_foundation.ml` | Generated exact-rational reference |
| `rocq/extracted/foundation_cli.ml` | Handwritten, tested input/output adapter |
| `ojasx2/math_foundation.py` | Opt-in PyTorch coefficient and multiplication references |
| `tools/test_broad_foundation.py` | Independent exact oracles, workload and matrix qualification |

Run the complete Rocq verification from the project root:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_rocq.py
```

Use a derivative reference explicitly:

```python
from ojasx2.math_foundation import layernorm_input_derivative

local = layernorm_input_derivative(x, gamma, eps=1e-5)
dx = local.pull(upstream)
```

This computes the LayerNorm **input** pullback. Gain/bias gradients are separate;
it is not a drop-in custom autograd layer. The reference operations do not
automatically replace the developer's model or enable compiler rewrites.

This version provides a broader mathematical basis and explicit extension
contracts. Large improvements across most complete models, global minimum
memory, floating-point proof refinement and an optimized native executor remain
open requirements. They must be demonstrated, not inferred from the number of
checked theorems.
