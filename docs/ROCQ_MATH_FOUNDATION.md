# OjasX: Rocq mathematical foundation and extracted reference engine

Date: 7 October 2026.

This document describes the initial 55-statement Rocq foundation. It is extended
by [backend mathematics v1](BACKEND_MATH_FOUNDATION_V1.md), whose current audit has
89 statements and ten Z3 checks. The linked verification report tracks that newer
source version; the initial counts below describe the original scope.

This extension installs Rocq 9.2 locally and adds checked, executable mathematics
for operation-reducing backpropagation schedules. It is an exact-rational
reference engine and a source of compiler rewrite specifications. It is not a
replacement GPU backend or evidence that arbitrary training can run at half cost.

## What is installed and organized

- Rocq core and standard library **9.2.0**, OCaml **4.14.2**, opam **2.6.0**.
- Isolated opam root: `.tools/opam-root`; switch: `ojasx-rocq-9.2`.
- Local GMP 6.3.0 and pkg-config 0.29.2 under `.tools/rocq-deps`.
- No global package installation or shell-profile modification.
- `tools/rocq.sh` selects this environment. On this machine it selects the
  installed macOS 15.4 SDK because the default macOS 27 SDK was incompatible with
  the linker used by the OCaml build. `SDKROOT` can explicitly override it.
- `rocq/opam-switch.export` records the installed package versions;
  `reports/rocq-installation.json` records the bootstrap checksums and location.

The installer and dependencies were obtained from official upstream sources.
Installation follows the [Rocq opam route](https://rocq-prover.org/docs/using-opam).
The local opam binary and downloaded GMP/pkg-config archives were SHA-256 checked.
Opam checked its package downloads. These checks establish source identity, not
that every dependency is formally verified.

## What the new proofs establish

There are **55 theorem/lemma statements** in five source modules. Every statement
is compiled, enumerated in `Audit.v`, and independently rechecked with `rocq check`.
The assumption audit reports **no axioms** for any of these statements. This is
separate from the earlier Lean proof set; no automatic Lean-to-Rocq equivalence is
claimed.

| Module | Mathematical result | Practical role |
|---|---|---|
| `rocq/Product.v` | Division-free product pullback; exact first-order polynomial identity; squared-loss gradient; instrumented linear multiplication count | Small executable proof of gradient computation, including zero factors |
| `rocq/Matrix.v` | Finite-sum associativity, transpose, first-order expansion and both adjoint identities; preservation of all linear-pair gradients | Correctness specification for joint forward/backward contraction |
| `rocq/Nonlinear.v` | One-bit ReLU derivative information, fixed-region rule, shared-mask forward and all-gradient contractions; impossibility of recovering square derivatives from outputs alone | Identifies when nonlinear memory/computation reductions are legal |
| `rocq/Planner.v` | Joint schedule cost identities; work and retained-intermediate guards; forward, all-gradient and parameter-SGD preservation | Exact arithmetic planner and backend contracts |
| `rocq/Optimizer.v` | Fast/reference SGD equality, repeated trajectory equality, length preservation and bounded safeguarded loss nonincrease | Verified optimizer reference for the squared-product objective |

All arithmetic values are canonical rationals (`Qc`). Dimension costs use
unbounded binary natural numbers (`N`). The proofs cover arbitrary finite
dimensions and rational inputs within their stated hypotheses, including zero
dimensions. They do not cover rounded GPU arithmetic merely by using the same
formulas.

## The main useful reduction: replace wide intermediates with narrow cores

Consider two consecutive linear layers, with row-vector convention:

\[
X\in\mathbb Q^{m\times k},\quad A\in\mathbb Q^{k\times n},\quad
B\in\mathbb Q^{n\times p},\quad Y=(XA)B.
\]

Let \(G\in\mathbb Q^{m\times p}\) be any upstream gradient. The ordinary schedule
forms the wide intermediate \(H=XA\), then a wide backward intermediate
\(E=GB^T\). Its gradients are

\[
dX=EA^T,\qquad dA=X^TE,\qquad dB=H^TG.
\]

The alternative forms two narrow cores:

\[
S=AB,\qquad T=X^TG.
\]

Then it computes

\[
Y=XS,\qquad dX=GS^T,\qquad dA=TB^T,\qquad dB=A^TT.
\]

Rocq proves equality of the output and **every** gradient. The separate adjoint
proofs connect the matrix rules to the first-order expansion, rather than only
showing that two unvalidated formulas match. Both original parameter matrices
remain trainable. The extracted planner also preserves their exact SGD updates.

This uses associativity, finite sums and adjoints. In the categorical language
used by the existing project, transposes implement the relevant linear daggers.
The arithmetic saving comes from choosing cheaper contraction dimensions and
sharing cores; it does not follow from the dagger notation alone.

Counting each multiply-accumulate as two leading GEMM operations:

\[
C_{native}=6mkn+6mnp,\qquad C_{contracted}=6pk(m+n).
\]

`Planner.v` records the six GEMM shapes for each joint schedule and proves these
cost identities. The planner chooses contraction only when

\[
C_{contracted}\leq C_{native}\quad\text{and}\quad kp\leq mn.
\]

The second guard compares the retained extra intermediate `S` with `H`. It does
not include common inputs/parameters, optimizer state, output gradients, temporary
GEMM workspaces, allocator overhead or total peak memory.

For the tested planner example \((m,k,n,p)=(512,32,2048,32)\):

| Quantity | Native schedule | Contracted schedule |
|---|---:|---:|
| Leading joint GEMM operations | 402,653,184 | 15,728,640 |
| Extra retained intermediate elements | 1,048,576 | 1,024 |

That is **25.6× fewer modeled leading operations**, or **96.09375% less**, and
**99.90234375% fewer elements in this particular saved intermediate**. These are
shape-specific mathematical counts, not a newly measured runtime or whole-model
memory result. The savings require executing each scheduled node once and reusing
it. Unfavorable shapes use the native schedule.

## Nonlinear operations: the exact boundary

For ReLU, the pullback only needs the boolean `x > 0` under the chosen convention
that the derivative at zero is zero. `relu_one_bit_sufficient` proves that storing
this boolean preserves the pullback. It does not claim differentiability at zero
or that OCaml's boolean representation physically packs one bit.

If every row in a batch has the **same hidden-unit mask**, the mask can be written
as one diagonal matrix \(D\). Then

\[
\operatorname{ReLU}(XA)B=(XAD)B=X((AD)B)
\]

at inputs with that mask, and the local pullbacks are

\[
dX=G((AD)B)^T,\quad dA=(TB^T)D,\quad dB=(AD)^TT.
\]

Rocq checks the shared-mask algebra and all three pullbacks. The backend must
establish that the actual ReLU masks agree across the relevant rows. Arbitrary
input-dependent masks are not interchangeable with one shared mask. Mask
detection or grouping also costs work, which the linear-pair cost model does not
include. This is not an automatic optimization for GELU or SiLU.

The square-activation counterexample gives a complementary storage limit:
\(1^2=(-1)^2\), but the derivatives are respectively \(2\) and \(-2\). Rocq proves
there is no function of the squared output alone that recovers both derivatives.
A valid backend must retain sufficient information, recompute it, or explicitly
change the numerical/algorithmic contract.

## Product gradients and descent: executable verification examples

For \(P(x)=\prod_i x_i\), the gradient is the product of all factors except each
selected factor. The extracted prefix/suffix pullback computes it without
division, so one or multiple zero factors are handled correctly.

The instrumented fast schedule uses exactly `3*n` rational multiplications. The
recursive naive gradient definition has the quadratic model `n*(n-1)`. The
comparison is nonincreasing for `n >= 4`. **Conventional reverse-mode autodiff
already computes product gradients in linear work**: this result validates the
extraction and proof method, not a quadratic-to-linear speedup over PyTorch.
Furthermore rational multiplication has variable bit cost, so this count is not
an OCaml elapsed-time prediction.

For the objective \((P(x)-target)^2\), the files prove its first-order coefficient,
the exact fast/reference gradient equality, one SGD-step equality, and equality
of repeated SGD trajectories. A bounded safeguard tries a candidate learning
rate, halves it if the loss increases, and returns the original parameters if its
trial budget is exhausted. The accepted result never increases this **exact
rational** loss. It may make no progress. There is no theorem here of universal
convergence, fewer training iterations, or improved model quality.

## Extraction, execution and the trust boundary

`rocq/Extract.v` produces `rocq/extracted/ojasx_math.ml` and its interface.
It retains arbitrary-precision inductive integer representations and exact
rational arithmetic. There are no extraction overrides replacing those values
with overflowing machine integers or floating point.

The handwritten `cli.ml` adapter uses Zarith for decimal input/output conversion,
normalizes rationals, checks dimensions and rejects malformed inputs. It compiles
with the extracted library into `rocq/extracted/ojasx_math_cli`. It exposes the
product, optimizer, ReLU, planner and finite-matrix reference operations.

The functional matrix representation is a **semantic oracle**. Nested matrix
functions can reevaluate intermediate entries. It does not materialize a fast
GEMM DAG, and the cost theorems do not describe this oracle's actual execution.
Implementing and verifying materialized scheduling is a separate backend step.

Rocq's [extraction documentation](https://rocq-prover.org/doc/v9.2/refman/addendum/extraction.html)
describes exporting executable functional definitions. The guarantee remains
scoped to the proved definitions and specifications. The extractor, OCaml
compiler/runtime, handwritten adapter and operating system remain in the trusted
execution path. Resource exhaustion is possible. This is not a guarantee of
“zero runtime bugs” in a complete backend.

## Validation and counterexample refinement

The authoritative fresh report is `reports/rocq-verification.json`, with source
hashes. Earlier Lean/PyTorch/GPU reports remain historical evidence for their own
checked source versions; they were not rerun as part of this math-only extension.

The exact engine suite includes zero and repeated-zero factors, empty vectors,
large integers, rational inputs, zero-size matrices, favorable and unfavorable
plans, both parameter updates, repeated training steps, rejected learning rates,
and malformed adapter inputs. Python `Fraction` is an independent oracle.
PyTorch comparisons use CPU FP64 and explicit `1e-11` relative/absolute tolerances;
they are sampled validation, not a floating-point equivalence proof.

`tools/verify_rocq_z3.py` checks three encoded identities/guards and retains three
expected counterexamples:

1. Unconditional contraction through ReLU fails when masks differ.
2. Square outputs alone cannot determine their derivatives.
3. FP32 reassociation can differ even for finite inputs; the concrete example
   overflows on one intermediate path.

These counterexamples constrain the implemented theorem scope. They are not
treated as failed tests or suppressed. General finite-matrix correctness is
proved in Rocq; the Z3 polynomial identities are only scalar checks.

## Use and reproduce

From the `OjasX_rebuilt` directory:

```sh
./tools/rocq.sh rocq --version
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_rocq.py
```

The runner compiles all modules, audits every lemma/theorem, independently checks
the compiled proof objects, extracts OCaml, builds the CLI, runs the exact and
PyTorch oracles, runs Z3, and records hashes. A failure stops the run. It does not
need network access after installation. The Python environment needs PyTorch and
z3-solver; the supplied environment already contains them.

Example planner query:

```sh
printf '%s\n' 'plan|512|32|2048|32' | rocq/extracted/ojasx_math_cli
```

Example exact gradient query, including a zero factor:

```sh
printf '%s\n' 'product|2,0,3|1' | rocq/extracted/ojasx_math_cli
```

Its gradient is `[0,6,0]`. The adapter also accepts `step|xs|target|rate|fuel`,
`relu|xs|upstream`, and `matrix|m,k,n,p|X|A|B|G`. Vectors/matrices use comma-separated
integers or `numerator/positive_denominator`; matrices are row-major. The matrix
query returns the forward result, all gradients, and parameter updates at rate
`1/10`. The adapter deliberately limits reference matrix dimensions to 12, vector
length to 512 and safeguard trials to 16. The mathematical theorems have no such
size limit. The binary-integer planner accepts much larger dimensions.

To recreate the installation on a matching Apple ARM machine, use the pinned
official opam binary recorded in the installation manifest, initialize an isolated
root, run `tools/bootstrap_rocq_deps.py`, then install the versions in
`rocq/opam-switch.export`. Export `SDKROOT`, `PATH`, `PKG_CONFIG_PATH`, `CFLAGS`
and `LDFLAGS` for the local dependencies as recorded in the installation logs.
This installation is host-specific; these paths are not a portable Linux or
Windows installer.

## How a later backend uses these results

1. Match a supported operation graph while preserving original parameters and
   optimizer ownership.
2. Establish the rewrite's shape and nonlinear-mask hypotheses.
3. Ask the exact planner for the schedule and cost comparison.
4. Materialize the selected DAG once per node; reuse `S` and `T`, with explicit
   lifetime management. This is not yet proved by the functional reference.
5. For real hardware, establish an explicit floating-point error contract and
   test output, input gradients, every parameter gradient and optimizer updates.
   A strict bit-preserving mode must reject unproved reassociations.
6. Dispatch only after workload-specific runtime and memory qualification;
   retain the ordinary path when the conditions fail.

Developers can eventually keep their usual model and optimizer interface. These
proofs make selected backend rewrites precise and reviewable. They do not yet
certify the complete compiler, convert every nonlinear layer into a linear one,
or establish that one GPU replaces three or four across arbitrary training jobs.
