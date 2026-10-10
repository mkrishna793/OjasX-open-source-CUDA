# OjasX mathematical compiler: closed representations and checked conditions

This version implements a bounded mathematical compiler research prototype. It translates particular nonlinear computations into a compact linear operator space, composes the operators there, decodes the result, and differentiates with respect to every original parameter. The successful new example preserves the nonlinear primal over real arithmetic; it does not replace GELU/SiLU with an approximation.

The useful result is specific: a trainable triangular nonlinear recurrence has a fixed three-observable representation. Both shared weights and separate original weights at every stage are supported. GPU tests cover forward values, all input/parameter gradients, four SGD steps, higher derivatives, and selected edge cases. Independent repeated full-step timing and retained-storage measurements are in [the repeat report](reports/koopman-repeat.json).

This is not a compiler that has found huge reductions for every modern ML operation. Dense GELU/SiLU/SwiGLU layers do not acquire a cheap exact representation in the tested chart grammar. Unsupported representations retain native execution. The prototype requires explicit `accuracy="algebraic"`; strict mode retains the native computation.

## The advanced mathematical structure

The construction uses nonlinear conjugacy/semiconjugacy, finite Koopman-invariant observable spaces, and composition of parameterized linear actions. The mathematical distinction is whether a representation is **closed**: after a stage, every represented observable must still be expressible using the same finite basis. Its width must remain small as the computation grows.

[Koopman invariant-subspace research](https://arxiv.org/abs/1510.03007) studies this exact finite-dimensional representation problem and documents restrictions on some proposed global linearizations. The compiler here searches a bounded grammar and proves specific closed cases. It makes no claim to solve the general finite-embedding problem. [Equality saturation research](https://arxiv.org/abs/2004.03082) supplies relevant compiler background; this prototype uses explicit candidate enumeration and counterexample-guided synthesis, not an implementation of the `egg` library.

The abstract compiler contract is:

1. Encode the original state into observables `E(x)`.
2. Find a compact operator `K(theta)` satisfying `E(f_theta(x)) = K(theta) E(x)` on a stated domain.
3. Prove the domain is preserved and decoding recovers the requested state.
4. Compose the operators in the original stage order.
5. Differentiate the whole encoded/operator/decoded expression, including every original parameter and the encoder/decoder derivatives.
6. Check representation construction, parameter-gradient recovery, residual storage, numerical error and complete-step GPU execution.

Lean and Rocq prove general sequence-composition contracts. The Lean contract supports explicit domain preservation and distinct parameters at each stage. Conditions are theorem hypotheses backed by supplied witnesses or proofs. No new foundational axioms are added.

## A nonlinear computation that really stays compact

Consider a stage with original trainable parameters `a`, `b`, `c`:

\[
 f_{a,b,c}(x,y)=(a x,\;b y+c x^2).
\]

The square makes the stage nonlinear. Encode the state as

\[
 E(x,y)=(x,y,x^2).
\]

The encoded update is exactly

\[
 E(f_{a,b,c}(x,y))=
 \begin{pmatrix}a&0&0\\0&b&c\\0&0&a^2\end{pmatrix}E(x,y).
\]

The third observable stays quadratic because the first state evolves independently as `a*x`. There is no feedback that produces degree four, eight, or a growing sequence of new observables. Decoding simply selects the first two coordinates. It is valid at zero and for negative states; no division or logarithm is needed.

For distinct trainable parameters at every stage,

\[
 f_{\theta_{L-1}}\circ\cdots\circ f_{\theta_0}(x)
 =D\big(K_{\theta_{L-1}}\cdots K_{\theta_0}E(x)\big).
\]

The implementation forms the small matrices and uses balanced ordered composition. It does not sort them, commute them, or merge the original parameter tensors into a new trainable parameter. If parameters are shared, it uses matrix exponentiation. The resulting full operator is applied to the batch once.

The original all-parameter adjoint for one stage is also generated and checked. For upstream `(u,v)`, it is

\[
 d_x=a u+2c x v,\quad d_y=bv,\quad
 d_a=xu,\quad d_b=yv,\quad d_c=x^2v.
\]

The compiler differentiates through `a²` inside the lifted matrix. Treating the composed operator as an independent parameter would lose these gradients; this implementation retains the original `theta` tensors and their dependencies.

For large batches, the wide batch is no longer propagated and retained at each stage. Transfer construction and its derivative graph have size tied to small per-feature matrices. With independent stage weights, the operator work is proportional to the number of stages times feature width, plus the batch contraction; the constant observable width is three. With shared weights, matrix exponentiation reduces the number of small operator compositions further. These are representation-level algorithmic observations, not a formally certified Metal instruction count.

## Tool-driven search and counterexamples

[tools/discover_coordinate_compiler.py](tools/discover_coordinate_compiler.py) enumerates 36 operation/chart pairs using identity, log, reciprocal and logit coordinates. It tests symbolic affine closure for affine, scaling, homogeneous powers, shifted squares, exponential, sigmoid, SiLU and GELU stages. Six pairs have symbolic affine closure in that grammar. Symbolic simplification alone does not certify a chart's domain or GPU implementation.

The Koopman example begins with a rejected state-only model. Z3 supplies a counterexample; adding `x²` permits sample-based coefficient synthesis, followed by universal SMT refutation of the negated identity. SymPy then derives the parameterized matrix and all parameter cotangents. It emits Lean and Rocq proof sources, which are kernel checked.

Sixteen new Z3 checks cover this search, including:

- A square-only chart loses sign information; positivity refines its injectivity condition.
- Positive domains are nonempty; contradictory assumptions are rejected.
- Output alone does not generally recover original parameter gradients. For `f(x,a,b)=a*x+b` at `a=0,b=1`, all inputs give output one but the weight derivative is `x`. The positive region of ReLU has the same example.
- Parameterized affine actions compose associatively and preserve all parameter cotangents.
- The synthesized triangular invariant operator and its full adjoint satisfy the tested identities.

For repeated squaring, monomial bases through degree six fail closure: the highest degree `d` generates degree `2d`. This rejects those finite bases, not every conceivable representation. New observables must earn their place by closure, correctness and cost checks.

## A second representation that remains limited

Positive homogeneous stages have a log conjugacy:

\[
 \log((a x)^p)=p\log x+p\log a.
\]

Their nonlinear chain becomes a composition of affine actions in log coordinates. Lean proves the exact positive-domain identity and actual encoder/decoder derivatives. Every original stage weight gradient is computed from the decoded output and its exponent.

The first GPU experiment found parameter-gradient pointwise tolerance failures in deeper square/cube chains, even with small relative L2 errors. The initial report is preserved in [coordinate-gpu-initial.json](reports/coordinate-gpu-initial.json). These are numerical counterexamples to the chosen FP32 contract, not failures of the real-arithmetic conjugacy.

The revised log prototype limits the total exponent to 16 and retains native execution for the rejected cases. Its final six-case GPU qualification and nine selected edge checks pass, with higher-derivative checks against the native nonlinear map. Active log candidates remain slower, so this route is research evidence and a counterexample-driven refinement, not a recommended acceleration.

## GPU qualification and boundaries

The Koopman tests use MPS FP32 with CPU ML fallback disabled. Five main cases include 8/32/64 shared stages and 32/64 separate-parameter stages. The small eight-stage case does not demonstrate a speed improvement. Larger cases do. Three independent seed repeats qualify the separate-parameter cases using all original gradients and complete SGD-step timings. The reports retain samples, accuracy tolerances and source hashes.

In the recorded larger 64-stage separate-parameter example, fresh retained activation storage decreases from about 11.88 MiB to 0.327 MiB, approximately **97.25% less**. Shared-parameter storage is smaller still. These are autograd-retained allocations, excluding input/parameter storage. Total GPU peak, driver workspace, energy and full-model memory are not certified.

Ten selected Koopman edge checks cover zero states, negative state/parameters, equal eigenvalues, zero coupling, NaN/domain rejection, large-parameter rejection, depth rejection, strict mode, odd numbers of separate stages and noncontiguous inputs. Higher derivatives are compared against the native recurrence. Matrix multiplication avoids a closed-form division by `a²-b`, so equal eigenvalues do not create a singular formula.

The current runtime restricts algebraic Koopman execution to 4–64 stages, finite bounded states/parameters, compatible MPS FP32 tensors and no autocast. It falls back outside these numerical/representation conditions. Tests are selected coverage, not a proof for every FP32 input. Real algebraic equivalence does not imply bitwise equivalence; pointwise and relative error checks remain necessary. Long-training convergence is not established by four SGD steps.

## Existing linear support and unsupported nonlinear layers

The existing dense linear-chain and matrix-chain planners remain the linear part of the foundation. Their reassociation contracts and all original parameter gradients are proved in the established modules. Algebraic FP32 execution changes rounding; the earlier unscaled linear stress failure remains recorded.

The new coordinate experiment handles a structured polynomial recurrence. It does **not** turn dense GELU, SiLU, SwiGLU, convolution, attention, or an arbitrary researcher-defined model into a cheap finite linear system. Such a stage needs an explicit compact representation with closure and derivative proofs. The tested chart grammar finds no usable general exact closed representation for those dense activations.

A future research model can be supported by providing a suitable encoding and stage operator, proving closure/domain/decoding and all-parameter derivatives, then qualifying the complete numerical execution. The framework is general; efficient representation discovery remains conditional.

## Using the prototype

The implementation is [ojasx2/coordinate_compiler.py](ojasx2/coordinate_compiler.py). Its strict mode defaults to the native recurrence. Algebraic mode is explicit:

```python
from ojasx2.coordinate_compiler import coordinate_koopman

# x: MPS FP32 [batch, width, 2]
# theta: [depth, 3, width], or [3, width] for shared parameters
# Both remain the original tensors used by the optimizer.
evidence = []
y = coordinate_koopman(x, theta, depth=32,
                       accuracy="algebraic", evidence=evidence)
loss = y.square().mean()
loss.backward()
```

This is a mathematical execution module and a bounded search/compiler prototype. It is not a transparent general PyTorch graph backend or a production native GPU code generator. No automatic substitution into unrelated developer models is enabled.

From this project directory:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_coordinate_foundation.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/test_coordinate_gpu.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/test_koopman_gpu.py
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/repeat_koopman_gpu.py
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/summarize_coordinate_compiler.py
```

Proof tools run on the host; all new ML references and benchmarks run on the GPU. Run benchmarks sequentially. Formal results, including the complete project theorem/statement counts and axiom audit, are in [coordinate-formal-verification.json](reports/coordinate-formal-verification.json). The combined current qualification is [coordinate-qualification.json](reports/coordinate-qualification.json).

The delivered improvement is a checked compiler contract plus a successful exact-real nonlinear representation for a particular family, with preserved original parameters and measured GPU savings. Broad huge reductions for modern dense activation workloads remain an open requirement of OjasX.
