# OjasX backpropagation mathematical structure

This phase was built and checked on 7 October 2026. It extends the earlier bilinear reference into a mathematical structure for the shared core of ML training: derivatives, joint input/parameter cotangents, composition, fanout and residual information. It is a foundation for qualifying ML operators, not certification of every existing PyTorch operator or a universal reduction in backward cost.

This foundational phase added 34 Lean statements, twelve Z3 models and 33 CPU tests. The later [joint contraction implementation](CONTRACTION_OPTIMIZATION.md) adds a measured, shape-dependent optimization. The current runner checks **121 Lean theorems, 33 scoped Z3 models and 209 CPU tests**, with two device tests skipped. There are no unfinished Lean proofs or added project axioms. The [verification manifest](reports/verification.json) records source hashes and runtime versions; [the mathematical exploration report](reports/math-structure.json) records symbolic/counting and sampled gradient evidence.

## 1. What a certified layer must supply

A `CertifiedStage I J S` in [CertifiedBackprop.lean](lean/OjasXMath/CertifiedBackprop.lean) carries:

- `forward : input → residual × output`;
- a continuous linear derivative at the actual input;
- a proof of `HasFDerivAt` for the actual output function;
- `pullback : residual → output_cotangent → input_cotangent`;
- a proof that the derivative and pullback have the adjoint pairing.

For finite real coordinate spaces, the contract is

\[
\langle Df(x)[v],g\rangle = \langle v,R_f(s_x,g)\rangle.
\]

The derivative proof and adjoint proof are separate obligations. An arbitrary function called a "dagger" is not accepted as a correct backward implementation merely because it composes.

The structure uses finite-coordinate dot products for cotangents. The topology used to state Fréchet differentiability is mathlib's finite Pi-space topology; a change of coordinate norm does not change the finite-dimensional derivative. Alternate metrics require a different adjoint identification. Complex Hermitian differentiation and generalized subgradients are not covered by this real contract.

Three primitive constructions are included:

1. A continuous linear map, given a proved pairing for its pullback. The derivative proof comes from actual continuity/linearity.
2. A coordinate-wise family of smooth scalar activations, given scalar derivative proofs. The retained input determines each derivative coefficient. A fully proved square layer is provided as an instance.
3. A finite scalar product, with a genuine Fréchet derivative and a joint, division-free pullback for every input slot.

These are usable formal building blocks. The construction for smooth scalar activations does not fill in derivative proofs for GELU, SiLU, sigmoid or every normalization function. ReLU at zero has no ordinary real derivative and needs a separately stated framework convention.

## 2. Composition and branching

For certified layers `f` and `h`, the constructor `f.then h` proves both

\[
D(h\circ f)(x)=Dh(f(x))\circ Df(x)
\]

and

\[
R_{h\circ f}(x,g)=R_f(s_f,R_h(s_h,g)).
\]

This is composition of correct derivatives and their adjoints, not just associativity of unqualified backward callbacks. Iterating the constructor gives a correctness argument for arbitrarily many compatible certified stages. Reverse composition is associative after residual reassociation.

`scalar_pair_separates` proves cotangents are determined by their pairing with every input direction. `certified_pullback_unique` consequently proves uniqueness of a certified input cotangent. `certified_fanout_accumulates` proves

\[
R_f(s,g_1+g_2)=R_f(s,g_1)+R_f(s,g_2).
\]

A shared input or parameter must receive contributions from all consumers. Joining gradients into a tuple does not remove an independent contribution. The theorem establishes accumulation for one certified stage; a full graph IR with explicit parameter-sharing/fanout constructors and mechanized compiler lowering is still future work.

The categorical organization follows the established [reverse derivative/dagger relationship](https://arxiv.org/abs/1910.07065). The present library checks concrete finite-coordinate contracts and composition laws; it is not a complete formalization of every axiom of an abstract Cartesian reverse differential category.

## 3. One joint rule for arbitrarily many operands

Let each operand `a_i` have its own finite coordinate set. For arbitrary compatible real linear encoders and decoder, set

\[
e_i=U_i a_i,\qquad y=W\left(\bigodot_i e_i\right),\qquad d=W^Tg.
\]

Then all input cotangents are given by the single indexed rule

\[
\boxed{\bar a_i=U_i^T\left(d\odot\bigodot_{j\ne i}e_j\right).}
\]

The same decoded cotangent `d` is available to every operand. The product excluding a slot is formed directly; it never divides by that slot. Zero factors and multiple zero factors are therefore valid. An empty excluded product is one.

[Multilinear.lean](lean/OjasXMath/Multilinear.lean) proves the general joint pairing identity, genuine analytic directional derivatives of every output coordinate, and the derivative of the scalarized output in every joint direction:

\[
\left.\frac{d}{dt}\langle F(a+t\,v),g\rangle\right|_{t=0}
=\sum_i\langle v_i,\bar a_i\rangle.
\]

The primitive finite product also has a full Fréchet derivative and is connected to `CertifiedStage`. The arbitrary encoded circuit is currently formalized by its analytic directional and joint-pairing theorems; it has not been packaged into a generic flattened-coordinate `CertifiedStage` constructor. This is an explicit remaining connection, not an assumption that directional tests alone prove arbitrary programs differentiable.

The mathematical rule generalizes the old two-operand theorem. It can express tensor contractions after coordinate expansion/decomposition. It does not make the encoder/decoder contractions free, certify efficient tensor decompositions, or delete GEMM input/weight gradients.

## 4. A real arithmetic sharing result—and the strongest comparison

The [offline mathematical reference](tools/math_structure.py) explores two division-free scalar circuits. Both return every gradient of `e_0*...*e_(r-1)` for the same upstream cotangent.

The prefix/suffix circuit builds the products on either side of each slot. For `r>=2`, it uses

\[
(r-2)+(r-2)+(r-2)+r=4r-6
\]

multiplications. A cheaper rolling-cotangent circuit rebuilds prefixes and consumes inputs backwards:

```text
prefix[0] = e[0]
for i = 1 .. r-2:
    prefix[i] = prefix[i-1] * e[i]
rolling = upstream
for i = r-1 .. 1:
    gradient[i] = rolling * prefix[i-1]
    rolling = rolling * e[i]
gradient[0] = rolling
```

It uses

\[
(r-2)+2(r-1)=3r-4.
\]

Lean proves the factor-sharing algebra and conditional count identities. Z3 checks both actual Python-generated scalar expression schedules for arities 2 through 8, over arbitrary symbolic reals, including zeros. SymPy checks the schedules against symbolic derivatives and tests instrument the multiplication calls. These are independent checks connecting the reference to the theorem; the Python interpreter is not mechanically verified by Lean.

| Factors | Independent gradient formulas | Prefix/suffix | Rebuilt-prefix rolling | Retained-prefix reverse-mode backward |
|---:|---:|---:|---:|---:|
| 2 | 2 | 2 | 2 | 2 |
| 3 | 6 | 6 | 5 | 4 |
| 4 | 12 | 10 | 8 | 6 |
| 8 | 56 | 26 | 20 | 14 |

Independent formulas form each excluded product afresh, including its upstream multiplication. The saved count is

\[
r(r-1)-(3r-4)=(r-2)^2.
\]

But ordinary reverse-mode already shares work and may retain the forward prefixes. With that tape, its backward count is only `2r-2`. The rebuilt-prefix circuit pays exactly `r-2` additional multiplications to reconstruct those prefixes. Forward multiplication costs are excluded from all columns; reconstruction is included in the rebuilt-prefix column. Encoders, decoder, additions, storage, tensor contractions, dispatch and timing are also excluded.

For eight scalar factors, 20 versus 56 is a 64.29% reduction against independent formulas. **It is not a 64.29% improvement over PyTorch, not a GEMM saving and not a whole-training result.** Retained-prefix reverse mode is a stronger arithmetic baseline. Prefix sharing/rolling reverse accumulation is established algebra, and historical novelty is not claimed.

The Python circuit uses `O(r)` temporary prefix entries plus required input/gradient storage. Rebuilding rather than retaining forward prefixes can shorten their lifetime, but this phase has not measured any net memory advantage, allocator peak or acceleration for this circuit. The earlier FFN memory measurements concern a different backend rewrite.

## 5. Residual information remains an obligation

`certified_residual_reencoding` proves changing a stage's residual representation preserves its certified pullback when a decoder reproduces the old pullback on every reachable residual and every cotangent.

The earlier [residual sufficiency theorem](lean/OjasXMath/Residuals.lean) supplies the information criterion:

\[
\mathrm{encode}(x)=\mathrm{encode}(x')
\Longrightarrow R_f(x,\cdot)=R_f(x',\cdot).
\]

Only then can a decoder exist. Its existence does not establish efficient reconstruction, reduced hardware allocation or acceptable recomputation cost. This rule prevents dropping primal information just because the backward map has a dagger notation. Memory planning must also account for aliases and other consumers that keep tensors alive.

## 6. Numerical linear algebra: propagate errors rather than hide them

`backward_error_composition` proves a local error bound. If an approximate upstream pullback has error `epsP` at the approximate downstream cotangent, the downstream approximation has error `epsQ`, and the exact upstream pullback has the stated local Lipschitz bound `L>=0`, then

\[
\|\widehat P(\widehat Q(g))-P(Q(g))\|
\le \varepsilon_P+L\varepsilon_Q.
\]

This accounts for amplification of downstream error. The norm theorem works for normed additive commutative groups; it does not magically supply an instruction-specific error or conditioning bound. Per-operator error certificates and rounding/overflow assumptions remain obligations. The existing FP32 counterexample continues to refute universal bit-exact S/A reconstruction.

No low-rank approximation is silently substituted for an exact dense gradient. Approximation, constrained parameterization and changed accumulation order need their own declared semantics and numerical contracts.

## 7. Geometric constraints

[Geometry.lean](lean/OjasXMath/Geometry.lean) adds real-algebra proofs:

- On the unit sphere, `P_x(g)=g-x<x,g>` is tangent, idempotent, and preserves the pairing with every tangent direction.
- For orthonormal-column matrices `X^T X=I`, the Stiefel expression

\[
P_X(G)=G-X\,\mathrm{sym}(X^TG),\qquad
\mathrm{sym}(A)=\tfrac12(A+A^T)
\]

satisfies `X^T P_X(G)+P_X(G)^T X=0`.

The Stiefel proof establishes tangency; its directional-pairing property is sampled numerically in this phase, not yet proved as a matrix theorem. These expressions use the ambient Euclidean/Frobenius metric. Updates that preserve constraints require a retraction or another justified map; no such update or convergence theorem is claimed. Manifold optimization is optional constrained-model mathematics, not an invisible change to ordinary unconstrained training. The broader context is [optimization on smooth manifolds](https://www.nicolasboumal.net/book/).

## 8. Verification and use in the future backend

Run the project-wide verification:

```sh
python -B tools/verify_all.py
```

Or inspect the mathematics alone:

```sh
ELAN_HOME="$PWD/.tools/elan" .tools/elan/bin/lake -d lean build
python -B tools/verify_z3.py
python -B tools/math_structure.py
```

This extension changes proof modules and the offline reference/tests. It does not enable new graph rewrites in ordinary training. A future compiler should attach the derivative/pullback contract to each qualified primitive, compose those contracts, retain enough residual information, and select a schedule only after arithmetic, liveness and numerical obligations pass. Existing developer APIs can remain unchanged; qualification must still cover their actual modes and edge semantics.

The remaining research target is a joint contraction schedule that beats strong existing AD/GEMM baselines while meeting the numerical contract and memory objective. This structure makes such a candidate checkable; it does not establish that one exists for every ML workload.
