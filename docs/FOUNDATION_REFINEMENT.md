# OjasX mathematical foundation: scoped refinement 1.1

The subsequent [abstract engine and polynomial discovery extension](ABSTRACT_BACKPROP_ENGINE.md)
records foundation 1.2. This document preserves the 1.1 normalization results.

This refinement strengthens the existing foundation by fixing demonstrated
numerical and validation failures, reducing the normalization residual model,
and extending exact proofs to complete affine gradients and repeated indexing.
It does not establish global minimum arithmetic or memory, or faster training
across most complete models. Verification evidence is in
[rocq-verification.json](reports/rocq-verification.json) and
[foundation-edges.json](reports/foundation-edges.json).

## Failures found and corrected

1. At zero FP32 input with epsilon `1e-30`, the old RMSNorm and LayerNorm
   factor construction evaluated the cube of the inverse scale. That cube
   overflowed, and multiplication by zero produced NaNs. Using normalized
   values removes that unnecessary cubic evaluation. The exact coefficient
   rewrite is proved in `rocq/Compact.v`; its floating execution is tested.
2. NaN and infinite epsilon values were accepted. The factories now require
   a finite, positive, representable scalar epsilon. Zero-width normalization
   and scalar derivative tensors are rejected. Empty leading batch axes remain
   supported. Unsupported FP16, integer and complex reference execution is not
   silently treated as qualified FP32/FP64 execution.
3. The old normalization interface described only input gradients. The new
   compact residual supplies input, shared gain and shared bias gradients.
   Direct residual construction also checks structural consistency.
4. A gather pullback must accumulate every occurrence of a repeated input
   index. Overwriting one contribution loses gradients. The new finite-index
   adjoint theorem and extracted scatter-sum reference handle repetitions.

## Compact normalization mathematics

Let a row have width n > 0. For RMSNorm set c = x; for LayerNorm set
c = x - mean(x). Define

```
r = 1 / sqrt(mean(c*c) + epsilon)
z = r*c
y = gamma*z + beta
q = gamma*g                  # q = g when gain is absent
```

For an output cotangent g, the input pullbacks are

```
RMSNorm:   dx = r * (q - z*mean(q*z))
LayerNorm: dx = r * (q - mean(q) - z*mean(q*z))
```

The shared parameter gradients are reductions over every leading batch axis:

```
dgamma = sum_batch(g*z)
dbeta  = sum_batch(g)
```

`Compact.rms_residual_sufficient` and `layer_residual_sufficient` prove the
equivalence to the diagonal plus low-rank derivative representation for every
finite dimension in exact rational arithmetic. `affine_all_gradients` proves
the full affine differential pairing, including both parameter gradients.
These identities do not prove that an arbitrary supplied z and r came from
the actual normalization primal. That analytic leaf obligation remains
separate; the Python factories are checked against PyTorch.

Positive epsilon must remain in the mathematics. RMSNorm does not have an
exactly zero radial derivative when epsilon is positive. The theorem
`epsilon_radial_correction` gives the retained correction under its stated
normalization identity. Removing it would change training.

For b rows, a literal diagonal/rank-one representation stores 3*b*n scalar
values; a rank-two representation stores 5*b*n. The compact normalization
residual needs b*n + b scalar values. This comparison counts required residual
values, not distinct allocated bytes. `compact_rms_residual_budget` and
`compact_layer_residual_budget` prove these count inequalities for every row
count and every positive width. Parameters, aliases, autograd graphs,
temporary tensors and allocator peak are excluded. Native PyTorch already
uses compact normalization derivatives, so this is a reduction against our
generic factor representation, not an established saving over native training.
Both compact pullbacks use O(b*n) arithmetic and avoid an n-by-n Jacobian.

## Generality and precise boundaries

The composition, branch accumulation and finite repetition contracts continue
to support building graphs from verified primitive derivatives. Pointwise
diagonal rules, low-rank corrections, bilinear matrix products and finite
gather/scatter cover useful shared structures across existing and experimental
models. A new primitive still needs a valid derivative and sufficient residual;
membership in a compositional graph does not prove its leaf rule.

The normalization reference supports any number of leading axes, a nonempty
last coordinate axis, FP32/FP64, optional shared coordinate gain and bias, and
fixed scalar epsilon. Noncontiguous layouts and higher input derivatives are
tested. Epsilon differentiation, arbitrary affine broadcasting in the compact
interface, complex derivatives, mixed precision and full higher-order
parameter/optimizer semantics need additional qualification.

Finite epsilon does not guarantee all intermediate quantities remain finite
for every possible input magnitude. Squaring extreme inputs, cancellation,
nonfinite operands, or large affine parameters can still overflow or lose
accuracy. This work does not supply a uniform floating error bound. Reassociated
GEMMs remain tolerance-based; exact real/rational algebra is not bit equality.

The extracted rational functions are correctness oracles. The handwritten
OCaml adapters and Python implementations are tested components, not entirely
verified backend code. The finite-index proof supports arbitrary finite maps;
its test adapter intentionally limits sizes to 64 for bounded oracle runs.

## Reproducing and using the reference

From the project root, with the recorded Python environment:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_rocq.py
```

The runner compiles 11 proof modules, audits all 100 theorem/lemma statements,
independently kernel-checks them, extracts three exact OCaml libraries, builds
their adapters, and runs the original, broad and new edge qualifications.
It rejects unfinished proofs or added axioms and records source hashes.
The new edge suite includes 80 compact exact cases, 13 indexed exact cases,
four invalid extracted requests, 124 floating normalization cases, two
noncontiguous/higher-order checks, and 24 invalid-domain checks. Four optimizer
configurations each run four steps with complete input/gain/bias gradients.
Another 20 checks cover activation boundaries, finite saturation inputs and
softmax rows of width one, tied logits and large finite logit differences.
The Z3 suite has 14 checks, including seven deliberately rejected claims.
These counts describe finite test coverage, not enumeration of every input.

```python
from ojasx2.math_foundation import normalization_residual

# x [..., n], gamma [n], beta [n], upstream [..., n]
residual = normalization_residual(x, gamma, beta, kind="layernorm")
dx, dgamma, dbeta = residual.pull(upstream)
```

For higher derivatives, create the residual with differentiable statistics.
For first-order manual VJPs, constructing it under `torch.no_grad()` avoids
retaining the factory's autograd graph. This reference is not a custom autograd
operator: calling it does not automatically change an ordinary model's storage
or compiler dispatch. Keep parameter versions consistent with the forward
pass and do not mutate saved values before the pullback.

An optimized backend can implement these formulas with row reductions and
parameter-gradient reductions, save only the qualified sufficient values, and
use normal developer-facing model/optimizer interfaces. Before dispatch it
must validate shapes, dtype, aliasing, requested gradients and numerical error;
compare actual peak memory and full training-step runtime against native
execution; and retain a fallback for unsupported cases.

## Optimization policy

The existing conditional linear-chain contraction and Strassen/Winograd
identities remain available with explicit preconditions. The rank-seven CPU
references remain slower than native matmul in the measured suite and stay
opt-in. This refinement adds no GPU speedup claim and performs no new GPU run.
It also does not replace nonlinear input-dependent derivatives with constant
linear maps.

The outcome is a broader, better checked mathematical reference foundation.
Large training gains still require an applicable arithmetic reduction, a
numerically qualified implementation and a measured whole-model benefit.
