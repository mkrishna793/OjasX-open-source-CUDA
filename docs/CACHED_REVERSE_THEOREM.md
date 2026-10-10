# Cached nonlinear reverse construction

The reversible idea now has a faster tested GPU variant. Saving branch
preactivations removes projection replay during backward; a native Metal SiLU
step computes the activation value, state reconstruction and activation VJP
together. Two M5 measurement runs show a modest advantage over native PyTorch:
the final run measures **1.06–1.11× native joint throughput**, and **1.60–1.70×
the old replay prototype**, with **46–49% less fresh activation tape**.
The first run measured 1.07–1.09× native; it is preserved independently.

This is a construction from established calculus, adjoints and sufficient
residuals. The contribution is the proved schedule and its implementation, not
a claim to have invented reversible networks or a universal faster derivative.

Evidence: [Lean proofs](lean/OjasXMath/CachedReverse.lean),
[Z3 models and counterexamples](reports/cached-reverse-z3.json),
[GPU measurements](reports/reversible-cache-mps.json),
[initial GPU measurements](reports/reversible-cache-mps-initial.json),
[CPU measurements](reports/reversible-cache-cpu.json),
[regressions](tests/test_cached_reverse.py), and
[complete verification](reports/verification.json).

| Final M5 SiLU experiment | Native | Old replay | Full cache + Metal | Speed vs native | Fresh tape reduction vs native |
|---|---:|---:|---:|---:|---:|
| Depth 16, batch 128, split width 32 | 5.532 ms | 8.544 ms | 5.213 ms | 1.061× | 46.03% |
| Depth 64, batch 128, split width 32 | 20.656 ms | 33.187 ms | 19.565 ms | 1.056× | 49.02% |
| Depth 16, batch 512, split width 128 | 5.929 ms | 8.567 ms | 5.341 ms | 1.110× | 46.03% |

The final cache-Metal path is faster than native in 9/9, 9/9 and 8/9 paired
rounds respectively. The speed margin is small and shape-specific. The plain
ATen cache variant and partial-cache hybrid still measure slower than native.
The old depth-64 replay path saves 99.22% of fresh tape but is slower; full
caching deliberately gives up that depth-independent storage property.

## Mathematical construction

An existing additive coupling block is

```text
zf = x2 Wfᵀ + bf
y1 = x1 + scale σ(zf)
zg = y1 Wgᵀ + bg
y2 = x2 + scale σ(zg)
```

The original replay prototype saves final states but recomputes both projections
`zf` and `zg` during backward. The new residual stores these preactivations for
selected blocks. It reconstructs inputs without replaying their projections:

```text
x2 = y2 - scale σ(zg)
x1 = y1 - scale σ(zf)
```

These are exact-real equalities. They do not require either branch or activation
to be invertible, and the branch weights remain independently trainable.

For incoming state cotangents `g1, g2`, use the ordered joint pullback:

```text
δg = scale g2 ⊙ σ′(zg)
dWg = flatten(δg)ᵀ flatten(y1)
dbg = sum_batch(δg)
adj1 = g1 + δg Wg

δf = scale adj1 ⊙ σ′(zf)
dWf = flatten(δf)ᵀ flatten(x2)
dbf = sum_batch(δf)
dx1 = adj1
dx2 = g2 + δf Wf
```

The G branch must contribute to `adj1` before the F branch is differentiated.
Caching the derivative inputs removes replay; it does not remove this causal
dependency. Frozen parameters omit their gradient products. Shared parameters
accumulate contributions through the ordinary custom-autograd argument mechanism.

For SiLU, a single common scalar `s = sigmoid(z)` gives

```text
σ(z)  = z s
σ′(z) = s [1 + z(1-s)]
```

The Metal kernel therefore generates two outputs from the same scalar terms:

```text
reconstructed = state - scale z s
δ = (scale incoming) s [1 + z(1-s)]
```

This jointly evaluates a primal reconstruction and its local cotangent action.
The cache produces the mathematical projection-work reduction; the joint Metal
step reduces pointwise launches and traffic. Both are needed for the measured
advantage over native GPU PyTorch. The unfused CPU/ATen cache variant remains
slower than native in the recorded cases.

## Work and storage theorem

For `L` blocks with all state/weight gradients demanded and `C` cached blocks:

```text
ordinary native joint GEMMs = 6 L
old replay joint GEMMs      = 8 L
new cached joint GEMMs      = 8 L - 2 C
new backward GEMMs         = 6 L - 2 C
fresh cached state elements = (2 + 2 C) batch split_width
```

Full caching, `C=L`, removes **25% of joint GEMMs** and **one third of backward
GEMMs relative to old replay**. It matches the ordinary native GEMM count. It does
not halve the dense backward arithmetic relative to native PyTorch.

Among cache counts under a uniform storage budget, taking the largest admissible
`C` minimizes this GEMM-count expression. This does not prove optimal GPU latency:
kernel launches, layout, throughput and memory pressure require measurements.
The implementation exposes `cache_interval`; caching every fourth block gives
`C=ceil(L/4)` for its selected index set.

For the measured ordinary coupling expression, native fresh retained tape is
`(4L-1) batch split_width` elements after supplied input/parameter aliases are
excluded. Full caching saves `(2+2L)` such elements, which is smaller for `L≥2`
and tends toward half of that native tape at large depth. At depth one this
particular cache can increase fresh storage. This is residual accounting for
the specified prototype, not a universal allocator or vendor-fused baseline.

The CPU profiler regression verifies actual `aten::mm`/`aten::addmm` counts for
full and partial caching. No saved-tensor hooks are used in timing or correctness.
Storage audits use hooks only on immutable tensors during measurement.

## Reconstruction error refinement

Without caching, reevaluating a branch at an approximate recovered input can
amplify a state error by the branch's Lipschitz factor. With a cached preactivation,
the branch value is independent of the reconstructed state. If the forward add,
cached value evaluation and reverse subtraction have additive errors `ea, ev, es`,
the reconstructed state error obeys

```text
|error| ≤ |ea| + |ev| + |es|
```

Lean checks this identity/bound. It is conditional on the stated error
decomposition; it does not supply hardware error magnitudes. Cached activation
slopes also use the original preactivations rather than replayed approximations.
Incoming gradient errors can still be amplified through actual Jacobians.
Partial caching retains replay-related sensitivity on uncached blocks.

Caching cannot restore bits lost in a rounded state addition. A concrete FP32
regression uses input 1 and branch offset 16,777,216. The recovered state is 0,
and a cached weight gradient becomes 0 where native saved-operand backprop gives
1. This is a deliberately passing *negative qualification test*. Use strict
execution for such regimes; this prototype has no universal stability guard.

## How counterexamples refined the theorem

The new Z3 tool proves nine scoped models and verifies four counterexamples:

1. **Parallel independent branch pullbacks are incorrect.** Unit slopes/weights,
   `g1=0`, `g2=1` produce the required contribution 2 where the shortcut gives 1.
   The implementation preserves G→cotangent-addition→F order.
2. **Activation output need not identify its derivative.** The solver finds
   `z=1` and `z=-1` for a square activation: same output, opposite derivatives.
   This refutes a generic output-only residual claim, not a specific SiLU result.
   The implementation saves preactivations.
3. **Partial caching does not remove all replay.** `L=1, C=0` refutes the claimed
   native-count equality. The theorem now uses `8L-2C`.
4. **Exact-real subtraction is not an FP32 inverse.** The 16,777,216 offset erases
   input 1. The claimed guarantee is scoped to real algebra; a negative regression
   preserves the floating limitation.

Z3 checks explicit scalar/finite models. Lean supplies general typed statements;
neither tool proves GPU wall-clock speed or the Metal binary's correctness.

## Lean proof scope

Thirteen new statements cover:

- Generic cached coupling reconstruction over additive groups.
- Actual real sigmoid and SiLU derivative proofs.
- A nonlinear affine directional derivative under a primitive derivative contract.
- Ordered coupling input/parameter cotangents for arbitrary real slopes.
- Shared SiLU activation-value/VJP algebra.
- Arbitrary finite-dimensional affine/pointwise dagger and weight-gradient pairing.
- Cache-dependent GEMM savings, full-cache native count and budget monotonicity.
- Retained storage reduction under the declared native accounting model.
- Additive cached-state reconstruction error.

The complete project has 152 kernel-checked statements, using only the audited
usual Lean foundations and no `sorry`, `admit` or custom axiom. Mathematical
derivatives and cache identities are distinct from a formal refinement proof of
the Python implementation or emitted GPU instructions, which is not supplied.

Final full regression verification passes 358 tests, including 64 actual MPS
checks; one CUDA test is skipped. All source hashes in the final GPU and full
verification reports match the delivered sources.

## Implementation and testing

The existing architecture and optimizer API remain intact. New options are
explicit; previous defaults retain their behavior:

```python
from ojasx2 import reversible_coupling_chain

y1, y2 = reversible_coupling_chain(
    x1, x2, blocks,
    activation='silu', scale=.1, accuracy='algebraic',
    residual_policy='preactivation', cache_interval=1,
    pointwise_backend='metal',
)
loss = loss_fn(y1, y2)
loss.backward()
optimizer.step()
```

Metal mode requires MPS FP32 SiLU and uses PyTorch 2.8's shader compiler. A native
pointwise backend supports CPU FP32/FP64 and sampled MPS FP32 GELU/SiLU. Leading
batch/token axes and strided incoming cotangents are handled explicitly. Saved
parameter/preactivation version checks reject mutation before backward. Algebraic
cached replay supports first derivatives; higher derivatives require strict mode.
There is no automatic conversion of an arbitrary model into a coupling network.

The GPU experiment covers three stable-scale SiLU configurations and a 30-step
AdamW trajectory. Its output/all-gradient contract is declared as `rtol=2e-4`,
`atol=2e-4`, plus relative-L2 below `2e-4`. The first measured maximum cached-Metal
gradient discrepancy is about 2.5e-7; the optimizer parameter drift is 2.98e-8.
The detailed JSON retains every comparison and timing sample, including slower
native-cache and hybrid modes. Timings include Python dispatch and synchronized
GPU execution of forward/all gradients, with warmup and shuffled mode order.
Memory numbers cover fresh activation tape, excluding supplied operands,
parameters, gradients, optimizer states and transient workspaces.

Reproduce from the rebuild directory:

```sh
python -B tools/verify_cached_reverse_z3.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B -m pytest tests/test_cached_reverse.py -q
python -B tools/reversible_cache_research.py --device cpu
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B tools/reversible_cache_research.py --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B tools/verify_all.py
```

GPU execution needs a context that exposes the MPS device. The benchmark fails
explicitly when it cannot access MPS and never substitutes CPU measurements.
Real-data training quality, sustained large-model throughput, complete peak
memory, GELU native Metal lowering and other accelerators remain unqualified.
