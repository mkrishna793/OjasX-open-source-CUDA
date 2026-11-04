# M2.5: residual replay, activation fusion, and joint memory optimization

M2.5 extends the M0–M2 core before the M3 production/PyTorch-autograd phase. Its
central object is a **semantics-preserving memory transformation**, not a stand-alone
scheduler: activation chains are re-encoded around residuals, their reverse programs
replay primal values on demand, and the resulting graph competes with the original
under the same live-storage certificate.

## Mathematical object

For a chain of elementwise maps

\[
  x_{\ell+1}=\phi_\ell(x_\ell),\qquad \ell=0,\ldots,L-1,
\]

the reverse recurrence is the adjoint composition

\[
  g_L=\bar y,\qquad
  g_\ell=(D\phi_\ell(x_\ell))^*g_{\ell+1}.
\]

The implementation retains the original chain input and reconstructs the tail state
needed by each local pullback. In residual terms, if `save` is the head residual and
`saveTail` the original tail residual, the replay map must satisfy

\[
  \mathsf{replay}(\mathsf{save}(x))
  =\mathsf{saveTail}(\mathsf{primal}(x)).
\]

Substituting this identity leaves the local pullback, adjoint, and downstream
optimizer update unchanged. This is stronger than merely saving storage: it gives an
explicit reason why the reverse program is the original reverse program after residual
re-encoding.

For the supported elementwise maps, the adjoint is diagonal. Its entries use the same
shared primitive derivative formulas as the unary operators and the normalized-jet
implementation. Examples include
`(D sigmoid)(x)[h]=sigmoid(x)(1-sigmoid(x))h` and, for `t=tanh(u)`,
`(D GELU_tanh)(x)[h]=[(1+t)/2+x(1-t^2)sqrt(2/pi)(1+3x^2/10)/2]h`,
with `u=sqrt(2/pi)(x+x^3/10)`. Nonfinite points and ReLU at zero follow the
operator-specific executable formulas; no claim of differentiability at a ReLU kink is
made.

## Safe fusion boundaries

`fuse_activation_chains` fuses only maximal, linear chains of registered pure
single-input/single-output elementwise activations. It preserves:

- every requested gradient or explicitly consumed intermediate value;
- fanout, nonlinear operators, and other graph boundaries;
- GELU `approximate` attributes, canonicalized to `none`/`tanh`;
- output and objective identity through graph-digest-based value replacement;
- the original builder as an unchanged fallback candidate.

The supported activation set is deliberately explicit: ReLU, sigmoid, tanh, exp, log,
sqrt, sin, cos, SiLU, GELU, square, and negation. Normalization, softmax, dropout, and
arbitrary Python/PyTorch callables are **not** included. GELU supports the exact and
`tanh` forms; unsupported approximations are rejected rather than silently changed.

## Two replay policies

Both policies retain only the original chain input as the fused residual. They differ
in backward replay and therefore in the memory/work tradeoff.

### Streaming replay

At reverse layer `k`, recompute the prefix from the original input until the required
local residual is available. Across `L` layers this performs

\[
  \sum_{k=1}^{L}k=L(L+1)/2
\]

activation evaluations. Its declared replay workspace is depth-independent: two chain
value slots plus the largest local backward workspace. This can save substantial
storage on long chains, but the quadratic recomputation is reported alongside the
memory result and is not hidden by an “optimal” label.

### Cached replay

Recompute the chain once at the start of backward and retain its intermediate outputs
while unwinding it. This performs `L` activation evaluations but uses depth-linear
workspace (`L` chain slots plus local backward workspace). It can be faster in work
count while losing the peak-memory contest; the optimizer therefore considers it
rather than assuming streaming is always best.

These counts describe activation evaluations in the fused replay operator. They are
not total FLOPs, latency measurements, or CUDA-device counters.

## Joint optimization and scoped optimality

`plan_m25_memory` constructs the original and maximally fused candidates and, for each
candidate, jointly searches:

1. residual policies, including packed-ReLU sign storage where the surrounding graph
   does not already make the original input resident;
2. exact or deterministic-greedy live-storage schedules;
3. streaming versus cached activation replay.

Candidates are compared by certified logical peak, then activation replay evaluations
and final live storage. A fused candidate is accepted only when its certified peak is
no larger than the original candidate. The original graph is retained when fusion is
not beneficial.

`optimal_for_considered_graphs` means exactly what its name says: optimality over the
original/maximally fused candidates, their searched residual-policy assignments, and
exact schedules when those searches completed. It does **not** mean global optimality
over all possible rewrites, all checkpoint partitions, or all allocator layouts.
Greedy fallback clears the optimality label. `replay_evaluations` reports only the
fused activation work defined above; certificate steps separately expose declared
workspace, and neither number is a total-FLOP or latency measurement.

## Executable example

```python
from ojasx2.core import JointGraphBuilder, plan_m25_memory

builder = JointGraphBuilder()
x = builder.input("x", (8,), requires_grad=True)
y = x
for _ in range(4):
    y = builder.forward("sigmoid", (y,))
builder.request_gradient(x)
builder.set_objective(y)

plan = plan_m25_memory(builder)
print(plan.certificate.peak_bytes)
print(plan.baseline_certificate.peak_bytes)
print(plan.replay_evaluations)
```

On the float64 test fixture, the managed logical peaks are **456 → 400 bytes**, with
10 streaming replay evaluations. The dispatched CPU tensor ledger on the same fixture
observes **576 → 520 bytes**; its absolute values include execution temporaries outside
the logical certificate. A two-sigmoid negative control retains the unfused graph at
328 logical bytes. The dense-mode packed-ReLU policy example reduces 964 → 908 logical
bytes without fusion; structured mode instead selects the lower-peak `resident_input`
policy at 712 bytes. These are deterministic synthetic memory-ledger results, not
hardware allocator benchmarks or speed claims.

## Formal and executable evidence

| Layer | M2.5 artifact | What it establishes |
|---|---|---|
| Lean source | `lean/OjasXMath/M25Activation.lean` | Replay-tail primal/pullback/adjoint/optimizer preservation and the minimum-candidate memory guard |
| Rocq source | `rocq/M25Activation.v` | Generic replay substitution, cotangent chain algebra, work comparison, memory guard, and a non-universal-fusion counterexample |
| Z3 | `tools/verify_core_z3.py` | Exact-real cotangent composition, integer work and selection guards, and intended counterexamples |
| Python | `tests/test_core_m25.py` | All supported activations under both replay policies, bitwise fused/unfused first-order execution, PyTorch gradient comparison, fourth-order fused jets, boundaries, budgets, and fallback labels |

The Lean and Rocq files are integrated into the source audit lists. On 2026-11-03,
jsCoq checked `rocq/M25Activation.v` and all six declarations reported `Closed under
the global context`; Bun/Lean 4.28.0-pre checked the dependency-free
`lean/OjasXMath/M25ReplayCore.lean`. Deliberately false negative controls were
rejected. These are real but **scoped** kernel runs, recorded in
`reports/formal-kernel-results.json`. The mathlib-dependent M2.5 adjoint file and
complete project were not checked by that WASM route, and its Lean axiom audit is
unavailable. Z3 proves only its stated mathematical models; it does not certify
floating-point code, the Python implementation, or a hardware allocator. Bitwise CPU
comparisons are implementation evidence for the tested first-order path, not formal
proofs about every backend. Fused primal jets are checked through order four; reverse
replay itself is a first-order pullback path, not a fourth-order reverse-jet claim.

## M3 and M4 continuation

M3 now registers eager/compiled PyTorch operators, fake kernels, and an autograd
formula that saves the original input and retains mutation version checks. Its
`ojasx_verified` Dynamo/AOT backend also plans eligible native activation regions
without changing model source, retains the original program on ties or when no
candidate improves the requested objective, and leaves unsupported neighbors in
PyTorch. RNG, mutation, CPU-autocast, dynamic/meta/noncontiguous execution,
gradient-accumulation, standard-optimizer, AOT, and Inductor behavior have focused
tests. M4 adds deterministic proof-gated equality saturation, backward-inclusive
memory/work/traffic objectives, exact-versus-bounded-error admission, and eight
rewrite rules across streaming, reconstruction, contraction, and structured-adjoint
families. Neither addition qualifies arbitrary whole-model rewriting,
CUDA/ROCm/MPS hardware, or backend allocators.

See [M3/M4 integration and scheduling](CORE_M3_M4.md),
[M0–M2 core](CORE_M0_M2.md), and the
[verification report](../reports/VERIFICATION.md) for the implementation and current
execution status.
