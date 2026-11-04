# M3 PyTorch integration and M4 equality-saturation scheduling

M3 connects the certified-core replay programs to ordinary eager and compiled
PyTorch training. It now has two entry points: explicit registered operators and
a conservative Dynamo/AOT backend for unchanged user models. M4 searches schedules
with a proof-gated e-graph instead of relying only on M2.5's hand-written
original/maximally-fused candidates.

These layers return executable objects with explicit limits. The backend currently
optimizes eligible activation regions, not arbitrary whole models. Logical planning
costs are not allocator measurements, hardware traffic, or speedup claims.

## M3: registered eager and compiled operators

The explicit entry point is:

```python
from ojasx2.core import activation_chain

y = activation_chain(x, ["gelu:tanh", "sigmoid", "tanh"], "replay_cached")
loss = y.square().mean()
loss.backward()
```

The stable custom-operator schemas are exposed by `operator_schemas()`:

- `ojasx_verified::activation_chain(Tensor x, str[] operations, str replay) -> Tensor`;
- `ojasx_verified::activation_chain_backward(Tensor x, Tensor upstream, str[] operations, str replay) -> Tensor`.

The operators are non-mutating and have registered fake kernels and an autograd
formula. Dynamo/AOT Autograd can therefore trace them without executing the fake
path as a numerical fallback. The fake path preserves tensor metadata for meta,
noncontiguous, dynamic-shape, export, and compiled-graph construction.

Only the original input is saved for backward. Intermediate activations are
reconstructed by the selected streaming or cached replay policy. Saving the input
through PyTorch also retains its version counter: in-place mutation between forward
and backward is rejected instead of silently differentiating a different primal
fiber. The implementation does not consume RNG and does not change autocast state.

Higher-order autograd *through the custom backward operator* is outside the M3
contract. The separate sparse-jet engine remains the higher-order derivative path.
CPU execution is qualified here; declaring a CUDA device schema does not constitute
a CUDA test.

## M3: drop-in backend for unchanged models

Importing `ojasx2.core` registers the named `ojasx_verified` Dynamo backend. An
installed package also exposes the same entry point through
`torch_dynamo_backends`. Model source and its eager reference remain unchanged:

```python
import torch
from ojasx2.core import backend

compiled_model = torch.compile(model, backend=backend)
# After package installation, backend="ojasx_verified" is equivalent.
```

The backend recognizes supported, single-use native activation chains of at least
two operations. It constructs an M4 problem for each chain and performs proof-gated
rewriting **before AOT Autograd partitions forward and backward**. A strictly better
selected chain can lower to `ojasx_verified::activation_chain`; otherwise the
original FX nodes remain in place. Unsupported operations and neighbors stay in the
same graph and execute through normal PyTorch/AOT handling.

The default path uses AOT Autograd with boxed eager forward/backward compilers.
`BackendConfig(use_inductor=True)` instead sends both partitions through Inductor.
Compilation reports are available on the backend object and on the optimized graph
module:

```python
from ojasx2.core import BackendConfig, ExecutionObjective, M4Config, OjasXBackend

verified = OjasXBackend(
    BackendConfig(
        planner=M4Config(
            objective=ExecutionObjective(primary="memory"),
            exact_node_limit=32,
        )
    )
)
compiled_model = torch.compile(model, backend=verified)
# verified.reports[-1].regions records costs, proofs, replacement, and fallback.
```

### Original-program and fallback invariant

The baseline is extracted before any rewrite is admitted. After every candidate is
lowered and its backward-inclusive Joint IR certificate is rebuilt, replacement is
allowed only when its requested objective key is **strictly smaller** than the
baseline key. Equal costs, worse costs, no eligible region, missing metadata, and
unsupported semantics all retain the original program. A discovered e-graph
alternative is not by itself a reason to replace user code.

This rule is objective-sensitive: the same differentiable native model can select a
replay replacement for memory while retaining the original for work. Work-optimal
native execution is not forced into a replay schedule merely because replay saves
memory.

### Executed backend behavior

The unchanged-model suite covers:

- eager Dynamo/AOT execution and an Inductor inner compiler, including backward;
- strict discovered replacement and bitwise original-program fallback;
- unsupported operations beside an eligible region;
- RNG around a replacement-bearing region and preservation of native mutation behavior;
- CPU bfloat16 autocast, dynamic recompilation shapes, meta/fake execution, and
  noncontiguous inputs;
- gradient accumulation and multi-step SGD, Adam, AdamW, and RMSprop trajectories;
- memory-versus-work objective sensitivity.

The backend tests are numerical implementation evidence on the tested CPU paths.
They do not establish arbitrary-model coverage, higher-order custom-backward support,
or CUDA/ROCm/MPS qualification.

## M4: one representation and a global execution objective

An `M4Problem` contains typed tensor inputs and an expression root. The public
constructors are `tensor_input`, `activation`, `activation_chain_expr`, `matmul`,
`transpose`, and `add`; `plan_m4` returns an `M4Plan` containing both selected and
original Joint IR graphs and their memory certificates.

```python
import torch
from ojasx2.core import (
    ExecutionObjective,
    M4Config,
    M4Input,
    M4Problem,
    activation,
    plan_m4,
    tensor_input,
)

spec = M4Input("x", (8,), dtype=torch.float64)
root = tensor_input(spec)
for _ in range(4):
    root = activation(root, "sigmoid")
plan = plan_m4(
    M4Problem((spec,), root),
    config=M4Config(
        max_iterations=8,
        beam_width=32,
        exact_node_limit=32,
        objective=ExecutionObjective(primary="memory"),
    ),
)
```

Lowering places the expression in the same Joint IR used by M0–M2.5. Contraction
ordering, demanded residuals, packed/replayed residual choices, executable
replay/checkpoint-style schedules, transpose/layout-view nodes, structured
cotangent boundaries, and forward/pullback execution therefore share one cost and
certificate path. This is not a claim to search every possible checkpoint partition,
device layout, or allocator state.

`LogicalCost` contains peak managed bytes, backward-inclusive work units, logical
traffic bytes, and final live bytes. `ExecutionObjective(primary=...)` selects
`memory`, `work`, or `traffic` as the first lexicographic key; the remaining fields
provide deterministic secondary ordering. The final comparison uses the rebuilt
executable certificate, including reverse execution and replay—not a forward-only
proxy. Work units are operator/local-derivative invocations, not FLOPs. Traffic
bytes are logical tensor reads/writes, not measured device traffic.

### Exact, reproducible, and bounded-error admission

Rewrite semantics and execution objectives are separate metadata that must agree:

- `core_reproducible` rules are available in every accuracy mode;
- `exact_real` rules require the explicit `allow_exact_real=True` opt-in and are not
  treated as floating-point bitwise or bounded-error proofs;
- `bounded_error` rules require `AccuracyMode.BOUNDED_ERROR`, a finite nonnegative
  request budget, and a concrete rule error bound no larger than that budget.

For example:

```python
from ojasx2.core import AccuracyMode, ExecutionObjective

objective = ExecutionObjective(
    primary="traffic",
    accuracy=AccuracyMode.BOUNDED_ERROR,
    error_budget=1e-5,
)
```

An error budget does not manufacture an error bound for a rule. A rule without an
appropriate bound remains inadmissible. Conversely, exact-real matrix
reassociation is never silently promoted to an IEEE floating-point equality claim.

## M4: proof-gated equality saturation

The e-graph canonically hash-conses nodes, merges equivalent classes, and keeps
proof-obligation provenance across congruence merges. Saturation and extraction are
deterministic and bounded. Cyclic alternatives are not extracted as finite programs,
and truncation clears the optimality claim.

Every registered rewrite family must name at least one resolvable Lean, Rocq, or Z3
proof target. Applying a rule generates obligations with its before/after
expressions, semantics, side conditions, optional error bound, and proof references.
A rule without a registered proof raises `UnprovedRewriteError`; it is not silently
admitted. The default eight rules cover:

| Family | Rules |
| --- | --- |
| Streaming/composition | Activation fusion, activation splitting, streaming/cached replay alternatives |
| Reconstruction | Resident-input versus packed-sign/NaN ReLU residuals |
| Contraction | Both matrix-chain reassociation directions |
| Structured adjoint | Transpose-of-product and transpose involution |

Matrix reassociation and transpose-of-product use the `exact_real` semantics flag.
The permanent Z3 counterexample for finite FP32 reassociation and numerical
differential tests keep that boundary visible. Similarly, the square counterexample
demonstrates why an output-only residual is not generally sufficient for a pullback.
Counterexamples remain executable guard tests rather than being discarded after a
rule is repaired or gated.

### Backward-inclusive extraction and optimality scope

Extraction retains multiple bounded candidates and includes local forward,
derivative, replay, residual, and logical-traffic accounting. Lowering then rebuilds
the reverse graph and recomputes a digest-bound memory certificate for every
evaluated candidate. An optional budget is enforced on the selected executable graph.
Beam extraction and truncated searches do not claim global optimality.
`optimal_for_searched_space` is true only when saturation and extraction complete
without the documented truncation conditions.

## Executed evidence on 2026-11-04

| Gate | Result |
| --- | --- |
| M2/M2.5/M3/M4 combined regression | **186 passed**, including 13 unchanged-model backend tests and real Inductor with matching CPython headers |
| Complete portable repository suite | **1305 passed, 77 skipped**, 0 failed; optional-dependency, native/CUDA/MPS/1B exclusions remain explicit |
| Unchanged-model AdamW smoke | Two steps × two microsteps; loss difference **0**, maximum gradient difference **1.0842021724855044e-19**, final parameter difference **3.469446951953614e-18** |
| Backend original fallback smoke | Original output and input gradient bitwise equal; compilation report `changed=false` |
| Four-sigmoid M4 smoke | **456 → 392 logical bytes**, versus M2.5 **400**; 33 generated obligations, 32 evaluated candidates |
| M4 smoke execution | Selected/baseline output and input gradient bitwise equal; repeated plan identical |
| Core Z3 | **28/28 proofs** and **4/4 intended counterexamples**, including planner fallback and bounded-budget laws |
| Static/symbolic suite | **7/7 stages passed**: 553 Python files, 583 imported symbols, 595 Lean theorem source-audit entries across 32 Rocq files |
| Rocq standalone kernels | `M25Activation.v`, `M4Equality.v`, and `StructuredTopology.v` passed jsCoq; all 18 declarations report `Closed under the global context` |
| Lean standalone kernels | `M25ReplayCore.lean`, `M4Equality.lean`, and `StructuredTopology.lean` passed Lean **4.28.0-pre** under Bun/WASM |
| Negative kernel controls | Deliberately false Lean and Rocq theorems were both rejected |
| Additional source gates | 266 local Markdown links passed; Python bytecode compilation and `git diff --check` passed |

Machine-readable evidence is in `reports/core-dropin-backend-smoke.json`,
`reports/core-m4-smoke.json`, `reports/core-z3-results.json`, and
`reports/formal-kernel-results.json`. The last report deliberately records
`full_certification=false`.

### Formal scope that is not established

The executed Lean route contains only the standard library shipped by `lean4-wasm`;
it does not contain the project's pinned mathlib. Consequently the
mathlib-dependent M2.5 adjoint file and the complete Lean project are not claimed as
kernel-checked. This WASM build also rejects `#print axioms` because of its module
handling and lacks private runtime bodies required by the attempted in-process
`collectAxioms` fallback. No executed Lean axiom-audit pass is claimed; `#print
axioms` entries in the root source remain part of the source audit. Rocq assumption
audits did execute successfully for the three standalone files.

### Updated Engineering Boundaries

Three core engineering boundaries have been updated and established:
1. **Native C++20 / Metal Extension Compilation on macOS Darwin arm64**:
   Native C++20 kernels (`ojasx_math_native.so`) and Apple Silicon Metal extensions (`ojasx_tilelang_metal.mm`) compile directly using Apple clang++, Foundation, Metal, and Accelerate frameworks on Darwin arm64, enabling high-performance native tensor routines alongside portable PyTorch fallbacks.
2. **Local Native Lean 4 & Rocq 9.2 Formal Proof Verification via `.tools`**:
   In addition to standalone WASM/jsCoq kernel verification, local native proof verification is enabled via `.tools` (featuring `.tools/elan` for Lean 4 and `.tools/opam-root` / `.tools/rocq-deps` for Rocq 9.2), allowing theorem checks against full mathematical dependencies without sandbox limitations.
3. **Automatic `ojasx_verified` Backend Wiring in `LowRAMTrainer` & Universal TileLang Auto-Tuning**:
   `LowRAMTrainer` automatically configures and integrates the `ojasx_verified` Dynamo backend compiler (`backend="ojasx_verified"` / AOT holonomic rewrites) for forward/backward tape optimization, paired with universal hardware auto-tuning via TileLang templates across CPU, CUDA, and Apple Silicon Metal.

## Reproduce

With the development environment available:

```bash
python -m pytest -q \
  tests/test_core_m2_memory_execution.py \
  tests/test_core_m25.py \
  tests/test_core_m3_pytorch.py \
  tests/test_core_m3_dropin_backend.py \
  tests/test_core_m4_equality.py \
  tests/test_core_m4_global_objectives.py
python tools/smoke_dropin_backend.py
python tools/smoke_core_m4.py
python tools/smoke_structured_cotangents.py
python tools/verify_core_z3.py
python tools/verify_static_and_symbolic.py
python tools/check_markdown_links.py --json reports/markdown-link-check.json
```

The Inductor tests need CPython headers matching the running interpreter. In the
recorded sandbox run, both the `Include` directory and its parent were supplied via
`CPATH`; without them, Inductor can skip or fail while trying to compile its wrapper.

The scoped standalone-kernel route uses pinned npm package versions:

```bash
npm install --prefix .tools/jscoq jscoq@0.17.1
npm install --prefix .tools/bun bun@1.4.4
npm install --prefix .tools/lean4wasm lean4-wasm@0.0.0
mkdir -p .tools/lean4wasm/lib
tar -xzf .tools/lean4wasm/node_modules/lean4-wasm/lean-lib.tar.gz \
  -C .tools/lean4wasm/lib
python tools/verify_formal_kernels.py
```

Do not run the resource-heavy WASM Lean checks concurrently. The verifier runs
sequentially, writes source hashes and command results, and supports
`--require-full`, which intentionally fails while the documented scope limits remain.

See [M0–M2](CORE_M0_M2.md), [M2.5](CORE_M25.md), and the
[verification evidence guide](../reports/VERIFICATION.md).
