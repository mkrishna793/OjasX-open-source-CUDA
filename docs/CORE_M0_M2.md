# Demanded cotangent and fourth-order jet core (M0–M2)

`ojasx2.core` is an executable operator and joint-compiler slice with two explicit execution modes:

1. the original demand-aware primal/pullback executor for first-order cotangents; and
2. sparse multivariate normalized Taylor jets through derivative order four.

It owns semantics for its 26 registered operators, including the M2.5 fused activation program. It does not rewrite arbitrary PyTorch models, implicitly register unknown operators, or select a generic eager fallback. See [M2.5 replay and fusion](CORE_M25.md) for the memory-transformation layer.

## Implemented scope

| Layer | Implementation |
| --- | --- |
| M0 | Unique inspectable registry; executable primal/pullback/jet implementations; arity, accuracy, derivative-order, smooth-domain, residual, alias, and proof metadata |
| M1 | Requested-gradient pruning; lazy demanded residuals; explicit fanout accumulation; first-order pullbacks; sparse normalized Taylor composition through order four |
| M2 | Joint IR; alias-group liveness; exact bounded and deterministic greedy scheduling; digest-bound memory certificates; scheduled first-order and jet execution |

Unsupported orders, arities, shapes, attributes, and checked smooth-domain violations fail explicitly. All built-in contracts advertise `max_derivative_order == 4`; the abstract `CoreOperator.jet_forward` remains an error unless an operator supplies a real implementation.

## Registry and qualified domains

| Category | Operators and semantics |
| --- | --- |
| Affine and arithmetic | `linear(X, W, bias=None)`, equal-shape `add(A, B)`, broadcast `mul(A, B)`, broadcast `div(A, B)`, `neg(X)`, `square(X)` |
| Activations and elementary functions | `relu`, `sigmoid`, `tanh`, `silu`, `gelu` (`approximate="none"` or `"tanh"`), `exp`, `log`, `sqrt`, `sin`, `cos` |
| M2.5 fused activation programs | `activation_chain(X, operations, replay)` over maximal safe elementwise chains; first-order streaming/cached replay and primal jets through order four |
| Linear algebra | `matmul(A, B)`, including vector/matrix and broadcast-batch cases; scalar operands are rejected |
| Reductions | full-scalar `sum(X)` and `mean(X)` |
| Normalized reductions | `softmax(X, dim)` and `log_softmax(X, dim)` |
| Normalization | `layer_norm(X, gamma=None, beta=None, eps)` and `rms_norm(X, gamma=None, eps)` over the last dimension; layer norm also supports gamma without beta |
| Views | `reshape(X, shape)` and `transpose(X, dim0, dim1)` |

Every contract has an inspectable `derivative_domain`. In particular:

- `log` and `sqrt` require finite strictly positive inputs;
- `div` requires every denominator element to be nonzero;
- ReLU orders two through four require finite inputs away from zero; order one retains the pinned PyTorch kink convention;
- softmax requires a nonscalar input and nonempty normalized dimension;
- normalization requires a nonempty last dimension and finite `eps > 0`;
- mean's smooth domain is nonempty; its empty-input NaN behavior is outside that domain but retained by the first-order PyTorch reference;
- the remaining arithmetic domains are finite real inputs. Orders two through four reject nonfinite inputs, directions, and resulting jet coefficients explicitly; first-order nonfinite behavior is qualified only where separately tested.

There is no implicit broadcasting in the original `add`; use `mul` or `div` where a broadcast contract is explicitly implemented.

## Sparse normalized Taylor jets

`Jet` stores coefficients indexed by canonical sorted multi-indices. For raw mixed derivative `D^α f`, the stored coefficient is

\[
  c_\alpha = \frac{D^\alpha f}{\alpha!}.
\]

This normalization makes multiplication ordinary multi-index convolution, without additional multinomial factors. Jets are sparse in multi-index support, truncated at the requested order one through four, and require all coefficients to share one primal shape, dtype, and device. Invalid, duplicate, negative, unsorted, or over-order indices are rejected.

`seed_jet` constructs either a constant or one independent first-order variable with an optional tangent. Addition supports broadcast coefficients, multiplication and matmul use truncated convolution, and unary functions use normalized composition. `derivative(α)` converts a stored coefficient back to the raw mixed directional derivative. `directional_derivative(k)` sums all order-`k` terms with the appropriate factorial to produce the total directional derivative along all seeded tangents.

The following generic order-four identities have exact real/rational source theorems and executable Z3 queries:

- normalized product convolution at order four equals the binomial Leibniz rule;
- the composition evaluator's Horner form equals its truncated degree-four polynomial.

These generic algebra results do **not** by themselves formally prove every operator-specific transcendental derivative coefficient. Those implementations are qualified by the differential matrix described below.

## Exact requested first-order pullbacks

For a linear node, let `G` have the output shape. The implemented cotangents are

\[
  dX = G W, \qquad
  dW = \operatorname{reshape}(G)^T \operatorname{reshape}(X), \qquad
  db = \sum_{\text{batch dimensions}} G.
\]

The flattening uses an explicit batch extent, so zero batch, zero input width, and zero output width do not rely on ambiguous `reshape(-1, 0)` inference.

Residual demand follows directly from these equations:

| Requested slot | Required forward information |
| --- | --- |
| `dX` | `W` |
| `dW` | `X` |
| `db` | output/batch shape only |

The expanded operators use the same demand principle. Unary nodes retain either their input or output as appropriate; multiplication retains only the operand needed by each requested slot; division retains the denominator and, only for its denominator cotangent, the output; matmul retains the opposite demanded operand; normalization materializes normalized rows and inverse scale only when required. Residuals are materialized lazily after reverse demand is known and may serve multiple gradient slots explicitly.

Unrequested leaf gradients are not emitted. A requested intermediate gradient is a boundary: reverse generation does not continue through its producer unless another requested target farther upstream requires that path. Nodes needed for the primal objective but not for any requested cotangent remain in the forward graph with residual demand removed.

Addition sends the upstream cotangent to each demanded input. Reduction creates a uniform cotangent for every nonempty input shape and explicitly materializes it only at a dense boundary; empty reductions retain the dense fallback. Broadcast binary and matmul pullbacks reduce back to the original operand shapes. Normalization implements every input/gamma/beta demand combination. Multiple contributions to one value are materialized as explicit `cotangent_add` nodes before the shared producer's pullback. Pullback nodes inherit forward attributes, so GELU approximation, softmax dimension, normalization epsilon, and view dimensions cannot silently change between primal and backward.

### Structured cotangent regions (M0/M1)

For a nonempty index set `I`, a uniform cotangent represents the field

\\[
  U_g(i)=g, \\qquad U_g=g\\mathbf 1_I .
\\]

The IR stores its logical tensor shape and one scalar witness, not an `|I|`-element allocation. Execution uses an independently owned scalar expanded as a stride-zero view. `DENSE_COTANGENT` remains the default representation and `JointGraphBuilder(structured_cotangents=False)` is the executable dense baseline.

The local representation laws are

\\[
\\begin{aligned}
B_{\\mathrm{sum}}(g)&=U_g, &
B_{\\mathrm{mean}}(g)&=U_{g/n} &&(n>0),\\\\
U_a+U_b&=U_{a+b}, &
\\rho_f(U_g)&=U_g,
\\end{aligned}
\\]

where `ρ_f` is contravariant reindexing along a view/index map. Reindexing is functorial: `ρ_g(ρ_f(U)) = ρ_{f∘g}(U)`. Reshape, transpose, additive fanout, and cotangent addition can therefore propagate one scalar through a connected reverse region. Nonempty scalar shapes are valid; empty tensors stay dense because they have no scalar witness.

Each operator declares which upstream encodings its backward accepts and which encoding every pullback output produces. An unqualified consumer receives an explicit `cotangent_materialize` node. Public gradient outputs also cross a dense materialization boundary, so no structured value escapes as an alleged dense allocation. Materialization is entry-for-entry decoding and is included in graph liveness and workspace accounting.

`analyze_cotangent_topology` computes maximal connected uniform regions with union-find over producer/consumer edges. Its certificate reports logical dense bytes, managed scalar bytes, their difference, region membership, dense boundaries, and explicit materialization nodes. These are **representation savings**, not automatically peak savings: a mandatory dense output can dominate peak liveness, and a materialization node must transiently coexist with its scalar source. The existing M2 scheduler remains the authority for peak bytes.

The default path is differential-tested against dense construction with bitwise floating-point comparisons. Exact real/rational laws do not claim IEEE reassociation; in particular, no algebraic formula that changes a general matrix reduction order is enabled by these rules. Representation metadata and the structured/dense configuration are part of `graph_digest`.

`python tools/smoke_structured_cotangents.py` executes three fixed-seed CPU comparisons and writes `reports/core-structured-smoke.json`:

| Case | Dense → structured peak | Uniform-region logical saving | Output/gradient |
| --- | ---: | ---: | --- |
| Sigmoid with dense gradient boundary | 1280 → 1032 bytes | 248 bytes | Bitwise equal |
| Reshape → transpose → sum | 196 → 104 bytes | 276 bytes | Bitwise equal |
| Additive fanout → sum | 152 → 104 bytes | 160 bytes | Bitwise equal |

The region saving can exceed the peak saving because logical and live-storage metrics answer different questions. The smoke also checks the residual-policy interaction: the single-ReLU planner chooses `packed_sign` in dense mode and lower-peak `resident_input` in structured mode.

### ReLU and nonfinite values

For a packed residual, element `i` is bit `i % 8` of byte `i // 8`; padding bits are discarded on unpack. This needs `ceil(n/8)` bytes rather than the input tensor's `n * element_size` bytes.

The Boolean gate is `(X > 0) OR isnan(X)`. The NaN term is intentional: the pinned PyTorch threshold backward propagates its upstream cotangent at NaN input. Inactive branches use selection rather than assuming that `0 * NaN` is zero. NaN, infinity, and signed-zero cases are covered by first-order differential tests; the FP32 SMT model separately checks the declared bitwise convention except for NaN upstream payload identity. Higher-order ReLU rejects the kink and nonfinite inputs rather than assigning them a fictitious smooth derivative.

## Joint IR and first-order certificates

A `JointGraph` contains external inputs, primal values, residual values, cotangent values, scalar seeds, and requested outputs. Nodes carry primal inputs, residual slots, effective demand, pullback output slots, sorted attributes, and declared explicit workspace. Validation checks sequential identities, producer relationships, residual placement, gradient signatures, policy names, supported derivative order, pullback slot mappings, cotangent encodings, scalar storage, and dense public-output/materialization boundaries.

`plan_memory` accounts for **storage groups**, not just tensor values. Resident residuals, output residuals, views, and explicitly aliased inputs therefore share allocations in the accounting. Declared input aliases must have identical layout and gradient eligibility and are checked against actual tensor storage at execution. They represent views of one autograd leaf: reverse mode traverses every aliased use, accumulates those contributions, and gives requested aliases one shared leaf-gradient output.

The first-order certificate model includes:

- live external inputs and scalar seeds;
- each distinct live storage group once;
- newly allocated node outputs before last-use release;
- declared explicit operator temporaries;
- requested outputs retained for the caller in `final_bytes`.

For at most `exact_node_limit` nodes (12 by default), subset dynamic programming minimizes peak bytes over every legal topological schedule and marks the result optimal **for that model**. Larger graphs use a deterministic greedy topological schedule and are explicitly not marked optimal. A budget failure reports required bytes and whether that requirement is the exact minimum or only the selected greedy schedule.

`execute_joint_graph` verifies the graph digest and recomputes schedule liveness, peak, and final bytes before running under `torch.no_grad()`. It checks output signatures and residual roles and releases managed references at last use. Its external inputs must be contiguous floating tensors with matching declared shape/dtype on one device. Most operators retain contiguous-input contracts; `matmul` and layout-independent `sum` also have scoped support for internally produced noncontiguous operands. Unsupported view/operator compositions fail explicitly rather than inserting untracked copies.

## Jet planning and execution

`plan_jet_memory` accepts a clean primal graph, a requested order, and an optional explicit sequence of variable external inputs. It specializes the graph with `execution_mode="jet"`, `derivative_order`, and `variable_inputs`; all three are part of `graph_digest`. Supplying a direction for a constant or unknown input is an error.

The planner executes the same sparse-jet operator implementations on meta tensors under a dispatch-mode explicit-tensor ledger. It observes actual coefficient, view, and temporary storage rather than applying a guessed dense-coefficient multiplier. The resulting per-node output sizes and measured explicit workspace feed the same exact/greedy M2 scheduler used by first order.

View aliasing is resolved during this meta execution. A contiguous reshape or transpose can share storage, but a reshape after a noncontiguous transpose may copy; the specialized graph records the observed distinction instead of assuming all reshapes alias. Execution requires contiguous external inputs and directions, checks declared input aliases, validates the specialized graph and certificate, follows the certified schedule, and releases jets at last use.

`execute_jet_graph` returns output `Jet` objects. It does not construct a hidden PyTorch autograd tape and does not silently lower an unsupported order to first order.

Both certificate modes cover **logical live storage plus declared or meta-observed explicit operator workspace**. They exclude allocator retention, library-internal workspace, Python objects, global caches, accelerator-library workspace, and runtime/framework overhead. They are not process-RSS or accelerator-allocator measurements, and graph-digest validation is internal consistency checking, not cryptographic provenance.

## Minimal use

### First-order cotangents

```python
import torch
from ojasx2.core import JointGraphBuilder, execute_joint_graph, plan_memory

builder = JointGraphBuilder()
x = builder.input("x", (2, 3), requires_grad=True)
w = builder.input("w", (4, 3), requires_grad=True)
y = builder.forward("linear", (x, w))
z = builder.forward("relu", (y,), residual_policy="packed_sign")
builder.request_gradient(w)             # dX is not computed
builder.set_objective(z)                 # full scalar sum
graph = builder.generate_reverse()
certificate = plan_memory(graph, budget=1_000_000)
outputs = execute_joint_graph(
    graph,
    certificate,
    {x: torch.randn(2, 3), w: torch.randn(4, 3)},
)
loss = outputs[graph.objective]
dw = outputs[graph.gradient_targets[w]]
```

### Fourth-order directional derivatives

```python
from ojasx2.core import execute_jet_graph, plan_jet_memory

builder = JointGraphBuilder()
x = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
y = builder.forward("exp", (x,))
builder.set_objective(y)                 # clean scalar primal objective
jet_graph, certificate = plan_jet_memory(builder.graph, order=4)
value = torch.randn(2, 3, dtype=torch.float64)
direction = torch.randn(2, 3, dtype=torch.float64)
result = execute_jet_graph(
    jet_graph,
    certificate,
    {x: value},
    directions={x: direction},
)[jet_graph.objective]
fourth = result.derivative((4,))         # one variable, raw fourth directional derivative
```

## Verification and qualification boundaries

Focused tests are organized as follows:

- `tests/test_core_m0_contracts.py`: registry uniqueness, order/domain metadata, and contract validation;
- `tests/test_core_m0_structured_cotangents.py`: dense/structured bitwise differentials, scalar storage, union-find regions, additive fanout, view composition, scalar/empty cases, dense-only consumers, and representation/digest validation;
- `tests/test_core_m1_quotients.py`: original demanded pullbacks, aliases, degenerate extents, fanout, and nonfinite partitions;
- `tests/test_core_m2_memory_execution.py`: exact/greedy certificates, budgets, tampering, liveness, and execution;
- `tests/test_core_extended.py`: all expanded first-order demand combinations, broadcast/vector matmul, normalization arities, and view behavior;
- `tests/test_core_jets.py`: direct sparse-jet arithmetic and nested PyTorch unary/multivariate differentials;
- `tests/test_core_jet_execute.py`: every registry operator against mixed PyTorch directional derivatives through order four, requested orders, budgets, digest binding, aliases, view/copy specialization, and domain failures;
- `tests/test_core_m25.py`: activation replay under both policies, bitwise fused/unfused execution, PyTorch gradients, fourth-order fused jets, fusion boundaries, joint residual policies, work reporting, budgets, and negative controls;
- `tests/test_core_m3_pytorch.py`: custom-operator contracts, fake/meta behavior, eager/AOT/Inductor execution, RNG, mutation, autocast, accumulation, and optimizer trajectories;
- `tests/test_core_m4_equality.py`: proof-gated saturation, discovered schedules, backward-inclusive costs, counterexamples, determinism, budgets, and public exports;
- `tests/test_core_proof_artifacts.py`: proof-target resolution and executable Z3 queries.

Run them with:

```bash
python -m pytest -q \
  tests/test_core_m0_contracts.py \
  tests/test_core_m0_structured_cotangents.py \
  tests/test_core_m1_quotients.py \
  tests/test_core_m2_memory_execution.py \
  tests/test_core_extended.py \
  tests/test_core_jets.py \
  tests/test_core_jet_execute.py \
  tests/test_core_m25.py \
  tests/test_core_m3_pytorch.py \
  tests/test_core_m4_equality.py \
  tests/test_core_proof_artifacts.py
python tools/smoke_core_m4.py
python tools/smoke_structured_cotangents.py
python tools/verify_core_z3.py
python tools/verify_formal_kernels.py
python tools/verify_static_and_symbolic.py
```

The higher-order differential matrix uses float64 CPU tensors and nontrivial independent directions, including mixed partials for multi-input operators. Torch 2.15's native higher-order `layer_norm` backward was not used as the order-three/four oracle after it disagreed with both the explicit differentiable formula and finite differences; first order is still checked against native `F.layer_norm`.

`lean/OjasXMath/CoreOperators.lean` and `rocq/CoreOperators.v` contain exact real/rational adjoint, finite-sum, fanout, residual-necessity, normalized product, and Horner source theorems. `lean/OjasXMath/M25Activation.lean` and `rocq/M25Activation.v` add replay-tail pullback/adjoint/optimizer preservation and replay memory/work source theorems. `lean/OjasXMath/StructuredTopology.lean` and `rocq/StructuredTopology.v` add the uniform decoding, sum/mean, addition, reindexing, materialization, and nonempty-region storage laws. `tools/verify_core_z3.py` executes 28 proof queries covering scalar linear demand projections, aggregate linear adjoint, order-four jet identities, packed-byte roundtrip, the FP32 ReLU convention, fanout, replay cotangents, work, memory selection, M4 obligations, six structured representation/storage laws, and the M4 original-fallback and bounded-error-budget planner laws, plus four intended counterexamples.

Lean/Rocq source presence and trust-escape scans are **not proof-kernel execution**. Separate scoped runs executed jsCoq on the standalone M2.5/M4/structured Rocq files with clean assumption audits for 18 declarations, and Bun/Lean 4.28.0-pre on the dependency-free M2.5 replay, M4, and structured modules. Eight checks passed, including both deliberately false controls; see `reports/formal-kernel-results.json` and [M3/M4](CORE_M3_M4.md). The full pinned Lean/mathlib project, an executed Lean axiom audit, native-kernel qualification, accelerator parity, arbitrary whole-model optimization, large-scale training results, and 31B/14 GB reproduction are not implied. Formal generic jet identities are not formal proofs of all transcendental coefficients. Repository-wide PyTorch tests and static/symbolic checks are separate gates from CUDA/MPS/native execution and large-model quality qualification.
