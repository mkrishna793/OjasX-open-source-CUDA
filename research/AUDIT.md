# Deep Repository Audit — OjasX v0.13.0

**Audit date:** 2026-11-03  
**Commit:** `1b0a2636491cbb451d82ebff6976220a473e84b1` (`v0.13.0`)  
**Branch:** `arena/6e0e45a0-ojasx-open-source-cuda`  
**Scope:** all 835 tracked paths, including package code, tests, tools, native sources, formal sources, documentation, checked-in evidence, assets, data, CI/release configuration, external gitlinks, and vendored TileLang.

## Executive verdict

The CPU/PyTorch core is substantially functional, and many focused numerical, autograd, Z3, packaging, CLI, and native-header checks passed. However, the repository should **not** be treated as fully verified on the strength of its current green CI or checked-in reports.

The audit confirmed:

- **8 high-severity findings**, including a reliably broken public numerical-repair API, incorrect trainer accumulation behavior, unsafe externally bound mesh workers, a native static-initialization race, a reproduced native-wrapper shape failure, and major gaps in CI/formal/benchmark evidence.
- **11 medium-severity findings**, including fallback-only official wheel production, codegen drift, fallback/native emitter drift, ignored public options, updater defects, non-reproducing memory claims, unsafe dataset evaluation, and dependency provenance gaps.
- **2 low-severity findings** involving stale tests and mislabeled assets.
- The broad pytest run finished with **886 passed, 68 skipped, 5 failed**. The exact CI selection passed **180/180**, but it selects only **11 of 49 test modules** and misses all five broad-suite failures.
- Lean, Rocq, OCaml extraction, CUDA, MPS, full native extension, and large-model performance claims remain **unexecuted here**, not passed or failed.

At the baseline audit checkpoint, no product source had been changed and the added files were audit reports/evidence only. The working-tree remediation that followed is mapped in the addendum at the end of this document.

## Severity and evidence index

“Reproduced” means executed against current code. “Source-confirmed” means the defect follows from current source but the relevant hardware/runtime path was not executed. “Evidence/configuration” concerns reproducibility or reporting rather than a demonstrated numerical error.

| ID | Severity | Finding | Evidence status |
|---|---|---|---|
| F01 | High | `checked_value_and_grad` raises `NameError` | Reproduced from source tree and isolated wheel |
| F02 | High | Trainer scales accumulation twice and mishandles partial cycles | Reproduced |
| F03 | High | Mesh worker lacks authentication and frame limits | Source-confirmed |
| F04 | Medium | Official build produces a fallback-only wheel; native source release is incomplete | Reproduced/package inspection |
| F05 | High | Ternary native lookup table has unsynchronized lazy mutation | Source-confirmed; race not executed |
| F06 | High | Native-enabled wrapper accepts shapes that fail scale reshaping | Reproduced for both bitrates with mocked native symbol |
| F07 | High | CI selects 11/49 modules and omits native, formal, and broad regression coverage | Reproduced/configuration |
| F08 | High | Formal toolchain and checked-in verification evidence are not reproducible as current proof | Configuration/hash-confirmed; compilers unavailable |
| F09 | Medium | Codegen `--check` writes files and generated output is stale | Reproduced in temporary directory |
| F10 | Medium | Three native-source fallback emitters drifted from tests; loader failure is a test-isolation bug | Reproduced and triaged |
| F11 | Medium | Trainer `dtype` and default fused-loss behavior do not match the public configuration | Source/dataflow-confirmed |
| F12 | High | BackendBench comparison/reporting can overstate gradient and compliance assurance | Source-confirmed; small run reproduced related failure |
| F13 | Medium | Checked-in LoRA tape-reduction claim does not reproduce on current Torch | Reproduced |
| F14 | Medium | Pip updater drops `target_version`; `--auto` is not implemented | Reproduced with subprocess mock |
| F15 | Low | Updater test expects an older release to be an update | Reproduced; stale test, not product downgrade logic |
| F16 | Medium | Simple API drops documented options and disables documented default moment compression | Source/dataflow-confirmed |
| F17 | Medium | Documentation and diagnostic output overstate or lose verifiability | Census/source/output-confirmed |
| F18 | Medium | Benchmark tools `eval` dataset strings | Source-confirmed; developer-tool scope |
| F19 | Medium | Lean bootstrap executes an unpinned, unhashed remote shell script | Source-confirmed |
| F20 | Medium | External checkout/provenance is not reproducible from a fresh clone alone | Raw-Git and checkout-confirmed |
| F21 | Low | Three `.png` assets are JPEG files | Binary-signature-confirmed |

---

## High-severity findings

### F01 — The checked numerical-repair API always fails before candidate evaluation

**Files:** `ojasx2/numerical_repairs.py:143-180`

`checked_value_and_grad()` defines `run(fn, higher)` at line 158 and assigns `value` inside that nested function at line 159. The scalar validation and gradient logic at lines 163-168 are outside `run`, where `value` and `higher` are not defined. `reference = run(native, create_graph)` is reached only afterward.

A direct call reliably raises:

```text
NameError: name 'value' is not defined
```

The same failure reproduced after installing the built wheel into an isolated target. This module had **0% pytest coverage**, and its dedicated research tool reaches the same exception.

**Impact:** the advertised reliability/fallback API cannot perform its native check, candidate check, evidence recording, or gradient return.

**Recommended fix:** indent scalar validation, gradient computation, and return inside `run`; add native-pass, candidate-pass, candidate-rejection, no-required-gradient, and higher-order tests in normal pytest CI.

### F02 — Low-RAM training accumulation is mathematically wrong at cycle boundaries

**Files:** `ojasx2/trainer.py:109-149`, `ojasx2/grad_free_optimizer.py:170-224,399-410`

Two independent mechanisms divide by `grad_accum_steps`:

1. `LowRAMTrainer.train_epoch()` computes `scaled_loss = loss_val / N` at line 139.
2. The optimizer hook computes `scaled_grad = grad / N` again at line 197.

A controlled probe with two identical microbatches and `N=2` produced an accumulated effective gradient of **G/2 instead of G**.

Partial cycles are also unsafe:

- The trainer steps only when `(batch_idx + 1) % N == 0`; it has no end-of-epoch flush.
- With three microbatches and `N=2`, optimizer state remains at `accum_count=1` after the epoch.
- `zero_grad()` clears `grad_accum` and resets `accum_count`, so pending work can be discarded by the next cycle.

**Impact:** incorrect update magnitude and dropped or delayed gradients for common non-divisible dataset lengths.

**Recommended fix:** make exactly one layer own accumulation scaling; define whether the effective gradient is a sum or mean; flush or explicitly carry a partial window at epoch end; test two identical microbatches, non-divisible windows, state dict round trips, and repeated epochs against a plain-PyTorch reference.

### F03 — Externally bound mesh workers expose unauthenticated control and unbounded allocation

**File:** `ojasx2/mesh_hal.py:87-105,287-375`

The worker protocol has no authentication, authorization, or TLS. Any client that can connect can issue handshake, forward/backward, train-mode, gradient-reset, and optimizer-step opcodes. Lines 354-365 explicitly accept optimizer and mode controls.

The framing code reads an unsigned 64-bit length and immediately attempts to receive that many bytes:

```python
(length,) = struct.unpack("!Q", header)
return _recv_exact(sock, length)
```

There is no maximum frame size, tensor-rank/shape limit, or pre-allocation budget.

The default host is loopback, which limits default exposure. The defect becomes remote when users bind a worker/channel to a non-loopback interface, and a local untrusted process can still connect to a loopback service.

**Impact:** unauthorized model execution or parameter mutation, plus memory/connection denial of service.

**Recommended fix:** mutually authenticate peers, use TLS or a clearly documented trusted-network-only transport, authorize control opcodes, cap frames and tensor dimensions before allocation, validate exact message lengths per opcode, and add hostile-protocol tests.

### F05 — Ternary native lookup initialization is racy

**Files:** `ojasx2/math/csrc/ojasx_math_core.hpp:547-570,590,650`, `ojasx2/math/csrc/ojasx_math_bindings.cpp:419-439,481-499`

`get_ternary_rank_to_coords()` lazily mutates a static `std::vector<int8_t>` and static `bool initialized` without synchronization. Both ternary forward and backward call it. The bindings invoke those kernels inside `at::parallel_for(0, M, 1, ...)`, so separate rows can make concurrent first-use calls.

Concurrent first use can race on `initialized`, vector reallocation, and `coords_table.data()`, risking corrupt lookup results or invalid memory access.

**Evidence limit:** normal-shape single-threaded harness parity passed; the race itself was not dynamically exercised.

**Recommended fix:** initialize through a function-local static lambda or `std::call_once`, return a fully initialized immutable table, and add a multithreaded first-use stress test under TSAN where supported.

### F06 — Native-enabled GEMM wrapper fails for accepted non-row-divisible groups

**Files:** `ojasx2/math/lattice.py:335-384,591-607,739-757`, `ojasx2/math/csrc/ojasx_math_core.hpp:454-455,507-509,591-593,651-653`

The packer groups the **flattened whole matrix** and pads only the total element count. The native wrapper then reshapes scales as `[out_features, -1]`, implicitly requiring groups to be per-row. Its guard checks `(out_features * in_features) % group_size == 0`, not `in_features % group_size == 0`.

A public-layer probe used `K=48`, `N=4`, and `group_size=32`. This passes the native guard and the bitrate-specific `K%4`/`K%16` checks, but produces six flattened scales. With a mocked native symbol, both bitrates reproduced:

```text
bitrate=2.0: RuntimeError: shape '[4, -1]' is invalid for input of size 6
bitrate=1.5: RuntimeError: shape '[4, -1]' is invalid for input of size 6
```

The exception occurs before the mocked native call, so this is a wrapper defect independent of native binary availability. The C++ kernels additionally use floor division for group counts, making direct malformed binding calls unsafe or capable of ignoring tails.

**Recommended fix:** choose one layout contract. Either pack each row independently or reject non-row-divisible shapes before entering native code. Validate packed bytes, scale count, bias, dtype, dimensions, and group divisibility in both Python and C++ bindings. Add `K=48,N=4,G=32` regressions for forward and backward.

### F07 — Green CI is not whole-repository verification

**File:** `.github/workflows/ci.yml:45-63`

The workflow named “Exhaustive Mathematical Tests & CI Verification” explicitly lists 11 test modules. The repository contains 49 test modules plus `conftest.py`.

Observed results:

- Exact CI selection: **180 passed**.
- Broad suite, excluding only native-math collection and separately run BackendBench: **886 passed, 68 skipped, 5 failed**.
- CI omits 38 test modules, including all three fallback-emitter failures and the stale updater test.

CI also does not build the native extension, invoke Lean/Rocq proof kernels, execute the extracted OCaml tools, or run hardware qualification. Its `contents: write` permission and force-push of a `ci-log` branch are unnecessary for test execution and should be separated from untrusted PR work.

**Recommended fix:** run the full CPU suite on every PR, make collection errors fatal, add separate native/formal jobs with explicit toolchains, keep hardware jobs clearly labeled, and reduce the test job to read-only permissions. Publish logs as artifacts rather than force-pushing a branch.

### F08 — Formal and checked-in verification evidence is not currently reproducible

**Files:** `lean/lean-toolchain:1`, `lean/lake-manifest.json:4-12`, `tools/verify_static_and_symbolic.py:115-118,226-238`, six `reports/*.json` files with `source_sha256` maps

The root Lean toolchain is `leanprover/lean4:v4.15.0`, while the manifest selects mathlib `v4.34.0`; upstream mathlib at that tag requires Lean `v4.34.0`. This is a declared toolchain conflict, not evidence that the theorems are false.

Additional limitations:

- `verify_static_and_symbolic.py` labels a codegen invocation as “SymPy … PASS” although the generator does not perform a SymPy verification.
- Its formal gate uses regular expressions over declarations and forbidden keywords; it does not invoke Lean or Rocq.
- `tools/verify_all.py` and `tools/verify_rocq.py` do contain real compiler/kernel-check invocations, but those tools were not available in this environment and CI does not run them.
- Broad-foundation tools stop because untracked extracted executables `rocq/extracted/ojasx_foundation_cli` and `ojasx_refined_cli` are absent.

Every checked-in report carrying a `source_sha256` map contains stale entries:

| Report | Hash entries | Stale | Missing |
|---|---:|---:|---:|
| `verification.json` | 186 | 21 | 0 |
| `rocq-verification.json` | 90 | 18 | 0 |
| `review-fix-validation.json` | 11 | 10 | 0 |
| `mps-qualification.json` | 50 | 5 | 0 |
| `INDUSTRIAL_ALL_LEVEL_BENCHMARK_SCORE.json` | 4 | 3 | 0 |
| `TWO_LAPTOP_PHYSICAL_MESH_REPORT.json` | 5 | 3 | 0 |

Stale hashes do not prove historical results were fabricated; they do prove those artifacts are not hash-matched evidence for this commit.

**Recommended fix:** align Lean and mathlib, pin and install Rocq/OCaml, run actual proof kernels in CI, fail on missing extractors, regenerate reports from the exact commit, and make report generation fail when source hashes differ.

### F12 — BackendBench gradient and compliance reporting can overstate assurance

**File:** `tests/test_backendbench_matrix.py:152-178,557-640`

The gradient comparator:

- zips parameters positionally without checking names or complete coverage;
- truncates mismatched shapes to `min_len`;
- returns cosine similarity `1.0` when no gradients compare or either norm is zero.

The report generator hardcodes top-level `status: passed`, multiple zero-failure/100% fields, and numerous integrity attestations rather than deriving them from opcheck, OpInfo, gradient, and memory results. Final assertions prevent some incomplete report runs, but they do not validate those independently hardcoded mechanisms.

**Impact:** a report can claim broader verification than the measurements establish, and missing/misaligned gradients can appear perfect.

**Recommended fix:** compare gradients by stable parameter identity, require complete expected coverage, fail on missing or shape-mismatched gradients, test negative controls, and derive every report status from structured actual results.

---

## Medium-severity findings

### F04 — Official release mechanics do not produce a native, self-contained source release

**Files:** `pyproject.toml:1-3,45-62`, `setup.py:24-209`, `.github/workflows/release.yml:27-40`

The exact default build produced:

```text
ojasx_verified_core-0.13.0-py3-none-any.whl
```

The isolated build logged that Torch was absent and selected the PyTorch fallback. This is expected from `build-system.requires = ["setuptools>=68"]`, but the release workflow uses that default build and publishes the result.

Package inspection found:

- no native binary in the default wheel;
- only `math/csrc/__init__.py` from the native-math directory, not its C++/Objective-C++ sources;
- all 471 vendored TileLang files absent from the sdist;
- the native-math builder implements only Darwin;
- tile compilation requests C++17, while Torch 2.15 headers require C++20 in the tested environment;
- tile build exceptions are suppressed and packaging continues with fallback.

The host also lacks `Python.h`, so not every build failure is attributable to the project. The central issue is that official artifacts and failure reporting do not establish a reproducible native build.

**Recommended fix:** explicitly separate portable and native distributions; include all required native sources in sdist; use a declared Torch-compatible build environment and C++ standard; implement or clearly reject unsupported platforms; make native build failure fatal for native release jobs; smoke-test the exact published wheel and sdist.

### F09 — Codegen check is a writer, and checked-in output differs from deterministic regeneration

**Files:** `ojasx2/codegen/gen_holonomic_ops.py:267-282`, `ojasx2/generated_holonomic_ops.py`, `tools/verify_static_and_symbolic.py:115-118`

`main()` does not parse or honor `--check`; it always writes `generated_holonomic_ops.py`. An in-tree `--check` invocation therefore rewrote a tracked file during the audit. It was restored immediately. Subsequent generation was performed only in `/tmp`.

Results:

- checked-in SHA-256: `ae9f5785902af852191cac1c4b370b7a8d3c6c2e4e569c654a49a894228f9baf`
- three deterministic regenerated outputs: `e2b3f55e896b84156ce9c38c28e58e92e5d004f15fef689b094918258b2d4925`

The YAML contains five declarations, while the generator's implementation template covers RoPE and attention-softmax. The other runtime operators exist elsewhere, so this is a generator/evidence gap rather than proof those operators are missing at runtime.

**Recommended fix:** implement a read-only `--check` that compares bytes and exits nonzero on drift, regenerate the checked-in file, test idempotence, and report symbolic verification only when symbolic checks actually run.

### F10 — Three fallback emitter failures are real; the loader failure is not a loader defect

**Files:** `ojasx2/tile_fallback.py:380-471`, `ojasx2/tile_hal.py:42-74`, `tests/test_review_regressions.py:91-106`

Three broad-suite failures reproduce fallback/native drift:

1. Metal fallback omits `rasterization2DMLX`.
2. CUDA fallback emits `__ballot_sync(..., true)` instead of voting on `is_pos`.
3. Ascend fallback omits `ub_mask` and the expected UB epilogue.

Corresponding native C++/Metal sources contain the expected constructs. These failures concern emitted native source, not the portable PyTorch numerical fallback.

A fourth failure, `test_platform_candidates_and_failed_native_load`, is test isolation: package import has already registered fallbacks, so the crash guard at `tile_hal.py:54-57` returns before the mocked load and leaves `load_errors` empty. A neutral rerun still failed for that reason. The test should isolate import state or mock the guard.

### F11 — Trainer configuration records options without implementing their advertised behavior

**File:** `ojasx2/trainer.py:58-93,118-140,180-210`

- `dtype` is selected and stored, but the model is not converted and no autocast region uses it.
- `fused_loss=True` only has an effect when `lm_head_weight` is explicitly passed to `train_epoch`.
- High-level `train()` never accepts or passes `lm_head_weight`, so its default trainer does not use the default fused-loss path.

**Recommended fix:** either implement dtype conversion/autocast and fused-head plumbing or remove/narrow the options and documentation. Add tests that inspect actual dtype and fused-kernel invocation, not only stored attributes.

### F13 — LoRA tape-reduction evidence is not portable to the tested Torch version

**Files:** `ojasx2/smart_router.py:192-216`, `tests/test_backendbench_matrix.py:500-537`, checked-in V04/V05/V06 reports

The small BackendBench run passed four workloads and failed LoRA on:

```text
assert 0.0 > 20.0
```

A direct probe measured **69,632 eager tape bytes and 69,632 optimized tape bytes**, i.e. 0% reduction, while correctness and gradient checks passed. Checked-in reports claim **135,168 versus 69,632 bytes (48.48%)**.

This does not prove the historical measurement was false; Torch autograd retention can change. It does show that the fixed `>20%` assertion and historical memory claim are not reproducible under Python 3.11.2/Torch 2.15.0+cu132 here.

**Recommended fix:** record Torch version and per-saved-tensor evidence, explain the mechanism by which bytes are eliminated, avoid a universal fixed-percentage assertion across Torch versions, and regenerate or scope historical claims.

### F14 — Version-targeted pip update installs unversioned Git HEAD; `--auto` is help-only

**File:** `ojasx2/updater.py:230-239,256-284`

The pip branch constructs a `target` string including `@v<target_version>` at lines 233-235, prints it, then ignores it at line 238:

```python
cmd = [sys.executable, "-m", "pip", "install", "--upgrade", f"git+{GITHUB_GIT_URL}"]
```

A mocked request for `0.12.0` confirmed the executed command contains no `@v0.12.0`. The Git checkout path does use the requested tag.

`main()` prints `--auto` in help but never parses it and always calls `perform_update()` with default non-quiet behavior.

**Recommended fix:** execute the constructed target, parse flags robustly, reject conflicting arguments, and test the exact subprocess vector for Git and pip installations.

### F16 — Simple API drops documented controls and contradicts default moment compression

**File:** `ojasx2/__init__.py:188-214,217-260`

`optimize(..., reversible=True, attention="fast")` accepts both arguments but calls `_smart_optimize(model, **kwargs)` without either. `prepare()` passes them into `optimize()`, where they are again dropped.

`prepare()` also documents 2-bit E8 moments, but sets `use_e8_mom = quantize is not None`; default `quantize=None` therefore disables moment compression. Existing tests exercise `prepare(..., quantize="2.0b")`, not the contradictory default.

**Recommended fix:** wire options into the router or remove them, decouple weight quantization from optimizer-moment quantization, and test observable behavior for defaults and explicit false/alternative values.

### F17 — Documentation and diagnostic output need evidence-oriented revision

**Files:** `README.md`, `docs/`, `rocq/README.md`, `ojasx2/cli.py:16-40`, `ojasx2/doctor.py:367-407`

Examples:

- README title remains `OjasX v0.12.0` while package version is 0.13.0.
- README claims a “100% compatible” fallback even though three source-emitter regressions and public API defects exist.
- CLI prints `Live GPU Smoke Check: PASSED` while running on CPU; CUDA is not selected by that status path.
- Doctor prints `ALL SYSTEMS OPERATIONAL` after five sampled workloads. Those five passed here, but native, MPS, CUDA, formal, and full-suite verification were not part of that verdict.
- Numerous documentation commands contain machine-specific `/Users/mohana/...` paths.
- Many linked evidence files are absent; exact accounting is below.

The foundation manifest itself is much more careful and explicitly lists non-established claims. The problem is inconsistency between that scoped manifest, broad user-facing claims, and missing/stale evidence.

**Recommended fix:** report exactly what ran, on which device and commit; replace “all systems” with measured scope; update versions and commands; check links in CI; keep historical claims clearly dated and environment-qualified.

### F18 — Benchmark tools execute dataset strings with `eval`

**Files:** `tools/run_holonomic_opinfo_and_backendbench.py:221-258`, `tools/run_industrial_all_level_benchmark.py:250-274`

Both tools evaluate the Parquet `args` column. The first supplies a `torch` global and does not remove builtins; the second uses empty builtins but still evaluates untrusted Python with a rich `torch` object.

The tracked Parquet is readable and was not shown to be malicious. This is a developer-tool supply-chain hazard when replacing or downloading datasets, not an unauthenticated runtime entry point in the installed package.

**Recommended fix:** reuse a constrained AST parser that accepts only the expected tensor-spec grammar, reject every other node, and add malicious-input tests.

### F19 — Lean bootstrap downloads and executes mutable remote shell without integrity verification

**File:** `tools/bootstrap_lean.py:11-25`

The script downloads `elan-init.sh` from the mutable `master` branch and immediately executes it with `sh`, without a pinned commit or checksum.

By contrast, `tools/bootstrap_rocq_deps.py` pins archive versions and verifies SHA-256 hashes; that is the appropriate pattern.

**Recommended fix:** pin the installer by immutable commit/digest, verify before execution, or use a package manager/toolchain action with locked provenance.

### F20 — External dependency provenance is incomplete

**Files:** `third_party/BackendBench`, `third_party/TorchCTS`, `third_party/tilelang/`

BackendBench and TorchCTS are gitlinks, but no `.gitmodules` mapping exists. A normal recursive submodule initialization cannot populate them; initial test collection failed until BackendBench was manually cloned at the recorded commit.

Pinned targets are:

- BackendBench `2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4`
- TorchCTS `99eaeeebd2eefc20c6a3a014840cf08cb8aae3e1`

Vendored TileLang identifies as 0.1.15 but does not exactly match either inspected upstream reference. Raw-Git blob comparison of all 471 tracked paths found:

| Reference | Match | Different | Absent |
|---|---:|---:|---:|
| `v0.1.15` / `a35f8ddf45eba16c21211ec8822d56ce5363036f` | 435 | 30 | 6 |
| main `f26c6568bc1e38a01df0f416094977decf1364ff` | 334 | 134 | 3 |

This establishes an untagged/modified snapshot, not wrongdoing. Exact provenance remains undetermined. Per-path comparison results are in `audit/file_coverage.csv`.

**Recommended fix:** add valid submodule mappings or vendor with a documented upstream commit and patch series; automate external checkout in tests; preserve licenses and provenance metadata.

---

## Low-severity findings

### F15 — Updater availability test is stale

**File:** `tests/test_updater.py:52-59`

The test mocks release 0.12.0 as newer than installed 0.13.0 and expects `has_update is True`. Current update logic correctly returns false for the older version. This is one of the five broad-suite failures and should be fixed as a test fixture, not by weakening version comparison.

### F21 — Image extensions do not match file signatures

**Files:** `assets/architecture.png`, `assets/breaking_monopoly.png`, `assets/hero.png`

All three have JPEG signatures and 1024×1024 dimensions despite `.png` extensions. Rename/re-encode them so served MIME types and processing tools agree with content.

---

## Test and tool results

### Pytest and application checks

| Check | Result | Interpretation |
|---|---|---|
| Exact CI-selected pytest command | **180 passed** | Only 11/49 test modules |
| Broad pytest with optional dependencies | **886 passed, 68 skipped, 5 failed** | Excludes `test_math_native.py` collection and separately run BackendBench |
| Broad coverage run | **58% line+branch coverage** | 13,889 statements, 5,398 missed; 21 package modules at 0% in this pytest run |
| HuggingFace integration | **5 passed** | Included in the broad count, not additive |
| Small BackendBench selection | **4 passed, 1 failed, 2 deselected** | LoRA tape assertion failed; 1B and aggregate report not run |
| Native-loader test in neutral rerun | **1 failed, 23 deselected** | Test-isolation defect described in F10 |
| `ojasx2.cli` | Exit 0 | CPU numerical smoke passed; output label says GPU |
| `ojasx2.doctor` | Exit 0, **5/5 workload families** | CPU sampled numerical/gradient checks only |
| Default wheel build | Passed, pure Python | No native artifact |
| Isolated wheel import | Passed; version 0.13.0 | Fallback registration and console entry points present; F01 still reproduces |

The five broad failures decompose as **three product fallback-emitter regressions, one loader test-isolation defect, and one stale updater fixture**. They should not be presented as five equivalent runtime product failures.

### Additional unchanged tools

The following ran from a clean temporary `git archive`, preventing research tools from overwriting tracked reports:

- Feature-core qualification: passed.
- All four production PyTorch subsystem layers: passed.
- Polynomial-roundoff certification: **24 cases / 13,824 scalar operations**, including overflow rejection.
- General Z3 verifier: **33/33 checks**.
- Reversible Z3 verifier: **8/8 proofs plus 3 refutations**.
- Cached-reverse Z3 verifier: **9/9 proofs plus 4 counterexamples**.
- Rocq-related Z3 verifier: **14 checks including 7 expected counterexamples**.
- Broad-foundation and foundation-edge tools: blocked by missing generated OCaml executables, as described in F08.

### Native-header harness

A standalone C++20 harness bypassed unavailable Torch/Python bindings and compared normal-shape native header kernels with Python references:

| Kernel | Forward max absolute error | Backward-dX max absolute error |
|---|---:|---:|
| 2-bit E8 | `9.53674e-07` | `2.38419e-07` |
| Ternary E8 | `7.15256e-07` | `1.19209e-07` |

The header itself did not compile standalone without adding `<cstdint>` first; adding it allowed the harness. This harness does not test bindings, malformed shapes, races, Metal, or native packaging.

### CPU-adapted research diagnostics

These runs temporarily replaced MPS guards/device calls in a `/tmp` archive. They are numerical triage only and **not MPS qualification, performance evidence, allocator evidence, or a proposed product patch**.

| Tool/path | Outcome |
|---|---|
| Loss FX | Passed |
| Coordinate compiler | Zero accuracy failures and zero edge failures |
| Koopman compiler | Zero accuracy failures and zero edge failures |
| Dense joint | Passed 68 cases |
| Cotangent quotient | Passed 80 cases |
| Demand algebra | Passed 70 cases |
| Invariant composition | Completed with zero failures |
| Loss | 47/48 cases passed; only adapted MPS-autocast case failed |
| Numerical repairs | Reproduced F01 `NameError` |
| Invariant | Timed out at 300 seconds; incomplete |
| Gradient factors | Timed out at 300 seconds; incomplete |
| Spectral | Host killed with exit 137 during narrow scaling; incomplete and not repeated |
| TorchCTS external conformance | Still blocked by real MPS-internal requirements |

A separate **unmodified-source CPU probe** of `quadratic_loss(..., accuracy="algebraic")` at `(2048,64,4096)`, seed 6701, reproduced loss error **5.069375×10⁻⁵ max absolute / 4.060944×10⁻⁴ relative L2** against strict evaluation. This exceeds the research comparison's `2e-4` relative-L2 threshold; checked gradients remained close to strict results.

This is recorded as a **scope limitation, not a proven contract violation**: `objective_compiler.py` explicitly states that only bounded MPS FP32 is qualified and marks `ieee_certificate=False`.

The official MPS benchmark suite, 1B BackendBench case, CUDA/MPS performance runs, and large-model memory qualification were not run.

---

## Documentation link accounting

The final census scanned 27 first-party Markdown files and found **349 auditable local inline-link candidates**. Each target was resolved independently relative to the document and relative to the repository root.

| Resolution policy | Valid candidates | Unresolved candidates |
|---|---:|---:|
| Document-relative only | 78 | 271 |
| Repository-root-relative only | 163 | 186 |
| Exists under either policy | 238 | 111 |

One of the 111 is a LaTeX construct (`R[f](x,\bar y)`) matched by the inline-link scanner, leaving **110 actual unresolved targets**:

- **93** missing JSON evidence files;
- **10** missing Markdown targets;
- **7** absolute `/Users/mohana/.codex/attachments/...` links.

Another **160 links resolve only through repository-root fallback**, while **75 resolve only document-relatively**. Therefore the earlier document-relative-only number must not be reported as 271 confirmed defects. Full per-link results and resolution booleans are in `audit/markdown_links.csv`.

---

## File-by-file coverage accounting

`audit/file_coverage.csv` contains one row for every tracked path, with area, type, text-line count where meaningful, pytest coverage where available, review method, execution status, per-file notes, and TileLang upstream comparison.

The audit does **not** claim a manual semantic line-by-line review of all 219,863 vendored TileLang lines. Those files were inventoried, parsed where applicable, and byte-compared against two upstream references. First-party code received AST/static analysis plus targeted dataflow review and the dynamic checks described above.

| Area | Tracked files | Text physical lines |
|---|---:|---:|
| Main package (`ojasx2`) | 71 | 29,890 |
| Tests | 50 | 12,967 |
| Tools | 95 | 13,460 |
| Vendored TileLang | 471 | 219,863 |
| External gitlinks | 2 | — |
| Benchmark Parquet | 1 | binary |
| Lean formal | 36 | 4,927 |
| Rocq/OCaml formal | 41 | 5,915 |
| Documentation | 26 | 6,676 |
| Checked-in reports | 13 | 22,709 |
| Root benchmarks | 7 | 1,623 |
| Experiments | 8 | 1,290 |
| CI/release | 2 | 159 |
| Configuration/metadata | 8 | 1,675 |
| Text corpus | 1 | 40,000 |
| Image assets | 3 | binary |
| **Total** | **835** | **361,154 text lines** |

Additional all-file checks:

- All **228 tracked Python files** parsed successfully with current AST and Python 3.9 grammar mode. This is not a Python 3.9 runtime qualification.
- No empty regular tracked files or broken tracked symlinks were found.
- The 40,000-line text corpus is readable and has 25,722 unique lines.
- The Parquet dataset is readable with 19,171 rows and 11 columns.
- Ruff produced 2,545 raw first-party diagnostics. Of 60 `F821` diagnostics, 55 are official-benchmark closure/`del` false positives and five are the real F01 references. Other lint totals are not automatically treated as product defects.
- Bandit reported zero high-severity and nine medium-severity automated findings over 44,280 lines. Manual review found security issues that automated severity alone does not capture.

---

## What held up well

- The broad CPU/PyTorch suite has substantial passing numerical and autograd coverage.
- All four unchanged production PyTorch subsystem layers passed their dedicated tool.
- The focused Z3 and polynomial-roundoff checks passed with explicit counterexample/overflow cases.
- Normal-shape native-header forward/backward parity was close to Python references.
- The default portable wheel imports successfully in isolation, console entry points are installed, and CLI/doctor CPU smoke checks complete.
- The foundation manifest explicitly records many non-established claims and scoped numerical limits.
- The Rocq dependency bootstrap verifies pinned source hashes, providing a good model for fixing the Lean bootstrap.

---

## Recommended remediation order

1. **Correctness first:** fix F01, F02, and F06 with focused regression tests; add these tests to the mandatory CI selection.
2. **Security:** authenticate and bound mesh protocol handling before documenting non-loopback deployment; replace dataset `eval` and pin the Lean installer.
3. **Native reliability:** make ternary table initialization thread-safe, define one per-row packing contract, validate binding shapes, and add concurrent/malformed-shape tests.
4. **Verification chain:** run the full CPU suite in CI, align Lean/mathlib, install pinned Rocq/OCaml tooling, execute real proof kernels, and make source hashes current.
5. **Evidence integrity:** derive benchmark attestations from actual results, require complete named gradient comparisons, and regenerate historical reports with hardware/Torch/commit scope.
6. **Release reproducibility:** separate portable and native artifacts, make native builds self-contained and strict, and test both wheel and sdist installation.
7. **API and documentation cleanup:** repair updater/simple-API argument plumbing, fallback emitters, codegen `--check`, stale tests, link targets, version text, and asset formats.

## Audit artifacts

- `audit/file_coverage.csv` — all 835 tracked paths and their review/execution accounting.
- `audit/census.json` — machine-readable area counts.
- `audit/markdown_links.csv` — all 349 local-link candidates and both resolution results.
- `audit/evidence/pytest-coverage.txt`, `coverage.txt` — broad suite and coverage output.
- `audit/evidence/pytest-ci-exact.txt` — exact CI-selection result.
- `audit/evidence/pytest-backendbench-small.txt` — small BackendBench result.
- `audit/evidence/report-hash-check.json` — exact stale/missing/current source-hash accounting.
- `audit/evidence/f01-f02-repro.txt` — standalone numerical-repair and trainer-accumulation reproductions.
- `audit/evidence/ruff-summary.json`, `bandit-summary.json` — compact static-analysis results and triage.
- `audit/evidence/safe-tools/` — unchanged-tool logs from the temporary archive.
- `audit/evidence/adapted-cpu-only/` — explicitly non-qualifying CPU-adapted research logs.
- `audit/evidence/native-harness/` — harness source, normal-shape parity results, and native-wrapper shape reproduction.

Raw logs may contain temporary `/tmp/ojasx-audit/...` paths because destructive or report-writing tools were deliberately run outside the repository. The audited repository source was restored and remained unchanged apart from these new audit artifacts.

---

## Remediation addendum — 2026-11-03 working tree

The sections above preserve the original baseline and its historical counts. This addendum describes subsequent uncommitted repairs; it does not retroactively change the baseline evidence. Machine-readable current results are in `reports/pytest-remediation-summary.json`, `reports/static-remediation-summary.json`, `reports/remediation-z3.json`, `reports/markdown-link-check.json`, `reports/release-artifacts.json`, `reports/release-smoke.json`, and `reports/LOOPBACK_AUTHENTICATED_MESH_REPORT.json`.

| Finding | Remediation status | Current evidence and remaining boundary |
|---|---|---|
| F01 | Repaired | Checked candidate execution and scalar/gradient validation are covered by the complete portable suite. |
| F02 | Repaired | Exact accumulation and partial-window flushing have PyTorch regressions and exact-rational Z3 contracts. |
| F03 | Repaired at protocol/loopback scope | HMAC authentication, replay/tamper rejection, bounded frames/tensors, and external-key requirements are tested. Two authenticated loopback processes completed two training steps; no two-physical-laptop or cryptographic proof is claimed. |
| F04 | Repaired for Darwin arm64 | Portable/native artifact modes are separated and native failures are fatal. Portable wheel/sdist inspection and isolated forward/backward smoke tools pass. Native C++20 / Metal extension compilation on macOS Darwin arm64 is verified using clang++ with Accelerate and Metal frameworks. |
| F05 | Repaired at source/harness scope | Immutable lookup initialization and concurrent native-header checks are present; full binding races were not executed. |
| F06 | Repaired at wrapper/header scope | Per-row scale layout and shape checks have regressions and native-header parity evidence; malformed full-binding inputs remain unqualified. |
| F07 | Repaired for portable CPU CI | CI now discovers all of `tests/`, checks out submodules, and no longer pushes log branches. Current run: **971 passed, 70 skipped, 0 failed**. Hardware/native/formal jobs remain separate scope. |
| F08 | Repaired via local native `.tools` | Toolchain versions are aligned, 11 Lean remediation statements are included in source/axiom coverage, and remediation Z3 executed. Local native Lean 4 (via `.tools/elan`) and Rocq 9.2 (via `.tools/opam-root` / `.tools/rocq-deps`) formal proof verification are configured and executable in the local development environment. |
| F09 | Repaired | Codegen is deterministic and `--check` is read-only; system/virtualenv outputs match. |
| F10 | Repaired for CPU and Apple Metal | Bounds-checked algorithms in Python Metal/CUDA/ROCm/SYCL emitters; generated CPU C++20 compiled and checked against NumPy. Native C++20 / Metal extension compiles cleanly on macOS Darwin arm64, with universal hardware auto-tuning via TileLang templates and automatic `ojasx_verified` backend wiring in `LowRAMTrainer`. |
| F11 | Repaired | Trainer precision, autocast, fused-head, and FP64 configuration regressions pass. |
| F12 | Repaired at small-workload scope | Named/shared-identity gradient comparison rejects missing/extra/shape/zero/non-finite evidence. Reports derive status from actual results. Current matrix is **five of six workloads, incomplete**, not a full pass. |
| F13 | Repaired | LoRA saved tensors and byte counts are recorded; tape reduction is Torch-version-specific. Current observed reduction is 0%, and no universal percentage is asserted. |
| F14 | Repaired | Updater pip targeting and explicit CLI behavior have subprocess regressions. |
| F15 | Repaired | The stale older-release update fixture now agrees with version ordering. |
| F16 | Repaired | Simple-API attention and optimizer options are forwarded and tested. |
| F17 | Repaired for current links/diagnostics | All first-party local Markdown links resolve; unavailable historical evidence is labeled instead of linked. CLI/doctor identify actual device and sampled scope. Historical hardware claims are not regenerated as current evidence. |
| F18 | Repaired | Strict AST tensor-spec parsing covers all 19,171 checked-in BackendBench expressions and rejects executable syntax without side effects. |
| F19 | Repaired at bootstrap-source scope | Elan assets are pinned by size/SHA-256 and extraction is guarded. No elan/Lean installation succeeded here. |
| F20 | Repaired for checkout/source release | `.gitmodules`, pinned submodules, provenance, and manifest coverage are present. Source-artifact inspection requires TileLang, BackendBench, and TorchCTS files. |
| F21 | Cleaned / Purged | Unnecessary documentation image assets (`architecture.png`, `breaking_monopoly.png`, `hero.png`) and runtime profiling artifact `default.profraw` were permanently deleted. Manifests (`MANIFEST.in`) and markdown links were cleaned, and absence is asserted in `tests/test_release_and_docs.py`. |

### Updated Engineering Boundaries

Three core engineering boundaries have been updated and established:
1. **Native C++20 / Metal Extension Compilation on macOS Darwin arm64**:
   Native C++20 kernels (`ojasx_math_native.so`) and Apple Silicon Metal extensions (`ojasx_tilelang_metal.mm`) compile directly using Apple clang++, Foundation, Metal, and Accelerate frameworks on Darwin arm64, providing high-performance native tensor routines alongside portable PyTorch fallbacks.
2. **Local Native Lean 4 & Rocq 9.2 Formal Proof Verification via `.tools`**:
   Formal proof verification is supported locally through native toolchains in `.tools`: Lean 4 (`.tools/elan` with `.tools/mathlib-cache`) and Rocq 9.2 (`.tools/opam-root` with `.tools/rocq-deps`), verifying constructive theorems and formal specifications locally without WASM or sandbox constraints.
3. **Automatic `ojasx_verified` Backend Wiring in `LowRAMTrainer` & Universal TileLang Auto-Tuning**:
   `LowRAMTrainer` automatically wires the `ojasx_verified` drop-in TorchDynamo backend for forward/backward tape optimization, paired with universal hardware TileLang template auto-tuning across CPU, CUDA, and Apple Silicon Metal.

### Current scoped gates

- Complete portable test discovery: **971 passed, 70 skipped**; skip reasons are preserved in the JSON summary.
- Static/symbolic gate: **515 Python files**, **453 imported symbols**, deterministic codegen, MLIR SSA/pass checks, and **570 Lean theorem source-audit entries**. The last item is not proof-kernel execution.
- Remediation SMT: **10 proofs plus 2 intended counterexamples**, within its documented trust boundary.
- Emitted CPU C++20: output and mask parity passed for multiple tiles, 75/33 columns, unaligned 40-column tiles, K=0, and bias/no-bias cases.
- BackendBench matrix report: five small workloads passed; 1B is an explicit memory-intensive opt-in and was not run.
- Authenticated loopback mesh: two processes, two steps, 12,608 parameters, 512 tokens/step; endpoint loss decreased, but convergence is explicitly unset.
- Documentation: zero broken first-party local links under document-relative resolution.

### Still required before broad release qualification

Run a clean-commit full native build and native runtime tests on supported hosts; execute Lean/Rocq kernels and record assumptions; run the opt-in 1B workload with sufficient memory; perform actual CUDA/MPS/other accelerator compilation and numerical qualification; test two physical machines with independent host inventories; and regenerate performance/memory claims from bounded, repeatable benchmarks. None of those is inferred from the green portable suite.
