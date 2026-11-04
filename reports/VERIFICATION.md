# Verification status and evidence guide

This guide separates executable checks from historical reports and unavailable hardware/toolchain scope. The original defect baseline and finding-by-finding remediation status are in [AUDIT.md](../AUDIT.md).

## Current structured-core and M3/M4 evidence (2026-11-04 working tree)

- Complete portable test discovery passed **1305 tests with 77 explicit skips and no failures**. The skips retain optional-dependency, native-extension, CUDA/MPS, and opt-in 1B boundaries. Matching CPython 3.11.2 headers were supplied through `CPATH`, so both the explicit-operator and drop-in-backend Inductor tests executed rather than skipped. See `pytest-core-m3-m4.txt`.
- The combined M2/M2.5/M3/M4 regression passed **186 tests**, including 13 unchanged-model drop-in backend tests. These cover original-program fallback, strict discovered replacement, unsupported neighbors, RNG and mutation behavior, CPU AMP, dynamic shapes, meta tensors, noncontiguous inputs, AOT/Inductor, gradient accumulation, and SGD/Adam/AdamW/RMSprop training. A separate objective test replaces a region for memory but retains its native work-optimal form.
- `tools/smoke_dropin_backend.py` passed an unchanged native MLP through two AdamW steps with two accumulated microsteps each. Maximum loss, gradient, and final-parameter differences against eager PyTorch were **0**, **1.0842021724855044e-19**, and **3.469446951953614e-18**, respectively. Its original-program fallback output and gradient were bitwise equal. See `core-dropin-backend-smoke.json`. This is small-model CPU training evidence, not arbitrary whole-model coverage or a speed result.
- `tools/smoke_structured_cotangents.py` passed dense/structured bitwise differentials. Logical peaks changed **1280 → 1032** for the sigmoid case, **196 → 104** for reshape/transpose/sum, and **152 → 104** for additive fanout/sum. Their uniform-region representation savings are 248, 276, and 160 bytes respectively. Region and peak savings are distinct metrics. See `core-structured-smoke.json`.
- `tools/smoke_core_m4.py` discovered a four-sigmoid schedule beyond the hand-written M2.5 candidates: logical peak **456 → 392 bytes**, compared with M2.5 **400 bytes**. It generated 33 obligations and evaluated 32 candidates; selected/baseline output and input gradient were bitwise equal and a repeated plan was identical. The report now includes the requested objective, baseline/selected cost keys, and planner-proof references. See `core-m4-smoke.json`. These are logical ledger results, not hardware traffic or allocator benchmarks.
- M3 retains its explicit registered custom-operator path with fake kernels, autograd, and replay mutation checks. The new `ojasx_verified` backend rewrites eligible regions before AOT partitioning and replaces one only on a strict backward-inclusive objective improvement. Unsupported regions remain native, and ties retain the original program.
- `tools/verify_static_and_symbolic.py` passed all seven stages: 553 Python files, 583 imported symbols, deterministic codegen, MLIR SSA, Metal swizzle, symplectic reversibility, and 595 Lean theorem source-audit entries across 32 Rocq files. Its last stage remains a source audit, with kernel evidence reported separately. The Markdown audit checked 266 local links with no broken target; Python bytecode compilation and `git diff --check` also passed.
- Focused core Z3 passed **28/28** proof queries and **4/4** intended counterexamples, including exact structured representation/storage laws and the new original-fallback and bounded-error-budget planner laws. The independent main and remediation models previously passed **33/33** and **10/10**, respectively, with their expected counterexamples.
- `tools/verify_formal_kernels.py` executed eight scoped checks successfully. jsCoq checked the standalone M2.5/M4/structured Rocq files and all 18 declarations reported `Closed under the global context`. Bun/Lean **4.28.0-pre** checked dependency-free `M25ReplayCore.lean`, `M4Equality.lean`, and `StructuredTopology.lean`. Deliberately false Lean and Rocq controls were rejected. `formal-kernel-results.json` records source hashes and `full_certification=false`: mathlib-dependent sources and an executed Lean axiom audit are not covered by this WASM route.
- Earlier core-expansion evidence passed 208 focused tests and 1190 portable repository tests with 70 skips; those counts predate M3/M4 and remain historical. The full-suite nonlinear squeeze theorem now uses Z3's complete `qfnra-nlsat` procedure after the generic strategy returned a resource-sensitive `UNKNOWN`; its proposition and SAT/UNSAT acceptance rule are unchanged. Generic jet algebra is not a formal proof of every transcendental coefficient. See the [core guide](../docs/CORE_M0_M2.md), [M2.5 guide](../docs/CORE_M25.md), and [M3/M4 guide](../docs/CORE_M3_M4.md).

## Earlier remediation evidence (pre-core-expansion)

- Complete portable test discovery passed **971 tests with 70 explicit skips and no failures**. `pytest-remediation-summary.json` records the command, source-tree hash, counts, and skip reasons. This is CPU/PyTorch scope, not whole-platform verification.
- The independent static/symbolic gate passed with 515 Python files and 453 imported symbols checked. Its 570 Lean theorem entries are a **source audit only**; neither Lean nor Rocq proof kernels executed. See `static-remediation-summary.json`.
- `tools/verify_remediation_z3.py` produces `remediation-z3.json`: 10 exact SMT proofs and 2 intended counterexamples. Its scope is explicit real/integer/selection/state algebra, not Python refinement, cryptography, IEEE accuracy, or hardware performance.
- `lean/OjasXMath/Remediation.lean` adds 11 mathematical statements to the root source/axiom audit. Lean 4.34.0 is aligned with mathlib v4.34.0, but compiler qualification has not run in this environment.
- Generated CPU C++20 was compiled and executed against NumPy for GEMM + ReLU² + mask packing, including unaligned tiles and K=0. CUDA, ROCm, Metal, SYCL, and Ascend source paths were not hardware-qualified. The full Torch native extension was not built on this host.
- The BackendBench matrix report is derived from actual results: five small workloads passed and the 1B workload was not run, so status is **incomplete** and the 1B attestation is null. LoRA tape retention is an observation for the installed Torch version, not a universal percentage guarantee.
- An authenticated two-process loopback mesh completed two tiny training steps. `LOOPBACK_AUTHENTICATED_MESH_REPORT.json` labels this as non-physical scope, sets convergence to null, and records source hashes and actual telemetry.
- Portable wheel/sdist inspection and smoke results are in `release-artifacts.json` and `release-smoke.json`. Portable artifacts contain no native binary; this does not qualify the separate native artifact.

## Historical artifacts

Several checked-in JSON reports contain source hashes from earlier revisions. They are historical artifacts, not hash-matched proof for the current working tree. `audit/evidence/report-hash-check.json` records the baseline mismatches. Do not cite their pass counts, speedups, or memory reductions as current results without rerunning the exact source and environment.

Missing historical evidence links in the documentation have been replaced by explicit “evidence not included” text. `markdown-link-check.json` covers first-party local Markdown targets using the document-relative paths readers actually follow; it does not validate external URLs or regenerate missing experiments.

## Not established here

- No NVIDIA CUDA or Apple MPS benchmark qualification was performed.
- The drop-in backend currently rewrites supported activation regions; arbitrary whole-model optimization and hardware speedups are not established by the small-model training tests.
- No 1B-model run, large-model memory qualification, or production performance certification was performed.
- The full pinned Lean/mathlib project and Lean axiom audit were not executed. Only the scoped standalone Lean/Rocq kernel runs described above passed; OCaml extraction kernels were not executed. Z3 did execute.
- No full Torch native extension was built or smoke-tested on this host.
- No two-physical-laptop test or independent remote hardware inventory was performed.
- HMAC frame tests establish protocol behavior and rejection cases; they are not a cryptographic security proof or a replacement for TLS on untrusted networks.

## Reproduce the scoped checks

```sh
python -m pytest -q --tb=short --junitxml=reports/pytest-remediation.xml
python tools/smoke_dropin_backend.py
python tools/smoke_core_m4.py
python tools/smoke_structured_cotangents.py
python tools/verify_core_z3.py
python tools/verify_formal_kernels.py
python tools/verify_z3.py
python tools/verify_remediation_z3.py
python tools/verify_static_and_symbolic.py
python tools/check_markdown_links.py --json reports/markdown-link-check.json
python ojasx2/codegen/gen_holonomic_ops.py --check
```

The 1B workload is deliberately excluded by default after an exit-137 host kill identified its allocation point. Run it only with sufficient memory and an explicit opt-in:

```sh
OJASX_RUN_1B=1 python -m pytest tests/test_backendbench_matrix.py -q
```

For a portable release, build into a temporary artifact directory, then inspect and smoke-test the exact outputs with a Python environment containing the declared runtime dependencies:

```sh
OJASX_BUILD_NATIVE=0 python -m build --sdist --wheel --outdir /tmp/ojasx-dist
python tools/inspect_release_artifacts.py --artifact-dir /tmp/ojasx-dist --report reports/release-artifacts.json
python tools/smoke_release_artifacts.py --artifact-dir /tmp/ojasx-dist --report reports/release-smoke.json
```

A full current verification report must be generated from a clean commit with its source hashes, dependency versions, device, commands, skips, and failures recorded. A green targeted or portable run is not whole-repository, formal, native, or hardware verification.
