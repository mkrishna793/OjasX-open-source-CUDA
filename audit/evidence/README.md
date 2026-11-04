# Audit evidence guide

These files support `../../AUDIT.md`. They are evidence artifacts, not product fixes.

## Environment

- Audit commit: `1b0a2636491cbb451d82ebff6976220a473e84b1` (`v0.13.0`)
- Python 3.11.2; Torch 2.15.0+cu132; Transformers 5.22.1
- Linux x86_64, 2 CPUs, approximately 3.8 GiB RAM, no swap
- CUDA and MPS unavailable
- Lean, Rocq, OCaml, CMake/Ninja, and CUDA toolchains unavailable

## Primary results

- `pytest-coverage.txt`: final broad run — 886 passed, 68 skipped, 5 failed; excludes native-math collection and separately executed BackendBench.
- `coverage.txt`: coverage table for that run.
- `pytest-full.txt`: earlier broad run before optional Transformers dependencies — 884 passed, 70 skipped, 5 failed.
- `pytest-ci-exact.txt`: exact CI-selected modules — 180 passed.
- `pytest-hf.txt`: five HuggingFace tests; also included in the final broad count.
- `pytest-backendbench-small.txt`: four passed, one failed, two deselected; no 1B case.
- `native-loader-neutral.txt`: isolated triage of the loader test-isolation failure.
- `f01-f02-repro.txt`: standalone numerical-repair `NameError`, double accumulation scaling, and partial-cycle state reproductions.
- `report-hash-check.json`: current source-hash comparison for all six reports containing `source_sha256` maps.
- `ruff-summary.json` and `bandit-summary.json`: compact static-analysis summaries. Raw lint counts are not treated as proof of defects.
- `ruff-high-signal.txt`: selected high-signal Ruff rules.
- `local-path-leaks.txt`: machine-specific documentation paths.
- `build-default.log`: exact default wheel build.
- `package-artifacts.json`: wheel/sdist native-content inspection.
- `wheel-smoke.txt`: isolated installed-wheel version and numerical-repair reproduction.
- `codegen-check.json`: checked-in versus deterministic regenerated hashes.
- `tilelang-provenance.json`: raw-Git upstream comparison totals.
- `environment.json`: concise execution environment.
- `cli.txt` and `doctor.txt`: CPU CLI/doctor smoke output.

## Unchanged tools

`safe-tools/` contains logs from tools run against an unchanged temporary `git archive`. This prevented research tools from overwriting tracked reports. Passing logs are scoped to what each tool actually checks. The broad-foundation logs preserve missing-OCaml-executable failures.

## CPU-adapted diagnostics

`adapted-cpu-only/` contains logs from a separate temporary archive in which selected MPS guards/device calls were adapted to CPU. These are numerical diagnostics only. They are not MPS/CUDA qualification, speed evidence, memory evidence, or a proposed product patch.

Several logs intentionally preserve failure or incomplete outcomes: numerical-repair `NameError`, objective tolerance failures, spectral host termination, invariant/gradient-factor timeouts, and TorchCTS MPS-internal requirements. `summary.json` extracts selected structured outcomes and repeats this limitation.

## Native checks

`native-harness/harness.cpp` is the standalone C++20 normal-shape header harness. `results.txt` contains forward/backward parity errors for 2-bit and ternary kernels. The harness explicitly includes `<cstdint>` before the project header.

`native-shape-repro.txt` records the `K=48,N=4,G=32` wrapper failure for both bitrates. A mocked native symbol enables the existing Python native branch; the exception occurs in scale reshaping before any native call. This does not claim that the unavailable native extension itself was executed.

## Temporary paths

Logs can contain `/tmp/ojasx-audit/tool-run` or `/tmp/ojasx-audit/adapted-run`. Those directories were temporary clean archives used to isolate generated reports and source adaptations. Large wheels, sdists, external clones, binary fixtures, and compiled harness artifacts are not duplicated in Git.
