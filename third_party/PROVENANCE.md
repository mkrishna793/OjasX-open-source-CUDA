# Third-party provenance

This directory contains external test dependencies and a vendored TileLang source snapshot. They are not first-party OjasX inventions. Preserve their licenses and upstream histories.

## BackendBench

- Upstream: <https://github.com/meta-pytorch/BackendBench>
- Checkout type: Git submodule at `third_party/BackendBench`.
- Pinned commit: `2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4`.
- License: `third_party/BackendBench/LICENSE.md` in the submodule.
- Used for external evaluation helpers and conformance workloads. Do not rewrite it as OjasX product code.

## TorchCTS

- Upstream: <https://github.com/TorchCTS/TorchCTS>
- Checkout type: Git submodule at `third_party/TorchCTS`.
- Pinned commit: `99eaeeebd2eefc20c6a3a014840cf08cb8aae3e1`.
- License and notices: preserved in the submodule.
- Hardware conformance still depends on the upstream backend requirements; a populated checkout alone is not a passing conformance result.

## TileLang

- Upstream: <https://github.com/tile-ai/tilelang>
- Checkout type: vendored source tree, not a Git submodule.
- The snapshot identifies itself as version 0.1.15, but it is **not byte-identical** to upstream tag `v0.1.15` (`a35f8ddf45eba16c21211ec8822d56ce5363036f`). Exact upstream provenance has not been established.
- The 2026-11-03 audit compared all 471 tracked vendored paths using raw Git blobs. Against `v0.1.15`: 435 matched, 30 differed, and 6 were absent. Against upstream main `f26c6568bc1e38a01df0f416094977decf1364ff`: 334 matched, 134 differed, and 3 were absent. These counts describe provenance, not correctness or wrongdoing.
- Upstream license and notices remain in `third_party/tilelang/`. The comparison counts above describe the original audit baseline, not the remediated working tree.
- Local remediation patch: `src/tl_templates/cpu/gemm.h` now initializes the full output mask and packs each positive result by global column. This fixes undefined shifts and missing/overwritten words for tiles wider than or not aligned to 32 columns. `tests/test_emitted_cpu_kernel.py` compiles the emitted C++20 source and checks outputs and every mask bit against NumPy. No upstream provenance is claimed for this local patch.

Run `git submodule update --init --recursive` after cloning to populate BackendBench and TorchCTS at the gitlink commits. Release and CI checkouts use the same mechanism.
