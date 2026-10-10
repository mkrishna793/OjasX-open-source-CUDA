# Rocq subproject

Latest formal-only audit: 133 closed statements across 12 proof modules, including generated quadratic/cubic feature identities and all coefficient/input pullback rules. Run `python -B tools/verify_nonlinear_foundation.py` from the project root to check proofs without executing CPU ML workloads. See [the GPU nonlinear research](../GPU_NONLINEAR_DISCOVERY.md). The older 100-statement report below describes the previous revision.

Current entry point: [scoped refinement 1.1](../FOUNDATION_REFINEMENT.md),
including compact normalization residuals, all affine gradients and finite
gather/scatter. The current audit contains 100 theorem/lemma statements across
11 proof modules. [Backend mathematics v1](../BACKEND_MATH_FOUNDATION_V1.md)
records the earlier foundation and cost models.

Read [the mathematical foundation and integration guide](../ROCQ_MATH_FOUNDATION.md).

Build order is specified by `_RocqProject`. `Audit.v` enumerates every lemma and
theorem in the 11 proof modules. `Extract.v` generates the exact OCaml reference
libraries. `extracted/cli.ml`, `extracted/foundation_cli.ml` and
`extracted/refined_cli.ml` are handwritten adapters.

Run from the project root:

```sh
/Users/mohana/.gemini/antigravity/scratch/venv/bin/python -B tools/verify_rocq.py
```

Evidence is saved in `../reports/rocq-verification.json`, `rocq-audit.txt`,
`rocq-kernel-check.txt`, `rocq-engine-tests.json` and `rocq-z3.json`.
`opam-switch.export` pins the locally installed package versions.

The extracted rational implementation is a correctness oracle. The matrix
functions do not implement materialized optimized GEMM execution.
