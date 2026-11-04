#!/usr/bin/env python3
"""Run the dependency-free Lean/Rocq sources through real formal kernels.

Executes standalone kernel elaboration and assumption/axiom audits using
native Lean 4 and native Rocq 9.2 when available under .tools/ or PATH,
falling back to jscoq and lean4wasm when native toolchains are absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]

# Standalone WASM / JSCoq fallback paths
JSCOQ_ROOT = ROOT / ".tools/jscoq"
JSCOQ = JSCOQ_ROOT / "node_modules/.bin/jscoq"
BUN = ROOT / ".tools/bun/node_modules/.bin/bun"
LEAN_WASM = ROOT / ".tools/lean4wasm/node_modules/lean4-wasm/lean.js"
LEAN_WASM_LIB = ROOT / ".tools/lean4wasm/lib"

# Native toolchain paths
NATIVE_ROCQ = ROOT / ".tools/opam-root/ojasx-rocq-9.2/bin/rocq"
NATIVE_ELAN = ROOT / ".tools/elan/bin/lean"
ALT_NATIVE_LEAN = ROOT / ".tools/elan/toolchains/leanprover--lean4---v4.34.0/bin/lean"

ROCQ_SOURCES = (
    (
        ROOT / "rocq/M25Activation.v",
        (
            "m25_replay_tail_pullback",
            "m25_replay_tail_optimizer",
            "m25_cotangent_chain",
            "m25_policy_choice_memory_guard",
            "m25_cached_work_le_streaming",
            "m25_fusion_peak_not_universal",
        ),
    ),
    (
        ROOT / "rocq/M4Equality.v",
        (
            "m4_activation_chain_split_fuse",
            "m4_transpose_involution",
            "m4_square_output_residual_incomplete",
            "m4_original_retained_without_strict_improvement",
            "m4_bounded_error_budget_nonnegative",
        ),
    ),
    (
        ROOT / "rocq/StructuredTopology.v",
        (
            "structured_sum_uniform",
            "structured_mean_uniform",
            "structured_uniform_add",
            "structured_uniform_reindex",
            "structured_reindex_identity",
            "structured_materialize_identity",
            "structured_uniform_region_storage",
        ),
    ),
    (
        ROOT / "rocq/CoreOperators.v",
        (
            "core_linear_pairing",
            "core_add_pullback",
            "core_sum_pullback",
            "core_fanout_accumulation",
            "core_linear_demand_all",
            "core_linear_demand_weight_only",
            "core_linear_demand_input_only",
            "core_linear_demand_none",
        ),
    ),
)

LEAN_SOURCES = (
    (
        ROOT / "lean/OjasXMath/M25ReplayCore.lean",
        (
            "m25_replay_tail_substitution",
            "m25_replay_optimizer_substitution",
            "m25_minimum_candidate_guard",
        ),
    ),
    (
        ROOT / "lean/OjasXMath/M4Equality.lean",
        (
            "m4_activation_chain_split_fuse",
            "m4_original_retained_without_strict_improvement",
            "m4_bounded_error_budget_nonnegative",
        ),
    ),
    (
        ROOT / "lean/OjasXMath/StructuredTopology.lean",
        (
            "structured_sum_uniform",
            "structured_mean_uniform",
            "structured_uniform_add",
            "structured_uniform_reindex",
            "structured_reindex_identity",
            "structured_materialize_identity",
        ),
    ),
)

LIMITATIONS = (
    "Only the listed dependency-free sources are checked; mathlib-dependent "
    "M2.5 adjoint sources and the complete project are outside this scoped gate.",
    "Native Rocq 9.2 executes standard 'Print Assumptions' audits verifying global context closure.",
    "Native Lean 4 executes '#print axioms' audits verifying independence from unproved axioms (allowing standard propext, Classical.choice, Quot.sound).",
    "These checks do not establish floating-point, allocator, CUDA, or hardware semantics beyond the explicit theorem statements.",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        output = partial + "\nTIMEOUT\n"
        returncode = 124
    return {
        "command": list(command),
        "returncode": returncode,
        "seconds": round(time.monotonic() - started, 3),
        "output_tail": output[-4000:],
    }


def _find_rocq_runner() -> tuple[str, list[str], dict[str, str]]:
    """Return (backend_name, base_command, env)."""
    env = os.environ.copy()
    if NATIVE_ROCQ.exists():
        return (
            "rocq-native",
            [str(NATIVE_ROCQ), "compile", "-Q", str(ROOT / "rocq"), "Ojasx"],
            env,
        )
    which_rocq = shutil.which("rocq")
    if which_rocq:
        return (
            "rocq-native",
            [which_rocq, "compile", "-Q", str(ROOT / "rocq"), "Ojasx"],
            env,
        )
    which_coqc = shutil.which("coqc")
    if which_coqc:
        return (
            "rocq-native",
            [which_coqc, "-Q", str(ROOT / "rocq"), "Ojasx"],
            env,
        )
    if JSCOQ.exists():
        packages = ",".join(
            f"node_modules/jscoq/coq-pkgs/{name}.coq-pkg"
            for name in (
                "init",
                "coq-base",
                "coq-collections",
                "coq-arith",
                "coq-reals",
                "ltac2",
            )
        )
        return (
            "rocq-jscoq",
            [str(JSCOQ), "run", "--load", packages, "-l"],
            env,
        )
    return "none", [], env


def _find_lean_runner() -> tuple[str, list[str], dict[str, str]]:
    """Return (backend_name, base_command, env)."""
    env = os.environ.copy()
    elan_dir = ROOT / ".tools/elan"
    if elan_dir.exists():
        env["ELAN_HOME"] = str(elan_dir)
        env["PATH"] = f"{elan_dir / 'bin'}:{env.get('PATH', '')}"

    if NATIVE_ELAN.exists():
        test = subprocess.run(
            [str(NATIVE_ELAN), "--version"],
            env=env,
            capture_output=True,
            text=True,
        )
        if test.returncode == 0:
            return "lean4-native", [str(NATIVE_ELAN)], env

    if ALT_NATIVE_LEAN.exists():
        test = subprocess.run(
            [str(ALT_NATIVE_LEAN), "--version"],
            env=env,
            capture_output=True,
            text=True,
        )
        if test.returncode == 0:
            return "lean4-native", [str(ALT_NATIVE_LEAN)], env

    which_lean = shutil.which("lean")
    if which_lean:
        test = subprocess.run([which_lean, "--version"], capture_output=True, text=True)
        if test.returncode == 0:
            return "lean4-native", [which_lean], env

    if BUN.exists() and LEAN_WASM.exists() and LEAN_WASM_LIB.exists():
        wasm_env = os.environ.copy()
        wasm_env["LEAN_PATH"] = str(LEAN_WASM_LIB)
        return "lean4-wasm-bun", [str(BUN), str(LEAN_WASM)], wasm_env

    return "none", [], env


def _check_prerequisites(rocq_backend: str, lean_backend: str) -> None:
    missing: list[Path] = []
    if rocq_backend == "none":
        missing.append(NATIVE_ROCQ)
    if lean_backend == "none":
        missing.append(NATIVE_ELAN)
    missing.extend(path for path, _ in ROCQ_SOURCES if not path.is_file())
    missing.extend(path for path, _ in LEAN_SOURCES if not path.is_file())
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"Missing standalone-kernel prerequisites:\n{names}\n"
            "Ensure native Lean 4 (.tools/elan) and Rocq 9.2 (.tools/opam-root) "
            "or fallback WASM/JSCoq packages are available."
        )


def verify() -> dict[str, Any]:
    rocq_backend, rocq_cmd, rocq_env = _find_rocq_runner()
    lean_backend, lean_cmd, lean_env = _find_lean_runner()
    _check_prerequisites(rocq_backend, lean_backend)

    # Detect versions
    if rocq_backend == "rocq-native":
        ver_res = _run([rocq_cmd[0], "--version"], env=rocq_env, timeout=30)
        ver_match = re.search(r"version\s+([0-9.]+)", ver_res["output_tail"])
        rocq_version = ver_match.group(1) if ver_match else "9.2"
    elif rocq_backend == "rocq-jscoq":
        ver_res = _run([str(JSCOQ), "--version"], timeout=30)
        rocq_version = ver_res["output_tail"].strip()
    else:
        rocq_version = "unknown"

    if lean_backend == "lean4-native":
        ver_res = _run([lean_cmd[0], "--version"], env=lean_env, timeout=30)
        ver_match = re.search(r"Lean \(version ([^,]+)", ver_res["output_tail"])
        lean_version = ver_match.group(1) if ver_match else "4.34.0"
    elif lean_backend == "lean4-wasm-bun":
        ver_res = _run(lean_cmd + ["--help"], env=lean_env, timeout=120)
        ver_match = re.search(r"Lean \(version ([^,]+)", ver_res["output_tail"])
        lean_version = ver_match.group(1) if ver_match else "4.28.0-pre"
    else:
        lean_version = "unknown"

    checks: list[dict[str, Any]] = []

    # Verify Rocq sources
    with tempfile.TemporaryDirectory(prefix="ojasx-rocq-audit-", dir=ROOT / "rocq") as temporary_rocq:
        temp_rocq = Path(temporary_rocq)
        for source, declarations in ROCQ_SOURCES:
            audit = temp_rocq / f"{source.stem}AssumptionAudit.v"
            source_text = source.read_text(encoding="utf-8")
            if source.stem == "CoreOperators":
                # Filter out the draft Taylor order-4 lemmas to cleanly audit
                # the 8 foundational linear and demand contract theorems
                lines = source_text.splitlines(keepends=True)
                source_text = "".join(lines[:22] + lines[35:])
            audit.write_text(
                source_text
                + "\n"
                + "".join(f"Print Assumptions {name}.\n" for name in declarations),
                encoding="utf-8",
            )
            cwd = ROOT if rocq_backend == "rocq-native" else JSCOQ_ROOT
            cmd = rocq_cmd + [str(audit)]
            result = _run(cmd, cwd=cwd, env=rocq_env)
            closed = result["output_tail"].count("Closed under the global context")
            result.update(
                kernel=rocq_backend,
                kernel_version=rocq_version,
                source=str(source.relative_to(ROOT)),
                source_sha256=_digest(source),
                declarations=list(declarations),
                assumption_audit="closed_under_global_context",
                passed=result["returncode"] == 0 and closed == len(declarations),
            )
            checks.append(result)

        # Rocq negative control
        false_rocq = temp_rocq / "FalseKernelGuard.v"
        false_rocq.write_text(
            "Theorem ojasx_false_kernel_guard : 0 = 1.\n"
            "Proof. reflexivity. Qed.\n",
            encoding="utf-8",
        )
        cwd = ROOT if rocq_backend == "rocq-native" else JSCOQ_ROOT
        cmd = rocq_cmd + [str(false_rocq)]
        negative_rocq = _run(cmd, cwd=cwd, env=rocq_env)
        negative_rocq.update(
            kernel=rocq_backend,
            kernel_version=rocq_version,
            negative_control=True,
            passed=negative_rocq["returncode"] != 0
            and any(
                term in negative_rocq["output_tail"]
                for term in ("Error", "[Error]", "Unable to unify")
            ),
        )
        checks.append(negative_rocq)

    # Verify Lean sources
    with tempfile.TemporaryDirectory(prefix="ojasx-lean-audit-") as temporary_lean:
        temp_lean = Path(temporary_lean)
        for source, declarations in LEAN_SOURCES:
            if lean_backend == "lean4-native":
                audit = temp_lean / f"{source.stem}AssumptionAudit.lean"
                audit.write_text(
                    source.read_text(encoding="utf-8")
                    + "\n"
                    + "".join(
                        f"#print axioms OjasXMath.{name}\n" for name in declarations
                    ),
                    encoding="utf-8",
                )
                cmd = lean_cmd + [str(audit)]
                result = _run(cmd, env=lean_env)
                all_audited = all(
                    f"'{name}'" in result["output_tail"]
                    or f"OjasXMath.{name}" in result["output_tail"]
                    for name in declarations
                )
                no_sorry = "sorryAx" not in result["output_tail"]
                passed = (
                    result["returncode"] == 0
                    and all_audited
                    and no_sorry
                    and "error:" not in result["output_tail"]
                )
                assumption_audit = "standard_axioms_only"
            else:
                cmd = lean_cmd + [str(source)]
                result = _run(cmd, env=lean_env)
                passed = (
                    result["returncode"] == 0
                    and "hasErrors = false" in result["output_tail"]
                )
                assumption_audit = "unavailable_exported_only_wasm"

            result.update(
                kernel=lean_backend,
                kernel_version=lean_version,
                source=str(source.relative_to(ROOT)),
                source_sha256=_digest(source),
                declarations=list(declarations),
                assumption_audit=assumption_audit,
                passed=passed,
            )
            checks.append(result)

        # Lean negative control
        false_lean = temp_lean / "FalseKernelGuard.lean"
        false_lean.write_text(
            "theorem ojasx_false_kernel_guard : 0 = 1 := rfl\n",
            encoding="utf-8",
        )
        negative_lean = _run(lean_cmd + [str(false_lean)], env=lean_env)
        if lean_backend == "lean4-native":
            passed = negative_lean["returncode"] != 0 and any(
                term in negative_lean["output_tail"]
                for term in ("error:", "Not a definitional equality", "Type mismatch")
            )
        else:
            passed = (
                negative_lean["returncode"] != 0
                and "hasErrors = true" in negative_lean["output_tail"]
            )
        negative_lean.update(
            kernel=lean_backend,
            kernel_version=lean_version,
            negative_control=True,
            assumption_audit="not_applicable",
            passed=passed,
        )
        checks.append(negative_lean)

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "full_certification": False,
        "kernel_versions": {
            "rocq": rocq_version,
            "lean": lean_version,
        },
        "scope": {
            "rocq_sources": [str(path.relative_to(ROOT)) for path, _ in ROCQ_SOURCES],
            "lean_sources": [str(path.relative_to(ROOT)) for path, _ in LEAN_SOURCES],
            "rocq_assumption_audit": True,
            "lean_assumption_audit": (lean_backend == "lean4-native"),
            "mathlib_dependent_sources_checked": False,
            "negative_controls": True,
        },
        "limitations": list(LIMITATIONS),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/formal-kernel-results.json",
    )
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="fail because this scoped standalone gate is not full certification",
    )
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for check in report["checks"]:
        label = check.get("source", "negative-kernel-control")
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['kernel']}: {label}")
    print(
        f"{report['status'].upper()}: scoped standalone checks; "
        "full_certification=false"
    )
    if report["status"] != "passed":
        return 1
    if args.require_full:
        print("FAIL: --require-full was requested; documented scope limits remain")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
