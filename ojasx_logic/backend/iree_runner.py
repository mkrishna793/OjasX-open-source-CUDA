"""
Phase 4 & 5: Multi-Target IREE Compilation & Live `.vmfb` Runtime Bridge.

Compiles emitted standard MLIR into portable `.vmfb` binaries via `iree-compile`
and executes them live via `iree.runtime` (`metal` or `local-task`) when installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


TARGET_PRESETS: Dict[str, List[str]] = {
    "metal-spirv": [
        "--iree-hal-target-backends=metal-spirv",
    ],
    "llvm-cpu": [
        "--iree-hal-target-backends=llvm-cpu",
    ],
    "cuda": [
        "--iree-hal-target-backends=cuda",
    ],
    "rocm": [
        "--iree-hal-target-backends=rocm",
    ],
}

HAL_DRIVERS: Dict[str, str] = {
    "metal-spirv": "metal",
    "llvm-cpu": "local-task",
    "cuda": "cuda",
    "rocm": "rocm",
}


@dataclass(frozen=True)
class IREETargetConfig:
    target: str = "metal-spirv"
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IREECompileResult:
    status: str  # "ok", "skipped_no_iree", "error"
    target: str
    vmfb_path: Optional[str] = None
    elapsed_ms: float = 0.0
    command: str = ""
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class IREERunResult:
    status: str  # "ok", "skipped_no_iree", "error"
    target: str
    compile_ms: float = 0.0
    run_ms: float = 0.0
    output: Optional[np.ndarray] = None
    error: str = ""


def detect_iree_toolchain() -> Dict[str, Optional[str]]:
    """Inspect whether `iree-compile` CLI and `iree.runtime` Python package exist."""
    iree_compile = shutil.which("iree-compile")
    iree_runtime_ver: Optional[str] = None
    try:
        import iree.runtime as iree_rt  # type: ignore

        iree_runtime_ver = getattr(iree_rt, "__version__", "installed")
    except Exception:
        iree_runtime_ver = None
    return {
        "iree_compile": iree_compile,
        "iree_runtime": iree_runtime_ver,
    }


def compile_mlir_to_vmfb(
    mlir_text: str,
    output_vmfb: Optional[str] = None,
    *,
    target: str = "metal-spirv",
    extra_flags: Optional[List[str]] = None,
    timeout_sec: int = 120,
) -> IREECompileResult:
    """Compile a standard MLIR string to an IREE `.vmfb` binary if `iree-compile` is installed."""
    if target not in TARGET_PRESETS:
        raise ValueError(f"Unsupported IREE target '{target}'. Choose from {list(TARGET_PRESETS)}")

    iree_compile = shutil.which("iree-compile")
    if not iree_compile:
        return IREECompileResult(
            status="skipped_no_iree",
            target=target,
            stderr="iree-compile not found on PATH. Install via: pip install iree-base-compiler iree-base-runtime",
        )

    with tempfile.TemporaryDirectory(prefix="ojasx_iree_") as tmpdir:
        mlir_path = Path(tmpdir) / "module.mlir"
        mlir_path.write_text(mlir_text, encoding="utf-8")
        out_path = Path(output_vmfb) if output_vmfb else (Path(tmpdir) / f"module_{target}.vmfb")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [iree_compile, *TARGET_PRESETS[target]]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend([str(mlir_path), "-o", str(out_path)])

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if proc.returncode != 0:
            return IREECompileResult(
                status="error",
                target=target,
                elapsed_ms=elapsed_ms,
                command=" ".join(cmd),
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        return IREECompileResult(
            status="ok",
            target=target,
            vmfb_path=str(out_path) if output_vmfb else None,
            elapsed_ms=elapsed_ms,
            command=" ".join(cmd),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def run_mlir_kernel_iree(
    mlir_text: str,
    entry_fn: str,
    args: Sequence[np.ndarray],
    *,
    target: str = "metal-spirv",
) -> IREERunResult:
    """Compile MLIR -> .vmfb and execute `entry_fn` live on `iree.runtime` (`metal` or `local-task`)."""
    tc = detect_iree_toolchain()
    if not tc["iree_compile"] or not tc["iree_runtime"]:
        return IREERunResult(
            status="skipped_no_iree",
            target=target,
            error="iree-compile or iree-runtime not installed",
        )

    import iree.runtime as iree_rt  # type: ignore

    with tempfile.TemporaryDirectory(prefix="ojasx_iree_run_") as tmpdir:
        vmfb_file = str(Path(tmpdir) / f"{entry_fn}_{target}.vmfb")
        cres = compile_mlir_to_vmfb(mlir_text, output_vmfb=vmfb_file, target=target)
        if cres.status != "ok":
            return IREERunResult(
                status="error",
                target=target,
                compile_ms=cres.elapsed_ms,
                error=cres.stderr,
            )

        try:
            driver = HAL_DRIVERS[target]
            config = iree_rt.Config(driver)
            vm_module = iree_rt.VmModule.mmap(config.vm_instance, vmfb_file)
            ctx = iree_rt.SystemContext(config=config)
            ctx.add_vm_module(vm_module)
            mod_name = vm_module.name
            fn = getattr(ctx.modules, mod_name)[entry_fn]
            t0 = time.perf_counter()
            raw_out = fn(*args)
            out_np = np.asarray(raw_out.to_host())
            run_ms = (time.perf_counter() - t0) * 1000.0
            return IREERunResult(
                status="ok",
                target=target,
                compile_ms=cres.elapsed_ms,
                run_ms=run_ms,
                output=out_np,
            )
        except Exception as exc:
            return IREERunResult(
                status="error",
                target=target,
                compile_ms=cres.elapsed_ms,
                error=str(exc),
            )


__all__ = [
    "TARGET_PRESETS",
    "HAL_DRIVERS",
    "IREETargetConfig",
    "IREECompileResult",
    "IREERunResult",
    "detect_iree_toolchain",
    "compile_mlir_to_vmfb",
    "run_mlir_kernel_iree",
]
