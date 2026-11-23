"""Hardened C++20 native zero-leakage gate and Merkle ledger bindings via ctypes."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch


AEGIS_OK = 0
AEGIS_ERR_NULL_PTR = -1
AEGIS_ERR_INVALID_COUNT = -2
AEGIS_ERR_NON_FINITE = -3
AEGIS_ERR_INVALID_WEIGHT = -4
AEGIS_ERR_FLAG_WEIGHT_CONFLICT = -5
AEGIS_ERR_MASK_TAMPERED = -6
AEGIS_ERR_ZERO_LEAKAGE_FAILED = -7

_STATUS_NAMES = {
    AEGIS_OK: "AEGIS_OK",
    AEGIS_ERR_NULL_PTR: "AEGIS_ERR_NULL_PTR",
    AEGIS_ERR_INVALID_COUNT: "AEGIS_ERR_INVALID_COUNT",
    AEGIS_ERR_NON_FINITE: "AEGIS_ERR_NON_FINITE",
    AEGIS_ERR_INVALID_WEIGHT: "AEGIS_ERR_INVALID_WEIGHT",
    AEGIS_ERR_FLAG_WEIGHT_CONFLICT: "AEGIS_ERR_FLAG_WEIGHT_CONFLICT",
    AEGIS_ERR_MASK_TAMPERED: "AEGIS_ERR_MASK_TAMPERED",
    AEGIS_ERR_ZERO_LEAKAGE_FAILED: "AEGIS_ERR_ZERO_LEAKAGE_FAILED",
}

EXPECTED_GATE_VERSION = "ojasx-aegis-gate-1.0.0"
_MASK_DOMAIN = b"OJASX_AEGIS_MASK_V1:"

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_CPP_SOURCE = _CSRC_DIR / "ojasx_aegis_gate.cpp"

_LOAD_LOCK = threading.Lock()
_LOADED_LIBS: dict[Tuple[str, int], ctypes.CDLL] = {}


class NativeGateError(RuntimeError):
    """Fail-closed exception raised when native gate build, load, or verification fails."""


def default_library_name() -> str:
    if sys.platform == "darwin":
        return "libojasx_aegis.dylib"
    if sys.platform == "win32":
        return "ojasx_aegis.dll"
    return "libojasx_aegis.so"


def default_library_path(csrc_dir: Optional[Path] = None) -> Path:
    base = Path(csrc_dir) if csrc_dir is not None else _CSRC_DIR
    return base / default_library_name()


def _resolve_cxx_compiler() -> str:
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        resolved = shutil.which(env_cxx) or env_cxx
        if Path(resolved).exists() or shutil.which(resolved):
            return resolved
    for candidate in ("clang++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    raise NativeGateError("No C++20 compiler (clang++/g++/c++) found to build ojasx_aegis_gate")


def build_native_gate(
    csrc_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """Compile ojasx_aegis_gate.cpp into the hardened shared library if missing or stale."""
    base_dir = Path(csrc_dir) if csrc_dir is not None else _CSRC_DIR
    src_path = base_dir / "ojasx_aegis_gate.cpp"
    out_path = default_library_path(base_dir)

    if not src_path.is_file():
        raise NativeGateError(f"Native gate C++ source not found: {src_path}")

    needs_build = force or (not out_path.is_file())
    if not needs_build:
        try:
            needs_build = src_path.stat().st_mtime_ns > out_path.stat().st_mtime_ns
        except OSError:
            needs_build = True

    if not needs_build:
        return out_path

    cxx = _resolve_cxx_compiler()
    base_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_libojasx_aegis_",
        suffix=out_path.suffix,
        dir=str(base_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        cxx,
        "-O3",
        "-std=c++20",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        str(src_path),
        "-o",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise NativeGateError(
                f"Failed to compile native gate ({proc.returncode}): {proc.stderr.strip()}"
            )
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return out_path


def _bind_abi(lib: ctypes.CDLL) -> ctypes.CDLL:
    required = (
        "aegis_gate_version",
        "aegis_verify_mask_integrity",
        "aegis_apply_zero_mask_f32",
        "aegis_compute_merkle_step",
    )
    for sym in required:
        if not hasattr(lib, sym):
            raise NativeGateError(f"Native gate library missing required export: {sym}")

    lib.aegis_gate_version.argtypes = []
    lib.aegis_gate_version.restype = ctypes.c_char_p

    lib.aegis_verify_mask_integrity.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.aegis_verify_mask_integrity.restype = ctypes.c_int32

    lib.aegis_apply_zero_mask_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.aegis_apply_zero_mask_f32.restype = ctypes.c_int32

    lib.aegis_compute_merkle_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    lib.aegis_compute_merkle_step.restype = ctypes.c_int32
    return lib


def verify_native_gate_selftest(lib: ctypes.CDLL) -> None:
    """Strictly validate exported ABI semantics against fail-closed invariants."""
    ver_raw = lib.aegis_gate_version()
    if not ver_raw:
        raise NativeGateError("aegis_gate_version() returned empty")
    ver = ver_raw.decode("utf-8", errors="strict")
    if ver != EXPECTED_GATE_VERSION:
        raise NativeGateError(f"Unexpected native gate version: {ver!r} != {EXPECTED_GATE_VERSION!r}")

    # 1. Valid mask integrity + Python SHA-256 agreement
    w_ok = np.array([1.0, 0.0], dtype=np.float32)
    f_ok = np.array([0, 1], dtype=np.uint8)
    out_digest = np.zeros(32, dtype=np.uint8)
    expected_digest = hashlib.sha256(_MASK_DOMAIN + struct.pack(">Q", 2) + b"\x01\x00").digest()
    rc = lib.aegis_verify_mask_integrity(
        w_ok.ctypes.data,
        f_ok.ctypes.data,
        2,
        None,
        out_digest.ctypes.data,
    )
    if rc != AEGIS_OK or bytes(out_digest) != expected_digest:
        raise NativeGateError(f"Self-test failed valid mask check: rc={rc}")

    # 2. Fail-closed rejection of NaN, fractional weights, flag conflict, tampered digest
    w_nan = np.array([1.0, float("nan")], dtype=np.float32)
    if lib.aegis_verify_mask_integrity(w_nan.ctypes.data, None, 2, None, None) != AEGIS_ERR_NON_FINITE:
        raise NativeGateError("Self-test failed: NaN mask weight was not rejected")

    w_frac = np.array([1.0, 0.5], dtype=np.float32)
    if lib.aegis_verify_mask_integrity(w_frac.ctypes.data, None, 2, None, None) != AEGIS_ERR_INVALID_WEIGHT:
        raise NativeGateError("Self-test failed: fractional mask weight was not rejected")

    f_conflict = np.array([1, 0], dtype=np.uint8)
    if lib.aegis_verify_mask_integrity(w_ok.ctypes.data, f_conflict.ctypes.data, 2, None, None) != AEGIS_ERR_FLAG_WEIGHT_CONFLICT:
        raise NativeGateError("Self-test failed: flag/weight conflict was not rejected")

    bad_digest = np.full(32, 0xFF, dtype=np.uint8)
    if lib.aegis_verify_mask_integrity(w_ok.ctypes.data, f_ok.ctypes.data, 2, bad_digest.ctypes.data, None) != AEGIS_ERR_MASK_TAMPERED:
        raise NativeGateError("Self-test failed: tampered mask digest was not rejected")

    # 3. Zero-leakage masking even with +inf on blocked slot
    vals_in = np.array([3.5, float("inf")], dtype=np.float32)
    vals_out = np.full(2, -99.0, dtype=np.float32)
    rc_mask = lib.aegis_apply_zero_mask_f32(
        vals_in.ctypes.data,
        vals_out.ctypes.data,
        w_ok.ctypes.data,
        f_ok.ctypes.data,
        2,
        1,
        out_digest.ctypes.data,
        None,
    )
    if rc_mask != AEGIS_OK or vals_out[0] != np.float32(3.5) or vals_out.view(np.uint32)[1] != 0:
        raise NativeGateError("Self-test failed zero-leakage mask application")

    # 4. Merkle step SHA-256 vector cross-check
    b = [bytes([i] * 32) for i in range(1, 6)]
    bufs = [np.frombuffer(x, dtype=np.uint8).copy() for x in b]
    chain_out = np.zeros(32, dtype=np.uint8)
    rc_step = lib.aegis_compute_merkle_step(
        bufs[0].ctypes.data,
        bufs[1].ctypes.data,
        bufs[2].ctypes.data,
        bufs[3].ctypes.data,
        bufs[4].ctypes.data,
        7,
        chain_out.ctypes.data,
    )
    py_expected = hashlib.sha256(b"".join(b) + struct.pack(">Q", 7)).digest()
    if rc_step != AEGIS_OK or bytes(chain_out) != py_expected:
        raise NativeGateError("Self-test failed Merkle step SHA-256 verification")


def load_native_gate(
    *,
    csrc_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> ctypes.CDLL:
    """Load the compiled C++20 native gate, building it automatically if missing or stale."""
    with _LOAD_LOCK:
        lib_path = build_native_gate(csrc_dir=csrc_dir, force=force_rebuild)
        resolved = lib_path.resolve()
        mtime_ns = resolved.stat().st_mtime_ns
        cache_key = (str(resolved), mtime_ns)
        cached = _LOADED_LIBS.get(cache_key)
        if cached is not None:
            return cached

        try:
            lib = ctypes.CDLL(str(resolved))
        except OSError as exc:
            raise NativeGateError(f"Failed to load native gate library {resolved}: {exc}") from exc

        _bind_abi(lib)
        verify_native_gate_selftest(lib)
        _LOADED_LIBS[cache_key] = lib
        return lib


def _raise_for_status(status: int, context: str) -> None:
    if status == AEGIS_OK:
        return
    name = _STATUS_NAMES.get(status, f"UNKNOWN_ERROR({status})")
    raise NativeGateError(f"{context} rejected fail-closed by native gate: {name} ({status})")


def _prepare_weights_f32(weights: Any) -> np.ndarray:
    if isinstance(weights, torch.Tensor):
        if weights.ndim != 1:
            raise NativeGateError(f"weights must be 1D [B], got shape {tuple(weights.shape)}")
        if weights.dtype == torch.float32:
            arr = weights.detach().cpu().contiguous().numpy()
            return np.ascontiguousarray(arr, dtype=np.float32)
        flat = weights.detach().cpu().reshape(-1).tolist()
    elif isinstance(weights, np.ndarray):
        if weights.ndim != 1:
            raise NativeGateError(f"weights must be 1D [B], got shape {weights.shape}")
        if weights.dtype == np.float32:
            return np.ascontiguousarray(weights, dtype=np.float32)
        flat = weights.reshape(-1).tolist()
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, bytearray)):
        flat = list(weights)
    else:
        raise NativeGateError(f"Unsupported weights type: {type(weights).__name__}")

    if not flat:
        raise NativeGateError("weights must be non-empty")

    out = np.empty(len(flat), dtype=np.float32)
    for idx, val in enumerate(flat):
        if isinstance(val, bool):
            raise NativeGateError(f"weight[{idx}] must be float 0.0 or 1.0, not bool")
        fval = float(val)
        if not math.isfinite(fval):
            # Pass NaN/Inf through so C++ gate explicitly rejects non-finite mask too.
            out[idx] = np.float32(fval)
        elif fval not in (0.0, 1.0):
            raise NativeGateError(
                f"weight[{idx}] must be strictly 0.0 or 1.0 (precision-checked), got {val!r}"
            )
        else:
            out[idx] = np.float32(fval)
    return out


def _prepare_flags_u8(flags: Optional[Any], expected_len: int) -> Optional[np.ndarray]:
    if flags is None:
        return None
    if isinstance(flags, torch.Tensor):
        if flags.ndim != 1 or flags.shape[0] != expected_len:
            raise NativeGateError(f"flags shape mismatch: expected ({expected_len},), got {tuple(flags.shape)}")
        arr = flags.detach().cpu().to(dtype=torch.uint8).contiguous().numpy()
        return np.ascontiguousarray(arr, dtype=np.uint8)
    arr = np.asarray(flags)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise NativeGateError(f"flags shape mismatch: expected ({expected_len},), got {arr.shape}")
    if np.issubdtype(arr.dtype, np.bool_):
        return np.ascontiguousarray(arr.astype(np.uint8))
    if not np.issubdtype(arr.dtype, np.integer):
        raise NativeGateError(f"flags must be integer or bool sequence, got dtype {arr.dtype}")
    if np.any((arr < 0) | (arr > 255)):
        raise NativeGateError("flags entries must fit in uint8 [0, 255]")
    return np.ascontiguousarray(arr.astype(np.uint8))


def _parse_digest_hex(digest_hex: Optional[str], label: str) -> Optional[np.ndarray]:
    if digest_hex is None:
        return None
    if not isinstance(digest_hex, str) or len(digest_hex) != 64:
        raise NativeGateError(f"{label} must be a 64-char hex SHA-256 string")
    try:
        raw = bytes.fromhex(digest_hex)
    except ValueError as exc:
        raise NativeGateError(f"{label} contains invalid hex characters") from exc
    if len(raw) != 32:
        raise NativeGateError(f"{label} must decode to exactly 32 bytes")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def gate_version() -> str:
    """Return compiled native gate version string."""
    lib = load_native_gate()
    return lib.aegis_gate_version().decode("utf-8", errors="strict")


def verify_mask_integrity(
    weights: Any,
    flags: Optional[Any] = None,
    expected_mask_hash: Optional[str] = None,
) -> str:
    """Verify binary mask weights/flags in C++20 and return canonical SHA-256 hex digest."""
    lib = load_native_gate()
    w_np = _prepare_weights_f32(weights)
    f_np = _prepare_flags_u8(flags, w_np.shape[0])
    exp_np = _parse_digest_hex(expected_mask_hash, "expected_mask_hash")
    out_digest = np.zeros(32, dtype=np.uint8)

    rc = lib.aegis_verify_mask_integrity(
        w_np.ctypes.data,
        f_np.ctypes.data if f_np is not None else None,
        w_np.shape[0],
        exp_np.ctypes.data if exp_np is not None else None,
        out_digest.ctypes.data,
    )
    _raise_for_status(rc, "verify_mask_integrity")
    return bytes(out_digest).hex()


def compute_mask_hash(weights: Any, flags: Optional[Any] = None) -> str:
    """Validate mask weights/flags and return canonical mask SHA-256 hex digest."""
    return verify_mask_integrity(weights, flags=flags, expected_mask_hash=None)


def compute_merkle_step(
    prev_chain_hash: str,
    merkle_root: str,
    policy_hash: str,
    mask_hash: str,
    grad_sketch: str,
    step: int,
) -> str:
    """Compute H_t = SHA256(H_{t-1} || merkle || policy_hash || mask_hash || grad_sketch || step) in C++20."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise NativeGateError(f"step must be a non-negative integer, got {step!r}")
    lib = load_native_gate()

    prev_np = _parse_digest_hex(prev_chain_hash, "prev_chain_hash")
    merkle_np = _parse_digest_hex(merkle_root, "merkle_root")
    policy_np = _parse_digest_hex(policy_hash, "policy_hash")
    mask_np = _parse_digest_hex(mask_hash, "mask_hash")
    sketch_np = _parse_digest_hex(grad_sketch, "grad_sketch")
    assert prev_np is not None and merkle_np is not None and policy_np is not None
    assert mask_np is not None and sketch_np is not None

    out_np = np.zeros(32, dtype=np.uint8)
    rc = lib.aegis_compute_merkle_step(
        prev_np.ctypes.data,
        merkle_np.ctypes.data,
        policy_np.ctypes.data,
        mask_np.ctypes.data,
        sketch_np.ctypes.data,
        ctypes.c_uint64(step),
        out_np.ctypes.data,
    )
    _raise_for_status(rc, "compute_merkle_step")
    return bytes(out_np).hex()


def _apply_zero_mask_numpy_raw(
    values_np: np.ndarray,
    w_np: np.ndarray,
    f_np: Optional[np.ndarray],
    exp_np: Optional[np.ndarray],
) -> Tuple[np.ndarray, str]:
    lib = load_native_gate()
    if values_np.ndim < 1:
        raise NativeGateError("values must have at least 1 batch dimension")
    n = values_np.shape[0]
    if n != w_np.shape[0]:
        raise NativeGateError(
            f"Batch dimension mismatch: values.shape[0]={n} vs len(weights)={w_np.shape[0]}"
        )
    if n == 0 or values_np.size == 0 or (values_np.size % n) != 0:
        raise NativeGateError(f"Invalid tensor shape {values_np.shape} for batch size {n}")

    stride = values_np.size // n
    in_f32 = np.ascontiguousarray(values_np, dtype=np.float32)
    out_f32 = np.empty_like(in_f32)
    out_digest = np.zeros(32, dtype=np.uint8)

    rc = lib.aegis_apply_zero_mask_f32(
        in_f32.ctypes.data,
        out_f32.ctypes.data,
        w_np.ctypes.data,
        f_np.ctypes.data if f_np is not None else None,
        n,
        stride,
        exp_np.ctypes.data if exp_np is not None else None,
        out_digest.ctypes.data,
    )
    _raise_for_status(rc, "apply_zero_mask")
    return out_f32, bytes(out_digest).hex()


class _AegisTensorMaskOnly(torch.autograd.Function):
    """Tensor-only autograd wrapper (single tensor return for seamless backward)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        values: torch.Tensor,
        w_np: np.ndarray,
        f_np: Optional[np.ndarray],
        exp_np: Optional[np.ndarray],
    ) -> torch.Tensor:
        orig_device = values.device
        orig_dtype = values.dtype
        values_cpu = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
        out_f32, mask_hash = _apply_zero_mask_numpy_raw(values_cpu.numpy(), w_np, f_np, exp_np)
        out_tensor = torch.from_numpy(out_f32).to(device=orig_device, dtype=orig_dtype)
        ctx.w_np = w_np
        ctx.f_np = f_np
        ctx.mask_hash = mask_hash
        ctx.orig_device = orig_device
        ctx.orig_dtype = orig_dtype
        return out_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:  # type: ignore[override]
        if grad_output is None:
            return None, None, None, None
        exp_np = _parse_digest_hex(ctx.mask_hash, "saved_mask_hash")
        grad_cpu = grad_output.detach().to(device="cpu", dtype=torch.float32).contiguous()
        masked_np, _ = _apply_zero_mask_numpy_raw(grad_cpu.numpy(), ctx.w_np, ctx.f_np, exp_np)
        masked_grad = torch.from_numpy(masked_np).to(device=ctx.orig_device, dtype=ctx.orig_dtype)
        return masked_grad, None, None, None


def apply_zero_mask_with_hash(
    values: Union[torch.Tensor, np.ndarray, Sequence[Any]],
    weights: Any,
    flags: Optional[Any] = None,
    expected_mask_hash: Optional[str] = None,
) -> Tuple[Any, str]:
    """Enforce zero-leakage masking via C++20 native gate and return (masked, mask_hash)."""
    w_np = _prepare_weights_f32(weights)
    f_np = _prepare_flags_u8(flags, w_np.shape[0])
    exp_np = _parse_digest_hex(expected_mask_hash, "expected_mask_hash")

    if isinstance(values, torch.Tensor):
        if values.requires_grad:
            mask_hash = verify_mask_integrity(w_np, flags=f_np, expected_mask_hash=expected_mask_hash)
            masked = _AegisTensorMaskOnly.apply(values, w_np, f_np, _parse_digest_hex(mask_hash, "mask_hash"))
            return masked, mask_hash
        orig_device = values.device
        orig_dtype = values.dtype
        values_cpu = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
        out_f32, mask_hash = _apply_zero_mask_numpy_raw(values_cpu.numpy(), w_np, f_np, exp_np)
        return torch.from_numpy(out_f32).to(device=orig_device, dtype=orig_dtype), mask_hash

    if isinstance(values, np.ndarray):
        return _apply_zero_mask_numpy_raw(values, w_np, f_np, exp_np)

    arr = np.asarray(values, dtype=np.float32)
    out_f32, mask_hash = _apply_zero_mask_numpy_raw(arr, w_np, f_np, exp_np)
    return out_f32.tolist(), mask_hash


def apply_zero_mask(
    values: Union[torch.Tensor, np.ndarray, Sequence[Any]],
    weights: Any,
    flags: Optional[Any] = None,
    expected_mask_hash: Optional[str] = None,
) -> Any:
    """Enforce zero-leakage masking via C++20 native gate and return masked values."""
    masked, _ = apply_zero_mask_with_hash(
        values,
        weights,
        flags=flags,
        expected_mask_hash=expected_mask_hash,
    )
    return masked


__all__ = [
    "AEGIS_ERR_FLAG_WEIGHT_CONFLICT",
    "AEGIS_ERR_INVALID_COUNT",
    "AEGIS_ERR_INVALID_WEIGHT",
    "AEGIS_ERR_MASK_TAMPERED",
    "AEGIS_ERR_NON_FINITE",
    "AEGIS_ERR_NULL_PTR",
    "AEGIS_ERR_ZERO_LEAKAGE_FAILED",
    "AEGIS_OK",
    "EXPECTED_GATE_VERSION",
    "NativeGateError",
    "apply_zero_mask",
    "apply_zero_mask_with_hash",
    "build_native_gate",
    "compute_mask_hash",
    "compute_merkle_step",
    "default_library_name",
    "default_library_path",
    "gate_version",
    "load_native_gate",
    "verify_mask_integrity",
    "verify_native_gate_selftest",
]
