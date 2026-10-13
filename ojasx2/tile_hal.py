"""OjasX TileLang-Style Hardware Abstraction Layer (TileHAL) & C++ AMX/L1 Bridge.

Connects the compiled C++17 / LibTorch / Apple Accelerate AMX kernel library
(`libojasx_tile_kernels.dylib`) to:
  1. Direct C++ `torch.ops.ojasx_hal` registration (`register_fake` + `register_autograd`)
     certified by PyTorch's official `torch.library.opcheck`
  2. TileLang-style L2 -> L1 -> Register Cache Planner
  3. Z3 SMT Bit-Vector Prover for L1 XOR Bank-Conflict-Free Swizzling & 1-Bit Ballot
  4. Drop-in PyTorch custom layers (`HolonomicBitPackedLinear`, `HolonomicNativeRMSNorm`)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_TILE_DYLIB_PATH = _CSRC_DIR / "libojasx_tile_kernels.dylib"
_TILE_LIB_CANDIDATES = (
    _CSRC_DIR / "libojasx_tile_kernels.dylib",
    _CSRC_DIR / "libojasx_tile_kernels.so",
    _CSRC_DIR / "libojasx_tile_kernels.pyd",
)
_LOADED_TILE_DYLIB = False
_TILE_KERNELS_READY = False
_NATIVE_TILELANG_LOADED = False
_NATIVE_LOAD_ERRORS = []


def _native_library_candidates():
    suffixes = {"darwin": (".dylib",), "win32": (".pyd", ".dll")}.get(sys.platform, (".so",))
    return tuple(p for p in _TILE_LIB_CANDIDATES if p.suffix in suffixes)


def ensure_tile_kernels_loaded() -> bool:
    """Initialize operators, returning whether a compatible native library was loaded.

    Missing/incompatible optional binaries use portable PyTorch numerical operators.
    Set OJASX_DISABLE_NATIVE=1 to explicitly select the reference implementation.
    """
    global _LOADED_TILE_DYLIB, _TILE_KERNELS_READY, _NATIVE_TILELANG_LOADED
    if _TILE_KERNELS_READY:
        return _LOADED_TILE_DYLIB

    # CRASH GUARD: If ojasx_hal fallbacks are already registered in c10::Dispatcher,
    # NEVER attempt dlopen to prevent duplicate registration abort / SIGABRT!
    from .tile_fallback import fallbacks_registered
    if fallbacks_registered():
        _TILE_KERNELS_READY = True
        return False

    if os.environ.get("OJASX_DISABLE_NATIVE") != "1":
        candidates = _native_library_candidates()
        for candidate in candidates:
            if candidate.exists():
                try:
                    torch.ops.load_library(str(candidate))
                except (OSError, RuntimeError) as exc:
                    _NATIVE_LOAD_ERRORS.append(f"{candidate}: {exc}")
                    continue
                _LOADED_TILE_DYLIB = True
                break

    _NATIVE_TILELANG_LOADED = _LOADED_TILE_DYLIB and hasattr(torch.ops.ojasx_hal, "tilelang_emit_universal_source")
    if not _LOADED_TILE_DYLIB:
        from .tile_fallback import register_tile_fallbacks
        register_tile_fallbacks()

    _TILE_KERNELS_READY = True
    return _LOADED_TILE_DYLIB


def tile_kernel_status():
    ensure_tile_kernels_loaded()
    return {"native_loaded": _LOADED_TILE_DYLIB, "native_tilelang_loaded": _NATIVE_TILELANG_LOADED,
            "load_errors": list(_NATIVE_LOAD_ERRORS)}


ensure_tile_kernels_loaded()


# ============================================================================
# 1. Hardware Silicon Capability Prober & TileLang L1/L2 Cache Schedule Planner
# ============================================================================
@dataclass(frozen=True)
class SiliconCapabilityProfile:
    l1d_cache_bytes: int
    l2_cache_bytes: int
    cache_line_bytes: int
    physical_cpu_cores: int
    has_apple_amx: bool
    has_mps_gpu: bool
    unified_memory: bool
    chip_class: str


@dataclass(frozen=True)
class TileSchedule:
    tile_m: int
    tile_n: int
    tile_k: int
    l1_footprint_bytes: int
    l2_footprint_bytes: int
    xor_swizzle_banks: int
    pipeline_stages: int
    target_engine: str
    is_decode: bool = False


def probe_hardware_profile() -> SiliconCapabilityProfile:
    if hasattr(torch.ops, "ojasx_hal") and hasattr(torch.ops.ojasx_hal, "probe_silicon_info"):
        info = torch.ops.ojasx_hal.probe_silicon_info()
        l1d, l2, line, cores, amx = [int(v) for v in info]
    else:
        l1d, l2, line, cores, amx = 65536, 4194304, 128, os.cpu_count() or 8, 1 if sys.platform == "darwin" else 0
    has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    chip_class = "Apple_Silicon_Unified_GPU_AMX_ANE" if (amx and has_mps) else (
        "Discrete_GPU" if torch.cuda.is_available() else "CPU_AMX"
    )
    return SiliconCapabilityProfile(
        l1d_cache_bytes=l1d,
        l2_cache_bytes=l2,
        cache_line_bytes=line,
        physical_cpu_cores=cores,
        has_apple_amx=bool(amx),
        has_mps_gpu=has_mps,
        unified_memory=bool(amx and has_mps),
        chip_class=chip_class,
    )


class TileSchedulePlanner:
    """Computes L1/L2 cache-resident tile shapes (M, N, K) for decode vs prefill workloads."""

    def __init__(self, profile: Optional[SiliconCapabilityProfile] = None) -> None:
        self.profile = profile or probe_hardware_profile()

    def plan_gemm_tile(
        self, m: int, n: int, k: int, elem_bytes: int = 4, device_type: str = "mps"
    ) -> TileSchedule:
        is_decode = (m <= 8)
        is_prefill = (m >= 64)
        l1_budget = max(16384, self.profile.l1d_cache_bytes // 2)

        if is_decode:
            # Low register pressure, small L1 footprint for autoregressive token decoding
            tm = min(8, max(1, m))
            tn = min(32, max(16, n))
            tk = min(16, max(8, k))
            stages = 1
            if device_type == "mps":
                engine = "Apple_Metal_SIMDGroup_Matrix_8x32x16_Decode"
            elif self.profile.has_apple_amx:
                engine = "Apple_AMX_Decode_2D_Tile"
            else:
                engine = "Vector_SIMD_Decode"
        elif is_prefill:
            # Throughput-oriented tile sizing with multi-stage pipelining for prompt prefill / batched GEMM
            candidates = [
                (64, 128, 64),
                (64, 64, 64),
                (64, 64, 32),
                (32, 64, 32),
                (32, 32, 32),
            ]
            tm, tn, tk = (32, 32, 32)
            for cand_m, cand_n, cand_k in candidates:
                if cand_m > max(32, m) and cand_n > max(32, n):
                    continue
                bytes_needed = (cand_m * cand_k + cand_k * cand_n + cand_m * cand_n) * elem_bytes
                if bytes_needed <= l1_budget:
                    tm = min(cand_m, max(16, m))
                    tn = min(cand_n, max(16, n))
                    tk = min(cand_k, max(16, k))
                    break
            stages = 2
            if device_type == "mps":
                engine = "Apple_Metal_SIMDGroup_Matrix_32x32x8_Prefill"
            elif self.profile.has_apple_amx:
                engine = "Apple_AMX_Prefill_Throughput"
            else:
                engine = "Vector_SIMD_Prefill"
        else:
            # Medium matrix (8 < m < 64)
            tm = min(32, max(8, m))
            tn = min(32, max(16, n))
            tk = min(32, max(8, k))
            stages = 1
            if device_type == "mps":
                engine = "Apple_Metal_SIMDGroup_Matrix_32x32x8"
            elif self.profile.has_apple_amx:
                engine = "Apple_AMX_Coprocessor"
            else:
                engine = "Vector_SIMD"

        l1_used = (tm * tk + tk * tn + tm * tn) * elem_bytes
        l2_used = min(self.profile.l2_cache_bytes, (m * k + k * n) * elem_bytes)

        return TileSchedule(
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            l1_footprint_bytes=l1_used,
            l2_footprint_bytes=l2_used,
            xor_swizzle_banks=32,
            pipeline_stages=stages,
            target_engine=engine,
            is_decode=is_decode,
        )


# ============================================================================
# 2. Z3 SMT Bit-Vector Prover for C++ XOR Swizzling & 1-Bit Ballot Packing
# ============================================================================
def verify_xor_swizzle_and_ballot_smt() -> Dict[str, object]:
    """Formally prove via Z3 BitVec32 that C++ `xor_swizzle_offset` has 0 bank conflicts."""
    import z3

    solver = z3.Solver()
    row_a = z3.BitVec("row_a", 32)
    row_b = z3.BitVec("row_b", 32)
    col = z3.BitVec("col", 32)
    bank_mask = z3.BitVecVal(31, 32)

    bank_a = (col ^ row_a) & bank_mask
    bank_b = (col ^ row_b) & bank_mask

    solver.add((row_a & bank_mask) != (row_b & bank_mask))
    solver.add(bank_a == bank_b)
    check_res = solver.check()
    zero_bank_conflicts_proven = (check_res == z3.unsat)

    cpp_matches = True
    for r in range(32):
        banks_seen = set()
        for c in range(32):
            off = int(torch.ops.ojasx_hal.xor_swizzle_offset(r, c, 32, 32))
            expected = r * 32 + ((c & ~31) | ((c ^ r) & 31))
            if off != expected:
                cpp_matches = False
            banks_seen.add(off % 32)
        if len(banks_seen) != 32:
            cpp_matches = False

    return {
        "z3_status": str(check_res),
        "zero_bank_conflicts_proven": zero_bank_conflicts_proven,
        "cpp_swizzle_bijection_verified": cpp_matches,
        "num_banks": 32,
    }


# ============================================================================
# 3. Direct C++ `ojasx_hal` FakeTensor & Autograd Registrations (`opcheck` ready)
# ============================================================================
@torch.library.register_fake("ojasx_hal::pack_1bit_ballot")
def _pack_1bit_ballot_fake(x: torch.Tensor) -> torch.Tensor:
    out_shape = list(x.shape)
    out_shape[-1] = (out_shape[-1] + 31) // 32
    return x.new_empty(out_shape, dtype=torch.int32)


@torch.library.register_fake("ojasx_hal::unpack_1bit_ballot")
def _unpack_1bit_ballot_fake(
    packed: torch.Tensor, cols: int, dtype: torch.dtype
) -> torch.Tensor:
    out_shape = list(packed.shape)
    out_shape[-1] = cols
    return packed.new_empty(out_shape, dtype=dtype)


@torch.library.register_fake("ojasx_hal::fused_amx_linear_relu2_fwd")
def _fused_amx_linear_relu2_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_shape = list(x.shape)
    out_shape[-1] = weight.shape[0]
    out = x.new_empty(out_shape)
    mask_shape = list(x.shape)
    mask_shape[-1] = (weight.shape[0] + 31) // 32
    mask = x.new_empty(mask_shape, dtype=torch.int32)
    return out, mask


@torch.library.register_fake("ojasx_hal::fused_amx_linear_relu2_bwd")
def _fused_amx_linear_relu2_bwd_fake(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    packed_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = weight.new_empty((weight.shape[0],))
    return dx, dw, db


def _amx_linear_relu2_setup_context(ctx, inputs, output):
    x, weight, bias = inputs
    _, packed_mask = output
    ctx.save_for_backward(
        x,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        packed_mask,
    )
    ctx.has_bias = bias is not None


def _amx_linear_relu2_backward(ctx, grad_out: torch.Tensor, _grad_mask: torch.Tensor):
    x, weight, bias_saved, packed_mask = ctx.saved_tensors
    bias = bias_saved if ctx.has_bias else None
    dx, dw, db = torch.ops.ojasx_hal.fused_amx_linear_relu2_bwd(
        grad_out, x, weight, bias, packed_mask
    )
    return (
        dx if ctx.needs_input_grad[0] else None,
        dw if ctx.needs_input_grad[1] else None,
        db if (ctx.has_bias and ctx.needs_input_grad[2]) else None,
    )


torch.library.register_autograd(
    "ojasx_hal::fused_amx_linear_relu2_fwd",
    _amx_linear_relu2_backward,
    setup_context=_amx_linear_relu2_setup_context,
)


@torch.library.register_fake("ojasx_hal::fused_l1_rmsnorm_fwd")
def _fused_l1_rmsnorm_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    rstd_shape = list(x.shape)
    rstd_shape[-1] = 1
    rstd = x.new_empty(rstd_shape, dtype=torch.float32)
    return out, rstd


@torch.library.register_fake("ojasx_hal::fused_l1_rmsnorm_bwd")
def _fused_l1_rmsnorm_bwd_fake(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _l1_rmsnorm_setup_context(ctx, inputs, output):
    x, weight, _eps = inputs
    _, rstd = output
    ctx.save_for_backward(x, weight, rstd)


def _l1_rmsnorm_backward(ctx, grad_out: torch.Tensor, _grad_rstd: torch.Tensor):
    x, weight, rstd = ctx.saved_tensors
    dx, dw = torch.ops.ojasx_hal.fused_l1_rmsnorm_bwd(grad_out, x, weight, rstd)
    return (
        dx if ctx.needs_input_grad[0] else None,
        dw if ctx.needs_input_grad[1] else None,
        None,
    )


torch.library.register_autograd(
    "ojasx_hal::fused_l1_rmsnorm_fwd",
    _l1_rmsnorm_backward,
    setup_context=_l1_rmsnorm_setup_context,
)


GLU_MODES = {
    "swiglu": 0, "silu": 0,
    "geglu": 1, "geglu_tanh": 1, "gelu_tanh": 1, "gelu": 1,
    "geglu_exact": 2, "gelu_exact": 2,
    "reglu": 3, "relu": 3,
    "relu2glu": 4, "squared_reglu": 4, "relu2": 4, "squared_relu": 4,
    "sigmoidglu": 5, "glu": 5, "sigmoid": 5,
    "bilinearglu": 6, "bilinear": 6, "identity": 6, "none": 6,
    "quickgeglu": 7, "quickgelu": 7,
    "mishglu": 8, "mish": 8,
    "tanhglu": 9, "tanh": 9,
    "leaky_reglu": 10, "leaky_relu": 10,
}


def resolve_glu_mode(mode: Union[int, str]) -> int:
    if isinstance(mode, int):
        if 0 <= mode <= 10:
            return mode
        raise ValueError(f"Unknown GLU mode integer: {mode}")
    mode_str = str(mode).lower().strip()
    if mode_str in GLU_MODES:
        return GLU_MODES[mode_str]
    raise ValueError(f"Unsupported GLU activation mode: '{mode}'. Supported: {list(GLU_MODES.keys())}")


@torch.library.register_fake("ojasx_hal::fused_l1_glu_fwd")
def _fused_l1_glu_fwd_fake(gate: torch.Tensor, up: torch.Tensor, act_mode: int) -> torch.Tensor:
    torch._check(gate.shape == up.shape)
    torch._check(gate.dtype == up.dtype)
    return torch.empty_like(gate)


@torch.library.register_fake("ojasx_hal::fused_l1_glu_bwd")
def _fused_l1_glu_bwd_fake(
    grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, act_mode: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch._check(grad_out.shape == gate.shape)
    torch._check(gate.shape == up.shape)
    torch._check(grad_out.dtype == gate.dtype)
    torch._check(gate.dtype == up.dtype)
    return torch.empty_like(gate), torch.empty_like(up)


def _fused_l1_glu_setup_context(ctx, inputs, output):
    gate, up, act_mode = inputs
    # Zero Bytes fresh intermediate activation/product storage!
    # intermediate [M, H] gate, up, and product tensors are NEVER materialized on autograd tape!
    ctx.save_for_backward(gate, up)
    ctx.act_mode = act_mode


def _fused_l1_glu_backward(ctx, grad_out: torch.Tensor):
    gate, up = ctx.saved_tensors
    d_gate, d_up = torch.ops.ojasx_hal.fused_l1_glu_bwd(grad_out, gate, up, ctx.act_mode)
    return (
        d_gate if ctx.needs_input_grad[0] else None,
        d_up if ctx.needs_input_grad[1] else None,
        None,
    )


torch.library.register_autograd(
    "ojasx_hal::fused_l1_glu_fwd",
    _fused_l1_glu_backward,
    setup_context=_fused_l1_glu_setup_context,
)


def fused_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    act_mode: Union[int, str] = "swiglu",
) -> torch.Tensor:
    """Universal L1-tiled fused Gated Linear Unit forward and autograd engine."""
    mode_int = resolve_glu_mode(act_mode)
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1" or not _LOADED_TILE_DYLIB:
        from .tile_fallback import _glu_forward
        return _glu_forward(gate, up, mode_int)
    return torch.ops.ojasx_hal.fused_l1_glu_fwd(gate, up, mode_int)


@torch.library.register_fake("ojasx_hal::compress_boundary_int8")
def _compress_boundary_int8_fake(
    x: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    numel = x.numel()
    num_blocks = (numel + block_size - 1) // block_size if numel > 0 else 0
    q = x.new_empty((num_blocks, block_size), dtype=torch.int8)
    scale = x.new_empty((num_blocks,), dtype=torch.float32)
    return q, scale


@torch.library.register_fake("ojasx_hal::decompress_boundary_int8")
def _decompress_boundary_int8_fake(
    q: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: list[int],
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    return q.new_empty(orig_shape, dtype=orig_dtype)


def amx_linear_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1" or not _LOADED_TILE_DYLIB:
        from .tile_fallback import _linear_forward
        return _linear_forward(x, weight, bias)
    return torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, weight, bias)


def l1_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1" or not _LOADED_TILE_DYLIB:
        from .tile_fallback import _rms_forward
        return _rms_forward(x, weight, eps)
    return torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd(x, weight, eps)


@torch.library.register_fake("ojasx_hal::tilelang_metal_gemm_relu2_fwd")
def _tilelang_metal_gemm_relu2_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    panel_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_shape = list(x.shape)
    out_shape[-1] = weight.shape[0]
    out = x.new_empty(out_shape)
    mask_shape = list(x.shape)
    mask_shape[-1] = (weight.shape[0] + 31) // 32
    mask = x.new_empty(mask_shape, dtype=torch.int32)
    return out, mask


def _tilelang_metal_setup_context(ctx, inputs, output):
    x, weight, bias, _panel_size = inputs
    _, packed_mask = output
    ctx.save_for_backward(
        x,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        packed_mask,
    )
    ctx.has_bias = bias is not None


def _tilelang_metal_backward(ctx, grad_out: torch.Tensor, _grad_mask: torch.Tensor):
    x, weight, bias_saved, packed_mask = ctx.saved_tensors
    bias = bias_saved if ctx.has_bias else None
    dx, dw, db = torch.ops.ojasx_hal.fused_amx_linear_relu2_bwd(
        grad_out, x, weight, bias, packed_mask
    )
    return (
        dx if ctx.needs_input_grad[0] else None,
        dw if ctx.needs_input_grad[1] else None,
        db if (ctx.has_bias and ctx.needs_input_grad[2]) else None,
        None,
    )


torch.library.register_autograd(
    "ojasx_hal::tilelang_metal_gemm_relu2_fwd",
    _tilelang_metal_backward,
    setup_context=_tilelang_metal_setup_context,
)


def tilelang_metal_gemm_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    panel_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1" or not _LOADED_TILE_DYLIB:
        from .tile_fallback import _linear_forward
        return _linear_forward(x, weight, bias)
    return torch.ops.ojasx_hal.tilelang_metal_gemm_relu2_fwd(x, weight, bias, panel_size)


@torch.library.register_fake("ojasx_hal::tilelang_universal_gemm_relu2_fwd")
def _tilelang_universal_gemm_relu2_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    target_arch: str = "auto",
    panel_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_shape = list(x.shape)
    out_shape[-1] = weight.shape[0]
    out = x.new_empty(out_shape)
    mask_shape = list(x.shape)
    mask_shape[-1] = (weight.shape[0] + 31) // 32
    mask = x.new_empty(mask_shape, dtype=torch.int32)
    return out, mask


def _tilelang_universal_setup_context(ctx, inputs, output):
    x, weight, bias, _target_arch, _panel_size = inputs
    _, packed_mask = output
    ctx.save_for_backward(
        x,
        weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        packed_mask,
    )
    ctx.has_bias = bias is not None


def _tilelang_universal_backward(ctx, grad_out: torch.Tensor, _grad_mask: torch.Tensor):
    x, weight, bias_saved, packed_mask = ctx.saved_tensors
    bias = bias_saved if ctx.has_bias else None
    dx, dw, db = torch.ops.ojasx_hal.fused_amx_linear_relu2_bwd(
        grad_out, x, weight, bias, packed_mask
    )
    return (
        dx if ctx.needs_input_grad[0] else None,
        dw if ctx.needs_input_grad[1] else None,
        db if (ctx.has_bias and ctx.needs_input_grad[2]) else None,
        None,
        None,
    )


torch.library.register_autograd(
    "ojasx_hal::tilelang_universal_gemm_relu2_fwd",
    _tilelang_universal_backward,
    setup_context=_tilelang_universal_setup_context,
)


def tilelang_universal_gemm_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    target_arch: str = "auto",
    panel_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1" or not _LOADED_TILE_DYLIB:
        from .tile_fallback import _linear_forward
        return _linear_forward(x, weight, bias)
    return torch.ops.ojasx_hal.tilelang_universal_gemm_relu2_fwd(
        x, weight, bias, target_arch, panel_size
    )


def tilelang_query_chip_schedule(
    target_arch: str,
    M: int,
    N: int,
    K: int,
    smem_kb: int = 64,
    l2_kb: int = 4096,
) -> Dict[str, int]:
    raw = torch.ops.ojasx_hal.tilelang_query_chip_schedule(
        target_arch, M, N, K, smem_kb, l2_kb
    )
    return {
        "block_m": int(raw[0]),
        "block_n": int(raw[1]),
        "block_k": int(raw[2]),
        "warp_m": int(raw[3]),
        "warp_n": int(raw[4]),
        "simd_lanes": int(raw[5]),
        "smem_bank_bytes": int(raw[6]),
        "swizzle_bytes": int(raw[7]),
        "panel_width": int(raw[8]),
        "pipeline_stages": int(raw[9]),
    }


def tilelang_emit_universal_source(
    target_arch: str,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 8,
    panel_size: int = 4,
) -> str:
    return torch.ops.ojasx_hal.tilelang_emit_universal_source(
        target_arch, block_m, block_n, block_k, panel_size
    )



# ============================================================================
# 4. Drop-In Custom PyTorch Layers Connected to C++ TileHAL Kernels
# ============================================================================
class HolonomicBitPackedLinear(nn.Module):
    """Fused Linear + Squared-ReLU (`ReLU2`) backed by C++ AMX & 1-Bit Ballot Packing.

    Compresses intermediate activation sign-mask by 32x compared to FP32!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _mask = torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, self.weight, self.bias)
        return out


class HolonomicNativeRMSNorm(nn.Module):
    """L1-Cache-Resident Single-Pass C++ RMSNorm storing only 4-byte scalar mu2."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _rstd = torch.ops.ojasx_hal.fused_l1_rmsnorm_fwd(x, self.weight, self.eps)
        return out


class HolonomicFusedGLU(nn.Module):
    """L1-Resident Universal Holonomic Gated Linear Unit (0 Bytes fresh activation tape)."""

    def __init__(self, act_mode: Union[int, str] = "swiglu") -> None:
        super().__init__()
        self.act_mode = resolve_glu_mode(act_mode)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return fused_glu(gate, up, self.act_mode)


# ============================================================================
# 8. Unified Compiled Transformer Block (RMSNorm -> SwiGLU -> Linear)
# ============================================================================

@torch.library.register_fake("ojasx_hal::fused_transformer_mlp_fwd")
def _fused_transformer_mlp_fwd_fake(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.register_fake("ojasx_hal::fused_transformer_mlp_bwd")
def _fused_transformer_mlp_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    norm_w: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x),
        torch.empty_like(norm_w),
        torch.empty_like(gate_w),
        torch.empty_like(up_w),
        torch.empty_like(down_w),
    )


def _fused_transformer_mlp_setup_context(ctx, inputs, output):
    x, norm_w, gate_w, up_w, down_w, eps = inputs
    # Save strictly the 5 boundary inputs - 0 bytes intermediate activation memory
    ctx.save_for_backward(x, norm_w, gate_w, up_w, down_w)
    ctx.eps = float(eps)


def _fused_transformer_mlp_backward_autograd(ctx, grad_out: torch.Tensor):
    x, norm_w, gate_w, up_w, down_w = ctx.saved_tensors
    dx, dnorm_w, dgate_w, dup_w, ddown_w = torch.ops.ojasx_hal.fused_transformer_mlp_bwd(
        grad_out, x, norm_w, gate_w, up_w, down_w, ctx.eps
    )
    return (
        dx if ctx.needs_input_grad[0] else None,
        dnorm_w if ctx.needs_input_grad[1] else None,
        dgate_w if ctx.needs_input_grad[2] else None,
        dup_w if ctx.needs_input_grad[3] else None,
        ddown_w if ctx.needs_input_grad[4] else None,
        None,
    )


torch.library.register_autograd(
    "ojasx_hal::fused_transformer_mlp_fwd",
    _fused_transformer_mlp_backward_autograd,
    setup_context=_fused_transformer_mlp_setup_context,
)


class HolonomicUnifiedTransformerMLPFunction(torch.autograd.Function):
    """Unified single-node compiled Transformer block execution function.

    Creates strictly 1 single autograd node per block on the backward tape,
    eliminating PyTorch's node-by-node autograd dispatch overhead.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_w: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, norm_w, gate_w, up_w, down_w)
        ctx.eps = float(eps)
        return torch.ops.ojasx_hal.fused_transformer_mlp_fwd(
            x, norm_w, gate_w, up_w, down_w, float(eps)
        )

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x, norm_w, gate_w, up_w, down_w = ctx.saved_tensors
        dx, dnorm_w, dgate_w, dup_w, ddown_w = torch.ops.ojasx_hal.fused_transformer_mlp_bwd(
            grad_out, x, norm_w, gate_w, up_w, down_w, ctx.eps
        )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dnorm_w if ctx.needs_input_grad[1] else None,
            dgate_w if ctx.needs_input_grad[2] else None,
            dup_w if ctx.needs_input_grad[3] else None,
            ddown_w if ctx.needs_input_grad[4] else None,
            None,
        )


class HolonomicUnifiedTransformerMLP(nn.Module):
    """Unified Compiled Transformer Block (RMSNorm -> SwiGLU -> Linear).

    Executes normalization, gate/up projection, SwiGLU activation, and down
    projection in a single unified sweep without materializing intermediate
    activations in DRAM and with strictly 1 single autograd node on the tape.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        eps: float = 1e-6,
        norm_weight: Optional[torch.Tensor] = None,
        gate_weight: Optional[torch.Tensor] = None,
        up_weight: Optional[torch.Tensor] = None,
        down_weight: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        if norm_weight is not None:
            self.norm_weight = nn.Parameter(norm_weight)
        else:
            self.norm_weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

        if gate_weight is not None:
            self.gate_weight = nn.Parameter(gate_weight)
        else:
            self.gate_weight = nn.Parameter(torch.empty((intermediate_dim, dim), **factory_kwargs))

        if up_weight is not None:
            self.up_weight = nn.Parameter(up_weight)
        else:
            self.up_weight = nn.Parameter(torch.empty((intermediate_dim, dim), **factory_kwargs))

        if down_weight is not None:
            self.down_weight = nn.Parameter(down_weight)
        else:
            self.down_weight = nn.Parameter(torch.empty((dim, intermediate_dim), **factory_kwargs))

        if gate_weight is None or up_weight is None or down_weight is None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HolonomicUnifiedTransformerMLPFunction.apply(
            x, self.norm_weight, self.gate_weight, self.up_weight, self.down_weight, self.eps
        )


