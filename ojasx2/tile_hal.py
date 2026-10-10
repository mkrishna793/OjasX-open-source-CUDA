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
from pathlib import Path
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


def ensure_tile_kernels_loaded() -> bool:
    """Load the compiled C++17 `libojasx_tile_kernels` shared library into `torch.ops.ojasx_hal`."""
    global _LOADED_TILE_DYLIB
    if _LOADED_TILE_DYLIB:
        return True
    for candidate in _TILE_LIB_CANDIDATES:
        if candidate.exists():
            torch.ops.load_library(str(candidate))
            _LOADED_TILE_DYLIB = True
            return True
    raise FileNotFoundError(
        f"Missing compiled C++ kernel library in {_CSRC_DIR}. "
        "Run `OJASX_REBUILD_NATIVE=1 python setup.py build_py` to build for your platform."
    )


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


def probe_hardware_profile() -> SiliconCapabilityProfile:
    info = torch.ops.ojasx_hal.probe_silicon_info()
    l1d, l2, line, cores, amx = [int(v) for v in info]
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
    """Computes L1/L2 cache-resident tile shapes (M, N, K) for a given workload."""

    def __init__(self, profile: Optional[SiliconCapabilityProfile] = None) -> None:
        self.profile = profile or probe_hardware_profile()

    def plan_gemm_tile(
        self, m: int, n: int, k: int, elem_bytes: int = 4, device_type: str = "mps"
    ) -> TileSchedule:
        l1_budget = max(16384, self.profile.l1d_cache_bytes // 2)
        candidates = [
            (128, 128, 64),
            (64, 128, 64),
            (64, 64, 64),
            (64, 64, 32),
            (32, 64, 32),
            (32, 32, 32),
            (16, 16, 16),
        ]
        chosen = (32, 32, 32)
        for tm, tn, tk in candidates:
            if tm > max(16, m) and tn > max(16, n):
                continue
            bytes_needed = (tm * tk + tk * tn + tm * tn) * elem_bytes
            if bytes_needed <= l1_budget:
                chosen = (min(tm, max(8, m)), min(tn, max(8, n)), min(tk, max(8, k)))
                break

        tm, tn, tk = chosen
        l1_used = (tm * tk + tk * tn + tm * tn) * elem_bytes
        l2_used = min(self.profile.l2_cache_bytes, (m * k + k * n) * elem_bytes)
        engine = (
            "Apple_Metal_SIMDGroup_Matrix_8x8"
            if device_type == "mps"
            else ("Apple_AMX_Coprocessor" if self.profile.has_apple_amx else "Vector_SIMD")
        )
        return TileSchedule(
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            l1_footprint_bytes=l1_used,
            l2_footprint_bytes=l2_used,
            xor_swizzle_banks=32,
            pipeline_stages=2 if l1_used * 2 <= self.profile.l1d_cache_bytes else 1,
            target_engine=engine,
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


def amx_linear_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.ojasx_hal.fused_amx_linear_relu2_fwd(x, weight, bias)


def l1_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
