"""
Zero-Copy Apple Silicon MPS Runtime Dispatcher for Fused Metal 4 E8 Dequant-GEMM Kernels.

Compiles `ojasx3/hardware/metal_e8_kernels.metal` directly via `torch.mps.compile_shader`
(PyTorch >= 2.4 on Apple Silicon) and dispatches on MPS tensors with ZERO CPU roundtrips
and ZERO full-weight FP16/FP32 DRAM materialization.
Falls back cleanly (returns None) on non-MPS devices or older PyTorch builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import threading
import torch

_METAL_LIB_LOCK = threading.Lock()
_METAL_LIB: Optional[Any] = None
_METAL_INIT_FAILED: bool = False


def _get_metal_e8_lib() -> Optional[Any]:
    global _METAL_LIB, _METAL_INIT_FAILED
    if _METAL_INIT_FAILED:
        return None
    if _METAL_LIB is not None:
        return _METAL_LIB
    if not (hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader")):
        _METAL_INIT_FAILED = True
        return None
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        _METAL_INIT_FAILED = True
        return None
    with _METAL_LIB_LOCK:
        if _METAL_LIB is not None:
            return _METAL_LIB
        if _METAL_INIT_FAILED:
            return None
        try:
            shader_path = Path(__file__).with_name("metal_e8_kernels.metal")
            source = shader_path.read_text(encoding="utf-8")
            _METAL_LIB = torch.mps.compile_shader(source)
            return _METAL_LIB
        except Exception:
            _METAL_INIT_FAILED = True
            return None


def metal_fused_e8_2bit_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> Optional[torch.Tensor]:
    """Run fused 2.0-bit E8 dequant-GEMM directly on Apple Silicon MPS GPU.
    Returns None if MPS shader compilation is unavailable or shapes are incompatible.
    """
    if x.device.type != "mps":
        return None
    if in_features % 4 != 0 or in_features % group_size != 0:
        return None
    lib = _get_metal_e8_lib()
    if lib is None or not hasattr(lib, "fused_e8_2bit_gemm_fwd"):
        return None

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d = x.reshape(-1, in_features).to(torch.float32).contiguous()
    M = int(x_2d.shape[0])
    K = int(in_features)
    N = int(out_features)
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((*orig_shape[:-1], N), dtype=orig_dtype, device=x.device)

    w_2d = packed_weight.reshape(N, K // 4).contiguous()
    s_2d = block_scales.reshape(N, K // group_size).to(torch.float16).contiguous()
    has_bias = 1 if bias is not None else 0
    b_t = bias.to(device=x.device, dtype=torch.float32).contiguous() if bias is not None else torch.zeros(1, dtype=torch.float32, device=x.device)
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)

    try:
        lib.fused_e8_2bit_gemm_fwd(
            x_2d,
            w_2d,
            s_2d,
            b_t,
            y,
            M,
            K,
            N,
            int(group_size),
            has_bias,
            threads=(N, M, 1),
        )
        return y.to(orig_dtype).reshape(*orig_shape[:-1], N)
    except Exception:
        return None


def metal_fused_ternary_e8_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    lut_coords: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> Optional[torch.Tensor]:
    """Run fused 1.5-bit Ternary E8 dequant-GEMM directly on Apple Silicon MPS GPU."""
    if x.device.type != "mps":
        return None
    if in_features % 16 != 0 or group_size % 16 != 0 or in_features % group_size != 0:
        return None
    lib = _get_metal_e8_lib()
    if lib is None or not hasattr(lib, "fused_ternary_e8_1_5bit_gemm_fwd"):
        return None

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d = x.reshape(-1, in_features).to(torch.float32).contiguous()
    M = int(x_2d.shape[0])
    K = int(in_features)
    N = int(out_features)
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((*orig_shape[:-1], N), dtype=orig_dtype, device=x.device)

    w_2d = packed_weight.reshape(N, (K * 3) // 16).contiguous()
    s_2d = block_scales.reshape(N, K // group_size).to(torch.float16).contiguous()
    lut_t = lut_coords.to(device=x.device, dtype=torch.int8).contiguous()
    has_bias = 1 if bias is not None else 0
    b_t = bias.to(device=x.device, dtype=torch.float32).contiguous() if bias is not None else torch.zeros(1, dtype=torch.float32, device=x.device)
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)

    try:
        lib.fused_ternary_e8_1_5bit_gemm_fwd(
            x_2d,
            w_2d,
            s_2d,
            lut_t,
            b_t,
            y,
            M,
            K,
            N,
            int(group_size),
            has_bias,
            threads=(N, M, 1),
        )
        return y.to(orig_dtype).reshape(*orig_shape[:-1], N)
    except Exception:
        return None


__all__ = ["metal_fused_e8_2bit_gemm", "metal_fused_ternary_e8_gemm"]
