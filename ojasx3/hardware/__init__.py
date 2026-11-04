"""OjasX3 Universal Multi-Hardware Acceleration & DeepSeek TileLang Subsystem."""

from .autotuner import TileLangHardwareAutoTuner, TileConfig
from .tilelang_jit import (
    compile_or_emit_universal_kernel,
    UniversalCompiledKernel,
    is_tilelang_available,
    dispatch_tilelang_e8_gemm,
)
from .emitters import emit_gemm_kernel_source, normalize_target_arch

__all__ = [
    "TileLangHardwareAutoTuner",
    "TileConfig",
    "compile_or_emit_universal_kernel",
    "UniversalCompiledKernel",
    "is_tilelang_available",
    "dispatch_tilelang_e8_gemm",
    "emit_gemm_kernel_source",
    "normalize_target_arch",
]
