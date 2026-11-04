"""DeepSeek TileLang JIT compilation and hardware kernel execution bridge."""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import torch

from ojasx2.tilelang_kernels import (
    compile_or_emit_universal_kernel,
    UniversalCompiledKernel,
    _ensure_vendored_tilelang_on_path,
    dispatch_tilelang_e8_gemm,
    is_tilelang_available,
)

__all__ = [
    "compile_or_emit_universal_kernel",
    "UniversalCompiledKernel",
    "_ensure_vendored_tilelang_on_path",
    "dispatch_tilelang_e8_gemm",
    "is_tilelang_available",
]
