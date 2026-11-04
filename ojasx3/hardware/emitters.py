"""Bounds-checked hardware kernel source emitters (Metal, CUDA, ROCm, SYCL, Ascend, CPU)."""

from __future__ import annotations
from typing import Tuple, Optional
import torch

from ojasx2.tile_fallback import emit_gemm_kernel_source, normalize_target_arch

__all__ = ["emit_gemm_kernel_source", "normalize_target_arch"]
