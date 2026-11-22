"""
Universal Architecture-Agnostic Activation Tape Compression via PyTorch saved_tensors_hooks.

Works on ANY PyTorch nn.Module (Transformers, Mamba/SSMs, MoE, ConvNeXt, ViT, DiT, GNNs,
or custom autograd functions) with zero hardcoded layer names and zero forward-recomputation
matmuls:
  - Leaves nn.Parameters, small tensors (< min_elements), and integer/bool tensors untouched.
  - Compresses large floating-point saved activations via Block-FWHT / CW(n,k) incoherence
    rotation + unbiased stochastic 2-bit quantization (E[Q(x)] == x).
  - Cuts activation tape RAM by 6.4x (vs FP16) to 12.8x (vs FP32) without re-running forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from ojasx2.math.lattice import stochastic_pack_2bit, stochastic_unpack_2bit


@dataclass
class TapeCompressionStats:
    tensors_compressed: int = 0
    tensors_skipped: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 1.0
        return self.original_bytes / float(self.compressed_bytes)


class E8ActivationTapeContext:
    """Context manager enabling architecture-agnostic Hadamard + Stochastic E8 tape compression.

    Usage:
        with E8ActivationTapeContext(group_size=32, incoherence=True) as tape:
            loss = model(x).sum()
            loss.backward()
        print(tape.stats.compression_ratio)
    """

    def __init__(
        self,
        *,
        group_size: int = 32,
        incoherence: bool = True,
        min_elements: int = 16384,
        enabled: bool = True,
    ) -> None:
        if group_size <= 0 or group_size % 8 != 0:
            raise ValueError(f"group_size must be a positive multiple of 8, got {group_size}")
        self.group_size = int(group_size)
        self.incoherence = bool(incoherence)
        self.min_elements = int(min_elements)
        self.enabled = bool(enabled)
        self.stats = TapeCompressionStats()
        self._hook_ctx: Optional[Any] = None

    def _pack_hook(self, tensor: torch.Tensor) -> Tuple[bool, Any]:
        # Never compress parameters, tiny tensors, or non-floating tensors (masks/indices)
        if (
            not self.enabled
            or isinstance(tensor, nn.Parameter)
            or not tensor.is_floating_point()
            or tensor.numel() < self.min_elements
        ):
            self.stats.tensors_skipped += 1
            return (False, tensor)

        packed, scales, orig_shape, orig_dtype, spec = stochastic_pack_2bit(
            tensor,
            group_size=self.group_size,
            incoherence=self.incoherence,
        )
        self.stats.tensors_compressed += 1
        self.stats.original_bytes += tensor.numel() * tensor.element_size()
        self.stats.compressed_bytes += (
            packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
        )
        return (True, (packed, scales, orig_shape, orig_dtype, spec))

    def _unpack_hook(self, handle: Tuple[bool, Any]) -> torch.Tensor:
        is_compressed, payload = handle
        if not is_compressed:
            return payload
        packed, scales, orig_shape, orig_dtype, spec = payload
        return stochastic_unpack_2bit(packed, scales, orig_shape, orig_dtype, spec=spec)

    def __enter__(self) -> "E8ActivationTapeContext":
        if self.enabled:
            self._hook_ctx = torch.autograd.graph.saved_tensors_hooks(
                self._pack_hook, self._unpack_hook
            )
            self._hook_ctx.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._hook_ctx is not None:
            self._hook_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._hook_ctx = None


__all__ = ["TapeCompressionStats", "E8ActivationTapeContext"]
