# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# This file is part of OjasX.
#
# OjasX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OjasX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OjasX.  If not, see <https://www.gnu.org/licenses/>.

"""
ojasx2/runtime/kv_cache_e8.py
==============================
8x-Compressed E8 Lattice Key-Value Cache for Autoregressive Generation.

Features:
- HuggingFace-compatible cache interface: `update(key_states, value_states, layer_idx)`.
- Packs intermediate key and value vectors into dense E8 crystal lattice codes.
- At 1.50 b/w (Ternary E8) or 1.00 b/w (Root E8), achieves 8x to 16x VRAM compression
  over standard FP16 KV caches, allowing 128k context windows to fit in <4 GB VRAM.
- Fully isolated: imports only from `ojasx2.math` and standard PyTorch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from ojasx2.math.lattice import _get_packer_for_bitrate


class E8KVCache:
    """
    HuggingFace-compatible 8x-compressed KV-Cache storing historical token keys and
    values as packed E8 crystal lattice codes.
    """

    def __init__(
        self,
        bitrate: float = 1.5,
        group_size: int = 32,
    ) -> None:
        self.bitrate = float(bitrate)
        self.group_size = int(group_size)

        # Per-layer storage for keys: (packed_codes, scales, shape [B, H, S, D])
        self.key_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]]] = []
        # Per-layer storage for values: (packed_codes, scales, shape [B, H, S, D])
        self.value_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]]] = []

        self._packer = _get_packer_for_bitrate(self.bitrate)

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx][2][2]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_decompressed_keys(self, layer_idx: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            raise ValueError(f"No key cache found for layer {layer_idx}")
        packed, scales, shape = self.key_cache[layer_idx]
        b, h, s, d = shape
        flat_2d_shape = (b * h * s, d)
        unpacked_flat = self._packer.unpack(packed, scales, flat_2d_shape, dtype=dtype)
        return unpacked_flat.view(b, h, s, d)

    def get_decompressed_values(self, layer_idx: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if layer_idx >= len(self.value_cache) or self.value_cache[layer_idx] is None:
            raise ValueError(f"No value cache found for layer {layer_idx}")
        packed, scales, shape = self.value_cache[layer_idx]
        b, h, s, d = shape
        flat_2d_shape = (b * h * s, d)
        unpacked_flat = self._packer.unpack(packed, scales, flat_2d_shape, dtype=dtype)
        return unpacked_flat.view(b, h, s, d)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache for layer_idx with incoming key_states and value_states [B, H, S, D].
        Returns decompressed full sequences (full_keys, full_values) ready for attention.
        """
        self._ensure_layer(layer_idx)
        B, H, S, D = key_states.shape
        dtype = key_states.dtype
        device = key_states.device

        if self.key_cache[layer_idx] is None:
            # First prompt prefill pass
            flat_k = key_states.contiguous().view(-1, D)
            flat_v = value_states.contiguous().view(-1, D)

            packed_k, scales_k, _ = self._packer.pack(flat_k, group_size=min(self.group_size, D))
            packed_v, scales_v, _ = self._packer.pack(flat_v, group_size=min(self.group_size, D))

            self.key_cache[layer_idx] = (packed_k, scales_k, (B, H, S, D))
            self.value_cache[layer_idx] = (packed_v, scales_v, (B, H, S, D))

            return key_states, value_states

        else:
            # Subsequent autoregressive token generation pass
            past_k = self.get_decompressed_keys(layer_idx, dtype=dtype)
            past_v = self.get_decompressed_values(layer_idx, dtype=dtype)

            # Concatenate along sequence dimension S
            full_k = torch.cat([past_k, key_states], dim=2)
            full_v = torch.cat([past_v, value_states], dim=2)

            total_s = full_k.shape[2]
            flat_k = full_k.contiguous().view(-1, D)
            flat_v = full_v.contiguous().view(-1, D)

            packed_k, scales_k, _ = self._packer.pack(flat_k, group_size=min(self.group_size, D))
            packed_v, scales_v, _ = self._packer.pack(flat_v, group_size=min(self.group_size, D))

            self.key_cache[layer_idx] = (packed_k, scales_k, (B, H, total_s, D))
            self.value_cache[layer_idx] = (packed_v, scales_v, (B, H, total_s, D))

            return full_k, full_v

    def reset(self) -> None:
        """Clears all cached tokens across all layers."""
        self.key_cache.clear()
        self.value_cache.clear()

    def to(self, device: Union[str, torch.device]) -> E8KVCache:
        """Moves all cached tensors to target device."""
        target_dev = torch.device(device)
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                pk, sk, shk = self.key_cache[i]
                self.key_cache[i] = (pk.to(target_dev), sk.to(target_dev), shk)
            if self.value_cache[i] is not None:
                pv, sv, shv = self.value_cache[i]
                self.value_cache[i] = (pv.to(target_dev), sv.to(target_dev), shv)
        return self

    def memory_bytes(self) -> int:
        """Calculates current total allocated memory in bytes for packed codes and scales."""
        total = 0
        for entry in self.key_cache + self.value_cache:
            if entry is not None:
                packed, scales, _ = entry
                total += packed.numel() * packed.element_size()
                total += scales.numel() * scales.element_size()
        return total

    def fp16_baseline_bytes(self) -> int:
        """Calculates memory that uncompressed FP16 key/value states would occupy."""
        total_elements = 0
        for entry in self.key_cache + self.value_cache:
            if entry is not None:
                _, _, shape = entry
                total_elements += shape[0] * shape[1] * shape[2] * shape[3]
        return total_elements * 2  # 2 bytes per float16

    def compression_ratio(self) -> float:
        """Calculates KV-cache compression factor vs FP16 baseline."""
        mem = self.memory_bytes()
        if mem == 0:
            return 1.0
        return self.fp16_baseline_bytes() / mem
