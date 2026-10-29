"""
ojasx2.math.lattice
===================

Canonical 8D E8 Lattice Sphere Packing and Vector Quantization Engine.

Upgraded with:
 - Thread- and CUDA-stream-safe ephemeral scratchpad (`_get_ephemeral_workspace`).
 - Automatic tail-padding and exact unpadded slicing in `E8LatticePacker` (2.0-bit),
   `TernaryE8LatticePacker` (1.5-bit), and `RootE8LatticePacker` (1.0-bit) so arbitrary
   matrix shapes (M, N) are supported seamlessly.
 - Mixed-precision LoRA dtype alignment in `E8LatticeLinear`.
"""

from __future__ import annotations

import copy
import itertools
import math
import threading
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .csrc import ojasx_math_native as _native
    _HAS_NATIVE = hasattr(_native, "fused_e8_gemm_fwd")
except Exception:
    _native = None
    _HAS_NATIVE = False

try:
    from ojasx2.tilelang_kernels import dispatch_tilelang_e8_gemm
except Exception:
    def dispatch_tilelang_e8_gemm(*args, **kwargs):
        return None


# ==============================================================================
# 0. Thread- & Stream-Safe Reusable Ephemeral Layer Scratchpad Manager
# ==============================================================================

_EPHEMERAL_WORKSPACES: Dict[Tuple[str, Optional[int], torch.dtype, int, int], torch.Tensor] = {}
_WORKSPACE_LOCK = threading.Lock()


def _get_ephemeral_workspace(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Returns a thread-local and CUDA-stream-local scratchpad buffer on `device`."""
    tid = threading.get_ident()
    stream_id = 0
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            stream_id = int(torch.cuda.current_stream(device).stream_id)
        except Exception:
            stream_id = 0

    key = (device.type, device.index, dtype, tid, stream_id)
    buf = _EPHEMERAL_WORKSPACES.get(key)
    if buf is None or buf.numel() < numel:
        with _WORKSPACE_LOCK:
            buf = _EPHEMERAL_WORKSPACES.get(key)
            if buf is None or buf.numel() < numel:
                buf = torch.empty(numel, dtype=dtype, device=device)
                _EPHEMERAL_WORKSPACES[key] = buf
    return buf[:numel]


# ==============================================================================
# 1. Randomized Incoherence Processing (Fast Walsh-Hadamard Transform)
# ==============================================================================

def fwht_block32(x: torch.Tensor) -> torch.Tensor:
    """Fast in-register 32-point Walsh-Hadamard Transform along the last dimension."""
    orig_shape = x.shape
    if orig_shape[-1] == 0 or orig_shape[-1] % 32 != 0:
        return x
    h = x.reshape(-1, 16, 2)
    h = torch.stack([h[..., 0] + h[..., 1], h[..., 0] - h[..., 1]], dim=-1).reshape(-1, 8, 2, 2)
    h = torch.stack([h[:, :, 0] + h[:, :, 1], h[:, :, 0] - h[:, :, 1]], dim=-2).reshape(-1, 4, 2, 4)
    h = torch.stack([h[:, :, 0] + h[:, :, 1], h[:, :, 0] - h[:, :, 1]], dim=-2).reshape(-1, 2, 2, 8)
    h = torch.stack([h[:, :, 0] + h[:, :, 1], h[:, :, 0] - h[:, :, 1]], dim=-2).reshape(-1, 2, 16)
    h = torch.stack([h[:, 0] + h[:, 1], h[:, 0] - h[:, 1]], dim=-2)
    return (h * 0.1767766952966369).reshape(orig_shape)


def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Computes the normalized Fast Walsh-Hadamard Transform (FWHT) along the last dimension."""
    orig_shape = x.shape
    n = orig_shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Last dimension ({n}) must be a positive power of 2 for FWHT.")
    if n == 32:
        return fwht_block32(x)

    out = x.clone()
    h = 1
    while h < n:
        out = out.view(-1, n // (2 * h), 2, h)
        a = out[:, :, 0, :].clone()
        b = out[:, :, 1, :].clone()
        out[:, :, 0, :] = a + b
        out[:, :, 1, :] = a - b
        h *= 2

    return out.view(orig_shape) / math.sqrt(n)


def randomized_hadamard_transform(
    w: torch.Tensor,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies Randomized Hadamard Transform (RHT) to weight matrix W."""
    m, n = w.shape
    device = w.device
    dtype = w.dtype

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    s1 = torch.randint(0, 2, (m,), generator=rng, device=device, dtype=dtype) * 2 - 1
    s2 = torch.randint(0, 2, (n,), generator=rng, device=device, dtype=dtype) * 2 - 1
    w_signed = (w * s2.unsqueeze(0)) * s1.unsqueeze(1)

    m_pow2 = 1 << (m - 1).bit_length() if (m & (m - 1)) != 0 else m
    n_pow2 = 1 << (n - 1).bit_length() if (n & (n - 1)) != 0 else n

    if m == m_pow2 and n == n_pow2:
        w_rot = fast_walsh_hadamard_transform(w_signed)
        w_rot = fast_walsh_hadamard_transform(w_rot.transpose(0, 1)).transpose(0, 1)
    elif n % 32 == 0:
        w_rot = fwht_block32(w_signed)
    else:
        w_rot = w_signed

    return w_rot, s1, s2


# ==============================================================================
# 2. Conway-Sloane O(8) Fast E8 Lattice Projection
# ==============================================================================

def conway_sloane_e8_project(x: torch.Tensor, prefer_native: bool = True) -> torch.Tensor:
    """Snaps an arbitrary continuous 8D vector or batch of 8D vectors to the nearest
    point in the 8-dimensional Gosset crystal lattice E8 = D8 U (D8 + 1/2 * 1)."""
    orig_shape = x.shape
    if orig_shape[-1] != 8:
        raise ValueError(f"Last dimension must be 8 for E8 lattice (got {orig_shape[-1]}).")
    if x.numel() == 0:
        return x.clone()

    x_flat = x.contiguous().view(-1, 8)

    if prefer_native and x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64):
        try:
            if _native is not None and hasattr(_native, "conway_sloane_e8_project_tensor"):
                return _native.conway_sloane_e8_project_tensor(x_flat).view(orig_shape)
        except Exception:
            pass

    f1 = torch.round(x_flat)
    sum1 = torch.sum(f1, dim=-1, keepdim=True).long()
    err1 = torch.abs(x_flat - f1)
    worst1 = torch.argmax(err1, dim=-1, keepdim=True)
    sign1 = torch.sign(x_flat - f1)
    sign1 = torch.where(sign1 == 0, torch.ones_like(sign1), sign1)
    f1_fixed = f1.scatter(-1, worst1, f1.gather(-1, worst1) + sign1.gather(-1, worst1))
    c1 = torch.where((sum1 % 2 != 0), f1_fixed, f1)

    x_shift = x_flat - 0.5
    f2 = torch.round(x_shift)
    sum2 = torch.sum(f2, dim=-1, keepdim=True).long()
    err2 = torch.abs(x_shift - f2)
    worst2 = torch.argmax(err2, dim=-1, keepdim=True)
    sign2 = torch.sign(x_shift - f2)
    sign2 = torch.where(sign2 == 0, torch.ones_like(sign2), sign2)
    f2_fixed = f2.scatter(-1, worst2, f2.gather(-1, worst2) + sign2.gather(-1, worst2))
    c2 = torch.where((sum2 % 2 != 0), f2_fixed, f2) + 0.5

    d1 = torch.sum((x_flat - c1) ** 2, dim=-1, keepdim=True)
    d2 = torch.sum((x_flat - c2) ** 2, dim=-1, keepdim=True)
    chosen = torch.where(d1 <= d2, c1, c2)

    return chosen.view(orig_shape)


# ==============================================================================
# 3a. 2.0-Bit Packing & Unpacking Engine
# ==============================================================================

class E8LatticePacker:
    """Packs weight matrices into 2.0 bits/weight (4 weights per uint8 byte) with auto-padding."""

    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        orig_shape = w.shape
        m, n = orig_shape
        total_elements = m * n
        if group_size <= 0 or group_size % 8 != 0:
            raise ValueError(f"group_size ({group_size}) must be a positive multiple of 8.")

        flat = w.reshape(-1).to(torch.float32)
        pad_len = (-total_elements) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        w_groups = flat.view(-1, group_size)
        scales = torch.norm(w_groups, dim=-1, keepdim=True) / math.sqrt(group_size) + 1e-8
        w_norm = (w_groups / scales).view(-1, 8)

        x_shift = w_norm - 0.5
        f2 = torch.clamp(torch.round(x_shift), -2.0, 1.0)
        sum2 = torch.sum(f2, dim=-1, keepdim=True).long()
        alt_f2 = torch.where(x_shift >= f2, torch.clamp(f2 + 1.0, -2.0, 1.0), torch.clamp(f2 - 1.0, -2.0, 1.0))
        penalty = (alt_f2 - x_shift) ** 2 - (f2 - x_shift) ** 2
        worst2 = torch.argmin(penalty, dim=-1, keepdim=True)
        f2_even = torch.where(sum2 % 2 != 0, f2.scatter(-1, worst2, alt_f2.gather(-1, worst2)), f2)
        w_e8 = f2_even + 0.5

        codes = torch.clamp(torch.round(w_e8 + 1.5), 0, 3).to(torch.uint8)
        codes_flat = codes.view(-1, 4)
        packed = (
            codes_flat[:, 0]
            | (codes_flat[:, 1] << 2)
            | (codes_flat[:, 2] << 4)
            | (codes_flat[:, 3] << 6)
        )
        return packed, scales.squeeze(-1).to(torch.float16), (m, n)

    @staticmethod
    def unpack(
        packed_codes: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, int],
        dtype: torch.dtype = torch.float32,
        use_workspace: bool = False,
    ) -> torch.Tensor:
        c0 = packed_codes & 0x03
        c1 = (packed_codes >> 2) & 0x03
        c2 = (packed_codes >> 4) & 0x03
        c3 = (packed_codes >> 6) & 0x03

        total_elements = shape[0] * shape[1]
        padded_elements = packed_codes.numel() * 4
        group_size = padded_elements // max(1, scales.numel())
        codes_unpacked = torch.stack([c0, c1, c2, c3], dim=-1).view(-1, group_size)

        if use_workspace and padded_elements == total_elements:
            out = _get_ephemeral_workspace(total_elements, dtype, packed_codes.device).view(-1, group_size)
            out.copy_(codes_unpacked)
            out.sub_(1.5)
            out.mul_(scales.to(dtype).view(-1, 1))
            return out.view(shape)

        w_e8 = codes_unpacked.to(dtype) - 1.5
        full = (w_e8 * scales.to(dtype).view(-1, 1)).reshape(-1)
        return full[:total_elements].view(shape)


# ==============================================================================
# 3b. Ternary E8 (1.50-Bit) Packing & Unpacking Engine
# ==============================================================================

_DEVICE_TERNARY_LUTS: Dict[Tuple[str, Optional[int]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_TERNARY_RANK_TO_COORDS: Optional[torch.Tensor] = None
_TERNARY_RAW_TO_RANK: Optional[torch.Tensor] = None
_POWERS_OF_3: Optional[torch.Tensor] = None


def _get_ternary_e8_luts(device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _DEVICE_TERNARY_LUTS, _TERNARY_RANK_TO_COORDS, _TERNARY_RAW_TO_RANK, _POWERS_OF_3

    if device is None:
        dev_key = ("cpu", None)
    else:
        if isinstance(device, str):
            device = torch.device(device)
        dev_key = (device.type, device.index)

    if dev_key in _DEVICE_TERNARY_LUTS:
        return _DEVICE_TERNARY_LUTS[dev_key]

    if _TERNARY_RANK_TO_COORDS is None:
        coords_list = []
        raw_to_rank = torch.zeros(6561, dtype=torch.int64)
        rank = 0
        for i in range(3**8):
            temp = i
            digits = []
            for _ in range(8):
                digits.append((temp % 3) - 1)
                temp //= 3
            if sum(digits) % 2 == 0:
                raw_to_rank[i] = rank
                coords_list.append(digits)
                rank += 1
        _TERNARY_RANK_TO_COORDS = torch.tensor(coords_list, dtype=torch.int8)
        _TERNARY_RAW_TO_RANK = raw_to_rank
        _POWERS_OF_3 = torch.tensor([3**i for i in range(8)], dtype=torch.int64)

    if dev_key[0] == "cpu":
        _DEVICE_TERNARY_LUTS[dev_key] = (_TERNARY_RANK_TO_COORDS, _TERNARY_RAW_TO_RANK, _POWERS_OF_3)
        return _DEVICE_TERNARY_LUTS[dev_key]

    target_dev = torch.device(dev_key[0], dev_key[1]) if dev_key[1] is not None else torch.device(dev_key[0])
    dev_r2c = _TERNARY_RANK_TO_COORDS.to(target_dev)
    dev_r2r = _TERNARY_RAW_TO_RANK.to(target_dev)
    dev_pow = _POWERS_OF_3.to(target_dev)
    _DEVICE_TERNARY_LUTS[dev_key] = (dev_r2c, dev_r2r, dev_pow)
    return _DEVICE_TERNARY_LUTS[dev_key]


class TernaryE8LatticePacker:
    """Packs weight matrices into 1.50 bits/weight using D8 cap {-1, 0, +1}^8 (3,281 points)."""

    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        orig_shape = w.shape
        m, n = orig_shape
        total_elements = m * n
        if group_size <= 0 or group_size % 16 != 0:
            group_size = 16

        device = w.device
        flat = w.reshape(-1).to(torch.float32)
        pad_len = (-total_elements) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        w_groups = flat.view(-1, group_size)
        scales = torch.norm(w_groups, dim=-1, keepdim=True) / math.sqrt(group_size) + 1e-8
        v = (w_groups / scales).view(-1, 8)

        q = torch.clamp(torch.round(v), -1.0, 1.0)
        parity = torch.sum(q, dim=-1, keepdim=True).long() % 2
        alt_q = torch.where(
            q == 0.0,
            torch.where(v >= 0.0, torch.ones_like(q), -torch.ones_like(q)),
            torch.zeros_like(q),
        )
        penalty = (alt_q - v) ** 2 - (q - v) ** 2
        worst = torch.argmin(penalty, dim=-1, keepdim=True)
        q_even = torch.where(parity != 0, q.scatter(-1, worst, alt_q.gather(-1, worst)), q)

        _, raw_to_rank, powers = _get_ternary_e8_luts(device)
        digits = q_even.long() + 1
        raw_codes = (digits * powers).sum(dim=-1)
        codes = raw_to_rank[raw_codes]

        codes_pairs = codes.view(-1, 2)
        c0 = codes_pairs[:, 0]
        c1 = codes_pairs[:, 1]

        b0 = (c0 & 0xFF).to(torch.uint8)
        b1 = (((c0 >> 8) & 0x0F) | ((c1 & 0x0F) << 4)).to(torch.uint8)
        b2 = ((c1 >> 4) & 0xFF).to(torch.uint8)

        packed = torch.stack([b0, b1, b2], dim=-1).flatten()
        return packed, scales.squeeze(-1).to(torch.float16), (m, n)

    @staticmethod
    def unpack(
        packed_codes: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, int],
        dtype: torch.dtype = torch.float32,
        use_workspace: bool = False,
    ) -> torch.Tensor:
        device = packed_codes.device
        rank_to_coords, _, _ = _get_ternary_e8_luts(device)

        triplets = packed_codes.view(-1, 3)
        b0 = triplets[:, 0].to(torch.int32)
        b1 = triplets[:, 1].to(torch.int32)
        b2 = triplets[:, 2].to(torch.int32)

        num_pairs = triplets.shape[0]
        codes = torch.empty((num_pairs, 2), dtype=torch.int32, device=device)
        codes[:, 0] = b0 | ((b1 & 0x0F) << 8)
        codes[:, 1] = ((b1 >> 4) & 0x0F) | (b2 << 4)

        total_elements = shape[0] * shape[1]
        padded_elements = num_pairs * 16
        group_size = padded_elements // max(1, scales.numel())
        coords = rank_to_coords[codes.view(-1)].view(-1, group_size)

        if use_workspace and padded_elements == total_elements:
            out = _get_ephemeral_workspace(total_elements, dtype, device).view(-1, group_size)
            out.copy_(coords)
            out.mul_(scales.to(dtype).view(-1, 1))
            return out.view(shape)

        full = (coords.to(dtype) * scales.to(dtype).view(-1, 1)).reshape(-1)
        return full[:total_elements].view(shape)


# ==============================================================================
# 3c. Root E8 (1.00-Bit) Packing & Unpacking Engine
# ==============================================================================

_DEVICE_ROOT_E8_LUTS: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_ROOT_E8_TABLE_CPU: Optional[torch.Tensor] = None


def _get_root_e8_lut(device: Optional[torch.device] = None) -> torch.Tensor:
    global _DEVICE_ROOT_E8_LUTS, _ROOT_E8_TABLE_CPU
    if device is None:
        dev_key = ("cpu", None)
    else:
        if isinstance(device, str):
            device = torch.device(device)
        dev_key = (device.type, device.index)

    if dev_key in _DEVICE_ROOT_E8_LUTS:
        return _DEVICE_ROOT_E8_LUTS[dev_key]

    if _ROOT_E8_TABLE_CPU is None:
        pts = []
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in (-1.0, 1.0):
                    for s2 in (-1.0, 1.0):
                        vec = [0.0] * 8
                        vec[i] = s1
                        vec[j] = s2
                        pts.append(vec)
        for signs in itertools.product((-0.5, 0.5), repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                pts.append(list(signs))
        rt2 = math.sqrt(2.0)
        for i in range(8):
            for s in (-rt2, rt2):
                vec = [0.0] * 8
                vec[i] = s
                pts.append(vec)
        _ROOT_E8_TABLE_CPU = torch.tensor(pts, dtype=torch.float32) * 2.0

    if dev_key[0] == "cpu":
        _DEVICE_ROOT_E8_LUTS[dev_key] = _ROOT_E8_TABLE_CPU
        return _ROOT_E8_TABLE_CPU

    target_dev = torch.device(dev_key[0], dev_key[1]) if dev_key[1] is not None else torch.device(dev_key[0])
    dev_lut = _ROOT_E8_TABLE_CPU.to(target_dev)
    _DEVICE_ROOT_E8_LUTS[dev_key] = dev_lut
    return dev_lut


class RootE8LatticePacker:
    """Packs weight matrices into 1.00 bit/weight (1 byte per 8D vector) using the 240 E8 roots."""

    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        orig_shape = w.shape
        m, n = orig_shape
        total_elements = m * n
        if group_size <= 0 or group_size % 8 != 0:
            group_size = 8

        device = w.device
        lut = _get_root_e8_lut(device)
        flat = w.reshape(-1).to(torch.float32)
        pad_len = (-total_elements) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        w_groups = flat.view(-1, group_size)
        scales = torch.norm(w_groups, dim=-1, keepdim=True) / math.sqrt(group_size) + 1e-8
        v = (w_groups / scales).view(-1, 8)

        dots = torch.matmul(v, lut.t())
        packed = torch.argmax(dots, dim=-1).to(torch.uint8)
        return packed, scales.squeeze(-1).to(torch.float16), (m, n)

    @staticmethod
    def unpack(
        packed_codes: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, int],
        dtype: torch.dtype = torch.float32,
        use_workspace: bool = False,
    ) -> torch.Tensor:
        device = packed_codes.device
        lut = _get_root_e8_lut(device).to(dtype)
        total_elements = shape[0] * shape[1]
        padded_elements = packed_codes.numel() * 8
        group_size = padded_elements // max(1, scales.numel())
        coords = lut[packed_codes.to(torch.int32)].view(-1, group_size)

        if use_workspace and padded_elements == total_elements:
            out = _get_ephemeral_workspace(total_elements, dtype, device).view(-1, group_size)
            out.copy_(coords)
            out.mul_(scales.to(dtype).view(-1, 1))
            return out.view(shape)

        full = (coords * scales.to(dtype).view(-1, 1)).reshape(-1)
        return full[:total_elements].view(shape)


def _get_packer_for_bitrate(bitrate: float):
    if bitrate == 1.0:
        return RootE8LatticePacker
    if bitrate == 1.5:
        return TernaryE8LatticePacker
    return E8LatticePacker


# ==============================================================================
# 4. Fused Register-Dequant GEMM Autograd Function
# ==============================================================================

class _FusedE8GEMMFunction(torch.autograd.Function):
    MAX_EPHEMERAL_ELEMENTS: int = 67_108_864

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        block_scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_features: int,
        in_features: int,
        group_size: int,
        bitrate: float,
        use_hadamard: bool = False,
    ) -> torch.Tensor:
        if use_hadamard and in_features % 32 == 0:
            x_rot = fwht_block32(x)
        else:
            x_rot = x

        ctx.save_for_backward(x_rot, packed_weight, block_scales)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.group_size = group_size
        ctx.bitrate = bitrate
        ctx.has_bias = bias is not None
        ctx.use_hadamard = use_hadamard

        x_orig_shape = x_rot.shape
        x_2d = x_rot.reshape(-1, in_features).contiguous()

        lut_coords = _get_ternary_e8_luts(x.device)[0] if bitrate == 1.5 else None
        y_tl = dispatch_tilelang_e8_gemm(
            x_2d, packed_weight, block_scales, bias,
            out_features, in_features, group_size, bitrate, lut_coords
        )
        if y_tl is not None:
            return y_tl.reshape(*x_orig_shape[:-1], out_features)

        if (
            _HAS_NATIVE
            and x.is_cpu
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and x_2d.shape[0] <= 64
            and (out_features * in_features) % group_size == 0
        ):
            if bitrate == 2.0 and in_features % 4 == 0:
                packed_2d = packed_weight.reshape(out_features, in_features // 4).contiguous()
                scales_2d = block_scales.reshape(out_features, -1).contiguous()
                y_2d = _native.fused_e8_gemm_fwd(x_2d, packed_2d, scales_2d, bias, group_size)
                return y_2d.reshape(*x_orig_shape[:-1], out_features)
            elif bitrate == 1.5 and hasattr(_native, "fused_ternary_e8_gemm_fwd") and (group_size % 16 == 0) and (in_features % 16 == 0):
                packed_2d = packed_weight.reshape(out_features, (in_features * 3) // 16).contiguous()
                scales_2d = block_scales.reshape(out_features, -1).contiguous()
                y_2d = _native.fused_ternary_e8_gemm_fwd(x_2d, packed_2d, scales_2d, bias, group_size)
                return y_2d.reshape(*x_orig_shape[:-1], out_features)

        packer = _get_packer_for_bitrate(bitrate)
        total_elements = out_features * in_features

        if total_elements <= _FusedE8GEMMFunction.MAX_EPHEMERAL_ELEMENTS:
            w_layer = packer.unpack(
                packed_weight, block_scales, (out_features, in_features),
                dtype=x_2d.dtype, use_workspace=True,
            )
            b_cast = bias.to(x_2d.dtype) if bias is not None else None
            y_2d = F.linear(x_2d, w_layer, b_cast)
            return y_2d.reshape(*x_orig_shape[:-1], out_features)

        chunk_size = 1024
        y_chunks = []
        bytes_per_row = int(in_features * bitrate) // 8
        scales_per_row = in_features // group_size

        for n_start in range(0, out_features, chunk_size):
            n_end = min(n_start + chunk_size, out_features)
            n_chunk = n_end - n_start
            p_chunk = packed_weight[n_start * bytes_per_row : n_end * bytes_per_row]
            s_chunk = block_scales[n_start * scales_per_row : n_end * scales_per_row]
            w_tile = packer.unpack(p_chunk, s_chunk, (n_chunk, in_features), dtype=x_2d.dtype, use_workspace=True)
            b_tile = bias[n_start:n_end].to(x_2d.dtype) if bias is not None else None
            y_chunks.append(F.linear(x_2d, w_tile, b_tile))

        return torch.cat(y_chunks, dim=-1).reshape(*x_orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_rot, packed_weight, block_scales = ctx.saved_tensors
        in_features = ctx.in_features
        out_features = ctx.out_features
        group_size = ctx.group_size
        bitrate = ctx.bitrate
        use_hadamard = ctx.use_hadamard

        gy_2d = grad_y.reshape(-1, out_features).contiguous()

        if (
            _HAS_NATIVE
            and grad_y.is_cpu
            and grad_y.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and gy_2d.shape[0] <= 64
            and (out_features * in_features) % group_size == 0
        ):
            if bitrate == 2.0 and in_features % 4 == 0:
                packed_2d = packed_weight.reshape(out_features, in_features // 4).contiguous()
                scales_2d = block_scales.reshape(out_features, -1).contiguous()
                grad_x_2d = _native.fused_e8_gemm_bwd_dx(gy_2d, packed_2d, scales_2d, in_features, group_size)
                grad_x = grad_x_2d.reshape_as(x_rot)
                if use_hadamard and in_features % 32 == 0:
                    grad_x = fwht_block32(grad_x)
                grad_bias = gy_2d.sum(dim=0) if ctx.has_bias else None
                return grad_x, None, None, grad_bias, None, None, None, None, None
            elif bitrate == 1.5 and hasattr(_native, "fused_ternary_e8_gemm_bwd_dx") and (group_size % 16 == 0) and (in_features % 16 == 0):
                packed_2d = packed_weight.reshape(out_features, (in_features * 3) // 16).contiguous()
                scales_2d = block_scales.reshape(out_features, -1).contiguous()
                grad_x_2d = _native.fused_ternary_e8_gemm_bwd_dx(gy_2d, packed_2d, scales_2d, in_features, group_size)
                grad_x = grad_x_2d.reshape_as(x_rot)
                if use_hadamard and in_features % 32 == 0:
                    grad_x = fwht_block32(grad_x)
                grad_bias = gy_2d.sum(dim=0) if ctx.has_bias else None
                return grad_x, None, None, grad_bias, None, None, None, None, None

        packer = _get_packer_for_bitrate(bitrate)
        total_elements = out_features * in_features

        if total_elements <= _FusedE8GEMMFunction.MAX_EPHEMERAL_ELEMENTS:
            w_layer = packer.unpack(
                packed_weight, block_scales, (out_features, in_features),
                dtype=x_rot.dtype, use_workspace=True,
            )
            grad_x_2d = gy_2d.matmul(w_layer)
            grad_x = grad_x_2d.reshape_as(x_rot)
        else:
            chunk_size = 1024
            bytes_per_row = int(in_features * bitrate) // 8
            scales_per_row = in_features // group_size
            grad_x_2d = torch.zeros(gy_2d.shape[0], in_features, dtype=x_rot.dtype, device=x_rot.device)

            for n_start in range(0, out_features, chunk_size):
                n_end = min(n_start + chunk_size, out_features)
                n_chunk = n_end - n_start
                p_chunk = packed_weight[n_start * bytes_per_row : n_end * bytes_per_row]
                s_chunk = block_scales[n_start * scales_per_row : n_end * scales_per_row]
                w_tile = packer.unpack(p_chunk, s_chunk, (n_chunk, in_features), dtype=x_rot.dtype, use_workspace=True)
                grad_x_2d.add_(gy_2d[:, n_start:n_end].matmul(w_tile))

            grad_x = grad_x_2d.reshape_as(x_rot)

        if use_hadamard and in_features % 32 == 0:
            grad_x = fwht_block32(grad_x)

        grad_bias = gy_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, None, None, grad_bias, None, None, None, None, None


# ==============================================================================
# 5. E8LatticeLinear: Low-Memory 1.0/1.5/2.0-Bit Linear Layer with LoRA Subspace
# ==============================================================================

class E8LatticeLinear(nn.Module):
    """Linear layer with base weights packed into the 8D E8 lattice (1.0, 1.5, or 2.0 b/w)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        lora_alpha: float = 16.0,
        group_size: int = 8,
        bitrate: float = 2.0,
        use_hadamard: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / rank) if rank > 0 else 0.0
        self.bitrate = float(bitrate)
        self.use_hadamard = bool(use_hadamard and (in_features % 32 == 0))

        total_elements = out_features * in_features
        if self.bitrate == 1.5:
            eff_group = group_size if (group_size >= 16 and group_size % 16 == 0) else 16
            padded_elements = total_elements + ((-total_elements) % eff_group)
            num_bytes = (padded_elements * 3) // 16
        elif self.bitrate == 1.0:
            eff_group = group_size if (group_size >= 8 and group_size % 8 == 0) else 8
            padded_elements = total_elements + ((-total_elements) % eff_group)
            num_bytes = padded_elements // 8
        else:
            eff_group = group_size if (group_size >= 8 and group_size % 8 == 0) else 8
            padded_elements = total_elements + ((-total_elements) % eff_group)
            num_bytes = padded_elements // 4

        self.group_size = eff_group
        num_scales = padded_elements // eff_group

        self.register_buffer("packed_weight", torch.zeros(num_bytes, dtype=torch.uint8, device=device))
        self.register_buffer("block_scales", torch.ones(num_scales, dtype=torch.float16, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 0,
        lora_alpha: float = 16.0,
        group_size: int = 8,
        bitrate: float = 2.0,
        use_hadamard: bool = False,
    ) -> "E8LatticeLinear":
        eff_gs = max(16, group_size) if bitrate == 1.5 else group_size
        m = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            rank=rank,
            lora_alpha=lora_alpha,
            group_size=eff_gs,
            bitrate=bitrate,
            use_hadamard=use_hadamard,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w_to_pack = fwht_block32(linear.weight) if m.use_hadamard else linear.weight
            packer = _get_packer_for_bitrate(bitrate)
            packed, scales, _ = packer.pack(w_to_pack, group_size=m.group_size)
            m.packed_weight.copy_(packed)
            m.block_scales.copy_(scales)
            if linear.bias is not None:
                m.bias.copy_(linear.bias)
        return m

    def unpack_weights(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        packer = _get_packer_for_bitrate(self.bitrate)
        w = packer.unpack(
            self.packed_weight,
            self.block_scales,
            (self.out_features, self.in_features),
            dtype=dtype,
            use_workspace=False,
        )
        return fwht_block32(w) if self.use_hadamard else w

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        out = _FusedE8GEMMFunction.apply(
            x,
            self.packed_weight,
            self.block_scales,
            self.bias,
            self.out_features,
            self.in_features,
            self.group_size,
            self.bitrate,
            self.use_hadamard,
        )
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            lora_b = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + self.scaling * ((x @ lora_a) @ lora_b)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fused(x)


# ==============================================================================
# 6. E8QuantizationAwareLinear & Automatic Model Packing Utility
# ==============================================================================

class _E8STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        orig_shape = w.shape
        total = w.numel()
        pad_len = (-total) % 8
        flat = F.pad(w.reshape(-1), (0, pad_len)) if pad_len > 0 else w.reshape(-1)
        w_flat = flat.view(-1, 8)
        scales = torch.norm(w_flat, dim=-1, keepdim=True) / math.sqrt(8) + 1e-8
        w_e8 = conway_sloane_e8_project(w_flat / scales)
        return (w_e8 * scales).reshape(-1)[:total].view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class E8QuantizationAwareLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = _E8STEFunction.apply(self.weight)
        return F.linear(x, w_q, self.bias)


def pack_model_e8(
    model: nn.Module,
    rank: int = 0,
    lora_alpha: float = 16.0,
    group_size: int = 32,
    bitrate: float = 2.0,
    use_hadamard: bool = False,
    target_modules: Optional[Union[list[str], tuple[str, ...], str]] = None,
    exclude_modules: Optional[Union[list[str], tuple[str, ...]]] = None,
    inplace: bool = True,
) -> nn.Module:
    """Recursively packs all nn.Linear modules in a model into E8LatticeLinear layers."""
    target_model = model if inplace else copy.deepcopy(model)
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    for name, module in target_model.named_children():
        if exclude_modules is not None and any(ex in name for ex in exclude_modules):
            continue
        is_targeted = True if target_modules is None else any(t in name for t in target_modules)

        if isinstance(module, nn.Linear) and is_targeted:
            e8_layer = E8LatticeLinear.from_linear(
                module,
                rank=rank,
                lora_alpha=lora_alpha,
                group_size=group_size,
                bitrate=bitrate,
                use_hadamard=use_hadamard,
            )
            setattr(target_model, name, e8_layer)
        else:
            pack_model_e8(
                module,
                rank=rank,
                lora_alpha=lora_alpha,
                group_size=group_size,
                bitrate=bitrate,
                use_hadamard=use_hadamard,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
                inplace=True,
            )

    return target_model
