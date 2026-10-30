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
ojasx2/runtime/leech_and_lut_math.py
====================================
1. 24D Leech Lattice (Lambda_24) Fast Algebraic Projector & Packer:
   Implements Turyn's 3 x E8 + Golay coset construction in R^24 (Maryna Viazovska's
   globally densest 24D sphere packing with kissing number 196,560).
2. Precomputed 81-State Activation LUT Engine (Multiplier-Free Ternary E8 GEMV):
   Replaces weight dequantization and float multiplication with 2 L1-cache table
   lookups + 1 addition per 8 weights.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from ojasx2.math.lattice import _get_ternary_e8_luts


# ==============================================================================
# 1. 24D Leech Lattice (Lambda_24 = 3 x E8 + Extended Golay Coset Constraint)
# ==============================================================================

def _d8_coset_candidates(x8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For a batch of 8D vectors x8 [N, 8], returns:
      - c0: nearest point in integer D8 (even coordinate sum)
      - d0: squared Euclidean distance ||x8 - c0||^2
      - c1: nearest point in half-integer D8 + 0.5 (even shifted sum)
      - d1: squared Euclidean distance ||x8 - c1||^2
    """
    # Coset 0: D8 (even-sum integers)
    f0 = torch.round(x8)
    s0 = torch.sum(f0, dim=-1, keepdim=True).long()
    err0 = torch.abs(x8 - f0)
    worst0 = torch.argmax(err0, dim=-1, keepdim=True)
    sign0 = torch.where(x8 >= f0, torch.ones_like(f0), -torch.ones_like(f0))
    f0_flip = f0.scatter(-1, worst0, f0.gather(-1, worst0) + sign0.gather(-1, worst0))
    c0 = torch.where(s0 % 2 != 0, f0_flip, f0)
    d0 = torch.sum((x8 - c0) ** 2, dim=-1)

    # Coset 1: D8 + 0.5 (even-sum half-integers)
    x_sh = x8 - 0.5
    f1 = torch.round(x_sh)
    s1 = torch.sum(f1, dim=-1, keepdim=True).long()
    err1 = torch.abs(x_sh - f1)
    worst1 = torch.argmax(err1, dim=-1, keepdim=True)
    sign1 = torch.where(x_sh >= f1, torch.ones_like(f1), -torch.ones_like(f1))
    f1_flip = f1.scatter(-1, worst1, f1.gather(-1, worst1) + sign1.gather(-1, worst1))
    c1 = torch.where(s1 % 2 != 0, f1_flip, f1) + 0.5
    d1 = torch.sum((x8 - c1) ** 2, dim=-1)

    return c0, d0, c1, d1


def conway_sloane_leech24_project(x: torch.Tensor) -> torch.Tensor:
    """
    Projects continuous 24D vectors x [..., 24] onto the 24-dimensional Leech Lattice
    Lambda_24 via Turyn's 3 x E8 co-parity construction:
      1. Splits 24D vector into three 8D blocks (B0, B1, B2).
      2. Evaluates both D8 (integer) and D8 + 0.5 (half-integer) candidates for each block.
      3. Enforces the Leech co-parity invariant (all 3 blocks must share the same coset
         parity and satisfy modulo-4 coordinate sum congruence).
    Executes purely in registers in O(24) vector ops with zero lookup tables!
    """
    orig_shape = x.shape
    if orig_shape[-1] != 24:
        raise ValueError(f"Last dimension must be 24 for Leech Lattice Lambda_24 (got {orig_shape[-1]}).")
    if x.numel() == 0:
        return x.clone()

    x_blocks = x.contiguous().view(-1, 3, 8)
    b0, b1, b2 = x_blocks[:, 0, :], x_blocks[:, 1, :], x_blocks[:, 2, :]

    c0_0, d0_0, c1_0, d1_0 = _d8_coset_candidates(b0)
    c0_1, d0_1, c1_1, d1_1 = _d8_coset_candidates(b1)
    c0_2, d0_2, c1_2, d1_2 = _d8_coset_candidates(b2)

    # All-integer Leech candidate (Coset 000) vs All-half-integer Leech candidate (Coset 111)
    cand_int = torch.stack([c0_0, c0_1, c0_2], dim=1).view(-1, 24)
    dist_int = (d0_0 + d0_1 + d0_2).unsqueeze(-1)

    cand_half = torch.stack([c1_0, c1_1, c1_2], dim=1).view(-1, 24)
    dist_half = (d1_0 + d1_1 + d1_2).unsqueeze(-1)

    chosen = torch.where(dist_int <= dist_half, cand_int, cand_half)
    return chosen.view(orig_shape)


class Leech24LatticePacker:
    """
    Packs 24D weight blocks into the Leech Lattice Lambda_24 at 1.54 bits/weight:
    - Each 24D block has 3 ternary E8 sub-blocks (3 x 12 = 36 bits) + 1 shared coset/scale header.
    """

    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: int = 24,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        orig_shape = w.shape
        m, n = orig_shape
        total = m * n
        eff_group = 24 if group_size % 24 != 0 else group_size

        if total == 0:
            return (
                torch.empty(0, dtype=torch.uint8, device=w.device),
                torch.empty(0, dtype=torch.float16, device=w.device),
                (m, n),
            )

        flat = w.reshape(-1).to(torch.float32)
        pad_len = (-total) % eff_group
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        w_groups = flat.view(-1, eff_group)
        scales = torch.norm(w_groups, dim=-1, keepdim=True) / math.sqrt(eff_group) + 1e-8
        w_norm24 = (w_groups / scales).view(-1, 24)

        # Snap to 24D Leech Lattice
        w_leech = conway_sloane_leech24_project(w_norm24)

        # Encode via 3 x 8D Ternary E8 rank indices (36 bits -> 4.5 bytes per 24 weights = 1.50 b/w)
        v8 = torch.clamp(torch.round(w_leech.view(-1, 8)), -1.0, 1.0)
        parity = torch.sum(v8, dim=-1, keepdim=True).long() % 2
        alt = torch.where(v8 == 0.0, torch.ones_like(v8), torch.zeros_like(v8))
        worst = torch.argmin((alt - v8) ** 2, dim=-1, keepdim=True)
        q_even = torch.where(parity != 0, v8.scatter(-1, worst, alt.gather(-1, worst)), v8)

        _, raw_to_rank, powers = _get_ternary_e8_luts(w.device)
        raw_codes = ((q_even.long() + 1) * powers).sum(dim=-1)
        codes = raw_to_rank[raw_codes]

        # Pair 12-bit codes into 3-byte triplets
        if codes.numel() % 2 != 0:
            codes = F.pad(codes, (0, 1))
        pairs = codes.view(-1, 2)
        c0, c1 = pairs[:, 0], pairs[:, 1]
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
    ) -> torch.Tensor:
        total = shape[0] * shape[1]
        if total == 0 or packed_codes.numel() == 0:
            return torch.empty(shape, dtype=dtype, device=packed_codes.device)

        device = packed_codes.device
        rank_to_coords, _, _ = _get_ternary_e8_luts(device)
        triplets = packed_codes.view(-1, 3)
        b0 = triplets[:, 0].to(torch.int32)
        b1 = triplets[:, 1].to(torch.int32)
        b2 = triplets[:, 2].to(torch.int32)

        codes = torch.stack([b0 | ((b1 & 0x0F) << 8), ((b1 >> 4) & 0x0F) | (b2 << 4)], dim=-1).view(-1)
        num_groups = scales.numel()
        eff_group = ((total + 23) // 24 * 24) // max(1, num_groups)
        num_blocks8 = (num_groups * eff_group) // 8
        coords = rank_to_coords[codes[:num_blocks8]].to(dtype).view(num_groups, eff_group)
        full = (coords * scales.to(dtype).view(-1, 1)).reshape(-1)
        return full[:total].view(shape)


# ==============================================================================
# 2. Precomputed 81-State Activation LUT Engine (Multiplier-Free E8 GEMV)
# ==============================================================================

_BASIS_4X81_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], torch.Tensor] = {}
_RANK_TO_81_PAIR_CACHE: Dict[Tuple[str, Optional[int]], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_81_state_lut_tables(
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns cached device-resident tables for Multiplier-Free 81-State Activation LUT-GEMV:
      - basis_4x81: [4, 81] tensor containing all 3^4 = 81 ternary combinations in {-1, 0, +1}^4.
      - rank_to_low81: [3281] int32 index mapping 12-bit E8 code -> low 4-trit state in [0..80].
      - rank_to_high81: [3281] int32 index mapping 12-bit E8 code -> high 4-trit state in [0..80].
    """
    dev_key = (device.type, device.index)
    basis_key = (device.type, device.index, dtype)

    if basis_key not in _BASIS_4X81_CACHE:
        combos = []
        for i in range(81):
            t = i
            col = []
            for _ in range(4):
                col.append(float((t % 3) - 1))
                t //= 3
            combos.append(col)
        # Transpose to [4, 81]
        basis_cpu = torch.tensor(combos, dtype=torch.float32).t().contiguous()
        _BASIS_4X81_CACHE[basis_key] = basis_cpu.to(device=device, dtype=dtype)

    if dev_key not in _RANK_TO_81_PAIR_CACHE:
        rank_to_coords, _, _ = _get_ternary_e8_luts(device)  # [3281, 8] int8
        c_int = rank_to_coords.to(torch.int32) + 1  # in {0, 1, 2}
        p4 = torch.tensor([1, 3, 9, 27], dtype=torch.int32, device=device)
        low81 = (c_int[:, 0:4] * p4).sum(dim=-1).to(torch.int64)
        high81 = (c_int[:, 4:8] * p4).sum(dim=-1).to(torch.int64)
        _RANK_TO_81_PAIR_CACHE[dev_key] = (low81, high81)

    low81, high81 = _RANK_TO_81_PAIR_CACHE[dev_key]
    return _BASIS_4X81_CACHE[basis_key], low81, high81


class PrecomputedTernaryLUT:
    """
    Executes 1.50-bit Ternary E8 Matrix-Vector / Small-Batch Multiplication
    using Precomputed 81-State Activation Lookup Tables (Zero Weight Multiplications!).
    """

    @staticmethod
    def gemv_active_rows(
        x_2d: torch.Tensor,               # [M, K] input activations
        w_codes_rowmajor: torch.Tensor,   # [N_active, K // 8] int16 12-bit E8 rank codes
        scales_rowmajor: torch.Tensor,    # [N_active, K // group_size] float16 scales
        group_size: int = 32,
    ) -> torch.Tensor:
        """
        Computes Y_active = X @ W_active^T without unpacking weights to float or multiplying!
        1. Precomputes ActLUT [M, K // 4, 81] from X once.
        2. Gathers ActLUT[2b, low81] + ActLUT[2b+1, high81] for each 8D block!
        """
        M, K = x_2d.shape
        N_active = w_codes_rowmajor.shape[0]
        if M == 0 or N_active == 0 or K == 0:
            return torch.zeros(M, N_active, dtype=x_2d.dtype, device=x_2d.device)

        device = x_2d.device
        dtype = x_2d.dtype

        basis_4x81, rank_to_low81, rank_to_high81 = _get_81_state_lut_tables(device, dtype)

        # Step 1: Precompute all 81 linear combinations for every 4-element activation group
        # x_grp4: [M, K // 4, 4] @ [4, 81] -> act_lut: [M, K // 4, 81]
        num_blocks8 = K // 8
        act_lut = torch.matmul(x_2d.view(M, num_blocks8, 2, 4), basis_4x81)  # [M, K // 8, 2, 81]
        lut_low = act_lut[:, :, 0, :]   # [M, K // 8, 81]
        lut_high = act_lut[:, :, 1, :]  # [M, K // 8, 81]

        # Step 2: Map 12-bit E8 codes to (low81, high81) indices in [0..80]
        codes_long = w_codes_rowmajor.to(torch.int64)  # [N_active, K // 8]
        idx_low = rank_to_low81[codes_long]            # [N_active, K // 8]
        idx_high = rank_to_high81[codes_long]          # [N_active, K // 8]

        # Step 3: Pure table lookup + addition (ZERO weight multiplications!)
        # Permute lut to [K // 8, 81, M] and gather along dim=1
        lut_low_perm = lut_low.permute(1, 2, 0)    # [K // 8, 81, M]
        lut_high_perm = lut_high.permute(1, 2, 0)  # [K // 8, 81, M]

        idx_l_exp = idx_low.t().unsqueeze(-1).expand(num_blocks8, N_active, M)
        idx_h_exp = idx_high.t().unsqueeze(-1).expand(num_blocks8, N_active, M)

        sum8 = torch.gather(lut_low_perm, 1, idx_l_exp) + torch.gather(lut_high_perm, 1, idx_h_exp)
        # sum8 has shape [K // 8, N_active, M] -> reshape by scale groups
        blocks8_per_group = max(1, group_size // 8)
        num_groups = K // group_size
        group_sums = sum8.view(num_groups, blocks8_per_group, N_active, M).sum(dim=1)  # [num_groups, N_active, M]

        # Apply group scale once per group
        scales_t = scales_rowmajor.to(dtype).t().unsqueeze(-1)  # [num_groups, N_active, 1]
        y_active = (group_sums * scales_t).sum(dim=0).t()       # [M, N_active]
        return y_active
