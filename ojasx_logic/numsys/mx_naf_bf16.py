"""
Phase 1: Block-Exponent NAF5 Dual-Rail Number System (`MX-NAF-BF16`).

Bridges dense BF16 tensors directly onto multiplier-free bitwise Dual-Rail POPCNT ALUs:
  1. Groups every 32 BF16 values under one shared block exponent e_b (Microscaling).
  2. Aligns all 32 mantissas to exact 8-bit signed integers q in [-127, +127].
  3. Encodes each mantissa with 5-instruction Canonical NAF5 (`naf5_encode`) ->
     average 2.67 non-zero trits per 8-bit mantissa (~66.7% zero sparsity, zero
     adjacent non-zero trits!).
  4. Executes exact inner products via pure bitwise AND + OR + POPCNT across
     Dual-Rail bitplanes (P, N) with ZERO floating-point multipliers in the inner loop!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class MXNAFTensor:
    """Block-Exponent NAF5 Dual-Rail tensor representation."""
    shape: Tuple[int, ...]
    block_size: int
    bitplanes: int
    exponents: np.ndarray      # int8 [..., num_blocks] shared block exponents
    pos_planes: np.ndarray     # uint32 [..., num_blocks, bitplanes] (1 bit per elem in 32-block)
    neg_planes: np.ndarray     # uint32 [..., num_blocks, bitplanes]
    nnz_trits: int
    total_trits: int

    @property
    def zero_sparsity(self) -> float:
        return 1.0 - (self.nnz_trits / max(1, self.total_trits))

    @property
    def effective_bdr_bits_per_elem(self) -> float:
        """Theoretical BDR storage: 1 occupancy bit per active plane + 1 sign bit per nz trit + 8/32 exp."""
        # Across 32 elements: 8-bit shared exp + (bitplanes * 32 occupancy bits) + nnz_trits sign bits
        num_elems = int(np.prod(self.shape))
        num_blocks = self.exponents.size
        total_bits = num_blocks * 8 + self.pos_planes.size * 32 + self.nnz_trits
        return total_bits / max(1, num_elems)


def naf5_encode_i32(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """5-instruction loop-free Canonical Non-Adjacent Form (NAF5) encoder.

    Guarantees:
      1. (P & N) == 0  (disjoint rails)
      2. ((P | N) & ((P | N) << 1)) == 0  (no adjacent non-zero trits!)
      3. P - N == q  (exact signed integer value)
    """
    q64 = np.asarray(q, dtype=np.int64)
    # Work in unsigned 16-bit two's complement domain for exact 8-bit signed q in [-128, 127]
    # Handle signed values via sign magnitude so P, N fit cleanly in non-negative bitplanes [0..8]
    sign = q64 < 0
    mag = np.abs(q64).astype(np.uint64)
    xh = mag >> 1
    x3 = mag + xh
    c = xh ^ x3
    p_mag = x3 & c
    n_mag = xh & c
    # Swap rails for negative inputs so (P - N) == q identically
    P = np.where(sign, n_mag, p_mag).astype(np.uint16)
    N = np.where(sign, p_mag, n_mag).astype(np.uint16)
    return P, N


def to_bf16_rounded_f32(x: np.ndarray) -> np.ndarray:
    """Round float32 array to exact IEEE-754 BF16 (round-to-nearest-even) represented in float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    # Round-to-nearest-even at bit 16
    lsb = (u >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    u_bf16 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u_bf16.view(np.float32)


def encode_mx_naf_bf16(
    x: np.ndarray,
    *,
    block_size: int = 32,
    bitplanes: int = 9,
) -> MXNAFTensor:
    """Encode a 2D matrix [M, K] (K divisible by 32) into MX-NAF-BF16 Dual-Rail bitplanes."""
    if block_size != 32:
        raise ValueError("MX-NAF-BF16 uses 32-element blocks matching 32-bit GPU/SIMD words")
    arr = to_bf16_rounded_f32(np.asarray(x, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] % 32 != 0:
        raise ValueError(f"Expected 2D array with K % 32 == 0, got {arr.shape}")
    M, K = arr.shape
    nb = K // 32
    blocks = arr.reshape(M, nb, 32)

    # Shared block exponent: scale max(|block|) to 127 (7-bit signed mantissa + sign)
    amax = np.max(np.abs(blocks), axis=-1)  # [M, nb]
    safe_amax = np.where(amax > 0.0, amax, 1.0)
    exp_f = np.ceil(np.log2(safe_amax)).astype(np.int32)
    exp_i8 = np.clip(exp_f, -120, 120).astype(np.int8)
    scale = np.ldexp(np.ones_like(amax, dtype=np.float32), exp_i8.astype(np.int32) - 7)  # 2^(e - 7)
    scale = np.where(amax > 0.0, scale, 1.0)

    q = np.clip(np.rint(blocks / scale[..., None]), -127, 127).astype(np.int64)
    P_raw, N_raw = naf5_encode_i32(q)  # [M, nb, 32] uint16

    # Bit-transpose 32 elements into `bitplanes` uint32 words!
    shifts = np.arange(32, dtype=np.uint32)
    pos_planes = np.zeros((M, nb, bitplanes), dtype=np.uint32)
    neg_planes = np.zeros((M, nb, bitplanes), dtype=np.uint32)
    nnz = 0
    for b in range(bitplanes):
        pb = ((P_raw >> b) & 1).astype(np.uint32)
        nb_bits = ((N_raw >> b) & 1).astype(np.uint32)
        nnz += int(pb.sum() + nb_bits.sum())
        pos_planes[:, :, b] = np.bitwise_or.reduce(pb << shifts, axis=-1)
        neg_planes[:, :, b] = np.bitwise_or.reduce(nb_bits << shifts, axis=-1)

    return MXNAFTensor(
        shape=(M, K),
        block_size=32,
        bitplanes=bitplanes,
        exponents=exp_i8,
        pos_planes=pos_planes,
        neg_planes=neg_planes,
        nnz_trits=nnz,
        total_trits=M * K * bitplanes,
    )


def decode_mx_naf_bf16(t: MXNAFTensor) -> np.ndarray:
    """Decode MXNAFTensor back to float32."""
    M, K = t.shape
    nb = K // 32
    shifts = np.arange(32, dtype=np.uint32)
    q = np.zeros((M, nb, 32), dtype=np.int32)
    for b in range(t.bitplanes):
        pb = ((t.pos_planes[:, :, b, None] >> shifts) & 1).astype(np.int32)
        nb_bits = ((t.neg_planes[:, :, b, None] >> shifts) & 1).astype(np.int32)
        q += (pb - nb_bits) << b
    scale = np.ldexp(np.ones((M, nb), dtype=np.float32), t.exponents.astype(np.int32) - 7)
    return (q.astype(np.float32) * scale[..., None]).reshape(M, K)


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _popcnt_u32(u: np.ndarray) -> np.ndarray:
    return (
        _POPCNT8[u & 0xFF]
        + _POPCNT8[(u >> 8) & 0xFF]
        + _POPCNT8[(u >> 16) & 0xFF]
        + _POPCNT8[(u >> 24) & 0xFF]
    )


def gemm_mx_naf_dual_rail(A: MXNAFTensor, Bt: MXNAFTensor) -> np.ndarray:
    """Multiplier-free Dual-Rail POPCNT GEMM: C[M, N] = A[M, K] @ Bt[N, K]^T.

    Inner loop uses strictly bitwise AND, OR, and POPCNT across bitplanes!
    """
    M, K = A.shape
    N, K2 = Bt.shape
    if K != K2:
        raise ValueError(f"K mismatch: {K} vs {K2}")
    nb = K // 32
    C = np.zeros((M, N), dtype=np.float64)

    for blk in range(nb):
        # Active bitplanes for this 32-element block
        ap = A.pos_planes[:, blk, :]    # [M, ba]
        an = A.neg_planes[:, blk, :]
        bp = Bt.pos_planes[:, blk, :]   # [N, bb]
        bn = Bt.neg_planes[:, blk, :]

        block_acc = np.zeros((M, N), dtype=np.int64)
        for a in range(A.bitplanes):
            ap_a = ap[:, a][:, None]    # [M, 1]
            an_a = an[:, a][:, None]
            if not (ap_a.any() or an_a.any()):
                continue
            for b in range(Bt.bitplanes):
                bp_b = bp[:, b][None, :]  # [1, N]
                bn_b = bn[:, b][None, :]
                if not (bp_b.any() or bn_b.any()):
                    continue
                pos_mask = (ap_a & bp_b) | (an_a & bn_b)
                neg_mask = (ap_a & bn_b) | (an_a & bp_b)
                diff = _popcnt_u32(pos_mask) - _popcnt_u32(neg_mask)
                block_acc += diff.astype(np.int64) << (a + b)

        exp_sum = A.exponents[:, blk].astype(np.int32)[:, None] + Bt.exponents[:, blk].astype(np.int32)[None, :] - 14
        C += np.ldexp(block_acc.astype(np.float64), exp_sum)

    return C.astype(np.float32)


__all__ = [
    "MXNAFTensor",
    "naf5_encode_i32",
    "to_bf16_rounded_f32",
    "encode_mx_naf_bf16",
    "decode_mx_naf_bf16",
    "gemm_mx_naf_dual_rail",
]
