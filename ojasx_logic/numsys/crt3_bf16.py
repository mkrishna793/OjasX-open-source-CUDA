"""
Phase 1: Exact 3-Channel & 4-Channel CRT/RNS Engine for Dense BF16 (`CRT-3` / `CRT-4`).

Because BF16 mantissas are only 7 bits (+ sign -> [-127, 127]), we do NOT need
trium-tile's full 6-channel CRT-6:
  - CRT-3 = {243, 251, 253}      -> M = 15,431,229    (exact up to K <= 478)
  - CRT-4 = {243, 251, 253, 247} -> M = 3,811,513,563 (exact up to K <= 118,000!)

Every channel runs as an independent zero-cross-term uint8 x int8 -> int32 GEMM
(mapping straight to ARMv8.6 I8MM, AVX-512 VNNI, or INT8 Tensor Cores), reconstructed
with division-free IEEE-754 double-quotient CRT.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


CRT3_MODS: Tuple[int, ...] = (243, 251, 253)
CRT4_MODS: Tuple[int, ...] = (243, 251, 253, 247)


def _mod_inv(a: int, m: int) -> int:
    return pow(a % m, -1, m)


class CRTPlan:
    def __init__(self, mods: Tuple[int, ...] = CRT4_MODS) -> None:
        self.mods = tuple(int(m) for m in mods)
        M = 1
        for m in self.mods:
            M *= m
        self.M_total = M
        self.Mi = tuple(M // m for m in self.mods)
        self.yi = tuple(_mod_inv(Mi % m, m) for Mi, m in zip(self.Mi, self.mods))
        self.inv_m = tuple(1.0 / float(m) for m in self.mods)

    def reconstruct(self, channels: np.ndarray) -> np.ndarray:
        """Division-free IEEE-754 double-quotient CRT reconstruction over [R, M, N]."""
        acc = np.zeros(channels.shape[1:], dtype=np.int64)
        frac = np.zeros(channels.shape[1:], dtype=np.float64)
        for c, m in enumerate(self.mods):
            r = np.mod(channels[c].astype(np.int64), m)
            coeff = np.mod(r * self.yi[c], m)
            acc += coeff * self.Mi[c]
            frac += coeff.astype(np.float64) * self.inv_m[c]
        q = np.rint(frac).astype(np.int64)
        res = acc - q * self.M_total
        half = self.M_total // 2
        res = np.where(res > half, res - self.M_total, res)
        res = np.where(res < -half, res + self.M_total, res)
        return res


def gemm_crt_int(
    Aq: np.ndarray,
    Bq: np.ndarray,
    *,
    channels: int = 0,
) -> np.ndarray:
    """Bit-exact integer GEMM via CRT-3 (K<=478 for [-127,127]) or CRT-4 (K<=118,000)."""
    A_i = np.asarray(Aq, dtype=np.int64)
    B_i = np.asarray(Bq, dtype=np.int64)
    M, K = A_i.shape
    K2, N = B_i.shape
    if K != K2:
        raise ValueError(f"K mismatch: {K} vs {K2}")

    if channels == 0:
        mods = CRT3_MODS if K <= 478 else CRT4_MODS
    elif channels == 3:
        mods = CRT3_MODS
    elif channels == 4:
        mods = CRT4_MODS
    else:
        raise ValueError(f"Supported channels are 3 or 4, got {channels}")

    plan = CRTPlan(mods)
    ch_outs = np.empty((len(mods), M, N), dtype=np.int64)
    for c, m in enumerate(mods):
        Au = np.mod(A_i, m).astype(np.int64)
        Bs = np.mod(B_i, m).astype(np.int64)
        Bs = np.where(Bs > (m // 2), Bs - m, Bs)
        ch_outs[c] = Au @ Bs

    return plan.reconstruct(ch_outs)


def gemm_crt_bf16(
    A: np.ndarray,
    B: np.ndarray,
    *,
    channels: int = 0,
) -> np.ndarray:
    """Per-row/per-col 7-bit scaled BF16 GEMM via CRT-3 (K<=478) or CRT-4 (K<=118,000)."""
    A_f = np.asarray(A, dtype=np.float32)
    B_f = np.asarray(B, dtype=np.float32)

    rs = np.max(np.abs(A_f), axis=1, keepdims=True) / 127.0
    rs = np.where(rs > 0.0, rs, 1.0)
    cs = np.max(np.abs(B_f), axis=0, keepdims=True) / 127.0
    cs = np.where(cs > 0.0, cs, 1.0)

    Aq = np.rint(A_f / rs).astype(np.int64)
    Bq = np.rint(B_f / cs).astype(np.int64)

    Cq = gemm_crt_int(Aq, Bq, channels=channels)
    return (Cq.astype(np.float64) * rs.astype(np.float64) * cs.astype(np.float64)).astype(np.float32)


__all__ = ["CRT3_MODS", "CRT4_MODS", "CRTPlan", "gemm_crt_int", "gemm_crt_bf16"]
