"""
Fused Triton E8 Dequant-GEMM Kernels for NVIDIA CUDA & AMD ROCm GPUs.

Unpacks 2.0-bit and 1.5-bit Ternary E8 weights directly in SRAM/registers during
tiled matmul accumulation, eliminating full-weight FP16/FP32 DRAM materialization.
Falls back cleanly (returns None) when Triton or a supported GPU is unavailable.
"""

from __future__ import annotations

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_e8_2bit_gemm_kernel(
        X_ptr,
        W_pack_ptr,
        Scales_ptr,
        Bias_ptr,
        Y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sg,
        stride_ym,
        stride_yn,
        GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            x_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            x_tile = tl.load(
                X_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
                mask=x_mask,
                other=0.0,
            ).to(tl.float32)

            # In-register 2-bit unpack: 4 weights per uint8 byte
            byte_idx = k_idx >> 2
            bit_shift = (k_idx & 3) * 2
            w_mask = (offs_n[:, None] < N) & (k_idx[None, :] < K)
            packed_bytes = tl.load(
                W_pack_ptr + offs_n[:, None] * stride_wn + byte_idx[None, :] * stride_wk,
                mask=w_mask,
                other=0,
            ).to(tl.int32)
            codes = ((packed_bytes >> bit_shift[None, :]) & 3).to(tl.float32) - 1.5

            # Group scale lookup (safe for any group_size, including group_size > BLOCK_K)
            g_idx = k_idx // GROUP_SIZE
            scales = tl.load(
                Scales_ptr + offs_n[:, None] * stride_sn + g_idx[None, :] * stride_sg,
                mask=w_mask,
                other=0.0,
            ).to(tl.float32)

            w_tile = codes * scales  # [BLOCK_N, BLOCK_K]
            acc += tl.dot(x_tile, tl.trans(w_tile))

        if HAS_BIAS:
            b = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += b[None, :]

        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc,
            mask=y_mask,
        )


def triton_fused_e8_2bit_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> Optional[torch.Tensor]:
    """Run fused 2.0-bit E8 dequant-GEMM on CUDA/ROCm via Triton. Returns None if unavailable."""
    if not HAS_TRITON or not x.is_cuda:
        return None
    if in_features % 4 != 0 or in_features % group_size != 0:
        return None

    orig_shape = x.shape
    x_2d = x.reshape(-1, in_features).contiguous()
    M, K = x_2d.shape
    N = int(out_features)
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((*orig_shape[:-1], N), dtype=x.dtype, device=x.device)

    w_2d = packed_weight.reshape(N, K // 4).contiguous()
    s_2d = block_scales.reshape(N, K // group_size).contiguous()
    y = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 16 if M < 32 else (32 if M < 64 else 64)
    BLOCK_N = 64
    BLOCK_K = 64 if K >= 64 else 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bias_ptr = bias.contiguous() if bias is not None else x_2d
    _triton_e8_2bit_gemm_kernel[grid](
        x_2d,
        w_2d,
        s_2d,
        bias_ptr,
        y,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w_2d.stride(0),
        w_2d.stride(1),
        s_2d.stride(0),
        s_2d.stride(1),
        y.stride(0),
        y.stride(1),
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y.to(x.dtype).reshape(*orig_shape[:-1], N)


__all__ = ["HAS_TRITON", "triton_fused_e8_2bit_gemm"]
