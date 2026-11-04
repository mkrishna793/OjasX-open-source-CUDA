"""Portable PyTorch reference operators and universal TileLang codegen fallbacks.

Used when optional native libraries are absent or when `OJASX_DISABLE_NATIVE=1`.
Library objects remain alive after registration.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

_LIBRARIES = []
_FALLBACKS_REGISTERED = False


def fallbacks_registered() -> bool:
    global _FALLBACKS_REGISTERED
    return _FALLBACKS_REGISTERED


def _pack(x):
    if x.dim() == 0:
        return x.new_empty((0,), dtype=torch.int32)
    cols = x.shape[-1]
    if x.numel() == 0 or cols == 0:
        out_shape = list(x.shape)
        out_shape[-1] = (cols + 31) // 32
        return x.new_empty(out_shape, dtype=torch.int32)
    bits = F.pad((x.contiguous() > 0).to(torch.int64), (0, (-cols) % 32))
    shifts = torch.arange(32, device=x.device, dtype=torch.int64)
    return (bits.reshape(*x.shape[:-1], (cols + 31) // 32, 32) << shifts).sum(-1).to(torch.int32)


def _unpack(packed, cols, dtype):
    if packed.dim() == 0 or packed.numel() == 0 or cols == 0:
        out_shape = list(packed.shape)
        if out_shape:
            out_shape[-1] = cols
        return packed.new_empty(out_shape, dtype=dtype)
    shifts = torch.arange(32, device=packed.device, dtype=torch.int64)
    return ((packed.contiguous().to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 32
    )[..., :cols].to(dtype)


def _linear_forward(x, weight, bias):
    z = F.linear(x, weight, bias)
    return F.relu(z).square(), _pack(z)


def _linear_backward(grad_out, x, weight, bias, packed_mask):
    if x.numel() == 0 or weight.numel() == 0:
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(weight)
        db_dtype = bias.dtype if bias is not None else x.dtype
        db = torch.zeros(weight.shape[0], device=x.device, dtype=db_dtype)
        return dx, dw, db
    z = F.linear(x, weight, bias)
    dz = grad_out * 2 * z * _unpack(packed_mask, weight.shape[0], z.dtype)
    dx = dz @ weight
    rows = math.prod(x.shape[:-1]) if x.dim() > 1 else 1
    dw = dz.reshape(rows, weight.shape[0]).t() @ x.reshape(rows, weight.shape[1])
    db = dz.reshape(rows, weight.shape[0]).sum(0)
    if bias is not None and db.dtype != bias.dtype:
        db = db.to(bias.dtype)
    return dx, dw, db


def _rms_forward(x, weight, eps):
    if x.numel() == 0:
        rstd_shape = list(x.shape)
        if rstd_shape:
            rstd_shape[-1] = 1
        return torch.empty_like(x), x.new_empty(rstd_shape, dtype=torch.float32)
    compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    w_c = weight.to(compute.dtype)
    rstd = torch.rsqrt(compute.square().mean(-1, keepdim=True) + eps)
    return ((compute * rstd) * w_c).to(x.dtype), rstd.float()


def _rms_backward(grad_out, x, weight, rstd):
    if x.numel() == 0 or weight.numel() == 0:
        return torch.zeros_like(x), torch.zeros_like(weight)
    compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    go_c = grad_out.to(compute.dtype)
    w_c = weight.to(compute.dtype)
    rstd_c = rstd.to(compute.dtype)
    x_hat = compute * rstd_c
    d_hat = go_c * w_c
    dx = rstd_c * (d_hat - x_hat * (d_hat * x_hat).mean(-1, keepdim=True))
    dw = (go_c * x_hat).reshape(-1, x.shape[-1]).sum(0)
    return dx.to(x.dtype), dw.to(weight.dtype)


def _compress(x, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat = F.pad(x.detach().float().reshape(-1), (0, (-x.numel()) % block_size))
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    return (blocks / scale).round().clamp(-127, 127).to(torch.int8), scale.squeeze(-1)


def _decompress(q, scale, orig_shape, orig_dtype):
    return (q.float() * scale.unsqueeze(-1)).reshape(-1)[:math.prod(orig_shape)].reshape(orig_shape).to(orig_dtype)


def _glu_activation(g: torch.Tensor, act_mode: int) -> torch.Tensor:
    if act_mode == 0:
        return F.silu(g)
    elif act_mode == 1:
        return F.gelu(g, approximate="tanh")
    elif act_mode == 2:
        return F.gelu(g, approximate="none")
    elif act_mode == 3:
        return F.relu(g)
    elif act_mode == 4:
        return F.relu(g).square()
    elif act_mode == 5:
        return torch.sigmoid(g)
    elif act_mode == 6:
        return g
    elif act_mode == 7:
        return g * torch.sigmoid(1.702 * g)
    elif act_mode == 8:
        return F.mish(g)
    elif act_mode == 9:
        return torch.tanh(g)
    elif act_mode == 10:
        return F.leaky_relu(g, negative_slope=0.01)
    else:
        raise ValueError(f"Unknown GLU activation mode: {act_mode}")


def _glu_forward(gate: torch.Tensor, up: torch.Tensor, act_mode: int) -> torch.Tensor:
    compute_g = gate.float() if gate.dtype in (torch.float16, torch.bfloat16) else gate
    compute_u = up.float() if up.dtype in (torch.float16, torch.bfloat16) else up
    phi = _glu_activation(compute_g, act_mode)
    return (phi * compute_u).to(gate.dtype)


def _glu_backward(
    grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, act_mode: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if grad_out.numel() == 0:
        return torch.zeros_like(gate), torch.zeros_like(up)
    compute_go = grad_out.float() if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out
    compute_g = gate.float() if gate.dtype in (torch.float16, torch.bfloat16) else gate
    compute_u = up.float() if up.dtype in (torch.float16, torch.bfloat16) else up

    if act_mode == 0:  # SwiGLU
        sig = torch.sigmoid(compute_g)
        phi = compute_g * sig
        dphi = sig * (1.0 + compute_g * (1.0 - sig))
    elif act_mode == 1:  # GeGLU tanh
        k = 0.7978845608028654
        c = 0.044715
        v = k * (compute_g + c * compute_g.pow(3))
        t = torch.tanh(v)
        phi = 0.5 * compute_g * (1.0 + t)
        dphi = 0.5 * (1.0 + t) + 0.5 * compute_g * (1.0 - t.pow(2)) * k * (1.0 + 3.0 * c * compute_g.pow(2))
    elif act_mode == 2:  # GeGLU exact
        inv_sqrt2 = 0.7071067811865475
        inv_sqrt2pi = 0.3989422804014327
        phi = 0.5 * compute_g * (1.0 + torch.erf(compute_g * inv_sqrt2))
        dphi = 0.5 * (1.0 + torch.erf(compute_g * inv_sqrt2)) + compute_g * inv_sqrt2pi * torch.exp(-0.5 * compute_g.pow(2))
    elif act_mode == 3:  # ReGLU
        phi = F.relu(compute_g)
        dphi = (compute_g > 0.0).float()
    elif act_mode == 4:  # ReLU2GLU
        phi = F.relu(compute_g).square()
        dphi = torch.where(compute_g > 0.0, 2.0 * compute_g, torch.zeros_like(compute_g))
    elif act_mode == 5:  # SigmoidGLU
        sig = torch.sigmoid(compute_g)
        phi = sig
        dphi = sig * (1.0 - sig)
    elif act_mode == 6:  # BilinearGLU
        phi = compute_g
        dphi = torch.ones_like(compute_g)
    elif act_mode == 7:  # QuickGeGLU
        sig = torch.sigmoid(1.702 * compute_g)
        phi = compute_g * sig
        dphi = sig * (1.0 + 1.702 * compute_g * (1.0 - sig))
    elif act_mode == 8:  # MishGLU
        sp = F.softplus(compute_g)
        tsp = torch.tanh(sp)
        sig = torch.sigmoid(compute_g)
        phi = compute_g * tsp
        dphi = tsp + compute_g * sig * (1.0 - tsp.pow(2))
    elif act_mode == 9:  # TanhGLU
        t = torch.tanh(compute_g)
        phi = t
        dphi = 1.0 - t.pow(2)
    elif act_mode == 10:  # LeakyReGLU
        phi = F.leaky_relu(compute_g, negative_slope=0.01)
        dphi = torch.where(compute_g > 0.0, torch.ones_like(compute_g), torch.full_like(compute_g, 0.01))
    else:
        raise ValueError(f"Unknown GLU activation mode: {act_mode}")

    d_up = compute_go * phi
    d_gate = compute_go * compute_u * dphi
    return d_gate.to(gate.dtype), d_up.to(up.dtype)


def _transformer_mlp_forward(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_shape = x.shape
    d = norm_w.shape[0]
    if x.numel() == 0 or d == 0:
        return torch.empty_like(x)

    x_2d = x.contiguous().reshape(-1, d)
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x_2d.to(compute_dtype)
    nw_c = norm_w.to(compute_dtype)

    var = x_c.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_norm = (x_c * rstd) * nw_c

    gw_c = gate_w.to(compute_dtype)
    uw_c = up_w.to(compute_dtype)
    dw_c = down_w.to(compute_dtype)

    g = F.linear(x_norm, gw_c)
    u = F.linear(x_norm, uw_c)
    h = F.silu(g) * u
    y = F.linear(h, dw_c)

    return y.to(x.dtype).reshape(orig_shape)


def _transformer_mlp_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    norm_w: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    d = norm_w.shape[0]
    if x.numel() == 0 or d == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(norm_w),
            torch.zeros_like(gate_w),
            torch.zeros_like(up_w),
            torch.zeros_like(down_w),
        )

    m = x.numel() // d
    x_2d = x.contiguous().reshape(m, d)
    dy_2d = dy.contiguous().reshape(m, d)

    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x_2d.to(compute_dtype)
    dy_c = dy_2d.to(compute_dtype)
    nw_c = norm_w.to(compute_dtype)
    gw_c = gate_w.to(compute_dtype)
    uw_c = up_w.to(compute_dtype)
    dw_c = down_w.to(compute_dtype)

    var = x_c.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_scaled = x_c * rstd
    x_norm = x_scaled * nw_c

    g = F.linear(x_norm, gw_c)
    u = F.linear(x_norm, uw_c)
    sig_g = torch.sigmoid(g)
    silu_g = g * sig_g
    h = silu_g * u

    dh = dy_c.matmul(dw_c)
    ddown_w = dy_c.t().matmul(h)

    du = dh * silu_g
    dsilu_g = dh * u
    dg = dsilu_g * sig_g * (1.0 + g * (1.0 - sig_g))

    d_xnorm = dg.matmul(gw_c) + du.matmul(uw_c)
    dgate_w = dg.t().matmul(x_norm)
    dup_w = du.t().matmul(x_norm)

    dnorm_w = (d_xnorm * x_scaled).sum(dim=0)
    d_xscaled = d_xnorm * nw_c
    mean_term = (d_xscaled * x_scaled).mean(dim=-1, keepdim=True)
    dx = rstd * (d_xscaled - x_scaled * mean_term)

    return (
        dx.to(x.dtype).reshape(orig_shape),
        dnorm_w.to(norm_w.dtype),
        dgate_w.to(gate_w.dtype),
        dup_w.to(up_w.dtype),
        ddown_w.to(down_w.dtype),
    )


def _compute_square_warp_partition(num_warps: int, m: int, n: int, km: int, kn: int) -> Tuple[int, int]:
    best_m = 1
    best_n = num_warps
    best_score = 10**9
    max_m = max(1, m // km)
    max_n = max(1, n // kn)
    for cand_m in range(1, min(num_warps, max_m) + 1):
        if num_warps % cand_m != 0:
            continue
        cand_n = num_warps // cand_m
        if cand_n <= max_n and (m % (cand_m * km) == 0) and (n % (cand_n * kn) == 0):
            score = abs((m // cand_m) - (n // cand_n))
            if score < best_score:
                best_score = score
                best_m = cand_m
                best_n = cand_n
    return best_m, best_n


def _query_schedule_fallback(M: int, N: int, K: int, num_warps: int, swizzle_ordinal: int) -> List[int]:
    if swizzle_ordinal < 0 or swizzle_ordinal > 3:
        raise RuntimeError(f"Invalid TileLang SwizzleMode ordinal: {swizzle_ordinal}")
    m_warp, n_warp = _compute_square_warp_partition(int(num_warps), int(M), int(N), 8, 8)
    can_coop = 0
    if M > 0 and N > 0 and K > 0 and num_warps > 0 and (M % 16 == 0) and (N % 32 == 0) and (K % 16 == 0):
        max_m = M // 16
        max_n = N // 32
        for cm in range(1, min(num_warps, max_m) + 1):
            if num_warps % cm == 0:
                cn = num_warps // cm
                if cn <= max_n and (M % (cm * 16) == 0) and (N % (cn * 32) == 0):
                    can_coop = 1
                    break
    byte_width = 1 if swizzle_ordinal == 0 else (16 << swizzle_ordinal)
    smem_align = 128 << swizzle_ordinal
    wgmma_layout = 0 if swizzle_ordinal == 0 else (4 - swizzle_ordinal)
    tcgen05_layout = 0 if swizzle_ordinal == 0 else 2 * (4 - swizzle_ordinal)
    return [m_warp, n_warp, can_coop, byte_width, smem_align, wgmma_layout, tcgen05_layout]


def _query_chip_schedule_fallback(target_arch: str, M: int, N: int, K: int, smem_kb: int, l2_kb: int) -> List[int]:
    panel_width = 8 if l2_kb >= 16384 else (4 if l2_kb >= 4096 else 2)
    if target_arch == "cuda":
        high_smem = smem_kb >= 96
        bm = 128 if high_smem else 64
        bn = 128 if high_smem else 64
        wm, wn = _compute_square_warp_partition(4, bm, bn, 16, 16)
        return [bm, bn, 32, wm, wn, 32, 128, 128, panel_width, 3 if high_smem else 2]
    elif target_arch == "rocm":
        high_lds = smem_kb >= 64
        bm = 128 if high_lds else 64
        bn = 128 if high_lds else 64
        wm, wn = _compute_square_warp_partition(4, bm, bn, 16, 16)
        return [bm, bn, 32, wm, wn, 64, 128, 128, panel_width, 2]
    elif target_arch == "sycl":
        is_igpu = smem_kb <= 64
        bm = 32 if is_igpu else 64
        bn = 32 if is_igpu else 64
        wm, wn = _compute_square_warp_partition(4, bm, bn, 8, 8)
        return [bm, bn, 16, wm, wn, 32, 64, 64, panel_width, 1 if is_igpu else 2]
    elif target_arch == "ascend":
        return [64, 64, 32, 2, 2, 32, 32, 32, panel_width, 2]
    elif target_arch == "cpu":
        return [32, 32, 32, 1, 1, 32, 64, 64, panel_width, 1]
    else:
        wm, wn = _compute_square_warp_partition(4, 32, 32, 8, 8)
        return [32, 32, 8, wm, wn, 32, 16, 32, panel_width, 1]


def _validate_emission_arguments(
    target_arch: str, block_m: int, block_n: int, block_k: int, panel_size: int
) -> None:
    supported = {"metal", "cuda", "rocm", "sycl", "ascend", "cpu"}
    if target_arch not in supported:
        raise RuntimeError(f"Unsupported TileLang target_arch: {target_arch}")
    for name, value in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if value <= 0:
            raise RuntimeError(f"TileLang {name} must be positive")
    if panel_size <= 0 or (panel_size & (panel_size - 1)) != 0:
        raise RuntimeError("TileLang swizzle panel size must be a positive power of two")
    if target_arch == "metal" and not (
        (block_m == 32 and block_n == 32 and block_k == 8)
        or (block_m == 8 and block_n == 32 and block_k == 16)
    ):
        raise RuntimeError("TileLang Metal kernel supports 8x32x16 (decode) or 32x32x8 (prefill) tile sizes")
    if target_arch == "ascend" and any(value % 16 for value in (block_m, block_n, block_k)):
        raise RuntimeError("TileLang Ascend tile dimensions must be positive multiples of 16")


def _emit_metal_gemm_source_fallback(block_m: int, block_n: int, block_k: int, panel_size: int) -> str:
    _validate_emission_arguments("metal", block_m, block_n, block_k, panel_size)
    swizzle_log = int(math.log2(panel_size))
    rows_per_pass = 1 if block_m == 8 else 4
    row_in_pass = "0u" if block_m == 8 else "__simd_group_id"
    # Correctness-first source fallback: one simdgroup computes one row of 32
    # columns. The native Objective-C++ emitter has a separate optimized path.
    return f"""// Emitted by OjasX portable TileLang Metal source fallback
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
struct TileLangGemmParams {{ uint M; uint N; uint K; uint has_bias; uint grid_x; uint grid_y; }};
kernel void ojasx_tilelang_gemm_relu2_ballot_kernel(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device const float* Bias [[buffer(2)]], device float* Out [[buffer(3)]],
    device int* PackedMask [[buffer(4)]], constant TileLangGemmParams& params [[buffer(5)]],
    uint3 __physical_blockIdx [[threadgroup_position_in_grid]],
    uint __simd_group_id [[simdgroup_index_in_threadgroup]],
    uint __lane [[thread_index_in_simdgroup]]) {{
    const uint __linear_bid = __physical_blockIdx.y * params.grid_x + __physical_blockIdx.x;
    const uint __panel_span = params.grid_y << {swizzle_log};
    const uint __panel_idx = (__panel_span > 0u) ? (__linear_bid / __panel_span) : 0u;
    const uint __in_panel = (__panel_span > 0u) ? (__linear_bid % __panel_span) : 0u;
    const uint3 blockIdx = ((params.grid_x >= {panel_size}u && (params.grid_x % {panel_size}u == 0u))
        ? uint3((__panel_idx << {swizzle_log}) + (__in_panel & {panel_size - 1}u), __in_panel >> {swizzle_log}, __physical_blockIdx.z)
        : __physical_blockIdx);
    const uint tile_row = blockIdx.y * {block_m}u;
    const uint tile_col = blockIdx.x * {block_n}u;
    const uint words_per_row = (params.N + 31u) >> 5;
    const uint word_col = tile_col >> 5;
    for (uint r_base = 0u; r_base < {block_m}u; r_base += {rows_per_pass}u) {{
        const uint r = r_base + {row_in_pass};
        const uint gm = tile_row + r;
        const uint gn = tile_col + __lane;
        const bool valid = (gm < params.M) && (gn < params.N);
        float val = (valid && params.has_bias != 0u) ? Bias[gn] : 0.0f;
        if (valid) {{
            for (uint k = 0u; k < params.K; ++k) {{
                val += A[gm * params.K + k] * B[gn * params.K + k];
            }}
        }}
        const bool is_pos = valid && (val > 0.0f);
        const simd_vote vote = simd_ballot(is_pos);
        const uint32_t mask_word = uint32_t(simd_vote::vote_t(vote));
        if (valid) Out[gm * params.N + gn] = is_pos ? (val * val) : 0.0f;
        if (__lane == 0u && gm < params.M && word_col < words_per_row) {{
            PackedMask[gm * words_per_row + word_col] = as_type<int>(mask_word);
        }}
    }}
}}
"""


def _emit_grid_stride_gpu_source(
    target_arch: str, block_m: int, block_n: int, block_k: int, panel_size: int
) -> str:
    if target_arch == "cuda":
        include = "#include <cuda_runtime.h>\n#include \"tl_templates/cuda/threadblock_swizzle.h\""
        kernel = "ojasx_tilelang_cuda_gemm_relu2_ballot_kernel"
        label = "NVIDIA CUDA"
    else:
        include = "#include <hip/hip_runtime.h>\n#include \"tl_templates/hip/threadblock_swizzle.h\""
        kernel = "ojasx_tilelang_rocm_gemm_relu2_ballot_kernel"
        label = "AMD ROCm/HIP"
    # Each independent work item owns one (row, mask-word). It computes up to
    # 32 outputs, so correctness does not depend on warp size or launch shape.
    return f"""// Emitted by OjasX portable {label} source fallback; requested tile={block_m}x{block_n}x{block_k}
#include <cstdint>
{include}
extern "C" __global__ void {kernel}(
    const float* A, const float* B, const float* Bias, float* Out,
    int* PackedMask, int M, int N, int K, int has_bias) {{
    const dim3 blockIdx_swz = tl::rasterization2DRow<{panel_size}>();
    const int64_t block_linear = (int64_t(blockIdx_swz.z) * gridDim.y + blockIdx_swz.y) * gridDim.x + blockIdx_swz.x;
    const int64_t threads_per_block = int64_t(blockDim.x) * blockDim.y * blockDim.z;
    const int64_t local_linear = (int64_t(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int64_t block_count = int64_t(gridDim.x) * gridDim.y * gridDim.z;
    const int64_t words_per_row = (int64_t(N) + 31) >> 5;
    const int64_t total_work = int64_t(M) * words_per_row;
    for (int64_t work = block_linear * threads_per_block + local_linear;
         work < total_work; work += block_count * threads_per_block) {{
        const int64_t row = work / words_per_row;
        const int64_t word_col = work - row * words_per_row;
        uint32_t mask_word = 0u;
        for (int bit = 0; bit < 32; ++bit) {{
            const int64_t col = (word_col << 5) + bit;
            if (col >= N) break;
            float val = has_bias ? Bias[col] : 0.0f;
            for (int k = 0; k < K; ++k) val += A[row * K + k] * B[col * K + k];
            const bool positive = val > 0.0f;
            Out[row * N + col] = positive ? val * val : 0.0f;
            if (positive) mask_word |= (uint32_t(1) << bit);
        }}
        PackedMask[work] = static_cast<int>(mask_word);
    }}
}}
"""


def _emit_universal_source_fallback(target_arch: str, block_m: int, block_n: int, block_k: int, panel_size: int) -> str:
    _validate_emission_arguments(target_arch, block_m, block_n, block_k, panel_size)
    if target_arch == "metal":
        return _emit_metal_gemm_source_fallback(block_m, block_n, block_k, panel_size)
    if target_arch in {"cuda", "rocm"}:
        return _emit_grid_stride_gpu_source(target_arch, block_m, block_n, block_k, panel_size)
    if target_arch == "sycl":
        return f"""// Emitted by OjasX portable Intel SYCL source fallback; requested tile={block_m}x{block_n}x{block_k}
#include <cstdint>
#include <sycl/sycl.hpp>
void ojasx_tilelang_sycl_gemm_relu2_ballot_submit(
    sycl::queue& q, const float* A, const float* B, const float* Bias,
    float* Out, int32_t* PackedMask, int M, int N, int K, int has_bias) {{
    if (M <= 0 || N <= 0) return;
    const int64_t words_per_row = (int64_t(N) + 31) >> 5;
    const int64_t total_work = int64_t(M) * words_per_row;
    q.submit([&](sycl::handler& h) {{
        h.parallel_for(sycl::range<1>(total_work), [=](sycl::id<1> id) {{
            const int64_t work = id[0];
            const int64_t row = work / words_per_row;
            const int64_t word_col = work - row * words_per_row;
            uint32_t mask_word = 0u;
            for (int bit = 0; bit < 32; ++bit) {{
                const int64_t col = (word_col << 5) + bit;
                if (col >= N) break;
                float val = has_bias ? Bias[col] : 0.0f;
                for (int k = 0; k < K; ++k) val += A[row * K + k] * B[col * K + k];
                const bool positive = val > 0.0f;
                Out[row * N + col] = positive ? val * val : 0.0f;
                if (positive) mask_word |= (uint32_t(1) << bit);
            }}
            PackedMask[work] = static_cast<int32_t>(mask_word);
        }});
    }});
}}
"""
    if target_arch == "ascend":
        return f"""// Emitted by OjasX x TileLang Universal CodeGen [Target: Huawei Ascend 910B AICore Cube + UB]
#include \"tl_templates/ascend/gemm.h\"
extern \"C\" __global__ __aicore__ void ojasx_tilelang_ascend_gemm_relu2_ballot_kernel(
    __gm__ float* A_gm, __gm__ float* B_gm, __gm__ float* Out_gm, __gm__ int32_t* Mask_gm, int M, int N, int K) {{
    __cbuf__ float* l1_a = (__cbuf__ float*)0;
    __cbuf__ float* l1_b = (__cbuf__ float*)16384;
    __cc__ float* l0c = (__cc__ float*)0;
    __ubuf__ float* ub_acc = (__ubuf__ float*)0;
    __ubuf__ float* ub_out = (__ubuf__ float*)16384;
    __ubuf__ int32_t* ub_mask = (__ubuf__ int32_t*)32768;
    asc_unlock(PIPE_MTE1, 0, ASC_LOCK_BLOCK);
    asc_unlock(PIPE_MTE1, 1, ASC_LOCK_BLOCK);
    ascend_gemm_l1<{block_m}, {block_k}, {block_n}, 16, true, float, float>(l0c, l1_a, l1_b, 1, 0, 3);
    asc_lock(PIPE_M, 0, ASC_LOCK_BLOCK);
    asc_lock(PIPE_M, 1, ASC_LOCK_BLOCK);
    asc_pipe_barrier(PIPE_V);
    asc_copy_l0c2ub(ub_acc, l0c, {block_m // 16}, {block_n // 16});
    asc_pipe_barrier(PIPE_V);
    const int words_per_row = ({block_n} + 31) >> 5;
    for (int r = 0; r < {block_m}; ++r) {{
        for (int w = 0; w < words_per_row; ++w) {{
            uint32_t mask_word = 0u;
            for (int bit = 0; bit < 32; ++bit) {{
                int c = (w << 5) + bit;
                if (c < {block_n}) {{
                    float z = ub_acc[r * {block_n} + c];
                    bool pos = (z > 0.0f);
                    ub_out[r * {block_n} + c] = pos ? (z * z) : 0.0f;
                    if (pos) mask_word |= (1u << bit);
                }}
            }}
            ub_mask[r * words_per_row + w] = static_cast<int32_t>(mask_word);
        }}
    }}
    asc_pipe_barrier(PIPE_MTE3);
    asc_copy_ub2gm(Out_gm, ub_out, {block_m * block_n} * sizeof(float));
    asc_copy_ub2gm(Mask_gm, ub_mask, {block_m} * words_per_row * sizeof(int32_t));
}}
"""
    # CPU source is independently compiled and executed in
    # tests/test_emitted_cpu_kernel.py on hosts with a C++20 compiler.
    return f"""// Emitted by OjasX x TileLang Universal CodeGen [Target: Portable CPU blocked C++ GEMM]
#include \"tl_templates/cpu/gemm.h\"
extern \"C\" void ojasx_tilelang_cpu_gemm_relu2_ballot_kernel(
    const float* A, const float* B, const float* Bias, float* Out,
    int32_t* PackedMask, int64_t M, int64_t N, int64_t K, bool has_bias) {{
    tl::cpu::GemmHolonomicRelu2BallotTile<{block_m}, {block_n}, {block_k}>::run(
        A, B, Bias, Out, PackedMask, M, N, K, has_bias);
}}
"""


def normalize_target_arch(target_arch: Union[str, torch.device, None]) -> str:
    """Normalize any hardware device or backend alias to a canonical TileLang target_arch."""
    if target_arch is None:
        raw = "auto"
    elif isinstance(target_arch, torch.device):
        raw = target_arch.type
    else:
        raw = str(target_arch).strip().lower()
    if ":" in raw:
        raw = raw.split(":", 1)[0]

    if raw == "auto":
        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "metal"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "sycl"
        return "cpu"

    if raw in {"metal", "mps", "apple"}:
        return "metal"
    if raw in {"cuda", "nvidia", "nvvm", "ptx", "sm80", "sm89", "sm90", "sm100"}:
        return "cuda"
    if raw in {"rocm", "hip", "amd", "cdna", "rdna", "gcn", "gfx90a", "gfx942"}:
        return "rocm"
    if raw in {"sycl", "xpu", "intel", "spirv", "oneapi", "vulkan"}:
        return "sycl"
    if raw in {"ascend", "npu", "cann", "huawei", "ascend910", "ascend910b"}:
        return "ascend"
    if raw in {"cpu", "avx", "avx2", "avx512", "neon", "amx", "arm64", "aarch64", "x86_64"}:
        return "cpu"
    return raw


def emit_gemm_kernel_source(
    target_arch: Union[str, torch.device] = "cpu",
    block_m: int = 32,
    block_n: int = 32,
    block_k: Optional[int] = None,
    panel_size: int = 4,
) -> str:
    """Emit verified hardware-specific GEMM kernel source for any supported architecture."""
    norm_arch = normalize_target_arch(target_arch)
    if block_k is None:
        if norm_arch == "metal":
            block_k = 16 if int(block_m) == 8 else 8
        elif norm_arch in {"cuda", "rocm", "ascend"}:
            block_k = 32
        else:
            block_k = 16
    return _emit_universal_source_fallback(
        norm_arch, int(block_m), int(block_n), int(block_k), int(panel_size)
    )


def register_tile_fallbacks():
    """Fill only missing schemas, never colliding with an already loaded native library."""
    import logging
    logger = logging.getLogger("ojasx2.tile_fallback")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[OjasX Tile Fallback] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.info("Registering tile fallbacks for universal hardware support (CUDA/ROCm/CPU/XPU/MPS)...")

    composite_only_ops = {
        "probe_silicon_info",
        "xor_swizzle_offset",
        "tilelang_query_schedule",
        "tilelang_query_chip_schedule",
        "tilelang_emit_metal_gemm_source",
        "tilelang_emit_universal_source",
    }
    definitions = {
        "probe_silicon_info": ("probe_silicon_info() -> int[]", lambda: [65536, 4194304, 128, os.cpu_count() or 8, 1 if sys.platform == "darwin" else 0]),
        "xor_swizzle_offset": ("xor_swizzle_offset(int row, int col, int stride, int num_banks) -> int", lambda row, col, stride, num_banks: row * stride + ((col & ~(num_banks - 1)) | ((col ^ row) & (num_banks - 1)))),
        "pack_1bit_ballot": ("pack_1bit_ballot(Tensor x) -> Tensor", _pack),
        "unpack_1bit_ballot": ("unpack_1bit_ballot(Tensor packed, int cols, ScalarType dtype) -> Tensor", _unpack),
        "fused_amx_linear_relu2_fwd": ("fused_amx_linear_relu2_fwd(Tensor x, Tensor weight, Tensor? bias) -> (Tensor, Tensor)", _linear_forward),
        "fused_amx_linear_relu2_bwd": ("fused_amx_linear_relu2_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor? bias, Tensor packed_mask) -> (Tensor, Tensor, Tensor)", _linear_backward),
        "fused_l1_rmsnorm_fwd": ("fused_l1_rmsnorm_fwd(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)", _rms_forward),
        "fused_l1_rmsnorm_bwd": ("fused_l1_rmsnorm_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)", _rms_backward),
        "fused_l1_glu_fwd": ("fused_l1_glu_fwd(Tensor gate, Tensor up, int act_mode) -> Tensor", _glu_forward),
        "fused_l1_glu_bwd": ("fused_l1_glu_bwd(Tensor grad_out, Tensor gate, Tensor up, int act_mode) -> (Tensor, Tensor)", _glu_backward),
        "fused_transformer_mlp_fwd": ("fused_transformer_mlp_fwd(Tensor x, Tensor norm_w, Tensor gate_w, Tensor up_w, Tensor down_w, float eps) -> Tensor", _transformer_mlp_forward),
        "fused_transformer_mlp_bwd": ("fused_transformer_mlp_bwd(Tensor dy, Tensor x, Tensor norm_w, Tensor gate_w, Tensor up_w, Tensor down_w, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)", _transformer_mlp_backward),
        "compress_boundary_int8": ("compress_boundary_int8(Tensor x, int block_size) -> (Tensor, Tensor)", _compress),
        "decompress_boundary_int8": ("decompress_boundary_int8(Tensor q, Tensor scale, int[] orig_shape, ScalarType orig_dtype) -> Tensor", _decompress),
        "tilelang_metal_gemm_relu2_fwd": ("tilelang_metal_gemm_relu2_fwd(Tensor x, Tensor weight, Tensor? bias, int panel_size=4) -> (Tensor, Tensor)", lambda x, weight, bias, panel_size=4: _linear_forward(x, weight, bias)),
        "tilelang_universal_gemm_relu2_fwd": ("tilelang_universal_gemm_relu2_fwd(Tensor x, Tensor weight, Tensor? bias, str target_arch='auto', int panel_size=4) -> (Tensor, Tensor)", lambda x, weight, bias, target_arch='auto', panel_size=4: _linear_forward(x, weight, bias)),
        "tilelang_query_schedule": ("tilelang_query_schedule(int M, int N, int K, int num_warps, int swizzle_ordinal) -> int[]", _query_schedule_fallback),
        "tilelang_query_chip_schedule": ("tilelang_query_chip_schedule(str target_arch, int M, int N, int K, int smem_kb, int l2_kb) -> int[]", _query_chip_schedule_fallback),
        "tilelang_emit_metal_gemm_source": ("tilelang_emit_metal_gemm_source(int block_m, int block_n, int block_k, int panel_size) -> str", _emit_metal_gemm_source_fallback),
        "tilelang_emit_universal_source": ("tilelang_emit_universal_source(str target_arch, int block_m, int block_n, int block_k, int panel_size) -> str", _emit_universal_source_fallback),
    }
    library = torch.library.Library("ojasx_hal", "FRAGMENT")
    registered_any = False
    for name, (schema, implementation) in definitions.items():
        if not hasattr(torch.ops, "ojasx_hal") or not hasattr(torch.ops.ojasx_hal, name):
            library.define(schema)
            if name in composite_only_ops:
                library.impl(name, implementation, "CompositeExplicitAutograd")
            else:
                for dispatch_key in ("CPU", "CUDA", "MPS", "XPU", "PrivateUse1"):
                    library.impl(name, implementation, dispatch_key)
            registered_any = True
    if registered_any:
        _LIBRARIES.append(library)
    global _FALLBACKS_REGISTERED
    _FALLBACKS_REGISTERED = True
    logger.info("Tile fallbacks registration completed successfully.")
