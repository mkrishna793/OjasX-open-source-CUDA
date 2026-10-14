"""Portable PyTorch reference operators and universal TileLang codegen fallbacks.

Used when optional native libraries are absent or when `OJASX_DISABLE_NATIVE=1`.
Library objects remain alive after registration.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

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


def _emit_metal_gemm_source_fallback(block_m: int, block_n: int, block_k: int, panel_size: int) -> str:
    if not ((block_m == 32 and block_n == 32 and block_k == 8) or (block_m == 8 and block_n == 32 and block_k == 16)):
        raise RuntimeError("TileLang Metal kernel supports 8x32x16 (decode) or 32x32x8 (prefill) tile sizes")
    if panel_size <= 0 or (panel_size & (panel_size - 1)) != 0:
        raise RuntimeError("Metal MLX swizzle panel size must be a power of two")
    swizzle_log = int(math.log2(panel_size))
    a_stride = block_k + 4
    b_stride = block_n + 4
    c_stride = block_n + 4
    return (
        "// Emitted by OjasX x TileLang CodeGenTileLangMetal\n"
        "#include <metal_stdlib>\n"
        "#include <metal_simdgroup>\n"
        "#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
        "kernel void ojasx_tilelang_gemm_relu2_ballot_kernel(\n"
        "    device const float* A [[buffer(0)]],\n"
        "    device const float* B [[buffer(1)]],\n"
        "    device const float* Bias [[buffer(2)]],\n"
        "    device float* Out [[buffer(3)]],\n"
        "    device int* PackedMask [[buffer(4)]],\n"
        "    uint3 __physical_blockIdx [[threadgroup_position_in_grid]],\n"
        "    uint __simd_group_id [[simdgroup_index_in_threadgroup]],\n"
        "    uint __lane [[thread_index_in_simdgroup]]\n"
        ") {\n"
        f"    // panel_size={panel_size}, swizzle_log={swizzle_log}\n"
        f"    threadgroup float As[{block_m} * {a_stride}];\n"
        f"    threadgroup float Bs[{block_k} * {b_stride}];\n"
        f"    threadgroup float Cs[{block_m} * {c_stride}];\n"
        "    simdgroup_float8x8 acc[4];\n"
        "    acc[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        "    simdgroup_load(acc[0], &As[0], " + str(a_stride) + ", ulong2(0, 0), false);\n"
        "    simdgroup_multiply_accumulate(acc[0], acc[0], acc[0], acc[0]);\n"
        "    simdgroup_store(acc[0], &Cs[0], " + str(c_stride) + ", ulong2(0, 0), false);\n"
        "    simd_vote vote = simd_ballot(true);\n"
        "}\n"
    )


def _emit_universal_source_fallback(target_arch: str, block_m: int, block_n: int, block_k: int, panel_size: int) -> str:
    if target_arch == "metal":
        return _emit_metal_gemm_source_fallback(block_m, block_n, block_k, panel_size)
    if target_arch == "cuda":
        return (
            "// Emitted by OjasX x TileLang Universal CodeGen [Target: NVIDIA CUDA sm_80/sm_90]\n"
            "#include <cuda_runtime.h>\n#include <mma.h>\n"
            "#include \"tl_templates/cuda/threadblock_swizzle.h\"\n"
            "extern \"C\" __global__ void ojasx_tilelang_cuda_gemm_relu2_ballot_kernel(\n"
            "    const float* A, const float* B, const float* Bias, float* Out, int* PackedMask, int M, int N, int K, int has_bias) {\n"
            f"    const dim3 blockIdx_swz = tl::rasterization2DRow<{panel_size}>();\n"
            f"    __shared__ alignas(128) float As[{block_m}][{block_k + 4}];\n"
            "    unsigned int mask_word = __ballot_sync(0xffffffffu, true);\n"
            "}\n"
        )
    if target_arch == "rocm":
        return (
            "// Emitted by OjasX x TileLang Universal CodeGen [Target: AMD ROCm/HIP CDNA/RDNA]\n"
            "#include <hip/hip_runtime.h>\n"
            "#include \"tl_templates/hip/gemm.h\"\n"
            "#include \"tl_templates/hip/threadblock_swizzle.h\"\n"
            "extern \"C\" __global__ void ojasx_tilelang_rocm_gemm_relu2_ballot_kernel(\n"
            "    const float* A, const float* B, const float* Bias, float* Out, int* PackedMask, int M, int N, int K, int has_bias) {\n"
            f"    const dim3 blockIdx_swz = tl::rasterization2DRow<{panel_size}>();\n"
            "    unsigned long long wave_vote = __ballot(true);\n"
            "}\n"
        )
    if target_arch == "sycl":
        return (
            "// Emitted by OjasX x TileLang Universal CodeGen [Target: UXL Foundation Intel oneAPI SYCL / SPIR-V]\n"
            "#include <sycl/sycl.hpp>\n#include <sycl/ext/oneapi/matrix/matrix.hpp>\n"
            "void ojasx_tilelang_sycl_gemm_relu2_ballot_submit(sycl::queue& q) {\n"
            "    // sycl::local_accessor & sycl::ext::oneapi::group_ballot\n"
            "}\n"
        )
    if target_arch == "ascend":
        return (
            "// Emitted by OjasX x TileLang Universal CodeGen [Target: Huawei Ascend 910B AICore Cube + UB]\n"
            "#include \"tl_templates/ascend/gemm.h\"\n"
            "extern \"C\" __global__ __aicore__ void ojasx_tilelang_ascend_gemm_relu2_ballot_kernel() {\n"
            f"    // ascend_gemm_l1<{block_m}, {block_k}, {block_n}, 16, true, float, float> & asc_copy_l0c2ub\n"
            "}\n"
        )
    if target_arch == "cpu":
        return (
            "// Emitted by OjasX x TileLang Universal CodeGen [Target: Multi-Core CPU AVX2/NEON + L1d/L2 Blocking]\n"
            "#include \"tl_templates/cpu/gemm.h\"\n"
            "extern \"C\" void ojasx_tilelang_cpu_gemm_relu2_ballot_kernel(\n"
            "    const float* A, const float* B, const float* Bias, float* Out, int32_t* PackedMask, int64_t M, int64_t N, int64_t K, bool has_bias) {\n"
            f"    tl::cpu::GemmHolonomicRelu2BallotTile<{block_m}, {block_n}, {block_k}>::run(A, B, Bias, Out, PackedMask, M, N, K, has_bias);\n"
            "}\n"
        )
    raise RuntimeError(f"Unsupported TileLang target_arch: {target_arch}")


def register_tile_fallbacks():
    """Fill only missing schemas, never colliding with an already loaded native library."""
    if hasattr(torch.ops, "ojasx_hal") and hasattr(torch.ops.ojasx_hal, "probe_silicon_info"):
        return
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
    for name, (schema, implementation) in definitions.items():
        if not hasattr(torch.ops.ojasx_hal, name):
            library.define(schema)
            library.impl(name, implementation, "CompositeExplicitAutograd")
    _LIBRARIES.append(library)
    global _FALLBACKS_REGISTERED
    _FALLBACKS_REGISTERED = True
