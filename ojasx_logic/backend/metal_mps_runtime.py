"""
Phase 5 Runtime Metal 4 Shader Dispatcher via `torch.mps.compile_shader`.

Compiles Metal Shading Language (MSL) directly at runtime through the macOS
Metal framework — ZERO full Xcode or `metallib` CLI dependency required!

Provides two zero-copy Apple Silicon MPS kernels:
  1. `mps_dual_rail_linear_forward`:
       Reads 2.125-bit Dual-Rail (pos_bytes, neg_bytes, scales) directly from
       DRAM in registers without ever allocating an unpacked BF16 weight matrix.
  2. `mps_dual_rail_popcnt_gemm_16x16`:
       16x16 threadgroup-tiled hardware `popcount()` Dual-Rail bitplane GEMM
       for bit-packed activations x bit-packed weights.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


_METAL_DUAL_RAIL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// 1. Zero-materialization 2.125-bit Dual-Rail Linear Forward:
//    y[m, o] = sum_k x[m, k] * (pos_bit - neg_bit) * scale[o, k / group_size]
//    dims = [M, O, I, I_pad, group_size]
kernel void dual_rail_linear_fwd_f32(
    device const float*   X         [[buffer(0)]],
    device const uint8_t* pos_bytes [[buffer(1)]],
    device const uint8_t* neg_bytes [[buffer(2)]],
    device const half*    scales    [[buffer(3)]],
    device float*         Y         [[buffer(4)]],
    device const int32_t* dims      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int M = dims[0];
    const int O = dims[1];
    const int I = dims[2];
    const int I_pad = dims[3];
    const int group_size = dims[4];

    const int col = int(gid.x); // output feature o in [0, O)
    const int row = int(gid.y); // batch*seq token m in [0, M)
    if (row >= M || col >= O) return;

    const int n_bytes = I_pad >> 3;
    const int n_groups = I_pad / group_size;
    const int bytes_per_group = group_size >> 3;

    device const float* x_row = X + row * I;
    device const uint8_t* p_row = pos_bytes + col * n_bytes;
    device const uint8_t* n_row = neg_bytes + col * n_bytes;
    device const half* s_row = scales + col * n_groups;

    float acc = 0.0f;
    for (int g = 0; g < n_groups; ++g) {
        const float sc = float(s_row[g]);
        if (sc == 0.0f) continue;
        float g_acc = 0.0f;
        const int b_start = g * bytes_per_group;
        const int k_base = g * group_size;
        for (int b = 0; b < bytes_per_group; ++b) {
            const uint8_t pb = p_row[b_start + b];
            const uint8_t nb = n_row[b_start + b];
            if ((pb | nb) == 0u) continue;
            const int k0 = k_base + (b << 3);
            #pragma unroll
            for (int bit = 0; bit < 8; ++bit) {
                const int k = k0 + bit;
                if (k < I) {
                    const float p = float((pb >> bit) & 1u);
                    const float n = float((nb >> bit) & 1u);
                    g_acc += x_row[k] * (p - n);
                }
            }
        }
        acc += g_acc * sc;
    }
    Y[row * O + col] = acc;
}

// 2. 16x16 Threadgroup-Tiled Dual-Rail Hardware `popcount()` GEMM:
//    A_pos/A_neg: [M, K_words] uint32
//    B_pos/B_neg: [N, K_words] uint32
//    B_scale:     [N, K_groups] half (group_size = 128 -> 4 words/group)
//    dims = [M, N, K_words]
constant uint TILE = 16;

kernel void dual_rail_popcnt_gemm_tiled_16x16(
    device const uint32_t* A_pos   [[buffer(0)]],
    device const uint32_t* A_neg   [[buffer(1)]],
    device const uint32_t* B_pos   [[buffer(2)]],
    device const uint32_t* B_neg   [[buffer(3)]],
    device const half*     B_scale [[buffer(4)]],
    device float*          C_out   [[buffer(5)]],
    device const int32_t*  dims    [[buffer(6)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup uint32_t As_p[TILE][TILE];
    threadgroup uint32_t As_n[TILE][TILE];
    threadgroup uint32_t Bs_p[TILE][TILE];
    threadgroup uint32_t Bs_n[TILE][TILE];

    const uint M = uint(dims[0]);
    const uint N = uint(dims[1]);
    const uint K_words = uint(dims[2]);
    const uint n_groups = (K_words + 3u) >> 2;

    const uint row = tgid.y * TILE + lid.y;
    const uint col = tgid.x * TILE + lid.x;

    float acc_f = 0.0f;
    for (uint kw_base = 0; kw_base < K_words; kw_base += TILE) {
        const uint k_col = kw_base + lid.x;
        if (row < M && k_col < K_words) {
            As_p[lid.y][lid.x] = A_pos[row * K_words + k_col];
            As_n[lid.y][lid.x] = A_neg[row * K_words + k_col];
        } else {
            As_p[lid.y][lid.x] = 0u;
            As_n[lid.y][lid.x] = 0u;
        }

        const uint b_row = tgid.x * TILE + lid.y;
        if (b_row < N && k_col < K_words) {
            Bs_p[lid.y][lid.x] = B_pos[b_row * K_words + k_col];
            Bs_n[lid.y][lid.x] = B_neg[b_row * K_words + k_col];
        } else {
            Bs_p[lid.y][lid.x] = 0u;
            Bs_n[lid.y][lid.x] = 0u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint g = 0; g < TILE; g += 4) {
            int32_t grp_acc = 0;
            for (uint w = 0; w < 4; ++w) {
                const uint k = g + w;
                const uint32_t ap = As_p[lid.y][k];
                const uint32_t an = As_n[lid.y][k];
                const uint32_t bp = Bs_p[lid.x][k];
                const uint32_t bn = Bs_n[lid.x][k];
                const uint32_t pos_hits = (ap & bp) | (an & bn);
                const uint32_t neg_hits = (ap & bn) | (an & bp);
                grp_acc += int32_t(popcount(pos_hits)) - int32_t(popcount(neg_hits));
            }
            const uint group_idx = (kw_base + g) >> 2;
            if (col < N && group_idx < n_groups) {
                acc_f += float(grp_acc) * float(B_scale[col * n_groups + group_idx]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C_out[row * N + col] = acc_f;
    }
}
"""

_COMPILED_LIB: Optional[Any] = None
_COMPILE_FAILED: bool = False
_LAST_ERROR: str = ""


def _get_mps_shader_lib() -> Optional[Any]:
    """Lazily compile Metal 4 shader via `torch.mps.compile_shader` (no Xcode/metallib needed)."""
    global _COMPILED_LIB, _COMPILE_FAILED, _LAST_ERROR
    if _COMPILED_LIB is not None:
        return _COMPILED_LIB
    if _COMPILE_FAILED:
        return None
    try:
        import torch

        if not (hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
            _COMPILE_FAILED = True
            _LAST_ERROR = "torch.mps.compile_shader unavailable on this host"
            return None
        _COMPILED_LIB = torch.mps.compile_shader(_METAL_DUAL_RAIL_SRC)
        return _COMPILED_LIB
    except Exception as exc:
        _COMPILE_FAILED = True
        _LAST_ERROR = str(exc)
        return None


def metal_runtime_status() -> Tuple[bool, str]:
    """Return (available, message) for runtime Metal 4 shader compilation."""
    lib = _get_mps_shader_lib()
    if lib is not None:
        return True, "torch.mps.compile_shader ready (zero-Xcode Metal 4 runtime)"
    return False, _LAST_ERROR or "unavailable"


def mps_dual_rail_linear_forward(
    x_2d: "torch.Tensor",
    pos_bytes: "torch.Tensor",
    neg_bytes: "torch.Tensor",
    scales: "torch.Tensor",
    in_features: int,
    group_size: int = 128,
) -> Optional["torch.Tensor"]:
    """
    Dispatch zero-materialization 2.125-bit Dual-Rail forward on Apple Silicon MPS.
    Returns None if MPS runtime shader is unavailable so caller cleanly falls back.
    """
    if x_2d.device.type != "mps":
        return None
    lib = _get_mps_shader_lib()
    if lib is None:
        return None

    import torch

    M = int(x_2d.shape[0])
    O = int(pos_bytes.shape[0])
    I_pad = int(pos_bytes.shape[1]) * 8

    x_f32 = x_2d.to(torch.float32).contiguous()
    out_f32 = torch.empty((M, O), dtype=torch.float32, device=x_2d.device)
    dims = torch.tensor([M, O, in_features, I_pad, group_size], dtype=torch.int32, device=x_2d.device)

    lib.dual_rail_linear_fwd_f32(
        x_f32,
        pos_bytes.contiguous(),
        neg_bytes.contiguous(),
        scales.contiguous(),
        out_f32,
        dims,
        threads=(O, M),
        group_size=(16, 16),
    )
    return out_f32.to(x_2d.dtype)


def mps_dual_rail_popcnt_gemm_16x16(
    a_pos: "torch.Tensor",
    a_neg: "torch.Tensor",
    b_pos: "torch.Tensor",
    b_neg: "torch.Tensor",
    b_scale: "torch.Tensor",
) -> Optional["torch.Tensor"]:
    """
    Dispatch 16x16 threadgroup-tiled hardware `popcount()` Dual-Rail GEMM on MPS.
    Inputs: a_pos/a_neg [M, K_words] uint32, b_pos/b_neg [N, K_words] uint32, b_scale [N, K_groups] fp16.
    """
    if a_pos.device.type != "mps":
        return None
    lib = _get_mps_shader_lib()
    if lib is None:
        return None

    import torch

    M, K_words = int(a_pos.shape[0]), int(a_pos.shape[1])
    N = int(b_pos.shape[0])
    out = torch.empty((M, N), dtype=torch.float32, device=a_pos.device)
    dims = torch.tensor([M, N, K_words], dtype=torch.int32, device=a_pos.device)

    # Round grid up to multiples of 16x16 threadgroups
    gx = ((N + 15) // 16) * 16
    gy = ((M + 15) // 16) * 16
    lib.dual_rail_popcnt_gemm_tiled_16x16(
        a_pos.contiguous(),
        a_neg.contiguous(),
        b_pos.contiguous(),
        b_neg.contiguous(),
        b_scale.to(torch.float16).contiguous(),
        out,
        dims,
        threads=(gx, gy),
        group_size=(16, 16),
    )
    return out


__all__ = [
    "metal_runtime_status",
    "mps_dual_rail_linear_forward",
    "mps_dual_rail_popcnt_gemm_16x16",
]
