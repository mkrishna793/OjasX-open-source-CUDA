// OjasX3 Native Apple Silicon Metal 4 Fused E8 Dequant-GEMM & FWHT Shaders
// Eliminates full-weight FP16/FP32 DRAM materialization by unpacking 2.0-bit and
// 1.5-bit Ternary E8 weights directly in registers during tile accumulation.

#include <metal_stdlib>
using namespace metal;

constant uint kTernaryLutSize = 3281u;

// -----------------------------------------------------------------------------
// 1. Fused 2.0-bit E8 Dequant-GEMM: Y[M, N] = X[M, K] @ dequant(W_packed[N, K/4])^T + Bias[N]
//    Each thread computes one output element (m, n) using register-level 2-bit unpack.
// -----------------------------------------------------------------------------
kernel void fused_e8_2bit_gemm_fwd(
    device const float* X          [[buffer(0)]],
    device const uchar* W_packed   [[buffer(1)]],
    device const half*  Scales     [[buffer(2)]],
    device const float* Bias       [[buffer(3)]],
    device float*       Y          [[buffer(4)]],
    constant uint&      M          [[buffer(5)]],
    constant uint&      K          [[buffer(6)]],
    constant uint&      N          [[buffer(7)]],
    constant uint&      group_size [[buffer(8)]],
    constant uint&      has_bias   [[buffer(9)]],
    uint2               gid        [[thread_position_in_grid]]
) {
    const uint n = gid.x;
    const uint m = gid.y;
    if (m >= M || n >= N) return;

    const uint packed_k_stride = K >> 2;
    const uint num_groups = K / group_size;
    const uint bytes_per_group = group_size >> 2;

    device const float* x_row = X + m * K;
    device const uchar* w_row = W_packed + n * packed_k_stride;
    device const half*  s_row = Scales + n * num_groups;

    float acc = (has_bias != 0u) ? Bias[n] : 0.0f;

    for (uint g = 0u; g < num_groups; ++g) {
        const float scale = float(s_row[g]);
        const uint byte_base = g * bytes_per_group;
        const uint k_base = g * group_size;
        float g_acc = 0.0f;

        for (uint b = 0u; b < bytes_per_group; ++b) {
            const uchar bv = w_row[byte_base + b];
            const uint k = k_base + (b << 2);
            const float4 x4 = float4(x_row[k], x_row[k + 1u], x_row[k + 2u], x_row[k + 3u]);
            const float4 w4 = float4(
                float(bv & 0x03u) - 1.5f,
                float((bv >> 2u) & 0x03u) - 1.5f,
                float((bv >> 4u) & 0x03u) - 1.5f,
                float((bv >> 6u) & 0x03u) - 1.5f
            );
            g_acc += dot(x4, w4);
        }
        acc = fma(g_acc, scale, acc);
    }

    Y[m * N + n] = acc;
}

// -----------------------------------------------------------------------------
// 2. Fused 1.5-bit Ternary E8 Dequant-GEMM: 3 bytes -> two 12-bit LUT codes -> 16 weights
//    Strict bounds check (code < 3281u) prevents out-of-bounds LUT reads.
// -----------------------------------------------------------------------------
kernel void fused_ternary_e8_1_5bit_gemm_fwd(
    device const float* X          [[buffer(0)]],
    device const uchar* W_packed   [[buffer(1)]],
    device const half*  Scales     [[buffer(2)]],
    device const char*  LUT        [[buffer(3)]],
    device const float* Bias       [[buffer(4)]],
    device float*       Y          [[buffer(5)]],
    constant uint&      M          [[buffer(6)]],
    constant uint&      K          [[buffer(7)]],
    constant uint&      N          [[buffer(8)]],
    constant uint&      group_size [[buffer(9)]],
    constant uint&      has_bias   [[buffer(10)]],
    uint2               gid        [[thread_position_in_grid]]
) {
    const uint n = gid.x;
    const uint m = gid.y;
    if (m >= M || n >= N) return;

    const uint packed_k_stride = (K * 3u) >> 4;
    const uint num_groups = K / group_size;
    const uint blocks_per_group = group_size >> 4;

    device const float* x_row = X + m * K;
    device const uchar* w_row = W_packed + n * packed_k_stride;
    device const half*  s_row = Scales + n * num_groups;

    float acc = (has_bias != 0u) ? Bias[n] : 0.0f;

    for (uint g = 0u; g < num_groups; ++g) {
        const float scale = float(s_row[g]);
        const uint byte_base = (g * group_size * 3u) >> 4;
        const uint k_group = g * group_size;
        float g_acc = 0.0f;

        for (uint blk = 0u; blk < blocks_per_group; ++blk) {
            const uint bi = byte_base + blk * 3u;
            const uint b0 = uint(w_row[bi + 0u]);
            const uint b1 = uint(w_row[bi + 1u]);
            const uint b2 = uint(w_row[bi + 2u]);

            const uint c0 = b0 | ((b1 & 0x0Fu) << 8u);
            const uint c1 = ((b1 >> 4u) & 0x0Fu) | (b2 << 4u);
            const uint sc0 = (c0 < kTernaryLutSize) ? c0 : 0u;
            const uint sc1 = (c1 < kTernaryLutSize) ? c1 : 0u;

            device const char* q0 = LUT + sc0 * 8u;
            device const char* q1 = LUT + sc1 * 8u;
            const uint k0 = k_group + (blk << 4);

            const float4 x0a = float4(x_row[k0 + 0u], x_row[k0 + 1u], x_row[k0 + 2u], x_row[k0 + 3u]);
            const float4 x0b = float4(x_row[k0 + 4u], x_row[k0 + 5u], x_row[k0 + 6u], x_row[k0 + 7u]);
            const float4 w0a = float4(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3]));
            const float4 w0b = float4(float(q0[4]), float(q0[5]), float(q0[6]), float(q0[7]));
            g_acc += dot(x0a, w0a) + dot(x0b, w0b);

            const float4 x1a = float4(x_row[k0 + 8u],  x_row[k0 + 9u],  x_row[k0 + 10u], x_row[k0 + 11u]);
            const float4 x1b = float4(x_row[k0 + 12u], x_row[k0 + 13u], x_row[k0 + 14u], x_row[k0 + 15u]);
            const float4 w1a = float4(float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3]));
            const float4 w1b = float4(float(q1[4]), float(q1[5]), float(q1[6]), float(q1[7]));
            g_acc += dot(x1a, w1a) + dot(x1b, w1b);
        }
        acc = fma(g_acc, scale, acc);
    }

    Y[m * N + n] = acc;
}

// -----------------------------------------------------------------------------
// 3. Fused In-Threadgroup Block-64 Fast Walsh-Hadamard Transform (FWHT)
//    6 butterfly stages, zero multiplications except final 1/8 normalization.
// -----------------------------------------------------------------------------
kernel void fwht_block64_fwd(
    device const float* X [[buffer(0)]],
    device float*       Y [[buffer(1)]],
    constant uint&      num_blocks [[buffer(2)]],
    uint                tgpig [[threadgroup_position_in_grid]],
    uint                tid   [[thread_index_in_threadgroup]]
) {
    if (tgpig >= num_blocks || tid >= 64u) return;
    threadgroup float buf[64];
    const uint base = tgpig * 64u;
    buf[tid] = X[base + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 64u; stride <<= 1u) {
        const uint partner = tid ^ stride;
        const float a = buf[tid];
        const float b = buf[partner];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[tid] = (tid < partner) ? (a + b) : (b - a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    Y[base + tid] = buf[tid] * 0.125f;  // 1 / sqrt(64) = 0.125 exact
}
