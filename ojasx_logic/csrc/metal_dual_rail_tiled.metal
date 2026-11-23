// Phase 5: Threadgroup-Tiled Metal 4 Dual-Rail POPCNT GEMM (16x16 Shared-Memory Tiles)
// Eliminates the 1-thread-per-output global-memory bottleneck from naive E8 Metal kernels.
// Each 16x16 threadgroup cooperatively stages Dual-Rail (P, N) uint32 bitplane words into
// fast threadgroup SRAM and accumulates dot(A, B) using hardware `popcount()`.

#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

kernel void dual_rail_popcnt_gemm_tiled_16x16(
    device const uint32_t* A_pos   [[buffer(0)]], // [M, K_words]
    device const uint32_t* A_neg   [[buffer(1)]], // [M, K_words]
    device const uint32_t* B_pos   [[buffer(2)]], // [N, K_words] (row-major transposed B^T)
    device const uint32_t* B_neg   [[buffer(3)]], // [N, K_words]
    device const half*     B_scale [[buffer(4)]], // [N, K_groups] (group_size = 128 -> 4 words/group)
    device float*          C_out   [[buffer(5)]], // [M, N]
    constant uint&         M       [[buffer(6)]],
    constant uint&         N       [[buffer(7)]],
    constant uint&         K_words [[buffer(8)]],
    uint2 gid                      [[thread_position_in_grid]],
    uint2 lid                      [[thread_position_in_threadgroup]],
    uint2 tgid                     [[threadgroup_position_in_grid]]
) {
    threadgroup uint32_t As_p[TILE][TILE];
    threadgroup uint32_t As_n[TILE][TILE];
    threadgroup uint32_t Bs_p[TILE][TILE];
    threadgroup uint32_t Bs_n[TILE][TILE];

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

        // 16 words = 4 groups of 4 words (128 weights per group)
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
            const uint n_groups = (K_words + 3) >> 2;
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
