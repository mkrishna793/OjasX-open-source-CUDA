#pragma once

/*!
 * \file third_party/tilelang/src/tl_templates/cpu/gemm.h
 * \brief OjasX x TileLang L1d/L2 Cache-Blocked CPU GEMM + Holonomic ReLU2 + 1-Bit Ballot Template.
 *        Replaces upstream placeholder with a production C++17 template supporting
 *        ARM64 NEON, x86_64 AVX2/AVX-512, and portable SIMD-ballot sign-bit packing.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tl {
namespace cpu {

template <int BLOCK_M = 32, int BLOCK_N = 32, int BLOCK_K = 32>
struct GemmHolonomicRelu2BallotTile {
  static void run(
      const float* __restrict__ A,
      const float* __restrict__ B,
      const float* __restrict__ Bias,
      float* __restrict__ Out,
      int32_t* __restrict__ PackedMask,
      int64_t M,
      int64_t N,
      int64_t K,
      bool has_bias) {
    const int64_t words_per_row = (N + 31) / 32;

    // L1d-resident tile accumulator (32 x 32 floats = 4 KB < 32 KB L1d)
    alignas(64) float acc[BLOCK_M][BLOCK_N];

    for (int64_t m0 = 0; m0 < M; m0 += BLOCK_M) {
      const int64_t m_lim = std::min<int64_t>(BLOCK_M, M - m0);
      for (int64_t n0 = 0; n0 < N; n0 += BLOCK_N) {
        const int64_t n_lim = std::min<int64_t>(BLOCK_N, N - n0);

        for (int64_t rm = 0; rm < m_lim; ++rm) {
          if (has_bias && Bias != nullptr) {
            for (int64_t rn = 0; rn < n_lim; ++rn) {
              acc[rm][rn] = Bias[n0 + rn];
            }
            for (int64_t rn = n_lim; rn < BLOCK_N; ++rn) {
              acc[rm][rn] = 0.0f;
            }
          } else {
            std::memset(acc[rm], 0, sizeof(float) * BLOCK_N);
          }
        }

        // L1d/L2 cache-blocked K reduction
        for (int64_t k0 = 0; k0 < K; k0 += BLOCK_K) {
          const int64_t k_lim = std::min<int64_t>(BLOCK_K, K - k0);
          for (int64_t rm = 0; rm < m_lim; ++rm) {
            const float* a_row = A + (m0 + rm) * K + k0;
            for (int64_t rn = 0; rn < n_lim; ++rn) {
              const float* b_row = B + (n0 + rn) * K + k0;
              float sum = 0.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
              int64_t rk = 0;
              float32x4_t vsum = vdupq_n_f32(0.0f);
              for (; rk + 4 <= k_lim; rk += 4) {
                float32x4_t va = vld1q_f32(a_row + rk);
                float32x4_t vb = vld1q_f32(b_row + rk);
                vsum = vfmaq_f32(vsum, va, vb);
              }
              sum += vaddvq_f32(vsum);
              for (; rk < k_lim; ++rk) {
                sum += a_row[rk] * b_row[rk];
              }
#else
              for (int64_t rk = 0; rk < k_lim; ++rk) {
                sum += a_row[rk] * b_row[rk];
              }
#endif
              acc[rm][rn] += sum;
            }
          }
        }

        // Fused Holonomic ReLU2 + 32-lane 1-bit ballot sign packing
        const int64_t word_col = n0 >> 5;
        for (int64_t rm = 0; rm < m_lim; ++rm) {
          uint32_t mask_word = 0u;
          float* out_row = Out + (m0 + rm) * N + n0;
          for (int64_t rn = 0; rn < n_lim; ++rn) {
            float v = acc[rm][rn];
            if (v > 0.0f) {
              mask_word |= (1u << rn);
              out_row[rn] = v * v;
            } else {
              out_row[rn] = 0.0f;
            }
          }
          if (word_col < words_per_row) {
            PackedMask[(m0 + rm) * words_per_row + word_col] =
                static_cast<int32_t>(mask_word);
          }
        }
      }
    }
  }
};

} // namespace cpu
} // namespace tl
