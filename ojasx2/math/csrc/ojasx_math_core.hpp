#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <limits>
#include <numbers>

namespace ojasx::math {

// ============================================================================
// 1. 2nd-order Störmer-Verlet FSAL Integrator
// ============================================================================
/**
 * 2nd-order Störmer-Verlet FSAL (First Same As Last) Symplectic Integrator.
 * 
 * Preserves symplectic form and quadratic invariants with persistent FP32 accumulators.
 * For Hamiltonian system:
 *   q' = f(p)
 *   p' = g(q)
 *
 * FSAL Execution per step:
 *   1. Initial half step:
 *      acc1 += 0.5 * h * f_eval
 *   2. Full step:
 *      g_eval = g(acc1)
 *      acc2 += h * g_eval
 *   3. Final half step:
 *      f_eval = f(acc2)  // Cached for next step's start!
 *      acc1 += 0.5 * h * f_eval
 */
template <typename T, typename F, typename G>
requires std::is_invocable_r_v<T, F, T> && std::is_invocable_r_v<T, G, T>
void stormer_verlet_step(
    T* __restrict__ acc1,
    T* __restrict__ acc2,
    size_t n,
    T h,
    F&& f,
    G&& g,
    T* __restrict__ f_eval_cache = nullptr,
    bool is_first_step = false
) {
    const T half_h = static_cast<T>(0.5) * h;
    for (size_t i = 0; i < n; ++i) {
        T f_val = (f_eval_cache && !is_first_step) ? f_eval_cache[i] : f(acc2[i]);
        acc1[i] += half_h * f_val;
        T g_val = g(acc1[i]);
        acc2[i] += h * g_val;
        T f_next = f(acc2[i]);
        if (f_eval_cache) {
            f_eval_cache[i] = f_next;
        }
        acc1[i] += half_h * f_next;
    }
}

template <typename T, typename F, typename G>
requires std::is_invocable_v<F, const T*, T*, size_t> && std::is_invocable_v<G, const T*, T*, size_t>
void stormer_verlet_step(
    T* __restrict__ acc1,
    T* __restrict__ acc2,
    size_t n,
    T h,
    F&& f,
    G&& g,
    T* __restrict__ f_eval_cache = nullptr,
    bool is_first_step = false
) {
    const T half_h = static_cast<T>(0.5) * h;
    std::vector<T> f_buf_local;
    T* f_eval = f_eval_cache;
    if (!f_eval) {
        f_buf_local.resize(n);
        f_eval = f_buf_local.data();
    }
    if (is_first_step || !f_eval_cache) {
        f(acc2, f_eval, n);
    }
    for (size_t i = 0; i < n; ++i) {
        acc1[i] += half_h * f_eval[i];
    }
    std::vector<T> g_buf(n);
    g(acc1, g_buf.data(), n);
    for (size_t i = 0; i < n; ++i) {
        acc2[i] += h * g_buf[i];
    }
    f(acc2, f_eval, n);
    for (size_t i = 0; i < n; ++i) {
        acc1[i] += half_h * f_eval[i];
    }
}

// Persistent FP32 Accumulators for Störmer-Verlet Integration
template <typename T = float>
struct StormerVerletAccumulators {
    std::vector<T> acc1;
    std::vector<T> acc2;
    std::vector<T> f_cache;
    bool has_cached_f = false;

    void resize(size_t n) {
        acc1.resize(n, static_cast<T>(0));
        acc2.resize(n, static_cast<T>(0));
        f_cache.resize(n, static_cast<T>(0));
        has_cached_f = false;
    }

    template <typename F, typename G>
    void step(size_t n, T h, F&& f, G&& g) {
        if (acc1.size() < n) resize(n);
        stormer_verlet_step(
            acc1.data(), acc2.data(), n, h,
            std::forward<F>(f), std::forward<G>(g),
            f_cache.data(), !has_cached_f
        );
        has_cached_f = true;
    }
};

// ============================================================================
// 2. Structured Low-Rank Pullback: J = diag(d) + U * V^T
// ============================================================================
/**
 * Low-rank vector-Jacobian product for J = diag(d) + U * V^T in O(n * Rank) FLOPs.
 * 
 * J^T = diag(d) + V * U^T
 * w = J^T * v = diag(d) * v + V * (U^T * v)
 *
 * Complexity:
 *   t = U^T * v : O(n * Rank) FLOPs
 *   w = d * v + V * t : O(n * Rank) FLOPs
 *   Total: 2 * n * Rank + n FLOPs (vs O(n^2) for explicit matrix multiplication).
 */
template <typename T, size_t Rank>
struct structured_pullback {
    static void apply(
        const T* __restrict__ v,  // [n]
        const T* __restrict__ d,  // [n]
        const T* __restrict__ U,  // [n, Rank] row-major
        const T* __restrict__ V,  // [n, Rank] row-major
        T* __restrict__ out,      // [n]
        size_t n
    ) {
        std::array<T, Rank> t{};
        for (size_t i = 0; i < n; ++i) {
            const T vi = v[i];
            const size_t offset = i * Rank;
            #pragma unroll
            for (size_t k = 0; k < Rank; ++k) {
                t[k] += U[offset + k] * vi;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            T sum_lr = static_cast<T>(0);
            const size_t offset = i * Rank;
            #pragma unroll
            for (size_t k = 0; k < Rank; ++k) {
                sum_lr += V[offset + k] * t[k];
            }
            out[i] = d[i] * v[i] + sum_lr;
        }
    }
};

template <typename T>
inline void structured_pullback_dynamic(
    const T* __restrict__ v,
    const T* __restrict__ d,
    const T* __restrict__ U,
    const T* __restrict__ V,
    T* __restrict__ out,
    size_t n,
    size_t rank
) {
    switch (rank) {
        case 1:  structured_pullback<T, 1>::apply(v, d, U, V, out, n); break;
        case 2:  structured_pullback<T, 2>::apply(v, d, U, V, out, n); break;
        case 4:  structured_pullback<T, 4>::apply(v, d, U, V, out, n); break;
        case 8:  structured_pullback<T, 8>::apply(v, d, U, V, out, n); break;
        case 16: structured_pullback<T, 16>::apply(v, d, U, V, out, n); break;
        case 32: structured_pullback<T, 32>::apply(v, d, U, V, out, n); break;
        case 64: structured_pullback<T, 64>::apply(v, d, U, V, out, n); break;
        default: {
            std::vector<T> t(rank, static_cast<T>(0));
            for (size_t i = 0; i < n; ++i) {
                const T vi = v[i];
                const size_t offset = i * rank;
                for (size_t k = 0; k < rank; ++k) {
                    t[k] += U[offset + k] * vi;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                T sum_lr = static_cast<T>(0);
                const size_t offset = i * rank;
                for (size_t k = 0; k < rank; ++k) {
                    sum_lr += V[offset + k] * t[k];
                }
                out[i] = d[i] * v[i] + sum_lr;
            }
            break;
        }
    }
}

// Batched structured pullback: v is [batch_size, n], d is [n], U is [n, rank], V is [n, rank]
template <typename T>
inline void structured_pullback_batch(
    const T* __restrict__ v,
    const T* __restrict__ d,
    const T* __restrict__ U,
    const T* __restrict__ V,
    T* __restrict__ out,
    size_t batch_size,
    size_t n,
    size_t rank
) {
    for (size_t b = 0; b < batch_size; ++b) {
        structured_pullback_dynamic(
            v + b * n,
            d,
            U,
            V,
            out + b * n,
            n,
            rank
        );
    }
}

// ============================================================================
// 3. 3rd-Order Halley-Padé Rational Root Solvers (SiLU and GELU)
// ============================================================================
/**
 * Numerically stable sigmoid function.
 */
inline float stable_sigmoid(float z) noexcept {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-z));
    } else {
        const float ez = std::exp(z);
        return ez / (1.0f + ez);
    }
}

/**
 * 3rd-order Halley-Padé rational root solver for SiLU: y = z * sigmoid(z).
 * Achieves cubic convergence in 2-3 iterations with < 1e-7 relative error.
 * 
 * @param y Output activation value to invert.
 * @param branch_sign >0 for right branch (z >= z*), <=0 for left branch (z < z*).
 * @param iters Number of Halley iterations (default 3).
 * @return Inverted root z such that z * sigmoid(z) == y.
 */
inline float halley_invert_silu(float y, int branch_sign, int iters = 3) noexcept {
    constexpr float z_star = -1.2784645427610738f;
    constexpr float y_star = -0.2784645427610738f;
    constexpr float phi_pp_star = 0.217809811f;

    const float delta = std::max(y - y_star, 0.0f);
    const float sqrt_delta = std::sqrt(2.0f * delta / phi_pp_star);

    float z;
    if (branch_sign > 0) {
        if (delta < 0.02f) {
            z = z_star + sqrt_delta * (1.0f - 0.16f * sqrt_delta);
        } else if (y > 1.5f) {
            z = y + std::exp(-std::min(y, 30.0f));
        } else if (y >= 0.0f) {
            z = 2.0f * y / (1.0f + 0.6f * y) + 0.3f * std::pow(y, 1.5f);
        } else {
            z = z_star + sqrt_delta;
        }
    } else {
        if (delta < 0.02f) {
            z = z_star - sqrt_delta * (1.0f + 0.16f * sqrt_delta);
        } else if (y > -0.01f) {
            const float L1 = std::log(std::max(-y, 1e-15f));
            const float L2 = std::log(std::max(-L1, 1e-5f));
            z = L1 - L2;
        } else {
            z = z_star - sqrt_delta;
        }
    }

    // 3rd-order Halley rational iteration
    for (int iter = 0; iter < iters; ++iter) {
        const float sig = stable_sigmoid(z);
        const float phi = z * sig;
        const float f = phi - y;
        const float phi_p = sig * (1.0f + z * (1.0f - sig));
        const float phi_pp = sig * (1.0f - sig) * (2.0f + z * (1.0f - 2.0f * sig));
        float denom = 2.0f * phi_p * phi_p - f * phi_pp;
        if (std::abs(denom) < 1e-10f) {
            denom = (denom >= 0.0f ? 1e-10f : -1e-10f);
        }
        const float raw_step = (2.0f * f * phi_p) / denom;
        const float step = std::clamp(raw_step, -2.0f, 2.0f);
        z -= step;
    }
    return z;
}

/**
 * 3rd-order Halley-Padé rational root solver for GELU (tanh approximation):
 *   y = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
 * Achieves cubic convergence in 2-3 iterations with < 1e-7 relative error.
 *
 * @param y Output activation value to invert.
 * @param branch_sign >0 for right branch (z >= z*), <=0 for left branch (z < z*).
 * @param iters Number of Halley iterations (default 3).
 * @return Inverted root z such that GELU(z) == y.
 */
inline float halley_invert_gelu(float y, int branch_sign, int iters = 3) noexcept {
    constexpr float c = 0.7978845608028654f; // sqrt(2 / pi)
    constexpr float z_star = -0.751791524138f;
    constexpr float y_star = -0.1700460395f;
    constexpr float phi_pp_star = 0.43091926f;

    const float delta = std::max(y - y_star, 0.0f);
    const float sqrt_delta = std::sqrt(2.0f * delta / phi_pp_star);

    float z;
    if (branch_sign > 0) {
        if (delta < 0.02f) {
            z = z_star + sqrt_delta * (1.0f - 0.15f * sqrt_delta);
        } else if (y > 1.5f) {
            z = y;
        } else if (y >= 0.0f) {
            z = 1.3f * y / (1.0f + 0.3f * y);
        } else {
            z = z_star + sqrt_delta;
        }
    } else {
        if (delta < 0.02f) {
            z = z_star - sqrt_delta * (1.0f + 0.15f * sqrt_delta);
        } else {
            const float val = -2.0f * std::log(std::max(-y, 1e-15f)) - 1.837877f;
            z = -std::sqrt(std::max(val, 1.0f));
        }
    }

    // 3rd-order Halley rational iteration
    for (int iter = 0; iter < iters; ++iter) {
        const float z2 = z * z;
        const float z3 = z2 * z;
        const float u = c * (z + 0.044715f * z3);
        const float t = std::tanh(u);
        const float phi = 0.5f * z * (1.0f + t);
        const float f = phi - y;
        const float u_p = c * (1.0f + 3.0f * 0.044715f * z2);
        const float t_p = (1.0f - t * t) * u_p;
        const float phi_p = 0.5f * (1.0f + t) + 0.5f * z * t_p;
        const float u_pp = c * (6.0f * 0.044715f * z);
        const float t_pp = -2.0f * t * t_p * u_p + (1.0f - t * t) * u_pp;
        const float phi_pp = t_p + 0.5f * z * t_pp;

        float denom = 2.0f * phi_p * phi_p - f * phi_pp;
        if (std::abs(denom) < 1e-10f) {
            denom = (denom >= 0.0f ? 1e-10f : -1e-10f);
        }
        const float raw_step = (2.0f * f * phi_p) / denom;
        const float step = std::clamp(raw_step, -2.0f, 2.0f);
        z -= step;
    }
    return z;
}

// ============================================================================
// 4. E8 Lattice Sphere Packing & Fast Conway-Sloane Decoding
// ============================================================================
namespace lattice {

/**
 * Snaps an 8-dimensional vector to the closest point in the E8 Gosset lattice.
 * Pure algebraic rounding in O(8) steps with zero table lookups.
 * E8 = D8 U (D8 + 1/2 * 1)
 */
template <typename T>
inline void conway_sloane_e8_project(const T* in8, T* out8) {
    T f1[8], f2[8];
    int sum1 = 0, sum2 = 0;
    int worst1 = 0, worst2 = 0;
    T max_err1 = 0, max_err2 = 0;

    for (int i = 0; i < 8; ++i) {
        // D8 candidate: nearest integers with even sum
        f1[i] = std::round(in8[i]);
        sum1 += static_cast<int>(f1[i]);
        T err1 = std::abs(in8[i] - f1[i]);
        if (err1 > max_err1) { max_err1 = err1; worst1 = i; }

        // D8 + 0.5 candidate: nearest half-integers with even sum
        T shifted = in8[i] - static_cast<T>(0.5);
        f2[i] = std::round(shifted);
        sum2 += static_cast<int>(f2[i]);
        T err2 = std::abs(shifted - f2[i]);
        if (err2 > max_err2) { max_err2 = err2; worst2 = i; }
    }

    if (sum1 % 2 != 0) {
        f1[worst1] += (in8[worst1] >= f1[worst1]) ? static_cast<T>(1.0) : static_cast<T>(-1.0);
    }
    if (sum2 % 2 != 0) {
        f2[worst2] += ((in8[worst2] - static_cast<T>(0.5)) >= f2[worst2]) ? static_cast<T>(1.0) : static_cast<T>(-1.0);
    }

    T d1 = 0, d2 = 0;
    for (int i = 0; i < 8; ++i) {
        f2[i] += static_cast<T>(0.5);
        d1 += (in8[i] - f1[i]) * (in8[i] - f1[i]);
        d2 += (in8[i] - f2[i]) * (in8[i] - f2[i]);
    }

    const T* chosen = (d1 <= d2) ? f1 : f2;
    for (int i = 0; i < 8; ++i) {
        out8[i] = chosen[i];
    }
}

/**
 * On-the-fly 2-bit register unpack of 4 weights from a single uint8 byte.
 */
template <typename T>
inline void unpack_2bit_byte(uint8_t byte_val, T scale, T* out4) {
    out4[0] = (static_cast<T>(byte_val & 0x03) - static_cast<T>(1.5)) * scale;
    out4[1] = (static_cast<T>((byte_val >> 2) & 0x03) - static_cast<T>(1.5)) * scale;
    out4[2] = (static_cast<T>((byte_val >> 4) & 0x03) - static_cast<T>(1.5)) * scale;
    out4[3] = (static_cast<T>((byte_val >> 6) & 0x03) - static_cast<T>(1.5)) * scale;
}

/**
 * Fused on-the-fly register-dequantization GEMM forward:
 * Y = X @ W_dequant^T + Bias
 * Eliminates intermediate [N, K] float weight tensor materialization in DRAM.
 */
template <typename T>
inline void fused_e8_gemm_forward(
    const T* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    const T* __restrict__ Scales,
    const T* __restrict__ Bias,
    T* __restrict__ Y,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size = 32
) {
    const size_t packed_k_stride = K / 4;
    const size_t num_scale_groups = K / group_size;

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = (Bias != nullptr) ? static_cast<float>(Bias[n]) : 0.0f;
            const uint8_t* w_row = W_packed + n * packed_k_stride;
            const T* s_row = Scales + n * num_scale_groups;
            const T* x_row = X + m * K;

            for (size_t g = 0; g < num_scale_groups; ++g) {
                const float scale = static_cast<float>(s_row[g]);
                const size_t byte_start = (g * group_size) / 4;
                const size_t bytes_per_group = group_size / 4;
                float group_acc = 0.0f;

                for (size_t b = 0; b < bytes_per_group; ++b) {
                    const uint8_t byte_val = w_row[byte_start + b];
                    const size_t k_idx = g * group_size + b * 4;

                    const float c0 = static_cast<float>(byte_val & 0x03) - 1.5f;
                    const float c1 = static_cast<float>((byte_val >> 2) & 0x03) - 1.5f;
                    const float c2 = static_cast<float>((byte_val >> 4) & 0x03) - 1.5f;
                    const float c3 = static_cast<float>((byte_val >> 6) & 0x03) - 1.5f;

                    group_acc += static_cast<float>(x_row[k_idx + 0]) * c0
                               + static_cast<float>(x_row[k_idx + 1]) * c1
                               + static_cast<float>(x_row[k_idx + 2]) * c2
                               + static_cast<float>(x_row[k_idx + 3]) * c3;
                }
                acc += group_acc * scale;
            }
            Y[m * N + n] = static_cast<T>(acc);
        }
    }
}

/**
 * Fused on-the-fly register-dequantization GEMM backward dx:
 * dX = dY @ W_dequant
 * Computes input gradients without materializing uncompressed weights.
 */
template <typename T>
inline void fused_e8_gemm_backward_dx(
    const T* __restrict__ dY,
    const uint8_t* __restrict__ W_packed,
    const T* __restrict__ Scales,
    T* __restrict__ dX,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size = 32
) {
    const size_t packed_k_stride = K / 4;
    const size_t num_scale_groups = K / group_size;
    const size_t bytes_per_group = group_size / 4;

    std::vector<float> dx_acc(K, 0.0f);

    for (size_t m = 0; m < M; ++m) {
        std::fill(dx_acc.begin(), dx_acc.end(), 0.0f);
        const T* dy_row = dY + m * N;

        // Stream row-by-row over N so W_packed is read sequentially from L1 cache
        for (size_t n = 0; n < N; ++n) {
            const float dy_val = static_cast<float>(dy_row[n]);
            if (dy_val == 0.0f) continue;

            const uint8_t* w_row = W_packed + n * packed_k_stride;
            const T* s_row = Scales + n * num_scale_groups;

            for (size_t g = 0; g < num_scale_groups; ++g) {
                const float dy_scaled = dy_val * static_cast<float>(s_row[g]);
                const size_t byte_start = g * bytes_per_group;
                const size_t k_base = g * group_size;

                for (size_t b = 0; b < bytes_per_group; ++b) {
                    const uint8_t byte_val = w_row[byte_start + b];
                    const size_t k_idx = k_base + b * 4;

                    dx_acc[k_idx + 0] += dy_scaled * (static_cast<float>(byte_val & 0x03) - 1.5f);
                    dx_acc[k_idx + 1] += dy_scaled * (static_cast<float>((byte_val >> 2) & 0x03) - 1.5f);
                    dx_acc[k_idx + 2] += dy_scaled * (static_cast<float>((byte_val >> 4) & 0x03) - 1.5f);
                    dx_acc[k_idx + 3] += dy_scaled * (static_cast<float>((byte_val >> 6) & 0x03) - 1.5f);
                }
            }
        }
        for (size_t k = 0; k < K; ++k) {
            dX[m * K + k] = static_cast<T>(dx_acc[k]);
        }
    }
}

// Coordinate LUT for Ternary E8 (D8 \cap {-1, 0, +1}^8, 3281 points x 8 coords)
inline const int8_t* get_ternary_rank_to_coords() {
    static std::vector<int8_t> coords_table;
    static bool initialized = false;
    if (!initialized) {
        coords_table.reserve(3281 * 8);
        for (int i = 0; i < 6561; ++i) {
            int temp = i;
            int sum = 0;
            int8_t digits[8];
            for (int d = 0; d < 8; ++d) {
                digits[d] = static_cast<int8_t>((temp % 3) - 1);
                sum += digits[d];
                temp /= 3;
            }
            if ((sum % 2 + 2) % 2 == 0) {
                for (int d = 0; d < 8; ++d) {
                    coords_table.push_back(digits[d]);
                }
            }
        }
        initialized = true;
    }
    return coords_table.data();
}

/**
 * Fused on-the-fly register-dequantization GEMM forward for 1.50 b/w Ternary E8:
 * Y = X @ W_dequant^T + Bias
 * Eliminates intermediate [N, K] float weight tensor materialization in DRAM.
 */
template <typename T>
inline void fused_ternary_e8_gemm_forward(
    const T* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    const T* __restrict__ Scales,
    const T* __restrict__ Bias,
    T* __restrict__ Y,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size = 32
) {
    const int8_t* rank_to_coords = get_ternary_rank_to_coords();
    const size_t packed_k_stride = (K * 3) / 16;
    const size_t num_scale_groups = K / group_size;
    const size_t blocks_per_group = group_size / 16;

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = (Bias != nullptr) ? static_cast<float>(Bias[n]) : 0.0f;
            const uint8_t* w_row = W_packed + n * packed_k_stride;
            const T* s_row = Scales + n * num_scale_groups;
            const T* x_row = X + m * K;

            for (size_t g = 0; g < num_scale_groups; ++g) {
                const float scale = static_cast<float>(s_row[g]);
                const size_t byte_start = (g * group_size * 3) / 16;
                float group_acc = 0.0f;

                for (size_t blk = 0; blk < blocks_per_group; ++blk) {
                    const size_t b_idx = byte_start + blk * 3;
                    const uint8_t b0 = w_row[b_idx + 0];
                    const uint8_t b1 = w_row[b_idx + 1];
                    const uint8_t b2 = w_row[b_idx + 2];

                    const uint32_t c0 = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1 & 0x0F) << 8);
                    const uint32_t c1 = static_cast<uint32_t>((b1 >> 4) & 0x0F) | (static_cast<uint32_t>(b2) << 4);

                    const int8_t* coords0 = rank_to_coords + c0 * 8;
                    const int8_t* coords1 = rank_to_coords + c1 * 8;
                    const size_t k_base = g * group_size + blk * 16;

                    for (size_t d = 0; d < 8; ++d) {
                        group_acc += static_cast<float>(x_row[k_base + d]) * static_cast<float>(coords0[d]);
                    }
                    for (size_t d = 0; d < 8; ++d) {
                        group_acc += static_cast<float>(x_row[k_base + 8 + d]) * static_cast<float>(coords1[d]);
                    }
                }
                acc += group_acc * scale;
            }
            Y[m * N + n] = static_cast<T>(acc);
        }
    }
}

/**
 * Fused on-the-fly register-dequantization GEMM backward dx for 1.50 b/w Ternary E8:
 * dX = dY @ W_dequant
 * Computes input gradients without materializing uncompressed weights.
 */
template <typename T>
inline void fused_ternary_e8_gemm_backward_dx(
    const T* __restrict__ dY,
    const uint8_t* __restrict__ W_packed,
    const T* __restrict__ Scales,
    T* __restrict__ dX,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size = 32
) {
    const int8_t* rank_to_coords = get_ternary_rank_to_coords();
    const size_t packed_k_stride = (K * 3) / 16;
    const size_t num_scale_groups = K / group_size;
    const size_t blocks_per_group = group_size / 16;

    std::vector<float> dx_acc(K, 0.0f);

    for (size_t m = 0; m < M; ++m) {
        std::fill(dx_acc.begin(), dx_acc.end(), 0.0f);
        const T* dy_row = dY + m * N;

        // Sequential L1-friendly row traversal over N
        for (size_t n = 0; n < N; ++n) {
            const float dy_val = static_cast<float>(dy_row[n]);
            if (dy_val == 0.0f) continue;

            const uint8_t* w_row = W_packed + n * packed_k_stride;
            const T* s_row = Scales + n * num_scale_groups;

            for (size_t g = 0; g < num_scale_groups; ++g) {
                const float dy_scaled = dy_val * static_cast<float>(s_row[g]);
                const size_t byte_start = (g * group_size * 3) / 16;
                const size_t k_group_base = g * group_size;

                for (size_t blk = 0; blk < blocks_per_group; ++blk) {
                    const size_t b_idx = byte_start + blk * 3;
                    const uint8_t b0 = w_row[b_idx + 0];
                    const uint8_t b1 = w_row[b_idx + 1];
                    const uint8_t b2 = w_row[b_idx + 2];

                    const uint32_t c0 = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1 & 0x0F) << 8);
                    const uint32_t c1 = static_cast<uint32_t>((b1 >> 4) & 0x0F) | (static_cast<uint32_t>(b2) << 4);

                    const int8_t* coords0 = rank_to_coords + c0 * 8;
                    const int8_t* coords1 = rank_to_coords + c1 * 8;
                    const size_t k_base = k_group_base + blk * 16;

                    for (size_t d = 0; d < 8; ++d) {
                        dx_acc[k_base + d] += dy_scaled * static_cast<float>(coords0[d]);
                    }
                    for (size_t d = 0; d < 8; ++d) {
                        dx_acc[k_base + 8 + d] += dy_scaled * static_cast<float>(coords1[d]);
                    }
                }
            }
        }
        for (size_t k = 0; k < K; ++k) {
            dX[m * K + k] = static_cast<T>(dx_acc[k]);
        }
    }
}

} // namespace lattice

} // namespace ojasx::math

