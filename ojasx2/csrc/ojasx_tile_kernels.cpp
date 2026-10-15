#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <Accelerate/Accelerate.h>
#endif

#ifdef __linux__
#include <unistd.h>
#include <fstream>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

namespace ojasx_hal {

static inline float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        float ex = std::exp(x);
        return ex / (1.0f + ex);
    }
}

// ============================================================================
// 1. Universal Hardware Silicon Prober (macOS, Linux x86_64/ARM64, Windows)
// ============================================================================
std::vector<int64_t> probe_silicon_info() {
    int64_t l1d_bytes = 65536;
    int64_t l2_bytes = 4194304;
    int64_t cache_line = 64;
    int64_t physical_cpus = static_cast<int64_t>(std::max(1u, std::thread::hardware_concurrency()));
    int64_t has_apple_amx = 0;

#ifdef __APPLE__
    cache_line = 128;
    size_t size = sizeof(int64_t);
    int64_t val = 0;
    if (sysctlbyname("hw.l1dcachesize", &val, &size, nullptr, 0) == 0 && val > 0) {
        l1d_bytes = val;
    } else if (sysctlbyname("hw.perflevel0.l1dcachesize", &val, &size, nullptr, 0) == 0 && val > 0) {
        l1d_bytes = val;
    }
    size = sizeof(int64_t);
    if (sysctlbyname("hw.l2cachesize", &val, &size, nullptr, 0) == 0 && val > 0) {
        l2_bytes = val;
    } else if (sysctlbyname("hw.perflevel0.l2cachesize", &val, &size, nullptr, 0) == 0 && val > 0) {
        l2_bytes = val;
    }
    size = sizeof(int64_t);
    if (sysctlbyname("hw.cachelinesize", &val, &size, nullptr, 0) == 0 && val > 0) {
        cache_line = val;
    }
    size = sizeof(int64_t);
    if (sysctlbyname("hw.physicalcpu", &val, &size, nullptr, 0) == 0 && val > 0) {
        physical_cpus = val;
    }
    has_apple_amx = 1;
#elif defined(__linux__)
#ifdef _SC_LEVEL1_DCACHE_SIZE
    long l1_sys = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (l1_sys > 0) l1d_bytes = static_cast<int64_t>(l1_sys);
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
    long l2_sys = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2_sys > 0) l2_bytes = static_cast<int64_t>(l2_sys);
#endif
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
    long line_sys = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (line_sys > 0) cache_line = static_cast<int64_t>(line_sys);
#endif
#ifdef _SC_NPROCESSORS_ONLN
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0) physical_cpus = static_cast<int64_t>(nproc);
#endif
    // Fallback to sysfs cache hierarchy if sysconf returned 0
    if (l1d_bytes <= 0) {
        std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index0/size");
        int64_t kb = 0;
        if (f >> kb && kb > 0) l1d_bytes = kb * 1024;
        else l1d_bytes = 65536;
    }
    if (l2_bytes <= 0) {
        std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index2/size");
        int64_t kb = 0;
        if (f >> kb && kb > 0) l2_bytes = kb * 1024;
        else l2_bytes = 4194304;
    }
#endif

    return {l1d_bytes, l2_bytes, cache_line, physical_cpus, has_apple_amx};
}

// ============================================================================
// 2. TileLang-Style L1 Shared-Memory XOR Bank-Conflict-Free Address Swizzling
// ============================================================================
int64_t xor_swizzle_offset(int64_t row, int64_t col, int64_t stride, int64_t num_banks) {
    TORCH_CHECK(num_banks > 0 && (num_banks & (num_banks - 1)) == 0,
                "num_banks must be a positive power of two");
    int64_t bank_mask = num_banks - 1;
    int64_t swizzled_col = (col & ~bank_mask) | ((col ^ row) & bank_mask);
    return row * stride + swizzled_col;
}

// ============================================================================
// 3. Row-Tiled 1-Bit Hardware Ballot Packing (32 Sign States -> 1 int32 Word)
//    Supports CPU SIMD threads AND native GPU (CUDA/ROCm/MPS/XPU) tensors
// ============================================================================
at::Tensor pack_1bit_ballot(const at::Tensor& x) {
    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "pack_1bit_ballot expects at least 1D tensor");
    int64_t cols = orig_sizes.back();
    int64_t words_per_row = (cols + 31) / 32;

    auto out_sizes = orig_sizes;
    out_sizes.back() = words_per_row;

    if (x.numel() == 0 || cols == 0) {
        return at::empty(out_sizes, x.options().dtype(at::kInt));
    }

    int64_t rows = x.numel() / cols;
    at::Tensor x_contig = x.contiguous();
    at::Tensor x_2d = x_contig.reshape({rows, cols});

    if (x_2d.device().is_cpu() && x_2d.scalar_type() == at::kFloat) {
        auto packed = at::empty({rows, words_per_row}, x_2d.options().dtype(at::kInt));
        const float* x_ptr = x_2d.data_ptr<float>();
        int32_t* out_ptr = packed.data_ptr<int32_t>();

        at::parallel_for(0, rows, std::max<int64_t>(1, 16384 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                const float* row_in = x_ptr + r * cols;
                int32_t* row_out = out_ptr + r * words_per_row;
                for (int64_t w = 0; w < words_per_row; ++w) {
                    uint32_t word = 0u;
                    int64_t base = w * 32;
                    int64_t limit = std::min<int64_t>(32, cols - base);
#if defined(__AVX2__)
                    if (limit == 32) {
                        __m256 zero_v = _mm256_setzero_ps();
                        uint32_t m0 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row_in + base + 0), zero_v, _CMP_GT_OQ))) & 0xFFu;
                        uint32_t m1 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row_in + base + 8), zero_v, _CMP_GT_OQ))) & 0xFFu;
                        uint32_t m2 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row_in + base + 16), zero_v, _CMP_GT_OQ))) & 0xFFu;
                        uint32_t m3 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(row_in + base + 24), zero_v, _CMP_GT_OQ))) & 0xFFu;
                        word = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
                        row_out[w] = static_cast<int32_t>(word);
                        continue;
                    }
#endif
                    for (int64_t b = 0; b < limit; ++b) {
                        if (row_in[base + b] > 0.0f) {
                            word |= (1u << b);
                        }
                    }
                    row_out[w] = static_cast<int32_t>(word);
                }
            }
        });
        return packed.reshape(out_sizes);
    }

    // Native GPU (CUDA / ROCm / MPS / XPU) vectorized bitwise ballot packing on device
    int64_t pad = words_per_row * 32 - cols;
    at::Tensor bool_mask = (x_2d > 0);
    if (pad > 0) {
        bool_mask = at::constant_pad_nd(bool_mask, {0, pad}, false);
    }
    auto bits = bool_mask.reshape({rows, words_per_row, 32}).to(at::kInt);
    auto shifts = at::arange(0, 32, x_2d.options().dtype(at::kInt)).reshape({1, 1, 32});
    auto weights = at::bitwise_left_shift(at::ones_like(shifts), shifts);
    auto packed = (bits * weights).sum(-1, false, at::kInt);
    return packed.reshape(out_sizes);
}

at::Tensor unpack_1bit_ballot(const at::Tensor& packed, int64_t cols, at::ScalarType dtype) {
    TORCH_CHECK(packed.scalar_type() == at::kInt, "packed tensor must be int32");
    TORCH_CHECK(cols >= 0, "cols must be non-negative");
    auto out_sizes = packed.sizes().vec();
    TORCH_CHECK(!out_sizes.empty(), "unpack_1bit_ballot expects at least 1D tensor");
    int64_t words_per_row = out_sizes.back();
    out_sizes.back() = cols;

    if (packed.numel() == 0 || cols == 0 || words_per_row == 0) {
        return at::empty(out_sizes, packed.options().dtype(dtype));
    }
    TORCH_CHECK(words_per_row * 32 >= cols,
                "packed tensor last dim (", words_per_row, " words) is too small for cols=", cols);

    int64_t rows = packed.numel() / words_per_row;
    at::Tensor p_contig = packed.contiguous();
    at::Tensor p_2d = p_contig.reshape({rows, words_per_row});

    if (p_2d.device().is_cpu() && dtype == at::kFloat) {
        auto out = at::empty({rows, cols}, p_2d.options().dtype(at::kFloat));
        const int32_t* p_ptr = p_2d.data_ptr<int32_t>();
        float* out_ptr = out.data_ptr<float>();

        at::parallel_for(0, rows, std::max<int64_t>(1, 16384 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                const int32_t* row_p = p_ptr + r * words_per_row;
                float* row_out = out_ptr + r * cols;
                for (int64_t w = 0; w < words_per_row; ++w) {
                    uint32_t word = static_cast<uint32_t>(row_p[w]);
                    int64_t base = w * 32;
                    int64_t limit = std::min<int64_t>(32, cols - base);
                    for (int64_t b = 0; b < limit; ++b) {
                        row_out[base + b] = ((word >> b) & 1u) ? 1.0f : 0.0f;
                    }
                }
            }
        });
        return out.reshape(out_sizes);
    }

    auto shifts = at::arange(0, 32, p_2d.options().dtype(at::kInt)).reshape({1, 1, 32});
    auto words = p_2d.unsqueeze(-1);
    auto bits = at::bitwise_and(at::bitwise_right_shift(words, shifts), 1);
    return bits.reshape({rows, words_per_row * 32}).slice(1, 0, cols).to(dtype).reshape(out_sizes);
}

// ============================================================================
// 4. Universal Hardware-Accelerated Fused Linear + ReLU2 (CPU AMX/AVX2/NEON & GPU)
// ============================================================================
std::tuple<at::Tensor, at::Tensor> fused_amx_linear_relu2_fwd(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_amx_linear_relu2_fwd expects at least 1D input tensor");
    TORCH_CHECK(weight.dim() == 2, "fused_amx_linear_relu2_fwd expects 2D weight tensor");
    int64_t in_features = orig_sizes.back();
    int64_t out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight in_features must match x.size(-1)");
    int64_t words_per_row = (out_features + 31) / 32;

    orig_sizes.back() = out_features;
    auto mask_sizes = orig_sizes;
    mask_sizes.back() = words_per_row;

    if (x.numel() == 0 || in_features == 0 || out_features == 0) {
        return std::make_tuple(
            at::empty(orig_sizes, x.options()),
            at::empty(mask_sizes, x.options().dtype(at::kInt))
        );
    }

    int64_t rows = x.numel() / in_features;
    at::Tensor x_contig = x.contiguous();
    at::Tensor w_contig = weight.contiguous();
    at::Tensor x_2d = x_contig.reshape({rows, in_features});
    at::Tensor w_2d = w_contig.reshape({out_features, in_features});

    // All CPU FP32 platforms (macOS AMX, Linux x86_64 AVX2/AVX-512, ARM64 NEON, Windows)
    if (x_2d.device().is_cpu() && w_2d.device().is_cpu() &&
        x_2d.scalar_type() == at::kFloat && w_2d.scalar_type() == at::kFloat) {
        at::Tensor out_2d = at::empty({rows, out_features}, x_2d.options());
#ifdef __APPLE__
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(rows),
            static_cast<int>(out_features),
            static_cast<int>(in_features),
            1.0f,
            x_2d.data_ptr<float>(),
            static_cast<int>(in_features),
            w_2d.data_ptr<float>(),
            static_cast<int>(in_features),
            0.0f,
            out_2d.data_ptr<float>(),
            static_cast<int>(out_features)
        );
#else
        // Direct GEMM into preallocated out_2d buffer using OpenBLAS/MKL/oneDNN
        at::matmul_out(out_2d, x_2d, w_2d.t());
#endif

        at::Tensor packed_mask = at::empty({rows, words_per_row}, x_2d.options().dtype(at::kInt));
        float* z_ptr = out_2d.data_ptr<float>();
        at::Tensor b_contig;
        const float* b_ptr = nullptr;
        if (bias_opt.has_value() && bias_opt->defined() && bias_opt->numel() > 0) {
            TORCH_CHECK(bias_opt->numel() == out_features, "bias numel must match weight out_features");
            b_contig = bias_opt->to(at::kCPU, at::kFloat).contiguous();
            b_ptr = b_contig.data_ptr<float>();
        }
        int32_t* mask_ptr = packed_mask.data_ptr<int32_t>();

        // Fused L1-hot Epilogue: Add bias + pack 1-bit ballot mask + apply ReLU2 in-place
        at::parallel_for(0, rows, std::max<int64_t>(1, 16384 / std::max<int64_t>(1, out_features)), [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                float* row_z = z_ptr + r * out_features;
                int32_t* row_m = mask_ptr + r * words_per_row;
                for (int64_t w = 0; w < words_per_row; ++w) {
                    uint32_t word = 0u;
                    int64_t base = w * 32;
                    int64_t limit = std::min<int64_t>(32, out_features - base);
                    for (int64_t bit = 0; bit < limit; ++bit) {
                        int64_t col = base + bit;
                        float val = row_z[col] + (b_ptr ? b_ptr[col] : 0.0f);
                        if (val > 0.0f) {
                            word |= (1u << bit);
                            row_z[col] = val * val;
                        } else {
                            row_z[col] = 0.0f;
                        }
                    }
                    row_m[w] = static_cast<int32_t>(word);
                }
            }
        });

        return std::make_tuple(out_2d.reshape(orig_sizes), packed_mask.reshape(mask_sizes));
    }

    // Native GPU (CUDA / ROCm / MPS / XPU) in-place fused execution (0 extra intermediate buffers)
    at::Tensor z_nd = at::linear(x_2d, w_2d, bias_opt).reshape(orig_sizes);
    at::Tensor packed_mask = pack_1bit_ballot(z_nd);
    z_nd.clamp_min_(0.0);
    z_nd.mul_(z_nd);
    return std::make_tuple(z_nd, packed_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_amx_linear_relu2_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& packed_mask) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_amx_linear_relu2_bwd expects at least 1D input tensor");
    TORCH_CHECK(weight.dim() == 2, "fused_amx_linear_relu2_bwd expects 2D weight tensor");
    int64_t in_features = orig_sizes.back();
    int64_t out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight in_features must match x.size(-1)");

    if (x.numel() == 0 || in_features == 0 || out_features == 0) {
        at::Tensor dx = at::zeros_like(x);
        at::Tensor dw = at::zeros_like(weight);
        at::Tensor db = (bias_opt.has_value() && bias_opt->defined())
            ? at::zeros_like(*bias_opt)
            : at::zeros({out_features}, x.options());
        return std::make_tuple(dx, dw, db);
    }

    int64_t rows = x.numel() / in_features;
    int64_t words_per_row = (out_features + 31) / 32;
    at::Tensor x_contig = x.contiguous();
    at::Tensor w_contig = weight.contiguous();
    at::Tensor go_contig = grad_out.contiguous();
    at::Tensor mask_contig = packed_mask.contiguous();

    at::Tensor x_2d = x_contig.reshape({rows, in_features});
    at::Tensor w_2d = w_contig.reshape({out_features, in_features});
    at::Tensor go_2d = go_contig.reshape({rows, out_features});

    if (x_2d.device().is_cpu() && w_2d.device().is_cpu() && go_2d.device().is_cpu() &&
        mask_contig.device().is_cpu() && mask_contig.scalar_type() == at::kInt &&
        mask_contig.numel() >= rows * words_per_row &&
        x_2d.scalar_type() == at::kFloat && w_2d.scalar_type() == at::kFloat && go_2d.scalar_type() == at::kFloat) {

        at::Tensor z_2d = at::empty({rows, out_features}, x_2d.options());
#ifdef __APPLE__
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(rows),
            static_cast<int>(out_features),
            static_cast<int>(in_features),
            1.0f,
            x_2d.data_ptr<float>(),
            static_cast<int>(in_features),
            w_2d.data_ptr<float>(),
            static_cast<int>(in_features),
            0.0f,
            z_2d.data_ptr<float>(),
            static_cast<int>(out_features)
        );
#else
        at::matmul_out(z_2d, x_2d, w_2d.t());
#endif

        at::Tensor b_contig;
        const float* b_ptr = nullptr;
        bool has_bias = (bias_opt.has_value() && bias_opt->defined() && bias_opt->numel() > 0);
        if (has_bias) {
            TORCH_CHECK(bias_opt->numel() == out_features, "bias numel must match weight out_features");
            b_contig = bias_opt->to(at::kCPU, at::kFloat).contiguous();
            b_ptr = b_contig.data_ptr<float>();
        }

        // Fused unpack + ReLU2 derivative in-place in z_2d: dZ = grad_out * 2 * Z * (Z > 0)
        at::Tensor db = at::zeros({out_features}, x_2d.options());
        const float* go_ptr = go_2d.data_ptr<float>();
        float* dz_ptr = z_2d.data_ptr<float>();
        float* db_ptr = db.data_ptr<float>();
        const int32_t* mask_ptr = mask_contig.data_ptr<int32_t>();

        std::vector<float> db_acc(out_features, 0.0f);
        std::mutex db_mutex;

        at::parallel_for(0, rows, std::max<int64_t>(1, 16384 / std::max<int64_t>(1, out_features)), [&](int64_t begin, int64_t end) {
            std::vector<float> local_db(out_features, 0.0f);
            for (int64_t r = begin; r < end; ++r) {
                const float* r_go = go_ptr + r * out_features;
                float* r_dz = dz_ptr + r * out_features;
                const int32_t* r_m = mask_ptr + r * words_per_row;

                for (int64_t w = 0; w < words_per_row; ++w) {
                    uint32_t word = static_cast<uint32_t>(r_m[w]);
                    int64_t base = w * 32;
                    int64_t limit = std::min<int64_t>(32, out_features - base);
                    for (int64_t bit = 0; bit < limit; ++bit) {
                        int64_t col = base + bit;
                        float z_val = r_dz[col] + (b_ptr ? b_ptr[col] : 0.0f);
                        bool is_pos = ((word >> bit) & 1u) != 0;
                        float dz_val = is_pos ? (r_go[col] * 2.0f * z_val) : 0.0f;
                        r_dz[col] = dz_val;
                        local_db[col] += dz_val;
                    }
                }
            }
            if (has_bias) {
                std::lock_guard<std::mutex> lock(db_mutex);
                for (int64_t col = 0; col < out_features; ++col) {
                    db_acc[col] += local_db[col];
                }
            }
        });

        if (has_bias) {
            std::memcpy(db_ptr, db_acc.data(), sizeof(float) * out_features);
        }

        at::Tensor dx_2d = at::empty({rows, in_features}, x_2d.options());
        at::Tensor dw = at::empty({out_features, in_features}, w_2d.options());
#ifdef __APPLE__
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(rows), static_cast<int>(in_features), static_cast<int>(out_features),
            1.0f, dz_ptr, static_cast<int>(out_features),
            w_2d.data_ptr<float>(), static_cast<int>(in_features),
            0.0f, dx_2d.data_ptr<float>(), static_cast<int>(in_features)
        );
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            static_cast<int>(out_features), static_cast<int>(in_features), static_cast<int>(rows),
            1.0f, dz_ptr, static_cast<int>(out_features),
            x_2d.data_ptr<float>(), static_cast<int>(in_features),
            0.0f, dw.data_ptr<float>(), static_cast<int>(in_features)
        );
#else
        at::matmul_out(dx_2d, z_2d, w_2d);
        at::matmul_out(dw, z_2d.t(), x_2d);
#endif

        if (has_bias && bias_opt->scalar_type() != at::kFloat) {
            db = db.to(bias_opt->scalar_type());
        }
        return std::make_tuple(dx_2d.reshape(orig_sizes), dw, db);
    }

    // Native GPU (CUDA / ROCm / MPS / XPU) backward contraction
    at::Tensor z = at::linear(x_2d, w_2d, bias_opt);
    at::Tensor sign_mask = unpack_1bit_ballot(mask_contig, out_features, z.scalar_type()).reshape({rows, out_features});
    at::Tensor dz = go_2d * (2.0 * z * sign_mask);

    at::Tensor dx = at::matmul(dz, w_2d).reshape(orig_sizes);
    at::Tensor dw = at::matmul(dz.t(), x_2d);
    at::Tensor db = dz.sum(0);
    if (bias_opt.has_value() && bias_opt->defined()) {
        db = db.to(bias_opt->scalar_type());
    }

    return std::make_tuple(dx, dw, db);
}

// ============================================================================
// 5. L1-Cache-Resident Single-Pass Holonomic RMSNorm (vDSP / AVX2 / NEON + GPU)
// ============================================================================
std::tuple<at::Tensor, at::Tensor> fused_l1_rmsnorm_fwd(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_l1_rmsnorm_fwd expects at least 1D tensor");
    int64_t dim = orig_sizes.back();
    TORCH_CHECK(weight.numel() == dim, "weight numel must match x.size(-1)");
    auto rstd_sizes = orig_sizes;
    rstd_sizes.back() = 1;

    if (x.numel() == 0 || dim == 0) {
        return std::make_tuple(
            at::empty(orig_sizes, x.options()),
            at::empty(rstd_sizes, x.options().dtype(x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat))
        );
    }

    int64_t rows = x.numel() / dim;
    at::Tensor x_contig = x.contiguous();
    at::Tensor w_contig = weight.contiguous();

    if (x_contig.device().is_cpu() && w_contig.device().is_cpu() &&
        x_contig.scalar_type() == at::kFloat && w_contig.scalar_type() == at::kFloat) {
        at::Tensor x_2d = x_contig.reshape({rows, dim});
        at::Tensor w_1d = w_contig.reshape({dim});
        at::Tensor out = at::empty_like(x_2d);
        at::Tensor rstd = at::empty({rows, 1}, x_2d.options());

        const float* x_ptr = x_2d.data_ptr<float>();
        const float* w_ptr = w_1d.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        float* rstd_ptr = rstd.data_ptr<float>();
        float eps_f = static_cast<float>(eps);
        float inv_dim = 1.0f / static_cast<float>(dim);

        at::parallel_for(0, rows, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, dim)), [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                const float* row_in = x_ptr + r * dim;
                float* row_out = out_ptr + r * dim;
                float sum_sq = 0.0f;
#ifdef __APPLE__
                vDSP_svesq(row_in, 1, &sum_sq, static_cast<vDSP_Length>(dim));
                float inv_rms = 1.0f / std::sqrt(sum_sq * inv_dim + eps_f);
                rstd_ptr[r] = inv_rms;
                vDSP_vsmul(row_in, 1, &inv_rms, row_out, 1, static_cast<vDSP_Length>(dim));
                vDSP_vmul(row_out, 1, w_ptr, 1, row_out, 1, static_cast<vDSP_Length>(dim));
#else
                int64_t c = 0;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 vsum = _mm256_setzero_ps();
                for (; c + 8 <= dim; c += 8) {
                    __m256 vx = _mm256_loadu_ps(row_in + c);
                    vsum = _mm256_fmadd_ps(vx, vx, vsum);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, vsum);
                for (int k = 0; k < 8; ++k) sum_sq += tmp[k];
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (; c + 4 <= dim; c += 4) {
                    float32x4_t vx = vld1q_f32(row_in + c);
                    vsum = vfmaq_f32(vsum, vx, vx);
                }
                sum_sq += vaddvq_f32(vsum);
#endif
                for (; c < dim; ++c) {
                    float v = row_in[c];
                    sum_sq += v * v;
                }
                float inv_rms = 1.0f / std::sqrt(sum_sq * inv_dim + eps_f);
                rstd_ptr[r] = inv_rms;
                for (int64_t j = 0; j < dim; ++j) {
                    row_out[j] = row_in[j] * inv_rms * w_ptr[j];
                }
#endif
            }
        });
        return std::make_tuple(out.reshape(orig_sizes), rstd.reshape(rstd_sizes));
    }

    at::ScalarType compute_type = (x.scalar_type() == at::kDouble) ? at::kDouble : at::kFloat;
    at::Tensor x_comp = x_contig.to(compute_type);
    at::Tensor w_comp = w_contig.to(compute_type);
    at::Tensor rstd = at::rsqrt(x_comp.pow(2).mean(-1, true) + eps);
    at::Tensor out = ((x_comp * rstd) * w_comp).to(x.scalar_type());
    return std::make_tuple(out, rstd);
}

std::tuple<at::Tensor, at::Tensor> fused_l1_rmsnorm_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& rstd) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_l1_rmsnorm_bwd expects at least 1D tensor");
    int64_t dim = orig_sizes.back();
    TORCH_CHECK(weight.numel() == dim, "weight numel must match x.size(-1)");

    if (x.numel() == 0 || dim == 0) {
        return std::make_tuple(at::zeros_like(x), at::zeros_like(weight));
    }

    int64_t rows = x.numel() / dim;
    at::Tensor x_contig = x.contiguous();
    at::Tensor go_contig = grad_out.contiguous();
    at::Tensor w_contig = weight.contiguous();
    at::Tensor rstd_contig = rstd.contiguous();

    // Universal L1-cache-resident single-pass backward on ALL CPUs (macOS vDSP, Linux/Windows AVX2/NEON)
    if (x_contig.device().is_cpu() && go_contig.device().is_cpu() &&
        w_contig.device().is_cpu() && rstd_contig.device().is_cpu() &&
        x_contig.scalar_type() == at::kFloat &&
        go_contig.scalar_type() == at::kFloat && w_contig.scalar_type() == at::kFloat &&
        rstd_contig.scalar_type() == at::kFloat) {

        at::Tensor x_2d = x_contig.reshape({rows, dim});
        at::Tensor go_2d = go_contig.reshape({rows, dim});
        at::Tensor w_1d = w_contig.reshape({dim});
        at::Tensor rstd_1d = rstd_contig.reshape({rows});

        at::Tensor dx = at::empty({rows, dim}, x_2d.options());
        at::Tensor dw = at::zeros({dim}, w_1d.options());

        const float* x_ptr = x_2d.data_ptr<float>();
        const float* go_ptr = go_2d.data_ptr<float>();
        const float* w_ptr = w_1d.data_ptr<float>();
        const float* rstd_ptr = rstd_1d.data_ptr<float>();
        float* dx_ptr = dx.data_ptr<float>();
        float* dw_ptr = dw.data_ptr<float>();

        std::vector<float> dw_acc(dim, 0.0f);
        std::mutex dw_mutex;
        float inv_dim = 1.0f / static_cast<float>(dim);

        at::parallel_for(0, rows, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, dim)), [&](int64_t begin, int64_t end) {
            std::vector<float> local_dw(dim, 0.0f);
            std::vector<float> local_d_xhat(dim, 0.0f);

            for (int64_t r = begin; r < end; ++r) {
                const float* rx = x_ptr + r * dim;
                const float* rgo = go_ptr + r * dim;
                float* rdx = dx_ptr + r * dim;
                float rstd_val = rstd_ptr[r];

                float dot_val = 0.0f;
#ifdef __APPLE__
                vDSP_vmul(rgo, 1, w_ptr, 1, local_d_xhat.data(), 1, static_cast<vDSP_Length>(dim));
                vDSP_dotpr(local_d_xhat.data(), 1, rx, 1, &dot_val, static_cast<vDSP_Length>(dim));
#else
                for (int64_t c = 0; c < dim; ++c) {
                    float dxh = rgo[c] * w_ptr[c];
                    local_d_xhat[c] = dxh;
                    dot_val += dxh * rx[c];
                }
#endif
                float mean_dot = dot_val * rstd_val * inv_dim;

                for (int64_t c = 0; c < dim; ++c) {
                    float x_hat = rx[c] * rstd_val;
                    rdx[c] = rstd_val * (local_d_xhat[c] - x_hat * mean_dot);
                    local_dw[c] += rgo[c] * x_hat;
                }
            }

            std::lock_guard<std::mutex> lock(dw_mutex);
            for (int64_t c = 0; c < dim; ++c) {
                dw_acc[c] += local_dw[c];
            }
        });

        std::memcpy(dw_ptr, dw_acc.data(), sizeof(float) * dim);
        return std::make_tuple(dx.reshape(orig_sizes), dw);
    }

    // Native GPU (CUDA / ROCm / MPS / XPU) vectorized backward
    at::ScalarType compute_type = (x.scalar_type() == at::kDouble) ? at::kDouble : at::kFloat;
    at::Tensor x_comp = x_contig.to(compute_type);
    at::Tensor go_comp = go_contig.to(compute_type);
    at::Tensor w_comp = w_contig.to(compute_type);
    at::Tensor rstd_comp = rstd_contig.to(compute_type);

    at::Tensor x_hat = x_comp * rstd_comp;
    at::Tensor dw = (go_comp * x_hat).reshape({-1, dim}).sum(0).to(weight.scalar_type());
    at::Tensor d_xhat = go_comp * w_comp;
    at::Tensor mean_dot = (d_xhat * x_hat).mean(-1, true);
    at::Tensor dx = (rstd_comp * (d_xhat - x_hat * mean_dot)).to(x.scalar_type());

    return std::make_tuple(dx, dw);
}

// ============================================================================
// 6. Block-Scaled 8-Bit Holonomic Boundary Compressor (For Multi-Node Mesh)
// ============================================================================
std::tuple<at::Tensor, at::Tensor> compress_boundary_int8(
    const at::Tensor& x,
    int64_t block_size) {
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    at::Tensor flat = x.contiguous().to(at::kFloat).view({-1});
    int64_t numel = flat.numel();
    int64_t num_blocks = (numel + block_size - 1) / block_size;
    int64_t pad = num_blocks * block_size - numel;

    if (pad > 0) {
        flat = at::constant_pad_nd(flat, {0, pad}, 0.0f);
    }
    at::Tensor blocks = flat.view({num_blocks, block_size});
    at::Tensor max_abs = std::get<0>(blocks.abs().max(-1, true)).clamp_min(1e-30f);
    at::Tensor scale = max_abs / 127.0f;
    at::Tensor q = at::clamp(at::round(blocks / scale), -127.0f, 127.0f).to(at::kChar);
    return std::make_tuple(q, scale.squeeze(-1));
}

at::Tensor decompress_boundary_int8(
    const at::Tensor& q,
    const at::Tensor& scale,
    std::vector<int64_t> orig_shape,
    at::ScalarType orig_dtype) {
    int64_t total_numel = 1;
    for (int64_t s : orig_shape) {
        total_numel *= s;
    }
    at::Tensor recon = (q.to(at::kFloat) * scale.unsqueeze(-1)).view({-1}).slice(0, 0, total_numel);
    return recon.view(orig_shape).to(orig_dtype);
}

// ============================================================================
// 7. Universal Holonomic L1-Tiled Fused Gated Linear Unit (GLU) Engine
// ============================================================================
enum GLUActivationMode : int64_t {
    GLU_ACT_SWIGLU = 0,
    GLU_ACT_GEGLU_TANH = 1,
    GLU_ACT_GEGLU_EXACT = 2,
    GLU_ACT_REGLU = 3,
    GLU_ACT_RELU2GLU = 4,
    GLU_ACT_SIGMOIDGLU = 5,
    GLU_ACT_BILINEARGLU = 6,
    GLU_ACT_QUICKGEGLU = 7,
    GLU_ACT_MISHGLU = 8,
    GLU_ACT_TANHGLU = 9,
    GLU_ACT_LEAKY_REGLU = 10,
};

at::Tensor fused_l1_glu_fwd(
    const at::Tensor& gate,
    const at::Tensor& up,
    int64_t act_mode) {
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up tensors must have identical shapes");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up tensors must have identical dtypes");
    TORCH_CHECK(gate.device() == up.device(), "gate and up tensors must be on the same device");

    auto orig_sizes = gate.sizes().vec();
    if (gate.numel() == 0) {
        return at::empty(orig_sizes, gate.options());
    }

    if (!gate.device().is_cpu()) {
        at::ScalarType compute_type = (gate.scalar_type() == at::kDouble) ? at::kDouble : at::kFloat;
        at::Tensor g_f = gate.to(compute_type);
        at::Tensor u_f = up.to(compute_type);
        at::Tensor phi;
        switch (act_mode) {
            case GLU_ACT_SWIGLU: phi = at::silu(g_f); break;
            case GLU_ACT_GEGLU_TANH: phi = at::gelu(g_f, "tanh"); break;
            case GLU_ACT_GEGLU_EXACT: phi = at::gelu(g_f, "none"); break;
            case GLU_ACT_REGLU: phi = at::relu(g_f); break;
            case GLU_ACT_RELU2GLU: { auto r = at::relu(g_f); phi = r * r; break; }
            case GLU_ACT_SIGMOIDGLU: phi = at::sigmoid(g_f); break;
            case GLU_ACT_BILINEARGLU: phi = g_f; break;
            case GLU_ACT_QUICKGEGLU: phi = g_f * at::sigmoid(1.702 * g_f); break;
            case GLU_ACT_MISHGLU: phi = at::mish(g_f); break;
            case GLU_ACT_TANHGLU: phi = at::tanh(g_f); break;
            case GLU_ACT_LEAKY_REGLU: phi = at::leaky_relu(g_f, 0.01); break;
            default: TORCH_CHECK(false, "Unknown GLU activation mode: ", act_mode);
        }
        return (phi * u_f).to(gate.scalar_type());
    }

    at::Tensor g_contig = gate.contiguous();
    at::Tensor u_contig = up.contiguous();
    at::Tensor out = at::empty_like(g_contig);
    int64_t total = g_contig.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, g_contig.scalar_type(), "fused_l1_glu_fwd_cpu", [&] {
        const scalar_t* g_ptr = g_contig.data_ptr<scalar_t>();
        const scalar_t* u_ptr = u_contig.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();

        at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                float g = static_cast<float>(g_ptr[i]);
                float u = static_cast<float>(u_ptr[i]);
                float phi_g = 0.0f;

                switch (act_mode) {
                    case GLU_ACT_SWIGLU: {
                        float sig = stable_sigmoid(g);
                        phi_g = g * sig;
                        break;
                    }
                    case GLU_ACT_GEGLU_TANH: {
                        const float k = 0.7978845608028654f;
                        const float c = 0.044715f;
                        float v = k * (g + c * g * g * g);
                        phi_g = 0.5f * g * (1.0f + std::tanh(v));
                        break;
                    }
                    case GLU_ACT_GEGLU_EXACT: {
                        const float inv_sqrt2 = 0.7071067811865475f;
                        phi_g = 0.5f * g * (1.0f + std::erff(g * inv_sqrt2));
                        break;
                    }
                    case GLU_ACT_REGLU: {
                        if (g > 0.0f) {
                            phi_g = g;
                        } else {
                            out_ptr[i] = static_cast<scalar_t>(0.0f);
                            continue;
                        }
                        break;
                    }
                    case GLU_ACT_RELU2GLU: {
                        if (g > 0.0f) {
                            phi_g = g * g;
                        } else {
                            out_ptr[i] = static_cast<scalar_t>(0.0f);
                            continue;
                        }
                        break;
                    }
                    case GLU_ACT_SIGMOIDGLU: {
                        phi_g = stable_sigmoid(g);
                        break;
                    }
                    case GLU_ACT_BILINEARGLU: {
                        phi_g = g;
                        break;
                    }
                    case GLU_ACT_QUICKGEGLU: {
                        float s = stable_sigmoid(1.702f * g);
                        phi_g = g * s;
                        break;
                    }
                    case GLU_ACT_MISHGLU: {
                        float sp = (g > 20.0f) ? g : std::log1pf(std::exp(g));
                        phi_g = g * std::tanh(sp);
                        break;
                    }
                    case GLU_ACT_TANHGLU: {
                        phi_g = std::tanh(g);
                        break;
                    }
                    case GLU_ACT_LEAKY_REGLU: {
                        phi_g = (g > 0.0f) ? g : 0.01f * g;
                        break;
                    }
                    default:
                        TORCH_CHECK(false, "Unknown GLU activation mode: ", act_mode);
                }
                out_ptr[i] = static_cast<scalar_t>(phi_g * u);
            }
        });
    });

    return out.reshape(orig_sizes);
}

std::tuple<at::Tensor, at::Tensor> fused_l1_glu_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& gate,
    const at::Tensor& up,
    int64_t act_mode) {
    TORCH_CHECK(grad_out.sizes() == gate.sizes(), "grad_out and gate tensors must have identical shapes");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up tensors must have identical shapes");
    TORCH_CHECK(grad_out.scalar_type() == gate.scalar_type(), "grad_out and gate tensors must have identical dtypes");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up tensors must have identical dtypes");
    TORCH_CHECK(grad_out.device() == gate.device(), "grad_out and gate tensors must be on the same device");
    TORCH_CHECK(gate.device() == up.device(), "gate and up tensors must be on the same device");

    auto orig_sizes = gate.sizes().vec();
    if (grad_out.numel() == 0) {
        return std::make_tuple(at::empty(orig_sizes, gate.options()), at::empty(orig_sizes, up.options()));
    }

    if (!gate.device().is_cpu()) {
        at::ScalarType compute_type = (gate.scalar_type() == at::kDouble) ? at::kDouble : at::kFloat;
        at::Tensor go_f = grad_out.to(compute_type);
        at::Tensor g_f = gate.to(compute_type);
        at::Tensor u_f = up.to(compute_type);

        at::Tensor phi;
        at::Tensor dphi;
        switch (act_mode) {
            case GLU_ACT_SWIGLU: {
                phi = at::silu(g_f);
                at::Tensor sig = at::sigmoid(g_f);
                dphi = sig * (1.0 + g_f * (1.0 - sig));
                break;
            }
            case GLU_ACT_GEGLU_TANH: {
                phi = at::gelu(g_f, "tanh");
                dphi = at::gelu_backward(at::ones_like(g_f), g_f, "tanh");
                break;
            }
            case GLU_ACT_GEGLU_EXACT: {
                phi = at::gelu(g_f, "none");
                dphi = at::gelu_backward(at::ones_like(g_f), g_f, "none");
                break;
            }
            case GLU_ACT_REGLU: {
                phi = at::relu(g_f);
                dphi = (g_f > 0.0).to(compute_type);
                break;
            }
            case GLU_ACT_RELU2GLU: {
                at::Tensor r = at::relu(g_f);
                phi = r * r;
                dphi = at::where(g_f > 0.0, 2.0 * g_f, at::zeros_like(g_f));
                break;
            }
            case GLU_ACT_SIGMOIDGLU: {
                phi = at::sigmoid(g_f);
                dphi = phi * (1.0 - phi);
                break;
            }
            case GLU_ACT_BILINEARGLU: {
                phi = g_f;
                dphi = at::ones_like(g_f);
                break;
            }
            case GLU_ACT_QUICKGEGLU: {
                at::Tensor s = at::sigmoid(1.702 * g_f);
                phi = g_f * s;
                dphi = s * (1.0 + 1.702 * g_f * (1.0 - s));
                break;
            }
            case GLU_ACT_MISHGLU: {
                phi = at::mish(g_f);
                at::Tensor sp = at::softplus(g_f);
                at::Tensor tsp = at::tanh(sp);
                at::Tensor sig = at::sigmoid(g_f);
                dphi = tsp + g_f * sig * (1.0 - tsp.pow(2));
                break;
            }
            case GLU_ACT_TANHGLU: {
                phi = at::tanh(g_f);
                dphi = 1.0 - phi.pow(2);
                break;
            }
            case GLU_ACT_LEAKY_REGLU: {
                phi = at::leaky_relu(g_f, 0.01);
                dphi = at::where(g_f > 0.0, at::ones_like(g_f), at::full_like(g_f, 0.01));
                break;
            }
            default:
                TORCH_CHECK(false, "Unknown GLU activation mode: ", act_mode);
        }
        at::Tensor du = (go_f * phi).to(up.scalar_type());
        at::Tensor dg = (go_f * u_f * dphi).to(gate.scalar_type());
        return std::make_tuple(dg, du);
    }

    at::Tensor go_contig = grad_out.contiguous();
    at::Tensor g_contig = gate.contiguous();
    at::Tensor u_contig = up.contiguous();
    at::Tensor d_gate = at::empty_like(g_contig);
    at::Tensor d_up = at::empty_like(u_contig);
    int64_t total = g_contig.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, g_contig.scalar_type(), "fused_l1_glu_bwd_cpu", [&] {
        const scalar_t* go_ptr = go_contig.data_ptr<scalar_t>();
        const scalar_t* g_ptr = g_contig.data_ptr<scalar_t>();
        const scalar_t* u_ptr = u_contig.data_ptr<scalar_t>();
        scalar_t* dg_ptr = d_gate.data_ptr<scalar_t>();
        scalar_t* du_ptr = d_up.data_ptr<scalar_t>();

        at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                float go = static_cast<float>(go_ptr[i]);
                float g = static_cast<float>(g_ptr[i]);
                float u = static_cast<float>(u_ptr[i]);

                float phi_g = 0.0f;
                float dphi_g = 0.0f;

                switch (act_mode) {
                    case GLU_ACT_SWIGLU: {
                        float sig = stable_sigmoid(g);
                        phi_g = g * sig;
                        dphi_g = sig * (1.0f + g * (1.0f - sig));
                        break;
                    }
                    case GLU_ACT_GEGLU_TANH: {
                        const float k = 0.7978845608028654f;
                        const float c = 0.044715f;
                        float v = k * (g + c * g * g * g);
                        float t = std::tanh(v);
                        phi_g = 0.5f * g * (1.0f + t);
                        dphi_g = 0.5f * (1.0f + t) + 0.5f * g * (1.0f - t * t) * k * (1.0f + 3.0f * c * g * g);
                        break;
                    }
                    case GLU_ACT_GEGLU_EXACT: {
                        const float inv_sqrt2 = 0.7071067811865475f;
                        const float inv_sqrt2pi = 0.3989422804014327f;
                        phi_g = 0.5f * g * (1.0f + std::erff(g * inv_sqrt2));
                        dphi_g = 0.5f * (1.0f + std::erff(g * inv_sqrt2)) + g * inv_sqrt2pi * std::exp(-0.5f * g * g);
                        break;
                    }
                    case GLU_ACT_REGLU: {
                        if (g > 0.0f) {
                            phi_g = g;
                            dphi_g = 1.0f;
                        } else {
                            dg_ptr[i] = static_cast<scalar_t>(0.0f);
                            du_ptr[i] = static_cast<scalar_t>(0.0f);
                            continue;
                        }
                        break;
                    }
                    case GLU_ACT_RELU2GLU: {
                        if (g > 0.0f) {
                            phi_g = g * g;
                            dphi_g = 2.0f * g;
                        } else {
                            dg_ptr[i] = static_cast<scalar_t>(0.0f);
                            du_ptr[i] = static_cast<scalar_t>(0.0f);
                            continue;
                        }
                        break;
                    }
                    case GLU_ACT_SIGMOIDGLU: {
                        float sig = stable_sigmoid(g);
                        phi_g = sig;
                        dphi_g = sig * (1.0f - sig);
                        break;
                    }
                    case GLU_ACT_BILINEARGLU: {
                        phi_g = g;
                        dphi_g = 1.0f;
                        break;
                    }
                    case GLU_ACT_QUICKGEGLU: {
                        float s = stable_sigmoid(1.702f * g);
                        phi_g = g * s;
                        dphi_g = s * (1.0f + 1.702f * g * (1.0f - s));
                        break;
                    }
                    case GLU_ACT_MISHGLU: {
                        float sp = (g > 20.0f) ? g : std::log1pf(std::exp(g));
                        float tsp = std::tanh(sp);
                        float sig = stable_sigmoid(g);
                        phi_g = g * tsp;
                        dphi_g = tsp + g * sig * (1.0f - tsp * tsp);
                        break;
                    }
                    case GLU_ACT_TANHGLU: {
                        float t = std::tanh(g);
                        phi_g = t;
                        dphi_g = 1.0f - t * t;
                        break;
                    }
                    case GLU_ACT_LEAKY_REGLU: {
                        if (g > 0.0f) {
                            phi_g = g;
                            dphi_g = 1.0f;
                        } else {
                            phi_g = 0.01f * g;
                            dphi_g = 0.01f;
                        }
                        break;
                    }
                    default:
                        TORCH_CHECK(false, "Unknown GLU activation mode: ", act_mode);
                }

                du_ptr[i] = static_cast<scalar_t>(go * phi_g);
                dg_ptr[i] = static_cast<scalar_t>((go * u) * dphi_g);
            }
        });
    });

    return std::make_tuple(d_gate.reshape(orig_sizes), d_up.reshape(orig_sizes));
}

// ============================================================================
// 8. Unified Compiled Transformer Block Kernels (RMSNorm -> SwiGLU -> Linear)
// ============================================================================
at::Tensor fused_transformer_mlp_fwd(
    const at::Tensor& x,
    const at::Tensor& norm_w,
    const at::Tensor& gate_w,
    const at::Tensor& up_w,
    const at::Tensor& down_w,
    double eps) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_transformer_mlp_fwd expects at least 1D tensor");
    int64_t dim = norm_w.numel();
    if (x.numel() == 0 || dim == 0) {
        return at::empty_like(x);
    }

    int64_t rows = x.numel() / dim;
    at::Tensor x_contig = x.contiguous();
    at::Tensor nw_contig = norm_w.contiguous();
    at::Tensor gw_contig = gate_w.contiguous();
    at::Tensor uw_contig = up_w.contiguous();
    at::Tensor dw_contig = down_w.contiguous();

    at::ScalarType compute_type = (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)
                                ? at::kFloat : x.scalar_type();

    at::Tensor x_2d_raw = x_contig.reshape({rows, dim});
    at::Tensor x_2d = (x_2d_raw.scalar_type() == compute_type) ? x_2d_raw : x_2d_raw.to(compute_type);
    at::Tensor nw_raw = nw_contig.reshape({dim});
    at::Tensor nw = (nw_raw.scalar_type() == compute_type) ? nw_raw : nw_raw.to(compute_type);
    at::Tensor gw = (gw_contig.scalar_type() == compute_type) ? gw_contig : gw_contig.to(compute_type);
    at::Tensor uw = (uw_contig.scalar_type() == compute_type) ? uw_contig : uw_contig.to(compute_type);
    at::Tensor dw = (dw_contig.scalar_type() == compute_type) ? dw_contig : dw_contig.to(compute_type);

    at::Tensor var = x_2d.pow(2).mean(-1, true);
    at::Tensor rstd = at::rsqrt(var + eps);
    at::Tensor x_norm = (x_2d * rstd) * nw;

    at::Tensor g = at::linear(x_norm, gw);
    at::Tensor u = at::linear(x_norm, uw);
    at::Tensor h = at::silu(g) * u;
    at::Tensor y = at::linear(h, dw);

    return y.to(x.scalar_type()).reshape(orig_sizes);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_transformer_mlp_bwd(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& norm_w,
    const at::Tensor& gate_w,
    const at::Tensor& up_w,
    const at::Tensor& down_w,
    double eps) {

    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "fused_transformer_mlp_bwd expects at least 1D tensor");
    int64_t dim = norm_w.numel();
    if (x.numel() == 0 || dim == 0) {
        return std::make_tuple(
            at::zeros_like(x),
            at::zeros_like(norm_w),
            at::zeros_like(gate_w),
            at::zeros_like(up_w),
            at::zeros_like(down_w)
        );
    }

    int64_t rows = x.numel() / dim;
    at::Tensor x_contig = x.contiguous();
    at::Tensor dy_contig = dy.contiguous();
    at::Tensor nw_contig = norm_w.contiguous();
    at::Tensor gw_contig = gate_w.contiguous();
    at::Tensor uw_contig = up_w.contiguous();
    at::Tensor dw_contig = down_w.contiguous();

    at::ScalarType compute_type = (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)
                                ? at::kFloat : x.scalar_type();

    at::Tensor x_2d_raw = x_contig.reshape({rows, dim});
    at::Tensor x_2d = (x_2d_raw.scalar_type() == compute_type) ? x_2d_raw : x_2d_raw.to(compute_type);
    at::Tensor dy_2d_raw = dy_contig.reshape({rows, dim});
    at::Tensor dy_2d = (dy_2d_raw.scalar_type() == compute_type) ? dy_2d_raw : dy_2d_raw.to(compute_type);
    at::Tensor nw_raw = nw_contig.reshape({dim});
    at::Tensor nw = (nw_raw.scalar_type() == compute_type) ? nw_raw : nw_raw.to(compute_type);
    at::Tensor gw = (gw_contig.scalar_type() == compute_type) ? gw_contig : gw_contig.to(compute_type);
    at::Tensor uw = (uw_contig.scalar_type() == compute_type) ? uw_contig : uw_contig.to(compute_type);
    at::Tensor dw = (dw_contig.scalar_type() == compute_type) ? dw_contig : dw_contig.to(compute_type);

    at::Tensor var = x_2d.pow(2).mean(-1, true);
    at::Tensor rstd = at::rsqrt(var + eps);
    at::Tensor x_scaled = x_2d * rstd;
    at::Tensor x_norm = x_scaled * nw;

    at::Tensor g = at::linear(x_norm, gw);
    at::Tensor u = at::linear(x_norm, uw);
    at::Tensor sig_g = at::sigmoid(g);
    at::Tensor silu_g = g * sig_g;
    at::Tensor h = silu_g * u;

    at::Tensor dh = at::matmul(dy_2d, dw);
    at::Tensor ddown_w = at::matmul(dy_2d.t(), h);

    at::Tensor du = dh * silu_g;
    at::Tensor dsilu_g = dh * u;
    at::Tensor dg = dsilu_g * sig_g * (1.0f + g * (1.0f - sig_g));

    at::Tensor d_xnorm = at::matmul(dg, gw) + at::matmul(du, uw);
    at::Tensor dgate_w = at::matmul(dg.t(), x_norm);
    at::Tensor dup_w = at::matmul(du.t(), x_norm);

    at::Tensor dnorm_w = (d_xnorm * x_scaled).sum(0);
    at::Tensor d_xscaled = d_xnorm * nw;
    at::Tensor mean_term = (d_xscaled * x_scaled).mean(-1, true);
    at::Tensor dx = rstd * (d_xscaled - x_scaled * mean_term);

    return std::make_tuple(
        dx.to(x.scalar_type()).reshape(orig_sizes),
        dnorm_w.to(norm_w.scalar_type()),
        dgate_w.to(gate_w.scalar_type()),
        dup_w.to(up_w.scalar_type()),
        ddown_w.to(down_w.scalar_type())
    );
}

} // namespace ojasx_hal

#define OJASX_REGISTER_HAL_BACKEND(DISPATCH_KEY)                                      \
    TORCH_LIBRARY_IMPL(ojasx_hal, DISPATCH_KEY, m) {                                  \
        m.impl("pack_1bit_ballot", ojasx_hal::pack_1bit_ballot);                      \
        m.impl("unpack_1bit_ballot", ojasx_hal::unpack_1bit_ballot);                  \
        m.impl("fused_amx_linear_relu2_fwd", ojasx_hal::fused_amx_linear_relu2_fwd);  \
        m.impl("fused_amx_linear_relu2_bwd", ojasx_hal::fused_amx_linear_relu2_bwd);  \
        m.impl("fused_l1_rmsnorm_fwd", ojasx_hal::fused_l1_rmsnorm_fwd);              \
        m.impl("fused_l1_rmsnorm_bwd", ojasx_hal::fused_l1_rmsnorm_bwd);              \
        m.impl("fused_l1_glu_fwd", ojasx_hal::fused_l1_glu_fwd);                      \
        m.impl("fused_l1_glu_bwd", ojasx_hal::fused_l1_glu_bwd);                      \
        m.impl("fused_transformer_mlp_fwd", ojasx_hal::fused_transformer_mlp_fwd);    \
        m.impl("fused_transformer_mlp_bwd", ojasx_hal::fused_transformer_mlp_bwd);    \
        m.impl("compress_boundary_int8", ojasx_hal::compress_boundary_int8);          \
        m.impl("decompress_boundary_int8", ojasx_hal::decompress_boundary_int8);      \
    }

TORCH_LIBRARY(ojasx_hal, m) {
    m.def("probe_silicon_info() -> int[]");
    m.def("xor_swizzle_offset(int row, int col, int stride, int num_banks) -> int");
    m.def("pack_1bit_ballot(Tensor x) -> Tensor");
    m.def("unpack_1bit_ballot(Tensor packed, int cols, ScalarType dtype) -> Tensor");
    m.def("fused_amx_linear_relu2_fwd(Tensor x, Tensor weight, Tensor? bias) -> (Tensor, Tensor)");
    m.def("fused_amx_linear_relu2_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor? bias, Tensor packed_mask) -> (Tensor, Tensor, Tensor)");
    m.def("fused_l1_rmsnorm_fwd(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
    m.def("fused_l1_rmsnorm_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
    m.def("fused_l1_glu_fwd(Tensor gate, Tensor up, int act_mode) -> Tensor");
    m.def("fused_l1_glu_bwd(Tensor grad_out, Tensor gate, Tensor up, int act_mode) -> (Tensor, Tensor)");
    m.def("fused_transformer_mlp_fwd(Tensor x, Tensor norm_w, Tensor gate_w, Tensor up_w, Tensor down_w, float eps) -> Tensor");
    m.def("fused_transformer_mlp_bwd(Tensor dy, Tensor x, Tensor norm_w, Tensor gate_w, Tensor up_w, Tensor down_w, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("compress_boundary_int8(Tensor x, int block_size) -> (Tensor, Tensor)");
    m.def("decompress_boundary_int8(Tensor q, Tensor scale, int[] orig_shape, ScalarType orig_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(ojasx_hal, CompositeExplicitAutograd, m) {
    m.impl("probe_silicon_info", ojasx_hal::probe_silicon_info);
    m.impl("xor_swizzle_offset", ojasx_hal::xor_swizzle_offset);
}

// Register across all hardware backends: CPU, NVIDIA CUDA / AMD ROCm, Apple MPS, Intel XPU, and PrivateUse1 (openreg)
OJASX_REGISTER_HAL_BACKEND(CPU)
OJASX_REGISTER_HAL_BACKEND(CUDA)
OJASX_REGISTER_HAL_BACKEND(MPS)
OJASX_REGISTER_HAL_BACKEND(XPU)
OJASX_REGISTER_HAL_BACKEND(PrivateUse1)
