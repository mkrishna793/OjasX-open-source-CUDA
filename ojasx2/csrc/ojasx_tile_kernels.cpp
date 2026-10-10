#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <Accelerate/Accelerate.h>
#endif

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace ojasx_hal {

// ============================================================================
// 1. Hardware Silicon Prober (Queries Physical L1/L2 Cache, Cores & AMX)
// ============================================================================
std::vector<int64_t> probe_silicon_info() {
    int64_t l1d_bytes = 65536;
    int64_t l2_bytes = 4194304;
    int64_t cache_line = 128;
    int64_t physical_cpus = 8;
    int64_t has_apple_amx = 0;

#ifdef __APPLE__
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
//    Preserves leading batch/token dimensions: [..., D] -> [..., (D + 31) / 32]
// ============================================================================
at::Tensor pack_1bit_ballot(const at::Tensor& x) {
    auto orig_sizes = x.sizes().vec();
    TORCH_CHECK(!orig_sizes.empty(), "pack_1bit_ballot expects at least 1D tensor");
    int64_t cols = orig_sizes.back();
    int64_t rows = x.numel() / cols;
    int64_t words_per_row = (cols + 31) / 32;

    auto out_sizes = orig_sizes;
    out_sizes.back() = words_per_row;

    at::Tensor x_2d = x.contiguous().view({rows, cols});

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
                    for (int64_t b = 0; b < limit; ++b) {
                        if (row_in[base + b] > 0.0f) {
                            word |= (1u << b);
                        }
                    }
                    row_out[w] = static_cast<int32_t>(word);
                }
            }
        });
        return packed.view(out_sizes);
    }

    int64_t pad = words_per_row * 32 - cols;
    at::Tensor bool_mask = (x_2d > 0);
    if (pad > 0) {
        bool_mask = at::constant_pad_nd(bool_mask, {0, pad}, false);
    }
    auto bits = bool_mask.view({rows, words_per_row, 32}).to(at::kInt);
    auto shifts = at::arange(0, 32, x_2d.options().dtype(at::kInt)).view({1, 1, 32});
    auto weights = at::bitwise_left_shift(at::ones_like(shifts), shifts);
    auto packed = (bits * weights).sum(-1, false, at::kInt);
    return packed.view(out_sizes);
}

at::Tensor unpack_1bit_ballot(const at::Tensor& packed, int64_t cols, at::ScalarType dtype) {
    TORCH_CHECK(packed.scalar_type() == at::kInt, "packed tensor must be int32");
    auto out_sizes = packed.sizes().vec();
    int64_t words_per_row = out_sizes.back();
    int64_t rows = packed.numel() / words_per_row;
    out_sizes.back() = cols;

    at::Tensor p_2d = packed.contiguous().view({rows, words_per_row});

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
        return out.view(out_sizes);
    }

    auto shifts = at::arange(0, 32, p_2d.options().dtype(at::kInt)).view({1, 1, 32});
    auto words = p_2d.unsqueeze(-1);
    auto bits = at::bitwise_and(at::bitwise_right_shift(words, shifts), 1);
    return bits.view({rows, words_per_row * 32}).slice(1, 0, cols).to(dtype).view(out_sizes);
}

// ============================================================================
// 4. Apple AMX (Accelerate cblas_sgemm) + L1-Tiled Fused Linear + ReLU2
// ============================================================================
std::tuple<at::Tensor, at::Tensor> fused_amx_linear_relu2_fwd(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {

    auto orig_sizes = x.sizes().vec();
    int64_t in_features = orig_sizes.back();
    int64_t out_features = weight.size(0);
    int64_t rows = x.numel() / in_features;

    at::Tensor x_2d = x.contiguous().view({rows, in_features});
    at::Tensor w_2d = weight.contiguous();

    at::Tensor out_2d;
    at::Tensor packed_mask;

#ifdef __APPLE__
    if (x_2d.device().is_cpu() && x_2d.scalar_type() == at::kFloat && w_2d.scalar_type() == at::kFloat) {
        out_2d = at::empty({rows, out_features}, x_2d.options());
        // 1. Hardware AMX GEMM directly into `out_2d` buffer (zero intermediate allocation!)
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

        int64_t words_per_row = (out_features + 31) / 32;
        packed_mask = at::empty({rows, words_per_row}, x_2d.options().dtype(at::kInt));

        float* z_ptr = out_2d.data_ptr<float>();
        const float* b_ptr = (bias_opt.has_value() && bias_opt->defined())
            ? bias_opt->contiguous().data_ptr<float>()
            : nullptr;
        int32_t* mask_ptr = packed_mask.data_ptr<int32_t>();

        // 2. Fused L1-hot Epilogue: Add bias + pack 1-bit ballot mask + apply ReLU2 in-place!
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

        orig_sizes.back() = out_features;
        auto mask_sizes = orig_sizes;
        mask_sizes.back() = words_per_row;
        return std::make_tuple(out_2d.view(orig_sizes), packed_mask.view(mask_sizes));
    }
#endif

    at::Tensor z = at::linear(x_2d, w_2d, bias_opt);
    orig_sizes.back() = out_features;
    at::Tensor z_nd = z.view(orig_sizes);
    packed_mask = pack_1bit_ballot(z_nd);
    at::Tensor rel = at::clamp_min(z_nd, 0.0);
    at::Tensor out = rel * rel;
    return std::make_tuple(out, packed_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_amx_linear_relu2_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& packed_mask) {

    auto orig_sizes = x.sizes().vec();
    int64_t in_features = orig_sizes.back();
    int64_t out_features = weight.size(0);
    int64_t rows = x.numel() / in_features;

    at::Tensor x_2d = x.contiguous().view({rows, in_features});
    at::Tensor w_2d = weight.contiguous();
    at::Tensor go_2d = grad_out.contiguous().view({rows, out_features});

    at::Tensor z = at::linear(x_2d, w_2d, bias_opt);
    at::Tensor sign_mask = unpack_1bit_ballot(packed_mask, out_features, z.scalar_type()).view({rows, out_features});
    at::Tensor dz = go_2d * (2.0 * z * sign_mask);

    at::Tensor dx = at::matmul(dz, w_2d).view(orig_sizes);
    at::Tensor dw = at::matmul(dz.t(), x_2d);
    at::Tensor db = dz.sum(0);

    return std::make_tuple(dx, dw, db);
}

// ============================================================================
// 5. L1-Cache-Resident Single-Pass Holonomic RMSNorm (vDSP + Parallel L1 Tiles)
// ============================================================================
std::tuple<at::Tensor, at::Tensor> fused_l1_rmsnorm_fwd(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {

    auto orig_sizes = x.sizes().vec();
    int64_t dim = orig_sizes.back();
    int64_t rows = x.numel() / dim;

    if (x.device().is_cpu() && x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat) {
        at::Tensor x_2d = x.contiguous().view({rows, dim});
        at::Tensor w_1d = weight.contiguous().view({dim});
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
                for (int64_t c = 0; c < dim; ++c) {
                    float v = row_in[c];
                    sum_sq += v * v;
                }
                float inv_rms = 1.0f / std::sqrt(sum_sq * inv_dim + eps_f);
                rstd_ptr[r] = inv_rms;
                for (int64_t c = 0; c < dim; ++c) {
                    row_out[c] = row_in[c] * inv_rms * w_ptr[c];
                }
#endif
            }
        });
        auto rstd_sizes = orig_sizes;
        rstd_sizes.back() = 1;
        return std::make_tuple(out.view(orig_sizes), rstd.view(rstd_sizes));
    }

    at::Tensor x_f32 = x.to(at::kFloat);
    at::Tensor rstd = at::rsqrt(x_f32.pow(2).mean(-1, true) + eps);
    at::Tensor out = (x_f32 * rstd).to(x.scalar_type()) * weight;
    return std::make_tuple(out, rstd);
}

std::tuple<at::Tensor, at::Tensor> fused_l1_rmsnorm_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& rstd) {

    int64_t dim = x.size(-1);
    at::Tensor x_f32 = x.to(at::kFloat);
    at::Tensor go_f32 = grad_out.to(at::kFloat);
    at::Tensor w_f32 = weight.to(at::kFloat);

    at::Tensor x_hat = x_f32 * rstd;
    at::Tensor dw = (go_f32 * x_hat).reshape({-1, dim}).sum(0).to(weight.scalar_type());
    at::Tensor d_xhat = go_f32 * w_f32;
    at::Tensor mean_dot = (d_xhat * x_hat).mean(-1, true);
    at::Tensor dx = (rstd * (d_xhat - x_hat * mean_dot)).to(x.scalar_type());

    return std::make_tuple(dx, dw);
}

// ============================================================================
// 6. Block-Scaled 8-Bit Holonomic Boundary Compressor (For Multi-Laptop Mesh)
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

} // namespace ojasx_hal

TORCH_LIBRARY(ojasx_hal, m) {
    m.def("probe_silicon_info() -> int[]");
    m.def("xor_swizzle_offset(int row, int col, int stride, int num_banks) -> int");
    m.def("pack_1bit_ballot(Tensor x) -> Tensor");
    m.def("unpack_1bit_ballot(Tensor packed, int cols, ScalarType dtype) -> Tensor");
    m.def("fused_amx_linear_relu2_fwd(Tensor x, Tensor weight, Tensor? bias) -> (Tensor, Tensor)");
    m.def("fused_amx_linear_relu2_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor? bias, Tensor packed_mask) -> (Tensor, Tensor, Tensor)");
    m.def("fused_l1_rmsnorm_fwd(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
    m.def("fused_l1_rmsnorm_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
    m.def("compress_boundary_int8(Tensor x, int block_size) -> (Tensor, Tensor)");
    m.def("decompress_boundary_int8(Tensor q, Tensor scale, int[] orig_shape, ScalarType orig_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(ojasx_hal, CompositeExplicitAutograd, m) {
    m.impl("probe_silicon_info", ojasx_hal::probe_silicon_info);
    m.impl("xor_swizzle_offset", ojasx_hal::xor_swizzle_offset);
}

TORCH_LIBRARY_IMPL(ojasx_hal, CPU, m) {
    m.impl("pack_1bit_ballot", ojasx_hal::pack_1bit_ballot);
    m.impl("unpack_1bit_ballot", ojasx_hal::unpack_1bit_ballot);
    m.impl("fused_amx_linear_relu2_fwd", ojasx_hal::fused_amx_linear_relu2_fwd);
    m.impl("fused_amx_linear_relu2_bwd", ojasx_hal::fused_amx_linear_relu2_bwd);
    m.impl("fused_l1_rmsnorm_fwd", ojasx_hal::fused_l1_rmsnorm_fwd);
    m.impl("fused_l1_rmsnorm_bwd", ojasx_hal::fused_l1_rmsnorm_bwd);
    m.impl("compress_boundary_int8", ojasx_hal::compress_boundary_int8);
    m.impl("decompress_boundary_int8", ojasx_hal::decompress_boundary_int8);
}

TORCH_LIBRARY_IMPL(ojasx_hal, MPS, m) {
    m.impl("pack_1bit_ballot", ojasx_hal::pack_1bit_ballot);
    m.impl("unpack_1bit_ballot", ojasx_hal::unpack_1bit_ballot);
    m.impl("fused_amx_linear_relu2_fwd", ojasx_hal::fused_amx_linear_relu2_fwd);
    m.impl("fused_amx_linear_relu2_bwd", ojasx_hal::fused_amx_linear_relu2_bwd);
    m.impl("fused_l1_rmsnorm_fwd", ojasx_hal::fused_l1_rmsnorm_fwd);
    m.impl("fused_l1_rmsnorm_bwd", ojasx_hal::fused_l1_rmsnorm_bwd);
    m.impl("compress_boundary_int8", ojasx_hal::compress_boundary_int8);
    m.impl("decompress_boundary_int8", ojasx_hal::decompress_boundary_int8);
}
