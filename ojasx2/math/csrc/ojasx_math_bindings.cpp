#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "ojasx_math_core.hpp"
#include "ojasx_math_metal.h"

#include <vector>
#include <string>
#include <tuple>
#include <iostream>

namespace ojasx_math {

// ============================================================================
// 1. Störmer-Verlet FSAL Symplectic Integrator Tensor Forward
// ============================================================================
/**
 * 2nd-order Störmer-Verlet FSAL Integrator forward pass using persistent FP32 accumulators.
 * 
 * Takes position state q1, momentum state q2, generic Python functors f and g,
 * and step size h. Performs symplectic time evolution.
 */
std::tuple<at::Tensor, at::Tensor> stormer_verlet_fwd(
    at::Tensor q1,
    at::Tensor q2,
    py::object f_func,
    py::object g_func,
    double h
) {
    TORCH_CHECK(q1.sizes() == q2.sizes(), "q1 and q2 must have identical shapes");
    const auto orig_dtype = q1.scalar_type();

    // Persistent FP32 accumulators to prevent numerical drift
    at::Tensor acc1 = q1.to(at::kFloat).contiguous();
    at::Tensor acc2 = q2.to(at::kFloat).contiguous();

    const float h_val = static_cast<float>(h);
    const float half_h = 0.5f * h_val;

    // FSAL: 1. Evaluate f(acc2)
    at::Tensor f1_res = f_func(acc2.to(orig_dtype)).cast<at::Tensor>().to(at::kFloat);
    acc1.add_(f1_res, half_h);

    // 2. Evaluate g(acc1)
    at::Tensor g_res = g_func(acc1.to(orig_dtype)).cast<at::Tensor>().to(at::kFloat);
    acc2.add_(g_res, h_val);

    // 3. Evaluate f(acc2) and complete second half-step
    at::Tensor f2_res = f_func(acc2.to(orig_dtype)).cast<at::Tensor>().to(at::kFloat);
    acc1.add_(f2_res, half_h);

    return std::make_tuple(acc1.to(orig_dtype), acc2.to(orig_dtype));
}

/**
 * Tensor-only Störmer-Verlet FSAL step with precomputed delta updates.
 */
std::tuple<at::Tensor, at::Tensor> stormer_verlet_step_tensors(
    at::Tensor q1,
    at::Tensor q2,
    at::Tensor delta_f1,
    at::Tensor delta_g,
    at::Tensor delta_f2,
    double h
) {
    TORCH_CHECK(q1.sizes() == q2.sizes(), "q1 and q2 must have identical shapes");
    const auto orig_dtype = q1.scalar_type();

    at::Tensor acc1 = q1.to(at::kFloat).contiguous();
    at::Tensor acc2 = q2.to(at::kFloat).contiguous();

    const float h_val = static_cast<float>(h);
    const float half_h = 0.5f * h_val;

    acc1.add_(delta_f1.to(at::kFloat), half_h);
    acc2.add_(delta_g.to(at::kFloat), h_val);
    acc1.add_(delta_f2.to(at::kFloat), half_h);

    return std::make_tuple(acc1.to(orig_dtype), acc2.to(orig_dtype));
}

// ============================================================================
// 2. Structured Low-Rank Pullback Tensor Operator: J = diag(d) + U * V^T
// ============================================================================
/**
 * Low-rank vector-Jacobian product for J = diag(d) + U * V^T in O(n * Rank) FLOPs.
 * 
 * v: [..., n] (or [batch_size, n])
 * d: [n] or matching v shape
 * U: [n, rank]
 * V: [n, rank]
 */
at::Tensor structured_pullback_tensor(
    at::Tensor v,
    at::Tensor d,
    at::Tensor U,
    at::Tensor V,
    bool prefer_metal = true
) {
    TORCH_CHECK(U.dim() == 2 && V.dim() == 2, "U and V must be 2D matrices [n, rank]");
    TORCH_CHECK(U.sizes() == V.sizes(), "U and V must have identical shapes");

    const int64_t n = U.size(0);
    const int64_t rank = U.size(1);
    TORCH_CHECK(v.size(-1) == n, "Last dimension of v must match n");

    const auto orig_dtype = v.scalar_type();
    at::Tensor v_fp32 = v.to(at::kFloat).contiguous();
    at::Tensor d_fp32 = d.to(at::kFloat).contiguous();
    at::Tensor U_fp32 = U.to(at::kFloat).contiguous();
    at::Tensor V_fp32 = V.to(at::kFloat).contiguous();

    const int64_t total_elements = v_fp32.numel();
    const int64_t batch_size = total_elements / n;

    at::Tensor out_fp32 = at::empty_like(v_fp32);

    const float* v_ptr = v_fp32.data_ptr<float>();
    const float* d_ptr = d_fp32.data_ptr<float>();
    const float* U_ptr = U_fp32.data_ptr<float>();
    const float* V_ptr = V_fp32.data_ptr<float>();
    float* out_ptr = out_fp32.data_ptr<float>();

    bool executed_metal = false;
    if (prefer_metal && ojasx::math::metal::is_metal_available() && batch_size == 1) {
        executed_metal = ojasx::math::metal::metal_structured_pullback(
            v_ptr, d_ptr, U_ptr, V_ptr, out_ptr, static_cast<size_t>(n), static_cast<size_t>(rank)
        );
    }

    if (!executed_metal) {
        // Parallel C++20 CPU execution across batches
        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
                const float* cur_d = (d_fp32.numel() == n) ? d_ptr : (d_ptr + b * n);
                ojasx::math::structured_pullback_dynamic<float>(
                    v_ptr + b * n,
                    cur_d,
                    U_ptr,
                    V_ptr,
                    out_ptr + b * n,
                    static_cast<size_t>(n),
                    static_cast<size_t>(rank)
                );
            }
        });
    }

    return out_fp32.to(orig_dtype);
}

// ============================================================================
// 3. Halley-Padé Branched Root Solver Tensor Operator
// ============================================================================
/**
 * Inverts y = phi(z) -> z to machine precision using 3rd-order Halley iterations.
 * Supports: 'silu' / 'swish', 'gelu' / 'quickgelu'.
 */
at::Tensor halley_invert_activation_tensor(
    at::Tensor y,
    at::Tensor branch_bit,
    std::string act = "silu",
    int64_t iters = 3,
    bool prefer_metal = true
) {
    TORCH_CHECK(y.sizes() == branch_bit.sizes(), "y and branch_bit must have identical shapes");
    const auto orig_dtype = y.scalar_type();

    at::Tensor y_fp32 = y.to(at::kFloat).contiguous();
    at::Tensor b_u8 = branch_bit.to(at::kByte).contiguous();
    at::Tensor z_fp32 = at::empty_like(y_fp32);

    const int64_t numel = y_fp32.numel();
    const float* y_ptr = y_fp32.data_ptr<float>();
    const uint8_t* b_ptr = b_u8.data_ptr<uint8_t>();
    float* z_ptr = z_fp32.data_ptr<float>();

    std::string act_lower = act;
    for (char& c : act_lower) c = std::tolower(c);

    bool is_silu = (act_lower == "silu" || act_lower == "swish");
    bool is_gelu = (act_lower == "gelu" || act_lower == "quickgelu" || act_lower == "geglu");

    bool executed_metal = false;
    if (prefer_metal && ojasx::math::metal::is_metal_available() && numel > 0) {
        if (is_silu) {
            executed_metal = ojasx::math::metal::metal_halley_invert_silu(
                y_ptr, b_ptr, z_ptr, static_cast<size_t>(numel), static_cast<int>(iters)
            );
        } else if (is_gelu) {
            executed_metal = ojasx::math::metal::metal_halley_invert_gelu(
                y_ptr, b_ptr, z_ptr, static_cast<size_t>(numel), static_cast<int>(iters)
            );
        }
    }

    if (!executed_metal) {
        // Multi-threaded C++20 CPU execution
        at::parallel_for(0, numel, 1024, [&](int64_t begin, int64_t end) {
            if (is_silu) {
                for (int64_t i = begin; i < end; ++i) {
                    const int branch_sign = b_ptr[i] ? 1 : -1;
                    z_ptr[i] = ojasx::math::halley_invert_silu(y_ptr[i], branch_sign, static_cast<int>(iters));
                }
            } else if (is_gelu) {
                for (int64_t i = begin; i < end; ++i) {
                    const int branch_sign = b_ptr[i] ? 1 : -1;
                    z_ptr[i] = ojasx::math::halley_invert_gelu(y_ptr[i], branch_sign, static_cast<int>(iters));
                }
            } else {
                for (int64_t i = begin; i < end; ++i) {
                    // Fallback identity
                    z_ptr[i] = y_ptr[i];
                }
            }
        });
    }

    return z_fp32.to(orig_dtype);
}

// ============================================================================
// 4. E8 Lattice Conway-Sloane Fast Projection Tensor Operator
// ============================================================================
at::Tensor conway_sloane_e8_project_tensor(at::Tensor x) {
    TORCH_CHECK(x.size(-1) == 8, "Last dimension of x must be 8 for E8 lattice");
    auto x_contig = x.contiguous();
    auto out = at::empty_like(x_contig);
    const int64_t num_blocks = x.numel() / 8;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conway_sloane_e8_project_cpu", [&] {
        const scalar_t* in_ptr = x_contig.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();

        at::parallel_for(0, num_blocks, 1024, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
                ojasx::math::lattice::conway_sloane_e8_project<scalar_t>(
                    in_ptr + b * 8,
                    out_ptr + b * 8
                );
            }
        });
    });
    return out;
}

/**
 * Fused on-the-fly register-dequantization GEMM forward:
 * Y = X @ W_dequant^T + Bias
 * Eliminates intermediate [N, K] float weight tensor materialization in DRAM.
 */
at::Tensor fused_e8_gemm_fwd(
    at::Tensor x,
    at::Tensor packed_weight,
    at::Tensor block_scales,
    c10::optional<at::Tensor> bias,
    int64_t group_size = 32
) {
    x = x.contiguous();
    packed_weight = packed_weight.contiguous();
    block_scales = block_scales.contiguous();
    TORCH_CHECK(packed_weight.scalar_type() == at::kByte, "packed_weight must be uint8");

    const int64_t K = x.size(-1);
    const int64_t N = packed_weight.size(0);
    const int64_t packed_k = packed_weight.size(1);
    TORCH_CHECK(packed_k == K / 4, "packed_weight dim 1 must match in_features / 4");

    auto x_flat = x.reshape({-1, K});
    const int64_t M = x_flat.size(0);

    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    at::Tensor y = at::empty(out_shape, x.options());
    auto y_flat = y.reshape({M, N});

    at::Tensor scales_contig = block_scales.to(x.scalar_type()).contiguous();
    at::Tensor bias_contig;
    const void* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined()) {
        bias_contig = bias->to(x.scalar_type()).contiguous();
        bias_ptr = bias_contig.data_ptr();
    }

    const uint8_t* w_ptr = packed_weight.data_ptr<uint8_t>();
    const void* s_data_ptr = scales_contig.data_ptr();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "fused_e8_gemm_fwd", [&] {
        const scalar_t* x_ptr = x_flat.data_ptr<scalar_t>();
        const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s_data_ptr);
        const scalar_t* b_ptr = reinterpret_cast<const scalar_t*>(bias_ptr);
        scalar_t* y_ptr = y_flat.data_ptr<scalar_t>();

        at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t m = start; m < end; ++m) {
                ojasx::math::lattice::fused_e8_gemm_forward<scalar_t>(
                    x_ptr + m * K,
                    w_ptr,
                    s_ptr,
                    b_ptr,
                    y_ptr + m * N,
                    1,
                    static_cast<size_t>(K),
                    static_cast<size_t>(N),
                    static_cast<size_t>(group_size)
                );
            }
        });
    });

    return y;
}

/**
 * Fused on-the-fly register-dequantization GEMM backward dx:
 * dX = dY @ W_dequant
 * Computes input gradients without materializing uncompressed weights.
 */
at::Tensor fused_e8_gemm_bwd_dx(
    at::Tensor grad_y,
    at::Tensor packed_weight,
    at::Tensor block_scales,
    int64_t in_features,
    int64_t group_size = 32
) {
    grad_y = grad_y.contiguous();
    packed_weight = packed_weight.contiguous();
    block_scales = block_scales.contiguous();

    const int64_t K = in_features;
    const int64_t N = packed_weight.size(0);

    auto gy_flat = grad_y.reshape({-1, N});
    const int64_t M = gy_flat.size(0);

    auto out_shape = grad_y.sizes().vec();
    out_shape.back() = K;
    at::Tensor grad_x = at::empty(out_shape, grad_y.options());
    auto gx_flat = grad_x.reshape({M, K});

    at::Tensor scales_contig = block_scales.to(grad_y.scalar_type()).contiguous();
    const uint8_t* w_ptr = packed_weight.data_ptr<uint8_t>();
    const void* s_data_ptr = scales_contig.data_ptr();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_y.scalar_type(), "fused_e8_gemm_bwd_dx", [&] {
        const scalar_t* gy_ptr = gy_flat.data_ptr<scalar_t>();
        const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s_data_ptr);
        scalar_t* gx_ptr = gx_flat.data_ptr<scalar_t>();

        at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t m = start; m < end; ++m) {
                ojasx::math::lattice::fused_e8_gemm_backward_dx<scalar_t>(
                    gy_ptr + m * N,
                    w_ptr,
                    s_ptr,
                    gx_ptr + m * K,
                    1,
                    static_cast<size_t>(K),
                    static_cast<size_t>(N),
                    static_cast<size_t>(group_size)
                );
            }
        });
    });

    return grad_x;
}

} // namespace ojasx_math

// ============================================================================
// PyBind11 Module Registration
// ============================================================================
PYBIND11_MODULE(ojasx_math_native, m) {
    m.doc() = "OjasX Native C++20 and Apple Silicon Metal Mathematical Engine";

    m.def("stormer_verlet_fwd", &ojasx_math::stormer_verlet_fwd,
          "2nd-order Störmer-Verlet FSAL symplectic forward integration with persistent FP32 accumulators",
          py::arg("q1"), py::arg("q2"), py::arg("f_func"), py::arg("g_func"), py::arg("h") = 1.0);

    m.def("stormer_verlet_step_tensors", &ojasx_math::stormer_verlet_step_tensors,
          "2nd-order Störmer-Verlet step using precomputed delta tensors",
          py::arg("q1"), py::arg("q2"), py::arg("delta_f1"), py::arg("delta_g"), py::arg("delta_f2"), py::arg("h") = 1.0);

    m.def("structured_pullback_tensor", &ojasx_math::structured_pullback_tensor,
          "Low-rank vector-Jacobian product for J = diag(d) + U * V^T in O(n * Rank) FLOPs",
          py::arg("v"), py::arg("d"), py::arg("U"), py::arg("V"), py::arg("prefer_metal") = true);

    m.def("halley_invert_activation_tensor", &ojasx_math::halley_invert_activation_tensor,
          "3rd-order Halley-Padé rational root inversion for non-monotonic activations (SiLU/GELU)",
          py::arg("y"), py::arg("branch_bit"), py::arg("act") = "silu", py::arg("iters") = 3, py::arg("prefer_metal") = true);

    m.def("conway_sloane_e8_project_tensor", &ojasx_math::conway_sloane_e8_project_tensor,
          "Fast O(8) algebraic Conway-Sloane snapping to 8D E8 Gosset lattice in C++20",
          py::arg("x"));

    m.def("fused_e8_gemm_fwd", &ojasx_math::fused_e8_gemm_fwd,
          "Fused on-the-fly register-dequantization GEMM forward without DRAM weight allocation",
          py::arg("x"), py::arg("packed_weight"), py::arg("block_scales"), py::arg("bias") = py::none(), py::arg("group_size") = 32);

    m.def("fused_e8_gemm_bwd_dx", &ojasx_math::fused_e8_gemm_bwd_dx,
          "Fused on-the-fly register-dequantization GEMM backward dx without DRAM weight allocation",
          py::arg("grad_y"), py::arg("packed_weight"), py::arg("block_scales"), py::arg("in_features"), py::arg("group_size") = 32);

    m.def("is_metal_available", &ojasx::math::metal::is_metal_available,
          "Check whether Apple Silicon Metal 4 GPU acceleration is available");
}

// ============================================================================
// PyTorch Library Registration (torch.ops.ojasx_math_native.*)
// ============================================================================
TORCH_LIBRARY(ojasx_math_native, m) {
    m.def("stormer_verlet_step_tensors(Tensor q1, Tensor q2, Tensor delta_f1, Tensor delta_g, Tensor delta_f2, float h) -> (Tensor, Tensor)");
    m.def("structured_pullback_tensor(Tensor v, Tensor d, Tensor U, Tensor V, bool prefer_metal=True) -> Tensor");
    m.def("halley_invert_activation_tensor(Tensor y, Tensor branch_bit, str act='silu', int iters=3, bool prefer_metal=True) -> Tensor");
    m.def("conway_sloane_e8_project_tensor(Tensor x) -> Tensor");
    m.def("fused_e8_gemm_fwd(Tensor x, Tensor packed_weight, Tensor block_scales, Tensor? bias=None, int group_size=32) -> Tensor");
    m.def("fused_e8_gemm_bwd_dx(Tensor grad_y, Tensor packed_weight, Tensor block_scales, int in_features, int group_size=32) -> Tensor");
}

TORCH_LIBRARY_IMPL(ojasx_math_native, CompositeExplicitAutograd, m) {
    m.impl("stormer_verlet_step_tensors", ojasx_math::stormer_verlet_step_tensors);
    m.impl("structured_pullback_tensor", ojasx_math::structured_pullback_tensor);
    m.impl("halley_invert_activation_tensor", ojasx_math::halley_invert_activation_tensor);
    m.impl("conway_sloane_e8_project_tensor", ojasx_math::conway_sloane_e8_project_tensor);
    m.impl("fused_e8_gemm_fwd", ojasx_math::fused_e8_gemm_fwd);
    m.impl("fused_e8_gemm_bwd_dx", ojasx_math::fused_e8_gemm_bwd_dx);
}
