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
}

TORCH_LIBRARY_IMPL(ojasx_math_native, CompositeExplicitAutograd, m) {
    m.impl("stormer_verlet_step_tensors", ojasx_math::stormer_verlet_step_tensors);
    m.impl("structured_pullback_tensor", ojasx_math::structured_pullback_tensor);
    m.impl("halley_invert_activation_tensor", ojasx_math::halley_invert_activation_tensor);
}
