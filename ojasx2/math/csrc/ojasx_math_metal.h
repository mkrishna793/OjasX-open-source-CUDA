#pragma once

#include <cstddef>
#include <cstdint>

namespace ojasx::math::metal {

bool is_metal_available();

bool metal_halley_invert_silu(
    const float* y,
    const uint8_t* branch,
    float* out,
    size_t n,
    int iters = 3
);

bool metal_halley_invert_gelu(
    const float* y,
    const uint8_t* branch,
    float* out,
    size_t n,
    int iters = 3
);

bool metal_structured_pullback(
    const float* v,
    const float* d,
    const float* U,
    const float* V,
    float* out,
    size_t n,
    size_t rank
);

} // namespace ojasx::math::metal
