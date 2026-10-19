#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ojasx_math_metal.h"

#include <iostream>
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ojasx::math::metal {

static const char* kMetalSource = R"(
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 1. Parallel Halley Inversion for SiLU (Apple Silicon Metal MSL)
// ============================================================================
kernel void halley_invert_silu_kernel(
    device const float* y [[buffer(0)]],
    device const uint8_t* branch [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant int& iters [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;

    const float yi = y[tid];
    const bool b = branch[tid] != 0;

    constexpr float z_star = -1.2784645427610738f;
    constexpr float y_star = -0.2784645427610738f;
    constexpr float phi_pp_star = 0.217809811f;

    const float delta = max(yi - y_star, 0.0f);
    const float sqrt_delta = sqrt(2.0f * delta / phi_pp_star);

    float z;
    if (b) {
        if (delta < 0.02f) {
            z = z_star + sqrt_delta * (1.0f - 0.16f * sqrt_delta);
        } else if (yi > 1.5f) {
            z = yi + exp(-min(yi, 30.0f));
        } else if (yi >= 0.0f) {
            z = 2.0f * yi / (1.0f + 0.6f * yi) + 0.3f * pow(yi, 1.5f);
        } else {
            z = z_star + sqrt_delta;
        }
    } else {
        if (delta < 0.02f) {
            z = z_star - sqrt_delta * (1.0f + 0.16f * sqrt_delta);
        } else if (yi > -0.01f) {
            const float L1 = log(max(-yi, 1e-15f));
            const float L2 = log(max(-L1, 1e-5f));
            z = L1 - L2;
        } else {
            z = z_star - sqrt_delta;
        }
    }

    for (int iter = 0; iter < iters; ++iter) {
        float sig;
        if (z >= 0.0f) {
            sig = 1.0f / (1.0f + exp(-z));
        } else {
            const float ez = exp(z);
            sig = ez / (1.0f + ez);
        }
        const float phi = z * sig;
        const float f = phi - yi;
        const float phi_p = sig * (1.0f + z * (1.0f - sig));
        const float phi_pp = sig * (1.0f - sig) * (2.0f + z * (1.0f - 2.0f * sig));
        float denom = 2.0f * phi_p * phi_p - f * phi_pp;
        if (abs(denom) < 1e-10f) {
            denom = (denom >= 0.0f ? 1e-10f : -1e-10f);
        }
        const float raw_step = (2.0f * f * phi_p) / denom;
        const float step = clamp(raw_step, -2.0f, 2.0f);
        z -= step;
    }

    out[tid] = z;
}

// ============================================================================
// 2. Parallel Halley Inversion for GELU (Apple Silicon Metal MSL)
// ============================================================================
kernel void halley_invert_gelu_kernel(
    device const float* y [[buffer(0)]],
    device const uint8_t* branch [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant int& iters [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;

    const float yi = y[tid];
    const bool b = branch[tid] != 0;

    constexpr float c = 0.7978845608028654f;
    constexpr float z_star = -0.751791524138f;
    constexpr float y_star = -0.1700460395f;
    constexpr float phi_pp_star = 0.43091926f;

    const float delta = max(yi - y_star, 0.0f);
    const float sqrt_delta = sqrt(2.0f * delta / phi_pp_star);

    float z;
    if (b) {
        if (delta < 0.02f) {
            z = z_star + sqrt_delta * (1.0f - 0.15f * sqrt_delta);
        } else if (yi > 1.5f) {
            z = yi;
        } else if (yi >= 0.0f) {
            z = 1.3f * yi / (1.0f + 0.3f * yi);
        } else {
            z = z_star + sqrt_delta;
        }
    } else {
        if (delta < 0.02f) {
            z = z_star - sqrt_delta * (1.0f + 0.15f * sqrt_delta);
        } else {
            const float val = -2.0f * log(max(-yi, 1e-15f)) - 1.837877f;
            z = -sqrt(max(val, 1.0f));
        }
    }

    for (int iter = 0; iter < iters; ++iter) {
        const float z2 = z * z;
        const float z3 = z2 * z;
        const float u = c * (z + 0.044715f * z3);
        const float t = tanh(u);
        const float phi = 0.5f * z * (1.0f + t);
        const float f = phi - yi;
        const float u_p = c * (1.0f + 3.0f * 0.044715f * z2);
        const float t_p = (1.0f - t * t) * u_p;
        const float phi_p = 0.5f * (1.0f + t) + 0.5f * z * t_p;
        const float u_pp = c * (6.0f * 0.044715f * z);
        const float t_pp = -2.0f * t * t_p * u_p + (1.0f - t * t) * u_pp;
        const float phi_pp = t_p + 0.5f * z * t_pp;

        float denom = 2.0f * phi_p * phi_p - f * phi_pp;
        if (abs(denom) < 1e-10f) {
            denom = (denom >= 0.0f ? 1e-10f : -1e-10f);
        }
        const float raw_step = (2.0f * f * phi_p) / denom;
        const float step = clamp(raw_step, -2.0f, 2.0f);
        z -= step;
    }

    out[tid] = z;
}

// ============================================================================
// 3. Structured Pullback with SIMD Group Reduction (Apple Silicon Metal MSL)
// ============================================================================
// Step 3a: Compute t_k = sum_{i=0}^{n-1} U[i * rank + k] * v[i] using SIMD group reductions
kernel void structured_pullback_reduce_tk(
    device const float* v [[buffer(0)]],
    device const float* U [[buffer(1)]],
    device float* t [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]], // corresponds to k in [0, rank)
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const uint k = tg_id;
    if (k >= rank) return;

    threadgroup float simd_sums[8]; // Up to 8 SIMD groups (256 threads)

    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        local_sum += U[i * rank + k] * v[i];
    }

    // Warp-level parallel reduction across SIMD group
    const float group_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        simd_sums[simd_id] = group_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group aggregates threadgroup partial sums
    if (simd_id == 0) {
        const float val = (simd_lane < 8) ? simd_sums[simd_lane] : 0.0f;
        const float total = simd_sum(val);
        if (simd_lane == 0) {
            t[k] = total;
        }
    }
}

// Step 3b: Evaluate w_i = d_i * v_i + sum_{k=0}^{rank-1} V[i * rank + k] * t_k
kernel void structured_pullback_eval_w(
    device const float* v [[buffer(0)]],
    device const float* d [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* t [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;

    float sum_lr = 0.0f;
    const uint offset = tid * rank;
    for (uint k = 0; k < rank; ++k) {
        sum_lr += V[offset + k] * t[k];
    }

    out[tid] = d[tid] * v[tid] + sum_lr;
}
)";

class MetalEngine {
private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> pso_silu_ = nil;
    id<MTLComputePipelineState> pso_gelu_ = nil;
    id<MTLComputePipelineState> pso_reduce_tk_ = nil;
    id<MTLComputePipelineState> pso_eval_w_ = nil;
    bool available_ = false;
    std::mutex mutex_;

    MetalEngine() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                available_ = false;
                return;
            }

            NSError* error = nil;
            NSString* src = [NSString stringWithUTF8String:kMetalSource];
            library_ = [device_ newLibraryWithSource:src options:nil error:&error];
            if (!library_) {
                std::cerr << "[OjasX Metal] Shader compilation error: "
                          << [[error localizedDescription] UTF8String] << std::endl;
                available_ = false;
                return;
            }

            auto load_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [library_ newFunctionWithName:name];
                if (!fn) return nil;
                NSError* err = nil;
                return [device_ newComputePipelineStateWithFunction:fn error:&err];
            };

            pso_silu_ = load_pso(@"halley_invert_silu_kernel");
            pso_gelu_ = load_pso(@"halley_invert_gelu_kernel");
            pso_reduce_tk_ = load_pso(@"structured_pullback_reduce_tk");
            pso_eval_w_ = load_pso(@"structured_pullback_eval_w");

            if (!pso_silu_ || !pso_gelu_ || !pso_reduce_tk_ || !pso_eval_w_) {
                available_ = false;
                return;
            }

            queue_ = [device_ newCommandQueue];
            available_ = (queue_ != nil);
        }
    }

public:
    static MetalEngine& instance() {
        static MetalEngine engine;
        return engine;
    }

    bool is_available() const {
        return available_;
    }

    bool halley_invert_silu(const float* y, const uint8_t* branch, float* out, size_t n, int iters) {
        if (!available_ || n == 0) return false;
        @autoreleasepool {
            std::lock_guard<std::mutex> lock(mutex_);

            const uint32_t count = static_cast<uint32_t>(n);
            const int32_t num_iters = iters;

            id<MTLBuffer> bufY = [device_ newBufferWithBytes:y length:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufB = [device_ newBufferWithBytes:branch length:n * sizeof(uint8_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufOut = [device_ newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufN = [device_ newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufIters = [device_ newBufferWithBytes:&num_iters length:sizeof(int32_t) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso_silu_];
            [enc setBuffer:bufY offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
            [enc setBuffer:bufN offset:0 atIndex:3];
            [enc setBuffer:bufIters offset:0 atIndex:4];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger tgSize = std::min((NSUInteger)256, pso_silu_.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            std::memcpy(out, [bufOut contents], n * sizeof(float));
            return true;
        }
    }

    bool halley_invert_gelu(const float* y, const uint8_t* branch, float* out, size_t n, int iters) {
        if (!available_ || n == 0) return false;
        @autoreleasepool {
            std::lock_guard<std::mutex> lock(mutex_);

            const uint32_t count = static_cast<uint32_t>(n);
            const int32_t num_iters = iters;

            id<MTLBuffer> bufY = [device_ newBufferWithBytes:y length:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufB = [device_ newBufferWithBytes:branch length:n * sizeof(uint8_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufOut = [device_ newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufN = [device_ newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufIters = [device_ newBufferWithBytes:&num_iters length:sizeof(int32_t) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso_gelu_];
            [enc setBuffer:bufY offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
            [enc setBuffer:bufN offset:0 atIndex:3];
            [enc setBuffer:bufIters offset:0 atIndex:4];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger tgSize = std::min((NSUInteger)256, pso_gelu_.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            std::memcpy(out, [bufOut contents], n * sizeof(float));
            return true;
        }
    }

    bool structured_pullback(const float* v, const float* d, const float* U, const float* V, float* out, size_t n, size_t rank) {
        if (!available_ || n == 0 || rank == 0) return false;
        @autoreleasepool {
            std::lock_guard<std::mutex> lock(mutex_);

            const uint32_t count = static_cast<uint32_t>(n);
            const uint32_t r = static_cast<uint32_t>(rank);

            id<MTLBuffer> bufV = [device_ newBufferWithBytes:v length:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufD = [device_ newBufferWithBytes:d length:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufU = [device_ newBufferWithBytes:U length:n * rank * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufMatV = [device_ newBufferWithBytes:V length:n * rank * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufT = [device_ newBufferWithLength:rank * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufOut = [device_ newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufN = [device_ newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufR = [device_ newBufferWithBytes:&r length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];

            // Pass 1: Reduce t_k
            id<MTLComputeCommandEncoder> enc1 = [cmdBuf computeCommandEncoder];
            [enc1 setComputePipelineState:pso_reduce_tk_];
            [enc1 setBuffer:bufV offset:0 atIndex:0];
            [enc1 setBuffer:bufU offset:0 atIndex:1];
            [enc1 setBuffer:bufT offset:0 atIndex:2];
            [enc1 setBuffer:bufN offset:0 atIndex:3];
            [enc1 setBuffer:bufR offset:0 atIndex:4];

            MTLSize threadgroupSize1 = MTLSizeMake(256, 1, 1);
            MTLSize threadgroups1 = MTLSizeMake(rank, 1, 1);
            [enc1 dispatchThreadgroups:threadgroups1 threadsPerThreadgroup:threadgroupSize1];
            [enc1 endEncoding];

            // Pass 2: Evaluate out
            id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
            [enc2 setComputePipelineState:pso_eval_w_];
            [enc2 setBuffer:bufV offset:0 atIndex:0];
            [enc2 setBuffer:bufD offset:0 atIndex:1];
            [enc2 setBuffer:bufMatV offset:0 atIndex:2];
            [enc2 setBuffer:bufT offset:0 atIndex:3];
            [enc2 setBuffer:bufOut offset:0 atIndex:4];
            [enc2 setBuffer:bufN offset:0 atIndex:5];
            [enc2 setBuffer:bufR offset:0 atIndex:6];

            MTLSize gridSize2 = MTLSizeMake(n, 1, 1);
            NSUInteger tgSize2 = std::min((NSUInteger)256, pso_eval_w_.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroupSize2 = MTLSizeMake(tgSize2, 1, 1);
            [enc2 dispatchThreads:gridSize2 threadsPerThreadgroup:threadgroupSize2];
            [enc2 endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            std::memcpy(out, [bufOut contents], n * sizeof(float));
            return true;
        }
    }
};

bool is_metal_available() {
    return MetalEngine::instance().is_available();
}

bool metal_halley_invert_silu(const float* y, const uint8_t* branch, float* out, size_t n, int iters) {
    return MetalEngine::instance().halley_invert_silu(y, branch, out, n, iters);
}

bool metal_halley_invert_gelu(const float* y, const uint8_t* branch, float* out, size_t n, int iters) {
    return MetalEngine::instance().halley_invert_gelu(y, branch, out, n, iters);
}

bool metal_structured_pullback(const float* v, const float* d, const float* U, const float* V, float* out, size_t n, size_t rank) {
    return MetalEngine::instance().structured_pullback(v, d, U, V, out, n, rank);
}

} // namespace ojasx::math::metal
