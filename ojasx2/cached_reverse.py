"""Cached sufficient branch residuals: no projection replay in cached backward.

Real-algebra reconstruction changes floating rounding. An optional Metal SiLU
kernel jointly reconstructs the previous state and applies the activation VJP.
Neither the cache nor this kernel guarantees universal speed or stable inversion.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F


METAL_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
kernel void silu_reverse(
    device const float* z,
    device const float* state,
    device const float* incoming,
    device float* reconstructed,
    device float* delta,
    constant float& scale,
    uint i [[thread_position_in_grid]]) {
    const float t = z[i];
    const float s = 1.0f / (1.0f + exp(-t));
    reconstructed[i] = state[i] - scale * (t * s);
    delta[i] = (scale * incoming[i]) * (s * (1.0f + t * (1.0f-s)));
}
'''


@lru_cache(maxsize=1)
def _metal_library():
    return torch.mps.compile_shader(METAL_SOURCE)


def cached_branch_reverse(z, state, incoming, scale, activation, pointwise_backend):
    if pointwise_backend == 'metal':
        if z.device.type != 'mps' or z.dtype != torch.float32 or activation != 'silu':
            raise ValueError('Metal cached reverse requires MPS FP32 SiLU')
        reconstructed = torch.empty_like(z)
        delta = torch.empty_like(z)
        # Explicit copies preserve arbitrary incoming cotangent strides.
        _metal_library().silu_reverse(z.contiguous(), state.contiguous(), incoming.contiguous(),
                                     reconstructed, delta, float(scale))
        return reconstructed, delta
    value = scale * (F.silu(z) if activation == 'silu' else F.gelu(z))
    scaled = incoming * scale
    delta = (torch.ops.aten.silu_backward(scaled, z) if activation == 'silu'
             else torch.ops.aten.gelu_backward(scaled, z, approximate='none'))
    return state-value, delta


class CachedReversibleChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, activation, scale, pointwise_backend, cache_interval, *parameters):
        cached = []
        indices = []
        for i in range(0, len(parameters), 4):
            wf,bf,wg,bg=parameters[i:i+4]
            zf=F.linear(x2,wf,bf)
            x1=x1+scale*(F.silu(zf) if activation=='silu' else F.gelu(zf))
            zg=F.linear(x1,wg,bg)
            x2=x2+scale*(F.silu(zg) if activation=='silu' else F.gelu(zg))
            if (i//4) % cache_interval == 0:
                indices.append(i)
                cached.extend((zf,zg))
        ctx.save_for_backward(x1,x2,*parameters,*cached)
        ctx.count=len(parameters)
        ctx.indices=tuple(indices)
        ctx.activation=activation;ctx.scale=scale;ctx.pointwise_backend=pointwise_backend
        return x1,x2

    @staticmethod
    def backward(ctx,g1,g2):
        if torch.is_grad_enabled():
            raise RuntimeError('Cached reversible backward supports first-order gradients only; use strict mode')
        y1,y2,*saved=ctx.saved_tensors
        parameters=saved[:ctx.count]
        cached=dict(zip(ctx.indices,zip(saved[ctx.count::2],saved[ctx.count+1::2])))
        gradients=[None]*ctx.count
        def reverse_branch(primal,state,w,b,incoming,z,needw,needb):
            # Uncached blocks also use an explicit pullback, but replay exactly
            # one projection. Cached blocks bypass it entirely.
            if z is None:z=F.linear(primal,w,b)
            reconstructed,delta=cached_branch_reverse(z,state,incoming,ctx.scale,
                ctx.activation,ctx.pointwise_backend)
            flat_delta=delta.reshape(-1,delta.shape[-1])
            flat_primal=primal.reshape(-1,primal.shape[-1])
            return (reconstructed, delta@w,
                    flat_delta.T@flat_primal if needw else None,
                    flat_delta.sum(0) if needb else None)
        for i in range(ctx.count-4,-1,-4):
            wf,bf,wg,bg=parameters[i:i+4]
            needs=ctx.needs_input_grad[6+i:10+i]
            zf,zg=cached.get(i,(None,None))
            x2,gdy,ggw,ggb=reverse_branch(y1,y2,wg,bg,g2,zg,needs[2],needs[3])
            adj1=g1+gdy
            x1,fdx,gfw,gfb=reverse_branch(x2,y1,wf,bf,adj1,zf,needs[0],needs[1])
            g1,g2=adj1,g2+fdx
            gradients[i:i+4]=[gfw,gfb,ggw,ggb]
            y1,y2=x1,x2
        return (g1 if ctx.needs_input_grad[0] else None,
                g2 if ctx.needs_input_grad[1] else None,None,None,None,None,*gradients)


def cached_reverse_cost(depth, cached_blocks, m, d):
    if not 0<=cached_blocks<=depth or min(depth,m,d)<=0:
        raise ValueError('Positive depth/state dimensions and bounded cache count required')
    return {'native_joint_gemms':6*depth,'replay_joint_gemms':8*depth,
            'cached_joint_gemms':8*depth-2*cached_blocks,
            'cached_backward_gemms':6*depth-2*cached_blocks,
            'fresh_cached_elements':(2+2*cached_blocks)*m*d,
            'scope':'All input and weight gradients; GEMM counts, not time/FLOPs; fresh tape excludes parameter/input storage and scratch'}
