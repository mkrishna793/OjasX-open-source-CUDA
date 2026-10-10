"""Portable PyTorch reference operators used when optional native kernels are absent.

These implement numerical operations, not native TileLang source generation or
hardware-specific acceleration. Library objects must remain alive after registration.
"""
import math
import os

import torch
import torch.nn.functional as F

_LIBRARIES = []


def _pack(x):
    cols = x.shape[-1]
    bits = F.pad((x > 0).to(torch.int64), (0, (-cols) % 32))
    shifts = torch.arange(32, device=x.device, dtype=torch.int64)
    return (bits.reshape(*x.shape[:-1], (cols + 31) // 32, 32) << shifts).sum(-1).to(torch.int32)


def _unpack(packed, cols, dtype):
    shifts = torch.arange(32, device=packed.device, dtype=torch.int64)
    return ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 32
    )[..., :cols].to(dtype)


def _linear_forward(x, weight, bias):
    z = F.linear(x, weight, bias)
    return F.relu(z).square(), _pack(z)


def _linear_backward(grad_out, x, weight, bias, packed_mask):
    z = F.linear(x, weight, bias)
    dz = grad_out * 2 * z * _unpack(packed_mask, weight.shape[0], z.dtype)
    dx = dz @ weight
    rows = math.prod(x.shape[:-1])
    dw = dz.reshape(rows, weight.shape[0]).t() @ x.reshape(rows, weight.shape[1])
    db = dz.reshape(rows, weight.shape[0]).sum(0)
    return dx, dw, db


def _rms_forward(x, weight, eps):
    compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rstd = torch.rsqrt(compute.square().mean(-1, keepdim=True) + eps)
    return ((compute * rstd) * weight).to(x.dtype), rstd.float()


def _rms_backward(grad_out, x, weight, rstd):
    compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    x_hat = compute * rstd
    d_hat = grad_out * weight
    dx = rstd * (d_hat - x_hat * (d_hat * x_hat).mean(-1, keepdim=True))
    dw = (grad_out * x_hat).reshape(-1, x.shape[-1]).sum(0)
    return dx.to(x.dtype), dw.to(weight.dtype)


def _compress(x, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat = F.pad(x.detach().float().reshape(-1), (0, (-x.numel()) % block_size))
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    return (blocks / scale).round().clamp(-127, 127).to(torch.int8), scale.squeeze(-1)


def _decompress(q, scale, orig_shape, orig_dtype):
    return (q.float() * scale.unsqueeze(-1)).reshape(-1)[:math.prod(orig_shape)].reshape(orig_shape).to(orig_dtype)


def _unavailable_codegen(*args):
    raise RuntimeError("Native TileLang source generation requires a compatible native tile library; the PyTorch fallback implements numerical operators only")


def register_tile_fallbacks():
    """Fill only missing schemas, including TileLang operations absent from a CPU-only .so."""
    definitions = {
        "probe_silicon_info": ("probe_silicon_info() -> int[]", lambda: [0, 0, 0, os.cpu_count() or 1, 0]),
        "xor_swizzle_offset": ("xor_swizzle_offset(int row, int col, int stride, int num_banks) -> int", lambda row, col, stride, num_banks: row * stride + ((col & ~(num_banks - 1)) | ((col ^ row) & (num_banks - 1)))),
        "pack_1bit_ballot": ("pack_1bit_ballot(Tensor x) -> Tensor", _pack),
        "unpack_1bit_ballot": ("unpack_1bit_ballot(Tensor packed, int cols, ScalarType dtype) -> Tensor", _unpack),
        "fused_amx_linear_relu2_fwd": ("fused_amx_linear_relu2_fwd(Tensor x, Tensor weight, Tensor? bias) -> (Tensor, Tensor)", _linear_forward),
        "fused_amx_linear_relu2_bwd": ("fused_amx_linear_relu2_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor? bias, Tensor packed_mask) -> (Tensor, Tensor, Tensor)", _linear_backward),
        "fused_l1_rmsnorm_fwd": ("fused_l1_rmsnorm_fwd(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)", _rms_forward),
        "fused_l1_rmsnorm_bwd": ("fused_l1_rmsnorm_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)", _rms_backward),
        "compress_boundary_int8": ("compress_boundary_int8(Tensor x, int block_size) -> (Tensor, Tensor)", _compress),
        "decompress_boundary_int8": ("decompress_boundary_int8(Tensor q, Tensor scale, int[] orig_shape, ScalarType orig_dtype) -> Tensor", _decompress),
        "tilelang_metal_gemm_relu2_fwd": ("tilelang_metal_gemm_relu2_fwd(Tensor x, Tensor weight, Tensor? bias, int panel_size=4) -> (Tensor, Tensor)", lambda x, weight, bias, panel_size=4: _linear_forward(x, weight, bias)),
        "tilelang_universal_gemm_relu2_fwd": ("tilelang_universal_gemm_relu2_fwd(Tensor x, Tensor weight, Tensor? bias, str target_arch='auto', int panel_size=4) -> (Tensor, Tensor)", lambda x, weight, bias, target_arch='auto', panel_size=4: _linear_forward(x, weight, bias)),
        "tilelang_query_schedule": ("tilelang_query_schedule(int M, int N, int K, int num_warps, int swizzle_ordinal) -> int[]", _unavailable_codegen),
        "tilelang_query_chip_schedule": ("tilelang_query_chip_schedule(str target_arch, int M, int N, int K, int smem_kb, int l2_kb) -> int[]", _unavailable_codegen),
        "tilelang_emit_metal_gemm_source": ("tilelang_emit_metal_gemm_source(int block_m, int block_n, int block_k, int panel_size) -> str", _unavailable_codegen),
        "tilelang_emit_universal_source": ("tilelang_emit_universal_source(str target_arch, int block_m, int block_n, int block_k, int panel_size) -> str", _unavailable_codegen),
    }
    library = torch.library.Library("ojasx_hal", "FRAGMENT")
    for name, (schema, implementation) in definitions.items():
        if not hasattr(torch.ops.ojasx_hal, name):
            library.define(schema)
            library.impl(name, implementation, "CompositeExplicitAutograd")
    _LIBRARIES.append(library)
