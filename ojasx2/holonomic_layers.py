"""
ojasx2/holonomic_layers.py

Holonomic Layer Primitives for Workload Families & Universal GLU Engine (Requirement R1):
1. `HolonomicLayerNorm`: Preserves compact mean and variance scalars (1 float/token)
   rather than materializing full normalized intermediate tensors.
2. `HolonomicGroupNorm`: Normalizes across channel groups preserving compact group statistics (N, G, 1).
3. `FrozenConvZeroTape`: Zero-tape autograd wrapper for frozen Conv1d/Conv2d/Conv3d backbones.
4. `PackedReLU`: 1-bit boolean sign mask retaining 1/32th the memory of FP32.
5. `HolonomicActivationLens`: Standalone activation lens with compact pullback.
6. Universal Holonomic GLU Engine:
   - All 10 GLU activation variants: SwiGLU, GeGLU (exact/tanh), ReGLU (1-bit mask g > 0),
     ReLU2GLU/SquaredReGLU, SigmoidGLU/nn.GLU, BilinearGLU, QuickGeGLU, MishGLU, TanhGLU, LeakyReGLU.
   - Both major layouts:
     * Split-projection (`gate_proj` + `up_proj` + `down_proj`) -> `HolonomicFusedGLUMLP`.
     * Chunked-projection (`fc_in` -> 2*H chunked by 2 -> `fc_out` or `nn.GLU`) -> `HolonomicFusedChunkedGLUMLP`, `HolonomicGLU`.
   - Zero fresh intermediate activation storage on tape.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .holonomic_math_compiler import (
    cull_zero_cotangent_gemm,
    get_activation_min_z,
    halley_activation_cotangent,
    halley_invert_activation,
    holonomic_geglu,
    holonomic_linear,
    holonomic_swiglu,
)
from .ops import packed_relu
from .residuals import pack_mask, unpack_mask


# =============================================================================
# 1. HolonomicLayerNorm (Compact Statistics: Mean + RSTD)
# =============================================================================

class _HolonomicLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        normalized_shape: Tuple[int, ...],
        eps: float,
    ) -> torch.Tensor:
        dim_count = len(normalized_shape)
        norm_dims = tuple(range(x.ndim - dim_count, x.ndim))
        m = 1
        for d in normalized_shape:
            m *= d

        # Compute mean and variance in float32 for high numerical stability
        x_f32 = x.float()
        mean = x_f32.mean(dim=norm_dims, keepdim=True)
        var = ((x_f32 - mean) ** 2).mean(dim=norm_dims, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x_f32 - mean) * rstd

        y = x_hat.to(x.dtype)
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias

        # Save ONLY x, weight, mean, rstd (compact statistics!)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.normalized_shape = normalized_shape
        ctx.norm_dims = norm_dims
        ctx.m = m
        ctx.has_bias = bias is not None
        ctx.x_dtype = x.dtype

        return y

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, weight, mean, rstd = ctx.saved_tensors
        norm_dims = ctx.norm_dims
        m = ctx.m
        x_dtype = ctx.x_dtype

        g_f32 = grad_out.float()
        x_f32 = x.float()
        x_hat = (x_f32 - mean) * rstd

        dw = None
        db = None
        if weight is not None and ctx.needs_input_grad[1]:
            leading_dims = tuple(range(x.ndim - len(ctx.normalized_shape)))
            dw = (g_f32 * x_hat).sum(dim=leading_dims).to(weight.dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            leading_dims = tuple(range(x.ndim - len(ctx.normalized_shape)))
            db = g_f32.sum(dim=leading_dims).to(x_dtype)

        dx = None
        if ctx.needs_input_grad[0]:
            if weight is not None:
                g_prime = g_f32 * weight.float()
            else:
                g_prime = g_f32

            mean_g_prime = g_prime.mean(dim=norm_dims, keepdim=True)
            mean_g_prime_x_hat = (g_prime * x_hat).mean(dim=norm_dims, keepdim=True)
            dx = (rstd * (g_prime - mean_g_prime - x_hat * mean_g_prime_x_hat)).to(x_dtype)

        return dx, dw, db, None, None


class HolonomicLayerNorm(nn.Module):
    """
    Drop-in replacement for `nn.LayerNorm` preserving compact statistics.
    Saves only (x, weight, mean, rstd) where mean and rstd are compact scalars (1 per token),
    eliminating the need to retain full intermediate normalized tensors.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HolonomicLayerNormFn.apply(
            x, self.weight, self.bias, self.normalized_shape, self.eps
        )

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# =============================================================================
# 2. HolonomicGroupNorm (Compact Statistics: Group Mean + Group RSTD)
# =============================================================================

class _HolonomicGroupNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        num_groups: int,
        num_channels: int,
        eps: float,
    ) -> torch.Tensor:
        n = x.size(0)
        c = x.size(1)
        orig_shape = x.shape

        x_reshaped = x.float().reshape(n, num_groups, -1)
        m = x_reshaped.size(-1)

        mean = x_reshaped.mean(dim=-1, keepdim=True)
        var = ((x_reshaped - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat_reshaped = (x_reshaped - mean) * rstd
        x_hat = x_hat_reshaped.reshape(orig_shape)

        y = x_hat.to(x.dtype)
        if weight is not None:
            w_shape = (1, c) + (1,) * (x.ndim - 2)
            y = y * weight.reshape(w_shape)
        if bias is not None:
            b_shape = (1, c) + (1,) * (x.ndim - 2)
            y = y + bias.reshape(b_shape)

        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.num_groups = num_groups
        ctx.num_channels = num_channels
        ctx.orig_shape = orig_shape
        ctx.m = m
        ctx.has_bias = bias is not None
        ctx.x_dtype = x.dtype

        return y

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, weight, mean, rstd = ctx.saved_tensors
        n = x.size(0)
        c = ctx.num_channels
        g = ctx.num_groups
        m = ctx.m
        x_dtype = ctx.x_dtype

        g_f32 = grad_out.float()
        x_reshaped = x.float().reshape(n, g, -1)
        x_hat_reshaped = (x_reshaped - mean) * rstd
        x_hat = x_hat_reshaped.reshape(x.shape)

        reduce_dims = (0,) + tuple(range(2, x.ndim))
        dw = None
        db = None
        if weight is not None and ctx.needs_input_grad[1]:
            dw = (g_f32 * x_hat).sum(dim=reduce_dims).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = g_f32.sum(dim=reduce_dims).to(x_dtype)

        dx = None
        if ctx.needs_input_grad[0]:
            if weight is not None:
                w_shape = (1, c) + (1,) * (x.ndim - 2)
                g_prime = g_f32 * weight.float().reshape(w_shape)
            else:
                g_prime = g_f32

            g_prime_reshaped = g_prime.reshape(n, g, -1)
            mean_gp = g_prime_reshaped.mean(dim=-1, keepdim=True)
            mean_gp_xhat = (g_prime_reshaped * x_hat_reshaped).mean(dim=-1, keepdim=True)
            dx_reshaped = rstd * (g_prime_reshaped - mean_gp - x_hat_reshaped * mean_gp_xhat)
            dx = dx_reshaped.reshape(x.shape).to(x_dtype)

        return dx, dw, db, None, None, None


class HolonomicGroupNorm(nn.Module):
    """
    Drop-in replacement for `nn.GroupNorm` preserving compact group statistics.
    Saves (x, weight, mean, rstd) with mean/rstd of shape (N, num_groups, 1).
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = float(eps)
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HolonomicGroupNormFn.apply(
            x, self.weight, self.bias, self.num_groups, self.num_channels, self.eps
        )

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


# =============================================================================
# 3. FrozenConvZeroTape (Conv1d, Conv2d, Conv3d with Zero Input Tape)
# =============================================================================

class _FrozenConvZeroTapeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: Tuple[int, ...],
        padding: Union[Tuple[int, ...], str],
        dilation: Tuple[int, ...],
        groups: int,
        dim: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.input_shape = x.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.dim = dim

        conv_fn = getattr(F, f"conv{dim}d")
        return conv_fn(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        (weight,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            grad_fn = getattr(torch.nn.grad, f"conv{ctx.dim}d_input")
            padding = ctx.padding
            if isinstance(padding, str):
                dim = ctx.dim
                k_size = weight.shape[2:]
                if padding == "valid":
                    padding = (0,) * dim
                elif padding == "same":
                    padding = tuple((k - 1) // 2 for k in k_size)
            dx = grad_fn(
                ctx.input_shape,
                weight,
                grad_out,
                stride=ctx.stride,
                padding=padding,
                dilation=ctx.dilation,
                groups=ctx.groups,
            )
        return dx, None, None, None, None, None, None, None


class FrozenConvZeroTape(nn.Module):
    """
    Zero-Tape Conv wrapper for frozen backbone layers (`weight.requires_grad=False`).
    Wraps Conv1d, Conv2d, and Conv3d. Never saves input activation `x` on autograd tape.
    """

    def __init__(self, conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]) -> None:
        super().__init__()
        if isinstance(conv, nn.Conv1d):
            self.dim = 1
        elif isinstance(conv, nn.Conv2d):
            self.dim = 2
        elif isinstance(conv, nn.Conv3d):
            self.dim = 3
        else:
            raise TypeError(f"Unsupported module for FrozenConvZeroTape: {type(conv)}")

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FrozenConvZeroTapeFn.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.dim,
        )

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, dim={self.dim}d (FrozenZeroTape)"


# =============================================================================
# 4. PackedReLU & Holonomic Activation Lenses
# =============================================================================

class PackedReLU(nn.Module):
    """1-Bit Boolean sign mask PackedReLU preserving exact semantics."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return packed_relu(x)

    def extra_repr(self) -> str:
        return "1-bit sign mask"


class _HalleyInverseActivationLensFunction(torch.autograd.Function):
    """
    Halley-Padé Branched Inverse Activation Adjoint (Innovation 2).
    Saves ONLY (y, pack_mask(x >= z_star)) where branch mask is 1 bit per element (32x smaller than FP32).
    Zero FP32 pre-activations saved on autograd tape.
    In backward, computes cotangent dz = dy * phi'(z) via 3rd-order Halley rational iteration
    in register FMA steps without requiring any matrix multiplications.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, act: str, neg_slope: float) -> torch.Tensor:
        orig_shape = x.shape
        z_star = get_activation_min_z(act)
        y = _eval_glu_activation(x, act, neg_slope)

        branch_bit = x >= z_star
        packed_branch = pack_mask(branch_bit)

        ctx.save_for_backward(y, packed_branch)
        ctx.orig_shape = orig_shape
        ctx.act = act
        ctx.neg_slope = neg_slope
        return y

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        y, packed_branch = ctx.saved_tensors
        branch_mask = unpack_mask(packed_branch, ctx.orig_shape)
        dx = halley_activation_cotangent(y, branch_mask, grad_out, ctx.act, ctx.neg_slope)
        return dx, None, None


class HalleyInverseActivationLens(nn.Module):
    """
    Drop-in activation lens utilizing Halley-Padé 3rd-order branched rational iteration.
    Zero pre-activation memory retention and zero backward recomputation GEMMs.
    """

    def __init__(self, act_name: str, negative_slope: float = 0.01, approximate: str = "tanh") -> None:
        super().__init__()
        self.act_name = act_name.lower()
        if self.act_name == "gelu":
            if approximate in ("none", "exact"):
                self.act_name = "gelu_exact"
            else:
                self.act_name = "gelu_tanh"
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HalleyInverseActivationLensFunction.apply(x, self.act_name, self.negative_slope)

    def extra_repr(self) -> str:
        return f"act={self.act_name}, halley_3rd_order_inverse=True"


class HolonomicActivationLens(HalleyInverseActivationLens):
    """
    Replaces standalone activations with memory-efficient Halley autograd lenses.
    """
    pass


# =============================================================================
# 5. Universal Holonomic GLU Mathematical Functions & Kernels
#    Supports All 10 Variants: SwiGLU, GeGLU, ReGLU, ReLU2GLU, SigmoidGLU,
#    BilinearGLU, QuickGeGLU, MishGLU, TanhGLU, LeakyReGLU
# =============================================================================

def _eval_glu_activation(g: torch.Tensor, act: str, neg_slope: float = 0.01) -> torch.Tensor:
    norm_act = act.lower()
    if norm_act in ("swiglu", "silu"):
        return F.silu(g)
    elif norm_act in ("geglu", "gelu", "gelu_tanh"):
        return F.gelu(g, approximate="tanh")
    elif norm_act in ("gelu_exact", "geglu_exact"):
        return F.gelu(g, approximate="none")
    elif norm_act in ("reglu", "relu"):
        return F.relu(g)
    elif norm_act in ("relu2glu", "squared_reglu", "relu2", "squared_relu"):
        return F.relu(g).square()
    elif norm_act in ("sigmoidglu", "sigmoid", "glu"):
        return torch.sigmoid(g)
    elif norm_act in ("bilinearglu", "bilinear", "identity", "none"):
        return g
    elif norm_act in ("quickgeglu", "quickgelu"):
        return g * torch.sigmoid(1.702 * g)
    elif norm_act in ("mishglu", "mish"):
        return F.mish(g)
    elif norm_act in ("tanhglu", "tanh"):
        return torch.tanh(g)
    elif norm_act in ("leaky_reglu", "leaky_relu"):
        return F.leaky_relu(g, negative_slope=neg_slope)
    else:
        raise ValueError(f"Unsupported Holonomic GLU activation: {act}")


def _eval_glu_gate_cotangent(g: torch.Tensor, d_phi_g: torch.Tensor, act: str, neg_slope: float = 0.01) -> torch.Tensor:
    norm_act = act.lower()
    if norm_act in ("swiglu", "silu"):
        sig = torch.sigmoid(g)
        return d_phi_g * sig * (1.0 + g * (1.0 - sig))
    elif norm_act in ("geglu", "gelu", "gelu_tanh"):
        return torch.ops.aten.gelu_backward(d_phi_g, g, approximate="tanh")
    elif norm_act in ("gelu_exact", "geglu_exact"):
        return torch.ops.aten.gelu_backward(d_phi_g, g, approximate="none")
    elif norm_act in ("reglu", "relu"):
        return torch.where(g > 0.0, d_phi_g, torch.zeros_like(d_phi_g))
    elif norm_act in ("relu2glu", "squared_reglu", "relu2", "squared_relu"):
        return torch.where(g > 0.0, d_phi_g * (2.0 * g), torch.zeros_like(d_phi_g))
    elif norm_act in ("sigmoidglu", "sigmoid", "glu"):
        sig = torch.sigmoid(g)
        return d_phi_g * sig * (1.0 - sig)
    elif norm_act in ("bilinearglu", "bilinear", "identity", "none"):
        return d_phi_g
    elif norm_act in ("quickgeglu", "quickgelu"):
        sig = torch.sigmoid(1.702 * g)
        return d_phi_g * sig * (1.0 + 1.702 * g * (1.0 - sig))
    elif norm_act in ("mishglu", "mish"):
        sp = F.softplus(g)
        tsp = torch.tanh(sp)
        sig = torch.sigmoid(g)
        dmish = tsp + g * sig * (1.0 - tsp.pow(2))
        return d_phi_g * dmish
    elif norm_act in ("tanhglu", "tanh"):
        t = torch.tanh(g)
        return d_phi_g * (1.0 - t.pow(2))
    elif norm_act in ("leaky_reglu", "leaky_relu"):
        return torch.where(g > 0.0, d_phi_g, d_phi_g * neg_slope)
    else:
        raise ValueError(f"Unsupported Holonomic GLU activation: {act}")


# =============================================================================
# 6. Layout A: Universal Split-Projection GLU (gate_proj + up_proj)
#    Zero Fresh Intermediate Activation Memory
# =============================================================================

class _HolonomicSplitGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        b_gate: Optional[torch.Tensor],
        b_up: Optional[torch.Tensor],
        act: str,
        neg_slope: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        in_features = x.size(-1)
        x_2d = x.reshape(-1, in_features)

        if b_gate is None and b_up is None and w_gate.is_contiguous() and w_up.is_contiguous():
            w_comb = torch.cat([w_gate, w_up], dim=0)
            g, u = F.linear(x_2d, w_comb).chunk(2, dim=-1)
        else:
            g = F.linear(x_2d, w_gate, b_gate)
            u = F.linear(x_2d, w_up, b_up)
        h = _eval_glu_activation(g, act, neg_slope) * u

        # Strictly save ONLY inputs & parameters (0 fresh intermediate activations saved on tape!)
        ctx.save_for_backward(x, w_gate, w_up, b_gate, b_up)
        ctx.orig_shape = orig_shape
        ctx.in_features = in_features
        ctx.act = act
        ctx.neg_slope = neg_slope

        out_shape = orig_shape[:-1] + (w_gate.size(0),)
        return h.reshape(out_shape)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, w_gate, w_up, b_gate, b_up = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        in_features = ctx.in_features
        act = ctx.act
        neg_slope = ctx.neg_slope

        x_2d = x.reshape(-1, in_features)
        dh_2d = grad_out.reshape(-1, w_gate.size(0))

        # Recompute G and U from (x, w_gate, w_up) in backward using single block GEMM when bias-free
        w_comb = torch.cat([w_gate.contiguous(), w_up.contiguous()], dim=0)
        if b_gate is None and b_up is None:
            g, u = F.linear(x_2d, w_comb).chunk(2, dim=-1)
        else:
            g = F.linear(x_2d, w_gate, b_gate)
            u = F.linear(x_2d, w_up, b_up)

        phi_g = _eval_glu_activation(g, act, neg_slope)
        du = dh_2d * phi_g
        d_phi_g = dh_2d * u
        dg = _eval_glu_gate_cotangent(g, d_phi_g, act, neg_slope)

        # Block-Matrix Cotangent Fusion (Innovation 1 & 3):
        # Concatenate [dg, du] into a single block tensor
        dG_U = torch.cat([dg, du], dim=-1)

        dw_gate = None
        dw_up = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            maybe_sparse = (dG_U.device.type == "cpu") and (
                act.lower() in ("relu", "reglu", "relu2", "relu2glu", "squared_reglu", "squared_relu")
            )
            dW_comb = cull_zero_cotangent_gemm(dG_U, x_2d, maybe_sparse=maybe_sparse)
            h_gate = w_gate.size(0)
            if ctx.needs_input_grad[1]:
                dw_gate = dW_comb[:h_gate]
            if ctx.needs_input_grad[2]:
                dw_up = dW_comb[h_gate:]

        db_gate = None
        if b_gate is not None and ctx.needs_input_grad[3]:
            db_gate = dg.sum(dim=0)

        db_up = None
        if b_up is not None and ctx.needs_input_grad[4]:
            db_up = du.sum(dim=0)

        dx = None
        if ctx.needs_input_grad[0]:
            dx_2d = torch.matmul(dG_U, w_comb)
            dx = dx_2d.reshape(orig_shape)

        return dx, dw_gate, dw_up, db_gate, db_up, None, None


class HolonomicFusedGLUMLP(nn.Module):
    """
    Universal Split-Projection Holonomic GLU MLP block.
    Supports all 10 GLU activation variants: SwiGLU, GeGLU, ReGLU, ReLU2GLU,
    SigmoidGLU, BilinearGLU, QuickGeGLU, MishGLU, TanhGLU, LeakyReGLU.
    Evaluates: `Y = down_proj(act(gate_proj(x)) * up_proj(x))`.
    Retains ZERO fresh intermediate activation tensors on the autograd tape.
    """

    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        act: str = "swiglu",
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act = act
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_gate = self.gate_proj.bias
        b_up = self.up_proj.bias
        hidden = _HolonomicSplitGLUFunction.apply(
            x, self.gate_proj.weight, self.up_proj.weight, b_gate, b_up, self.act, self.negative_slope
        )
        if not self.down_proj.weight.requires_grad:
            return holonomic_linear(hidden, self.down_proj.weight, self.down_proj.bias, act="none")
        return F.linear(hidden, self.down_proj.weight, self.down_proj.bias)

    def extra_repr(self) -> str:
        return f"act={self.act}, in={self.gate_proj.in_features}, hidden={self.gate_proj.out_features}, out={self.down_proj.out_features}"


# Convenience aliases for specific GLU variants
class HolonomicFusedSwiGLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        super().__init__(gate_proj, up_proj, down_proj, act="swiglu")


class HolonomicFusedGeGLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, approximate: str = "tanh") -> None:
        act = "geglu" if approximate == "tanh" else "gelu_exact"
        super().__init__(gate_proj, up_proj, down_proj, act=act)


class HolonomicFusedReGLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        super().__init__(gate_proj, up_proj, down_proj, act="reglu")


class HolonomicFusedReLU2GLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        super().__init__(gate_proj, up_proj, down_proj, act="relu2glu")


class HolonomicFusedSigmoidGLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        super().__init__(gate_proj, up_proj, down_proj, act="sigmoidglu")


class HolonomicFusedBilinearGLUMLP(HolonomicFusedGLUMLP):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        super().__init__(gate_proj, up_proj, down_proj, act="bilinearglu")


# =============================================================================
# 7. Layout B: Universal Chunked-Projection GLU (fc_in -> 2*H chunk -> act(g)*u -> fc_out)
#    and Standalone HolonomicGLU (nn.GLU drop-in)
# =============================================================================

class _HolonomicChunkedGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w_in: torch.Tensor,
        b_in: Optional[torch.Tensor],
        act: str,
        neg_slope: float,
        dim: int,
        gate_first: bool,
    ) -> torch.Tensor:
        orig_shape = x.shape
        in_features = x.size(-1)
        x_2d = x.reshape(-1, in_features)

        z = F.linear(x_2d, w_in, b_in)
        chunk_a, chunk_b = z.chunk(2, dim=-1)
        g, u = (chunk_a, chunk_b) if gate_first else (chunk_b, chunk_a)
        h = _eval_glu_activation(g, act, neg_slope) * u

        # Save ONLY (x, w_in, b_in) - 0 intermediate z, g, u on tape!
        ctx.save_for_backward(x, w_in, b_in)
        ctx.orig_shape = orig_shape
        ctx.in_features = in_features
        ctx.act = act
        ctx.neg_slope = neg_slope
        ctx.hidden_dim = g.size(-1)
        ctx.gate_first = gate_first

        out_shape = orig_shape[:-1] + (g.size(-1),)
        return h.reshape(out_shape)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, w_in, b_in = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        in_features = ctx.in_features
        act = ctx.act
        neg_slope = ctx.neg_slope
        hidden_dim = ctx.hidden_dim
        gate_first = ctx.gate_first

        x_2d = x.reshape(-1, in_features)
        dh_2d = grad_out.reshape(-1, hidden_dim)

        z = F.linear(x_2d, w_in, b_in)
        chunk_a, chunk_b = z.chunk(2, dim=-1)
        g, u = (chunk_a, chunk_b) if gate_first else (chunk_b, chunk_a)

        phi_g = _eval_glu_activation(g, act, neg_slope)
        du = dh_2d * phi_g
        d_phi_g = dh_2d * u
        dg = _eval_glu_gate_cotangent(g, d_phi_g, act, neg_slope)

        dz = torch.cat([dg, du], dim=-1) if gate_first else torch.cat([du, dg], dim=-1)

        dx = None
        if ctx.needs_input_grad[0]:
            dx_2d = torch.matmul(dz, w_in)
            dx = dx_2d.reshape(orig_shape)

        dw_in = None
        if ctx.needs_input_grad[1]:
            dw_in = torch.matmul(dz.transpose(0, 1), x_2d)

        db_in = None
        if b_in is not None and ctx.needs_input_grad[2]:
            db_in = dz.sum(dim=0)

        return dx, dw_in, db_in, None, None, None, None


class HolonomicFusedChunkedGLUMLP(nn.Module):
    """
    Universal Chunked-Projection Holonomic GLU MLP block.
    Evaluates: `Z = in_proj(x)`, splits `Z` into `(G, U)`, evaluates `H = act(G) * U`,
    and optionally projects with `out_proj`.
    Retains ZERO fresh intermediate tensors on the autograd tape.
    """

    def __init__(
        self,
        in_proj: nn.Linear,
        out_proj: Optional[nn.Linear] = None,
        act: str = "sigmoidglu",
        negative_slope: float = 0.01,
        dim: int = -1,
        gate_first: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.act = act
        self.negative_slope = float(negative_slope)
        self.dim = dim
        self.gate_first = gate_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_in = self.in_proj.bias
        hidden = _HolonomicChunkedGLUFunction.apply(
            x, self.in_proj.weight, b_in, self.act, self.negative_slope, self.dim, self.gate_first
        )
        if self.out_proj is not None:
            if not self.out_proj.weight.requires_grad:
                return holonomic_linear(hidden, self.out_proj.weight, self.out_proj.bias, act="none")
            return F.linear(hidden, self.out_proj.weight, self.out_proj.bias)
        return hidden

    def extra_repr(self) -> str:
        return f"act={self.act}, in={self.in_proj.in_features}, 2H={self.in_proj.out_features}, gate_first={self.gate_first}, has_out={self.out_proj is not None}"


class _HolonomicStandaloneGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, dim: int, act: str, neg_slope: float) -> torch.Tensor:
        # Standard nn.GLU splits into (a, b) and computes a * act(b) (where b is the gate)
        u, g = z.chunk(2, dim=dim)
        phi_g = _eval_glu_activation(g, act, neg_slope)
        h = u * phi_g
        ctx.save_for_backward(z)
        ctx.dim = dim
        ctx.act = act
        ctx.neg_slope = neg_slope
        return h

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        (z,) = ctx.saved_tensors
        dim = ctx.dim
        act = ctx.act
        neg_slope = ctx.neg_slope

        u, g = z.chunk(2, dim=dim)
        phi_g = _eval_glu_activation(g, act, neg_slope)
        du = grad_out * phi_g
        d_phi_g = grad_out * u
        dg = _eval_glu_gate_cotangent(g, d_phi_g, act, neg_slope)
        dz = torch.cat([du, dg], dim=dim)
        return dz, None, None, None


class HolonomicGLU(nn.Module):
    """
    Drop-in replacement for `nn.GLU` supporting arbitrary GLU activations
    (`sigmoid`, `silu`, `gelu`, `relu`, `relu2`, etc.) with zero extra tape retention.
    Matches PyTorch nn.GLU semantics: `input.chunk(2)[0] * act(input.chunk(2)[1])`.
    """

    def __init__(self, dim: int = -1, act: str = "sigmoid", negative_slope: float = 0.01) -> None:
        super().__init__()
        self.dim = dim
        self.act = act
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HolonomicStandaloneGLUFunction.apply(x, self.dim, self.act, self.negative_slope)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, act={self.act}"
