"""PyTorch-registered operators with explicit residual and backward contracts."""
from typing import Tuple
import torch
import torch.nn.functional as F
from .residuals import pack_mask, unpack_mask


def _floating(x):
    if not x.is_floating_point():
        raise TypeError("OjasX operators require real floating tensors")


@torch.library.custom_op("ojasx_verified::packed_relu", mutates_args=())
def packed_relu(x: torch.Tensor) -> torch.Tensor:
    _floating(x)
    return F.relu(x)


@packed_relu.register_fake
def _relu_fake(x):
    return torch.empty_like(x)


def _relu_setup(ctx, inputs, output):
    (x,) = inputs
    ctx.shape = x.shape
    # NaN follows PyTorch's threshold_backward convention: it passes dY.
    ctx.save_for_backward(pack_mask(~(x <= 0)))


def _relu_backward(ctx, dy):
    (packed,) = ctx.saved_tensors
    return torch.where(unpack_mask(packed, ctx.shape), dy, torch.zeros_like(dy))


packed_relu.register_autograd(_relu_backward, setup_context=_relu_setup)


@torch.library.custom_op("ojasx_verified::gelu", mutates_args=())
def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    _floating(x)
    if approximate not in ("none", "tanh"):
        raise ValueError("approximate must be 'none' or 'tanh'")
    return F.gelu(x, approximate=approximate)


@gelu.register_fake
def _gelu_fake(x, approximate="none"):
    return torch.empty_like(x)


def _gelu_setup(ctx, inputs, output):
    x, ctx.approximate = inputs
    ctx.save_for_backward(x)


def _gelu_backward(ctx, dy):
    (x,) = ctx.saved_tensors
    return torch.ops.aten.gelu_backward(dy, x, approximate=ctx.approximate), None


gelu.register_autograd(_gelu_backward, setup_context=_gelu_setup)


@torch.library.custom_op("ojasx_verified::silu", mutates_args=())
def silu(x: torch.Tensor) -> torch.Tensor:
    _floating(x)
    return F.silu(x)


@silu.register_fake
def _silu_fake(x):
    return torch.empty_like(x)


def _silu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _silu_backward(ctx, dy):
    x = ctx.saved_tensors[0]
    if torch.is_grad_enabled():
        sig = torch.sigmoid(x)
        return dy * sig * (1 + x * (1 - sig))
    return torch.ops.aten.silu_backward(dy, x)


silu.register_autograd(_silu_backward, setup_context=_silu_setup)


@torch.library.custom_op("ojasx_verified::swiglu", mutates_args=())
def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    _floating(gate)
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("gate and up must have identical shape, dtype and device")
    return F.silu(gate) * up


@swiglu.register_fake
def _swiglu_fake(gate, up):
    return torch.empty_like(gate)


def _swiglu_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _swiglu_backward(ctx, dy):
    gate, up = ctx.saved_tensors
    # Recompute only pointwise SiLU, never a linear projection or a GEMM.
    if torch.is_grad_enabled():
        sig = torch.sigmoid(gate)
        dgate = (dy * up) * sig * (1 + gate * (1 - sig))
    else:
        dgate = torch.ops.aten.silu_backward(dy * up, gate)
    return dgate, dy * F.silu(gate)


swiglu.register_autograd(_swiglu_backward, setup_context=_swiglu_setup)


@torch.library.custom_op("ojasx_verified::rmsnorm_aux", mutates_args=())
def rmsnorm_aux(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    _floating(x)
    if x.ndim < 1 or x.shape[-1] == 0 or gamma.shape != (x.shape[-1],):
        raise ValueError("RMSNorm requires a nonempty last dimension and matching 1D gamma")
    if gamma.device != x.device or gamma.dtype != x.dtype:
        raise ValueError("x and gamma must have matching dtype and device")
    if eps <= 0:
        raise ValueError("eps must be positive")
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x * r) * gamma, r


@rmsnorm_aux.register_fake
def _rms_fake(x, gamma, eps=1e-6):
    return torch.empty_like(x), x.new_empty((*x.shape[:-1], 1))


def _rms_setup(ctx, inputs, output):
    x, gamma, ctx.eps = inputs
    ctx.save_for_backward(x, gamma, output[1])


def _rms_backward(ctx, dy, dr):
    x, gamma, saved_r = ctx.saved_tensors
    # Recompute the row statistic under higher-order differentiation so its
    # dependence on x remains represented. First-order uses the saved statistic.
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + ctx.eps) if torch.is_grad_enabled() else saved_r
    n = x.shape[-1]
    if dy is None:
        dy = torch.zeros_like(x)
    g = dy * gamma
    dx = r * g - (x * r.pow(3) / n) * (g * x).sum(-1, keepdim=True)
    if dr is not None:
        dx = dx - dr * x * r.pow(3) / n
    dg = dy * (x * r)
    if x.ndim > 1:
        dg = dg.sum(tuple(range(x.ndim - 1)))
    return dx, dg, None


rmsnorm_aux.register_autograd(_rms_backward, setup_context=_rms_setup)


def rmsnorm(x, gamma, eps=1e-6):
    return rmsnorm_aux(x, gamma, eps)[0]
