"""Holonomic Fused Linear Cross-Entropy Loss Engine.

Computes cross-entropy loss and its backward pullback directly from hidden representations
H in [M, D] and LM head weights W in [V, D] without ever materializing the giant [B, S, V]
vocabulary logits tensor in DRAM.

Dual-pass streaming accumulation over token tiles (B_tok=128) and vocabulary tiles (B_voc=4096)
bounds cross-entropy workspace memory to O(B x S) rather than O(B x S x V).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn


# =============================================================================
# Core Streaming Mathematical Kernels
# =============================================================================

def _is_concrete_eager_tensor(t: torch.Tensor) -> bool:
    if type(t) is not torch.Tensor:
        return False
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling():
        return False
    return True


def _streaming_fused_lce_forward(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    tile_tok: int = 512,
    tile_voc: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dual-pass streaming forward accumulation over token and vocabulary tiles."""
    d = h.shape[-1]
    v = w.shape[0]
    m = (h.numel() // d) if d > 0 else 0

    if w.shape[1] != d:
        raise ValueError(f"Weight feature dimension {w.shape[1]} does not match hidden dimension {d}")
    if bias is not None and bias.shape[0] != v:
        raise ValueError(f"Bias dimension {bias.shape[0]} does not match vocabulary size {v}")

    # Zero-element handling
    if m == 0 or d == 0 or v == 0:
        loss = (
            torch.zeros((), dtype=h.dtype, device=h.device)
            if reduction != "none"
            else torch.zeros(targets.shape, dtype=h.dtype, device=h.device)
        )
        lse = torch.zeros(targets.shape, dtype=torch.float32, device=h.device)
        return loss, lse

    compute_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    h_2d = h.reshape(m, d)
    h_2d_comp = h_2d if h_2d.dtype == compute_dtype else h_2d.to(compute_dtype)
    w_comp = w if w.dtype == compute_dtype else w.to(compute_dtype)
    b_comp = (bias if bias.dtype == compute_dtype else bias.to(compute_dtype)) if bias is not None else None
    t_1d = targets.reshape(m)

    # Valid mask without data-dependent python branches
    mask = (t_1d != ignore_index) & (t_1d >= 0) & (t_1d < v)
    clamped_t = torch.clamp(t_1d, 0, v - 1)

    # Target dot-product z_{m, y_m} computed in O(M x D) without logits tensor
    w_target = w_comp[clamped_t]
    z_target = (h_2d_comp * w_target).sum(dim=-1)
    if b_comp is not None:
        z_target = z_target + b_comp[clamped_t]
    z_target = torch.where(mask, z_target, torch.zeros_like(z_target))

    lse = torch.zeros(m, dtype=compute_dtype, device=h.device)
    sum_z = torch.zeros(m, dtype=compute_dtype, device=h.device) if label_smoothing > 0.0 else None

    # Dual-pass tile streaming loop: transient memory <= O(tile_tok x tile_voc)
    for tok_s in range(0, m, tile_tok):
        tok_e = min(tok_s + tile_tok, m)
        h_chunk = h_2d_comp[tok_s:tok_e]
        m_chunk = torch.full((tok_e - tok_s,), -float("inf"), dtype=compute_dtype, device=h.device)
        s_chunk = torch.zeros(tok_e - tok_s, dtype=compute_dtype, device=h.device)
        sum_z_chunk = (
            torch.zeros(tok_e - tok_s, dtype=compute_dtype, device=h.device)
            if label_smoothing > 0.0
            else None
        )

        for v_s in range(0, v, tile_voc):
            v_e = min(v_s + tile_voc, v)
            w_chunk = w_comp[v_s:v_e]
            z_tile = torch.matmul(h_chunk, w_chunk.t())
            if b_comp is not None:
                z_tile = z_tile + b_comp[v_s:v_e]

            m_tile = z_tile.amax(dim=-1)
            m_new = torch.maximum(m_chunk, m_tile)
            exp_tile = torch.exp(z_tile - m_new.unsqueeze(-1))
            s_chunk = s_chunk * torch.exp(m_chunk - m_new) + exp_tile.sum(dim=-1)
            m_chunk = m_new

            if sum_z_chunk is not None:
                sum_z_chunk = sum_z_chunk + z_tile.sum(dim=-1)

        lse[tok_s:tok_e] = m_chunk + torch.log(s_chunk)
        if sum_z is not None:
            sum_z[tok_s:tok_e] = sum_z_chunk

    # Exact Cross-Entropy loss with optional label smoothing
    if label_smoothing > 0.0:
        loss_1d = lse - (1.0 - label_smoothing) * z_target - (label_smoothing / float(v)) * sum_z
    else:
        loss_1d = lse - z_target

    loss_1d = torch.where(mask, loss_1d, torch.zeros_like(loss_1d))

    if reduction == "mean":
        m_valid = mask.sum(dtype=compute_dtype)
        loss = loss_1d.sum() / torch.clamp(m_valid, min=1.0)
    elif reduction == "sum":
        loss = loss_1d.sum()
    elif reduction == "none":
        loss = loss_1d.reshape(targets.shape)
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Expected 'mean', 'sum', or 'none'.")

    return loss.to(h.dtype), lse.reshape(targets.shape).to(torch.float32)


def _streaming_fused_lce_backward(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor],
    lse: torch.Tensor,
    grad_loss: Optional[torch.Tensor],
    grad_lse: Optional[torch.Tensor],
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    tile_tok: int = 512,
    tile_voc: int = 4096,
    needs_dh: Optional[bool] = None,
    needs_dw: Optional[bool] = None,
    needs_db: Optional[bool] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None]:
    """Streaming chunked pullback directly to dH and dW without materializing logits."""
    d = h.shape[-1]
    v = w.shape[0]
    m = (h.numel() // d) if d > 0 else 0

    if needs_dh is None:
        needs_dh = bool(h.requires_grad)
    if needs_dw is None:
        needs_dw = bool(w.requires_grad)
    if needs_db is None:
        needs_db = bool(bias is not None and bias.requires_grad)

    if m == 0 or d == 0 or v == 0:
        dh = torch.zeros_like(h) if needs_dh else None
        dw = torch.zeros_like(w) if needs_dw else None
        db = torch.zeros_like(bias) if (needs_db and bias is not None) else None
        return dh, dw, None, db, None, None, None

    compute_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    h_2d = h.reshape(m, d)
    h_2d = h_2d if h_2d.dtype == compute_dtype else h_2d.to(compute_dtype)
    w_comp = w if w.dtype == compute_dtype else w.to(compute_dtype)
    b_comp = (bias if bias.dtype == compute_dtype else bias.to(compute_dtype)) if bias is not None else None
    t_1d = targets.reshape(m)
    lse_1d = lse.reshape(m)
    lse_1d = lse_1d if lse_1d.dtype == compute_dtype else lse_1d.to(compute_dtype)

    mask = (t_1d != ignore_index) & (t_1d >= 0) & (t_1d < v)
    clamped_t = torch.clamp(t_1d, 0, v - 1)
    m_valid = mask.sum(dtype=compute_dtype)

    # Token loss weight c_m
    c = torch.zeros(m, dtype=compute_dtype, device=h.device)
    if grad_loss is not None:
        gl = grad_loss.to(compute_dtype)
        if reduction == "mean":
            c_val = gl / torch.clamp(m_valid, min=1.0)
            c = torch.where(mask, c_val, torch.zeros_like(c))
        elif reduction == "sum":
            c = torch.where(mask, gl, torch.zeros_like(c))
        elif reduction == "none":
            c = torch.where(mask, gl.reshape(m), torch.zeros_like(c))

    # Total softmax scalar coefficient: alpha_m = c_m + g_lse
    alpha = c.clone()
    if grad_lse is not None:
        alpha = alpha + grad_lse.reshape(m).to(compute_dtype)

    dh_2d = torch.zeros((m, d), dtype=compute_dtype, device=h.device) if needs_dh else None
    dw = torch.zeros_like(w, dtype=compute_dtype) if needs_dw else None
    db = torch.zeros_like(bias, dtype=compute_dtype) if (needs_db and bias is not None) else None

    # Step 1: Target subtraction term: - c_m (1 - eps) * 1[v = y_m]
    eps = label_smoothing
    target_scale = c * (1.0 - eps)
    if needs_dh and dh_2d is not None:
        dh_2d -= target_scale.unsqueeze(-1) * w_comp[clamped_t]
    if needs_dw and dw is not None:
        dw.index_add_(0, clamped_t, -(target_scale.unsqueeze(-1) * h_2d))
    if needs_db and db is not None:
        db.index_add_(0, clamped_t, -target_scale)

    # Step 2: Uniform label smoothing shift: - c_m * (eps / V)
    if eps > 0.0:
        c_sum = c.sum()
        if needs_dh and dh_2d is not None:
            w_mean = w_comp.mean(dim=0)
            dh_2d -= (c * eps).unsqueeze(-1) * w_mean
        if needs_dw and dw is not None:
            h_weighted = (c.unsqueeze(-1) * h_2d).sum(dim=0)
            dw -= (eps / float(v)) * h_weighted.unsqueeze(0).expand(v, d)
        if needs_db and db is not None:
            db -= (eps / float(v)) * c_sum

    # Step 3: Softmax streaming accumulation (with FakeTensor-safe zero-cotangent chunk culling)
    can_cull_chunks = _is_concrete_eager_tensor(alpha)
    for tok_s in range(0, m, tile_tok):
        tok_e = min(tok_s + tile_tok, m)
        alpha_chunk = alpha[tok_s:tok_e]
        if can_cull_chunks and not bool(torch.any(alpha_chunk != 0)):
            continue
        h_chunk = h_2d[tok_s:tok_e]
        lse_chunk = lse_1d[tok_s:tok_e]

        for v_s in range(0, v, tile_voc):
            v_e = min(v_s + tile_voc, v)
            w_chunk = w_comp[v_s:v_e]
            z_tile = torch.matmul(h_chunk, w_chunk.t())
            if b_comp is not None:
                z_tile = z_tile + b_comp[v_s:v_e]

            p_tile = torch.exp(z_tile - lse_chunk.unsqueeze(-1))
            p_scaled = alpha_chunk.unsqueeze(-1) * p_tile

            if needs_dh and dh_2d is not None:
                dh_2d[tok_s:tok_e] += torch.matmul(p_scaled, w_chunk)
            if needs_dw and dw is not None:
                dw[v_s:v_e] += torch.matmul(p_scaled.t(), h_chunk)
            if needs_db and db is not None:
                db[v_s:v_e] += p_scaled.sum(dim=0)

    dh = dh_2d.reshape(h.shape).to(h.dtype) if (needs_dh and dh_2d is not None) else None
    dw = dw.to(w.dtype) if (needs_dw and dw is not None) else None
    db = db.to(bias.dtype) if (needs_db and db is not None and bias is not None) else None

    return dh, dw, None, db, None, None, None


# =============================================================================
# PyTorch Custom Operator Registration (ojasx_holonomic::fused_linear_cross_entropy)
# =============================================================================

@torch.library.custom_op("ojasx_holonomic::fused_linear_cross_entropy", mutates_args=())
def _fused_linear_cross_entropy_op(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluates Fused Linear Cross-Entropy loss and row-wise LogSumExp vector."""
    return _streaming_fused_lce_forward(
        h=h,
        w=w,
        targets=targets,
        bias=bias,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


@torch.library.register_fake("ojasx_holonomic::fused_linear_cross_entropy")
def _fake_fused_linear_cross_entropy(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Abstract FakeTensor / Meta propagation kernel for torch.compile and AOTAutograd."""
    if reduction == "none":
        loss = torch.empty(targets.shape, dtype=h.dtype, device=h.device)
    else:
        loss = torch.empty((), dtype=h.dtype, device=h.device)
    lse = torch.empty(targets.shape, dtype=torch.float32, device=h.device)
    return loss, lse


def _fused_lce_setup(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    h, w, targets, bias, ignore_index, label_smoothing, reduction = inputs
    loss, lse = output
    ctx.save_for_backward(h, w, targets, bias, lse)
    ctx.ignore_index = int(ignore_index)
    ctx.label_smoothing = float(label_smoothing)
    ctx.reduction = str(reduction)


def _fused_lce_backward(
    ctx: Any, grad_loss: Optional[torch.Tensor], grad_lse: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None]:
    h, w, targets, bias, lse = ctx.saved_tensors
    return _streaming_fused_lce_backward(
        h=h,
        w=w,
        targets=targets,
        bias=bias,
        lse=lse,
        grad_loss=grad_loss,
        grad_lse=grad_lse,
        ignore_index=ctx.ignore_index,
        label_smoothing=ctx.label_smoothing,
        reduction=ctx.reduction,
        needs_dh=bool(ctx.needs_input_grad[0]),
        needs_dw=bool(ctx.needs_input_grad[1]),
        needs_db=bool(bias is not None and ctx.needs_input_grad[3]),
    )


_fused_linear_cross_entropy_op.register_autograd(_fused_lce_backward, setup_context=_fused_lce_setup)


# =============================================================================
# Explicit Autograd Function
# =============================================================================

class _HolonomicFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    Autograd Function implementing Fused Linear Cross-Entropy:
    - Retains only H, W, targets, bias, and 1D vector LSE in [M] (0 bytes [M, V] logits!).
    - Dual-pass streaming accumulation over token tiles and vocabulary tiles.
    """

    @staticmethod
    def forward(
        ctx: Any,
        h: torch.Tensor,
        w: torch.Tensor,
        targets: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, lse = _streaming_fused_lce_forward(
            h=h,
            w=w,
            targets=targets,
            bias=bias,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        ctx.save_for_backward(h, w, targets, bias, lse)
        ctx.ignore_index = int(ignore_index)
        ctx.label_smoothing = float(label_smoothing)
        ctx.reduction = str(reduction)
        return loss, lse

    @staticmethod
    def backward(
        ctx: Any, grad_loss: Optional[torch.Tensor], grad_lse: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None]:
        h, w, targets, bias, lse = ctx.saved_tensors
        return _streaming_fused_lce_backward(
            h=h,
            w=w,
            targets=targets,
            bias=bias,
            lse=lse,
            grad_loss=grad_loss,
            grad_lse=grad_lse,
            ignore_index=ctx.ignore_index,
            label_smoothing=ctx.label_smoothing,
            reduction=ctx.reduction,
            needs_dh=bool(ctx.needs_input_grad[0]),
            needs_dw=bool(ctx.needs_input_grad[1]),
            needs_db=bool(bias is not None and ctx.needs_input_grad[3]),
        )


# =============================================================================
# Functional and Module Interfaces
# =============================================================================

def holonomic_fused_linear_cross_entropy(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cross-entropy loss directly from hidden representations H and weights W.

    Returns (loss, lse).
    """
    return torch.ops.ojasx_holonomic.fused_linear_cross_entropy(
        h, w, targets, bias, ignore_index, label_smoothing, reduction
    )


class HolonomicFusedLinearCrossEntropy(nn.Module):
    """
    Drop-in replacement for `nn.Linear + nn.CrossEntropyLoss`.

    Eliminates DRAM materialization of the full [B, S, V] vocabulary logits tensor
    by fusing linear projection and cross-entropy loss into a dual-pass streaming
    token-and-vocabulary accumulation kernel.
    """

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        bias: bool = False,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        self.weight = nn.Parameter(torch.empty((vocab_size, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(vocab_size, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        h: torch.Tensor,
        targets: torch.Tensor,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward computation taking hidden representations H and integer targets.

        If `return_lse=True`, returns `(loss, lse)`.
        Otherwise, returns `loss`.
        """
        loss, lse = torch.ops.ojasx_holonomic.fused_linear_cross_entropy(
            h,
            self.weight,
            targets,
            self.bias,
            self.ignore_index,
            self.label_smoothing,
            self.reduction,
        )
        if return_lse:
            return loss, lse
        return loss

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, vocab_size={self.vocab_size}, "
            f"bias={self.bias is not None}, ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, reduction='{self.reduction}'"
        )
