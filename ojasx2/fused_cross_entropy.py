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

def _streaming_fused_lce_forward(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    tile_tok: int = 128,
    tile_voc: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dual-pass streaming forward accumulation over token and vocabulary tiles."""
    d = h.shape[-1]
    m = h.numel() // d
    v = w.shape[0]

    if w.shape[1] != d:
        raise ValueError(f"Weight feature dimension {w.shape[1]} does not match hidden dimension {d}")
    if bias is not None and bias.shape[0] != v:
        raise ValueError(f"Bias dimension {bias.shape[0]} does not match vocabulary size {v}")

    # Zero-element handling
    if m == 0:
        loss = (
            torch.zeros((), dtype=h.dtype, device=h.device)
            if reduction != "none"
            else torch.zeros(targets.shape, dtype=h.dtype, device=h.device)
        )
        lse = torch.zeros(targets.shape, dtype=torch.float32, device=h.device)
        return loss, lse

    compute_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    h_2d = h.reshape(m, d)
    t_1d = targets.reshape(m)

    # Valid mask without data-dependent python branches
    mask = (t_1d != ignore_index) & (t_1d >= 0) & (t_1d < v)
    clamped_t = torch.clamp(t_1d, 0, v - 1)

    # Target dot-product z_{m, y_m} computed in O(M x D) without logits tensor
    w_target = w[clamped_t].to(compute_dtype)
    z_target = (h_2d.to(compute_dtype) * w_target).sum(dim=-1)
    if bias is not None:
        z_target = z_target + bias[clamped_t].to(compute_dtype)
    z_target = torch.where(mask, z_target, torch.zeros_like(z_target))

    lse = torch.zeros(m, dtype=compute_dtype, device=h.device)
    sum_z = torch.zeros(m, dtype=compute_dtype, device=h.device) if label_smoothing > 0.0 else None

    # Dual-pass tile streaming loop: transient memory <= O(tile_tok x tile_voc)
    for tok_s in range(0, m, tile_tok):
        tok_e = min(tok_s + tile_tok, m)
        h_chunk = h_2d[tok_s:tok_e].to(compute_dtype)
        m_chunk = torch.full((tok_e - tok_s,), -float("inf"), dtype=compute_dtype, device=h.device)
        s_chunk = torch.zeros(tok_e - tok_s, dtype=compute_dtype, device=h.device)
        sum_z_chunk = (
            torch.zeros(tok_e - tok_s, dtype=compute_dtype, device=h.device)
            if label_smoothing > 0.0
            else None
        )

        for v_s in range(0, v, tile_voc):
            v_e = min(v_s + tile_voc, v)
            w_chunk = w[v_s:v_e].to(compute_dtype)
            z_tile = torch.matmul(h_chunk, w_chunk.t())
            if bias is not None:
                z_tile = z_tile + bias[v_s:v_e].to(compute_dtype)

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
    tile_tok: int = 128,
    tile_voc: int = 4096,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, Optional[torch.Tensor], None, None, None]:
    """Streaming chunked pullback directly to dH and dW without materializing logits."""
    d = h.shape[-1]
    m = h.numel() // d
    v = w.shape[0]

    needs_dh = h.requires_grad
    needs_dw = w.requires_grad
    needs_db = bias is not None and bias.requires_grad

    if m == 0:
        dh = torch.zeros_like(h) if needs_dh else None
        dw = torch.zeros_like(w) if needs_dw else None
        db = torch.zeros_like(bias) if needs_db else None
        return dh, dw, None, db, None, None, None

    compute_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    h_2d = h.reshape(m, d).to(compute_dtype)
    t_1d = targets.reshape(m)
    lse_1d = lse.reshape(m).to(compute_dtype)

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
    db = torch.zeros_like(bias, dtype=compute_dtype) if needs_db else None

    # Step 1: Target subtraction term: - c_m (1 - eps) * 1[v = y_m]
    eps = label_smoothing
    target_scale = c * (1.0 - eps)
    if needs_dh:
        dh_2d -= target_scale.unsqueeze(-1) * w[clamped_t].to(compute_dtype)
    if needs_dw:
        dw.index_add_(0, clamped_t, -(target_scale.unsqueeze(-1) * h_2d))
    if needs_db:
        db.index_add_(0, clamped_t, -target_scale)

    # Step 2: Uniform label smoothing shift: - c_m * (eps / V)
    if eps > 0.0:
        c_sum = c.sum()
        if needs_dh:
            w_mean = w.to(compute_dtype).mean(dim=0)
            dh_2d -= (c * eps).unsqueeze(-1) * w_mean
        if needs_dw:
            h_weighted = (c.unsqueeze(-1) * h_2d).sum(dim=0)
            dw -= (eps / float(v)) * h_weighted.unsqueeze(0).expand(v, d)
        if needs_db:
            db -= (eps / float(v)) * c_sum

    # Step 3: Softmax streaming accumulation
    for tok_s in range(0, m, tile_tok):
        tok_e = min(tok_s + tile_tok, m)
        h_chunk = h_2d[tok_s:tok_e]
        alpha_chunk = alpha[tok_s:tok_e]
        lse_chunk = lse_1d[tok_s:tok_e]

        for v_s in range(0, v, tile_voc):
            v_e = min(v_s + tile_voc, v)
            w_chunk = w[v_s:v_e].to(compute_dtype)
            z_tile = torch.matmul(h_chunk, w_chunk.t())
            if bias is not None:
                z_tile = z_tile + bias[v_s:v_e].to(compute_dtype)

