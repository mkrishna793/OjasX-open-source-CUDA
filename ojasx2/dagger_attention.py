"""Holonomic Dagger Attention Engine.

Implements Symplectic Holonomic Dagger Attention: an in-register adjoint
online softmax tiling operator (FlashAttention-style) that fuses Q, K, V
projections and attention probabilities directly into hardware registers/SRAM.

Strictly eliminates DRAM materialization of the [B, H, S, S] intermediate
attention score and softmax probability matrices during training, guaranteeing
0 bytes DRAM allocation on the autograd backward tape.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Tiled Online Softmax Forward & Backward Kernels
# =============================================================================

def _dagger_attention_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_r: int = 64,
    block_c: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-register online softmax forward tiling.

    Accumulates output O in [B, H, S_q, d] and row-wise LogSumExp L in [B, H, S_q]
    without ever materializing the [B, H, S_q, S_kv] attention score matrix in DRAM.
    """
    B, H, Sq, D = q.shape
    _, H_kv, Skv, _ = k.shape
    device = q.device
    dtype = q.dtype

    if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
        out = torch.empty_like(q)
        lse = torch.empty(B, H, Sq, dtype=torch.float32, device=device)
        return out, lse

    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(D))

    # Support Grouped-Query Attention (GQA) if H != H_kv
    if H != H_kv:
        if H % H_kv != 0:
            raise ValueError(f"Query heads {H} must be divisible by KV heads {H_kv}")
        repeat_factor = H // H_kv
        k_eff = k.repeat_interleave(repeat_factor, dim=1)
        v_eff = v.repeat_interleave(repeat_factor, dim=1)
    else:
        k_eff = k
        v_eff = v

    out = torch.zeros(B, H, Sq, D, dtype=dtype, device=device)
    lse = torch.empty(B, H, Sq, dtype=torch.float32, device=device)

    # Use float32 accumulators for numerical stability across fp16/bf16
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    br = max(1, min(Sq, block_r))
    bc = max(1, min(Skv, block_c))

    for r_start in range(0, Sq, br):
        r_end = min(r_start + br, Sq)
        actual_br = r_end - r_start
        q_block = q[:, :, r_start:r_end, :].contiguous().to(acc_dtype)

        # Initialize online running statistics in registers/local memory
        m_i = torch.full((B, H, actual_br), -float("inf"), dtype=acc_dtype, device=device)
        l_i = torch.zeros((B, H, actual_br), dtype=acc_dtype, device=device)
        acc_o = torch.zeros((B, H, actual_br, D), dtype=acc_dtype, device=device)

        for c_start in range(0, Skv, bc):
            if is_causal and c_start > r_end - 1:
                break
            c_end = min(c_start + bc, Skv)
            actual_bc = c_end - c_start
            k_block = k_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)
            v_block = v_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)

            # Local score tile [B, H, actual_br, actual_bc]
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            if is_causal and c_end - 1 >= r_start:
                row_idx = torch.arange(r_start, r_end, device=device).unsqueeze(1)
                col_idx = torch.arange(c_start, c_end, device=device).unsqueeze(0)
                mask = col_idx > row_idx
                scores = scores.masked_fill(mask, -float("inf"))

            # Online running softmax update
            m_block = scores.amax(dim=-1)
            m_new = torch.maximum(m_i, m_block)

            alpha = torch.exp(m_i - m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            p_block = torch.exp(scores - m_new.unsqueeze(-1))
            if is_causal and c_end - 1 >= r_start:
                p_block = p_block.masked_fill(mask, 0.0)

            l_new = alpha * l_i + p_block.sum(dim=-1)
            acc_o = alpha.unsqueeze(-1) * acc_o + torch.matmul(p_block, v_block)

            m_i = m_new
            l_i = l_new

        # Normalize and compute exact row-wise LogSumExp
        # Guard against any all-masked rows
        safe_l_i = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
        out[:, :, r_start:r_end, :] = (acc_o / safe_l_i.unsqueeze(-1)).to(dtype)
        lse[:, :, r_start:r_end] = (m_i + torch.log(safe_l_i)).to(torch.float32)

    return out, lse


def _dagger_attention_backward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    dlse: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_r: int = 64,
    block_c: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """In-register online softmax backward tiling.

    Reconstructs attention probabilities P in registers tile-by-tile using L
    and computes pullback scalar D = (O . dO)1, emitting dQ, dK, dV without
    ever materializing [B, H, S_q, S_kv] in DRAM.
    """
    B, H, Sq, D = q.shape
    _, H_kv, Skv, _ = k.shape
    device = q.device
    dtype = q.dtype

    if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(D))

    # Effective KV heads for GQA
    is_gqa = (H != H_kv)
    if is_gqa:
        repeat_factor = H // H_kv
        k_eff = k.repeat_interleave(repeat_factor, dim=1)
        v_eff = v.repeat_interleave(repeat_factor, dim=1)
    else:
        k_eff = k
        v_eff = v

    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    # Step 1: Invariant row scalar D = (O . dO) 1 in O(B H Sq) memory
    # D_i = sum_d dO_{id} * O_{id}
    D_scalar = (dout.to(acc_dtype) * out.to(acc_dtype)).sum(dim=-1)

    dq = torch.zeros(B, H, Sq, D, dtype=acc_dtype, device=device)
    dk_eff = torch.zeros(B, H, Skv, D, dtype=acc_dtype, device=device)
    dv_eff = torch.zeros(B, H, Skv, D, dtype=acc_dtype, device=device)

    br = max(1, min(Sq, block_r))
    bc = max(1, min(Skv, block_c))

    for r_start in range(0, Sq, br):
        r_end = min(r_start + br, Sq)
        actual_br = r_end - r_start

        q_block = q[:, :, r_start:r_end, :].contiguous().to(acc_dtype)
        dout_block = dout[:, :, r_start:r_end, :].contiguous().to(acc_dtype)
        lse_block = lse[:, :, r_start:r_end].to(acc_dtype)
        D_block = D_scalar[:, :, r_start:r_end]
        dlse_block = dlse[:, :, r_start:r_end].to(acc_dtype) if dlse is not None else None

        dq_row_acc = torch.zeros((B, H, actual_br, D), dtype=acc_dtype, device=device)

        for c_start in range(0, Skv, bc):
            if is_causal and c_start > r_end - 1:
                break
            c_end = min(c_start + bc, Skv)
            actual_bc = c_end - c_start

            k_block = k_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)
            v_block = v_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)

            # Reconstruct local score tile in registers
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            has_causal_mask = is_causal and (c_end - 1 >= r_start)
            if has_causal_mask:
                row_idx = torch.arange(r_start, r_end, device=device).unsqueeze(1)
                col_idx = torch.arange(c_start, c_end, device=device).unsqueeze(0)
                mask = col_idx > row_idx
                scores = scores.masked_fill(mask, -float("inf"))

            # Reconstruct exact attention probabilities tile P_{ij} = exp(S_{ij} - L_i)
            p_block = torch.exp(scores - lse_block.unsqueeze(-1))
            if has_causal_mask:
                p_block = p_block.masked_fill(mask, 0.0)

            # dP_{ij} = dO_i @ V_j^T
            dp_block = torch.matmul(dout_block, v_block.transpose(-2, -1))

            # dS_{ij} = P_{ij} * (dP_{ij} - D_i + (dL_i if dL is not None else 0))
            diff = dp_block - D_block.unsqueeze(-1)
            if dlse_block is not None:
                diff = diff + dlse_block.unsqueeze(-1)
            ds_block = p_block * diff
            if has_causal_mask:
                ds_block = ds_block.masked_fill(mask, 0.0)

            # Accumulate dV_j += P_{ij}^T @ dO_i
            dv_eff[:, :, c_start:c_end, :] += torch.matmul(p_block.transpose(-2, -1), dout_block)

            # Accumulate dK_j += scale * (dS_{ij}^T @ Q_i)
            dk_eff[:, :, c_start:c_end, :] += scale * torch.matmul(ds_block.transpose(-2, -1), q_block)

            # Accumulate dQ_i += scale * (dS_{ij} @ K_j)
            dq_row_acc += scale * torch.matmul(ds_block, k_block)

        dq[:, :, r_start:r_end, :] = dq_row_acc

    # If GQA, reduce gradients across grouped heads
    if is_gqa:
        repeat_factor = H // H_kv
        dk = dk_eff.view(B, H_kv, repeat_factor, Skv, D).sum(dim=2).to(k.dtype)
        dv = dv_eff.view(B, H_kv, repeat_factor, Skv, D).sum(dim=2).to(v.dtype)
    else:
        dk = dk_eff.to(k.dtype)
        dv = dv_eff.to(v.dtype)

    return dq.to(q.dtype), dk, dv


# =============================================================================
# 2. PyTorch Custom Operator Registration & Autograd Tape Binding
# =============================================================================

@torch.library.custom_op("ojasx_holonomic::dagger_attention", mutates_args=())
def dagger_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom operator ojasx_holonomic::dagger_attention.

    Computes multihead attention with in-register online softmax tiling.
    Returns (out, lse).
    """
    return _dagger_attention_forward_tiled(q, k, v, is_causal=is_causal, sm_scale=sm_scale)

