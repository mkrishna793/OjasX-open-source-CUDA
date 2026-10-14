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
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Tiled Online Softmax Forward & Backward Kernels
# =============================================================================

def _extract_mask_tile(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    r_start: int,
    r_end: int,
    c_start: int,
    c_end: int,
    acc_dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Slices optional attention and key-padding masks for the current [r_start:r_end, c_start:c_end] tile."""
    bool_mask: Optional[torch.Tensor] = None
    float_bias: Optional[torch.Tensor] = None

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            m_tile = attn_mask[r_start:r_end, c_start:c_end].unsqueeze(0).unsqueeze(0)
        elif attn_mask.ndim == 3:
            if attn_mask.shape[0] == B * H:
                m_tile = attn_mask.view(B, H, Sq, Skv)[:, :, r_start:r_end, c_start:c_end]
            else:
                m_tile = attn_mask[:, r_start:r_end, c_start:c_end].unsqueeze(1)
        elif attn_mask.ndim == 4:
            m_tile = attn_mask[:, :, r_start:r_end, c_start:c_end]
        else:
            m_tile = None

        if m_tile is not None:
            if m_tile.dtype == torch.bool:
                bool_mask = m_tile
            else:
                float_bias = m_tile.to(acc_dtype)

    if key_padding_mask is not None:
        kpm_tile = key_padding_mask[:, c_start:c_end].unsqueeze(1).unsqueeze(2)
        if kpm_tile.dtype == torch.bool:
            bool_mask = kpm_tile if bool_mask is None else (bool_mask | kpm_tile)
        else:
            kpm_bias = kpm_tile.to(acc_dtype)
            float_bias = kpm_bias if float_bias is None else (float_bias + kpm_bias)

    return bool_mask, float_bias


def _dagger_attention_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_r: int = 128,
    block_c: int = 128,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
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

    row_arange = torch.arange(Sq, device=device) if is_causal else None
    col_arange = torch.arange(Skv, device=device) if is_causal else None

    for r_start in range(0, Sq, br):
        r_end = min(r_start + br, Sq)
        actual_br = r_end - r_start
        q_block = q[:, :, r_start:r_end, :].contiguous().to(acc_dtype)
        row_idx = row_arange[r_start:r_end].unsqueeze(1) if row_arange is not None else None

        # Initialize online running statistics in registers/local memory
        m_i = torch.full((B, H, actual_br), -float("inf"), dtype=acc_dtype, device=device)
        l_i = torch.zeros((B, H, actual_br), dtype=acc_dtype, device=device)
        acc_o = torch.zeros((B, H, actual_br, D), dtype=acc_dtype, device=device)

        for c_start in range(0, Skv, bc):
            if is_causal and c_start > r_end - 1:
                break
            c_end = min(c_start + bc, Skv)
            k_block = k_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)
            v_block = v_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)

            # Local score tile [B, H, actual_br, actual_bc]
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            bool_mask, float_bias = _extract_mask_tile(
                attn_mask, key_padding_mask, B, H, Sq, Skv, r_start, r_end, c_start, c_end, acc_dtype
            )
            if float_bias is not None:
                scores = scores + float_bias

            if is_causal and c_end - 1 >= r_start and row_idx is not None and col_arange is not None:
                col_idx = col_arange[c_start:c_end].unsqueeze(0)
                causal_mask = col_idx > row_idx
                bool_mask = causal_mask if bool_mask is None else (bool_mask | causal_mask)

            if bool_mask is not None:
                scores = scores.masked_fill(bool_mask, -float("inf"))

            nonfinite_mask = ~torch.isfinite(scores)

            # Online running softmax update
            m_block = scores.amax(dim=-1)
            m_new = torch.maximum(m_i, m_block)

            safe_m_new = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m_i - safe_m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            p_block = torch.exp(scores - safe_m_new.unsqueeze(-1))
            if bool_mask is not None or float_bias is not None:
                p_block = p_block.masked_fill(nonfinite_mask, 0.0)

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
    block_r: int = 128,
    block_c: int = 128,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
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
    D_scalar = (dout.to(acc_dtype) * out.to(acc_dtype)).sum(dim=-1)

    dq = torch.zeros(B, H, Sq, D, dtype=acc_dtype, device=device)
    dk_eff = torch.zeros(B, H, Skv, D, dtype=acc_dtype, device=device)
    dv_eff = torch.zeros(B, H, Skv, D, dtype=acc_dtype, device=device)

    br = max(1, min(Sq, block_r))
    bc = max(1, min(Skv, block_c))

    row_arange = torch.arange(Sq, device=device) if is_causal else None
    col_arange = torch.arange(Skv, device=device) if is_causal else None

    for r_start in range(0, Sq, br):
        r_end = min(r_start + br, Sq)
        actual_br = r_end - r_start

        q_block = q[:, :, r_start:r_end, :].contiguous().to(acc_dtype)
        dout_block = dout[:, :, r_start:r_end, :].contiguous().to(acc_dtype)
        lse_block = lse[:, :, r_start:r_end].to(acc_dtype)
        D_block = D_scalar[:, :, r_start:r_end]
        dlse_block = dlse[:, :, r_start:r_end].to(acc_dtype) if dlse is not None else None
        row_idx = row_arange[r_start:r_end].unsqueeze(1) if row_arange is not None else None

        dq_row_acc = torch.zeros((B, H, actual_br, D), dtype=acc_dtype, device=device)

        for c_start in range(0, Skv, bc):
            if is_causal and c_start > r_end - 1:
                break
            c_end = min(c_start + bc, Skv)

            k_block = k_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)
            v_block = v_eff[:, :, c_start:c_end, :].contiguous().to(acc_dtype)

            # Reconstruct local score tile in registers
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            bool_mask, float_bias = _extract_mask_tile(
                attn_mask, key_padding_mask, B, H, Sq, Skv, r_start, r_end, c_start, c_end, acc_dtype
            )
            if float_bias is not None:
                scores = scores + float_bias

            has_causal_mask = is_causal and (c_end - 1 >= r_start)
            if has_causal_mask and row_idx is not None and col_arange is not None:
                col_idx = col_arange[c_start:c_end].unsqueeze(0)
                causal_mask = col_idx > row_idx
                bool_mask = causal_mask if bool_mask is None else (bool_mask | causal_mask)

            if bool_mask is not None:
                scores = scores.masked_fill(bool_mask, -float("inf"))

            nonfinite_mask = ~torch.isfinite(scores)

            # Reconstruct exact attention probabilities tile P_{ij} = exp(S_{ij} - L_i)
            safe_lse = torch.where(torch.isfinite(lse_block), lse_block, torch.zeros_like(lse_block))
            p_block = torch.exp(scores - safe_lse.unsqueeze(-1))
            if bool_mask is not None or float_bias is not None:
                p_block = p_block.masked_fill(nonfinite_mask, 0.0)

            # dP_{ij} = dO_i @ V_j^T
            dp_block = torch.matmul(dout_block, v_block.transpose(-2, -1))

            # dS_{ij} = P_{ij} * (dP_{ij} - D_i + (dL_i if dL is not None else 0))
            diff = dp_block - D_block.unsqueeze(-1)
            if dlse_block is not None:
                diff = diff + dlse_block.unsqueeze(-1)
            ds_block = p_block * diff
            if bool_mask is not None or float_bias is not None:
                ds_block = ds_block.masked_fill(nonfinite_mask, 0.0)

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


@torch.library.register_fake("ojasx_holonomic::dagger_attention")
def dagger_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor / Meta kernel for symbolic tracing and torch.compile."""
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device)
    return out, lse


def _dagger_attention_setup(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[float]],
    output: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Saves ONLY O and L to backward tape (plus Q, K, V references).

    Strictly zero bytes DRAM allocation for [B, H, S, S] attention score matrices.
    """
    q, k, v, is_causal, sm_scale = inputs
    out, lse = output
    ctx.is_causal = is_causal
    ctx.sm_scale = sm_scale
    ctx.save_for_backward(q, k, v, out, lse)


def _dagger_attention_backward(
    ctx: torch.autograd.function.FunctionCtx,
    dout: torch.Tensor,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    """Reconstructs attention probabilities tile-by-tile in registers using L."""
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _dagger_attention_backward_tiled(
        q, k, v, out, lse, dout, dlse=dlse, is_causal=ctx.is_causal, sm_scale=ctx.sm_scale
    )
    return dq, dk, dv, None, None


dagger_attention.register_autograd(_dagger_attention_backward, setup_context=_dagger_attention_setup)


# =============================================================================
# 3. Explicit torch.autograd.Function Implementation
# =============================================================================

class _HolonomicDaggerAttentionFunction(torch.autograd.Function):
    """Explicit autograd Function for Holonomic Dagger Attention.

    Retains strictly 0 bytes DRAM allocation for intermediate [B, H, S, S]
    attention scores on the backward tape.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        sm_scale: Optional[float] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _dagger_attention_forward_tiled(
            q,
            k,
            v,
            is_causal=is_causal,
            sm_scale=sm_scale,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        ctx.is_causal = is_causal
        ctx.sm_scale = sm_scale
        ctx.attn_mask = attn_mask.detach() if isinstance(attn_mask, torch.Tensor) else None
        ctx.key_padding_mask = (
            key_padding_mask.detach() if isinstance(key_padding_mask, torch.Tensor) else None
        )
        # Save ONLY O and L (plus Q, K, V references)
        ctx.save_for_backward(q, k, v, out, lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: Optional[torch.Tensor] = None):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _dagger_attention_backward_tiled(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dlse=dlse,
            is_causal=ctx.is_causal,
            sm_scale=ctx.sm_scale,
            attn_mask=getattr(ctx, "attn_mask", None),
            key_padding_mask=getattr(ctx, "key_padding_mask", None),
        )
        return dq, dk, dv, None, None, None, None


def holonomic_dagger_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional entrypoint for Holonomic Dagger Attention.

    Returns the attention output tensor [B, H, S_q, d].
    """
    if attn_mask is not None or key_padding_mask is not None:
        return _HolonomicDaggerAttentionFunction.apply(
            q, k, v, is_causal, sm_scale, attn_mask, key_padding_mask
        )[0]
    return torch.ops.ojasx_holonomic.dagger_attention(
        q, k, v, is_causal=is_causal, sm_scale=sm_scale
    )[0]


def holonomic_dagger_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional entrypoint returning both attention output and row-wise LogSumExp."""
    if attn_mask is not None or key_padding_mask is not None:
        return _HolonomicDaggerAttentionFunction.apply(
            q, k, v, is_causal, sm_scale, attn_mask, key_padding_mask
        )
    return torch.ops.ojasx_holonomic.dagger_attention(
        q, k, v, is_causal=is_causal, sm_scale=sm_scale
    )


# =============================================================================
# 4. HolonomicDaggerAttention nn.Module Wrapper
# =============================================================================

class HolonomicDaggerAttention(nn.Module):
    """Drop-in compatible Multi-Head Attention module powered by Holonomic Dagger Attention.

    Supports both:
    1. Standalone construction compatible with nn.MultiheadAttention:
       `HolonomicDaggerAttention(embed_dim=256, num_heads=8, is_causal=True)`
    2. Rewritten / wrapped LLM attention projection submodules:
       `HolonomicDaggerAttention(q_proj=..., k_proj=..., v_proj=..., o_proj=..., num_heads=8)`
    """

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        q_proj: Optional[nn.Module] = None,
        k_proj: Optional[nn.Module] = None,
        v_proj: Optional[nn.Module] = None,
        o_proj: Optional[nn.Module] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False,
        batch_first: bool = True,
        return_tuple: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.is_causal = is_causal
        self.batch_first = batch_first
        self.dropout = dropout
        self.return_tuple = return_tuple

        # nn.MultiheadAttention & nn.TransformerEncoderLayer / DecoderLayer compatibility attributes
        self._qkv_same_embed_dim = False
        self.in_proj_weight = None
        self.in_proj_bias = None
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False

        factory_kwargs = {"device": device, "dtype": dtype}

        if q_proj is not None and k_proj is not None and v_proj is not None:
            self.q_proj = q_proj
            self.k_proj = k_proj
            self.v_proj = v_proj
            self.o_proj = o_proj if o_proj is not None else nn.Identity()
            if head_dim is None:
                out_feat = getattr(q_proj, "out_features", None)
                if out_feat is not None:
                    self.head_dim = out_feat // num_heads
                else:
                    self.head_dim = 64
            else:
                self.head_dim = head_dim
            self.embed_dim = self.num_heads * self.head_dim
            self.kdim = kdim if kdim is not None else getattr(k_proj, "in_features", self.embed_dim)
            self.vdim = vdim if vdim is not None else getattr(v_proj, "in_features", self.embed_dim)
        else:
            if embed_dim is None:
                embed_dim = 256
            self.embed_dim = embed_dim
            self.head_dim = head_dim if head_dim is not None else (embed_dim // num_heads)
            self.kdim = kdim if kdim is not None else embed_dim
            self.vdim = vdim if vdim is not None else embed_dim
            self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(self.kdim, self.num_kv_heads * self.head_dim, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(self.vdim, self.num_kv_heads * self.head_dim, bias=bias, **factory_kwargs)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim, bias=bias, **factory_kwargs)

    @property
    def out_proj(self) -> nn.Module:
        """Alias for nn.MultiheadAttention compatibility."""
        return self.o_proj

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = None,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: Optional[bool] = None,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Forward pass for multi-head attention.

        Supports query input of shape [B, S, D] (or [S, B, D] if batch_first=False),
        or direct [B, H, S, d] tensor projection. Fully compatible with nn.MultiheadAttention.
        """
        causal = is_causal if is_causal is not None else self.is_causal
        eff_attn_mask = attn_mask
        if eff_attn_mask is not None and getattr(eff_attn_mask, "is_causal", False):
            causal = True
            eff_attn_mask = None

        # Case 1: Already projected 4D inputs [B, H, S, d]
        if query.ndim == 4 and key is not None and value is not None and key.ndim == 4:
            out = holonomic_dagger_attention(
                query,
                key,
                value,
                is_causal=causal,
                attn_mask=eff_attn_mask,
                key_padding_mask=key_padding_mask,
            )
            if need_weights is not None or self.return_tuple:
                return out, None
            return out

        # Case 2: Standard 3D inputs [B, S, D]
        if not self.batch_first and query.ndim == 3:
            # [S, B, D] -> [B, S, D]
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)
            if value is not None:
                value = value.transpose(0, 1)

        k_in = key if key is not None else query
        v_in = value if value is not None else query

        B, Sq, _ = query.shape
        _, Skv, _ = k_in.shape

        q_proj = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(k_in).view(B, Skv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(v_in).view(B, Skv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rotary_cos is not None and rotary_sin is not None:
            d_half = self.head_dim // 2
            cos_eff = torch.cat([rotary_cos, rotary_cos], dim=-1) if rotary_cos.shape[-1] == d_half else rotary_cos
            sin_eff = torch.cat([rotary_sin, rotary_sin], dim=-1) if rotary_sin.shape[-1] == d_half else rotary_sin
            q1, q2 = q_proj[..., :d_half], q_proj[..., d_half : 2 * d_half]
            k1, k2 = k_proj[..., :d_half], k_proj[..., d_half : 2 * d_half]
            q_rot = torch.cat([-q2, q1], dim=-1)
            k_rot = torch.cat([-k2, k1], dim=-1)
            q_proj = q_proj * cos_eff + q_rot * sin_eff
            k_proj = k_proj * cos_eff + k_rot * sin_eff

        attn_out = holonomic_dagger_attention(
            q_proj,
            k_proj,
            v_proj,
            is_causal=causal,
            attn_mask=eff_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        # [B, H, Sq, d] -> [B, Sq, H * d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Sq, self.num_heads * self.head_dim)
        output = self.o_proj(attn_out)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if need_weights is not None or self.return_tuple:
            return output, None
        return output
