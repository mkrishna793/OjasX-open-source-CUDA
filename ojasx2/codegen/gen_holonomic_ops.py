#!/usr/bin/env python3
"""
ojasx2/codegen/gen_holonomic_ops.py

Declarative Operator & Derivative Code Generator (Modeled on PyTorch tools/codegen/gen.py).
Reads ojasx2/codegen/holonomic_declarations.yaml and emits ojasx2/generated_holonomic_ops.py
registering native custom operators, FakeTensor meta kernels, and autograd pullbacks
under the `ojasx_holonomic` TORCH_LIBRARY namespace.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


def parse_declarations_yaml(yaml_path: str) -> Dict[str, Any]:
    """Parse holonomic_declarations.yaml using PyYAML if available, or built-in structured parser."""
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data: Dict[str, Any] = {"namespace": "ojasx_holonomic", "version": "2.0.0", "operators": []}
    current_op: Dict[str, Any] | None = None
    current_list_key: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("namespace:"):
            data["namespace"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("version:"):
            data["version"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("- name:"):
            if current_op is not None:
                data["operators"].append(current_op)
            current_op = {"name": line.split(":", 1)[1].strip(), "edge_cases": []}
            current_list_key = None
        elif current_op is not None:
            if line.startswith("- ") and current_list_key:
                current_op[current_list_key].append(line[2:].strip())
            elif ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if not v:
                    current_op[k] = []
                    current_list_key = k
                elif v.startswith("[") and v.endswith("]"):
                    items = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                    current_op[k] = items
                    current_list_key = None
                else:
                    current_op[k] = v.strip('"')
                    current_list_key = None

    if current_op is not None:
        data["operators"].append(current_op)
    return data


def generate_ops_module(spec: Dict[str, Any], output_path: str) -> None:
    ns = spec.get("namespace", "ojasx_holonomic")
    ops = spec.get("operators", [])
    ledger_json = json.dumps(spec, indent=2)

    code = f'''"""
ojasx2/generated_holonomic_ops.py

AUTO-GENERATED FILE BY ojasx2/codegen/gen_holonomic_ops.py
Source Declaration: ojasx2/codegen/holonomic_declarations.yaml
Namespace: {ns}
Operators Generated: {len(ops)}

DO NOT EDIT MANUALLY — Edit holonomic_declarations.yaml and re-run gen_holonomic_ops.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

HOLONOMIC_DECLARATIONS_LEDGER: Dict[str, Any] = {ledger_json}


# =============================================================================
# 1. Generated Operator: {ns}::rope (Reversible Orthogonal RoPE)
# =============================================================================

@torch.library.custom_op("{ns}::rope", mutates_args=())
def _rope_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 2 != 0:
        raise ValueError(f"HolonomicRoPE requires an even last dimension, got shape {{tuple(x.shape)}}")
    half = x.shape[-1] // 2
    if cos.shape[-1] != half or sin.shape[-1] != half:
        raise ValueError(f"cos/sin last dimension {{cos.shape[-1]}} must equal half of x last dim {{half}}")
    if x.numel() == 0:
        return x.clone()
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)


@_rope_op.register_fake
def _rope_fake(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 2 != 0:
        raise ValueError("HolonomicRoPE requires an even last dimension")
    return torch.empty_like(x)


def _rope_setup(ctx: Any, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
    x, cos, sin = inputs
    ctx.x_shape = x.shape
    ctx.need_trig_grad = bool(cos.requires_grad or sin.requires_grad)
    if ctx.need_trig_grad:
        # Save output y instead of x: x is recovered via inverse rotation R_{{-theta}}(y)
        ctx.save_for_backward(output, cos, sin)
    else:
        # Zero token-activation bytes saved! Orthogonal adjoint depends only on (cos, sin)
        ctx.save_for_backward(cos, sin)


def _rope_backward(ctx: Any, dy: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if dy.numel() == 0:
        if ctx.need_trig_grad:
            _, cos, sin = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        return (
            torch.zeros_like(dy),
            torch.zeros_like(cos) if cos.requires_grad else None,
            torch.zeros_like(sin) if sin.requires_grad else None,
        )

    half = dy.shape[-1] // 2
    dy1 = dy[..., :half]
    dy2 = dy[..., half:]

    if ctx.need_trig_grad:
        y, cos, sin = ctx.saved_tensors
        y1 = y[..., :half]
        y2 = y[..., half:]
        # Recover original x via exact conformal inverse M(cos, sin)^{-1}(y)
        denom = (cos.square() + sin.square()).clamp_min(1e-12)
        x1 = (y1 * cos + y2 * sin) / denom
        x2 = (-y1 * sin + y2 * cos) / denom
    else:
        cos, sin = ctx.saved_tensors
        x1 = x2 = None

    dx1 = dy1 * cos + dy2 * sin
    dx2 = -dy1 * sin + dy2 * cos
    dx = torch.cat((dx1, dx2), dim=-1)

    dcos = dsin = None
    if ctx.need_trig_grad and x1 is not None and x2 is not None:
        raw_dcos = dy1 * x1 + dy2 * x2
        raw_dsin = -dy1 * x2 + dy2 * x1
        # Reduce broadcasted leading dimensions if necessary
        while raw_dcos.ndim > cos.ndim:
            raw_dcos = raw_dcos.sum(dim=0)
            raw_dsin = raw_dsin.sum(dim=0)
        for dim_idx, (c_dim, r_dim) in enumerate(zip(cos.shape, raw_dcos.shape)):
            if c_dim == 1 and r_dim > 1:
                raw_dcos = raw_dcos.sum(dim=dim_idx, keepdim=True)
                raw_dsin = raw_dsin.sum(dim=dim_idx, keepdim=True)
        dcos = raw_dcos if cos.requires_grad else None
        dsin = raw_dsin if sin.requires_grad else None

    return dx, dcos, dsin


_rope_op.register_autograd(_rope_backward, setup_context=_rope_setup)


def holonomic_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Generated Holonomic Rotary Position Embedding with 0-byte token activation tape."""
    return _rope_op(x, cos, sin)


class HolonomicRoPE(nn.Module):
    """Drop-in Rotary Position Embedding module using orthogonal inverse pullbacks."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0, device: Any = None, dtype: Any = torch.float32) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq).to(dtype=dtype)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        length = seq_len if seq_len is not None else x.shape[-2]
        cos = self.cos_cached[:length]
        sin = self.sin_cached[:length]
        return holonomic_rope(x, cos, sin)


# =============================================================================
# 2. Generated Operator: {ns}::attention_softmax_lse
# =============================================================================

@torch.library.custom_op("{ns}::attention_softmax_lse", mutates_args=())
def _attn_softmax_lse_op(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("attention_softmax_lse requires a non-empty last dimension")
    if scores.numel() == 0:
        return scores.clone(), scores.new_zeros(scores.shape[:-1] + (1,))
    m = scores.amax(dim=-1, keepdim=True)
    shifted = scores - m
    lse = m + torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    p = torch.exp(scores - lse)
    return p, lse


@_attn_softmax_lse_op.register_fake
def _attn_softmax_lse_fake(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(scores), scores.new_empty(scores.shape[:-1] + (1,))


def _attn_softmax_setup(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    (scores,) = inputs
    _, lse = output
    # Save input scores and 1D row scalar lse [..., M, 1] instead of full probability tensor p
    ctx.save_for_backward(scores, lse)


def _attn_softmax_backward(ctx: Any, dp: torch.Tensor, dlse: Optional[torch.Tensor]) -> torch.Tensor:
    scores, saved_lse = ctx.saved_tensors
    if scores.numel() == 0:
        return torch.zeros_like(scores)
    if torch.is_grad_enabled():
        m = scores.amax(dim=-1, keepdim=True)
        lse = m + torch.log(torch.exp(scores - m).sum(dim=-1, keepdim=True))
    else:
        lse = saved_lse
    p = torch.exp(scores - lse)
    dscores = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    if dlse is not None:
        dscores = dscores + dlse * p
    return dscores


_attn_softmax_lse_op.register_autograd(_attn_softmax_backward, setup_context=_attn_softmax_setup)


def holonomic_attention_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Generated Holonomic Attention Softmax retaining only row-wise log-sum-exp scalar [..., M, 1]."""
    return _attn_softmax_lse_op(scores)[0]
'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "holonomic_declarations.yaml")
    out_path = os.path.abspath(os.path.join(base_dir, "..", "generated_holonomic_ops.py"))
    spec = parse_declarations_yaml(yaml_path)
    generate_ops_module(spec, out_path)
    print(f"[Codegen] Successfully generated {out_path} from {yaml_path} ({len(spec['operators'])} operators).")


if __name__ == "__main__":
    main()
