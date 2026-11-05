"""Seamless Hugging Face PEFT LoRA integration with E8 / Linear frozen backbone."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from ojasx2.streaming_e8 import load_hf_checkpoint_e8
from ojasx2.math.lattice import E8LatticeLinear


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper around a frozen base linear layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_f = getattr(base_layer, "in_features", None)
        out_f = getattr(base_layer, "out_features", None)
        if in_f is None:
            in_f = getattr(base_layer, "in_dim", None)
        if out_f is None:
            out_f = getattr(base_layer, "out_dim", None)
        if in_f is None or out_f is None:
            raise ValueError(f"Could not infer in/out features for layer {base_layer}")

        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.rank)

        params = list(base_layer.parameters())
        device = params[0].device if params else None
        dtype = params[0].dtype if params else torch.float32
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            dtype = torch.float32

        self.lora_A = nn.Parameter(
            torch.randn(self.rank, in_f, device=device, dtype=dtype) / math.sqrt(self.rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, self.rank, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.lora_A.shape[1]

    @property
    def out_features(self) -> int:
        return self.lora_B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        drop_x = self.dropout(x)
        lora_out = (drop_x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


def attach_lora_adapters(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: float = 32.0,
    target_modules: Optional[Sequence[str]] = None,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """Attach trainable LoRA adapters to linear layers in-place and freeze base weights."""
    # First, freeze all existing parameters
    for p in model.parameters():
        p.requires_grad = False

    def _replace(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.Linear, E8LatticeLinear)):
                if not isinstance(child, LoRALinear):
                    if target_modules is None or any(t in full_name for t in target_modules):
                        wrapped = LoRALinear(
                            child,
                            rank=rank,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        )
                        setattr(parent, child_name, wrapped)
            else:
                _replace(child, full_name)

    _replace(model, "")
    return model


__all__ = ["load_hf_checkpoint_e8", "attach_lora_adapters", "LoRALinear"]
