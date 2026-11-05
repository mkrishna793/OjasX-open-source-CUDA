"""Turnkey Drop-In Public API for OjasX3."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from ojasx2.core.backend import backend, register_backend, OjasXBackend, BackendConfig
from ojasx3.trainer.trainer import LowRAMTrainer, train as _low_ram_train, finetune as _low_ram_finetune
from ojasx3.trainer.optimizer import HolonomicGradFreeOptimizer
from ojasx3.lattice.streaming import load_hf_checkpoint_e8
from ojasx3.lattice.layers import pack_model_e8
from ojasx2 import optimize as _legacy_optimize, prepare as _legacy_prepare


def compile(
    model: nn.Module,
    *,
    backend: Optional[Union[str, Callable]] = "ojasx3",
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    options: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Drop-in 1-line PyTorch compiler using certified OjasX3 equality saturation backend."""
    register_backend()
    return torch.compile(
        model,
        backend=backend or "ojasx3",
        fullgraph=fullgraph,
        dynamic=dynamic,
        options=options,
    )


def train(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    compile_backend: Optional[Union[str, bool]] = "ojasx3",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train any PyTorch model from scratch in 1 line with automated low-RAM scaling."""
    res = _low_ram_train(
        model,
        dataset,
        epochs=epochs,
        lr=lr,
        device=device,
        dtype=dtype,
        compile_backend=compile_backend,
        **kwargs,
    )
    if isinstance(res, dict):
        res["model"] = model
    return res


def finetune(
    checkpoint: Union[str, Path, nn.Module],
    dataset: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
    *,
    bitrate: float = 1.5,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lr: float = 2e-4,
    epochs: int = 1,
    device: str = "auto",
    **kwargs: Any,
) -> Union[Tuple[nn.Module, HolonomicGradFreeOptimizer], Dict[str, Any]]:
    """Fine-tune any large Hugging Face checkpoint or model on consumer RAM in 1 line."""
    if dataset is None:
        if isinstance(checkpoint, (str, Path)):
            model, load_report = load_hf_checkpoint_e8(
                checkpoint,
                bitrate=bitrate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                **kwargs,
            )
        else:
            from ojasx3.trainer.peft import attach_lora_adapters
            model = attach_lora_adapters(checkpoint, rank=lora_rank, lora_alpha=lora_alpha)
        opt = HolonomicGradFreeOptimizer(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        return model, opt

    return _low_ram_finetune(
        checkpoint,
        dataset,
        bitrate=bitrate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lr=lr,
        epochs=epochs,
        device=device,
        **kwargs,
    )


def optimize(
    model: nn.Module,
    quantize: Optional[Union[str, float]] = None,
    attention: str = "fast",
    group_size: int = 32,
    compile_backend: Optional[Union[str, bool]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Transform any standard PyTorch model in-place with DaggerAttention and E8 packing."""
    return _legacy_optimize(
        model,
        quantize=quantize,
        attention=attention,
        group_size=group_size,
        compile_backend=compile_backend,
        **kwargs,
    )


def prepare(
    model: nn.Module,
    lr: float = 1e-4,
    optimizer: str = "adamw",
    quantize: Optional[Union[str, float]] = None,
    attention: str = "fast",
    quantize_moments_e8: bool = True,
    grad_accum_steps: int = 1,
    weight_decay: float = 0.01,
    compile_backend: Optional[Union[str, bool]] = None,
    **kwargs: Any,
) -> Tuple[nn.Module, HolonomicGradFreeOptimizer]:
    """1-Line turn-key preparation of model and optimizer for production training."""
    return _legacy_prepare(
        model,
        lr=lr,
        optimizer=optimizer,
        quantize=quantize,
        attention=attention,
        quantize_moments_e8=quantize_moments_e8,
        grad_accum_steps=grad_accum_steps,
        weight_decay=weight_decay,
        compile_backend=compile_backend,
        **kwargs,
    )


__all__ = [
    "compile",
    "train",
    "finetune",
    "optimize",
    "prepare",
]
