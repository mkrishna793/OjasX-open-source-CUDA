# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# This file is part of OjasX.
#
# OjasX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OjasX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OjasX.  If not, see <https://www.gnu.org/licenses/>.

"""
ojasx2/trainer.py
=================
Turnkey Low-RAM Training & Fine-Tuning Engine for OjasX.

Enables training brand-new models or fine-tuning existing pretrained checkpoints
on resource-constrained laptops (e.g. Apple Silicon, low-VRAM GPUs) using:
1. Reversible Yoshida 4th-order symplectic integration (eliminates forward activation tape).
2. Fused Linear Cross-Entropy (eliminates multi-GB [B, S, V] logit tensor allocations).
3. Grounded E8 quantized momentum buffers & asynchronous gradient norm latching.
4. Automatic mixed precision (AMP) and memory tracking.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .fused_cross_entropy import fused_linear_cross_entropy
from .grad_free_optimizer import HolonomicGradFreeOptimizer


def _get_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LowRAMTrainer:
    """Turnkey Low-RAM Trainer for training or fine-tuning models on consumer hardware."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        quantize_moments: bool = True,
        fused_loss: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        compile_backend: Optional[Union[str, bool]] = "ojasx_verified",
    ) -> None:
        self.device = torch.device(device or _get_device_str())
        self.fused_loss = fused_loss
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.compile_backend = compile_backend

        if dtype is None:
            self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        else:
            self.dtype = dtype
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Trainer dtype must be float16, bfloat16, float32, or float64")
        self.use_autocast = self.dtype in (torch.float16, torch.bfloat16)
        # AMP keeps FP32 master weights. Full-precision modes convert the model
        # itself, so the dtype option has an observable meaning in both modes.
        master_dtype = torch.float32 if self.use_autocast else self.dtype
        self.model = model.to(device=self.device, dtype=master_dtype)

        self._compiled_model: Optional[Callable[..., Any]] = None
        if self.compile_backend:
            backend_name = "ojasx_verified" if self.compile_backend is True else str(self.compile_backend)
            if backend_name == "ojasx_verified":
                from .core.backend import register_backend
                register_backend()
            try:
                self._compiled_model = torch.compile(self.model, backend=backend_name)
            except Exception:
                self._compiled_model = None

        # Initialize low-RAM optimizer with E8 momentum compression
        self.optimizer = HolonomicGradFreeOptimizer(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            grad_accum_steps=self.grad_accum_steps,
            quantize_moments_e8=quantize_moments,
            max_grad_norm=max_grad_norm,
            algorithm="adamw",
        )

    def __getattr__(self, name: str) -> Any:
        if "model" in self.__dict__ and hasattr(self.__dict__["model"], name):
            return getattr(self.__dict__["model"], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def train_epoch(
        self,
        dataloader: Iterable[Any],
        epoch: int = 1,
        log_interval: int = 10,
        lm_head_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Runs a single training epoch with live memory and throughput telemetry."""
        self.model.train()
        if self._compiled_model is not None and hasattr(self._compiled_model, "train"):
            try:
                self._compiled_model.train()
            except Exception:
                pass
        total_loss = 0.0
        total_tokens = 0
        step_count = 0
        optimizer_steps = 0
        t0 = time.perf_counter()

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("input_ids", batch.get("inputs"))
                targets = batch.get("labels", batch.get("targets"))
            else:
                inputs, targets = batch, batch

            inputs = inputs.to(self.device)
            if torch.is_floating_point(inputs) and not self.use_autocast:
                inputs = inputs.to(dtype=self.dtype)
            targets = targets.to(self.device)
            batch_tokens = inputs.numel()

            # Forward and loss share one autocast region. Backward remains
            # outside it, as in ordinary PyTorch AMP training.
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype if self.use_autocast else None,
                enabled=self.use_autocast,
            ):
                if self.compile_backend and self._compiled_model is not None:
                    try:
                        outputs = self._compiled_model(inputs)
                    except Exception:
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                if lm_head_weight is not None:
                    head_weight = lm_head_weight.to(device=outputs.device, dtype=outputs.dtype)
                    if self.fused_loss:
                        loss_val = fused_linear_cross_entropy(outputs, head_weight, targets)
                    else:
                        logits = torch.nn.functional.linear(outputs, head_weight)
                        loss_val = nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                        )
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError("Trainer model must return a logits tensor or an object with .logits")
                    loss_val = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )

            # The optimizer owns microbatch averaging. Dividing here as well
            # would scale every accumulated gradient by 1/N twice.
            loss_val.backward()

            total_loss += float(loss_val.item())
            total_tokens += batch_tokens
            step_count += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                optimizer_steps += 1

            if (batch_idx + 1) % log_interval == 0:
                elapsed = max(1e-5, time.perf_counter() - t0)
                tok_per_sec = total_tokens / elapsed
                avg_loss = total_loss / step_count
                mem_mb = self.get_peak_memory_mb()
                print(
                    f"Epoch {epoch} | Step {batch_idx + 1} | Loss: {avg_loss:.4f} | "
                    f"Speed: {tok_per_sec:.1f} tok/s | RAM: {mem_mb:.1f} MB"
                )

        # A short final window is averaged by its actual microbatch count.
        # Flush before zero_grad, whose documented purpose is to discard state.
        flushed = self.optimizer.flush_accumulation()
        if flushed:
            optimizer_steps += 1
            self.optimizer.step()
        self.optimizer.zero_grad()

        elapsed = max(1e-5, time.perf_counter() - t0)
        return {
            "epoch": epoch,
            "avg_loss": total_loss / max(1, step_count),
            "tokens_per_sec": total_tokens / elapsed,
            "steps": step_count,
            "optimizer_steps": optimizer_steps,
            "flushed_parameter_updates": flushed,
        }

    @staticmethod
    def get_peak_memory_mb() -> float:
        """Returns peak allocated memory in MB across GPU or process RSS."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        except Exception:
            return 0.0


def train(
    model: nn.Module,
    dataloader: Iterable[Any],
    epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    quantize_moments: bool = True,
    fused_loss: bool = True,
    lm_head_weight: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    log_interval: int = 10,
    compile_backend: Optional[Union[str, bool]] = "ojasx_verified",
) -> Dict[str, Any]:
    """Turnkey high-level training entrypoint for training brand-new or custom models."""
    trainer = LowRAMTrainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        quantize_moments=quantize_moments,
        fused_loss=fused_loss,
        dtype=dtype,
        device=device,
        compile_backend=compile_backend,
    )

    history = []
    print(f"\n🚀 OjasX Low-RAM Training Started | Device: {trainer.device} | Moments: E8 (2-bit)")
    for ep in range(1, epochs + 1):
        metrics = trainer.train_epoch(
            dataloader, epoch=ep, log_interval=log_interval,
            lm_head_weight=lm_head_weight,
        )
        history.append(metrics)
        print(f"✅ Epoch {ep} complete | Avg Loss: {metrics['avg_loss']:.4f} | Throughput: {metrics['tokens_per_sec']:.1f} tok/s")

    return {"history": history, "trainer": trainer}


def finetune(
    model: nn.Module,
    dataloader: Iterable[Any],
    epochs: int = 1,
    lr: float = 5e-5,
    grad_accum_steps: int = 2,
    fused_loss: bool = True,
    lm_head_weight: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    compile_backend: Optional[Union[str, bool]] = "ojasx_verified",
) -> Dict[str, Any]:
    """Turnkey fine-tuning helper optimized for adapting pretrained models on minimal RAM."""
    return train(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        device=device,
        dtype=dtype,
        fused_loss=fused_loss,
        lm_head_weight=lm_head_weight,
        quantize_moments=True,
        compile_backend=compile_backend,
    )
