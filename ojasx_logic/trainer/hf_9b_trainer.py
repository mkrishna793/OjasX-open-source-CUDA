"""
Phase 5: Up-to-9B Official HuggingFace Model Fine-Tuning & Benchmark Engine.

Supports streaming layer-by-layer 2.125-bit Dual-Rail quantization + BF16 LoRA
fine-tuning on Apple M5 (24 GB unified RAM), CUDA, or CPU, plus an optional
`safety_policy` hook compatible with OjasX-Aegis.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import resource
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from ojasx_logic.trainer.dual_rail_lora import (
    DEFAULT_LORA_TARGETS,
    quantize_module_to_dual_rail_lora,
)


RECOMMENDED_HF_MODELS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "model_id": "Qwen/Qwen2.5-1.5B",
        "params_b": 1.54,
        "gated": False,
        "notes": "Fast smoke baseline (already cached on most dev setups)",
    },
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.62,
        "gated": False,
        "notes": "Official Apache-2.0 7.6B modern instruction model (ungated)",
    },
    "qwen3_8b": {
        "model_id": "Qwen/Qwen3-8B",
        "params_b": 8.19,
        "gated": False,
        "notes": "Official modern 8.2B flagship (36 layers, GQA 32Q/8KV, ungated)",
    },
    "mistral7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.25,
        "gated": False,
        "notes": "Official 7.2B Mistral v0.3 instruct model",
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "params_b": 8.03,
        "gated": True,
        "notes": "Official Meta Llama 3.1 8B Instruct (requires HF_TOKEN)",
    },
    "gemma9b": {
        "model_id": "google/gemma-2-9b-it",
        "params_b": 9.24,
        "gated": True,
        "notes": "Official Google Gemma 2 9.2B Instruct (requires HF_TOKEN)",
    },
}


@dataclass
class TrainBenchmarkResult:
    model_id: str
    device: str
    steps: int
    seq_len: int
    batch_size: int
    layers_quantized: int
    bf16_linear_gb: float
    packed_dual_rail_gb: float
    compression_ratio: float
    mean_zero_sparsity: float
    trainable_lora_params: int
    loss_start: float
    loss_end: float
    tokens_per_sec: float
    peak_device_alloc_mb: float
    peak_device_driver_mb: float
    host_rss_mb: float
    safety_checks_run: int = 0


def _get_host_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB
    if sys.platform == "darwin":
        return rss / (1024.0 ** 2)
    return rss / 1024.0


def _detect_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _device_sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _device_mem_mb(device: str) -> tuple[float, float]:
    import torch

    if device == "mps":
        alloc = torch.mps.current_allocated_memory() / (1024.0 ** 2)
        driver = torch.mps.driver_allocated_memory() / (1024.0 ** 2)
        return alloc, driver
    if device.startswith("cuda"):
        alloc = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        resv = torch.cuda.max_memory_reserved() / (1024.0 ** 2)
        return alloc, resv
    return 0.0, 0.0


def load_and_quantize_hf_model(
    model_id: str,
    *,
    device: str = "auto",
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    group_size: int = 128,
    chunk_rows: int = 1024,
    gradient_checkpointing: bool = True,
):
    """Load an official HuggingFace CausalLM on CPU and stream-quantize to 2.125-bit Dual-Rail + BF16 LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _detect_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load onto CPU with low_cpu_mem_usage to avoid doubling weights before quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    stats = quantize_module_to_dual_rail_lora(
        model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        group_size=group_size,
        chunk_rows=chunk_rows,
        target_suffixes=DEFAULT_LORA_TARGETS,
        device=dev,
        compute_dtype_name="bfloat16",
    )

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    gc.collect()
    return model, tokenizer, stats, dev


def run_dual_rail_lora_training(
    model: Any,
    *,
    model_id: str,
    stats: Dict[str, float],
    device: str,
    steps: int = 10,
    seq_len: int = 256,
    batch_size: int = 1,
    lr: float = 2e-4,
    vocab_size: Optional[int] = None,
    input_batches: Optional[List["torch.Tensor"]] = None,
    safety_policy: Optional[Callable[[int, float], None]] = None,
) -> TrainBenchmarkResult:
    """Run real forward + backward + AdamW LoRA fine-tuning steps with telemetry & optional safety_policy hook."""
    import torch

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    if vocab_size is None:
        vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 32000))

    losses: List[float] = []
    peak_alloc = 0.0
    peak_driver = 0.0
    safety_checks = 0

    _device_sync(device)
    t_start = time.perf_counter()

    for step in range(steps):
        if input_batches is not None and len(input_batches) > 0:
            input_ids = input_batches[step % len(input_batches)].to(device)
        else:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().float().item())
        losses.append(loss_val)

        if safety_policy is not None:
            safety_policy(step, loss_val)
            safety_checks += 1

        _device_sync(device)
        a_mb, d_mb = _device_mem_mb(device)
        peak_alloc = max(peak_alloc, a_mb)
        peak_driver = max(peak_driver, d_mb)

    elapsed = max(time.perf_counter() - t_start, 1e-6)
    total_tokens = steps * batch_size * seq_len

    return TrainBenchmarkResult(
        model_id=model_id,
        device=device,
        steps=steps,
        seq_len=seq_len,
        batch_size=batch_size,
        layers_quantized=int(stats["layers_quantized"]),
        bf16_linear_gb=float(stats["bf16_linear_gb"]),
        packed_dual_rail_gb=float(stats["packed_dual_rail_gb"]),
        compression_ratio=float(stats["compression_ratio"]),
        mean_zero_sparsity=float(stats["mean_zero_sparsity"]),
        trainable_lora_params=int(stats["trainable_lora_params"]),
        loss_start=losses[0] if losses else 0.0,
        loss_end=losses[-1] if losses else 0.0,
        tokens_per_sec=total_tokens / elapsed,
        peak_device_alloc_mb=peak_alloc,
        peak_device_driver_mb=peak_driver,
        host_rss_mb=_get_host_rss_mb(),
        safety_checks_run=safety_checks,
    )


__all__ = [
    "RECOMMENDED_HF_MODELS",
    "TrainBenchmarkResult",
    "load_and_quantize_hf_model",
    "run_dual_rail_lora_training",
]
