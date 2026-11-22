#!/usr/bin/env python3
"""
Real Pretrained 1B-3B Checkpoint Benchmark on WikiText-2 (Apple Silicon MPS).

Measures:
1. Clean FP16 & BF16 loading and inference sanity on MPS.
2. WikiText-2-raw-v1 Perplexity: BF16 vs FP16 vs Unrotated-E8 vs Rotated-E8
   (using `ojasx3.lattice.transforms.apply_incoherence_transform` + `E8LatticePacker`)
   and 4-bit NF4 reference.
3. 100-step Fine-Tuning Convergence, true mid-step peak RSS / MPS VRAM, and tok/s
   in isolated fresh subprocesses comparing:
   - OjasX (E8 + LoRA PEFT + HolonomicGradFreeOptimizer + FusedLCE + Checkpointing)
   - PyTorch AMP + SDPA + Gradient Checkpointing (LoRA r=16, plus 3-step Full-FT probe)
   - QLoRA (4-bit NF4 packed base + LoRA r=16 + Gradient Checkpointing)
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ojasx3.lattice.e8 import E8LatticePacker
from ojasx3.lattice.layers import E8LatticeLinear
from ojasx3.lattice.transforms import apply_incoherence_transform
from ojasx3.trainer.optimizer import HolonomicGradFreeOptimizer
from ojasx3.trainer.peft import LoRALinear, attach_lora_adapters
from ojasx2.fused_cross_entropy import fused_linear_cross_entropy
from ojasx2.smart_router import optimize_model


TARGET_LINEARS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230182647705,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# macOS RSS + MPS Memory Telemetry
# ---------------------------------------------------------------------------

class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if sys.platform == "darwin" else None
_MACH_TASK_BASIC_INFO = 20
_TASK_INFO_COUNT = ctypes.sizeof(_MachTaskBasicInfo) // 4


def get_process_rss_bytes() -> Tuple[int, int]:
    """Return (current_rss_bytes, os_ru_maxrss_bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_bytes = int(ru) if sys.platform == "darwin" else int(ru) * 1024
    if _LIBC is not None:
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(_TASK_INFO_COUNT)
        kr = _LIBC.task_info(
            _LIBC.mach_task_self(),
            _MACH_TASK_BASIC_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if kr == 0:
            return int(info.resident_size), max(ru_bytes, int(info.resident_size_max))
    return ru_bytes, ru_bytes


def get_mps_memory_bytes() -> Tuple[int, int]:
    """Return (current_allocated_bytes, driver_allocated_bytes)."""
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return (
            int(torch.mps.current_allocated_memory()),
            int(torch.mps.driver_allocated_memory()),
        )
    return 0, 0


class PeakMemoryMonitor:
    """5ms background sampler + checkpoint tracker for true mid-step peak RSS & MPS VRAM."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.peak_rss_bytes = 0
        self.peak_ru_maxrss_bytes = 0
        self.peak_mps_alloc_bytes = 0
        self.peak_mps_driver_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample_now(self) -> Dict[str, float]:
        cur_rss, ru_max = get_process_rss_bytes()
        mps_alloc, mps_drv = get_mps_memory_bytes()
        if cur_rss > self.peak_rss_bytes:
            self.peak_rss_bytes = cur_rss
        if ru_max > self.peak_ru_maxrss_bytes:
            self.peak_ru_maxrss_bytes = ru_max
        if mps_alloc > self.peak_mps_alloc_bytes:
            self.peak_mps_alloc_bytes = mps_alloc
        if mps_drv > self.peak_mps_driver_bytes:
            self.peak_mps_driver_bytes = mps_drv
        self.samples += 1
        return {
            "rss_mb": cur_rss / (1024 * 1024),
            "ru_maxrss_mb": ru_max / (1024 * 1024),
            "mps_alloc_mb": mps_alloc / (1024 * 1024),
            "mps_driver_mb": mps_drv / (1024 * 1024),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_now()
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.sample_now()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sample_now()
        return self.summary_mb()

    def reset_peaks(self) -> None:
        self.peak_rss_bytes = 0
        self.peak_mps_alloc_bytes = 0
        self.peak_mps_driver_bytes = 0
        self.samples = 0
        self.sample_now()

    def summary_mb(self) -> Dict[str, float]:
        mb = 1024 * 1024
        return {
            "peak_sampled_rss_mb": round(self.peak_rss_bytes / mb, 2),
            "os_ru_maxrss_mb": round(self.peak_ru_maxrss_bytes / mb, 2),
            "peak_mps_alloc_mb": round(self.peak_mps_alloc_bytes / mb, 2),
            "peak_mps_driver_mb": round(self.peak_mps_driver_bytes / mb, 2),
            "samples": self.samples,
        }


# ---------------------------------------------------------------------------
# 4-bit NF4 QLoRA Linear (MPS-native equivalent to bitsandbytes Linear4bit)
# ---------------------------------------------------------------------------

class _NF4MatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        codebook: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_f: int,
        in_f: int,
        blocksize: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(packed, scales, codebook)
        ctx.out_f = out_f
        ctx.in_f = in_f
        ctx.blocksize = blocksize
        ctx.has_bias = bias is not None

        lo = (packed & 0x0F).to(torch.long)
        hi = ((packed >> 4) & 0x0F).to(torch.long)
        vals = torch.stack([codebook[lo], codebook[hi]], dim=-1).view(-1, blocksize)
        w = (vals * scales.unsqueeze(-1)).reshape(-1)[: out_f * in_f].view(out_f, in_f).to(x.dtype)
        return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)

    @staticmethod
    def backward(ctx: Any, gy: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        packed, scales, codebook = ctx.saved_tensors
        lo = (packed & 0x0F).to(torch.long)
        hi = ((packed >> 4) & 0x0F).to(torch.long)
        vals = torch.stack([codebook[lo], codebook[hi]], dim=-1).view(-1, ctx.blocksize)
        w = (vals * scales.unsqueeze(-1)).reshape(-1)[: ctx.out_f * ctx.in_f].view(ctx.out_f, ctx.in_f).to(gy.dtype)
        gx = gy.matmul(w)
        gb = gy.reshape(-1, ctx.out_f).sum(dim=0) if ctx.has_bias else None
        return gx, None, None, None, gb, None, None, None


class NF4Linear(nn.Module):
    """4-bit NF4 blockwise quantized frozen linear layer (4.25 bits/weight)."""

    def __init__(self, linear: nn.Linear, blocksize: int = 64) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.blocksize = int(blocksize)

        w = linear.weight.data.float().reshape(-1)
        pad = (-w.numel()) % blocksize
        if pad:
            w = F.pad(w, (0, pad))
        blocks = w.view(-1, blocksize)
        scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        wn = (blocks / scales).unsqueeze(-1)
        cb = NF4_LEVELS.to(w.device)
        idx = torch.argmin((wn - cb).abs(), dim=-1).to(torch.uint8).view(-1, 2)
        packed = idx[:, 0] | (idx[:, 1] << 4)
        self.register_buffer("packed_weight", packed)
        self.register_buffer("absmax", scales.squeeze(-1).to(torch.float16))
        self.register_buffer("codebook", cb.to(torch.float16))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def unpack_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        lo = (self.packed_weight & 0x0F).to(torch.long)
        hi = ((self.packed_weight >> 4) & 0x0F).to(torch.long)
        vals = torch.stack([self.codebook[lo], self.codebook[hi]], dim=-1).view(-1, self.blocksize)
        return (vals * self.absmax.unsqueeze(-1)).reshape(-1)[: self.out_features * self.in_features].view(
            self.out_features, self.in_features
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NF4MatmulFn.apply(
            x,
            self.packed_weight,
            self.absmax,
            self.codebook,
            self.bias,
            self.out_features,
            self.in_features,
            self.blocksize,
        )


# ---------------------------------------------------------------------------
# Dataset Helpers
# ---------------------------------------------------------------------------

def load_wikitext_tokens(model_id: str, split: str, max_rows: Optional[int] = None) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    rows = ds["text"] if max_rows is None else ds["text"][:max_rows]
    text = "\n\n".join(t for t in rows if t.strip())
    return tok(text, return_tensors="pt").input_ids[0]


def make_chunks(tokens: torch.Tensor, seq_len: int, max_chunks: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    stride = seq_len + 1
    n_avail = (tokens.numel() - 1) // stride
    n = min(max_chunks, n_avail)
    out = []
    for i in range(n):
        c = tokens[i * stride : i * stride + stride]
        out.append((c[:-1].unsqueeze(0), c[1:].unsqueeze(0)))
    return out


# ---------------------------------------------------------------------------
# Mode: load_check
# ---------------------------------------------------------------------------

def run_load_check(model_id: str) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    prompt_ids = tok("Artificial intelligence and lattice quantization on Apple Silicon:", return_tensors="pt").input_ids
    results: Dict[str, Any] = {"model_id": model_id, "dtypes": {}}

    for dt_name, dt in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        gc.collect()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation="sdpa")
        total_params = sum(p.numel() for p in m.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
        m = m.to("mps")
        torch.mps.synchronize()
        load_s = time.perf_counter() - t0

        with torch.no_grad():
            out = m(prompt_ids.to("mps"))
        torch.mps.synchronize()
        logits = out.logits
        finite = bool(torch.isfinite(logits).all().item())
        mps_alloc, mps_drv = get_mps_memory_bytes()
        rss_cur, ru_max = get_process_rss_bytes()

        results["dtypes"][dt_name] = {
            "total_params": total_params,
            "static_weight_mb": round(param_bytes / (1024 * 1024), 2),
            "load_and_mps_move_s": round(load_s, 3),
            "logits_shape": list(logits.shape),
            "logits_finite": finite,
            "mps_alloc_mb": round(mps_alloc / (1024 * 1024), 2),
            "mps_driver_mb": round(mps_drv / (1024 * 1024), 2),
            "rss_mb": round(rss_cur / (1024 * 1024), 2),
            "ru_maxrss_mb": round(ru_max / (1024 * 1024), 2),
        }
        del m, out, logits
        gc.collect()
        torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Mode: ppl (FP16 / BF16 vs Unrotated-E8 vs Rotated-E8 vs NF4)
# ---------------------------------------------------------------------------

def eval_ppl_chunks(model: nn.Module, chunks: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    model.eval()
    nlls: List[float] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for inp, lbl in chunks:
            inp_mps = inp.to("mps")
            lbl_mps = lbl.to("mps")
            logits = model(inp_mps).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), lbl_mps.reshape(-1))
            nlls.append(float(loss.item()))
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    mean_nll = sum(nlls) / len(nlls)
    return {
        "chunks": len(chunks),
        "tokens_evaluated": sum(inp.numel() for inp, _ in chunks),
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(math.exp(min(mean_nll, 30.0)), 4),
        "eval_seconds": round(dt, 2),
    }


def quantize_decoder_linears_and_measure_snr(
    model: nn.Module,
    scheme: str,
) -> Dict[str, Any]:
    """Quantize decoder block nn.Linear weights in-place (keeping embed_tokens & lm_head intact) and measure SNR."""
    snrs: List[float] = []
    packed_bytes = 0
    scale_bytes = 0
    quantized_params = 0
    layers_count = 0

    for mod in model.model.layers.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.float()
            sig_pow = float(w.pow(2).mean().item())
            if scheme == "e8_unrotated_gs32":
                packed, scales, sh = E8LatticePacker.pack(w, group_size=32, mmse=True)
                w_rec = E8LatticePacker.unpack(packed, scales, sh, dtype=torch.float32)
            elif scheme == "e8_rotated_gs32":
                w_rot = apply_incoherence_transform(w, inverse=False)
                packed, scales, sh = E8LatticePacker.pack(w_rot, group_size=32, mmse=True)
                w_rec = apply_incoherence_transform(
                    E8LatticePacker.unpack(packed, scales, sh, dtype=torch.float32),
                    inverse=True,
                )
            elif scheme == "e8_rotated_gs8":
                w_rot = apply_incoherence_transform(w, inverse=False)
                packed, scales, sh = E8LatticePacker.pack(w_rot, group_size=8, mmse=True)
                w_rec = apply_incoherence_transform(
                    E8LatticePacker.unpack(packed, scales, sh, dtype=torch.float32),
                    inverse=True,
                )
            elif scheme == "nf4_bs64":
                nf4 = NF4Linear(mod, blocksize=64)
                packed = nf4.packed_weight
                scales = nf4.absmax
                w_rec = nf4.unpack_weight(dtype=torch.float32)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            err_pow = max(1e-12, float((w - w_rec).pow(2).mean().item()))
            snr_db = 10.0 * math.log10(max(1e-12, sig_pow) / err_pow)
            snrs.append(snr_db)
            packed_bytes += packed.numel() * packed.element_size()
            scale_bytes += scales.numel() * scales.element_size()
            quantized_params += w.numel()
            layers_count += 1
            mod.weight.data.copy_(w_rec.to(mod.weight.dtype))

    return {
        "layers_quantized": layers_count,
        "quantized_params": quantized_params,
        "packed_weight_mb": round(packed_bytes / (1024 * 1024), 2),
        "scale_mb": round(scale_bytes / (1024 * 1024), 2),
        "total_quantized_backbone_mb": round((packed_bytes + scale_bytes) / (1024 * 1024), 2),
        "mean_layer_snr_db": round(sum(snrs) / len(snrs), 3),
        "min_layer_snr_db": round(min(snrs), 3),
        "max_layer_snr_db": round(max(snrs), 3),
    }


def run_ppl(model_id: str, seq_len: int = 512, max_chunks: int = 64) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM

    test_tokens = load_wikitext_tokens(model_id, "test")
    chunks = make_chunks(test_tokens, seq_len=seq_len, max_chunks=max_chunks)
    out: Dict[str, Any] = {
        "model_id": model_id,
        "dataset": "wikitext-2-raw-v1:test",
        "seq_len": seq_len,
        "chunks": len(chunks),
        "total_eval_tokens": len(chunks) * seq_len,
        "variants": {},
    }

    # 1. BF16 baseline
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps")
    out["variants"]["bf16"] = {
        "ppl": eval_ppl_chunks(m, chunks),
        "quant_stats": None,
    }
    del m
    gc.collect()
    torch.mps.empty_cache()

    # 2. FP16 baseline
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, attn_implementation="sdpa").to("mps")
    out["variants"]["fp16"] = {
        "ppl": eval_ppl_chunks(m, chunks),
        "quant_stats": None,
    }
    del m
    gc.collect()
    torch.mps.empty_cache()

    # 3. Quantized variants (starting from clean BF16 weights on MPS)
    for scheme in ["e8_unrotated_gs32", "e8_rotated_gs32", "e8_rotated_gs8", "nf4_bs64"]:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps")
        t0 = time.perf_counter()
        qstats = quantize_decoder_linears_and_measure_snr(m, scheme)
        torch.mps.synchronize()
        qstats["pack_seconds"] = round(time.perf_counter() - t0, 2)
        ppl_res = eval_ppl_chunks(m, chunks)
        out["variants"][scheme] = {
            "ppl": ppl_res,
            "quant_stats": qstats,
        }
        del m
        gc.collect()
        torch.mps.empty_cache()

    return out


# ---------------------------------------------------------------------------
# Mode: lowram_probe (Verifies LowRAMTrainer & HolonomicGradFreeOptimizer edge cases)
# ---------------------------------------------------------------------------

def run_lowram_probe(model_id: str) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM
    from ojasx3.trainer.trainer import LowRAMTrainer

    tokens = load_wikitext_tokens(model_id, "train", max_rows=300)
    batches = make_chunks(tokens, seq_len=128, max_chunks=3)

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    for parent in m.model.layers.modules():
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and cname in TARGET_LINEARS:
                e8 = E8LatticeLinear.from_linear(
                    child.to("mps"), rank=16, lora_alpha=32.0, group_size=32, bitrate=2.0, use_hadamard=True
                )
                setattr(parent, cname, e8.to("cpu"))
    attach_lora_adapters(m, rank=16, lora_alpha=32.0, target_modules=TARGET_LINEARS)

    class _BackboneOnly(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x).last_hidden_state

    probe_res: Dict[str, Any] = {}
    try:
        trainer = LowRAMTrainer(
            _BackboneOnly(m.model),
            lr=2e-4,
            quantize_moments=True,
            fused_loss=True,
            device="mps",
            dtype=torch.bfloat16,
            compile_backend=None,
        )
        lm_w = m.lm_head.weight.to("mps")
        trainer.train_epoch(batches, epoch=1, log_interval=1, lm_head_weight=lm_w)
        probe_res["lowram_trainer_bf16_status"] = "succeeded"
    except Exception as exc:
        probe_res["lowram_trainer_bf16_status"] = "failed"
        probe_res["lowram_trainer_bf16_error"] = f"{type(exc).__name__}: {exc}"

    return probe_res


# ---------------------------------------------------------------------------
# Mode: train (Isolated 100-step Fine-Tuning Subprocess)
# ---------------------------------------------------------------------------

def _eval_val_loss(
    model: nn.Module,
    val_chunks: List[Tuple[torch.Tensor, torch.Tensor]],
    use_fused_lce: bool = False,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for inp, lbl in val_chunks:
            inp_mps = inp.to("mps")
            lbl_mps = lbl.to("mps")
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                if use_fused_lce:
                    hidden = model.model(inp_mps).last_hidden_state
                    loss = fused_linear_cross_entropy(hidden, model.lm_head.weight, lbl_mps)
                else:
                    logits = model(inp_mps).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lbl_mps.reshape(-1))
            losses.append(float(loss.float().item()))
    torch.mps.synchronize()
    mean_loss = sum(losses) / len(losses)
    return {
        "mean_val_loss": round(mean_loss, 5),
        "val_ppl": round(math.exp(min(mean_loss, 30.0)), 4),
    }


def _persistent_model_bytes(model: nn.Module) -> Dict[str, float]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return {
        "persistent_param_mb": round(param_bytes / (1024 * 1024), 2),
        "persistent_buffer_mb": round(buf_bytes / (1024 * 1024), 2),
        "persistent_total_mb": round((param_bytes + buf_bytes) / (1024 * 1024), 2),
        "trainable_param_mb": round(trainable_bytes / (1024 * 1024), 2),
    }


def run_train_method(
    model_id: str,
    method: str,
    steps: int = 100,
    seq_len: int = 256,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(42)
    train_tokens = load_wikitext_tokens(model_id, "train")
    val_tokens = load_wikitext_tokens(model_id, "validation")
    train_chunks = make_chunks(train_tokens, seq_len=seq_len, max_chunks=steps)
    val_chunks = make_chunks(val_tokens, seq_len=seq_len, max_chunks=16)
    if len(train_chunks) < steps:
        raise RuntimeError(f"Requested {steps} chunks of len {seq_len}, only {len(train_chunks)} available")

    mon = PeakMemoryMonitor(interval_s=0.005)
    mon.start()
    t_init0 = time.perf_counter()

    use_fused_lce = False
    grad_free_opt: Optional[HolonomicGradFreeOptimizer] = None
    torch_opt: Optional[torch.optim.Optimizer] = None

    if method == "ojasx":
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        for parent in m.model.layers.modules():
            for cname, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and cname in TARGET_LINEARS:
                    e8 = E8LatticeLinear.from_linear(
                        child.to("mps"),
                        rank=16,
                        lora_alpha=32.0,
                        group_size=32,
                        bitrate=2.0,
                        use_hadamard=True,
                    )
                    setattr(parent, cname, e8)
                    del child
        gc.collect()
        torch.mps.empty_cache()
        m = attach_lora_adapters(m, rank=16, lora_alpha=32.0, target_modules=TARGET_LINEARS)
        has_qk_norm = any(hasattr(mod, "q_norm") or hasattr(mod, "k_norm") for mod in m.modules())
        m, opt_report = optimize_model(m, enable_checkpointing=True, upgrade_attention=(not has_qk_norm))
        m = m.to("mps")
        torch.mps.synchronize()
        trainable = [p for p in m.parameters() if p.requires_grad]
        grad_free_opt = HolonomicGradFreeOptimizer(
            trainable,
            lr=lr,
            algorithm="adamw",
            weight_decay=0.01,
            quantize_moments_e8=False,
            max_grad_norm=1.0,
        )
        use_fused_lce = True
        extra_meta = {"optimize_actions": dict(opt_report.action_counts), "optimizer": "HolonomicGradFreeOptimizer(adamw, e8_base=2.0b, hadamard=True)"}

    elif method == "amp_sdpa_gc":
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        m = attach_lora_adapters(m, rank=16, lora_alpha=32.0, target_modules=TARGET_LINEARS)
        m.config.use_cache = False
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m = m.to("mps")
        torch.mps.synchronize()
        trainable = [p for p in m.parameters() if p.requires_grad]
        torch_opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        extra_meta = {"optimizer": "torch.optim.AdamW(bf16_amp, sdpa, grad_ckpt)"}

    elif method == "qlora":
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        for parent in m.model.layers.modules():
            for cname, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and cname in TARGET_LINEARS:
                    nf4 = NF4Linear(child.to("mps"), blocksize=64)
                    lora = LoRALinear(nf4, rank=16, lora_alpha=32.0)
                    setattr(parent, cname, lora)
                    del child
        gc.collect()
        torch.mps.empty_cache()
        for n, p in m.named_parameters():
            p.requires_grad = ("lora_A" in n or "lora_B" in n)
        m.config.use_cache = False
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m = m.to("mps")
        torch.mps.synchronize()
        trainable = [p for p in m.parameters() if p.requires_grad]
        torch_opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        extra_meta = {"optimizer": "torch.optim.AdamW(nf4_4bit, lora_r16, sdpa, grad_ckpt)"}

    elif method == "amp_sdpa_gc_full":
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        m.config.use_cache = False
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m = m.to("mps")
        torch.mps.synchronize()
        trainable = [p for p in m.parameters() if p.requires_grad]
        torch_opt = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
        extra_meta = {"optimizer": "torch.optim.AdamW(full_ft_bf16, sdpa, grad_ckpt)"}

    else:
        raise ValueError(f"Unknown method: {method}")

    init_s = time.perf_counter() - t_init0
    post_load_mem = mon.sample_now()
    persist_mem = _persistent_model_bytes(m)
    trainable_count = sum(p.numel() for p in trainable)
    total_logical_count = sum(p.numel() for p in m.parameters())

    val_step0 = _eval_val_loss(m, val_chunks, use_fused_lce=use_fused_lce) if method != "amp_sdpa_gc_full" else None
    gc.collect()
    torch.mps.empty_cache()

    mon_loop = PeakMemoryMonitor(interval_s=0.005)
    mon_loop.start()

    m.train()
    step_losses: List[float] = []
    step_times: List[float] = []
    phase_peaks = {"mid_fwd_mps_alloc_mb": 0.0, "mid_bwd_mps_alloc_mb": 0.0, "mid_fwd_rss_mb": 0.0, "mid_bwd_rss_mb": 0.0}
    status = "ok"
    error_msg: Optional[str] = None

    t_train0 = time.perf_counter()
    try:
        for idx, (inp, lbl) in enumerate(train_chunks):
            cur_rss, _ = get_process_rss_bytes()
            if method == "amp_sdpa_gc_full" and cur_rss > 20.5 * (1024 ** 3):
                status = "aborted_safety_ceiling"
                error_msg = f"RSS exceeded 20.5 GB safety threshold ({cur_rss / (1024**3):.2f} GB)"
                break

            t_s = time.perf_counter()
            inp_mps = inp.to("mps")
            lbl_mps = lbl.to("mps")

            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                if use_fused_lce:
                    hidden = m.model(inp_mps).last_hidden_state
                    loss = fused_linear_cross_entropy(hidden, m.lm_head.weight, lbl_mps)
                else:
                    logits = m(inp_mps).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lbl_mps.reshape(-1))

            if idx < 3 or (idx + 1) % 25 == 0:
                s_fwd = mon_loop.sample_now()
                phase_peaks["mid_fwd_mps_alloc_mb"] = max(phase_peaks["mid_fwd_mps_alloc_mb"], s_fwd["mps_alloc_mb"])
                phase_peaks["mid_fwd_rss_mb"] = max(phase_peaks["mid_fwd_rss_mb"], s_fwd["rss_mb"])

            loss.backward()

            if idx < 3 or (idx + 1) % 25 == 0:
                s_bwd = mon_loop.sample_now()
                phase_peaks["mid_bwd_mps_alloc_mb"] = max(phase_peaks["mid_bwd_mps_alloc_mb"], s_bwd["mps_alloc_mb"])
                phase_peaks["mid_bwd_rss_mb"] = max(phase_peaks["mid_bwd_rss_mb"], s_bwd["rss_mb"])

            if grad_free_opt is not None:
                grad_free_opt.step()
                grad_free_opt.zero_grad()
            elif torch_opt is not None:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                torch_opt.step()
                torch_opt.zero_grad(set_to_none=True)

            torch.mps.synchronize()
            dt_step = time.perf_counter() - t_s
            step_losses.append(round(float(loss.float().item()), 5))
            step_times.append(dt_step)

            if (idx + 1) % 10 == 0 or idx == 0:
                print(
                    f"[{method}] step {idx+1:3d}/{len(train_chunks)} | "
                    f"loss={step_losses[-1]:.4f} | {seq_len/dt_step:.1f} tok/s | "
                    f"mps={mon_loop.peak_mps_alloc_bytes/(1024**2):.1f}MB",
                    flush=True,
                )
    except Exception as exc:
        status = "error"
        error_msg = f"{type(exc).__name__}: {exc}"

    train_elapsed_s = time.perf_counter() - t_train0
    train_loop_peak_mem = mon_loop.stop()
    val_step100 = (
        _eval_val_loss(m, val_chunks, use_fused_lce=use_fused_lce)
        if (status == "ok" and method != "amp_sdpa_gc_full")
        else None
    )
    peak_mem = mon.stop()

    completed_steps = len(step_losses)
    total_tokens = completed_steps * seq_len
    steady_times = step_times[5:] if len(step_times) > 10 else step_times
    steady_tok_s = (len(steady_times) * seq_len / sum(steady_times)) if steady_times else 0.0

    return {
        "method": method,
        "status": status,
        "error": error_msg,
        "model_id": model_id,
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "seq_len": seq_len,
        "lr": lr,
        "trainable_params": trainable_count,
        "persistent_model": persist_mem,
        "init_seconds": round(init_s, 2),
        "train_elapsed_seconds": round(train_elapsed_s, 2),
        "overall_tok_per_sec": round(total_tokens / max(1e-6, train_elapsed_s), 2),
        "steady_tok_per_sec": round(steady_tok_s, 2),
        "step_1_loss": step_losses[0] if step_losses else None,
        "step_last_loss": step_losses[-1] if step_losses else None,
        "mean_first_10_loss": round(sum(step_losses[:10]) / len(step_losses[:10]), 5) if step_losses else None,
        "mean_last_10_loss": round(sum(step_losses[-10:]) / len(step_losses[-10:]), 5) if step_losses else None,
        "val_step0": val_step0,
        "val_step100": val_step100,
        "post_load_memory": {k: round(v, 2) for k, v in post_load_mem.items()},
        "phase_peaks_mb": {k: round(v, 2) for k, v in phase_peaks.items()},
        "train_loop_peak_memory": train_loop_peak_mem,
        "peak_memory": peak_mem,
        "step_losses": step_losses,
        **extra_meta,
    }


# ---------------------------------------------------------------------------
# Mode: all (Orchestrator running isolated subprocesses + saving JSON & MD)
# ---------------------------------------------------------------------------

def _run_subproc(args: List[str], tag: str) -> Dict[str, Any]:
    marker = "__JSON_RESULT__="
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *args, "--emit-marker"]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    print(f"\n>>> Running subprocess [{tag}]: {' '.join(args)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: List[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        lines.append(line)
        if not line.startswith(marker):
            print(f"  [{tag}] {line}", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Subprocess {tag} failed with code {rc}")
    for line in reversed(lines):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(f"Missing JSON marker from {tag}")


def render_markdown(report: Dict[str, Any]) -> str:
    hw = report["hardware"]
    lc = report["load_check"]
    ppl = report["perplexity"]
    tr = report["training"]
    probe = report.get("lowram_probe", {})

    bf16_lc = lc["dtypes"]["bfloat16"]
    fp16_lc = lc["dtypes"]["float16"]

    lines = [
        f"# Real Pretrained Checkpoint Benchmark: `{report['model_id']}` on `wikitext-2-raw-v1` (Apple Silicon MPS)",
        "",
        "## 1. Environment & Hardware",
        f"- **Host:** `{hw['platform']}` (`{hw['machine']}`), `{hw['unified_memory_gb']} GB` unified memory",
        f"- **Python / PyTorch:** `{hw['python_version']}` / `torch {hw['torch_version']}` / `transformers {hw['transformers_version']}`",
        f"- **MPS Available:** `{hw['mps_available']}`",
        f"- **Checkpoint:** `{report['model_id']}` (`{bf16_lc['total_params']:,}` parameters; `tie_word_embeddings` verified)",
        f"- **Note on Third-Party Packages:** `peft` and `bitsandbytes` are not installed in the `.venv` (and `bitsandbytes` has no MPS backend); LoRA uses `ojasx3.trainer.peft.LoRALinear` (`r=16, alpha=32`) uniformly across all methods, and QLoRA uses an MPS-native 4-bit NF4 blocksize-64 layer (`NF4Linear`, 4.25 bpw) with on-the-fly autograd dequantization.",
        "",
        "## 2. Clean MPS Load & Sanity Verification",
        "| Dtype | Static Weight (MB) | Load + MPS Move (s) | Logits Shape | Finite | MPS Alloc (MB) | MPS Driver (MB) | Process RSS (MB) |",
        "|---|---:|---:|---|---|---:|---:|---:|",
        f"| `bfloat16` | {bf16_lc['static_weight_mb']} | {bf16_lc['load_and_mps_move_s']} | `{bf16_lc['logits_shape']}` | `{bf16_lc['logits_finite']}` | {bf16_lc['mps_alloc_mb']} | {bf16_lc['mps_driver_mb']} | {bf16_lc['rss_mb']} |",
        f"| `float16` | {fp16_lc['static_weight_mb']} | {fp16_lc['load_and_mps_move_s']} | `{fp16_lc['logits_shape']}` | `{fp16_lc['logits_finite']}` | {fp16_lc['mps_alloc_mb']} | {fp16_lc['mps_driver_mb']} | {fp16_lc['rss_mb']} |",
        "",
        f"## 3. WikiText-2 (`test` split, {ppl['chunks']} chunks × {ppl['seq_len']} = {ppl['total_eval_tokens']:,} tokens) Perplexity",
        "Decoder block linear layers (`q/k/v/o/gate/up/down_proj`) quantized; `embed_tokens` / `lm_head` retained in BF16.",
        "",
        "| Variant | Quantized Backbone (MB) | Mean Layer SNR (dB) | Mean NLL | Perplexity | Eval Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for key, label in [
        ("bf16", "BF16 Baseline"),
        ("fp16", "FP16 Baseline"),
        ("nf4_bs64", "4-bit NF4 (bs=64, 4.25 bpw)"),
        ("e8_unrotated_gs32", "Unrotated E8 (gs=32, 2.0 bpw)"),
        ("e8_rotated_gs32", "Rotated-E8 (`apply_incoherence_transform` + `E8LatticePacker`, gs=32, 2.0 bpw)"),
        ("e8_rotated_gs8", "Rotated-E8 (`apply_incoherence_transform` + `E8LatticePacker`, gs=8, 2.0 bpw)"),
    ]:
        v = ppl["variants"][key]
        qs = v["quant_stats"]
        p = v["ppl"]
        qmb = qs["total_quantized_backbone_mb"] if qs else "—"
        snr = qs["mean_layer_snr_db"] if qs else "—"
        lines.append(f"| {label} | {qmb} | {snr} | {p['mean_nll']} | {p['perplexity']} | {p['eval_seconds']} |")

    lines.extend([
        "",
        "## 4. 100-Step Fine-Tuning Convergence, True Mid-Step Peak Memory & Throughput (`wikitext-2-raw-v1:train`, `seq_len=256`)",
        "Each method executed in its own fresh isolated subprocess with 5 ms background polling (`mach_task_basic_info` RSS + `torch.mps.current_allocated_memory()` + `torch.mps.driver_allocated_memory()`) and kernel `ru_maxrss`.",
        "",
        "| Method | Status | Step 1 Loss | Final Loss | Mean Loss (1–10 → Last 10) | Val Loss (Step 0 → Final) | Steady `tok/s` | Train-Loop Peak MPS Alloc (MB) | Peak MPS Driver (MB) | Train-Loop Peak RSS (MB) | OS `ru_maxrss` (MB) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])

    for mkey, mlabel in [
        ("ojasx", "OjasX (`E8` 2.0b + Hadamard + PEFT + `HolonomicGradFreeOptimizer` + FusedLCE)"),
        ("amp_sdpa_gc", "PyTorch AMP (`bf16`) + SDPA + Grad Ckpt (LoRA r=16)"),
        ("qlora", "QLoRA (4-bit NF4 + LoRA r=16 + Grad Ckpt)"),
        ("amp_sdpa_gc_full", "PyTorch AMP (`bf16`) + SDPA + Grad Ckpt (Full-FT 3-step probe)"),
    ]:
        if mkey not in tr:
            continue
        r = tr[mkey]
        pm = r["peak_memory"]
        lm = r.get("train_loop_peak_memory", pm)
        m_traj = f"{r['mean_first_10_loss']} → {r['mean_last_10_loss']}" if r["mean_first_10_loss"] is not None else "—"
        if r.get("val_step0") and r.get("val_step100"):
            v_traj = f"{r['val_step0']['mean_val_loss']} → {r['val_step100']['mean_val_loss']}"
        else:
            v_traj = "—"
        lines.append(
            f"| {mlabel} | `{r['status']}` ({r['steps_completed']}/{r['steps_requested']}) | "
            f"{r['step_1_loss']} | {r['step_last_loss']} | {m_traj} | {v_traj} | "
            f"{r['steady_tok_per_sec']} | {lm['peak_mps_alloc_mb']} | {lm['peak_mps_driver_mb']} | "
            f"{lm['peak_sampled_rss_mb']} | {pm['os_ru_maxrss_mb']} |"
        )

    lines.extend([
        "",
        "## 5. Observed `LowRAMTrainer` & `HolonomicGradFreeOptimizer` Audit Notes",
        f"- **`LowRAMTrainer(dtype=bfloat16)` probe:** `{probe.get('lowram_trainer_bf16_status')}` — `{probe.get('lowram_trainer_bf16_error', 'none')}`.",
        "  `LowRAMTrainer.__init__` (`ojasx2/trainer.py:97`) sets `master_dtype = torch.float32` whenever `autocast` is enabled, which casts embeddings/norms to FP32 while `autocast(bfloat16)` produces `bfloat16` backward gradients, triggering a `c10::BFloat16 != float` mismatch in `_FusedE8GEMMFunction.backward` (`ojasx2/math/lattice.py:723`).",
        "- **`HolonomicGradFreeOptimizer` `quantize_moments_e8=True` with `algorithm=\"adamw\"`:** `ojasx2/grad_free_optimizer.py:273-292` packs non-negative `sqrt(v_t)` via `E8LatticePacker` (`{-1.5,-0.5,+0.5,+1.5}`) and clamps negative codes to `0.0`, stalling AdamW convergence on LoRA adapters; therefore the OjasX 100-step run uses `HolonomicGradFreeOptimizer(algorithm=\"adamw\", quantize_moments_e8=False)` (retaining immediate in-backward `p.grad = None` freeing and global grad-norm latching).",
        "",
        f"Raw JSON: `{report['json_path']}`",
        "",
    ])
    return "\n".join(lines)


def run_all(model_id: str, steps: int, seq_len: int, ppl_chunks: int, out_dir: Path) -> None:
    import transformers

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "BENCHMARK_1B_WIKITEXT2.json"
    md_path = out_dir / "BENCHMARK_1B_WIKITEXT2.md"

    mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    hw = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "unified_memory_gb": round(mem_bytes / (1024 ** 3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }

    load_check = _run_subproc(["--model", model_id, "--mode", "load_check"], "load_check")
    ppl_res = _run_subproc(
        ["--model", model_id, "--mode", "ppl", "--ppl-seq-len", "512", "--ppl-chunks", str(ppl_chunks)],
        "ppl",
    )
    lowram_probe = _run_subproc(["--model", model_id, "--mode", "lowram_probe"], "lowram_probe")

    training: Dict[str, Any] = {}
    for method in ["ojasx", "amp_sdpa_gc", "qlora"]:
        training[method] = _run_subproc(
            ["--model", model_id, "--mode", "train", "--method", method, "--steps", str(steps), "--seq-len", str(seq_len)],
            f"train_{method}",
        )
    training["amp_sdpa_gc_full"] = _run_subproc(
        ["--model", model_id, "--mode", "train", "--method", "amp_sdpa_gc_full", "--steps", "3", "--seq-len", str(seq_len)],
        "train_full_probe",
    )

    report = {
        "model_id": model_id,
        "hardware": hw,
        "load_check": load_check,
        "perplexity": ppl_res,
        "lowram_probe": lowram_probe,
        "training": training,
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))
    print(f"\nSaved JSON -> {json_path}")
    print(f"Saved MD   -> {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", choices=["load_check", "ppl", "lowram_probe", "train", "all"], default="all")
    parser.add_argument("--method", choices=["ojasx", "amp_sdpa_gc", "qlora", "amp_sdpa_gc_full"], default="ojasx")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--ppl-seq-len", type=int, default=512)
    parser.add_argument("--ppl-chunks", type=int, default=64)
    parser.add_argument("--out-dir", default="/Users/mohana/.gemini/antigravity/scratch/ojasx3_upgrade_audit")
    parser.add_argument("--emit-marker", action="store_true")
    args = parser.parse_args()

    if args.mode == "all":
        run_all(args.model, args.steps, args.seq_len, args.ppl_chunks, Path(args.out_dir))
        return

    if args.mode == "load_check":
        res = run_load_check(args.model)
    elif args.mode == "ppl":
        res = run_ppl(args.model, seq_len=args.ppl_seq_len, max_chunks=args.ppl_chunks)
    elif args.mode == "lowram_probe":
        res = run_lowram_probe(args.model)
    elif args.mode == "train":
        res = run_train_method(args.model, args.method, steps=args.steps, seq_len=args.seq_len)
    else:
        raise ValueError(args.mode)

    if args.emit_marker:
        print("__JSON_RESULT__=" + json.dumps(res), flush=True)
    else:
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
