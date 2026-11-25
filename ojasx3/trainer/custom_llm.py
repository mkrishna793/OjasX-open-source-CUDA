"""
Turnkey Custom LLM Pretraining & Fine-Tuning Engine for OjasX3.

Allows users to:
1. Build and train a custom LLM from scratch (`preset="tiny-50m"`, `"small-150m"`,
   `"medium-350m"`, `"1b"`, `"3b"`, `"7b"`, or custom `CustomLLMConfig`) using
   OjasX low-RAM optimizations:
     - `FusedLinearCrossEntropy` (0-byte vocabulary logit materialization)
     - Activation gradient checkpointing
     - `HolonomicGradFreeOptimizer` (in-backward gradient freeing)
     - Optional `2.125-bit Dual-Rail` (`mode="dual_rail"`) or `2.0-bit Rotated-E8`
       (`mode="e8"`) or full `bfloat16` (`mode="bf16"`) weights
2. Fine-tune any HuggingFace or local LLM checkpoint on custom `.txt` / `.jsonl` /
   in-memory text data with 2.125-bit Dual-Rail + BF16 LoRA (`7.53x` weight compression)
   and zero-lock-in export (`export_peft_adapter` / `merge_and_export_hf`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.fused_cross_entropy import FusedLinearCrossEntropy
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.math.lattice import E8LatticeLinear
from ojasx_logic.trainer.dual_rail_lora import (
    DualRailLoRALinear,
    quantize_module_to_dual_rail_lora,
)
from ojasx_logic.trainer.hf_9b_trainer import (
    _detect_device,
    _device_mem_mb,
    _device_sync,
    _get_host_rss_mb,
    load_and_quantize_hf_model,
    run_dual_rail_lora_training,
)


@dataclass
class CustomLLMConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    n_layers: int = 12
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


CUSTOM_LLM_PRESETS: Dict[str, CustomLLMConfig] = {
    "nano-15m": CustomLLMConfig(vocab_size=8192, d_model=384, n_heads=6, n_kv_heads=2, d_ff=1024, n_layers=6),
    "tiny-50m": CustomLLMConfig(vocab_size=16384, d_model=512, n_heads=8, n_kv_heads=4, d_ff=1408, n_layers=8),
    "small-150m": CustomLLMConfig(vocab_size=32000, d_model=768, n_heads=12, n_kv_heads=4, d_ff=2048, n_layers=12),
    "medium-350m": CustomLLMConfig(vocab_size=32000, d_model=1024, n_heads=16, n_kv_heads=4, d_ff=2816, n_layers=20),
    "1b": CustomLLMConfig(vocab_size=32000, d_model=2048, n_heads=32, n_kv_heads=8, d_ff=5632, n_layers=22),
    "3b": CustomLLMConfig(vocab_size=32000, d_model=3072, n_heads=24, n_kv_heads=8, d_ff=8192, n_layers=28),
    "7b": CustomLLMConfig(vocab_size=32000, d_model=4096, n_heads=32, n_kv_heads=8, d_ff=11008, n_layers=32),
}


class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight


class CustomDecoderBlock(nn.Module):
    """Production GQA Causal SDPA + 3-Projection SwiGLU Decoder Block."""

    def __init__(self, cfg: CustomLLMConfig, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.attn_norm = CustomRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, device=device, dtype=dtype)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, device=device, dtype=dtype)

        self.ffn_norm = CustomRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, device=device, dtype=dtype)
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn)

        h2 = self.ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: Optional[torch.Tensor] = None


class OjasXCustomCausalLM(nn.Module):
    """Turnkey custom Causal LLM with FusedLinearCrossEntropy (0-byte full logit materialization during training)."""

    def __init__(
        self,
        cfg: CustomLLMConfig,
        *,
        gradient_checkpointing: bool = True,
        device=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.config = cfg
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [CustomDecoderBlock(cfg, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        )
        self.norm = CustomRMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, device=device, dtype=dtype)

        if cfg.tie_word_embeddings:
            self.lm_head_weight = self.embed_tokens.weight
        else:
            self.lm_head_weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype))
            nn.init.normal_(self.lm_head_weight, mean=0.0, std=0.02)

        self.fused_ce = FusedLinearCrossEntropy(
            in_features=cfg.d_model,
            vocab_size=cfg.vocab_size,
            bias=False,
            chunk_size=1024,
            weight=self.lm_head_weight,
            device=device,
            dtype=dtype,
        )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        if labels is not None:
            shift_h = h[:, :-1, :].contiguous().view(-1, h.shape[-1])
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = self.fused_ce(shift_h, shift_labels)
            return CausalLMOutput(loss=loss, logits=None)
        logits = F.linear(h, self.lm_head_weight.to(h.dtype))
        return CausalLMOutput(loss=None, logits=logits)

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        was_training = self.training
        self.eval()
        cur = prompt_ids
        for _ in range(max_new_tokens):
            out = self(cur)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            cur = torch.cat([cur, next_tok], dim=-1)
        if was_training:
            self.train()
        return cur


def resolve_custom_llm_config(
    preset_or_config: Union[str, CustomLLMConfig, Dict[str, Any]] = "small-150m"
) -> CustomLLMConfig:
    if isinstance(preset_or_config, CustomLLMConfig):
        return preset_or_config
    if isinstance(preset_or_config, dict):
        return CustomLLMConfig(**preset_or_config)
    key = str(preset_or_config).strip().lower()
    if key not in CUSTOM_LLM_PRESETS:
        raise ValueError(
            f"Unknown custom LLM preset '{preset_or_config}'. Choose from {list(CUSTOM_LLM_PRESETS)}"
        )
    return CUSTOM_LLM_PRESETS[key]


def build_custom_llm(
    preset_or_config: Union[str, CustomLLMConfig, Dict[str, Any]] = "small-150m",
    *,
    mode: str = "dual_rail",
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    group_size: int = 128,
    gradient_checkpointing: bool = True,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[OjasXCustomCausalLM, Dict[str, Any]]:
    """Build a custom LLM ready for low-RAM training in 1 line.

    `mode`:
      - `"dual_rail"`: 2.125-bit Dual-Rail base weights (7.53x compression) + trainable BF16 LoRA + embeddings/norms
      - `"e8"`: 2.0-bit Hadamard-Rotated E8 lattice weights + trainable LoRA + embeddings/norms
      - `"bf16"`: Full dense BF16 pretraining with FusedLinearCrossEntropy + GradCkpt
    """
    cfg = resolve_custom_llm_config(preset_or_config)
    dev = _detect_device(device)
    mode_norm = mode.strip().lower()

    # Build on CPU first when quantizing so peak GPU memory never holds uncompressed dense weights
    init_dev = "cpu" if mode_norm in ("dual_rail", "e8") else dev
    model = OjasXCustomCausalLM(
        cfg,
        gradient_checkpointing=gradient_checkpointing,
        device=init_dev,
        dtype=dtype,
    )

    total_arch_params = sum(p.numel() for p in model.parameters())
    stats: Dict[str, Any] = {
        "mode": mode_norm,
        "total_arch_params": int(total_arch_params),
        "layers_quantized": 0,
        "bf16_linear_gb": 0.0,
        "packed_dual_rail_gb": 0.0,
        "compression_ratio": 1.0,
    }

    if mode_norm == "dual_rail":
        eff_gs = group_size if (cfg.d_model % group_size == 0 and cfg.d_ff % group_size == 0) else 32
        q_stats = quantize_module_to_dual_rail_lora(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            group_size=eff_gs,
            device=dev,
            compute_dtype_name="bfloat16" if dtype == torch.bfloat16 else "float16",
        )
        model.to(dev)
        # Keep embeddings & RMSNorms trainable alongside LoRA for custom LLM training
        for p in model.embed_tokens.parameters():
            p.requires_grad_(True)
        for mod in model.modules():
            if isinstance(mod, CustomRMSNorm):
                for p in mod.parameters():
                    p.requires_grad_(True)
        stats.update(q_stats)
    elif mode_norm == "e8":
        eff_gs = 32 if (cfg.d_model % 32 == 0 and cfg.d_ff % 32 == 0) else 8
        layers_q = 0
        bf16_bytes = 0
        packed_bytes = 0
        for blk in model.layers:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                lin = getattr(blk, attr)
                if isinstance(lin, nn.Linear):
                    bf16_bytes += lin.weight.numel() * 2
                    e8_lin = E8LatticeLinear.from_linear(
                        lin,
                        rank=lora_r,
                        lora_alpha=lora_alpha,
                        group_size=eff_gs,
                        bitrate=2.0,
                        use_hadamard=(lin.in_features % 32 == 0),
                    ).to(device=dev, dtype=dtype)
                    packed_bytes += e8_lin.packed_weight.nbytes + e8_lin.block_scales.nbytes
                    setattr(blk, attr, e8_lin)
                    layers_q += 1
        model.to(dev)
        stats.update(
            {
                "layers_quantized": layers_q,
                "bf16_linear_gb": bf16_bytes / (1024.0 ** 3),
                "packed_dual_rail_gb": packed_bytes / (1024.0 ** 3),
                "compression_ratio": bf16_bytes / max(packed_bytes, 1),
            }
        )
    elif mode_norm in ("bf16", "fp16", "fp32", "dense"):
        model.to(dev)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from 'dual_rail', 'e8', or 'bf16'.")

    stats["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, stats


def load_text_or_jsonl_batches(
    data: Union[str, Path, Sequence[str], Sequence[torch.Tensor]],
    *,
    tokenizer: Any = None,
    vocab_size: int = 32000,
    seq_len: int = 256,
    batch_size: int = 1,
    max_batches: int = 256,
) -> List[torch.Tensor]:
    """Convert a `.txt` file, `.jsonl` file, list of strings, or tensor list into `[batch_size, seq_len]` token batches."""
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], torch.Tensor):
        out: List[torch.Tensor] = []
        for t in data[:max_batches]:
            if t.ndim == 1:
                t = t.unsqueeze(0)
            out.append(t.to(torch.long))
        return out

    texts: List[str] = []
    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.exists() and p.is_file():
            raw = p.read_text(encoding="utf-8", errors="ignore")
            if p.suffix.lower() == ".jsonl":
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        txt = (
                            obj.get("text")
                            or (f"{obj.get('prompt', '')}\n{obj.get('completion', '')}".strip())
                            or (f"{obj.get('instruction', '')}\n{obj.get('output', '')}".strip())
                            or json.dumps(obj)
                        )
                        if txt:
                            texts.append(str(txt))
            elif p.suffix.lower() == ".json":
                arr = json.loads(raw)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict):
                            texts.append(
                                str(
                                    item.get("text")
                                    or f"{item.get('prompt', '')}\n{item.get('completion', '')}".strip()
                                    or json.dumps(item)
                                )
                            )
            else:
                texts.append(raw)
        else:
            # Treat string directly as corpus text
            texts.append(str(data))
    elif isinstance(data, Sequence):
        texts.extend(str(x) for x in data if str(x).strip())

    full_text = "\n\n".join(t for t in texts if t.strip())
    if not full_text:
        full_text = "OjasX low-RAM custom LLM training engine. " * 256

    if tokenizer is not None:
        ids_list = tokenizer.encode(full_text)
        ids = torch.tensor(ids_list, dtype=torch.long)
    else:
        # Zero-dependency deterministic byte-pair hash tokenizer mapped into [4, vocab_size)
        raw_bytes = full_text.encode("utf-8")
        eff_mod = max(vocab_size - 4, 256)
        toks = [4 + ((b + (raw_bytes[i - 1] << 4 if i > 0 else 0)) % eff_mod) for i, b in enumerate(raw_bytes)]
        ids = torch.tensor(toks, dtype=torch.long)

    needed = seq_len * batch_size
    if ids.numel() < needed:
        reps = (needed // max(ids.numel(), 1)) + 2
        ids = ids.repeat(reps)

    batches: List[torch.Tensor] = []
    for start in range(0, ids.numel() - needed + 1, needed):
        batches.append(ids[start : start + needed].view(batch_size, seq_len).clone())
        if len(batches) >= max_batches:
            break
    if not batches:
        batches.append(ids[:needed].view(batch_size, seq_len).clone())
    return batches


@dataclass
class CustomLLMTrainResult:
    model_name: str
    mode: str
    device: str
    steps: int
    seq_len: int
    batch_size: int
    total_arch_params: int
    trainable_params: int
    layers_quantized: int
    bf16_linear_gb: float
    packed_weights_gb: float
    compression_ratio: float
    loss_start: float
    loss_end: float
    loss_history: List[float]
    tokens_per_sec: float
    peak_device_alloc_mb: float
    peak_device_driver_mb: float
    host_rss_mb: float
    saved_dir: Optional[str] = None
    model: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("model", None)
        return d


def train_custom_llm(
    data: Optional[Union[str, Path, Sequence[str], Sequence[torch.Tensor]]] = None,
    *,
    preset: Union[str, CustomLLMConfig, Dict[str, Any]] = "tiny-50m",
    checkpoint: Optional[Union[str, Path, nn.Module]] = None,
    mode: str = "dual_rail",
    steps: int = 20,
    seq_len: int = 128,
    batch_size: int = 1,
    lr: float = 1e-3,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    group_size: int = 128,
    device: str = "auto",
    safety_policy: Optional[Callable[[int, float], None]] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> CustomLLMTrainResult:
    """Turnkey 1-line training for a custom LLM from scratch OR fine-tuning an existing HF checkpoint.

    Examples:
      # 1. Train your own custom LLM from scratch on a text/JSONL file:
      res = ojasx3.train_custom_llm("my_corpus.txt", preset="small-150m", mode="dual_rail", steps=100)

      # 2. Fine-tune a pretrained HuggingFace model on your custom JSONL dataset:
      res = ojasx3.train_custom_llm("my_sft.jsonl", checkpoint="Qwen/Qwen2.5-1.5B", steps=100, save_dir="./out")
    """
    dev = _detect_device(device)

    # Branch A: Pretrained HuggingFace checkpoint ID / local HF directory
    if isinstance(checkpoint, (str, Path)):
        model_id = str(checkpoint)
        model, tokenizer, stats, dev = load_and_quantize_hf_model(
            model_id,
            device=dev,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            group_size=group_size,
            gradient_checkpointing=True,
        )
        vocab_sz = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
        batches = (
            load_text_or_jsonl_batches(
                data,
                tokenizer=tokenizer,
                vocab_size=vocab_sz,
                seq_len=seq_len,
                batch_size=batch_size,
                max_batches=max(steps, 16),
            )
            if data is not None
            else None
        )
        hf_res = run_dual_rail_lora_training(
            model,
            model_id=model_id,
            stats=stats,
            device=dev,
            steps=steps,
            seq_len=seq_len,
            batch_size=batch_size,
            lr=lr,
            vocab_size=vocab_sz,
            input_batches=batches,
            safety_policy=safety_policy,
        )
        saved_path_str: Optional[str] = None
        if save_dir is not None:
            out_p = Path(save_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            adapter_sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
            try:
                from safetensors.torch import save_file
                save_file(adapter_sd, str(out_p / "adapter_model.safetensors"))
            except Exception:
                torch.save(adapter_sd, str(out_p / "adapter_model.bin"))
            (out_p / "ojasx_train_summary.json").write_text(
                json.dumps(asdict(hf_res), indent=2), encoding="utf-8"
            )
            saved_path_str = str(out_p)
        return CustomLLMTrainResult(
            model_name=model_id,
            mode="dual_rail",
            device=dev,
            steps=steps,
            seq_len=seq_len,
            batch_size=batch_size,
            total_arch_params=int(sum(p.numel() for p in model.parameters())),
            trainable_params=hf_res.trainable_lora_params,
            layers_quantized=hf_res.layers_quantized,
            bf16_linear_gb=hf_res.bf16_linear_gb,
            packed_weights_gb=hf_res.packed_dual_rail_gb,
            compression_ratio=hf_res.compression_ratio,
            loss_start=hf_res.loss_start,
            loss_end=hf_res.loss_end,
            loss_history=[hf_res.loss_start, hf_res.loss_end],
            tokens_per_sec=hf_res.tokens_per_sec,
            peak_device_alloc_mb=hf_res.peak_device_alloc_mb,
            peak_device_driver_mb=hf_res.peak_device_driver_mb,
            host_rss_mb=hf_res.host_rss_mb,
            saved_dir=saved_path_str,
            model=model,
        )

    # Branch B: Build & train custom OjasX LLM from scratch (or user-supplied nn.Module)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(dev)
        cfg_vocab = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
        stats = {
            "mode": mode,
            "total_arch_params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "layers_quantized": 0,
            "bf16_linear_gb": 0.0,
            "packed_dual_rail_gb": 0.0,
            "compression_ratio": 1.0,
        }
        model_name = type(model).__name__
    else:
        cfg = resolve_custom_llm_config(preset)
        cfg_vocab = cfg.vocab_size
        model, stats = build_custom_llm(
            cfg,
            mode=mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            group_size=group_size,
            gradient_checkpointing=True,
            device=dev,
        )
        model_name = preset if isinstance(preset, str) else f"custom-{cfg.n_layers}L-{cfg.d_model}d"

    batches = load_text_or_jsonl_batches(
        data if data is not None else "OjasX custom LLM training corpus. " * 256,
        vocab_size=cfg_vocab,
        seq_len=seq_len,
        batch_size=batch_size,
        max_batches=max(steps, 16),
    )

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = HolonomicGradFreeOptimizer(
        trainable,
        lr=lr,
        weight_decay=0.01,
        quantize_moments_e8=False,
    )

    losses: List[float] = []
    peak_alloc = 0.0
    peak_driver = 0.0

    _device_sync(dev)
    t0 = time.perf_counter()

    for step in range(steps):
        batch_ids = batches[step % len(batches)].to(dev)
        optimizer.zero_grad()
        out = model(batch_ids, labels=batch_ids)
        loss = out.loss if hasattr(out, "loss") else out
        loss.backward()
        a_mid, d_mid = _device_mem_mb(dev)
        peak_alloc = max(peak_alloc, a_mid)
        peak_driver = max(peak_driver, d_mid)
        optimizer.step()
        optimizer.zero_grad()

        loss_val = float(loss.detach().float().item())
        losses.append(loss_val)
        if safety_policy is not None:
            safety_policy(step, loss_val)

        _device_sync(dev)
        a_end, d_end = _device_mem_mb(dev)
        peak_alloc = max(peak_alloc, a_end)
        peak_driver = max(peak_driver, d_end)

    elapsed = max(time.perf_counter() - t0, 1e-6)
    tok_s = (steps * batch_size * seq_len) / elapsed

    saved_path_str = None
    if save_dir is not None:
        out_p = Path(save_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        cpu_sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(cpu_sd, str(out_p / "model.safetensors"))
        except Exception:
            torch.save(cpu_sd, str(out_p / "model.bin"))
        if hasattr(model, "config") and isinstance(model.config, CustomLLMConfig):
            (out_p / "config.json").write_text(json.dumps(asdict(model.config), indent=2), encoding="utf-8")
        saved_path_str = str(out_p)

    res = CustomLLMTrainResult(
        model_name=str(model_name),
        mode=str(stats["mode"]),
        device=dev,
        steps=steps,
        seq_len=seq_len,
        batch_size=batch_size,
        total_arch_params=int(stats["total_arch_params"]),
        trainable_params=int(stats["trainable_params"]),
        layers_quantized=int(stats.get("layers_quantized", 0)),
        bf16_linear_gb=float(stats.get("bf16_linear_gb", 0.0)),
        packed_weights_gb=float(stats.get("packed_dual_rail_gb", 0.0)),
        compression_ratio=float(stats.get("compression_ratio", 1.0)),
        loss_start=losses[0] if losses else 0.0,
        loss_end=losses[-1] if losses else 0.0,
        loss_history=losses,
        tokens_per_sec=tok_s,
        peak_device_alloc_mb=peak_alloc,
        peak_device_driver_mb=peak_driver,
        host_rss_mb=_get_host_rss_mb(),
        saved_dir=saved_path_str,
        model=model,
    )
    if save_dir is not None:
        (Path(save_dir) / "ojasx_train_summary.json").write_text(
            json.dumps(res.to_dict(), indent=2), encoding="utf-8"
        )
    return res


__all__ = [
    "CustomLLMConfig",
    "CUSTOM_LLM_PRESETS",
    "OjasXCustomCausalLM",
    "CustomLLMTrainResult",
    "resolve_custom_llm_config",
    "build_custom_llm",
    "load_text_or_jsonl_batches",
    "train_custom_llm",
]
