#!/usr/bin/env python3
"""Run a complete local HF checkpoint through checkpoint-direct E8 LoRA training.

Examples (load-only and synthetic mechanics respectively)::

    python tools/run_streaming_e8_finetune.py --checkpoint /models/qwen32b
    python tools/run_streaming_e8_finetune.py --checkpoint /models/qwen32b \
        --max-steps 1 --allow-synthetic --output reports/qwen32b-e8.json

Synthetic tokens establish execution/memory mechanics only.  Use --text-file
for a real data path, and use a held-out evaluation separately for quality.
The driver intentionally uses the model's stock HF attention and labels-loss
path; it does not claim OjasX fused attention or streaming cross-entropy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import psutil
import torch

# Allow execution directly from a source checkout as well as an installed tool.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ojasx2
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer
from ojasx2.streaming_e8 import load_hf_checkpoint_e8


class RSSSampler:
    """Low-overhead process RSS sampler; OS high-water is recorded separately."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample(self) -> None:
        value = int(self.process.memory_info().rss)
        self.peak_bytes = max(self.peak_bytes, value)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._run, name="ojasx-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()


def os_peak_rss_bytes() -> int:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return int(value if sys.platform == "darwin" else value * 1024.0)


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local HF config + safetensors directory")
    parser.add_argument("--output", default="reports/streaming-e8-run.json")
    parser.add_argument("--bitrate", type=float, choices=(1.0, 1.5, 2.0), default=1.5)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--row-block", type=int, default=32)
    parser.add_argument("--use-hadamard", action="store_true")
    parser.add_argument("--adapter-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=[],
        help="Module glob to leave dense (repeatable); tied lm_head must be explicitly excluded",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Zero performs load verification only")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--text-file", help="Optional UTF-8 training text; tokenized locally")
    parser.add_argument("--max-text-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit random-token mechanics tests; these are not quality evidence",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--algorithm", choices=("adafactor", "adamw", "sgd"), default="adafactor")
    parser.add_argument("--memory-budget-gib", type=float, default=14.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--skip-movement-check", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.max_steps and not args.text_file and not args.allow_synthetic:
        raise ValueError("Training requires --text-file or explicit --allow-synthetic")
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    if args.memory_budget_gib <= 0 or args.sample_interval <= 0:
        raise ValueError("Memory budget and sample interval must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")


def load_text_tokens(args: argparse.Namespace, checkpoint: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    path = Path(args.text_file).expanduser().resolve()
    if path.stat().st_size > args.max_text_bytes:
        raise ValueError(
            f"Text input is {path.stat().st_size} bytes, above --max-text-bytes={args.max_text_bytes}; "
            "use a bounded shard rather than tokenizing an unbounded corpus"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    text = path.read_text(encoding="utf-8")
    tokens = tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids[0]
    if tokens.numel() < args.sequence_length:
        raise ValueError("Tokenized text is shorter than --sequence-length")
    return tokens


def main(argv: Optional[List[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)

    output_path = Path(args.output).expanduser().resolve()
    sampler = RSSSampler(args.sample_interval)
    started = time.time()
    report: Dict[str, object] = {
        "schema_version": 1,
        "status": "loading",
        "complete_model_loaded": False,
        "training_completed": False,
        "training_data_kind": "text" if args.text_file else ("synthetic" if args.max_steps else "none"),
        "quality_verified": False,
        "execution_integration": {
            "weights": "checkpoint-direct E8 with LoRA",
            "attention": "stock HuggingFace model implementation",
            "loss": "stock HuggingFace labels path; not OjasX streaming cross-entropy",
            "activation_checkpointing": bool(args.max_steps and not args.no_activation_checkpointing),
            "max_grad_norm": None,
            "moment_quantization": False,
        },
        "configuration": {
            key: value for key, value in vars(args).items() if key not in {"output"}
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ojasx": ojasx2.__version__,
        },
        "steps": [],
        "started_unix": started,
    }
    sampler.start()
    write_report(output_path, report)

    try:
        load_started = time.perf_counter()
        dtype = getattr(torch, args.adapter_dtype)
        model, load_report = load_hf_checkpoint_e8(
            args.checkpoint,
            bitrate=args.bitrate,
            group_size=args.group_size,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            row_block=args.row_block,
            use_hadamard=args.use_hadamard,
            adapter_dtype=dtype,
            exclude_modules=tuple(args.exclude_module),
        )
        report["load_seconds"] = time.perf_counter() - load_started
        report["persistent_tensor_ledger"] = load_report.to_dict()
        report["complete_model_loaded"] = True
        report["status"] = "load_complete" if not args.max_steps else "training"
        report["rss"] = {
            "sample_count": sampler.samples,
            "sampled_peak_bytes": sampler.peak_bytes,
            "os_high_water_bytes": os_peak_rss_bytes(),
        }
        write_report(output_path, report)

        if args.max_steps:
            model.config.use_cache = False
            if not args.no_activation_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model.train()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            initial = None
            if not args.skip_movement_check:
                # This reference is an explicit runtime allocation included in
                # measured RSS; it is not hidden in the persistent-model ledger.
                initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
                report["movement_reference_bytes"] = sum(
                    tensor.numel() * tensor.element_size() for tensor in initial.values()
                )

            gradient_observation = {"sum_squares": 0.0, "count": 0}

            def observe_gradient(gradient: torch.Tensor) -> None:
                norm = float(gradient.detach().float().norm())
                gradient_observation["sum_squares"] += norm * norm
                gradient_observation["count"] += 1

            gradient_handles = [parameter.register_hook(observe_gradient) for parameter in trainable]
            optimizer = HolonomicGradFreeOptimizer(
                trainable,
                lr=args.learning_rate,
                algorithm=args.algorithm,
                max_grad_norm=None,
                quantize_moments_e8=False,
            )
            text_tokens = load_text_tokens(args, args.checkpoint) if args.text_file else None
            vocab_size = int(model.config.vocab_size)
            training_started = time.perf_counter()

            for step in range(1, args.max_steps + 1):
                gradient_observation["sum_squares"] = 0.0
                gradient_observation["count"] = 0
                if text_tokens is None:
                    input_ids = torch.randint(
                        0,
                        vocab_size,
                        (1, args.sequence_length),
                        generator=torch.Generator().manual_seed(args.seed + step),
                    )
                else:
                    max_start = int(text_tokens.numel() - args.sequence_length)
                    start_index = (step - 1) % (max_start + 1)
                    input_ids = text_tokens[start_index : start_index + args.sequence_length].unsqueeze(0)
                step_started = time.perf_counter()
                output = model(input_ids=input_ids, labels=input_ids)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
                loss_value = float(loss.detach())
                optimizer.zero_grad()
                loss.backward()  # In-backward hooks update and release each adapter gradient.
                gradient_sum_squares = float(gradient_observation["sum_squares"])
                if gradient_observation["count"] != len(trainable):
                    raise RuntimeError(
                        f"Only {gradient_observation['count']} of {len(trainable)} adapter gradients were observed"
                    )
                if not math.isfinite(gradient_sum_squares) or gradient_sum_squares <= 0:
                    raise FloatingPointError("Adapter gradients were zero or non-finite")
                step_record = {
                    "step": step,
                    "loss": loss_value,
                    "seconds": time.perf_counter() - step_started,
                    "observed_adapter_gradients": gradient_observation["count"],
                    "adapter_global_gradient_norm": math.sqrt(gradient_sum_squares),
                    "sampled_peak_rss_bytes": sampler.peak_bytes,
                    "os_high_water_rss_bytes": os_peak_rss_bytes(),
                }
                report["steps"].append(step_record)
                report["rss"] = {
                    "sample_count": sampler.samples,
                    "sampled_peak_bytes": sampler.peak_bytes,
                    "os_high_water_bytes": os_peak_rss_bytes(),
                }
                write_report(output_path, report)
                print(
                    f"step={step} loss={loss_value:.6f} "
                    f"grad_norm={step_record['adapter_global_gradient_norm']:.6f} "
                    f"seconds={step_record['seconds']:.3f} "
                    f"peak_rss_gib={max(sampler.peak_bytes, os_peak_rss_bytes()) / 1024**3:.3f}",
                    flush=True,
                )

            report["training_seconds"] = time.perf_counter() - training_started
            report["training_completed"] = True
            if initial is not None:
                delta_sum_squares = 0.0
                for name, parameter in model.named_parameters():
                    if name in initial:
                        delta = parameter.detach().float() - initial[name].float()
                        delta_sum_squares += float(delta.square().sum())
                report["trainable_parameter_delta_l2"] = math.sqrt(delta_sum_squares)
                if not math.isfinite(delta_sum_squares) or delta_sum_squares <= 0:
                    raise RuntimeError("No finite nonzero trainable-parameter movement was observed")
            for handle in gradient_handles:
                handle.remove()

        sampler.stop()
        observed_peak = max(sampler.peak_bytes, os_peak_rss_bytes())
        budget = int(args.memory_budget_gib * 1024 ** 3)
        report["status"] = "completed"
        report["finished_unix"] = time.time()
        report["rss"] = {
            "sample_count": sampler.samples,
            "sampled_peak_bytes": sampler.peak_bytes,
            "os_high_water_bytes": os_peak_rss_bytes(),
            "observed_peak_bytes": observed_peak,
            "budget_bytes": budget,
            "within_budget": observed_peak <= budget,
        }
        report["verdict"] = {
            "scope": "complete_model_load" if not args.max_steps else "complete_model_e8_lora_training",
            "all_checkpoint_tensors_materialized_or_packed": True,
            "representative_layer_extrapolation": False,
            "quality_verified": False,
            "within_observed_memory_budget": observed_peak <= budget,
        }
        write_report(output_path, report)
        print(json.dumps(report["verdict"], indent=2), flush=True)
        print(f"Evidence: {output_path}", flush=True)
        return 0
    except Exception as exc:
        sampler.stop()
        report["status"] = "failed"
        report["finished_unix"] = time.time()
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
        report["rss"] = {
            "sample_count": sampler.samples,
            "sampled_peak_bytes": sampler.peak_bytes,
            "os_high_water_bytes": os_peak_rss_bytes(),
        }
        write_report(output_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
