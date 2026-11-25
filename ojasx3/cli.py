"""Turnkey Command-Line Interface for OjasX3."""

from __future__ import annotations
import argparse
import json
import sys
import torch

import ojasx3
from ojasx3.core.backend import detect_hardware_target
from ojasx2.doctor import print_doctor_report
from ojasx3.trainer.custom_llm import CUSTOM_LLM_PRESETS, train_custom_llm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ojasx3",
        description="OjasX3 Turnkey AI Compute & Custom LLM Training CLI",
    )
    parser.add_argument("--version", action="version", version=f"OjasX3 v{ojasx3.__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Doctor command
    subparsers.add_parser("doctor", help="Run hardware and system diagnostic check")

    # Benchmark command
    bench_p = subparsers.add_parser("benchmark", help="Run multi-layer GPU training benchmark")
    bench_p.add_argument("--scale", default="1B", choices=["1B", "4B", "8B", "14B", "27B"], help="Model scale")

    # Train custom LLM from scratch
    train_p = subparsers.add_parser(
        "train-llm",
        help="Train a custom LLM from scratch on .txt / .jsonl data with Dual-Rail, E8, or BF16",
    )
    train_p.add_argument(
        "--preset",
        default="tiny-50m",
        choices=list(CUSTOM_LLM_PRESETS.keys()),
        help="Custom LLM architecture preset",
    )
    train_p.add_argument("--data", default=None, help="Path to training .txt / .jsonl file or raw text")
    train_p.add_argument("--mode", default="dual_rail", choices=["dual_rail", "e8", "bf16"], help="Weight quantization/training mode")
    train_p.add_argument("--steps", type=int, default=20, help="Number of training steps")
    train_p.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    train_p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    train_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_p.add_argument("--lora-r", type=int, default=16, help="LoRA rank when using dual_rail or e8 mode")
    train_p.add_argument("--device", default="auto", help="Device (auto, mps, cuda, cpu)")
    train_p.add_argument("--save-dir", default=None, help="Optional output directory to save model & telemetry")

    # Fine-tune pretrained HuggingFace LLM
    ft_p = subparsers.add_parser(
        "finetune-llm",
        help="Fine-tune any HuggingFace or local LLM checkpoint in 2.125-bit Dual-Rail + BF16 LoRA",
    )
    ft_p.add_argument("--model", required=True, help="HuggingFace model ID (e.g., Qwen/Qwen2.5-1.5B or Qwen/Qwen3-8B) or local path")
    ft_p.add_argument("--data", default=None, help="Path to custom .txt / .jsonl fine-tuning dataset")
    ft_p.add_argument("--steps", type=int, default=20, help="Number of fine-tuning steps")
    ft_p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    ft_p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    ft_p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ft_p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    ft_p.add_argument("--device", default="auto", help="Device (auto, mps, cuda, cpu)")
    ft_p.add_argument("--save-dir", default=None, help="Optional directory to export LoRA adapter & telemetry")

    args = parser.parse_args(argv)

    if args.command == "doctor" or args.command is None:
        hw = detect_hardware_target()
        print(f"OjasX3 v{ojasx3.__version__} Doctor")
        print(f"Active Hardware: {hw.architecture} ({hw.device_type}), Vector Width: {hw.vector_width_bits}b, Preferred Executor: {hw.preferred_executor}")
        print_doctor_report()
        return 0

    if args.command == "benchmark":
        import time
        from ojasx2.math.lattice import E8LatticeLinear
        print(f"Starting OjasX3 self-contained benchmark for scale {args.scale}...")
        dim_map = {"1B": (2048, 2048), "4B": (3072, 3072), "8B": (4096, 4096), "14B": (5120, 5120), "27B": (6144, 6144)}
        in_f, out_f = dim_map[args.scale]
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        layer = E8LatticeLinear(in_f, out_f, bitrate=1.5, group_size=32, rank=16).to(dev)
        x = torch.randn(2, 32, in_f, device=dev)
        t0 = time.perf_counter()
        y = layer(x)
        y.sum().backward()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Completed {args.scale} E8+LoRA layer step on {dev}: {dt_ms:.2f} ms")
        return 0

    if args.command == "train-llm":
        res = train_custom_llm(
            args.data,
            preset=args.preset,
            mode=args.mode,
            steps=args.steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            device=args.device,
            save_dir=args.save_dir,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.command == "finetune-llm":
        res = train_custom_llm(
            args.data,
            checkpoint=args.model,
            mode="dual_rail",
            steps=args.steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            device=args.device,
            save_dir=args.save_dir,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
