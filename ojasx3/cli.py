"""Turnkey Command-Line Interface for OjasX3."""

from __future__ import annotations
import argparse
import sys
import torch

import ojasx3
from ojasx3.core.backend import detect_hardware_target
from ojasx2.doctor import print_doctor_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ojasx3",
        description="OjasX3 Turnkey AI Compute & Training CLI",
    )
    parser.add_argument("--version", action="version", version=f"OjasX3 v{ojasx3.__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Doctor command
    doc_p = subparsers.add_parser("doctor", help="Run hardware and system diagnostic check")

    # Benchmark command
    bench_p = subparsers.add_parser("benchmark", help="Run multi-layer GPU training benchmark")
    bench_p.add_argument("--scale", default="1B", choices=["1B", "4B", "8B", "14B", "27B"], help="Model scale")

    args = parser.parse_args(argv)

    if args.command == "doctor" or args.command is None:
        hw = detect_hardware_target()
        print(f"OjasX3 v{ojasx3.__version__} Doctor")
        print(f"Active Hardware: {hw.architecture} ({hw.device_type}), Vector Width: {hw.vector_width_bits}b, Preferred Executor: {hw.preferred_executor}")
        print_doctor_report()
        return 0

    if args.command == "benchmark":
        print(f"Starting OjasX3 benchmark for scale {args.scale}...")
        from tools.run_1b_to_27b_apple_m5_gpu_benchmark import run_single_scale
        ladder = {"1B": (20, 2048, 5632, 16, 16384), "4B": (34, 3072, 8192, 24, 16384), "8B": (32, 4096, 14336, 32, 16384), "14B": (44, 5120, 13824, 40, 16384), "27B": (60, 6144, 16384, 48, 16384)}
        nl, dm, df, nh, vs = ladder[args.scale]
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        res = run_single_scale(args.scale, nl, dm, df, nh, vs, steps=1, device=dev)
        print(f"Completed {args.scale} benchmark: Peak RAM = {res['memory_gb']['mps_peak_training_allocated_gb']} GB")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
