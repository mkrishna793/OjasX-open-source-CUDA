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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
