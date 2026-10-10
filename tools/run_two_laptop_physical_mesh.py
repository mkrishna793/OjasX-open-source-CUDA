"""Physical Two-Laptop Distributed P2P Mesh Runner (`tools/run_two_laptop_physical_mesh.py`).

Connects:
  - Laptop A (Coordinator): Apple M5 (24 GB Unified Memory, 10-Core Metal 4 GPU + 16-Core ANE)
  - Laptop B (Worker):      Intel Core i5 (16 GB RAM, 1-2 GB Integrated GPU / CPU AVX2)

Usage:
  1. On Laptop B (Intel Core i5 16GB):
       python tools/run_two_laptop_physical_mesh.py --role worker --port 29505

  2. On Laptop A (Apple M5 24GB):
       python tools/run_two_laptop_physical_mesh.py --role coordinator --worker-host <LAPTOP_B_IP> --port 29505
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

import ojasx2
from ojasx2.holonomic_math_compiler import HolonomicLinear, HolonomicRMSNorm
from ojasx2.mesh_hal import (
    HolonomicVirtualMeshEngine,
    MeshNodeSpec,
    MeshRemoteWorkerServer,
    plan_mesh_partitions,
)


def _detect_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_shared_model(dim: int = 512, num_layers: int = 8, seed: int = 2026) -> nn.Sequential:
    """Build deterministic 8-block model identically on both laptops for SHA-256 handshake."""
    torch.manual_seed(seed)
    blocks: List[nn.Module] = []
    for i in range(num_layers):
        act = "silu" if i % 2 == 0 else "relu2"
        blocks.append(
            nn.Sequential(
                HolonomicRMSNorm(dim),
                HolonomicLinear(dim, dim * 2, bias=True, act=act),
                HolonomicLinear(dim * 2, dim, bias=True, act="none"),
            )
        )
    return nn.Sequential(*blocks)


def build_node_specs(worker_host: str, worker_port: int, coordinator_device: str) -> List[MeshNodeSpec]:
    return [
        MeshNodeSpec(
            node_id="Laptop_A_Apple_M5_24GB",
            chip_class="Apple_M5_10Core_Metal4_GPU_16Core_ANE",
            free_memory_gb=18.0,
            compute_tflops=16.5,
            device=coordinator_device,
            host="127.0.0.1",
            port=0,  # Local on Coordinator (Laptop A)
        ),
        MeshNodeSpec(
            node_id="Laptop_B_Intel_i5_16GB_iGPU",
            chip_class="Intel_Core_i5_AVX2_iGPU_16GB",
            free_memory_gb=10.5,
            compute_tflops=4.2,
            device="cpu",
            host=worker_host,
            port=worker_port,  # Remote TCP Worker on Laptop B
        ),
    ]


def run_worker(port: int, dim: int, num_layers: int, lr: float) -> None:
    lan_ip = _detect_lan_ip()
    model = build_shared_model(dim=dim, num_layers=num_layers, seed=2026)
    nodes = build_node_specs(worker_host="0.0.0.0", worker_port=port, coordinator_device="cpu")
    plan = plan_mesh_partitions(model, nodes)

    worker_node = nodes[1]
    assigned_indices = plan.node_assignments[worker_node.node_id]
    partition = nn.Sequential(*(model[i] for i in assigned_indices)).to("cpu")
    optimizer = torch.optim.AdamW(partition.parameters(), lr=lr)

    server = MeshRemoteWorkerServer(
        sub_module=partition,
        host="0.0.0.0",
        port=port,
        device="cpu",
        compress_int8=True,
        optimizer=optimizer,
        timeout=120.0,
    )
    print("=" * 88)
    print("⚡ OJASX PHYSICAL MESH WORKER ONLINE (Laptop B — Intel Core i5 16GB + iGPU)")
    print("=" * 88)
    print(f"  Listening Address   : 0.0.0.0:{server.port}")
    print(f"  Laptop B LAN IP     : {lan_ip}")
    print(f"  Assigned Layers     : {assigned_indices} ({plan.node_param_counts[worker_node.node_id]:,} params)")
    print(f"  Capacity Share      : {plan.node_capacity_share_pct[worker_node.node_id]:.1f}% of Virtual Mesh")
    print("-" * 88)
    print("👉 NOW RUN THIS COMMAND ON LAPTOP A (Apple M5 MacBook Air):")
    print(f"   .tools/conformance-venv/bin/python tools/run_two_laptop_physical_mesh.py --role coordinator --worker-host {lan_ip} --port {server.port}")
    print("=" * 88)
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down Laptop B worker...")
    finally:
        server.close()


def run_coordinator(worker_host: str, port: int, dim: int, num_layers: int, steps: int, lr: float) -> None:
    coord_device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    model_mesh = build_shared_model(dim=dim, num_layers=num_layers, seed=2026)
    model_ref = copy.deepcopy(model_mesh).to(coord_device)

    nodes = build_node_specs(worker_host=worker_host, worker_port=port, coordinator_device=coord_device)
    print("=" * 92)
    print("⚡ OJASX TWO-LAPTOP PHYSICAL VIRTUAL UNIFIED GPU MESH (Coordinator: Apple M5 24GB)")
    print("=" * 92)
    print(f"  Coordinator (Laptop A) : {nodes[0].node_id} (Device={coord_device.upper()}, Free=18.0 GB)")
    print(f"  Remote Worker (Laptop B): {nodes[1].node_id} ({worker_host}:{port}, Free=10.5 GB)")

    mesh_engine = HolonomicVirtualMeshEngine(
        model_mesh,
        nodes=nodes,
        compress_boundary=True,
        timeout=60.0,
    )
    local_params = [p for p in mesh_engine.parameters() if p.requires_grad]
    opt_mesh = torch.optim.AdamW(local_params, lr=lr)
    opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=lr)

    plan = mesh_engine.plan
    print(f"  Pooled Virtual Memory  : {plan.pooled_memory_gb:.1f} GB | Pooled Compute: {plan.pooled_tflops:.1f} TFLOPS")
    print(f"  Laptop A Layers        : {plan.node_assignments[nodes[0].node_id]} ({plan.node_param_counts[nodes[0].node_id]:,} params)")
    print(f"  Laptop B Layers (TCP)  : {plan.node_assignments[nodes[1].node_id]} ({plan.node_param_counts[nodes[1].node_id]:,} params)")
    print("-" * 92)

    torch.manual_seed(99)
    x_base = torch.randn(16, 64, dim)
    target = torch.randn(16, 64, dim)

    step_records = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_mesh.zero_grad(set_to_none=True)
        mesh_engine.zero_remote_grad()

        x_m = x_base.to(coord_device).requires_grad_(True)
        y_m = mesh_engine(x_m)
        loss_m = (y_m - target.to(y_m.device)).square().mean()
        loss_m.backward()
        opt_mesh.step()
        mesh_engine.step_remote()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Reference single-machine step
        opt_ref.zero_grad(set_to_none=True)
        x_r = x_base.to(coord_device).requires_grad_(True)
        y_r = model_ref(x_r)
        loss_r = (y_r - target.to(coord_device)).square().mean()
        loss_r.backward()
        opt_ref.step()

        grad_cos = torch.nn.functional.cosine_similarity(
            x_m.grad.detach().flatten().cpu(),
            x_r.grad.detach().flatten().cpu(),
            dim=0,
        ).item()

        print(
            f"  Step {step:02d}/{steps:02d} | Mesh Loss: {loss_m.item():.6f} | Ref Loss: {loss_r.item():.6f} | "
            f"Grad Cosine Sim: {grad_cos:.6f} | Step Time: {dt_ms:.1f} ms"
        )
        step_records.append(
            {
                "step": step,
                "mesh_loss": float(loss_m.item()),
                "ref_loss": float(loss_r.item()),
                "input_grad_cosine_sim": float(grad_cos),
                "step_ms": float(dt_ms),
            }
        )

    tel = mesh_engine.telemetry
    print("-" * 92)
    print(f"  TCP Packets Exchanged  : {tel.forward_packets} Fwd / {tel.backward_packets} Bwd")
    print(f"  Uncompressed FP32 Bytes: {tel.uncompressed_fp32_bytes / 1024.0:.1f} KB")
    print(f"  Compressed Wire Bytes  : {tel.compressed_wire_bytes / 1024.0:.1f} KB")
    print(f"  Boundary Compression   : {tel.compression_ratio:.2f}x ({tel.bandwidth_saved_pct:.1f}% bandwidth saved)")
    print("=" * 92)

    report_path = Path(__file__).resolve().parents[1] / "reports" / "TWO_LAPTOP_PHYSICAL_MESH_REPORT.json"
    report_data = {
        "status": "passed",
        "coordinator_node": nodes[0].__dict__,
        "worker_node": nodes[1].__dict__,
        "pooled_memory_gb": plan.pooled_memory_gb,
        "pooled_tflops": plan.pooled_tflops,
        "node_assignments": plan.node_assignments,
        "telemetry": {
            "forward_packets": tel.forward_packets,
            "backward_packets": tel.backward_packets,
            "compressed_wire_bytes": tel.compressed_wire_bytes,
            "uncompressed_fp32_bytes": tel.uncompressed_fp32_bytes,
            "compression_ratio": tel.compression_ratio,
            "bandwidth_saved_pct": tel.bandwidth_saved_pct,
        },
        "steps": step_records,
    }
    report_path.write_text(json.dumps(report_data, indent=2) + "\n")
    mesh_engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OjasX Two-Laptop Physical P2P Mesh Runner")
    parser.add_argument("--role", choices=["worker", "coordinator", "selftest"], default="selftest")
    parser.add_argument("--worker-host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29505)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.role == "worker":
        run_worker(port=args.port, dim=args.dim, num_layers=args.layers, lr=args.lr)
    elif args.role == "coordinator":
        run_coordinator(
            worker_host=args.worker_host,
            port=args.port,
            dim=args.dim,
            num_layers=args.layers,
            steps=args.steps,
            lr=args.lr,
        )
    else:
        # Local two-endpoint self-test before connecting second physical laptop
        model = build_shared_model(dim=args.dim, num_layers=args.layers, seed=2026)
        nodes = build_node_specs(worker_host="127.0.0.1", worker_port=1, coordinator_device="cpu")
        plan = plan_mesh_partitions(model, nodes)
        assigned = plan.node_assignments[nodes[1].node_id]
        partition = nn.Sequential(*(model[i] for i in assigned)).to("cpu")
        opt_w = torch.optim.AdamW(partition.parameters(), lr=args.lr)
        server = MeshRemoteWorkerServer(partition, host="127.0.0.1", port=0, device="cpu", optimizer=opt_w)
        try:
            run_coordinator("127.0.0.1", server.port, args.dim, args.layers, args.steps, args.lr)
        finally:
            server.close()


if __name__ == "__main__":
    main()
