#!/usr/bin/env python3
"""Exercise M4 discovery and bitwise execution on a four-sigmoid chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ojasx2.core import (
    JointGraphBuilder,
    M4Config,
    M4Input,
    M4Problem,
    activation,
    execute_joint_graph,
    plan_m25_memory,
    plan_m4,
    tensor_input,
)


def run() -> dict[str, object]:
    spec = M4Input("x", (8,), dtype=torch.float64)
    expression = tensor_input(spec)
    for _ in range(4):
        expression = activation(expression, "sigmoid")
    problem = M4Problem((spec,), expression)
    config = M4Config(
        max_iterations=8,
        beam_width=32,
        max_candidates=64,
        exact_node_limit=32,
    )
    plan = plan_m4(problem, config=config)
    repeated = plan_m4(problem, config=config)

    builder = JointGraphBuilder()
    x = builder.input("x", (8,), dtype=torch.float64, requires_grad=True)
    y = x
    for _ in range(4):
        y = builder.forward("sigmoid", (y,))
    builder.request_gradient(x)
    builder.set_objective(y)
    m25 = plan_m25_memory(builder, exact_node_limit=32)

    value = torch.linspace(-0.3, 0.4, 8, dtype=torch.float64)
    selected = execute_joint_graph(plan.graph, plan.certificate, {0: value})
    baseline = execute_joint_graph(
        plan.baseline_graph,
        plan.baseline_certificate,
        {0: value},
    )
    output_equal = torch.equal(
        selected[plan.graph.objective],
        baseline[plan.baseline_graph.objective],
    )
    gradient_equal = torch.equal(
        selected[plan.graph.gradient_targets[0]],
        baseline[plan.baseline_graph.gradient_targets[0]],
    )
    deterministic = plan.to_dict() == repeated.to_dict()
    passed = all(
        (
            plan.discovered,
            plan.saturation.saturated,
            plan.evaluated_candidates > 2,
            bool(plan.proof_obligations),
            plan.certificate.peak_bytes < plan.baseline_certificate.peak_bytes,
            plan.certificate.peak_bytes < m25.certificate.peak_bytes,
            output_equal,
            gradient_equal,
            deterministic,
        )
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "torch_version": torch.__version__,
        "case": "four float64 sigmoid activations, eight elements, input gradient",
        "plan": plan.to_dict(),
        "m25_peak_bytes": m25.certificate.peak_bytes,
        "execution": {
            "output_bitwise_equal": output_equal,
            "input_gradient_bitwise_equal": gradient_equal,
        },
        "deterministic_repeated_plan": deterministic,
        "scope": (
            "Logical memory/work/traffic certificate and CPU execution smoke; "
            "not measured hardware traffic, FLOPs, or CUDA qualification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/core-m4-smoke.json",
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    plan = report["plan"]
    print(
        f"M4 smoke: {report['status']}; "
        f"peak {plan['baseline_peak_bytes']} -> {plan['peak_bytes']} bytes; "
        f"M2.5 {report['m25_peak_bytes']} bytes; "
        f"{plan['evaluated_candidates']} candidates"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
