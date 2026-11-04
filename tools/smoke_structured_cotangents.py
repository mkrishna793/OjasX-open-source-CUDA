#!/usr/bin/env python3
"""Execute dense/structured M0/M1 differential and logical-storage smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ojasx2.core import (
    JointGraphBuilder,
    analyze_cotangent_topology,
    execute_joint_graph,
    plan_m25_memory,
    plan_memory,
)


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    integer_dtype = {torch.float32: torch.int32, torch.float64: torch.int64}[left.dtype]
    return bool(torch.equal(left.view(integer_dtype), right.view(integer_dtype)))


def _execute_case(name, build: Callable, value: torch.Tensor) -> dict:
    results = {}
    for structured in (False, True):
        builder = JointGraphBuilder(structured_cotangents=structured)
        source, objective = build(builder)
        builder.request_gradient(source)
        builder.set_objective(objective)
        graph = builder.generate_reverse()
        certificate = plan_memory(graph)
        output = execute_joint_graph(graph, certificate, {source: value})
        results[structured] = (graph, certificate, output)
    dense_graph, dense_certificate, dense = results[False]
    graph, certificate, result = results[True]
    topology = analyze_cotangent_topology(graph)
    gradient_bitwise = _bits_equal(
        dense[dense_graph.gradient_targets[0]],
        result[graph.gradient_targets[0]],
    )
    objective_bitwise = _bits_equal(
        dense[dense_graph.objective], result[graph.objective]
    )
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "dense_peak_bytes": dense_certificate.peak_bytes,
        "structured_peak_bytes": certificate.peak_bytes,
        "peak_saved_bytes": dense_certificate.peak_bytes - certificate.peak_bytes,
        "topology": topology.to_dict(),
        "objective_bitwise_equal": objective_bitwise,
        "gradient_bitwise_equal": gradient_bitwise,
        "structured_graph_digest": certificate.graph_digest,
    }


def _sigmoid(builder):
    source = builder.input("x", (4, 8), dtype=torch.float64, requires_grad=True)
    return source, builder.forward("sigmoid", (source,))


def _views(builder):
    source = builder.input("x", (2, 3, 4), dtype=torch.float32, requires_grad=True)
    value = builder.forward("reshape", (source,), attrs={"shape": (4, 6)})
    value = builder.forward("transpose", (value,), attrs={"dim0": 0, "dim1": 1})
    return source, builder.forward("sum", (value,))


def _fanout(builder):
    source = builder.input("x", (2, 3), dtype=torch.float64, requires_grad=True)
    value = builder.forward("add", (source, source))
    return source, builder.forward("sum", (value,))


def verify() -> dict:
    torch.manual_seed(20261103)
    cases = [
        _execute_case("sigmoid_dense_boundary", _sigmoid, torch.randn(4, 8, dtype=torch.float64)),
        _execute_case("reshape_transpose_sum", _views, torch.randn(2, 3, 4)),
        _execute_case("additive_fanout_sum", _fanout, torch.randn(2, 3, dtype=torch.float64)),
    ]
    policies = {}
    for structured in (False, True):
        builder = JointGraphBuilder(structured_cotangents=structured)
        source = builder.input("x", (64,), dtype=torch.float32, requires_grad=True)
        objective = builder.forward("relu", (source,))
        builder.request_gradient(source)
        builder.set_objective(objective)
        plan = plan_m25_memory(builder, fuse=False)
        policies["structured" if structured else "dense"] = {
            "policy": dict(plan.policy_choices)[0],
            "peak_bytes": plan.certificate.peak_bytes,
        }
    passed = all(
        case["objective_bitwise_equal"]
        and case["gradient_bitwise_equal"]
        and case["structured_peak_bytes"] <= case["dense_peak_bytes"]
        and case["topology"]["managed_bytes"] <= case["topology"]["logical_bytes"]
        for case in cases
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "torch_version": torch.__version__,
        "cases": cases,
        "relu_policy_selection": policies,
        "scope": (
            "CPU bitwise differential execution and logical certificate metrics; "
            "region savings are not necessarily peak savings; no allocator, "
            "FLOP, hardware-traffic, CUDA, or speed claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/core-structured-smoke.json"
    )
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(
            f"{case['name']}: peak {case['dense_peak_bytes']} -> "
            f"{case['structured_peak_bytes']}; region saved "
            f"{case['topology']['saved_bytes']} bytes; bitwise="
            f"{case['gradient_bitwise_equal']}"
        )
    print(report["status"].upper())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
