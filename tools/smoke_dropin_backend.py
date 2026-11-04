#!/usr/bin/env python3
"""Exercise the unchanged-model drop-in backend through accumulated training."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ojasx2.core import OjasXBackend


class NativeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.input_projection = torch.nn.Linear(4, 6, dtype=torch.float64)
        self.output_projection = torch.nn.Linear(6, 3, dtype=torch.float64)

    def forward(self, value):
        hidden = self.input_projection(value)
        for _ in range(4):
            hidden = torch.sigmoid(hidden)
        return self.output_projection(hidden)


def _fallback_check() -> dict:
    def original(value):
        return torch.tanh(torch.sigmoid(value)).square().mean()

    backend = OjasXBackend()
    compiled = torch.compile(original, backend=backend, fullgraph=True)
    value = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64, requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = compiled(value)
    expected = original(reference)
    actual.backward()
    expected.backward()
    return {
        "output_bitwise_equal": torch.equal(actual, expected),
        "gradient_bitwise_equal": torch.equal(value.grad, reference.grad),
        "changed": backend.reports[-1].changed,
        "report": asdict(backend.reports[-1]),
    }


def verify() -> dict:
    actual_model = NativeMLP()
    expected_model = copy.deepcopy(actual_model)
    backend = OjasXBackend()
    compiled = torch.compile(actual_model, backend=backend, fullgraph=True)
    actual_optimizer = torch.optim.AdamW(actual_model.parameters(), lr=0.01, weight_decay=0.02)
    expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=0.01, weight_decay=0.02)
    max_loss_difference = 0.0
    max_gradient_difference = 0.0
    max_parameter_difference = 0.0
    for step in range(2):
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        for microstep in range(2):
            torch.manual_seed(700 + step * 2 + microstep)
            value = torch.randn(5, 4, dtype=torch.float64)
            target = torch.randn(5, 3, dtype=torch.float64)
            actual_loss = (compiled(value) - target).square().mean() / 2
            expected_loss = (expected_model(value) - target).square().mean() / 2
            max_loss_difference = max(
                max_loss_difference,
                abs((actual_loss - expected_loss).detach().item()),
            )
            actual_loss.backward()
            expected_loss.backward()
        for actual, expected in zip(actual_model.parameters(), expected_model.parameters()):
            max_gradient_difference = max(
                max_gradient_difference,
                (actual.grad - expected.grad).abs().max().item(),
            )
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(actual_model.parameters(), expected_model.parameters()):
            max_parameter_difference = max(
                max_parameter_difference,
                (actual - expected).abs().max().item(),
            )
    fallback = _fallback_check()
    passed = (
        max_loss_difference <= 2e-11
        and max_gradient_difference <= 2e-10
        and max_parameter_difference <= 2e-9
        and any(report.changed for report in backend.reports)
        and not fallback["changed"]
        and fallback["output_bitwise_equal"]
        and fallback["gradient_bitwise_equal"]
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "torch_version": torch.__version__,
        "case": "unchanged native MLP, four sigmoid activations, AdamW, two steps, two microsteps",
        "max_loss_difference": max_loss_difference,
        "max_gradient_difference": max_gradient_difference,
        "max_parameter_difference": max_parameter_difference,
        "tolerances": {"loss": 2e-11, "gradient": 2e-10, "parameters": 2e-9},
        "compilation_reports": [asdict(report) for report in backend.reports],
        "original_fallback": fallback,
        "scope": (
            "CPU eager reference versus Dynamo/AOT backend with accumulated "
            "model-level training; logical planning costs only, no speed, "
            "allocator, CUDA, or whole-model coverage claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/core-dropin-backend-smoke.json",
    )
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"loss={report['max_loss_difference']:.3e} "
        f"gradient={report['max_gradient_difference']:.3e} "
        f"parameters={report['max_parameter_difference']:.3e}"
    )
    print(report["status"].upper())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
