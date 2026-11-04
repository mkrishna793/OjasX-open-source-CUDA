"""Shared elementwise activation semantics for unfused and replay-fused nodes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .jets import Jet


ACTIVATION_NAMES = (
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "silu",
    "gelu",
    "square",
    "neg",
)


def activation_program(attrs: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Parse a stable ``(name, approximation)`` program from graph attributes."""

    operations = attrs.get("operations")
    if not isinstance(operations, (tuple, list)) or not operations:
        raise ValueError("activation_chain requires a nonempty operations attribute")
    program = []
    for operation in operations:
        if not isinstance(operation, str) or not operation:
            raise ValueError("Activation-chain entries must be nonempty strings")
        parts = operation.split(":")
        if len(parts) > 2:
            raise ValueError(f"Invalid activation-chain entry {operation!r}")
        kind = parts[0]
        approximation = parts[1] if len(parts) == 2 else "none"
        if kind not in ACTIVATION_NAMES:
            raise ValueError(f"Unsupported activation-chain operation {kind!r}")
        if kind == "gelu":
            if approximation not in {"none", "tanh"}:
                raise ValueError("GELU approximation must be 'none' or 'tanh'")
        elif len(parts) == 2:
            raise ValueError(f"Activation {kind!r} has no approximation parameter")
        program.append((kind, approximation))
    return tuple(program)


def activation_forward(kind: str, tensor: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if kind == "relu":
        return F.relu(tensor)
    if kind == "sigmoid":
        return tensor.sigmoid()
    if kind == "tanh":
        return tensor.tanh()
    if kind == "exp":
        return tensor.exp()
    if kind == "log":
        return tensor.log()
    if kind == "sqrt":
        return tensor.sqrt()
    if kind == "sin":
        return tensor.sin()
    if kind == "cos":
        return tensor.cos()
    if kind == "silu":
        return F.silu(tensor)
    if kind == "gelu":
        return F.gelu(tensor, approximate=approximate)
    if kind == "square":
        return tensor.square()
    if kind == "neg":
        return -tensor
    raise ValueError(f"Unknown elementwise activation {kind!r}")


def activation_backward(
    kind: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    upstream: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    """Apply the established first-order pullback with explicit residuals."""

    if kind == "relu":
        mask = (x > 0) | torch.isnan(x)
        return torch.where(mask, upstream, 0)
    if kind == "sigmoid":
        return upstream * y * (1.0 - y)
    if kind == "tanh":
        return upstream * (1.0 - y.square())
    if kind == "exp":
        return upstream * y
    if kind == "log":
        return upstream / x
    if kind == "sqrt":
        return upstream / (2.0 * y)
    if kind == "sin":
        return upstream * x.cos()
    if kind == "cos":
        return -upstream * x.sin()
    if kind == "silu":
        s = x.sigmoid()
        return upstream * (s + x * s * (1.0 - s))
    if kind == "gelu" and approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x.pow(3))
        t = u.tanh()
        du = c * (1.0 + 3.0 * 0.044715 * x.square())
        return upstream * 0.5 * (1.0 + t + x * (1.0 - t.square()) * du)
    if kind == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        density = torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        return upstream * (cdf + x * density)
    if kind == "square":
        return 2.0 * x * upstream
    if kind == "neg":
        return -upstream
    raise ValueError(f"Unknown elementwise activation {kind!r}")


def activation_jet(jet: Jet, kind: str, approximate: str = "none") -> Jet:
    if kind == "relu":
        value = jet.value
        if not value.is_meta and jet.order > 1 and (
            bool(torch.any(value == 0)) or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError("Higher-order ReLU jets require finite inputs away from the kink")
        mask = (value > 0).to(value.dtype)
        return (jet * mask).set_value(F.relu(value))
    if kind == "gelu" and approximate == "tanh":
        result = 0.5 * jet * (
            1.0 + (math.sqrt(2.0 / math.pi) * (jet + 0.044715 * jet.pow_scalar(3))).tanh()
        )
        return result.set_value(F.gelu(jet.value, approximate="tanh"))
    if kind == "neg":
        return -jet
    if kind == "square":
        return jet.pow_scalar(2)
    return getattr(jet, kind)()


def layer_backward_elements(kind: str) -> int:
    if kind == "relu":
        return 3
    if kind in {"gelu", "silu"}:
        return 8
    return 2


def streaming_replay_workspace_elements(program: Sequence[Tuple[str, str]]) -> int:
    # Two recomputed primal tensors (layer input and output) plus the largest
    # per-layer pullback temporary declaration. It is independent of depth.
    return 2 + max((layer_backward_elements(kind) for kind, _ in program), default=0)


def cached_replay_workspace_elements(program: Sequence[Tuple[str, str]]) -> int:
    # One full recomputed value per layer plus the largest pullback workspace.
    return len(program) + max((layer_backward_elements(kind) for kind, _ in program), default=0)


def streaming_replay_evaluations(program: Sequence[Tuple[str, str]]) -> int:
    # Layer i replays i forward operations: sum(i, i=1..L).
    return len(program) * (len(program) + 1) // 2


__all__ = [
    "ACTIVATION_NAMES",
    "activation_backward",
    "activation_forward",
    "activation_jet",
    "activation_program",
    "cached_replay_workspace_elements",
    "layer_backward_elements",
    "streaming_replay_evaluations",
    "streaming_replay_workspace_elements",
]
