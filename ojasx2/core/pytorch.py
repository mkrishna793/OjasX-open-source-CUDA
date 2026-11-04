"""M3 eager/compiled PyTorch integration for M2.5 activation replay.

The registered operators are opaque to Dynamo, while AOT Autograd uses the
registered pullback and its fake kernel. Only the original input is saved for
backward; intermediate activations are replayed by the selected M2.5 policy.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch

from .activation_math import (
    activation_backward,
    activation_forward,
    activation_program,
)


_FORWARD_NAME = "ojasx_verified::activation_chain"
_BACKWARD_NAME = "ojasx_verified::activation_chain_backward"
_FORWARD_SCHEMA = (
    '(Tensor x, str[] operations, str replay="replay_streaming") -> Tensor'
)
_BACKWARD_SCHEMA = (
    "(Tensor x, Tensor upstream, str[] operations, "
    'str replay="replay_streaming") -> Tensor'
)
_REPLAY_POLICIES = frozenset({"replay_streaming", "replay_cached"})


def _validated_program(operations: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    if isinstance(operations, (str, bytes)):
        raise TypeError("operations must be a sequence of activation names, not a string")
    return activation_program({"operations": tuple(operations)})


def _validate_replay(replay: str) -> None:
    if replay not in _REPLAY_POLICIES:
        raise ValueError(f"Unknown activation replay policy {replay!r}")


def _validate_forward(x: torch.Tensor, operations: Sequence[str], replay: str) -> None:
    if not x.is_floating_point():
        raise TypeError("activation_chain requires a floating-point tensor")
    _validated_program(operations)
    _validate_replay(replay)


def _validate_backward(
    x: torch.Tensor,
    upstream: torch.Tensor,
    operations: Sequence[str],
    replay: str,
) -> None:
    _validate_forward(x, operations, replay)
    if upstream.shape != x.shape:
        raise ValueError("The activation-chain upstream cotangent must match x's shape")
    if upstream.dtype != x.dtype or upstream.device != x.device:
        raise ValueError("The upstream cotangent and x must have matching dtype and device")


def _replay_backward(
    x: torch.Tensor,
    upstream: torch.Tensor,
    operations: Sequence[str],
    replay: str,
) -> torch.Tensor:
    program = _validated_program(operations)
    if replay == "replay_cached":
        values = [x]
        for kind, approximate in program:
            values.append(activation_forward(kind, values[-1], approximate))
        for index in range(len(program) - 1, -1, -1):
            output = values.pop()
            kind, approximate = program[index]
            upstream = activation_backward(
                kind,
                values[-1],
                output,
                upstream,
                approximate,
            )
        return upstream

    # Keep this recurrence identical to ActivationChainOperator: every local
    # pullback replays only its required prefix and releases it on return.
    for index in range(len(program) - 1, -1, -1):
        value = x
        for prefix in range(index + 1):
            kind, approximate = program[prefix]
            output = activation_forward(kind, value, approximate)
            if prefix == index:
                upstream = activation_backward(
                    kind,
                    value,
                    output,
                    upstream,
                    approximate,
                )
            else:
                value = output
    return upstream


@torch.library.custom_op(
    _FORWARD_NAME,
    mutates_args=(),
    device_types=("cpu", "cuda"),
    schema=_FORWARD_SCHEMA,
)
def _activation_chain_op(
    x: torch.Tensor,
    operations: list[str],
    replay: str = "replay_streaming",
) -> torch.Tensor:
    _validate_forward(x, operations, replay)
    value = x
    for kind, approximate in _validated_program(operations):
        value = activation_forward(kind, value, approximate)
    return value


@_activation_chain_op.register_fake
def _activation_chain_fake(
    x: torch.Tensor,
    operations: list[str],
    replay: str = "replay_streaming",
) -> torch.Tensor:
    _validate_forward(x, operations, replay)
    return torch.empty_like(x)


@torch.library.custom_op(
    _BACKWARD_NAME,
    mutates_args=(),
    device_types=("cpu", "cuda"),
    schema=_BACKWARD_SCHEMA,
)
def _activation_chain_backward_op(
    x: torch.Tensor,
    upstream: torch.Tensor,
    operations: list[str],
    replay: str = "replay_streaming",
) -> torch.Tensor:
    _validate_backward(x, upstream, operations, replay)
    return _replay_backward(x, upstream, operations, replay)


@_activation_chain_backward_op.register_fake
def _activation_chain_backward_fake(
    x: torch.Tensor,
    upstream: torch.Tensor,
    operations: list[str],
    replay: str = "replay_streaming",
) -> torch.Tensor:
    _validate_backward(x, upstream, operations, replay)
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output) -> None:
    x, operations, replay = inputs
    # save_for_backward retains PyTorch's input version counter. In-place input
    # mutation between forward and backward therefore raises instead of silently
    # replaying a different primal fiber.
    ctx.save_for_backward(x)
    ctx.operations = tuple(operations)
    ctx.replay = replay


def _backward(ctx, upstream: torch.Tensor):
    (x,) = ctx.saved_tensors
    gradient = None
    if ctx.needs_input_grad[0]:
        gradient = torch.ops.ojasx_verified.activation_chain_backward(
            x,
            upstream,
            list(ctx.operations),
            ctx.replay,
        )
    return gradient, None, None


_activation_chain_op.register_autograd(_backward, setup_context=_setup_context)


def activation_chain(
    x: torch.Tensor,
    operations: Sequence[str],
    replay: str = "replay_streaming",
) -> torch.Tensor:
    """Apply a registered M2.5 activation program in eager or compiled code.

    ``operations`` entries are activation names, with ``gelu:tanh`` selecting
    the tanh approximation. The operator is deterministic and non-mutating. Its
    custom autograd formula saves only ``x`` and replays intermediate values.
    Higher-order *autograd through the custom backward operator* is outside the
    M3 contract; the separate jet core remains the higher-order engine.
    """

    program = _validated_program(operations)
    _validate_replay(replay)
    encoded = [
        kind if kind == "gelu" and approximate == "none" else
        f"{kind}:{approximate}" if kind == "gelu" else kind
        for kind, approximate in program
    ]
    return torch.ops.ojasx_verified.activation_chain(x, encoded, replay)


def operator_schemas() -> Mapping[str, str]:
    """Return stable public schemas for integration and deployment audits."""

    return {
        _FORWARD_NAME: _FORWARD_SCHEMA,
        _BACKWARD_NAME: _BACKWARD_SCHEMA,
    }


__all__ = ["activation_chain", "operator_schemas"]
