"""Residual-carrying reverse maps. Linear daggers are a special case."""
from dataclasses import dataclass
from typing import Any, Callable, Tuple
import torch


@dataclass(frozen=True)
class Lens:
    forward: Callable[[Any], Tuple[Any, Any]]
    pullback: Callable[[Any, Any], Any]
    name: str = "lens"

    def then(self, other: "Lens") -> "Lens":
        def forward(x):
            s, y = self.forward(x)
            t, z = other.forward(y)
            return (s, t), z

        def pullback(tape, dz):
            s, t = tape
            return self.pullback(s, other.pullback(t, dz))

        return Lens(forward, pullback, f"{other.name} ∘ {self.name}")

    def value_and_vjp(self, x, dy):
        tape, y = self.forward(x)
        return y, self.pullback(tape, dy)


def linear_lens(weight: torch.Tensor) -> Lens:
    """Fixed W: f(X)=X Wᵀ, dagger(dY)=dY W. No activation inversion."""
    return Lens(lambda x: (None, x @ weight.T), lambda _tape, dy: dy @ weight, "linear")


def parameterized_linear_lens() -> Lens:
    """Input is (X,W); output is XWᵀ. Pullback returns both parameter gradients."""
    return Lens(lambda pair: (pair, pair[0] @ pair[1].T),
                lambda pair, dy: (dy @ pair[1], dy.T @ pair[0]), "parameterized-linear")


def activation_lens(kind: str, approximate: str = "none") -> Lens:
    if kind == "relu":
        from .residuals import pack_mask, unpack_mask

        def forward(x):
            return (pack_mask(~(x <= 0)), x.shape), torch.relu(x)

        def backward(tape, dy):
            packed, shape = tape
            return torch.where(unpack_mask(packed, shape), dy, torch.zeros_like(dy))

        return Lens(forward, backward, "relu")
    if kind == "gelu":
        return Lens(lambda x: (x, torch.nn.functional.gelu(x, approximate=approximate)),
                    lambda x, dy: torch.ops.aten.gelu_backward(dy, x, approximate=approximate), "gelu")
    if kind == "silu":
        return Lens(lambda x: (x, torch.nn.functional.silu(x)),
                    lambda x, dy: torch.ops.aten.silu_backward(dy, x), "silu")
    raise ValueError(f"Unsupported activation: {kind}")
