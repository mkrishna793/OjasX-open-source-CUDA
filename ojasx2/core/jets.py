"""Sparse multivariate truncated Taylor jets through fourth order.

Coefficients use the normalized convention ``D^α f / α!``. Products therefore
use ordinary multi-index convolution without additional multinomial factors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch

Index = Tuple[Tuple[int, int], ...]
ZERO: Index = ()
Scalar = Union[int, float, torch.Tensor]


def index_from_powers(powers: Sequence[int], variables: Optional[int] = None) -> Index:
    if variables is not None and len(powers) != variables:
        raise ValueError("Multi-index arity mismatch")
    result = tuple((index, int(power)) for index, power in enumerate(powers) if power)
    if any(power < 0 for power in powers):
        raise ValueError("Multi-index powers must be non-negative")
    return result


def index_degree(index: Index) -> int:
    return sum(power for _, power in index)


def index_add(left: Index, right: Index) -> Index:
    merged: Dict[int, int] = dict(left)
    for variable, power in right:
        merged[variable] = merged.get(variable, 0) + power
    return tuple(sorted(merged.items()))


def _factorial(index: Index) -> int:
    result = 1
    for _, power in index:
        result *= math.factorial(power)
    return result


@dataclass(eq=False)
class Jet:
    coefficients: Dict[Index, torch.Tensor]
    variables: int
    order: int

    def __post_init__(self) -> None:
        if not 1 <= self.variables or not 1 <= self.order <= 4:
            raise ValueError("Jets require at least one variable and order one through four")
        if ZERO not in self.coefficients:
            raise ValueError("A jet requires a primal zero-order coefficient")
        value = self.coefficients[ZERO]
        if not value.dtype.is_floating_point:
            raise TypeError("Taylor jets require floating coefficients")
        for index, coefficient in self.coefficients.items():
            if index_degree(index) > self.order:
                raise ValueError("Coefficient exceeds the truncated jet order")
            variables = [variable for variable, _ in index]
            if (
                tuple(sorted(index)) != index
                or len(set(variables)) != len(variables)
                or any(
                    variable < 0 or variable >= self.variables or power <= 0
                    for variable, power in index
                )
            ):
                raise ValueError("Coefficient multi-indices must be canonical")
            if coefficient.shape != value.shape or coefficient.dtype != value.dtype or coefficient.device != value.device:
                raise ValueError("All jet coefficients must share the primal signature")

    @property
    def value(self) -> torch.Tensor:
        return self.coefficients[ZERO]

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.value.dtype

    @property
    def device(self) -> torch.device:
        return self.value.device

    @property
    def term_count(self) -> int:
        return len(self.coefficients)

    def set_value(self, value: torch.Tensor) -> "Jet":
        if value.shape != self.shape or value.dtype != self.dtype or value.device != self.device:
            raise ValueError("Replacement primal signature mismatch")
        self.coefficients[ZERO] = value
        return self

    def coefficient(self, powers: Sequence[int]) -> torch.Tensor:
        index = index_from_powers(powers, self.variables)
        if index_degree(index) > self.order:
            raise ValueError("Requested coefficient exceeds the jet order")
        if index not in self.coefficients:
            return torch.zeros_like(self.value)
        return self.coefficients[index]

    def derivative(self, powers: Sequence[int]) -> torch.Tensor:
        index = index_from_powers(powers, self.variables)
        if index_degree(index) > self.order:
            raise ValueError("Requested derivative exceeds the jet order")
        if index not in self.coefficients:
            return torch.zeros_like(self.value)
        return self.coefficients[index] * _factorial(index)

    def directional_derivative(self, order: int) -> torch.Tensor:
        """Return the total order-directional derivative from seeded tangents."""

        if not 0 <= order <= self.order:
            raise ValueError("Requested directional derivative exceeds the jet order")
        terms = [
            math.factorial(order) * coefficient
            for index, coefficient in self.coefficients.items()
            if index_degree(index) == order
        ]
        if not terms:
            return torch.zeros_like(self.value)
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def clone(self) -> "Jet":
        return Jet({index: value.clone() for index, value in self.coefficients.items()}, self.variables, self.order)

    @classmethod
    def constant(
        cls,
        value: Scalar,
        *,
        variables: Optional[int] = None,
        order: Optional[int] = None,
        like: Optional["Jet"] = None,
    ) -> "Jet":
        if like is not None:
            variables, order = like.variables, like.order
            value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        elif not isinstance(value, torch.Tensor):
            raise TypeError("A standalone constant jet requires a tensor or a like jet")
        if variables is None or order is None:
            raise ValueError("Standalone constant jets require variables and order")
        return cls({ZERO: value}, variables, order)

    def _coerce(self, other: Scalar) -> "Jet":
        if isinstance(other, Jet):
            if other.variables != self.variables or other.order != self.order:
                raise ValueError("Jet variable/order mismatch")
            return other
        return Jet.constant(other, like=self)

    def _combine(self, other: Scalar, function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> "Jet":
        other = self._coerce(other)
        result: Dict[Index, torch.Tensor] = {}
        for left_index, left in self.coefficients.items():
            for right_index, right in other.coefficients.items():
                index = index_add(left_index, right_index)
                if index_degree(index) > self.order:
                    continue
                term = function(left, right)
                if index in result:
                    term = result[index] + term
                result[index] = term
        return Jet(result, self.variables, self.order)

    def __add__(self, other: Scalar) -> "Jet":
        other = self._coerce(other)
        result: Dict[Index, torch.Tensor] = {}
        shape = torch.broadcast_shapes(self.shape, other.shape)
        for index in self.coefficients.keys() | other.coefficients.keys():
            left, right = self.coefficients.get(index), other.coefficients.get(index)
            if left is None:
                result[index] = torch.broadcast_to(right, shape).clone()
            elif right is None:
                result[index] = torch.broadcast_to(left, shape).clone()
            else:
                result[index] = left + right
        return Jet(result, self.variables, self.order)

    def __radd__(self, other: Scalar) -> "Jet":
        return self + other

    def __neg__(self) -> "Jet":
        return Jet({index: -value for index, value in self.coefficients.items()}, self.variables, self.order)

    def __sub__(self, other: Scalar) -> "Jet":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Scalar) -> "Jet":
        return self._coerce(other) + (-self)

    def __mul__(self, other: Scalar) -> "Jet":
        return self._combine(other, torch.mul)

    def __rmul__(self, other: Scalar) -> "Jet":
        return self * other

    def __truediv__(self, other: Scalar) -> "Jet":
        other = self._coerce(other)
        result = self * other.reciprocal()
        if not self.value.is_meta:
            result.set_value(self.value / other.value)
        return result

    def __rtruediv__(self, other: Scalar) -> "Jet":
        return self._coerce(other) * self.reciprocal()

    def __matmul__(self, other: "Jet") -> "Jet":
        return self._combine(other, torch.matmul)

    def reciprocal(self) -> "Jet":
        return self.pow_scalar(-1)

    def pow_scalar(self, exponent: float) -> "Jet":
        if isinstance(exponent, int) and exponent >= 0:
            result = Jet.constant(torch.ones_like(self.value), like=self)
            for _ in range(exponent):
                result = result * self
            return result
        if exponent == 0:
            return Jet.constant(torch.ones_like(self.value), like=self)
        if not self.value.is_meta:
            if isinstance(exponent, int) and exponent < 0 and bool(torch.any(self.value == 0)):
                raise ValueError("Negative integer jet powers require nonzero inputs")
            if not isinstance(exponent, int) and bool(torch.any(self.value <= 0)):
                raise ValueError("Noninteger jet powers require a strictly positive primal domain")
        base = torch.pow(self.value, exponent)
        delta = self - Jet.constant(self.value, like=self)
        scaled = delta * (1.0 / self.value)
        coefficients = [base]
        for order in range(1, self.order + 1):
            binomial = math.prod((exponent - index) / (index + 1) for index in range(order))
            coefficients.append(base * binomial)
        return _compose(scaled, coefficients, base)

    def exp(self) -> "Jet":
        base = self.value.exp()
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [base] + [base / math.factorial(order) for order in range(1, self.order + 1)]
        return _compose(delta, coefficients, base)

    def log(self) -> "Jet":
        if not self.value.is_meta and bool(torch.any(self.value <= 0)):
            raise ValueError("Logarithm jets require strictly positive inputs")
        base = self.value.log()
        delta = self - Jet.constant(self.value, like=self)
        scaled = delta * (1.0 / self.value)
        coefficients = [base] + [
            torch.full_like(self.value, ((-1.0) ** (order + 1)) / order)
            for order in range(1, self.order + 1)
        ]
        return _compose(scaled, coefficients, base)

    def sqrt(self) -> "Jet":
        return self.pow_scalar(0.5)

    def rsqrt(self) -> "Jet":
        return self.pow_scalar(-0.5)

    def sin(self) -> "Jet":
        sine, cosine = self.value.sin(), self.value.cos()
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [sine, cosine, -sine / 2, -cosine / 6, sine / 24]
        return _compose(delta, coefficients[: self.order + 1], sine)

    def cos(self) -> "Jet":
        sine, cosine = self.value.sin(), self.value.cos()
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [cosine, -sine, -cosine / 2, sine / 6, cosine / 24]
        return _compose(delta, coefficients[: self.order + 1], cosine)

    def sigmoid(self) -> "Jet":
        s = self.value.sigmoid()
        q = s * (1.0 - s)
        raw = [
            s,
            q,
            q * (1.0 - 2.0 * s),
            q * (1.0 - 6.0 * q),
            q * (1.0 - 2.0 * s) * (1.0 - 12.0 * q),
        ]
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [raw[0]] + [raw[order] / math.factorial(order) for order in range(1, self.order + 1)]
        return _compose(delta, coefficients, s)

    def tanh(self) -> "Jet":
        s = self.value.tanh()
        q = 1.0 - s.square()
        raw = [
            s,
            q,
            -2.0 * s * q,
            q * (6.0 * s.square() - 2.0),
            8.0 * s * q * (2.0 - 3.0 * s.square()),
        ]
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [raw[0]] + [raw[order] / math.factorial(order) for order in range(1, self.order + 1)]
        return _compose(delta, coefficients, s)

    def silu(self) -> "Jet":
        x = self.value
        s = x.sigmoid()
        q = s * (1.0 - s)
        ds = [
            s,
            q,
            q * (1.0 - 2.0 * s),
            q * (1.0 - 6.0 * q),
            q * (1.0 - 2.0 * s) * (1.0 - 12.0 * q),
        ]
        base = x * s
        raw = [base] + [x * ds[order] + order * ds[order - 1] for order in range(1, 5)]
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [base] + [raw[order] / math.factorial(order) for order in range(1, self.order + 1)]
        return _compose(delta, coefficients, base)

    def gelu(self) -> "Jet":
        x = self.value
        base = torch.nn.functional.gelu(x)
        normal_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        density = torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        raw = [
            base,
            normal_cdf + x * density,
            (2.0 - x.square()) * density,
            x * (x.square() - 4.0) * density,
            (-x.pow(4) + 7.0 * x.square() - 4.0) * density,
        ]
        delta = self - Jet.constant(self.value, like=self)
        coefficients = [base] + [raw[order] / math.factorial(order) for order in range(1, self.order + 1)]
        return _compose(delta, coefficients, base)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Jet":
        return Jet(
            {index: value.sum(dim=dim, keepdim=keepdim) for index, value in self.coefficients.items()},
            self.variables,
            self.order,
        )

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Jet":
        return Jet(
            {index: value.mean(dim=dim, keepdim=keepdim) for index, value in self.coefficients.items()},
            self.variables,
            self.order,
        )

    def reshape(self, shape: Sequence[int]) -> "Jet":
        return Jet(
            {index: value.reshape(tuple(shape)) for index, value in self.coefficients.items()},
            self.variables,
            self.order,
        )

    def transpose(self, dim0: int, dim1: int) -> "Jet":
        return Jet(
            {index: value.transpose(dim0, dim1) for index, value in self.coefficients.items()},
            self.variables,
            self.order,
        )


def _compose(delta: Jet, coefficients: Sequence[torch.Tensor], primal: torch.Tensor) -> Jet:
    """Evaluate Σ c_k delta^k by Horner with normalized coefficients."""

    if len(coefficients) != delta.order + 1:
        raise ValueError("Composition coefficient count does not match jet order")
    result = Jet.constant(coefficients[-1], like=delta)
    for coefficient in reversed(coefficients[:-1]):
        result = result * delta + Jet.constant(coefficient, like=delta)
    return result.set_value(primal)


def seed_jet(
    value: torch.Tensor,
    *,
    variables: int,
    order: int,
    variable: Optional[int] = None,
    direction: Optional[torch.Tensor] = None,
) -> Jet:
    """Seed one constant or first-order variable in a shared jet space."""

    coefficients: Dict[Index, torch.Tensor] = {ZERO: value}
    if variable is not None:
        if not 0 <= variable < variables:
            raise ValueError("Invalid jet variable index")
        if direction is None:
            direction = torch.ones_like(value)
        if direction.shape != value.shape or direction.dtype != value.dtype or direction.device != value.device:
            raise ValueError("Jet direction must match the input signature")
        coefficients[((variable, 1),)] = direction
    return Jet(coefficients, variables, order)


__all__ = [
    "Index",
    "Jet",
    "ZERO",
    "index_add",
    "index_degree",
    "index_from_powers",
    "seed_jet",
]
