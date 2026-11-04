"""Fourth-order sparse Taylor-jet differential tests."""

import math

import pytest
import torch

from ojasx2.core.jets import Jet, index_from_powers, seed_jet


def _elementwise_derivatives(function, x, direction, order=4):
    current = function(x)
    result = []
    for derivative_order in range(1, order + 1):
        current = torch.autograd.grad(
            current,
            x,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
        )[0] * direction
        result.append(current)
    return result


@pytest.mark.parametrize(
    "jet_method,torch_function,positive",
    [
        ("exp", torch.exp, False),
        ("log", torch.log, True),
        ("sqrt", torch.sqrt, True),
        ("rsqrt", torch.rsqrt, True),
        ("sin", torch.sin, False),
        ("cos", torch.cos, False),
        ("sigmoid", torch.sigmoid, False),
        ("tanh", torch.tanh, False),
        ("silu", torch.nn.functional.silu, False),
        ("gelu", torch.nn.functional.gelu, False),
    ],
)
def test_unary_jets_match_four_nested_pytorch_derivatives(jet_method, torch_function, positive):
    low = 0.4 if positive else -1.3
    value = torch.linspace(low, 1.2, 7, dtype=torch.float64)
    direction = torch.linspace(0.3, 1.1, 7, dtype=torch.float64)
    jet = seed_jet(value, variables=1, order=4, variable=0, direction=direction)
    actual = getattr(jet, jet_method)()
    expected = _elementwise_derivatives(torch_function, value.requires_grad_(True), direction)
    torch.testing.assert_close(actual.value, torch_function(value), rtol=1e-12, atol=1e-12)
    for order, reference in enumerate(expected, 1):
        torch.testing.assert_close(actual.derivative((order,)), reference, rtol=2e-8, atol=2e-8)


def test_multivariate_division_total_directional_derivatives():
    x = torch.linspace(0.7, 1.4, 5, dtype=torch.float64)
    y = torch.linspace(1.5, 2.4, 5, dtype=torch.float64)
    u = torch.linspace(0.2, 0.8, 5, dtype=torch.float64)
    v = torch.linspace(-0.4, 0.3, 5, dtype=torch.float64)
    jets = (
        seed_jet(x, variables=2, order=4, variable=0, direction=u),
        seed_jet(y, variables=2, order=4, variable=1, direction=v),
    )
    actual = jets[0] / jets[1]

    xr, yr = x.requires_grad_(True), y.requires_grad_(True)
    current = xr / yr
    for order in range(1, 5):
        current = (
            torch.autograd.grad(current, xr, torch.ones_like(current), create_graph=True, retain_graph=True)[0] * u
            + torch.autograd.grad(current, yr, torch.ones_like(current), create_graph=True, retain_graph=True)[0] * v
        )
        torch.testing.assert_close(actual.directional_derivative(order), current, rtol=2e-8, atol=2e-8)
    torch.testing.assert_close(actual.derivative((1, 1)), -u * v / y.square() - 0 * x, rtol=1e-10, atol=1e-10)


def test_normalized_multiindex_and_constant_arithmetic():
    assert index_from_powers((0, 2, 1), 3) == ((1, 2), (2, 1))
    value = torch.tensor(2.0, dtype=torch.float64)
    jet = seed_jet(value, variables=1, order=4, variable=0)
    result = (jet + 3.0) * 2.0
    assert result.value == 10.0
    assert result.derivative((1,)) == 2.0
    for order in range(2, 5):
        assert result.derivative((order,)) == 0.0
    with pytest.raises(ValueError, match="exceeds"):
        result.derivative((5,))


def test_jet_rejects_invalid_coefficients_and_direction():
    value = torch.tensor(1.0)
    with pytest.raises(ValueError, match="direction"):
        seed_jet(value, variables=1, order=4, variable=0, direction=torch.ones(2))
    with pytest.raises(ValueError, match="zero-order"):
        Jet({}, 1, 4)


@pytest.mark.parametrize("index", [((-1, 1),), ((0, 1), (0, 1)), ((1, 1), (0, 1))])
def test_jet_rejects_noncanonical_multiindices(index):
    with pytest.raises(ValueError, match="canonical"):
        Jet({(): torch.tensor(1.0), index: torch.tensor(2.0)}, 2, 4)


def test_negative_integer_powers_allow_negative_nonzero_inputs():
    value = torch.tensor([-2.0, -1.5, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    jet = seed_jet(value, variables=1, order=4, variable=0, direction=direction)
    actual = jet.pow_scalar(-1)
    expected = _elementwise_derivatives(torch.reciprocal, value.requires_grad_(True), direction)
    for order, reference in enumerate(expected, 1):
        torch.testing.assert_close(actual.derivative((order,)), reference)
    with pytest.raises(ValueError, match="nonzero"):
        seed_jet(
            torch.tensor([0.0], dtype=torch.float64),
            variables=1,
            order=4,
            variable=0,
        ).pow_scalar(-1)
    with pytest.raises(ValueError, match="strictly positive"):
        jet.pow_scalar(-0.5)
