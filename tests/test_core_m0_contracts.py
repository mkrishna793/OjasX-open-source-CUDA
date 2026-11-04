"""M0 declarative registry and proof/residual contract tests."""

import pytest
import torch

from ojasx2.core import (
    AccuracyMode,
    CORE_REGISTRY,
    OperatorRegistry,
    ProofRef,
    ProofSystem,
    ResidualPolicy,
    tensor_nbytes,
)
from ojasx2.core.operators import LinearOperator


def test_builtin_registry_is_unique_inspectable_and_complete():
    assert CORE_REGISTRY.names() == (
        "activation_chain", "add", "cos", "div", "exp", "gelu", "layer_norm",
        "linear", "log", "log_softmax", "matmul", "mean", "mul", "neg", "relu",
        "reshape", "rms_norm", "sigmoid", "silu", "sin", "softmax", "sqrt",
        "square", "sum", "tanh", "transpose",
    )
    assert len(CORE_REGISTRY) == 26
    snapshot = CORE_REGISTRY.snapshot()
    assert set(snapshot) == set(CORE_REGISTRY.names())
    for contract in snapshot.values():
        assert contract.residual_policies
        assert contract.max_derivative_order == 4
        assert contract.derivative_domain
        assert "operator's smooth domain" not in contract.derivative_domain
        for policy in contract.residual_policies:
            assert policy.satisfies
            for proof in policy.proofs:
                assert proof.system in set(ProofSystem)
                assert proof.identifier
    with pytest.raises(TypeError, match="mappingproxy"):
        snapshot["new"] = snapshot["add"]


def test_smooth_domain_metadata_is_operator_specific():
    contracts = CORE_REGISTRY.snapshot()
    assert "strictly positive" in contracts["log"].derivative_domain
    assert "denominator" in contracts["div"].derivative_domain
    assert "away from zero" in contracts["relu"].derivative_domain
    assert "epsilon > 0" in contracts["layer_norm"].derivative_domain
    assert "nonempty normalized dimension" in contracts["softmax"].derivative_domain
    assert contracts["matmul"].derivative_domain == "all finite real inputs"
    assert contracts["gelu"].attribute_names == ("approximate",)
    assert contracts["softmax"].attribute_names == ("dim",)
    assert contracts["layer_norm"].attribute_names == ("eps",)
    assert contracts["reshape"].attribute_names == ("shape",)
    assert contracts["transpose"].attribute_names == ("dim0", "dim1")
    assert contracts["exp"].attribute_names == ()
    assert contracts["activation_chain"].attribute_names == ("operations",)
    assert tuple(policy.name for policy in contracts["activation_chain"].residual_policies) == (
        "replay_streaming",
        "replay_cached",
    )


def test_registry_rejects_duplicate_and_unknown_operators():
    registry = OperatorRegistry()
    registry.register(LinearOperator())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(LinearOperator())
    with pytest.raises(KeyError, match="Unknown certified-core operator"):
        registry.get("not_registered")


def test_strict_residual_cannot_be_lossy():
    with pytest.raises(ValueError, match="must be lossless"):
        ResidualPolicy(
            name="bad",
            accuracy=AccuracyMode.STRICT,
            lossless=False,
            satisfies=("input",),
        )


def test_proof_reference_rejects_empty_identity_or_premise():
    with pytest.raises(ValueError):
        ProofRef(ProofSystem.LEAN, "")
    with pytest.raises(ValueError):
        ProofRef(ProofSystem.Z3, "guard", premises=("valid", ""))


@pytest.mark.parametrize(
    "shape,dtype,expected",
    [
        ((), torch.float32, 4),
        ((0,), torch.float32, 0),
        ((2, 0, 5), torch.float64, 0),
        ((2, 3), torch.float64, 48),
        ((17,), torch.uint8, 17),
    ],
)
def test_tensor_nbytes(shape, dtype, expected):
    assert tensor_nbytes(shape, dtype) == expected


def test_tensor_nbytes_rejects_negative_dimensions():
    with pytest.raises(ValueError, match="Negative"):
        tensor_nbytes((2, -1), torch.float32)
