"""Observable contracts for the one-line public API."""

import pytest
import torch.nn as nn

import ojasx2
from ojasx2.dagger_attention import DaggerAttention


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)
        self.num_heads = 2
        self.head_dim = 4

    def forward(self, x):
        return self.out_proj(self.q_proj(x))


def test_attention_native_preserves_and_fast_upgrades_with_weights():
    native_model = nn.Sequential(DummyAttention())
    original = native_model[0]
    result = ojasx2.optimize(native_model, attention="native")
    assert result[0] is original

    fast_model = nn.Sequential(DummyAttention())
    original_q = fast_model[0].q_proj
    result = ojasx2.optimize(fast_model, attention="fast")
    assert isinstance(result[0], DaggerAttention)
    assert result[0].q_proj is original_q


def test_prepare_moment_quantization_is_independent_of_weight_quantization():
    _, default_optimizer = ojasx2.prepare(nn.Linear(4, 4), quantize=None)
    assert default_optimizer.quantize_moments_e8 is True
    _, dense_optimizer = ojasx2.prepare(
        nn.Linear(4, 4), quantize="2.0b", quantize_moments_e8=False
    )
    assert dense_optimizer.quantize_moments_e8 is False


@pytest.mark.parametrize("quantize", ["not-a-bitrate", 3.0, 0.0])
def test_simple_api_rejects_unknown_quantization(quantize):
    with pytest.raises(ValueError, match="1.0, 1.5, or 2.0"):
        ojasx2.optimize(nn.Linear(4, 4), quantize=quantize)


def test_simple_api_rejects_unknown_attention_mode():
    with pytest.raises(ValueError, match="fast.*native"):
        ojasx2.optimize(nn.Linear(4, 4), attention="sometimes")


def test_simple_api_wires_compile_backend():
    import torch

    opt_model = ojasx2.optimize(nn.Linear(4, 4), compile_backend="ojasx_verified")
    assert hasattr(opt_model, "_orig_mod") or callable(opt_model)
    x = torch.randn(2, 4)
    out = opt_model(x)
    assert out.shape == (2, 4)

    prepared_model, opt = ojasx2.prepare(nn.Linear(4, 4), compile_backend="ojasx_verified")
    assert hasattr(prepared_model, "_orig_mod") or callable(prepared_model)
    out = prepared_model(x)
    loss = out.sum()
    loss.backward()
    opt.step()


def test_simple_api_defaults_to_uncompiled():
    opt_model = ojasx2.optimize(nn.Linear(4, 4))
    assert not hasattr(opt_model, "_orig_mod")

    prepared_model, _ = ojasx2.prepare(nn.Linear(4, 4))
    assert not hasattr(prepared_model, "_orig_mod")
