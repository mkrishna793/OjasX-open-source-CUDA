"""Turnkey API and integration tests for OjasX3."""

import pytest
import torch
import torch.nn as nn
import ojasx3


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        h = torch.tanh(torch.sigmoid(self.fc1(x)))
        return self.fc2(h)


def test_ojasx3_version():
    assert ojasx3.__version__ == "0.16.0"


def test_ojasx3_compile_eager_and_backward():
    model = SimpleMLP()
    compiled = ojasx3.compile(model)
    x = torch.randn(2, 8, requires_grad=True)
    out = compiled(x)
    loss = out.sum()
    loss.backward()
    assert out.shape == (2, 4)
    assert x.grad is not None
    assert x.grad.shape == (2, 8)


def test_ojasx3_train_from_scratch_one_line():
    model = SimpleMLP()
    batch = [(torch.randn(4, 8), torch.randint(0, 4, (4,)))]
    res = ojasx3.train(model, batch, epochs=1, lr=1e-3, device="cpu", compile_backend=None)
    assert res["model"] is model
    assert "history" in res
    assert len(res["history"]) == 1


def test_ojasx3_optimize_and_prepare():
    model = SimpleMLP()
    opt_model, optimizer = ojasx3.prepare(model, quantize=None, lr=1e-4)
    assert isinstance(optimizer, ojasx3.HolonomicGradFreeOptimizer)
    x = torch.randn(2, 8)
    y = opt_model(x)
    assert y.shape == (2, 4)


def test_ojasx3_e8_lattice_linear_forward_backward():
    layer = ojasx3.E8LatticeLinear(16, 16, bitrate=1.5, group_size=16)
    x = torch.randn(2, 16, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    assert y.shape == (2, 16)
    assert x.grad is not None


def test_ojasx3_autotuner_hardware_selection():
    from ojasx3.hardware import TileLangHardwareAutoTuner
    cfg_cuda = TileLangHardwareAutoTuner.select_tile_config(128, 128, 128, device="cuda")
    assert cfg_cuda.target_arch == "cuda"
    cfg_metal = TileLangHardwareAutoTuner.select_tile_config(32, 32, 32, device="mps")
    assert cfg_metal.target_arch == "metal"
