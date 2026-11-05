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


def test_ojasx3_train_default_auto_device():
    """Verify train() does NOT crash on its default device='auto'."""
    model = SimpleMLP()
    batch = [(torch.randn(4, 8), torch.randint(0, 4, (4,)))]
    res = ojasx3.train(model, batch, epochs=1, lr=1e-3, device="auto", compile_backend=None)
    assert res["model"] is model
    assert len(res["history"]) == 1


def test_ojasx3_finetune_reconciled_signatures():
    """Verify finetune() accepts model, bitrate, lora_rank, device='auto' without TypeError."""
    model = SimpleMLP()
    batch = [(torch.randn(4, 8), torch.randint(0, 4, (4,)))]
    res = ojasx3.finetune(
        model,
        batch,
        bitrate=1.5,
        lora_rank=4,
        lora_alpha=8.0,
        lr=1e-3,
        epochs=1,
        device="auto",
        compile_backend=None,
    )
    assert "history" in res
    assert len(res["history"]) == 1


def test_ojasx3_finetune_load_only_mode():
    """Verify finetune(model, dataset=None) returns (model, optimizer) with active grad on LoRA."""
    model = SimpleMLP()
    model_ret, opt = ojasx3.finetune(model, dataset=None, lora_rank=4, lora_alpha=8.0)
    assert model_ret is model
    assert isinstance(opt, ojasx3.HolonomicGradFreeOptimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "LoRA adapters must be attached and require grad"


def test_ojasx3_attach_lora_adapters_actual_wrapping():
    """Verify attach_lora_adapters freezes base params and attaches trainable LoRA weights."""
    from ojasx3.trainer.peft import attach_lora_adapters, LoRALinear

    model = SimpleMLP()
    orig_params = list(model.parameters())
    assert all(p.requires_grad for p in orig_params)

    attach_lora_adapters(model, rank=4, lora_alpha=8.0)
    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, LoRALinear)

    # Base parameters must be frozen
    assert not model.fc1.base_layer.weight.requires_grad
    assert not model.fc2.base_layer.weight.requires_grad

    # LoRA parameters must be trainable
    assert model.fc1.lora_A.requires_grad
    assert model.fc1.lora_B.requires_grad
    assert model.fc2.lora_A.requires_grad
    assert model.fc2.lora_B.requires_grad

    # Forward & backward pass
    x = torch.randn(2, 8)
    out = model(x)
    out.sum().backward()
    assert model.fc1.lora_A.grad is not None
    assert model.fc1.base_layer.weight.grad is None


def test_ojasx3_backend_reports_unified():
    """Verify ojasx3.core.backend and ojasx2.core.backend are identical and audit trail works."""
    from ojasx3.core.backend import backend as b3
    from ojasx2.core.backend import backend as b2
    assert b3 is b2, "ojasx3 and ojasx2 backends must be the exact same instance"

    model = SimpleMLP()
    compiled = ojasx3.compile(model)
    x = torch.randn(2, 8)
    out = compiled(x)
    assert out.shape == (2, 4)
    # The unified backend recorded the compilation report
    assert len(b3.reports) > 0, "Backend reports audit trail must be reachable"

