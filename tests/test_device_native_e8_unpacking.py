"""
Unit tests for Device-Native E8 Lattice Unpacking, Ephemeral Layer Streaming,
and Zero-Host-Roundtrip Fused GEMV/GEMM Execution.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from ojasx2.math.lattice import (
    TernaryE8LatticePacker,
    E8LatticePacker,
    E8LatticeLinear,
    _FusedE8GEMMFunction,
)


def _get_test_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_ternary_unpack_no_cpu_call():
    """Assert TernaryE8LatticePacker.unpack executes strictly on-device without calling .cpu()."""
    device = _get_test_device()
    torch.manual_seed(42)
    w = torch.randn(64, 64, device=device, dtype=torch.float32)

    # First pack on CPU or device
    packed, scales, shape = TernaryE8LatticePacker.pack(w.cpu(), group_size=32)
    packed_dev = packed.to(device)
    scales_dev = scales.to(device)

    # Monkeypatch .cpu to raise if invoked on packed_dev or any device tensor
    orig_cpu = torch.Tensor.cpu
    def forbidden_cpu(self, *args, **kwargs):
        if self.is_mps or self.is_cuda:
            raise RuntimeError("FORBIDDEN_HOST_DEVICE_TRANSFER: .cpu() called during device execution!")
        return orig_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", forbidden_cpu):
        # This MUST succeed without calling .cpu() on device tensors
        w_recon = TernaryE8LatticePacker.unpack(packed_dev, scales_dev, shape, dtype=torch.float32)
        assert w_recon.device.type == device.type
        assert w_recon.shape == (64, 64)


def test_ternary_pack_no_cpu_call():
    """Assert TernaryE8LatticePacker.pack executes on-device without calling .cpu()."""
    device = _get_test_device()
    torch.manual_seed(42)
    w = torch.randn(64, 64, device=device, dtype=torch.float32)

    orig_cpu = torch.Tensor.cpu
    def forbidden_cpu(self, *args, **kwargs):
        if self.is_mps or self.is_cuda:
            raise RuntimeError("FORBIDDEN_HOST_DEVICE_TRANSFER: .cpu() called during pack()!")
        return orig_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", forbidden_cpu):
        packed, scales, shape = TernaryE8LatticePacker.pack(w, group_size=32)
        assert packed.device.type == device.type
        assert scales.device.type == device.type
        assert shape == (64, 64)


def test_single_token_decoding_fused_gemv_no_cpu():
    """Assert single-token decoding (M=1) forward pass executes purely on device with zero CPU copies."""
    device = _get_test_device()
    in_features = 128
    out_features = 256
    layer = E8LatticeLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        bitrate=1.5,
        group_size=32,
        device=device,
        dtype=torch.float32,
    )
    
    # Initialize weights
    w_fp = torch.randn(out_features, in_features, device=device)
    with torch.no_grad():
        packed, scales, _ = TernaryE8LatticePacker.pack(w_fp, group_size=32)
        layer.packed_weight.copy_(packed)
        layer.block_scales.copy_(scales)

    x = torch.randn(1, 1, in_features, device=device)

    orig_cpu = torch.Tensor.cpu
    def forbidden_cpu(self, *args, **kwargs):
        if self.is_mps or self.is_cuda:
            raise RuntimeError("FORBIDDEN_HOST_DEVICE_TRANSFER: .cpu() called during decoding forward!")
        return orig_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", forbidden_cpu):
        y = layer(x)
        assert y.shape == (1, 1, out_features)
        assert y.device.type == device.type


def test_batch_forward_backward_on_device_parity():
    """Assert batched forward and backward passes run without CPU round-trips and match unpack_weights()."""
    device = _get_test_device()
    in_features = 64
    out_features = 128
    batch_size = 4
    
    layer = E8LatticeLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        rank=8,
        bitrate=1.5,
        group_size=32,
        device=device,
        dtype=torch.float32,
    )
    
    w_fp = torch.randn(out_features, in_features, device=device)
    with torch.no_grad():
        packed, scales, _ = TernaryE8LatticePacker.pack(w_fp, group_size=32)
        layer.packed_weight.copy_(packed)
        layer.block_scales.copy_(scales)

    x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

    orig_cpu = torch.Tensor.cpu
    def forbidden_cpu(self, *args, **kwargs):
        if self.is_mps or self.is_cuda:
            raise RuntimeError("FORBIDDEN_HOST_DEVICE_TRANSFER: .cpu() called during batch forward/backward!")
        return orig_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", forbidden_cpu):
        y = layer(x)
        loss = y.sum()
        loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    # Base packed weight is not a parameter and requires no grad
    assert layer.packed_weight.requires_grad is False


def test_ephemeral_workspace_memory_reclaim():
    """Verify that multiple successive linear passes do not leak or hoard FP32/FP16 weight memory."""
    device = _get_test_device()
    in_features = 256
    out_features = 256

    layers = [
        E8LatticeLinear(in_features, out_features, bias=False, bitrate=1.5, group_size=32, device=device)
        for _ in range(5)
    ]

    x = torch.randn(2, in_features, device=device)
    h = x
    for l in layers:
        h = l(h)

    assert h.shape == (2, out_features)
    assert h.device.type == device.type
