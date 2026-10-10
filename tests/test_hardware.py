"""Hardware gates stay visible when no GPU is available."""
import pytest
import torch
from ojasx2 import packed_relu


@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_device_packed_relu_conformance(device):
    available = torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f'{device} hardware unavailable; no hardware verification claimed')
    x = torch.randn(127, 513, device=device, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    y, ref = packed_relu(x), torch.relu(z)
    dx = torch.autograd.grad(y, x, dy)[0]
    dz = torch.autograd.grad(ref, z, dy)[0]
    for a, b in ((y, ref), (dx, dz)):
        assert torch.equal(a.detach().contiguous().view(torch.uint8).cpu(), b.detach().contiguous().view(torch.uint8).cpu())
