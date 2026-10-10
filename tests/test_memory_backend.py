import copy
import pytest
import torch
import torch.nn.functional as F
from ojasx2 import make_backend
from ojasx2.residuals import SavedTensorAudit


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=torch.nn.Linear(128,512,bias=False)
        self.up=torch.nn.Linear(128,512,bias=False)
        self.down=torch.nn.Linear(512,128,bias=False)
    def forward(self,x):
        return x+self.down(F.silu(self.gate(x))*self.up(x))


def bits(x,y):
    assert torch.equal(x.detach().contiguous().view(torch.uint8),y.detach().contiguous().view(torch.uint8))


def test_ordinary_ffn_memory_backend_bits_and_saved_storage():
    model=FFN()
    other=copy.deepcopy(model)
    backend=make_backend(memory_policy='pointwise')
    compiled=torch.compile(other,backend=backend,fullgraph=True)
    x=torch.randn(64,128,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    with SavedTensorAudit([x,*model.parameters()]) as ref_audit:y=model(x)
    with SavedTensorAudit([z,*other.parameters()]) as new_audit:v=compiled(z)
    bits(y,v)
    y.sum().backward();v.sum().backward()
    bits(x.grad,z.grad)
    for a,b in zip(model.parameters(),other.parameters()):bits(a.grad,b.grad)
    assert ref_audit.summary()['fresh_saved_storage_bytes']-new_audit.summary()['fresh_saved_storage_bytes']==64*512*4
    assert backend.evidence[-1]['pointwise_fusions']==1
    assert backend.evidence[-1]['rewritten_nodes']==0


def test_broadcast_multiply_falls_back():
    backend=make_backend(memory_policy='pointwise')
    fn=torch.compile(lambda x,u:F.silu(x)*u,backend=backend,fullgraph=True)
    x=torch.randn(3,7,requires_grad=True)
    u=torch.randn(7,requires_grad=True)
    y=fn(x,u);ref=F.silu(x)*u
    bits(y,ref)
    for a,b in zip(torch.autograd.grad(y.sum(),(x,u)),torch.autograd.grad(ref.sum(),(x,u))):bits(a,b)


def test_shared_activation_not_fused():
    def fn(x,u):
        s=F.silu(x)
        return s*u+s
    backend=make_backend(memory_policy='pointwise')
    compiled=torch.compile(fn,backend=backend,fullgraph=True)
    x,u=torch.randn(3,7),torch.randn(3,7)
    bits(compiled(x,u),fn(x,u))
    assert backend.evidence[-1]['pointwise_fusions']==0


def test_deep_residual_stack_all_gradients_match():
    model=torch.nn.Sequential(*(FFN() for _ in range(4)))
    other=copy.deepcopy(model)
    backend=make_backend(memory_policy='pointwise')
    compiled=torch.compile(other,backend=backend,fullgraph=True)
    x=torch.randn(16,128,requires_grad=True)
    z=x.detach().clone().requires_grad_()
    y=model(x);v=compiled(z)
    y.square().mean().backward();v.square().mean().backward()
    bits(y,v);bits(x.grad,z.grad)
    for a,b in zip(model.parameters(),other.parameters()):bits(a.grad,b.grad)
    assert backend.evidence[-1]['pointwise_fusions']==4
