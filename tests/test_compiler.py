import pytest
import torch
from ojasx2 import make_backend
from ojasx2.residuals import SavedTensorAudit


@pytest.mark.parametrize('shape',[(32,128,64),(31,127,63),(2,128,1024)])
def test_backend_guards_and_gradients(shape):
    m,k,n=shape
    backend=make_backend('algebraic',minimum_inner=16)
    compiled=torch.compile(lambda a,b:a@b,backend=backend,fullgraph=True)
    a=torch.randn(m,k,dtype=torch.double,requires_grad=True)
    b=torch.randn(k,n,dtype=torch.double,requires_grad=True)
    y=compiled(a,b)
    ref=a@b
    dy=torch.randn_like(ref)
    actual=torch.autograd.grad(y,(a,b),dy)
    expected=torch.autograd.grad(ref,(a,b),dy)
    for x,z in zip((y,*actual),(ref,*expected)):
        torch.testing.assert_close(x,z,rtol=2e-11,atol=2e-11)
    assert backend.evidence[-1]['rewritten_nodes']>=1


def test_backend_preserves_normal_training_and_parameter_identity():
    model=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,64))
    backend=make_backend('algebraic',minimum_inner=16)
    compiled=torch.compile(model,backend=backend,fullgraph=True)
    assert list(compiled.parameters())[0] is list(model.parameters())[0]
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    before=model[0].weight.detach().clone()
    for _ in range(3):
        x=torch.randn(32,128)
        optimizer.zero_grad()
        loss=compiled(x).square().mean()
        loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before,model[0].weight)
    assert backend.evidence[-1]['rewritten_nodes']==2


def test_strict_backend_preserves_bits():
    backend=make_backend()
    fn=torch.compile(lambda x,w:torch.nn.functional.linear(x,w),backend=backend,fullgraph=True)
    x=torch.randn(7,13,requires_grad=True)
    w=torch.randn(11,13,requires_grad=True)
    y=fn(x,w);ref=torch.nn.functional.linear(x,w)
    assert torch.equal(y,ref)
    assert backend.evidence[-1]['rewritten_nodes']==0


def test_gradient_demand_saves_only_required_operand():
    from ojasx2 import operand_tape_gemm
    a=torch.randn(16,32,requires_grad=True)
    b=torch.randn(32,16)
    with SavedTensorAudit([a,b]) as audit:
        y=operand_tape_gemm(a,b)
    assert audit.summary()['saved_tensors']==1
    assert audit.summary()['unique_saved_storage_bytes']==b.numel()*b.element_size()
    y.sum().backward()
    assert a.grad is not None
