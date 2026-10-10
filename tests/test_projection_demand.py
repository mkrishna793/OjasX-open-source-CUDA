import pytest
import torch
from ojasx2 import shared_projections
from ojasx2.residuals import SavedTensorAudit


def test_unused_outputs_preserve_none_and_adamw_updates():
    x=torch.randn(4,8)
    ws=tuple(torch.nn.Parameter(torch.randn(6,8)) for _ in range(3))
    zs=tuple(torch.nn.Parameter(w.detach().clone()) for w in ws)
    before=tuple(w.detach().clone() for w in ws)
    opt1=torch.optim.AdamW(ws,lr=1e-3,weight_decay=0.1)
    opt2=torch.optim.AdamW(zs,lr=1e-3,weight_decay=0.1)
    shared_projections(x,ws,'strict')[0].square().mean().backward()
    shared_projections(x,zs,'algebraic')[0].square().mean().backward()
    for a,b in zip(ws,zs):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
    opt1.step();opt2.step()
    for i in (1,2):
        assert torch.equal(ws[i],before[i]) and torch.equal(zs[i],before[i])


def test_projection_tape_has_no_concatenated_weights():
    x=torch.randn(8,13,requires_grad=True)
    weights=tuple(torch.randn(n,13,requires_grad=True) for n in (7,9,5))
    with SavedTensorAudit([x,*weights]) as audit:
        ys=shared_projections(x,weights,'algebraic')
    assert audit.summary()['fresh_saved_storage_bytes']==0
    assert audit.summary()['unique_saved_storage_bytes']==sum(t.numel()*t.element_size() for t in (x,*weights))
    assert all(y.is_contiguous() for y in ys)
    sum(y.sum() for y in ys).backward()


@pytest.mark.parametrize('shape',[(13,),(2,3,13),(0,13),(2,0)])
def test_batched_empty_projection_gradients(shape):
    x=torch.randn(shape,dtype=torch.double,requires_grad=True)
    weights=tuple(torch.randn(n,shape[-1],dtype=torch.double,requires_grad=True) for n in (7,9,5))
    ys=shared_projections(x,weights,'algebraic')
    ref=shared_projections(x,weights,'strict')
    dys=tuple(torch.randn_like(y) for y in ys)
    actual=torch.autograd.grad(ys,(x,*weights),dys)
    expected=torch.autograd.grad(ref,(x,*weights),dys)
    for a,b in zip((*ys,*actual),(*ref,*expected)):
        torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)


def test_projection_gradcheck_gradgradcheck():
    x=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    weights=tuple(torch.randn(n,3,dtype=torch.double,requires_grad=True) for n in (2,4))
    fn=lambda x,*ws:shared_projections(x,ws,'algebraic')
    assert torch.autograd.gradcheck(fn,(x,*weights))
    assert torch.autograd.gradgradcheck(fn,(x,*weights))
