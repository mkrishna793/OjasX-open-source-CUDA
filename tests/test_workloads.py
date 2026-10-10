import copy
import pytest
import torch
from torch import nn
from ojasx2 import optimize_relu_residuals, shared_projections, rmsnorm, swiglu
from test_activations import assert_bits


def test_mlp_20_optimizer_steps_bit_exact():
    torch.manual_seed(71)
    eager=nn.Sequential(nn.Linear(17,31),nn.ReLU(),nn.Linear(31,23),nn.ReLU(),nn.Linear(23,7))
    rebuilt=optimize_relu_residuals(eager, policy='pack')
    opt1,opt2=torch.optim.Adam(eager.parameters(),lr=1e-3),torch.optim.Adam(rebuilt.parameters(),lr=1e-3)
    for _ in range(20):
        data=torch.randn(5,17)
        a,b=data.clone().requires_grad_(),data.clone().requires_grad_()
        y1,y2=eager(a),rebuilt(b)
        assert_bits(y1,y2)
        y1.square().mean().backward()
        y2.square().mean().backward()
        assert_bits(a.grad,b.grad)
        for p,q in zip(eager.parameters(),rebuilt.parameters()):
            assert_bits(p.grad,q.grad)
        opt1.step();opt2.step()
        opt1.zero_grad(set_to_none=True);opt2.zero_grad(set_to_none=True)
        for p,q in zip(eager.parameters(),rebuilt.parameters()):
            assert_bits(p,q)


def test_residual_branch_gradient_accumulation():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first=nn.Linear(13,13)
            self.relu=nn.ReLU()
            self.last=nn.Linear(13,5)
        def forward(self,x):
            y=self.first(x)
            return self.last(self.relu(y)+y)
    eager=Model()
    rebuilt=optimize_relu_residuals(eager)
    a=torch.randn(3,13,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    y1,y2=eager(a),rebuilt(b)
    y1.sum().backward();y2.sum().backward()
    assert_bits(y1,y2);assert_bits(a.grad,b.grad)
    for p,q in zip(eager.parameters(),rebuilt.parameters()): assert_bits(p.grad,q.grad)


def test_no_implicit_output_activation_or_inplace_rewrite():
    eager=nn.Sequential(nn.Linear(5,3),nn.ReLU(inplace=True),nn.Linear(3,2))
    result=optimize_relu_residuals(eager)
    assert isinstance(result[0],nn.Linear) and isinstance(result[2],nn.Linear)
    assert isinstance(result[1],nn.ReLU) and result[1].inplace
    x=torch.randn(2,5)
    assert_bits(eager(x),result(x))


def test_aot_compiled_packed_model():
    model=optimize_relu_residuals(nn.Sequential(nn.Linear(7,11),nn.ReLU(),nn.Linear(11,3)), policy='pack')
    compiled=torch.compile(model,backend='aot_eager',fullgraph=True)
    a=torch.randn(3,7,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    ya,yb=model(a),compiled(b)
    ga=torch.autograd.grad(ya.sum(),a)[0]
    gb=torch.autograd.grad(yb.sum(),b)[0]
    torch.testing.assert_close(ya,yb,rtol=0,atol=0)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['strict','algebraic'])
def test_shared_qkv_projection_gradients(mode):
    x=torch.randn(3,17,dtype=torch.double,requires_grad=True)
    weights=tuple(torch.randn(n,17,dtype=torch.double,requires_grad=True) for n in (11,13,7))
    expected=shared_projections(x,weights,'strict')
    observed=shared_projections(x,weights,mode)
    dy=tuple(torch.randn_like(y) for y in expected)
    refgrads=torch.autograd.grad(expected,(x,*weights),dy)
    grads=torch.autograd.grad(observed,(x,*weights),dy)
    for a,b in zip(observed,expected): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for a,b in zip(grads,refgrads): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_small_transformer_mlp_block_gradients():
    torch.manual_seed(13)
    x=torch.randn(2,3,17,dtype=torch.double,requires_grad=True)
    gamma=torch.randn(17,dtype=torch.double,requires_grad=True)
    wg=torch.randn(31,17,dtype=torch.double,requires_grad=True)
    wu=torch.randn(31,17,dtype=torch.double,requires_grad=True)
    wd=torch.randn(17,31,dtype=torch.double,requires_grad=True)
    args=(x,gamma,wg,wu,wd)
    n=rmsnorm(x,gamma)
    y=x+torch.nn.functional.linear(swiglu(n@wg.T,n@wu.T),wd)
    r=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*gamma
    ref=x+torch.nn.functional.linear(torch.nn.functional.silu(r@wg.T)*(r@wu.T),wd)
    dy=torch.randn_like(y)
    got=torch.autograd.grad(y,args,dy)
    expected=torch.autograd.grad(ref,args,dy)
    torch.testing.assert_close(y,ref,rtol=1e-12,atol=1e-12)
    for a,b in zip(got,expected): torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_conservative_policy_avoids_redundant_relu_masks():
    from ojasx2 import PackedReLU
    model=nn.Sequential(nn.Linear(5,7),nn.ReLU(),nn.Linear(7,3),nn.ReLU())
    adapted=optimize_relu_residuals(model)
    assert isinstance(adapted[1],nn.ReLU)
    assert isinstance(adapted[3],PackedReLU)
    with pytest.raises(ValueError): optimize_relu_residuals(model,policy='unverified')
