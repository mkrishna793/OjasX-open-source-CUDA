import copy
import itertools
import pytest
import torch
from ojasx2 import linear_chain,chain_cost,factored_gemm_pullback
from ojasx2.contractions import _LinearChain


@pytest.fixture(autouse=True)
def isolate_independent_compiler_models():
    # This file qualifies many different Sequential models sharing one Python
    # forward frame. Reset their test-local cache rather than raising or masking
    # PyTorch's production recompilation limit.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('mode',['joint','backward_only'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_chain_forward_and_all_gradients(mode,dtype):
    torch.manual_seed(23)
    x=torch.randn(11,7,dtype=dtype,requires_grad=True)
    a=torch.randn(19,7,dtype=dtype,requires_grad=True)
    b=torch.randn(3,19,dtype=dtype,requires_grad=True)
    y=torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    z=linear_chain(x,a,b,'algebraic',mode);g=torch.randn_like(y)
    gy=torch.autograd.grad(y,(x,a,b),g)
    gz=torch.autograd.grad(z,(x,a,b),g)
    tol=3e-5 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(z,y,rtol=tol,atol=tol)
    for h,k in zip(gy,gz):torch.testing.assert_close(h,k,rtol=tol,atol=tol)
    if mode=='backward_only':assert torch.equal(y.view(torch.uint8),z.view(torch.uint8))


@pytest.mark.parametrize('mode',['joint','backward_only'])
def test_chain_gradcheck_and_gradgradcheck(mode):
    torch.manual_seed(10)
    args=(torch.randn(5,3,dtype=torch.double,requires_grad=True),
          torch.randn(7,3,dtype=torch.double,requires_grad=True),
          torch.randn(1,7,dtype=torch.double,requires_grad=True))
    fn=lambda *xs:_LinearChain.apply(*xs,mode)
    assert torch.autograd.gradcheck(fn,args)
    assert torch.autograd.gradgradcheck(fn,args)


@pytest.mark.parametrize('needs',list(itertools.product([False,True],repeat=3))[1:])
def test_chain_requested_gradients(needs):
    torch.manual_seed(55)
    args=tuple(torch.randn(*shape,dtype=torch.double,requires_grad=need)
               for shape,need in zip(((9,5),(13,5),(2,13)),needs))
    wanted=tuple(t for t in args if t.requires_grad)
    y=torch.nn.functional.linear(torch.nn.functional.linear(args[0],args[1]),args[2])
    z=_LinearChain.apply(*args,'joint');g=torch.randn_like(y)
    ref=torch.autograd.grad(y,wanted,g)
    got=torch.autograd.grad(z,wanted,g)
    for a,b in zip(got,ref):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_chain_strict_and_unprofitable_fallback():
    x=torch.randn(3,5,requires_grad=True);a=torch.randn(7,5,requires_grad=True);b=torch.randn(11,7,requires_grad=True)
    ref=torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    for mode in ['strict','algebraic']:
        got=linear_chain(x,a,b,mode)
        assert torch.equal(got,ref)
    assert chain_cost(3,5,7,11)['scalar_saving']<0


def test_chain_version_check_and_noncontiguous_cotangent():
    x=torch.randn(9,5,requires_grad=True);a=torch.randn(13,5,requires_grad=True);b=torch.randn(2,13,requires_grad=True)
    y=_LinearChain.apply(x,a,b,'joint')
    g=torch.randn(2,9).T
    assert not g.is_contiguous()
    torch.autograd.grad(y,(x,a,b),g)
    y=_LinearChain.apply(x,a,b,'joint')
    with torch.no_grad():a.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):y.sum().backward()


def test_factored_cotangent_gradients():
    torch.manual_seed(50)
    a=torch.randn(17,13,dtype=torch.double);b=torch.randn(13,19,dtype=torch.double)
    l=torch.randn(17,3,dtype=torch.double);r=torch.randn(19,3,dtype=torch.double)
    g=l@r.T
    ga,gb=factored_gemm_pullback(a,b,l,r)
    torch.testing.assert_close(ga,g@b.T,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gb,a.T@g,rtol=1e-12,atol=1e-12)


def test_activation_can_destroy_low_rank():
    # A ReLU derivative mask can turn a rank-one cotangent into rank two.
    g=torch.ones(2,2,dtype=torch.double)
    mask=torch.eye(2,dtype=torch.double)
    assert torch.linalg.matrix_rank(g)==1
    assert torch.linalg.matrix_rank(g*mask)==2


def test_20_step_training_trajectory_matches_with_tolerance():
    torch.manual_seed(92)
    a=torch.nn.Parameter(torch.randn(19,7,dtype=torch.double)*.1)
    b=torch.nn.Parameter(torch.randn(3,19,dtype=torch.double)*.1)
    c=torch.nn.Parameter(a.detach().clone());d=torch.nn.Parameter(b.detach().clone())
    o1=torch.optim.Adam((a,b),lr=1e-3);o2=torch.optim.Adam((c,d),lr=1e-3)
    for _ in range(20):
        x=torch.randn(11,7,dtype=torch.double);target=torch.randn(11,3,dtype=torch.double)
        y=torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
        z=linear_chain(x,c,d,'algebraic')
        o1.zero_grad(set_to_none=True);o2.zero_grad(set_to_none=True)
        (y-target).square().mean().backward();(z-target).square().mean().backward()
        torch.testing.assert_close(a.grad,c.grad,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(b.grad,d.grad,rtol=1e-11,atol=1e-11)
        o1.step();o2.step()
        torch.testing.assert_close(a,c,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(b,d,rtol=1e-11,atol=1e-11)


def test_compiler_linear_chain_preserves_optimizer_parameters():
    from ojasx2 import make_backend
    torch.manual_seed(44)
    model=torch.nn.Sequential(torch.nn.Linear(7,19,bias=False),torch.nn.Linear(19,3,bias=False)).double()
    other=copy.deepcopy(model)
    backend=make_backend(accuracy='algebraic',contraction_policy='linear_chain')
    compiled=torch.compile(other,backend=backend,fullgraph=True)
    original=tuple(other.parameters())
    x=torch.randn(11,7,dtype=torch.double,requires_grad=True)
    y=model(x);z=compiled(x)
    wanted=torch.autograd.grad(y.square().mean(),(x,*model.parameters()))
    got=torch.autograd.grad(z.square().mean(),(x,*other.parameters()))
    for a,b in zip(wanted,got):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    assert backend.evidence[-1]['linear_chain_rewrites']==1
    assert all(p is q for p,q in zip(original,other.parameters()))


@pytest.mark.parametrize('middle',[torch.nn.ReLU(),torch.nn.SiLU(),torch.nn.Dropout(.1)])
def test_compiler_does_not_cross_activation_or_dropout(middle):
    from ojasx2 import make_backend
    model=torch.nn.Sequential(torch.nn.Linear(7,19,bias=False),middle,torch.nn.Linear(19,3,bias=False))
    backend=make_backend(accuracy='algebraic',contraction_policy='linear_chain')
    compiled=torch.compile(model,backend=backend,fullgraph=True)
    compiled(torch.randn(11,7))
    assert backend.evidence[-1]['linear_chain_rewrites']==0


def test_compiler_does_not_rewrite_fanout():
    from ojasx2 import make_backend
    class Branched(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Linear(7,19,bias=False);self.b=torch.nn.Linear(19,3,bias=False)
        def forward(self,x):
            h=self.a(x);return self.b(h),h.square().sum()
    for model in [Branched()]:
        backend=make_backend(accuracy='algebraic',contraction_policy='linear_chain')
        torch.compile(model,backend=backend,fullgraph=True)(torch.randn(11,7))
        assert backend.evidence[-1]['linear_chain_rewrites']==0


def test_cpu_autocast_uses_native_fallback():
    x=torch.randn(11,7,requires_grad=True);a=torch.randn(19,7,requires_grad=True);b=torch.randn(3,19,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        ref=torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
        got=linear_chain(x,a,b,'algebraic','backward_only')
    assert torch.equal(got,ref)
    r=torch.autograd.grad(ref.float().square().sum(),(x,a,b))
    g=torch.autograd.grad(got.float().square().sum(),(x,a,b))
    for p,q in zip(r,g):assert torch.equal(p,q)


@pytest.mark.parametrize('bias1,bias2',[(True,True),(True,False),(False,True)])
def test_joint_affine_chain_all_parameter_gradients(bias1,bias2):
    from ojasx2 import make_backend
    torch.manual_seed(15)
    model=torch.nn.Sequential(torch.nn.Linear(7,19,bias=bias1),torch.nn.Linear(19,3,bias=bias2)).double()
    other=copy.deepcopy(model)
    backend=make_backend(accuracy='algebraic',contraction_policy='linear_chain')
    compiled=torch.compile(other,backend=backend,fullgraph=True)
    x=torch.randn(11,7,dtype=torch.double,requires_grad=True)
    y=model(x);z=compiled(x);g=torch.randn_like(y)
    want=torch.autograd.grad(y,(x,*model.parameters()),g)
    got=torch.autograd.grad(z,(x,*other.parameters()),g)
    torch.testing.assert_close(z,y,rtol=1e-12,atol=1e-12)
    for a,b in zip(want,got):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert backend.evidence[-1]['linear_chain_rewrites']==1


def test_backward_only_bias_preserves_native_fallback():
    x=torch.randn(11,7,requires_grad=True);a=torch.randn(19,7,requires_grad=True);b=torch.randn(3,19,requires_grad=True)
    bias1=torch.randn(19,requires_grad=True);bias2=torch.randn(3,requires_grad=True)
    got=linear_chain(x,a,b,'algebraic','backward_only',bias1,bias2)
    ref=torch.nn.functional.linear(torch.nn.functional.linear(x,a,bias1),b,bias2)
    assert torch.equal(got,ref)
