import pytest
import torch
import torch.nn.functional as F
from ojasx2 import packed_relu, gelu, silu, swiglu, rmsnorm, rmsnorm_aux
from ojasx2.residuals import pack_mask, unpack_mask, SavedTensorAudit


def assert_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


CASES=[('relu',packed_relu,F.relu),('gelu',gelu,F.gelu),
       ('gelu_tanh',lambda x:gelu(x,'tanh'),lambda x:F.gelu(x,approximate='tanh')),
       ('silu',silu,F.silu)]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('name,op,ref',CASES)
def test_activation_forward_backward_bits(dtype,name,op,ref):
    x=torch.tensor([-20.,-3.,-2.,-1.,-0.8,-0.1,-0.,0.,0.5,1.,3.,20.],dtype=dtype)
    dy=torch.linspace(-2,2,len(x),dtype=dtype)
    a,b=x.clone().requires_grad_(),x.clone().requires_grad_()
    ya,yb=op(a),ref(b)
    ga=torch.autograd.grad(ya,a,dy)[0]
    gb=torch.autograd.grad(yb,b,dy)[0]
    assert_bits(ya,yb)
    assert_bits(ga,gb)


@pytest.mark.parametrize('shape',[(0,7),(1,),(3,5),(2,3,7)])
def test_mask_roundtrip_shapes(shape):
    mask=torch.randn(shape)>0
    packed=pack_mask(mask)
    assert packed.numel()==(mask.numel()+7)//8
    assert torch.equal(unpack_mask(packed,shape),mask)


def test_relu_nan_infinity_signed_zero():
    x=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.,1.,-1.])
    dy=torch.tensor([2.,3.,4.,-0.,-0.,float('nan'),float('nan')])
    a,b=x.clone().requires_grad_(),x.clone().requires_grad_()
    ya,yb=packed_relu(a),F.relu(b)
    ga=torch.autograd.grad(ya,a,dy)[0]
    gb=torch.autograd.grad(yb,b,dy)[0]
    assert_bits(ya,yb)
    assert_bits(ga,gb)


@pytest.mark.parametrize('name,op,ref',CASES)
def test_noncontiguous(name,op,ref):
    x=torch.randn(5,7,dtype=torch.double).T
    a,b=x.detach().requires_grad_(),x.clone().detach().requires_grad_()
    dy=torch.randn_like(x)
    assert_bits(op(a),ref(b))
    assert_bits(torch.autograd.grad(op(a),a,dy)[0],torch.autograd.grad(ref(b),b,dy)[0])


@pytest.mark.parametrize('op',[packed_relu,gelu,silu])
def test_gradcheck_and_gradgradcheck(op):
    x=(torch.randn(2,3,dtype=torch.double)+0.23).requires_grad_()
    assert torch.autograd.gradcheck(op,(x,))
    assert torch.autograd.gradgradcheck(op,(x,))


@pytest.mark.parametrize('op',[packed_relu,gelu,silu])
def test_pytorch_registration_and_aot(op):
    result=torch.library.opcheck(op,(torch.randn(2,7,requires_grad=True),))
    assert set(result.values())=={'SUCCESS'}


def test_swiglu_bits_and_saved_storage():
    gate=torch.randn(8,17,requires_grad=True)
    up=torch.randn(8,17,requires_grad=True)
    dy=torch.randn_like(gate)
    with SavedTensorAudit([gate,up]) as base:
        yref=F.silu(gate)*up
    with SavedTensorAudit([gate,up]) as packed:
        y=swiglu(gate,up)
    assert_bits(y,yref)
    got=torch.autograd.grad(y,(gate,up),dy)
    ref=torch.autograd.grad(yref,(gate,up),dy)
    for a,b in zip(got,ref):
        assert_bits(a,b)
    assert base.summary()['fresh_saved_storage_bytes']==gate.numel()*gate.element_size()
    assert packed.summary()['fresh_saved_storage_bytes']==0


def test_swiglu_gradchecks_and_registration():
    args=tuple(torch.randn(2,5,dtype=torch.double,requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(swiglu,args)
    assert torch.autograd.gradgradcheck(swiglu,args)
    assert set(torch.library.opcheck(swiglu,args).values())=={'SUCCESS'}


@pytest.mark.parametrize('shape',[(7,),(3,7),(2,3,7)])
@pytest.mark.parametrize('scale',[0.001,1.,100.])
def test_rmsnorm_full_domain(shape,scale):
    x=(torch.randn(shape,dtype=torch.double)*scale).requires_grad_()
    gamma=torch.linspace(-1,1,shape[-1],dtype=torch.double).requires_grad_()
    gamma.data[0]=0
    dy=torch.randn_like(x)
    y=rmsnorm(x,gamma)
    ref=(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))*gamma
    assert_bits(y,ref)
    got=torch.autograd.grad(y,(x,gamma),dy)
    expected=torch.autograd.grad(ref,(x,gamma),dy)
    for a,b in zip(got,expected):
        torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)


def test_rmsnorm_gradchecks_aux_and_registration():
    x=torch.randn(2,5,dtype=torch.double,requires_grad=True)
    gamma=torch.randn(5,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(rmsnorm,(x,gamma))
    assert torch.autograd.gradgradcheck(rmsnorm,(x,gamma))
    assert torch.autograd.gradcheck(rmsnorm_aux,(x,gamma))
    assert set(torch.library.opcheck(rmsnorm_aux,(x,gamma,1e-6)).values())=={'SUCCESS'}


def test_relu_saved_tensor_bytes():
    x=torch.randn(256,1024,requires_grad=True)
    with SavedTensorAudit([x]) as eager:
        F.relu(x)
    with SavedTensorAudit([x]) as packed:
        packed_relu(x)
    assert eager.summary()['fresh_saved_storage_bytes']==1048576
    assert packed.summary()['fresh_saved_storage_bytes']==32768


def test_reject_invalid_contracts():
    with pytest.raises(TypeError): packed_relu(torch.ones(3,dtype=torch.int64))
    with pytest.raises(ValueError): gelu(torch.ones(3),'inverse')
    with pytest.raises(ValueError): swiglu(torch.ones(3),torch.ones(4))
    with pytest.raises(ValueError): rmsnorm(torch.ones(2,3),torch.ones(2))
    with pytest.raises(ValueError): rmsnorm(torch.ones(2,3),torch.ones(3),0.)
