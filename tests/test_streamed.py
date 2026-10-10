import pytest
import torch
from ojasx2 import operand_tape_gemm, streamed_gemm
from ojasx2.bilinear import winograd_forward, winograd_pullback
from ojasx2.streamed import _joint_pullback, streamed_cost
from ojasx2.residuals import SavedTensorAudit


@pytest.mark.parametrize('shape',[(2,2,2),(6,10,14),(32,64,16),(128,128,128)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_streamed_gradients(shape,dtype):
    m,k,n=shape
    a=torch.randn(m,k,dtype=dtype,requires_grad=True)
    b=torch.randn(k,n,dtype=dtype,requires_grad=True)
    g=torch.randn(m,n,dtype=dtype)
    y=streamed_gemm(a,b,'algebraic')
    grads=torch.autograd.grad(y,(a,b),g)
    expected=torch.autograd.grad(a@b,(a,b),g)
    tol=4e-5 if dtype==torch.float32 else 2e-12
    # Scale-aware absolute tolerance also covers cancellation near zero.
    for x,z in zip((y,*grads),(a@b,*expected)):
        torch.testing.assert_close(x,z,rtol=tol,atol=tol)


def test_joint_output_slot_schedule():
    a=torch.randn(12,16,dtype=torch.double)
    b=torch.randn(16,10,dtype=torch.double)
    g=torch.randn(12,10,dtype=torch.double)
    tape,_=winograd_forward(a,b)
    expected=winograd_pullback(tape,g)
    for x,z in zip(_joint_pullback(a,b,g),expected):
        torch.testing.assert_close(x,z,rtol=2e-12,atol=2e-12)


def test_operand_only_saved_storage():
    a=torch.randn(128,256,requires_grad=True)
    b=torch.randn(256,64,requires_grad=True)
    with SavedTensorAudit([a,b]) as encoded:
        _,y=winograd_forward(a,b)
    with SavedTensorAudit([a,b]) as streamed:
        z=streamed_gemm(a,b,'algebraic')
    elements=a.numel()+b.numel()
    assert encoded.summary()['fresh_saved_storage_bytes']==elements*4
    assert streamed.summary()['fresh_saved_storage_bytes']==0
    assert encoded.summary()['unique_saved_storage_bytes']==2*elements*4
    assert streamed.summary()['unique_saved_storage_bytes']==elements*4
    torch.autograd.grad(y,(a,b),torch.ones_like(y))
    torch.autograd.grad(z,(a,b),torch.ones_like(z))


@pytest.mark.parametrize('needs',[(True,False),(False,True)])
def test_gradient_demand(needs):
    a=torch.randn(8,12,dtype=torch.double,requires_grad=needs[0])
    b=torch.randn(12,6,dtype=torch.double,requires_grad=needs[1])
    y=streamed_gemm(a,b,'algebraic')
    leaf=a if needs[0] else b
    x=torch.autograd.grad(y.sum(),leaf)[0]
    z=torch.autograd.grad((a@b).sum(),leaf)[0]
    torch.testing.assert_close(x,z,rtol=2e-12,atol=2e-12)


def test_gradcheck_gradgradcheck_and_registration():
    args=tuple(torch.randn(4,4,dtype=torch.double,requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(operand_tape_gemm,args)
    assert torch.autograd.gradgradcheck(operand_tape_gemm,args)
    assert set(torch.library.opcheck(operand_tape_gemm,args).values())=={'SUCCESS'}


def test_input_mutation_version_safety():
    a=torch.randn(4,4,requires_grad=True)
    b=torch.randn(4,4,requires_grad=True)
    y=streamed_gemm(a,b,'algebraic')
    with torch.no_grad(): a.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        y.sum().backward()


def test_strict_default_and_contracts():
    a,b=torch.randn(3,7),torch.randn(7,5)
    assert torch.equal(streamed_gemm(a,b),(a@b))
    with pytest.raises(ValueError): streamed_gemm(a,b,'algebraic')
    with pytest.raises(ValueError): streamed_gemm(a,b,'approximate')
    assert streamed_cost(12,12,12)['scalar_saving']<0
    assert streamed_cost(14,14,14)['scalar_saving']>0


def test_medium_training_trajectory():
    torch.manual_seed(991)
    # Fixed schedule of real bilinear operations, nonlinear activation, trainable
    # parameters and fanout. Algebraic mode receives tolerance qualification.
    w=torch.randn(128,64,dtype=torch.double)/8
    w1=torch.nn.Parameter(w.clone());w2=torch.nn.Parameter(w.clone())
    opt1=torch.optim.SGD([w1],lr=1e-3)
    opt2=torch.optim.SGD([w2],lr=1e-3)
    for _ in range(12):
        x=torch.randn(32,128,dtype=torch.double,requires_grad=True)
        z=x.detach().clone().requires_grad_()
        y=torch.nn.functional.silu(x@w1)
        v=torch.nn.functional.silu(streamed_gemm(z,w2,'algebraic'))
        loss1=y.square().mean()+0.01*y.mean()
        loss2=v.square().mean()+0.01*v.mean()
        loss1.backward();loss2.backward()
        for a,b in ((y,v),(x.grad,z.grad),(w1.grad,w2.grad)):
            torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-11)
        opt1.step();opt2.step()
        opt1.zero_grad();opt2.zero_grad()
    torch.testing.assert_close(w1,w2,rtol=2e-10,atol=2e-11)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_noncontiguous_operands_and_cotangent(dtype):
    a=torch.randn(16,12,dtype=dtype).T.detach().requires_grad_()
    b=torch.randn(8,16,dtype=dtype).T.detach().requires_grad_()
    g=torch.randn(8,12,dtype=dtype).T
    y=streamed_gemm(a,b,'algebraic')
    got=torch.autograd.grad(y,(a,b),g)
    ref=a@b
    expected=torch.autograd.grad(ref,(a,b),g)
    tol=5e-5 if dtype==torch.float32 else 2e-12
    for x,z in zip((y,*got),(ref,*expected)):
        torch.testing.assert_close(x,z,rtol=tol,atol=tol)


def test_same_leaf_operand_accumulation():
    a=torch.randn(8,8,dtype=torch.double,requires_grad=True)
    y=streamed_gemm(a,a,'algebraic')
    g=torch.randn_like(y)
    actual=torch.autograd.grad(y,a,g)[0]
    expected=torch.autograd.grad(a@a,a,g)[0]
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
