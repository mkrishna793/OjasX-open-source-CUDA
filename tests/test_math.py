import pytest
import torch
import sympy as sp
from ojasx2 import linear_lens, parameterized_linear_lens, activation_lens
from ojasx2.bilinear import U,V,W,winograd_forward,winograd_pullback,decoder_pullback,forward_cost


def test_noncommutative_scheme_polynomial():
    a=sp.Matrix(sp.symbols('a11 a12 a21 a22',commutative=False))
    b=sp.Matrix(sp.symbols('b11 b12 b21 b22',commutative=False))
    ea,eb=sp.Matrix(U)*a,sp.Matrix(V)*b
    c=sp.Matrix(W)*sp.Matrix([ea[i]*eb[i] for i in range(7)])
    ref=[a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]]
    assert all(sp.expand(x-y)==0 for x,y in zip(c,ref))


@pytest.mark.parametrize('shape',[(2,2,2),(6,10,14),(8,16,4),(32,32,32)])
def test_winograd_forward_pullback_rectangular(shape):
    m,k,n=shape
    a=torch.randn(m,k,dtype=torch.double,requires_grad=True)
    b=torch.randn(k,n,dtype=torch.double,requires_grad=True)
    g=torch.randn(m,n,dtype=torch.double)
    tape,c=winograd_forward(a,b)
    reference=a@b
    got=winograd_pullback(tape,g)
    grads=torch.autograd.grad(reference,(a,b),g)
    torch.testing.assert_close(c,reference,rtol=1e-12,atol=1e-12)
    for x,y in zip(got,grads):
        torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)


def test_decoder_is_adjoint():
    from ojasx2.bilinear import _decode,_join
    ps=tuple(torch.randn(2,3,dtype=torch.double,requires_grad=True) for _ in range(7))
    g=torch.randn(4,6,dtype=torch.double)
    actual=decoder_pullback(g)
    ref=torch.autograd.grad(_join(_decode(ps)),ps,g)
    for a,b in zip(actual,ref):
        torch.testing.assert_close(a,b,rtol=0,atol=2e-15)


def test_linear_adjoint_pairing_and_parameter_gradients():
    x=torch.randn(7,5,dtype=torch.double,requires_grad=True)
    w=torch.randn(3,5,dtype=torch.double,requires_grad=True)
    dy=torch.randn(7,3,dtype=torch.double)
    lens=linear_lens(w)
    tape,y=lens.forward(x)
    torch.testing.assert_close((y*dy).sum(),(x*lens.pullback(tape,dy)).sum())
    p=parameterized_linear_lens()
    tape,y=p.forward((x,w))
    got=p.pullback(tape,dy)
    for a,b in zip(got,torch.autograd.grad(y,(x,w),dy)):
        torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('activation',['relu','gelu','silu'])
def test_nonlinear_lens_composition_uses_primal_residual(activation):
    w1=torch.randn(9,5,dtype=torch.double)
    w2=torch.randn(3,9,dtype=torch.double)
    x=torch.randn(7,5,dtype=torch.double,requires_grad=True)
    lens=linear_lens(w1).then(activation_lens(activation)).then(linear_lens(w2))
    tape,y=lens.forward(x)
    dy=torch.randn_like(y)
    expected=torch.autograd.grad(y,x,dy)[0]
    torch.testing.assert_close(lens.pullback(tape,dy),expected,rtol=1e-12,atol=1e-12)


def test_corrected_arithmetic_cost():
    assert forward_cost(16,16,16)['winograd_scalar_ops']==7*16**3//4+2*16**2
    assert forward_cost(12,12,12)['saving']==0
    assert forward_cost(14,14,14)['saving']>0
    assert forward_cost(2,4096,14336)['saving']<0


def test_invalid_block_shapes():
    with pytest.raises(ValueError): winograd_forward(torch.randn(3,4),torch.randn(4,2))
    with pytest.raises(ValueError): winograd_forward(torch.randn(2,4),torch.randn(6,2))


@pytest.mark.parametrize('side',['a','b'])
def test_encoder_adjoint_common_subexpressions(side):
    from ojasx2.bilinear import encoder_a_pullback,encoder_b_pullback,_transpose_transform
    channels=tuple(torch.randn(3,5,dtype=torch.double) for _ in range(7))
    op,coefficients=(encoder_a_pullback,U) if side=='a' else (encoder_b_pullback,V)
    for actual,expected in zip(op(channels),_transpose_transform(coefficients,channels)):
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
