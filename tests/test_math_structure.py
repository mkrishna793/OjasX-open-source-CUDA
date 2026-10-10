from fractions import Fraction
import pytest
import sympy as sp
import torch
from tools.math_structure import (joint_product_pullback, prefix_suffix_pullback, multilinear_circuit,
                                  sphere_project, stiefel_project)


@pytest.mark.parametrize('circuit,cost',[(joint_product_pullback,lambda r:3*r-4),
                                        (prefix_suffix_pullback,lambda r:4*r-6)])
@pytest.mark.parametrize('r',range(0,10))
def test_product_symbolic_and_count(r,circuit,cost):
    factors=sp.symbols('e0:'+str(r));g=sp.Symbol('g')
    calls=[]
    def counted(a,b):
        calls.append(1)
        return a*b
    gradients=circuit(factors,g,counted)
    expected=tuple(g*sp.diff(sp.prod(factors),e) for e in factors)
    assert len(gradients)==r
    assert all(sp.expand(a-b)==0 for a,b in zip(gradients,expected))
    assert len(calls)==(0 if r<2 else cost(r))


@pytest.mark.parametrize('factors',[(0,2),(0,2,3,4),(0,0,3,4),(0,0,0,4),
                                   (-2,3,-5,7),(Fraction(1,3),Fraction(2,7),0)])
def test_product_zero_and_exact_rationals(factors):
    g=Fraction(3,5)
    expected=[]
    for i in range(len(factors)):
        value=g
        for j,e in enumerate(factors):
            if j!=i:value*=e
        expected.append(value)
    assert joint_product_pullback(factors,g)==tuple(expected)


@pytest.mark.parametrize('dims',[(2,3),(2,3,4),(3,2,5,4),(2,3,2,3,2,3,2,3)])
def test_multilinear_all_input_gradients(dims):
    torch.manual_seed(123)
    u=tuple(torch.randn(5,d,dtype=torch.float64) for d in dims)
    a=tuple(torch.randn(d,dtype=torch.float64,requires_grad=True) for d in dims)
    w=torch.randn(3,5,dtype=torch.float64);g=torch.randn(3,dtype=torch.float64)
    y,got=multilinear_circuit(u,w,a,g)
    want=torch.autograd.grad(y,a,g,create_graph=True)
    for x,z in zip(got,want):torch.testing.assert_close(x,z,rtol=1e-10,atol=1e-10)
    # The algebraic reference remains differentiable for higher-order reasoning.
    got_second=torch.autograd.grad(sum(h.sum() for h in got),a,retain_graph=True)
    want_second=torch.autograd.grad(sum(h.sum() for h in want),a)
    for x,z in zip(got_second,want_second):
        torch.testing.assert_close(x,z,rtol=1e-9,atol=1e-9)


def test_sphere_projection_tangent_pairing_and_idempotence():
    torch.manual_seed(222)
    x=torch.randn(7,dtype=torch.float64);x/=torch.linalg.vector_norm(x)
    g=torch.randn_like(x);v=sphere_project(x,torch.randn_like(x));p=sphere_project(x,g)
    torch.testing.assert_close(x@p,torch.zeros((),dtype=x.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close(v@p,v@g,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(sphere_project(x,p),p,atol=1e-12,rtol=1e-12)


def test_stiefel_projection_tangent_and_pairing():
    torch.manual_seed(333)
    x=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q
    g=torch.randn_like(x);p=stiefel_project(x,g)
    v=stiefel_project(x,torch.randn_like(x))
    torch.testing.assert_close(x.T@p+p.T@x,torch.zeros(3,3,dtype=x.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close((v*p).sum(),(v*g).sum(),atol=1e-12,rtol=1e-12)


def test_error_bound_needs_amplification_term():
    # A small downstream error can grow greatly through an upstream pullback.
    lip=100.;eq=.01;ep=.001;g=2.
    q=lambda z:z
    qhat=lambda z:z+eq
    p=lambda z:lip*z
    phat=lambda z:p(z)+ep
    error=abs(phat(qhat(g))-p(q(g)))
    assert error==pytest.approx(ep+lip*eq)
    assert error>ep+eq
