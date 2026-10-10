"""Exact-real homogeneous polynomial feature contraction reference.

No automatic compiler dispatch. Floating reassociation is tolerance based.
The logical cost model includes all input/weight VJPs and feature construction,
but excludes tensor launches, layout copies and allocator behavior.
"""
from collections import Counter
from itertools import combinations_with_replacement
import math
import torch

def feature_basis(width, degree):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (width,degree)):
        raise ValueError('Positive integer width and degree required')
    if degree>3 or math.comb(width+degree-1,degree)>4096:
        raise ValueError('Reference feature expansion exceeds its explicit budget')
    terms=tuple(combinations_with_replacement(range(width),degree))
    weights=tuple(math.factorial(degree)//math.prod(math.factorial(c) for c in Counter(t).values()) for t in terms)
    return terms,weights

def _domain(x,a,b,degree):
    if x.ndim!=2 or a.ndim!=2 or b.ndim!=2 or x.shape[1]!=a.shape[0] or a.shape[1]!=b.shape[0]:
        raise ValueError('Expected X[m,k], A[k,n], B[n,p]')
    if min(*x.shape,*a.shape,*b.shape)<=0:
        raise ValueError('Nonempty matrices required by this reference')
    if x.dtype not in (torch.float32,torch.float64) or any(t.dtype!=x.dtype or t.device!=x.device for t in (a,b)):
        raise ValueError('Matching real FP32/FP64 tensors required')
    return feature_basis(x.shape[1],degree)

def _lift(x,terms):
    return torch.stack([x[:,list(t)].prod(-1) for t in terms],-1)

def _lift_pull(x,up,terms):
    # Division-free product differentiation, including repeated factors and zero.
    if len(terms[0])==1:
        return up
    out=torch.zeros_like(x)
    for j,t in enumerate(terms):
        for pos,i in enumerate(t):
            rest=t[:pos]+t[pos+1:]
            factor=x[:,list(rest)].prod(-1) if rest else torch.ones_like(x[:,i])
            contribution=up[:,j]*factor
            # Functional accumulation retains correct higher-order derivatives.
            out=out.index_add(1,torch.tensor([i],device=x.device),contribution[:,None])
    return out

def polynomial_core(x,a,b,degree=2):
    """Return ((X@A)**degree)@B using symmetric monomial features."""
    terms,weights=_domain(x,a,b,degree)
    t=_lift(x,terms);h=_lift(a.T,terms).T*a.new_tensor(weights)[:,None]
    return t@(h@b)

def polynomial_core_vjp(x,a,b,g,degree=2):
    """Manual reference: output plus all three original operand gradients."""
    terms,weights=_domain(x,a,b,degree)
    if g.shape!=(x.shape[0],b.shape[1]) or g.dtype!=x.dtype or g.device!=x.device:
        raise ValueError('Cotangent must match output shape, dtype and device')
    w=a.new_tensor(weights)
    t=_lift(x,terms);h=_lift(a.T,terms).T*w[:,None];c=h@b
    dc=t.T@g;dt=g@c.T
    dh=dc@b.T;db=h.T@dc
    dx=_lift_pull(x,dt,terms)
    da=_lift_pull(a.T,dh.T*w,terms).T
    return t@c,(dx,da,db)

def _joint_mm(m,k,n):
    return 3*m*k*n, m*n*(k-1)+m*k*(n-1)+k*n*(m-1)

def polynomial_cost(m,k,n,p,degree=2):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (m,k,n,p)):
        raise ValueError('Positive integer shapes required')
    terms,weights=feature_basis(k,degree);s=len(terms);nonunit=sum(w!=1 for w in weights)
    nm0,na0=_joint_mm(m,k,n);nm1,na1=_joint_mm(m,n,p)
    cm0,ca0=_joint_mm(s,n,p);cm1,ca1=_joint_mm(m,s,p)
    native_mul=nm0+nm1+(2*degree-1)*m*n if degree>1 else nm0+nm1
    native_add=na0+na1
    # Scalar monomial reference circuit: d-1 forward products and d products
    # of d factors in reverse, initialized-zero accumulation per occurrence.
    overhead_mul=(degree*degree-1)*s*(m+n)+2*nonunit*n
    overhead_add=degree*s*(m+n) if degree>1 else 0
    core_mul=cm0+cm1+overhead_mul;core_add=ca0+ca1+overhead_add
    parameters=k*n+n*p
    optimizer={'multiplications':parameters,'additions':parameters}
    native_total=native_mul+native_add+2*parameters
    core_total=core_mul+core_add+2*parameters
    # The specified native schedule retains Z and Z**d. Retaining just Z
    # instead would require extra recomputation in the second-weight VJP.
    native_residual=(2 if degree>1 else 1)*m*n;core_residual=s*(m+n+p)
    return {'shape':[m,k,n,p],'degree':degree,'features':s,
        'native':{'multiplications':native_mul,'additions':native_add,'total_with_sgd':native_total,'residual_values':native_residual},
        'core':{'multiplications':core_mul,'additions':core_add,'total_with_sgd':core_total,'residual_values':core_residual},
        'sgd':optimizer,'eligible':core_total<native_total and core_residual<=native_residual,
        'arithmetic_ratio':native_total/core_total,'residual_ratio':native_residual/core_residual,
        'scope':'Logical scalar monomial circuit + six GEMMs + shared SGD update. Parameter/input storage, transient workspace, copies and launches excluded. Python tensor overhead is not this scalar circuit.'}
