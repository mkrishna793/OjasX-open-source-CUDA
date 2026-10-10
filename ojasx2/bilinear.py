"""Rank-7 block algebra. Floating equality and real equality are separate claims."""
from dataclasses import dataclass
from typing import Tuple
import torch

U = ((1,0,0,0),(0,1,0,0),(1,1,-1,-1),(0,0,0,1),(0,0,1,1),(-1,0,1,1),(1,0,-1,0))
V = ((1,0,0,0),(0,0,1,0),(0,0,0,1),(1,-1,-1,1),(-1,1,0,0),(1,-1,0,1),(0,-1,0,1))
W = ((1,1,0,0,0,0,0),(1,0,1,0,1,1,0),(1,0,0,-1,0,1,1),(1,0,0,0,1,1,1))


def _blocks(x):
    if x.ndim != 2 or any(d % 2 for d in x.shape):
        raise ValueError("The block experiment requires 2D even dimensions")
    m, n = x.shape[0] // 2, x.shape[1] // 2
    return x[:m,:n], x[:m,n:], x[m:,:n], x[m:,n:]


def _join(blocks):
    a,b,c,d = blocks
    return torch.cat((torch.cat((a,b),1),torch.cat((c,d),1)),0)


def _encode_a(a):
    a11,a12,a21,a22 = a
    s1=a21+a22
    s2=s1-a11
    s3=a11-a21
    s4=a12-s2
    return a11,a12,s4,a22,s1,s2,s3


def _encode_b(b):
    b11,b12,b21,b22 = b
    t1=b12-b11
    t2=b22-t1
    t3=b22-b12
    t4=t2-b21
    return b11,b21,b22,t4,t1,t2,t3


def _decode(products):
    p1,p2,p3,p4,p5,p6,p7 = products
    t1=p1+p6
    t2=t1+p7
    return p1+p2, (t1+p5)+p3, t2-p4, t2+p5


def decoder_pullback(g):
    """Seven cotangent channels using four matrix additions and one negation."""
    g11,g12,g21,g22 = _blocks(g)
    v1=g21+g22
    v2=g12+g22
    v3=g12+v1
    v4=g11+v3
    return v4,g11,g12,-g21,v2,v3,v1


def _transpose_transform(coefficients, channels):
    result=[]
    for j in range(len(coefficients[0])):
        terms=[t if row[j]==1 else -t for row,t in zip(coefficients,channels) if row[j]]
        acc=terms[0]
        for term in terms[1:]:
            acc=acc+term
        result.append(acc)
    return result


def encoder_a_pullback(channels):
    """U transpose with seven additions/subtractions instead of ten."""
    c1,c2,c3,c4,c5,c6,c7 = channels
    ds2 = c6-c3
    ds1 = c5+ds2
    return c1-ds2+c7, c2+c3, ds1-c7, c4+ds1


def encoder_b_pullback(channels):
    """V transpose with seven additions/subtractions instead of ten."""
    c1,c2,c3,c4,c5,c6,c7 = channels
    dt2 = c6+c4
    dt1 = c5-dt2
    return c1-dt1, dt1-c7, c2-c4, c3+dt2+c7


@dataclass
class WinogradTape:
    encoded_a: Tuple[torch.Tensor, ...]
    encoded_b: Tuple[torch.Tensor, ...]


def winograd_forward(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible matrices A(m,k) and B(k,n)")
    if a.dtype != b.dtype or a.device != b.device or not a.is_floating_point():
        raise ValueError("A and B must share a real floating dtype and device")
    ea,eb=_encode_a(_blocks(a)),_encode_b(_blocks(b))
    products=tuple(x@y for x,y in zip(ea,eb))
    return WinogradTape(ea,eb),_join(_decode(products))


def winograd_pullback(tape, g):
    dm=decoder_pullback(g)
    da=[d@y.T for d,y in zip(dm,tape.encoded_b)]
    db=[x.T@d for x,d in zip(tape.encoded_a,dm)]
    return _join(encoder_a_pullback(da)),_join(encoder_b_pullback(db))


def forward_cost(m, k, n):
    """Scalar additions and multiplications for even positive dimensions.

    Assumes four A encodings, four B encodings and seven decoder additions.
    FLOP count is not a runtime estimate. Conventional leaves use 2mkn-mn.
    """
    if min(m,k,n)<=0 or any(d%2 for d in (m,k,n)):
        raise ValueError("Cost model requires positive even dimensions")
    dense=2*m*k*n-m*n
    winograd=7*m*k*n//4+m*k+k*n
    return {"dense_scalar_ops":dense,"winograd_scalar_ops":winograd,
            "saving":dense-winograd,"forward_gemm_calls":7,
            "backward_gemm_calls":14,
            "dense_backward_scalar_ops":4*m*k*n-m*k-k*n,
            "winograd_backward_scalar_ops":7*m*k*n//2+m*n,
            "dense_joint_scalar_ops":6*m*k*n-m*n-m*k-k*n,
            "winograd_joint_scalar_ops":21*m*k*n//4+m*k+k*n+m*n,
            "claim":"one-level scalar operation model, not measured speed"}
