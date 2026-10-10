"""Opt-in mathematical reference operations; no compiler dispatch or speed promise.

Rocq proves the supplied diagonal/low-rank adjoint identity and rank-seven
algebra. These Python realizations and analytic coefficient builders are tested;
there is no formal Rocq-to-PyTorch refinement or universal error theorem.
"""
from dataclasses import dataclass
import math
import torch

def _vector_domain(x):
    if x.ndim < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('Expected a real FP32/FP64 tensor with a coordinate axis')

def _normalization_domain(x, gamma, eps):
    _vector_domain(x)
    if x.shape[-1] == 0:
        raise ValueError('Normalization dimension must be nonempty')
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Expected finite positive scalar epsilon')
    info=torch.finfo(x.dtype)
    if eps < info.tiny*info.eps or eps > info.max:
        raise ValueError('Epsilon must lie in the representable positive dtype range')
    if gamma.dtype != x.dtype or gamma.device != x.device:
        raise ValueError('Gain must share input dtype and device')
    if torch.broadcast_shapes(x.shape,gamma.shape) != x.shape:
        raise ValueError('Gain must broadcast without expanding the input shape')

@dataclass(frozen=True)
class StructuredDerivative:
    """J = diag(d) + U V^T, batching over every axis before the last two."""
    diagonal: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self):
        _vector_domain(self.diagonal)
        if (self.left.shape != self.right.shape or
            self.left.shape[:-1] != self.diagonal.shape):
            raise ValueError('Expected diagonal [..., n], factors [..., n, r]')
        if any(t.dtype != self.diagonal.dtype or t.device != self.diagonal.device
               for t in (self.left, self.right)):
            raise ValueError('All derivative tensors must share dtype and device')

    def _apply(self, x, left, right):
        if x.shape != self.diagonal.shape or x.dtype != self.diagonal.dtype or x.device != self.diagonal.device:
            raise ValueError('Direction/cotangent must match diagonal shape, dtype and device')
        if left.shape[-1] == 0:
            return self.diagonal*x
        return self.diagonal*x + torch.einsum('...ir,...r->...i',left,
            torch.einsum('...ir,...i->...r',right,x))

    def push(self, tangent): return self._apply(tangent,self.left,self.right)
    def pull(self, cotangent): return self._apply(cotangent,self.right,self.left)

def diagonal_derivative(diagonal):
    empty=diagonal.new_empty((*diagonal.shape,0))
    return StructuredDerivative(diagonal,empty,empty)

def activation_derivative(x, kind):
    _vector_domain(x)
    if kind=='relu': d=(x>0).to(x.dtype)
    elif kind=='silu':
        s=x.sigmoid();d=s+x*s*(1-s)
    elif kind=='gelu':
        d=.5*(1+torch.erf(x/math.sqrt(2)))+x*torch.exp(-.5*x*x)/math.sqrt(2*math.pi)
    elif kind=='tanh': d=1-x.tanh().square()
    elif kind=='sigmoid':
        s=x.sigmoid();d=s*(1-s)
    elif kind=='square': d=2*x
    else: raise ValueError('Unsupported activation derivative: '+kind)
    return diagonal_derivative(d)

def softmax_derivative(probabilities):
    """Caller supplies the actual softmax output; no recomputation here."""
    _vector_domain(probabilities)
    return StructuredDerivative(probabilities,-probabilities.unsqueeze(-1),probabilities.unsqueeze(-1))

def rmsnorm_input_derivative(x, gamma, eps=1e-5):
    _normalization_domain(x,gamma,eps)
    n=x.shape[-1]
    if n==0: raise ValueError('Normalization dimension must be nonempty')
    r=torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
    z=x*r
    diagonal=(r*gamma).expand_as(x)
    return StructuredDerivative(diagonal,(-r*gamma*z/n).unsqueeze(-1),z.unsqueeze(-1))

def layernorm_input_derivative(x, gamma, eps=1e-5):
    _normalization_domain(x,gamma,eps)
    n=x.shape[-1]
    if n==0: raise ValueError('Normalization dimension must be nonempty')
    c=x-x.mean(-1,keepdim=True)
    r=torch.rsqrt(c.square().mean(-1,keepdim=True)+eps)
    z=c*r
    diagonal=(r*gamma).expand_as(x)
    left=torch.stack((-diagonal/n,-r*gamma*z/n),dim=-1)
    right=torch.stack((torch.ones_like(x),z),dim=-1)
    return StructuredDerivative(diagonal,left,right)

@dataclass(frozen=True)
class NormalizationResidual:
    """First-order sufficient values; autograd graphs are excluded from the
    residual-element model. Keep differentiable stats for higher derivatives.
    """
    kind: str
    normalized: torch.Tensor
    inverse_scale: torch.Tensor
    gamma: object
    has_bias: bool

    def __post_init__(self):
        z=self.normalized;r=self.inverse_scale
        _vector_domain(z)
        if self.kind not in ('layernorm','rmsnorm') or z.shape[-1]==0:
            raise ValueError('Expected supported normalization with nonempty coordinates')
        if r.shape!=(*z.shape[:-1],1) or r.dtype!=z.dtype or r.device!=z.device:
            raise ValueError('Inverse scale must match one value per normalization row')
        if self.gamma is not None and (self.gamma.shape!=(z.shape[-1],) or
                self.gamma.dtype!=z.dtype or self.gamma.device!=z.device):
            raise ValueError('Gain must match shared coordinate shape, dtype and device')
        if not isinstance(self.has_bias,bool):
            raise ValueError('Bias-presence flag must be Boolean')

    @property
    def residual_elements(self):
        return self.normalized.numel()+self.inverse_scale.numel()

    def pull(self, upstream):
        z=self.normalized;r=self.inverse_scale
        if upstream.shape!=z.shape or upstream.dtype!=z.dtype or upstream.device!=z.device:
            raise ValueError('Cotangent must match normalization output')
        q=upstream if self.gamma is None else upstream*self.gamma
        correction=z*(q*z).mean(-1,keepdim=True)
        if self.kind=='layernorm':
            dx=r*((q-q.mean(-1,keepdim=True))-correction)
        else:
            dx=r*(q-correction)
        dgamma=None if self.gamma is None else (upstream*z).sum_to_size(self.gamma.shape)
        dbeta=upstream.sum_to_size((z.shape[-1],)) if self.has_bias else None
        return dx,dgamma,dbeta

def normalization_residual(x, gamma=None, beta=None, eps=1e-5, kind='layernorm'):
    """Compact reference VJP for x, shared affine gain and shared affine bias.
    Epsilon is a fixed scalar. This is not a custom autograd/storage backend.
    """
    if kind not in ('layernorm','rmsnorm'):raise ValueError('Unsupported normalization kind')
    gain=x.new_ones((x.shape[-1],)) if gamma is None and x.ndim else gamma
    _normalization_domain(x,gain,eps)
    if gamma is not None and gamma.shape!=(x.shape[-1],):
        raise ValueError('Compact gain must have the shared coordinate shape')
    if beta is not None and (beta.shape!=(x.shape[-1],) or beta.dtype!=x.dtype or beta.device!=x.device):
        raise ValueError('Compact bias must match shared coordinate shape, dtype and device')
    centered=x-x.mean(-1,keepdim=True) if kind=='layernorm' else x
    r=torch.rsqrt(centered.square().mean(-1,keepdim=True)+eps)
    return NormalizationResidual(kind,centered*r,r,gamma,beta is not None)

def rank7_reference(a,b):
    """One Strassen level, conventional differentiable PyTorch operations.

    Both dimensions of each operand must be positive and even. Float results
    can differ from a@b. Extra additions and temporaries can make this slower.
    """
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError('Expected compatible matrices')
    if any(d<=0 or d%2 for d in (*a.shape,*b.shape)):
        raise ValueError('Expected positive even dimensions')
    if a.device!=b.device or a.dtype!=b.dtype or a.dtype not in (torch.float32,torch.float64):
        raise ValueError('Operands must share dtype and device')
    a00,a01=a[:a.shape[0]//2,:a.shape[1]//2],a[:a.shape[0]//2,a.shape[1]//2:]
    a10,a11=a[a.shape[0]//2:,:a.shape[1]//2],a[a.shape[0]//2:,a.shape[1]//2:]
    b00,b01=b[:b.shape[0]//2,:b.shape[1]//2],b[:b.shape[0]//2,b.shape[1]//2:]
    b10,b11=b[b.shape[0]//2:,:b.shape[1]//2],b[b.shape[0]//2:,b.shape[1]//2:]
    p1=(a00+a11)@(b00+b11);p2=(a10+a11)@b00;p3=a00@(b01-b11)
    p4=a11@(b10-b00);p5=(a00+a01)@b11;p6=(a10-a00)@(b00+b01);p7=(a01-a11)@(b10+b11)
    return torch.cat((torch.cat((p1+p4-p5+p7,p3+p5),1),
                      torch.cat((p2+p4,p1-p2+p3+p6),1)),0)

def winograd_reference(a,b):
    """Known seven-product, fifteen-addition variant; opt-in reference only."""
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError('Expected compatible matrices')
    if (any(d<=0 or d%2 for d in (*a.shape,*b.shape)) or a.device!=b.device or a.dtype!=b.dtype
        or a.dtype not in (torch.float32,torch.float64)):
        raise ValueError('Expected positive even dimensions and matching dtype/device')
    m,k,n=a.shape[0]//2,a.shape[1]//2,b.shape[1]//2
    a00,a01,a10,a11=a[:m,:k],a[:m,k:],a[m:,:k],a[m:,k:]
    b00,b01,b10,b11=b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
    s1=a10+a11;s2=s1-a00;s3=a00-a10;s4=a01-s2
    t1=b01-b00;t2=b11-t1;t3=b11-b01;t4=t2-b10
    p1=a00@b00;p2=a01@b10;p3=s4@b11;p4=a11@t4
    p5=s1@t1;p6=s2@t2;p7=s3@t3
    u1=p1+p6;u2=u1+p7;u3=u1+p5
    return torch.cat((torch.cat((p1+p2,u3+p3),1),torch.cat((u2-p4,u2+p5),1)),0)
