"""GPU-only experimental small-domain Taylor feature contraction.

Changes GELU/SiLU/SwiGLU to a polynomial surrogate inside an explicit gate.
Opt-in research, no automatic compiler dispatch or universal error guarantee.
"""
import math
import torch
import torch.nn.functional as F
from .taylor_coefficients import BUILDERS,PULLS,TERMS,COUNTS

def native_block(x,a,b,v=None,kind='gelu'):
    if any(t.device.type!='mps' for t in (x,a,b) if t is not None) or (v is not None and v.device.type!='mps'):
        raise ValueError('GPU-only research workload')
    z=x@a
    if kind=='gelu':h=F.gelu(z)
    elif kind=='silu':h=F.silu(z)
    elif kind=='swiglu':h=F.silu(z)*(x@v)
    else:raise ValueError('Unsupported activation')
    return h@b

def _key(x,kind):return ('swiglu' if kind=='swiglu' else 'activation',x.shape[1])
def _alpha(kind):return 1/math.sqrt(2*math.pi) if kind=='gelu' else .25

def _lift(x,terms):
    return torch.stack([x[:,list(t)].prod(-1) if len(t)>1 else x[:,t[0]] for t in terms],-1)

def _lift_pull(x,dt,terms):
    result=torch.zeros_like(x)
    for j,t in enumerate(terms):
        for pos,i in enumerate(t):
            rest=t[:pos]+t[pos+1:]
            contribution=dt[:,j]
            if rest:contribution=contribution*x[:,list(rest)].prod(-1)
            result=result.index_add(1,torch.tensor([i],device=x.device),contribution[:,None])
    return result

def feature_values(x,a,b,v,kind):
    # Full tensor powers use slightly more features than the compressed
    # symmetric basis, but evaluate the same polynomial with fewer dispatches.
    k=x.shape[1]
    q=(x[:,:,None]*x[:,None,:]).flatten(1)
    if kind=='swiglu':
        cubic=(q[:,:,None]*x[:,None,:]).flatten(1)
        t=torch.cat((q,cubic),1)
        h2=(.5*a[:,None,:]*v[None,:,:]).flatten(0,1)
        h3=(.25*a[:,None,None,:]*a[None,:,None,:]*v[None,None,:,:]).flatten(0,2)
        h=torch.cat((h2,h3),0)
    else:
        t=torch.cat((x,q),1)
        h=torch.cat((.5*a,(_alpha(kind)*a[:,None,:]*a[None,:,:]).flatten(0,1)),0)
    return t,h,h@b

def surrogate_block(x,a,b,v=None,kind='gelu'):
    """Ordinary evaluation of the same polynomial, for derivative checks."""
    z=x@a;h=.5*z+_alpha(kind)*z.square()
    if kind=='swiglu':h=h*(x@v)
    return h@b

class _TaylorFeature(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,b,v,kind):
        t,h,c=feature_values(x,a,b,v,kind)
        ctx.kind=kind;ctx.save_for_backward(x,a,b,v,t,h,c)
        return t@c
    @staticmethod
    def backward(ctx,g):
        x,a,b,v,t,h,c=ctx.saved_tensors;kind=ctx.kind;k=x.shape[1]
        if torch.is_grad_enabled():t,h,c=feature_values(x,a,b,v,kind)
        dc=t.T@g;dt=g@c.T;dh=dc@b.T;db=h.T@dc
        if kind=='swiglu':
            d2=dh[:k*k].reshape(k,k,-1);d3=dh[k*k:].reshape(k,k,k,-1)
            da=.5*(d2*v[None,:,:]).sum(1)+.25*((d3+d3.transpose(0,1))*a[None,:,None,:]*v[None,None,:,:]).sum((1,2))
            dv=.5*(d2*a[:,None,:]).sum(0)+.25*(d3*a[:,None,None,:]*a[None,:,None,:]).sum((0,1))
            q2=dt[:,:k*k].reshape(-1,k,k);q3=dt[:,k*k:].reshape(-1,k,k,k)
            dx=((q2+q2.transpose(1,2))*x[:,None,:]).sum(2)
            dx=dx+((q3+q3.transpose(1,2)+q3.permute(0,3,1,2))*x[:,None,:,None]*x[:,None,None,:]).sum((2,3))
        else:
            alpha=_alpha(kind);d1=dh[:k];d2=dh[k:].reshape(k,k,-1)
            da=.5*d1+alpha*((d2+d2.transpose(0,1))*a[None,:,:]).sum(1);dv=None
            q2=dt[:,k:].reshape(-1,k,k)
            dx=dt[:,:k]+((q2+q2.transpose(1,2))*x[:,None,:]).sum(2)
        return dx,da,db,dv,None

def _mm(m,k,n):return 3*m*k*n,m*n*(k-1)+m*k*(n-1)+k*n*(m-1)

def taylor_cost(m,k,n,p,kind):
    key=('swiglu' if kind=='swiglu' else 'activation',k)
    if key not in TERMS:return {'eligible':False,'reason':'unsupported_feature_width'}
    # Conservative straight-line scalar upper bounds for the tensor-power
    # implementation. All coefficient and feature pulls are charged.
    s=k*k+k*k*k if kind=='swiglu' else k+k*k
    native0=_mm(m,k,n);native1=_mm(m,n,p)
    native_mul=(2 if kind=='swiglu' else 1)*native0[0]+native1[0]
    native_add=(2 if kind=='swiglu' else 1)*native0[1]+native1[1]+(m*k if kind=='swiglu' else 0)
    core0=_mm(s,n,p);core1=_mm(m,s,p)
    if kind=='swiglu':
        # Tensor powers, coefficient construction, and their two pullbacks.
        extra_mul=2*m*k*k+3*m*k**3+4*n*k*k+9*n*k**3+3*k*n+2*m*k
        extra_add=m*k*k+4*m*k**3+2*n*k*k+3*n*k**3+2*k*n+m*k
    else:
        extra_mul=2*m*k*k+3*n*k*k+3*k*n
        extra_add=2*m*k*k+m*k+2*n*k*k+k*n
    guard_mul=m*k*n;guard_add=m*n*(k-1)
    core_mul=core0[0]+core1[0]+extra_mul+guard_mul
    core_add=core0[1]+core1[1]+extra_add+guard_add
    params=k*n+n*p+(k*n if kind=='swiglu' else 0)
    nt=native_mul+native_add+2*params;ct=core_mul+core_add+2*params
    # Native lower reference storage: two activations (three for a gate).
    nr=(3 if kind=='swiglu' else 2)*m*n;cr=s*(m+n+p)
    return {'eligible':ct<nt and cr<=nr,'features':s,'native_arithmetic_lower_model_with_sgd':nt,
        'candidate_arithmetic_model_with_sgd':ct,'arithmetic_ratio':nt/ct,
        'native_logical_residual_values':nr,'candidate_logical_residual_values':cr,
        'scope':'All operand gradients, native gate projection guard, full tensor coefficients, feature VJPs and plain weight SGD. Native transcendental activation cost is assigned zero. Min/max domain reductions, launches, copies, synchronization and transient workspace are not scalar arithmetic counts; actual GPU timings include them.'}

def taylor_block(x,a,b,v=None,kind='gelu',delta=.05,evidence=None):
    values=[x,a,b]+([v] if v is not None else [])
    if kind not in ('gelu','silu','swiglu'):raise ValueError('Unsupported activation')
    if kind=='swiglu' and v is None:raise ValueError('SwiGLU needs an up-projection weight')
    if any(t.device.type!='mps' or t.dtype!=torch.float32 for t in values):
        raise ValueError('This research path requires MPS FP32; CPU ML execution is forbidden')
    if x.ndim!=2 or a.ndim!=2 or b.ndim!=2 or x.shape[1]!=a.shape[0] or a.shape[1]!=b.shape[0] or (v is not None and v.shape!=a.shape):
        raise ValueError('Expected compatible X[m,k], A[k,n], B[n,p], optional V[k,n]')
    if any(t.numel()==0 for t in values):raise ValueError('Positive dimensions required')
    if not isinstance(delta,(int,float)) or isinstance(delta,bool) or not math.isfinite(delta) or delta<=0 or delta>.05:
        raise ValueError('Explicit domain radius in (0,0.05] required')
    if torch.is_autocast_enabled('mps'):raise ValueError('MPS autocast is outside the verified FP32 contract')
    if a.shape[1]>1000000:raise ValueError('Research hidden-width bound exceeded')
    cost=taylor_cost(x.shape[0],x.shape[1],a.shape[1],b.shape[1],kind)
    path='cost_or_width_fallback'
    if cost['eligible']:
        with torch.no_grad():
            lo,hi=torch.aminmax(x@a)
            domain=torch.maximum(lo.abs(),hi.abs())
            magnitudes=[t.abs().amax() for t in values]
            valid=torch.stack([v<=1000 for v in magnitudes]+[domain<=delta,magnitudes[0]*magnitudes[1]*x.shape[1]<=1]).all()
        # NaN and infinity fail this comparison. Host synchronization cost is
        # real and included in timed candidate execution.
        if bool(valid.item()):
            if evidence is not None:evidence.append({'path':'approximate_feature','gate_max_abs':domain.item(),'cost':cost})
            return _TaylorFeature.apply(x,a,b,v,kind)
        path='domain_fallback'
    if evidence is not None:evidence.append({'path':path,'cost':cost})
    return native_block(x,a,b,v,kind)
