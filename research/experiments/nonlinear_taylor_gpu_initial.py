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
    key=_key(x,kind);t=_lift(x,TERMS[key]);h=BUILDERS[key](a,v,_alpha(kind));c=h@b
    return t,h,c

def surrogate_block(x,a,b,v=None,kind='gelu'):
    """Unconditional polynomial reference for qualification, not dispatch."""
    t,h,c=feature_values(x,a,b,v,kind)
    return t@c

class _TaylorFeature(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,b,v,kind):
        t,h,c=feature_values(x,a,b,v,kind)
        ctx.kind=kind;ctx.save_for_backward(x,a,b,v,t,h,c)
        return t@c
    @staticmethod
    def backward(ctx,g):
        x,a,b,v,t,h,c=ctx.saved_tensors;kind=ctx.kind;key=_key(x,kind)
        if torch.is_grad_enabled():t,h,c=feature_values(x,a,b,v,kind)
        dc=t.T@g;dt=g@c.T;dh=dc@b.T;db=h.T@dc
        da,dv=PULLS[key](a,v,_alpha(kind),dh)
        dx=_lift_pull(x,dt,TERMS[key])
        return dx,da,db,dv,None

def _mm(m,k,n):return 3*m*k*n,m*n*(k-1)+m*k*(n-1)+k*n*(m-1)

def taylor_cost(m,k,n,p,kind):
    key=('swiglu' if kind=='swiglu' else 'activation',k)
    if key not in TERMS:return {'eligible':False,'reason':'unsupported_feature_width'}
    terms=TERMS[key];s=len(terms);counts=COUNTS[key]
    native0=_mm(m,k,n);native1=_mm(m,n,p)
    native_mul=(2 if kind=='swiglu' else 1)*native0[0]+native1[0]
    native_add=(2 if kind=='swiglu' else 1)*native0[1]+native1[1]+(m*k if kind=='swiglu' else 0)
    core0=_mm(s,n,p);core1=_mm(m,s,p)
    lift_mul=m*sum((len(t)-1)+len(t)*max(0,len(t)-1) for t in terms)
    lift_add=m*sum(len(t) for t in terms)
    # One native gate projection is evaluated to enforce the approximation
    # domain. It is not omitted from the arithmetic or measured runtime.
    guard_mul=m*k*n;guard_add=m*n*(k-1)
    core_mul=core0[0]+core1[0]+n*(counts['coefficient_mul']+counts['pull_mul'])+lift_mul+guard_mul
    core_add=core0[1]+core1[1]+n*(counts['coefficient_add']+counts['pull_add'])+lift_add+guard_add
    params=k*n+n*p+(k*n if kind=='swiglu' else 0)
    nt=native_mul+native_add+2*params;ct=core_mul+core_add+2*params
    # Native lower reference storage: two activations (three for a gate).
    nr=(3 if kind=='swiglu' else 2)*m*n;cr=s*(m+n+p)
    return {'eligible':ct<nt and cr<=nr,'features':s,'native_arithmetic_lower_model_with_sgd':nt,
        'candidate_arithmetic_model_with_sgd':ct,'arithmetic_ratio':nt/ct,
        'native_logical_residual_values':nr,'candidate_logical_residual_values':cr,
        'scope':'All operand gradients, native gate projection guard, generated coefficients, feature VJPs and plain weight SGD. Native transcendental activation cost is assigned zero. Abs/max domain reductions, launches, copies, synchronization and transient workspace are not scalar arithmetic counts; actual GPU timings include them.'}

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
    cost=taylor_cost(x.shape[0],x.shape[1],a.shape[1],b.shape[1],kind)
    path='cost_or_width_fallback'
    if cost['eligible']:
        with torch.no_grad():
            domain=(x@a).abs().amax()
            valid=torch.stack([torch.isfinite(t).all() for t in values]+[domain<=delta]).all()
        # NaN and infinity fail this comparison. Host synchronization cost is
        # real and included in timed candidate execution.
        if bool(valid.item()):
            if evidence is not None:evidence.append({'path':'approximate_feature','gate_max_abs':domain.item(),'cost':cost})
            return _TaylorFeature.apply(x,a,b,v,kind)
        path='domain_fallback'
    if evidence is not None:evidence.append({'path':path,'cost':cost})
    return native_block(x,a,b,v,kind)
