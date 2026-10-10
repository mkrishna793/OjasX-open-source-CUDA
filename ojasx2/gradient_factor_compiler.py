"""Combined activation/backprop experiment. Exact-real factors, no truncation.

The public projection block retains every original parameter. Only MPS FP32
first-order algebraic execution is optimized. Higher-order AD recomputes the
native differentiable block. This is an explicit API, not a graph compiler.
"""
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F

KINDS = ('identity', 'relu', 'square', 'gelu', 'silu', 'tanh', 'sigmoid')

def activation(z, kind):
    if kind == 'identity': return z
    if kind == 'relu': return F.relu(z)
    if kind == 'square': return z.square()
    if kind == 'gelu': return F.gelu(z, approximate='none')
    if kind == 'silu': return F.silu(z)
    if kind == 'tanh': return z.tanh()
    if kind == 'sigmoid': return z.sigmoid()
    raise ValueError('Unsupported activation: '+str(kind))

def native_projection(x, w, b, q, kind):
    return activation(x @ w + b, kind) @ q

@dataclass(frozen=True)
class FactorPlan:
    shape: tuple
    activation: str
    factor_width: int
    native_backward_arithmetic: int
    candidate_backward_arithmetic: int
    eligible: bool
    condition: str
    scope: str = 'Leading multiply/add model for dx,dw,db; common forward,dq and SGD excluded. No latency guarantee; comparisons/copies/allocator work not FLOPs.'

def factor_plan(m, k, n, r, kind):
    if min(m,k,n,r) <= 0: raise ValueError('Positive dimensions required')
    width = r*(k+1) if kind == 'square' else r
    # Native G=P Q^T, activation multiply, dx, dw, db; dot products
    # use conventional 2abc leading counts, not exact instruction counts.
    native = 2*m*n*r + 4*m*n*k + 2*m*n
    candidate = 4*k*width*(m+n) + (m+n)*width
    if kind == 'square': candidate += (m+n)*width + 2*k*n
    if kind == 'relu': candidate += n*r + m*n  # numeric mask/guard work
    eligible = kind in ('identity','square','relu') and candidate < native
    condition = ('Known upstream factors; no activation restriction' if kind=='identity' else
                 'Exact polynomial slope factorization; no degree truncation' if kind=='square' else
                 'Each feature has the same ReLU mask throughout this batch' if kind=='relu' else
                 'No proved cheap exact slope factorization in this grammar; native VJP')
    return FactorPlan((m,k,n,r),kind,width,native,candidate,eligible,condition)

def _slope(z, y, kind):
    if kind=='relu': return (y>0).to(y.dtype)
    if kind=='square': return 2*z
    if kind=='tanh': return 1-y.square()
    if kind=='sigmoid': return y*(1-y)
    if kind=='silu':
        s=z.sigmoid(); return s*(1+z*(1-s))
    if kind=='gelu':
        import math
        return .5*(1+torch.erf(z/math.sqrt(2)))+z*torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    return torch.ones_like(y)

def factor_vjp(x,w,b,q,p,kind,z,y,plan,evidence=None):
    """Derivative factors are derived from the primal, never guessed/truncated."""
    use = plan.eligible
    mask = None
    if use and kind=='relu':
        mask=y>0
        use=bool((mask==mask[:1]).all())
    if use:
        if kind=='identity': a,bt=p,q
        elif kind=='relu': a,bt=p,q*mask[0,:,None]
        else:
            s=torch.cat((x,torch.ones_like(x[:,:1])),dim=1)
            t=torch.cat((2*w.T,2*b[:,None]),dim=1)
            a=(p[:,:,None]*s[:,None,:]).reshape(x.shape[0],-1)
            bt=(q[:,:,None]*t[:,None,:]).reshape(q.shape[0],-1)
        dx=a@(w@bt).T
        dw=(x.T@a)@bt.T
        db=a.sum(0)@bt.T
        path='exact_factored_'+kind
    else:
        g=p@q.T
        if kind!='identity': g=g*_slope(z,y,kind)
        dx=g@w.T;dw=x.T@g;db=g.sum(0)
        path='native_dense_vjp'
    if evidence is not None: evidence.append({'path':path,'plan':asdict(plan)})
    return dx,dw,db

class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,kind,evidence):
        ctx.kind=kind;ctx.evidence=evidence
        ctx.plan=factor_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],kind)
        # Square slopes factor directly through x,w,b. Linear needs no wide
        # saved activation. ReLU/tanh/sigmoid use their output statistic.
        if kind=='identity':
            wq=w@q;h=x@wq+b@q;saved=(wq,)
        else:
            z=x@w+b;y=activation(z,kind);h=y@q
            if kind in ('square','relu','tanh','sigmoid'): saved=(y,)
            else: saved=(z,y)
        ctx.save_for_backward(x,w,b,q,*saved)
        return h

    @staticmethod
    def backward(ctx,p):
        x,w,b,q,*saved=ctx.saved_tensors;kind=ctx.kind
        higher_order=torch.is_grad_enabled()
        # A nonfinite or extreme cotangent can expose IEEE differences such as
        # 0*inf when products are reassociated. Preserve the native VJP there.
        with torch.no_grad():
            numeric_safe=bool(torch.isfinite(p).all() & (p.abs().amax()<=8))
        if higher_order or not numeric_safe:
            # A detached cached residual is insufficient for correct HVPs.
            inputs=(x,w,b,q);active=[t for t in inputs if t.requires_grad]
            with torch.enable_grad():
                h=native_projection(x,w,b,q,kind)
                grads=torch.autograd.grad(h,active,p,create_graph=higher_order)
            it=iter(grads)
            result=tuple(next(it) if t.requires_grad else None for t in inputs)
            if ctx.evidence is not None:ctx.evidence.append({'path':'native_higher_order_recompute' if higher_order else 'native_cotangent_domain'})
            return (*result,None,None)
        z=None
        if kind=='identity':
            # Share both small contractions across every original parameter.
            t=x.T@p;ps=p.sum(0)
            dx=p@saved[0].T;dw=t@q.T;db=ps@q.T
            dq=w.T@t+b[:,None]*ps[None,:]
            if ctx.evidence is not None:ctx.evidence.append({'path':'exact_factored_identity','plan':asdict(ctx.plan)})
            return dx,dw,db,dq,None,None
        elif kind in ('square','relu','tanh','sigmoid'): y=saved[0]
        else:z,y=saved
        plan=ctx.plan
        # Reconstruct z only on a square cost fallback; no division by y.
        if kind=='square' and not plan.eligible:z=x@w+b
        dx,dw,db=factor_vjp(x,w,b,q,p,kind,z,y,plan,ctx.evidence)
        dq=y.T@p
        return dx,dw,db,dq,None,None

def projection_block(x,w,b,q,kind='gelu',accuracy='strict',evidence=None):
    if kind not in KINDS:raise ValueError('Unsupported activation: '+str(kind))
    if accuracy not in ('strict','algebraic'):raise ValueError('Unknown accuracy policy')
    if not (x.ndim==w.ndim==q.ndim==2 and b.ndim==1 and x.shape[1]==w.shape[0]
            and w.shape[1]==b.shape[0]==q.shape[0]):raise ValueError('Incompatible shapes')
    if any(t.device.type!='mps' for t in (x,w,b,q)):raise ValueError('GPU-only MPS experiment')
    if accuracy=='strict' or any(t.dtype!=torch.float32 for t in (x,w,b,q)) or torch.is_autocast_enabled('mps') or min(*x.shape,*w.shape,*q.shape)==0:
        if evidence is not None:evidence.append({'path':'native_policy'})
        return native_projection(x,w,b,q,kind)
    plan=factor_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],kind)
    if not plan.eligible:
        if evidence is not None:evidence.append({'path':'native_no_profitable_certificate','plan':asdict(plan)})
        return native_projection(x,w,b,q,kind)
    # Nonfinite inputs retain native semantics, including NaN propagation.
    with torch.no_grad():
        finite=torch.isfinite(x).all() & torch.isfinite(w).all() & torch.isfinite(b).all() & torch.isfinite(q).all()
    if not bool(finite):
        if evidence is not None:evidence.append({'path':'native_nonfinite'})
        return native_projection(x,w,b,q,kind)
    with torch.no_grad():
        bounded=(x.abs().amax()<=8) & (w.abs().amax()<=8) & (b.abs().amax()<=8) & (q.abs().amax()<=8)
    if not bool(bounded):
        if evidence is not None:evidence.append({'path':'native_numeric_domain'})
        return native_projection(x,w,b,q,kind)
    return _Projection.apply(x,w,b,q,kind,evidence)
