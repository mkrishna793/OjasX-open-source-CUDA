"""Dense nonlinear residuals and batched existing rank-seven arithmetic.

No sparsity, saturation, low-rank cotangent or activation approximation required.
Formal algebra does not assert floating-point equality or hardware speed.
"""
import torch
from .cotangent_quotient import native_mlp,activation,vjp,KINDS
from .bilinear import (_blocks,_join,_encode_a,_encode_b,_decode,
                       decoder_pullback,encoder_a_pullback,encoder_b_pullback,forward_cost)

def rank7_plan(m,k,n,cache=False,budget=256*1024*1024,grad_a=True,grad_b=True):
    if min(m,k,n)<1 or any(d%2 for d in (m,k,n)):
        return dict(eligible=False,reason='Positive even dimensions required')
    c=forward_cost(m,k,n)
    # Charge only demanded gradient products, plus the actual shared encoders
    # replayed by this implementation. Count the decoder's unary negation.
    grads=int(grad_a)+int(grad_b)
    c['dense_joint_scalar_ops']=2*m*k*n*(1+grads)-m*n-int(grad_a)*m*k-int(grad_b)*k*n
    c['candidate_joint_scalar_ops']=7*m*k*n*(1+grads)//4+m*k+k*n+(5*m*n//4 if grads else 0)+(m*k+k*n if grads and not cache else 0)
    c['requested_matrix_gradients']=[bool(grad_a),bool(grad_b)]
    c['workspace_estimate_bytes']=8*(m*k+k*n+m*n)*4
    c['cached_extra_estimate_bytes']=(7*(m*k+k*n)//4)*4 if cache else 0
    c['eligible']=(c['candidate_joint_scalar_ops']<c['dense_joint_scalar_ops'] and
                   c['workspace_estimate_bytes']<=budget)
    c['scope']='Requested-gradient scalar arithmetic and conservative logical scratch estimate; copies, indexing, allocation, traffic and hardware peak separate'
    return c

def raw_seven(a,b):
    ea=torch.stack(_encode_a(_blocks(a)));eb=torch.stack(_encode_b(_blocks(b)))
    return _join(_decode(torch.bmm(ea,eb).unbind(0))),ea,eb

def seven_reverse(ea,eb,g,need_a=True,need_b=True):
    channels=torch.stack(decoder_pullback(g))
    da=_join(encoder_a_pullback(torch.bmm(channels,eb.transpose(1,2)).unbind(0))) if need_a else None
    db=_join(encoder_b_pullback(torch.bmm(ea.transpose(1,2),channels).unbind(0))) if need_b else None
    return da,db

def _agrees(a,b,tol=2e-4):
    # A runtime numerical check, not a floating-point proof. Charge the native
    # reference products as extra work; this is qualification, not speed mode.
    if a is None or b is None:return a is b
    finite=torch.isfinite(a)&torch.isfinite(b)
    special=((torch.isnan(a)&torch.isnan(b))|(a==b)|finite).all()
    aa=torch.where(finite,a,0);bb=torch.where(finite,b,0)
    relative=torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb).clamp_min(1e-20)
    return bool(special) and bool(torch.allclose(aa,bb,atol=tol,rtol=tol)) and bool(relative<=tol)

class _Seven(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,cache,checked,evidence):
        out,ea,eb=raw_seven(a,b)
        ctx.cache=cache;ctx.checked=checked;ctx.evidence=evidence;ctx.native=False
        if checked:
            ref=a@b
            if not _agrees(out,ref):
                out=ref;ctx.native=True
                if evidence is not None:evidence.append(dict(route='native_primal_numerical_rejection'))
        ctx.save_for_backward(a,b,*((ea,eb) if cache and not ctx.native else ()))
        return out
    @staticmethod
    def backward(ctx,g):
        a,b,*enc=ctx.saved_tensors
        higher=torch.is_grad_enabled()
        if higher or not bool(torch.isfinite(g).all()):
            with torch.enable_grad():
                y=a@b;demand=tuple(t for t in (a,b) if t.requires_grad)
                gs=iter(torch.autograd.grad(y,demand,g,create_graph=higher))
            return (*tuple(next(gs) if t.requires_grad else None for t in (a,b)),None,None,None)
        if ctx.native:
            return (g@b.T if ctx.needs_input_grad[0] else None,
                    a.T@g if ctx.needs_input_grad[1] else None,None,None,None)
        if enc:ea,eb=enc
        else:ea=torch.stack(_encode_a(_blocks(a)));eb=torch.stack(_encode_b(_blocks(b)))
        da,db=seven_reverse(ea,eb,g,*ctx.needs_input_grad[:2])
        if ctx.checked:
            ra=g@b.T if ctx.needs_input_grad[0] else None
            rb=a.T@g if ctx.needs_input_grad[1] else None
            if not (_agrees(da,ra) and _agrees(db,rb)):
                da,db=ra,rb
                if ctx.evidence is not None:ctx.evidence.append(dict(route='native_gradient_numerical_rejection'))
        return da,db,None,None,None

def dense_matmul(a,b,*,accuracy='strict',cache=False,workspace_budget=256*1024*1024,evidence=None):
    if a.device.type!='mps' or b.device.type!='mps':raise ValueError('MPS only')
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('Shape mismatch')
    if accuracy not in ('strict','algebraic','checked'):raise ValueError('Invalid accuracy')
    plan=rank7_plan(a.shape[0],a.shape[1],b.shape[1],cache,workspace_budget,a.requires_grad,b.requires_grad)
    eligible=(accuracy in ('algebraic','checked') and plan['eligible'] and a.dtype==b.dtype==torch.float32
              and a is not b and not torch.is_autocast_enabled('mps'))
    if eligible:
        with torch.no_grad():
            maxima=torch.stack((a.abs().amax(),b.abs().amax()))
            eligible=bool(torch.isfinite(maxima).all() & (maxima.amax()<=4))
    if evidence is not None:evidence.append(dict(route='batched_rank7' if eligible else 'native',plan=plan))
    return _Seven.apply(a,b,cache,accuracy=='checked',evidence) if eligible else a@b

class _ResidualMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,u,ub,kind):
        z=x@w
        if b is not None:z=z+b
        if kind=='swiglu':
            up=x@u
            if ub is not None:up=up+ub
            h=activation(z,'silu')*up
            stored=(z,up)
        else:
            h=activation(z,kind)
            # For activations whose native VJP is determined by output, retain
            # that output directly. GELU/SiLU/square keep the preactivation.
            stored=(h,) if kind in ('relu','tanh','sigmoid','identity') else (z,)
        ctx.kind=kind;ctx.save_for_backward(x,w,b,q,u,ub,*stored)
        return h@q
    @staticmethod
    def backward(ctx,g):
        x,w,b,q,u,ub,*r=ctx.saved_tensors;kind=ctx.kind;needed=ctx.needs_input_grad[:6]
        higher=torch.is_grad_enabled()
        if higher:
            values=(x,w,b,q,u,ub)
            with torch.enable_grad():
                y=native_mlp(*values,kind)
                demand=tuple(t for t in values if t is not None and t.requires_grad)
                gs=iter(torch.autograd.grad(y,demand,g,create_graph=True))
            return (*tuple(next(gs) if t is not None and t.requires_grad else None for t in values),None)
        dx=dw=db=dq=du=dub=None
        if kind=='swiglu':
            z,up=r;s=activation(z,'silu');h=s*up
        elif kind in ('relu','tanh','sigmoid','identity'):
            h=r[0];z=h
        else:z=r[0];h=activation(z,kind)
        if needed[3]:dq=h.T@g
        if any(needed[i] for i in (0,1,2,4,5)):
            gh=g@q.T
            if kind=='swiglu':
                d=vjp(z,s,gh*up,'silu');dp=gh*s
                if needed[4]:du=x.T@dp
                if ub is not None and needed[5]:dub=dp.sum(0)
            else:d=vjp(z,h,gh,kind)
            if needed[0]:dx=d@w.T+(dp@u.T if kind=='swiglu' else 0)
            if needed[1]:dw=x.T@d
            if b is not None and needed[2]:db=d.sum(0)
        return dx,dw,db,dq,du,dub,None

def residual_mlp(x,w,b,q,up_weight=None,up_bias=None,kind='gelu',*,accuracy='strict',derivative_order=1,evidence=None):
    values=(x,w,b,q,up_weight,up_bias);vals=[t for t in values if t is not None]
    if any(t.device.type!='mps' for t in vals):raise ValueError('MPS only')
    if kind not in (*KINDS,'swiglu') or accuracy not in ('strict','algebraic'):raise ValueError('Invalid policy')
    if not(x.ndim==w.ndim==q.ndim==2 and x.shape[1]==w.shape[0] and w.shape[1]==q.shape[0]):raise ValueError('Shape mismatch')
    if b is not None and b.shape!=(w.shape[1],):raise ValueError('Bias mismatch')
    if kind=='swiglu' and (up_weight is None or up_weight.shape!=w.shape or up_bias is not None and up_bias.shape!=(w.shape[1],)):raise ValueError('Gate mismatch')
    if kind!='swiglu' and (up_weight is not None or up_bias is not None):raise ValueError('Unused gate')
    if derivative_order<1:raise ValueError('Derivative order')
    native=(accuracy=='strict' or derivative_order>1 or torch.is_autocast_enabled('mps')
            or min(*x.shape,*w.shape,*q.shape)==0 or any(t.dtype!=torch.float32 for t in vals)
            or len({id(t) for t in vals})!=len(vals))
    # Avoid replacing an already minimal native tape. GELU/SiLU and gates
    # admit dense savings when upstream and projection gradients are both used.
    native=native or kind in ('relu','tanh','sigmoid','identity') or not q.requires_grad or not any(
        t is not None and t.requires_grad for t in (x,w,b,up_weight,up_bias))
    if evidence is not None:evidence.append(dict(route='native' if native else 'nonlinear_sufficient_residual',
                                                  matrix_replay=False,dense_gradients_supported=True))
    return native_mlp(*values,kind) if native else _ResidualMLP.apply(*values,kind)
