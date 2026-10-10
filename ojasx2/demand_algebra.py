"""Two-sided demand restriction for private deterministic nonlinear blocks.

Exact zero cotangents only; original full forward and all parameter gradients.
This MPS reference is numerically qualified, not extracted verified IEEE code.
The construction uses existing support algebra; no historical novelty claim.
"""
import torch
from .cotangent_quotient import native_mlp,activation,KINDS


def demand_plan(m,k,n,r,a,c,gated=False):
    if min(m,k,n,r)<1 or not 0<=a<=m or not 0<=c<=r:raise ValueError('Invalid shape/support')
    projections=2 if gated else 1
    forward=2*m*n*(projections*k+r)
    native_backward=4*m*n*(projections*k+r)
    replay=6*a*n*projections*k+4*a*n*c
    scan=2*m*r
    return dict(native_joint=forward+native_backward,candidate_joint=forward+replay+scan,
                native_backward=native_backward,replay_backward=replay,scan_estimate=scan,
                eligible=replay+scan<native_backward,active_rows=a,active_columns=c,
                scope='Leading GEMM arithmetic plus cotangent scan estimate; pointwise, gather/scatter, allocation and synchronization measured separately')


class _Demand(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,u,ub,kind,evidence):
        ctx.save_for_backward(x,w,b,q,u,ub);ctx.kind=kind;ctx.evidence=evidence
        return native_mlp(x,w,b,q,u,ub,kind)

    @staticmethod
    def backward(ctx,g):
        values=ctx.saved_tensors;x,w,b,q,u,ub=values
        higher=torch.is_grad_enabled();needed=ctx.needs_input_grad[:6]
        def full(reason):
            with torch.enable_grad():
                y=native_mlp(*values,ctx.kind)
                demand=tuple(t for t in values if t is not None and t.requires_grad)
                grads=iter(torch.autograd.grad(y,demand,g,create_graph=higher))
            if ctx.evidence is not None:ctx.evidence.append(dict(route='native_recompute',reason=reason))
            return (*tuple(next(grads) if t is not None and t.requires_grad else None for t in values),None,None)
        if higher:return full('Higher-order dependency must not be erased by value-based support')
        if not bool(torch.isfinite(g).all()):return full('Nonfinite upstream')
        rows=(g!=0).any(1).nonzero().flatten()
        cols=(g!=0).any(0).nonzero().flatten()
        plan=demand_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],rows.numel(),cols.numel(),ctx.kind=='swiglu')
        if not plan['eligible']:return full('Dense demand does not pay replay cost')
        # Only scattered outputs require zero initialization. Dense parameter
        # gradients are produced directly, avoiding a redundant full write.
        empty=not rows.numel() or not cols.numel()
        result=[torch.zeros_like(t) if t is not None and needed[i] and (empty or i in (0,3))
                else None for i,t in enumerate(values)]
        if rows.numel() and cols.numel():
            with torch.enable_grad():
                local=[]
                for i,t in enumerate(values):
                    if t is None:local.append(None);continue
                    v=t.index_select(0,rows) if i==0 else t.index_select(1,cols) if i==3 else t
                    local.append(v.detach().requires_grad_(needed[i]))
                lx,lw,lb,lq,lu,lub=local
                z=lx@lw
                if lb is not None:z=z+lb
                if ctx.kind=='swiglu':
                    up=lx@lu
                    if lub is not None:up=up+lub
                    h=activation(z,'silu')*up
                else:h=activation(z,ctx.kind)
                active=g.index_select(0,rows).index_select(1,cols)
                demand=tuple(t for i,t in enumerate(local) if i!=3 and t is not None and t.requires_grad)
                grads=iter(torch.autograd.grad(h,demand,active@lq.T) if demand else ())
            for i,t in enumerate(local):
                if t is None or not needed[i]:continue
                grad=h.detach().T@active if i==3 else next(grads)
                if i==0:result[i].index_copy_(0,rows,grad)
                elif i==3:result[i].index_copy_(1,cols,grad)
                else:result[i]=grad
        if ctx.evidence is not None:ctx.evidence.append(dict(route='two_sided_demand',plan=plan,
            exact_zero_support=True,all_original_gradients=True))
        return (*result,None,None)


def demand_mlp(x,w,b,q,up_weight=None,up_bias=None,kind='gelu',*,accuracy='strict',
               derivative_order=1,evidence=None):
    vals=tuple(t for t in (x,w,b,q,up_weight,up_bias) if t is not None)
    if any(t.device.type!='mps' for t in vals):raise ValueError('MPS only; CPU fallback prohibited')
    if kind not in (*KINDS,'swiglu') or accuracy not in ('strict','algebraic'):raise ValueError('Invalid policy')
    if not(x.ndim==w.ndim==q.ndim==2 and x.shape[1]==w.shape[0] and w.shape[1]==q.shape[0]):raise ValueError('Shape mismatch')
    if b is not None and b.shape!=(w.shape[1],):raise ValueError('Bias mismatch')
    if kind=='swiglu' and (up_weight is None or up_weight.shape!=w.shape or up_bias is not None and up_bias.shape!=(w.shape[1],)):raise ValueError('Gate mismatch')
    if kind!='swiglu' and (up_weight is not None or up_bias is not None):raise ValueError('Unused gate operands')
    if derivative_order<1:raise ValueError('Invalid derivative order')
    def native(reason):
        if evidence is not None:evidence.append(dict(route='native',reason=reason))
        return native_mlp(x,w,b,q,up_weight,up_bias,kind)
    if accuracy=='strict' or derivative_order>1 or torch.is_autocast_enabled('mps'):return native('Strict/higher/autocast')
    if min(*x.shape,*w.shape,*q.shape)==0 or any(t.dtype!=torch.float32 for t in vals):return native('Shape/dtype')
    if len({id(t) for t in vals})!=len(vals):return native('Repeated operand')
    # Finite inputs alone do not rule out overflow. Bound the dot-product and
    # gate-product inputs before applying zero-cotangent elision.
    with torch.no_grad():
        maxima=torch.stack([t.detach().abs().amax() for t in vals]);a=maxima.amax()
        safe=bool(torch.isfinite(maxima).all() & (a<=8))
    if not safe or max(x.shape[1],w.shape[1])>65536:return native('Overflow envelope')
    return _Demand.apply(x,w,b,q,up_weight,up_bias,kind,evidence)
