"""Observable-gradient contraction with exact-support nonlinear defects.

Primal evaluation is native. A compact first-order tape is admitted only when
both native values and unit-cotangent slopes match a per-column linear part
outside a small explicit support. No nonzero defect is threshold-pruned.
Real calculus is proved separately; rounded GELU/SiLU saturation is a numerical
certificate, not a theorem that their real derivatives are piecewise constant.
"""
import torch
import torch.nn.functional as F

KINDS=('relu','gelu','silu','tanh','sigmoid','square','identity')


def activation(z,kind):
    if kind=='relu':return F.relu(z)
    if kind=='gelu':return F.gelu(z)
    if kind=='silu':return F.silu(z)
    if kind=='tanh':return torch.tanh(z)
    if kind=='sigmoid':return torch.sigmoid(z)
    if kind=='square':return z.square()
    if kind=='identity':return z
    raise ValueError('Unknown activation')


def vjp(z,y,g,kind):
    if kind=='relu':return torch.ops.aten.threshold_backward(g,z,0)
    if kind=='gelu':return torch.ops.aten.gelu_backward(g,z,approximate='none')
    if kind=='silu':return torch.ops.aten.silu_backward(g,z)
    if kind=='tanh':return torch.ops.aten.tanh_backward(g,y)
    if kind=='sigmoid':return torch.ops.aten.sigmoid_backward(g,y)
    if kind=='square':return g*(2*z)
    if kind=='identity':return g
    raise ValueError('Unknown activation')


def native_projection(x,w,b,q,kind):
    z=x@w
    if b is not None:z=z+b
    return activation(z,kind)@q


def defect_plan(m,k,n,r,entries,element_bytes=4):
    if min(m,k,n,r)<1 or entries<0 or entries>m*n:raise ValueError('Invalid shape/support')
    common=2*m*n*(k+r)
    dense=4*m*n*(k+r)
    compact=4*m*k*r+6*k*n*r+4*entries*(k+r)+4*n*r+2*m*r
    # A deliberately visible arithmetic estimate for certificate construction.
    # Transcendental evaluation, indexing and synchronization are not modeled as
    # cheap FLOPs and must be timed separately; no operation-optimality claim.
    certificate=8*m*n
    dense_tape=2*m*n*element_bytes
    compact_tape=entries*(16+2*element_bytes)+2*n*element_bytes
    return dict(native_joint_flops=common+dense,candidate_joint_flops=common+compact+certificate,
                common_forward_flops=common,native_backward_flops=dense,
                compact_backward_flops=compact,certificate_scalar_estimate=certificate,
                dense_saved_estimate=dense_tape,compact_saved_estimate=compact_tape,
                entries=entries,eligible=(compact+certificate<dense and compact_tape<dense_tape),
                scope='Leading joint matrix/support arithmetic plus scalar certificate estimate; transcendental latency, indexing, reductions and memory traffic measured separately. All X,W,Q gradients demanded.')


def cotangent_representative(d,left,right):
    """Mathematical probe; forming this dense representative itself is not cheap."""
    return left@d+d@right-(left@d)@right


class _DefectProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,kind,evidence,max_support_fraction,chunk_entries):
        z=x@w
        if b is not None:z=z+b
        y=activation(z,kind);output=y@q
        ctx.kind=kind;ctx.evidence=evidence;ctx.chunk_entries=chunk_entries
        # The first row supplies a data-dependent candidate slope. It is held
        # as a local coefficient; the exact slope defects restore all gradients.
        slope=vjp(z,y,torch.ones_like(z),kind)
        c=slope[0].clone();offset=y[0]-z[0]*c
        values=y-(z*c+offset)
        support=(values!=0)|(slope!=c)
        entries=int(support.sum().item())
        plan=defect_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],entries)
        finite=bool(torch.isfinite(output).all() & torch.isfinite(slope).all() & torch.isfinite(values).all())
        cols=support.any(0).nonzero().flatten()
        column_plan=defect_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],x.shape[0]*cols.numel())
        column_plan['compact_saved_estimate']=2*x.shape[0]*cols.numel()*4+8*cols.numel()+8*w.shape[1]
        use_columns=finite and column_plan['eligible'] and cols.numel()<=max_support_fraction*w.shape[1]
        ctx.mode='columns' if use_columns else 'sparse' if finite and plan['eligible'] and entries<=max_support_fraction*z.numel() else 'dense'
        if ctx.mode=='columns':
            c=c.clone();offset=offset.clone();c[cols]=0;offset[cols]=0
            ctx.save_for_backward(x,w,b,q,c,offset,cols,z.index_select(1,cols),y.index_select(1,cols))
            plan=column_plan;route='compact_nonlinear_columns'
        elif ctx.mode=='sparse':
            ids=support.nonzero();rows,cols=ids[:,0].contiguous(),ids[:,1].contiguous()
            zs=z[rows,cols];ef=values[rows,cols]
            ctx.save_for_backward(x,w,b,q,c,offset,rows,cols,zs,ef)
            route='compact_defect'
        else:
            ctx.save_for_backward(x,w,b,q,z,y)
            route='dense_backward_after_certificate'
        if evidence is not None:evidence.append(dict(route=route,activation=kind,plan=plan,
            support_fraction=entries/z.numel(),certificate='Native forward values AND native unit-cotangent slopes; no tolerance pruning'))
        return output

    @staticmethod
    def backward(ctx,p):
        saved=ctx.saved_tensors;x,w,b,q=saved[:4];kind=ctx.kind
        higher=torch.is_grad_enabled()
        # A safe upstream guard is necessary for rounded saturation: a very
        # large cotangent can expose terms that rounded away for unit slopes.
        safe=bool(torch.isfinite(p).all() & (p.detach().abs().amax()<=1)) if p.numel() else True
        if higher or not safe:
            originals=(x,w,b,q);demand=tuple(t for t in originals if t is not None and t.requires_grad)
            with torch.enable_grad():
                output=native_projection(x,w,b,q,kind)
                grads=iter(torch.autograd.grad(output,demand,p,create_graph=higher))
            out=tuple(next(grads) if t is not None and t.requires_grad else None for t in originals)
            if ctx.evidence is not None:ctx.evidence.append(dict(route='native_recompute_higher_or_cotangent',higher=higher))
            return (*out,None,None,None,None)
        if ctx.mode=='dense':
            z,y=saved[4:];d=vjp(z,y,p@q.T,kind)
            return d@w.T,x.T@d,d.sum(0) if b is not None else None,y.T@p,None,None,None,None
        c,offset=saved[4:6]
        qc=c[:,None]*q;t=x.T@p;s=p.sum(0)
        dx=p@(w@qc).T
        dw=(t@q.T)*c
        db=(q@s)*c if b is not None else None
        dq=(w.T@t)*c[:,None]
        affine=offset if b is None else b*c+offset
        dq=dq+affine[:,None]*s[None,:]
        if ctx.mode=='columns':
            cols,zs,ys=saved[6:]
            if cols.numel():
                d=vjp(zs,ys,p@q.index_select(0,cols).T,kind)
                dx.add_(d@w.index_select(1,cols).T)
                dw.index_copy_(1,cols,x.T@d)
                if db is not None:db.index_copy_(0,cols,d.sum(0))
                dq.index_copy_(0,cols,ys.T@p)
            return dx,dw,db,dq,None,None,None,None
        rows,cols,zs,ef=saved[6:]
        # Exact support, bounded scratch. Scatter accumulation rounding can
        # differ from dense GEMMs; the GPU qualification checks every gradient.
        for offset in range(0,rows.numel(),ctx.chunk_entries):
            ri=rows[offset:offset+ctx.chunk_entries];ci=cols[offset:offset+ctx.chunk_entries]
            zi=zs[offset:offset+ctx.chunk_entries];fi=ef[offset:offset+ctx.chunk_entries]
            pi=p.index_select(0,ri);qi=q.index_select(0,ci)
            g=(pi*qi).sum(-1)
            dg=vjp(zi,activation(zi,kind),g,kind)-c.index_select(0,ci)*g
            dx.index_add_(0,ri,dg[:,None]*w.T.index_select(0,ci))
            dw.T.index_add_(0,ci,dg[:,None]*x.index_select(0,ri))
            if db is not None:db.index_add_(0,ci,dg)
            dq.index_add_(0,ci,fi[:,None]*pi)
        return dx,dw,db,dq,None,None,None,None


def defect_projection(x,w,b,q,kind='gelu',*,accuracy='strict',derivative_order=1,
                      max_support_fraction=.03,chunk_entries=1024,evidence=None):
    """Private XW+b -> activation -> Q block; original parameters retained.

    Exact-real defect identities are general. Cheap sparse support is conditional.
    This first implementation is MPS FP32, all X/W/Q gradients, optional bias.
    A rejected certificate still pays a scan; no zero-overhead fallback promise.
    """
    vals=tuple(t for t in (x,w,b,q) if t is not None)
    if any(t.device.type!='mps' for t in vals):raise ValueError('MPS required; CPU fallback prohibited')
    if kind not in KINDS or accuracy not in ('strict','algebraic'):raise ValueError('Invalid activation/policy')
    if not (x.ndim==w.ndim==q.ndim==2 and x.shape[1]==w.shape[0] and w.shape[1]==q.shape[0]):
        raise ValueError('Expected X[M,K], W[K,N], Q[N,R]')
    if b is not None and (b.ndim!=1 or b.shape[0]!=w.shape[1]):raise ValueError('Invalid bias')
    if not 0<=max_support_fraction<=1 or chunk_entries<1 or derivative_order<1:raise ValueError('Invalid execution contract')
    def native(reason):
        if evidence is not None:evidence.append(dict(route='native',reason=reason))
        return native_projection(x,w,b,q,kind)
    if accuracy=='strict' or derivative_order>1 or torch.is_autocast_enabled('mps'):
        return native('Strict, higher derivative or autocast')
    if len({id(t) for t in vals})!=len(vals):return native('Repeated operand object')
    if min(*x.shape,*w.shape,*q.shape)==0 or any(t.dtype!=torch.float32 for t in vals):
        return native('Empty or dtype')
    if not all(t.requires_grad for t in (x,w,q)) or b is not None and not b.requires_grad:
        return native('Frozen operand: conservative demand policy')
    plan=defect_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],0)
    if not plan['eligible']:return native('Even zero-defect plan is not cheaper')
    with torch.no_grad():
        bounded=all(bool(torch.isfinite(t).all() & (t.abs().amax()<=32)) for t in vals)
    if not bounded:return native('Numerical domain')
    return _DefectProjection.apply(x,w,b,q,kind,evidence,max_support_fraction,chunk_entries)


def native_mlp(x,w,b,q,up_weight=None,up_bias=None,kind='gelu'):
    z=x@w
    if b is not None:z=z+b
    if kind=='swiglu':
        up=x@up_weight
        if up_bias is not None:up=up+up_bias
        y=F.silu(z)*up
    else:y=activation(z,kind)
    return y@q


class _StreamedMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,up_weight,up_bias,kind,chunk):
        ctx.save_for_backward(x,w,b,q,up_weight,up_bias);ctx.kind=kind;ctx.chunk=chunk
        return native_mlp(x,w,b,q,up_weight,up_bias,kind)
    @staticmethod
    def backward(ctx,p):
        values=ctx.saved_tensors;needs=ctx.needs_input_grad[:6]
        if torch.is_grad_enabled():
            demand=tuple(t for t in values if t is not None and t.requires_grad)
            with torch.enable_grad():
                y=native_mlp(*values,ctx.kind)
                gs=iter(torch.autograd.grad(y,demand,p,create_graph=True))
            out=tuple(next(gs) if t is not None and t.requires_grad else None for t in values)
            return (*out,None,None)
        result=[torch.zeros_like(t) if t is not None and needs[i] else None for i,t in enumerate(values)]
        for start in range(0,values[0].shape[0],ctx.chunk):
            end=min(start+ctx.chunk,values[0].shape[0])
            with torch.enable_grad():
                local=tuple((t[start:end] if i==0 else t).detach().requires_grad_(needs[i]) if t is not None else None
                            for i,t in enumerate(values))
                demand=tuple(t for t in local if t is not None and t.requires_grad)
                y=native_mlp(*local,ctx.kind)
                gs=iter(torch.autograd.grad(y,demand,p[start:end]))
            for i,t in enumerate(local):
                if t is not None and needs[i]:
                    g=next(gs)
                    if i==0:result[i][start:end].copy_(g)
                    else:result[i].add_(g)
        return (*result,None,None)


def streamed_mlp(x,w,b,q,up_weight=None,up_bias=None,kind='gelu',*,accuracy='strict',
                 derivative_order=1,chunk_rows=32,evidence=None):
    """Keep only block operands; stream full nonlinear backward contractions.

    Applies to all supported deterministic activation values, including dense
    GELU/SiLU/SwiGLU. It spends recomputation to reduce saved activation memory;
    it does not claim fewer arithmetic operations. No attention rows are erased.
    """
    vals=tuple(t for t in (x,w,b,q,up_weight,up_bias) if t is not None)
    if any(t.device.type!='mps' for t in vals):raise ValueError('MPS required')
    if kind not in (*KINDS,'swiglu') or accuracy not in ('strict','algebraic'):raise ValueError('Invalid policy')
    if chunk_rows<1 or derivative_order<1:raise ValueError('Invalid chunk or derivative order')
    if not(x.ndim==w.ndim==q.ndim==2 and x.shape[1]==w.shape[0] and w.shape[1]==q.shape[0]):raise ValueError('Shape mismatch')
    if b is not None and b.shape!=(w.shape[1],):raise ValueError('Bias mismatch')
    if kind=='swiglu' and (up_weight is None or up_weight.shape!=w.shape or up_bias is not None and up_bias.shape!=(w.shape[1],)):
        raise ValueError('SwiGLU up projection mismatch')
    if kind!='swiglu' and (up_weight is not None or up_bias is not None):raise ValueError('Extra operands for ungated MLP')
    native=(accuracy=='strict' or derivative_order>1 or torch.is_autocast_enabled('mps')
            or min(*x.shape,*w.shape,*q.shape)==0 or any(t.dtype!=torch.float32 for t in vals)
            or len({id(t) for t in vals})!=len(vals))
    if evidence is not None:evidence.append(dict(route='native' if native else 'streamed_nonlinear_moments',
        activation=kind,chunk_rows=chunk_rows,recompute=True,arithmetic_reduction=False))
    if native:return native_mlp(x,w,b,q,up_weight,up_bias,kind)
    return _StreamedMoments.apply(x,w,b,q,up_weight,up_bias,kind,chunk_rows)
