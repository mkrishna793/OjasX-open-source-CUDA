"""Exact certified-chamber ReLU VJP; no approximate clustering.
The certificate is supplied and checked on the GPU. Discovery of chambers is
not free and is not implemented as an automatic clustering algorithm here.
"""
from dataclasses import asdict
import torch
from .gradient_factor_compiler import native_projection

class _Chamber(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,labels,masks,evidence):
        y=torch.relu(x@w+b);valid=bool(((y>0)==masks[labels]).all())
        ctx.valid=valid;ctx.evidence=evidence
        ctx.save_for_backward(x,w,b,q,labels,masks,y)
        if evidence is not None:evidence.append({'path':'exact_chamber' if valid else 'native_invalid_chamber','chambers':masks.shape[0]})
        return y@q
    @staticmethod
    def backward(ctx,p):
        x,w,b,q,labels,masks,y=ctx.saved_tensors
        higher=torch.is_grad_enabled()
        with torch.no_grad():safe=bool(torch.isfinite(p).all() & (p.abs().amax()<=8))
        if higher or not safe:
            inputs=(x,w,b,q);active=[t for t in inputs if t.requires_grad]
            with torch.enable_grad():h=native_projection(x,w,b,q,'relu');grads=torch.autograd.grad(h,active,p,create_graph=higher)
            it=iter(grads);g=tuple(next(it) if t.requires_grad else None for t in inputs)
        elif ctx.valid:
            s=(labels[:,None]==torch.arange(masks.shape[0],device='mps')[None,:]).to(x.dtype)
            a=(p[:,:,None]*s[:,None,:]).reshape(x.shape[0],-1)
            bt=(q[:,:,None]*masks.T[:,None,:]).reshape(q.shape[0],-1)
            g=(a@(w@bt).T,(x.T@a)@bt.T,a.sum(0)@bt.T,y.T@p)
        else:
            d=(p@q.T)*(y>0);g=(d@w.T,x.T@d,d.sum(0),y.T@p)
        return (*g,None,None,None)

def chamber_projection(x,w,b,q,labels,masks,accuracy='strict',evidence=None):
    values=(x,w,b,q,labels,masks)
    if any(t.device.type!='mps' for t in values):raise ValueError('MPS GPU only')
    if not (x.ndim==w.ndim==q.ndim==masks.ndim==2 and b.ndim==1 and labels.shape==(x.shape[0],) and x.shape[1]==w.shape[0] and w.shape[1]==q.shape[0]==b.shape[0]==masks.shape[1]):raise ValueError('Incompatible chamber shapes')
    if masks.dtype!=torch.bool or labels.dtype!=torch.long:raise ValueError('Boolean masks and int64 labels required')
    if accuracy not in ('strict','algebraic'):raise ValueError('Unknown accuracy policy')
    m,k=x.shape;n,r=q.shape;c=masks.shape[0];width=r*c
    candidate=4*k*width*(m+n)+(m+n)*width+m*n
    native=4*m*n*k+2*m*n*r+2*m*n
    if accuracy=='strict' or min(m,k,n,r,c)==0 or candidate>=native or any(t.dtype!=torch.float32 for t in values[:4]) or torch.is_autocast_enabled('mps'):
        if evidence is not None:evidence.append({'path':'native_policy_or_cost'})
        return native_projection(x,w,b,q,'relu')
    with torch.no_grad():
        ok=(labels>=0).all() & (labels<c).all()
        for t in values[:4]:ok=ok & torch.isfinite(t).all() & (t.abs().amax()<=8)
    if not bool(ok):
        if evidence is not None:evidence.append({'path':'native_invalid_domain'})
        return native_projection(x,w,b,q,'relu')
    return _Chamber.apply(x,w,b,q,labels,masks,evidence)
