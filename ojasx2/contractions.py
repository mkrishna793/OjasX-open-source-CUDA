"""Explicit algebraic contraction schedules; strict default preserves native ops.

The two-layer rule never crosses an activation or approximates/truncates weights.
Qualification covers CPU FP32/FP64 and sampled MPS FP32 2D inputs.
"""
import torch
import torch.nn.functional as F
from .devices import algebraic_device_supported


def _mm_ops(m,k,n):return 2*m*k*n-m*n


def chain_cost(m,k,n,p,needs=(True,True,True),mode='joint'):
    if min(m,k,n,p)<=0 or mode not in ('joint','backward_only'):
        raise ValueError('Positive dimensions and a declared mode are required')
    dx,dw1,dw2=needs
    native=_mm_ops(m,k,n)+_mm_ops(m,n,p)
    if dx or dw1:native+=_mm_ops(m,p,n)
    if dx:native+=_mm_ops(m,n,k)
    if dw1:native+=_mm_ops(n,m,k)
    if dw2:native+=_mm_ops(p,m,n)
    # T=g.T@x is shared by both weight gradients.
    improved=(_mm_ops(p,n,k)+_mm_ops(m,k,p) if mode=='joint' else
              _mm_ops(m,k,n)+_mm_ops(m,n,p))
    if dx:
        if mode=='backward_only':improved+=_mm_ops(p,n,k)
        improved+=_mm_ops(m,p,k)
    if dw1 or dw2:improved+=_mm_ops(p,m,k)
    if dw1:improved+=_mm_ops(n,p,k)
    if dw2:improved+=_mm_ops(p,k,n)
    return {'native_scalar_ops':native,'reassociated_scalar_ops':improved,
            'scalar_saving':native-improved,
            'saved_hidden_elements_native':m*n if dw2 else 0,
            'saved_effective_weight_elements':p*k if mode=='joint' and dx else 0,
            'scope':'Conditional first-order scalar model; no timing, allocator, transfer or bit-equivalence guarantee'}


def _native(x,w1,w2,bias1=None,bias2=None):return F.linear(F.linear(x,w1,bias1),w2,bias2)


class _LinearChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w1,w2,mode):
        dx,dw1,dw2=ctx.needs_input_grad[:3]
        ctx.mode=mode
        if mode=='joint':
            effective=w2@w1
            y=F.linear(x,effective)
        else:
            effective=None
            y=_native(x,w1,w2)
        ctx.save_for_backward(x if dw1 or dw2 else None,
                              w1 if dx or dw2 else None,
                              w2 if dx or dw1 else None,
                              effective if dx else None)
        return y

    @staticmethod
    def backward(ctx,g):
        x,w1,w2,effective=ctx.saved_tensors
        dx,dw1,dw2=ctx.needs_input_grad[:3]
        higher=torch.is_grad_enabled()
        # Reconnect the effective weight to the original parameters for higher
        # derivatives; a saved temporary from Function.forward has no such graph.
        if dx and (ctx.mode=='backward_only' or higher):effective=w2@w1
        t=g.T@x if dw1 or dw2 else None
        gx=g@effective if dx else None
        gw1=w2.T@t if dw1 else None
        gw2=t@w1.T if dw2 else None
        return gx,gw1,gw2,None


def linear_chain(x,w1,w2,accuracy='strict',mode='joint',bias1=None,bias2=None):
    """Same two original dense weights, optional real-algebra reassociation.

    'backward_only' keeps native forward evaluation and removes its hidden tape.
    'joint' uses the shared effective weight in forward and input-gradient work.
    Rounded results can differ. No approximation to matrix rank is performed.
    """
    if accuracy not in ('strict','algebraic') or mode not in ('joint','backward_only'):
        raise ValueError('Invalid accuracy or schedule contract')
    if accuracy=='strict':return _native(x,w1,w2,bias1,bias2)
    if mode=='backward_only' and (bias1 is not None or bias2 is not None):
        return _native(x,w1,w2,bias1,bias2)
    eligible=(x.ndim==w1.ndim==w2.ndim==2 and x.shape[1]==w1.shape[1]
              and w1.shape[0]==w2.shape[1] and min(*x.shape,*w1.shape,*w2.shape)>0
              and x.device==w1.device==w2.device and algebraic_device_supported(x)
              and x.dtype==w1.dtype==w2.dtype and x.dtype in (torch.float32,torch.float64)
              )
    eligible=eligible and all(b is None or (b.ndim==1 and b.shape[0]==size and b.dtype==x.dtype and b.device==x.device)
                                 for b,size in ((bias1,w1.shape[0]),(bias2,w2.shape[0])))
    if not eligible:return _native(x,w1,w2,bias1,bias2)
    cost=chain_cost(x.shape[0],x.shape[1],w1.shape[0],w2.shape[0],
                    (x.requires_grad,w1.requires_grad,w2.requires_grad),mode)
    if cost['scalar_saving']<=0:return _native(x,w1,w2,bias1,bias2)
    # Native autograd already realizes the shared P/T dagger on this graph.
    # Keep it for joint mode; custom backward is needed to retain native forward
    # evaluation while omitting its wide hidden residual.
    if mode=='joint':
        bias=w2@bias1 if bias1 is not None else None
        if bias2 is not None:bias=bias2 if bias is None else bias+bias2
        return F.linear(x,w2@w1,bias)
    return _LinearChain.apply(x,w1,w2,mode)


def factored_gemm_pullback(a,b,left,right):
    """G=left@right.T supplied by structure. Not an SVD/rank approximation."""
    return left@(b@right).T,(a.T@left)@right.T
