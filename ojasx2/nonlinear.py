"""Conditional exact-real nonlinear schedules, with native default/fallback.

ReLU common-mask fast path keeps native forward evaluation. It does not assume
random masks coincide, truncate rank, or silently approximate smooth activations.
"""
import torch
import torch.nn.functional as F
from .contractions import chain_cost


def _all_finite(v):
    # CPU/MPS aminmax propagates NaN and exposes infinity at an endpoint.
    # Scalar endpoint checks avoid several full-size pointwise temporaries.
    low, high = torch.aminmax(v)
    return bool(torch.isfinite(low) & torch.isfinite(high))


class _CommonMaskReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2, native_y, mask):
        ctx.save_for_backward(x, w1, w2, b1, mask)
        return native_y

    @staticmethod
    def backward(ctx, g):
        x, w1, w2, b1, mask = ctx.saved_tensors
        dx, dw1, dw2, db1, db2 = ctx.needs_input_grad[:5]
        # D is exact shared Boolean branch information. Multiplication by D
        # is selection, not an approximation to the activation derivative.
        selected_w2 = torch.where(mask[None, :], w2, 0)
        t = g.T @ x if dw1 or dw2 else None
        gx = g @ (selected_w2 @ w1) if dx else None
        gw1 = selected_w2.T @ t if dw1 else None
        gw2 = None
        s = g.sum(0) if db1 or db2 or (dw2 and b1 is not None) else None
        if dw2:
            gw2 = t @ w1.T
            if b1 is not None:
                gw2 = gw2 + s[:, None]*b1[None, :]
            gw2 = torch.where(mask[None, :], gw2, 0)
        gb1 = torch.where(mask, s @ w2, 0) if db1 else None
        gb2 = s if db2 else None
        return gx, gw1, gw2, gb1, gb2, None, None


def relu_linear_chain(x, w1, w2, bias1=None, bias2=None, accuracy='strict', evidence=None):
    """Linear-ReLU-Linear; opt-in sharing only for exactly identical row masks.

Native forward is evaluated once, including its ReLU. A common mask and
positive modeled saving replace only its tape/pullback. First-order CPU
FP32/FP64 qualification; unsupported contexts retain the native graph.
"""
    if accuracy not in ('strict', 'algebraic'):
        raise ValueError('Invalid accuracy contract')
    z = F.linear(x, w1, bias1)
    reason = 'strict'
    eligible = (accuracy == 'algebraic' and x.ndim == w1.ndim == w2.ndim == 2
                and min(*x.shape, *w1.shape, *w2.shape) > 0
                and x.device.type == 'cpu' and x.device == w1.device == w2.device
                and x.dtype in (torch.float32, torch.float64) and x.dtype == w1.dtype == w2.dtype
                and w2.shape[1] == w1.shape[0]
                and not torch.is_autocast_enabled('cpu')
                and all(b is None or (b.ndim == 1 and b.dtype == x.dtype and b.device == x.device)
                        for b in (bias1, bias2)))
    if eligible:
        with torch.no_grad():
            mask = z > 0
            common = bool((mask == mask[:1]).all())
        cost = chain_cost(x.shape[0], x.shape[1], w1.shape[0], w2.shape[0],
                          (x.requires_grad, w1.requires_grad, w2.requires_grad), 'backward_only')
        # Full finite checks are paid only by a candidate that passes the exact
        # mask/cost guards. Random distinct-mask batches keep their native tape.
        with torch.no_grad():
            finite = all(_all_finite(v) for v in (x, w1, w2, z)) if common and cost['scalar_saving'] > 0 else True
        if finite and common and cost['scalar_saving'] > 0:
            reason = 'common_mask'
            # Avoid constructing a native hidden-activation tape that would
            # immediately be discarded. The first product is evaluated once;
            # its graph retains only the already supplied X/W1 references.
            with torch.no_grad():
                y = F.linear(F.relu(z), w2, bias2)
            if not _all_finite(y):
                reason = 'nonfinite'
                y = F.linear(F.relu(z), w2, bias2)
                if evidence is not None:
                    evidence.append({'path': reason})
                return y
            y = _CommonMaskReLU.apply(x, w1, w2, bias1, bias2, y.detach(), mask[0].clone())
            if evidence is not None:
                evidence.append({'path': reason, 'scope': 'Native forward; selected first-order real-algebra backward'})
            return y
        else:
            reason = 'nonfinite' if not finite else 'distinct_masks' if not common else 'unprofitable'
    elif accuracy != 'strict':
        reason = 'unsupported'
    y = F.linear(F.relu(z), w2, bias2)
    if evidence is not None:
        evidence.append({'path': reason, 'scope': 'Native forward; selected first-order real-algebra backward'})
    return y


def softmax_pullback(probabilities, upstream):
    """Diagonal minus rank-one VJP; native softmax already uses this identity."""
    return probabilities*(upstream-(probabilities*upstream).sum(-1, keepdim=True))


def layernorm_input_pullback(x, upstream, weight=None, eps=1e-5):
    """One last-axis normalization VJP with differentiable recomputed statistics.

Reference formula only; native LayerNorm is the performance baseline.
"""
    center = x-x.mean(-1, keepdim=True)
    r = torch.rsqrt(center.square().mean(-1, keepdim=True)+eps)
    normalized = center*r
    g = upstream if weight is None else upstream*weight
    return r*(g-g.mean(-1, keepdim=True)-normalized*(g*normalized).mean(-1, keepdim=True))


def attention_softmax_row_scalar(output, output_cotangent):
    """Reuse O=P@V: <P, dO@V.T> per row equals <O,dO> in real algebra.

Known FlashAttention identity; caller already supplies O and dO. This is not
a replacement SDPA kernel and does not claim superiority to native attention.
"""
    return (output*output_cotangent).sum(-1, keepdim=True)


def layernorm_chain_cost(m,k,n,p,needs=(True,True,True,True)):
    """Conditional GEMM arithmetic, excluding normalization/vector/guard work.

needs: input, first weight, second weight, normalization scale gradients.
"""
    from .contractions import _mm_ops
    if min(m,k,n,p)<=0:raise ValueError('Positive dimensions required')
    dx,dw1,dw2,dgamma=needs
    forward=_mm_ops(m,k,n)+_mm_ops(m,n,p)
    native=forward
    if dx or dw1 or dgamma:native+=_mm_ops(m,p,n)
    if dx:native+=_mm_ops(m,n,k)
    if dw1:native+=_mm_ops(n,m,k)
    if dw2:native+=_mm_ops(p,m,n)
    improved=forward
    if dw1 or dw2 or dgamma:improved+=_mm_ops(p,m,k)
    if dw2 or dgamma:improved+=_mm_ops(p,k,n)
    if dw1:improved+=_mm_ops(n,p,k)+_mm_ops(k,m,k)+_mm_ops(n,k,k)
    if dx:improved+=_mm_ops(p,n,k)+_mm_ops(m,p,k)+_mm_ops(k,n,k)+_mm_ops(m,k,k)
    return {'native_scalar_ops':native,'structured_scalar_ops':improved,'scalar_saving':native-improved,
            'scope':'First-order main-GEMM model; excludes LayerNorm, vector products, pointwise work, guard/transfer costs and timing'}


class _LayerNormChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w1,w2,gamma,beta,bias2,eps,y,mean,r):
        ctx.eps=eps
        ctx.save_for_backward(x,w1,w2,gamma,beta,bias2,y,mean,r)
        return y

    @staticmethod
    def backward(ctx,g):
        x,w1,w2,gamma,beta,bias2,y,mean,r=ctx.saved_tensors
        dx,dw1,dw2,dgamma,dbeta,dbias2=ctx.needs_input_grad[:6]
        higher=torch.is_grad_enabled()
        if higher or not _all_finite(g):
            # Native replay reconnects statistics/output to original parameters
            # for higher derivatives and preserves native exceptional upstream.
            args=(x,w1,w2,gamma,beta,bias2)
            with torch.enable_grad():
                z=F.linear(x,w1)
                out=F.linear(F.layer_norm(z,(w1.shape[0],),gamma,beta,ctx.eps),w2,bias2)
                wanted=[a for a,need in zip(args,ctx.needs_input_grad[:6]) if need]
                gradients=iter(torch.autograd.grad(out,wanted,g,create_graph=higher))
                result=[next(gradients) if need else None for need in ctx.needs_input_grad[:6]]
            return (*result,None,None,None,None)
        n=w1.shape[0]
        c=w2 if gamma is None else w2*gamma
        t=g.T@(r*x) if dw1 or dw2 or dgamma else None
        s=g.sum(0) if dw2 or dbeta or dbias2 else None
        centered_product=None
        if dw2 or dgamma:
            correction=g.T@(r*mean)
            centered_product=t@w1.T-correction
        gx=gw1=None
        if dx or dw1:
            actual_y=y if bias2 is None else y-bias2
            if beta is not None:actual_y=actual_y-(w2@beta)
            alpha=r.square()*(g*actual_y).sum(-1,keepdim=True)/n
            avg_g=(g@c.sum(-1,keepdim=True))/n
            u=alpha*mean-r*avg_g
            if dx:
                gx=(r*g)@(c@w1)+u*w1.sum(0)-(alpha*x)@(w1.T@w1)
            if dw1:
                gw1=c.T@t+(u.T@x)-w1@(x.T@(alpha*x))
        gw2=centered_product if dw2 else None
        if dw2:
            if gamma is not None:gw2=gw2*gamma
            if beta is not None:gw2=gw2+s[:,None]*beta
        gg=(w2*centered_product).sum(0) if dgamma else None
        gb=s@w2 if dbeta else None
        return gx,gw1,gw2,gg,gb,s if dbias2 else None,None,None,None,None


def layernorm_linear_chain(x,w1,w2,gamma=None,beta=None,bias2=None,eps=1e-5,
                          accuracy='strict',evidence=None):
    """Bias-free first Linear -> affine LayerNorm -> second Linear.

Native forward, structured first-order VJPs; higher orders replay natively.
All original weights/normalization parameters remain independently trainable.
"""
    if accuracy not in ('strict','algebraic'):raise ValueError('Invalid accuracy contract')
    z=F.linear(x,w1)
    eligible=(accuracy=='algebraic' and eps>0 and x.ndim==w1.ndim==w2.ndim==2
              and min(*x.shape,*w1.shape,*w2.shape)>0 and w2.shape[1]==w1.shape[0]
              and x.device.type=='cpu' and x.device==w1.device==w2.device
              and x.dtype in (torch.float32,torch.float64) and x.dtype==w1.dtype==w2.dtype
              and not torch.is_autocast_enabled('cpu')
              and all(a is None or (a.ndim==1 and a.shape[0]==size and a.dtype==x.dtype and a.device==x.device)
                      for a,size in [(gamma,w1.shape[0]),(beta,w1.shape[0]),(bias2,w2.shape[0])]))
    reason='strict' if accuracy=='strict' else 'unsupported'
    if eligible:
        cost=layernorm_chain_cost(x.shape[0],x.shape[1],w1.shape[0],w2.shape[0],
                                  (x.requires_grad,w1.requires_grad,w2.requires_grad,gamma is not None and gamma.requires_grad))
        if cost['scalar_saving']>0:
            with torch.no_grad():
                finite=all(_all_finite(a) for a in (x,w1,w2,z,gamma,beta,bias2) if a is not None)
            if finite:
                with torch.no_grad():
                    h,mean,r=torch.native_layer_norm(z,(w1.shape[0],),gamma,beta,eps)
                    y=F.linear(h,w2,bias2)
                    finite=_all_finite(y) and _all_finite(r)
                if finite:
                    reason='structured_layernorm'
                    if evidence is not None:evidence.append({'path':reason})
                    return _LayerNormChain.apply(x,w1,w2,gamma,beta,bias2,eps,y,mean,r)
            reason='nonfinite'
        else:reason='unprofitable'
    if evidence is not None:evidence.append({'path':reason})
    return F.linear(F.layer_norm(z,(w1.shape[0],),gamma,beta,eps),w2,bias2)


def gram_layernorm_chain_cost(m,k,n,p,needs=(True,True,True,True)):
    """Main-GEMM costs of a centered-weight Gram forward and native AD tree."""
    from .planner import _node_cost
    dx,dw1,dw2,dgamma=needs
    native=layernorm_chain_cost(m,k,n,p,needs)['native_scalar_ops']
    cost=(_node_cost(k,n,k,dw1,dw1)+_node_cost(m,k,k,dx,dw1)
          +_node_cost(p,n,k,dw2 or dgamma,dw1)+_node_cost(m,k,p,dx,dw1 or dw2 or dgamma))
    return {'native_scalar_ops':native,'gram_scalar_ops':cost,'scalar_saving':native-cost,
            'scope':'Main-GEMM forward/native-AD model; excludes mean/vector/pointwise operations, copies, finite/variance guards and dispatch'}


def gram_layernorm_linear_chain(x,w1,w2,gamma=None,beta=None,bias2=None,eps=1e-5,
                               accuracy='strict',evidence=None):
    """Experimental all-algebraic Linear-LayerNorm-Linear using centered weights.

Avoids the m×n primal activation, with ordinary native AD on the reduced graph.
Output rounding also changes. Conservative variance/native fallback is not a
universal conditioning/error certificate; strict remains the default.
"""
    if accuracy not in ('strict','algebraic'):raise ValueError('Invalid accuracy contract')
    def native():return F.linear(F.layer_norm(F.linear(x,w1),(w1.shape[0],),gamma,beta,eps),w2,bias2)
    eligible=(accuracy=='algebraic' and eps>0 and x.ndim==w1.ndim==w2.ndim==2
              and min(*x.shape,*w1.shape,*w2.shape)>0 and x.shape[1]==w1.shape[1] and w2.shape[1]==w1.shape[0]
              and x.device.type=='cpu' and x.device==w1.device==w2.device
              and x.dtype in (torch.float32,torch.float64) and x.dtype==w1.dtype==w2.dtype
              and not torch.is_autocast_enabled('cpu')
              and all(a is None or (a.ndim==1 and a.shape[0]==size and a.dtype==x.dtype and a.device==x.device)
                      for a,size in [(gamma,w1.shape[0]),(beta,w1.shape[0]),(bias2,w2.shape[0])]))
    reason='strict' if accuracy=='strict' else 'unsupported'
    if eligible:
        cost=gram_layernorm_chain_cost(x.shape[0],x.shape[1],w1.shape[0],w2.shape[0],
                                       (x.requires_grad,w1.requires_grad,w2.requires_grad,gamma is not None and gamma.requires_grad))
        if cost['scalar_saving']>0:
            with torch.no_grad():finite=all(_all_finite(a) for a in (x,w1,w2,gamma,beta,bias2) if a is not None)
            if finite:
                mean_weight=w1.mean(0)
                centered_weight=w1-mean_weight
                gram=centered_weight.T@centered_weight
                variance=((x@gram)*x).sum(-1,keepdim=True)/w1.shape[0]
                with torch.no_grad():
                    # Guard clearly risky small/negative variances and large
                    # common weight offsets; this is a conservative heuristic.
                    small=bool((variance<=10*eps).any())
                    low,high=torch.aminmax(centered_weight)
                    center_max=torch.maximum(low.abs(),high.abs())
                    offset_large=bool(mean_weight.abs().max()>center_max)
                    safe=_all_finite(variance) and not small and not offset_large
                if safe:
                    r=torch.rsqrt(variance+eps)
                    c=w2 if gamma is None else w2*gamma
                    y=(x@(c@w1).T-(x@mean_weight)[:,None]*c.sum(-1))*r
                    if beta is not None:y=y+w2@beta
                    if bias2 is not None:y=y+bias2
                    with torch.no_grad():finite=_all_finite(y)
                    if finite:
                        if evidence is not None:evidence.append({'path':'gram_layernorm','conditioning_guard':'heuristic, not certified'})
                        return y
                reason='conditioning' if not safe else 'nonfinite'
            else:reason='nonfinite'
        else:reason='unprofitable'
    if evidence is not None:evidence.append({'path':reason})
    return native()
