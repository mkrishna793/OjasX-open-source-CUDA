"""First-order CPU/MPS reversible coupling experiment; strict native default.

Requires an existing coupling architecture and deterministic GELU/SiLU branches.
Rounded reconstruction is approximate. No conversion of ordinary layers into a
different model, optimizer mutation, RNG replay or native kernel generation is performed.
"""
import math
import torch
import torch.nn.functional as F
from .devices import algebraic_device_supported
from .cached_reverse import CachedReversibleChain


def _branch(x,w,b,activation,scale):
    z=F.linear(x,w,b)
    return scale*(F.gelu(z) if activation=='gelu' else F.silu(z))


class _ReversibleChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x1,x2,activation,scale,*parameters):
        for i in range(0,len(parameters),4):
            wf,bf,wg,bg=parameters[i:i+4]
            x1=x1+_branch(x2,wf,bf,activation,scale)
            x2=x2+_branch(x1,wg,bg,activation,scale)
        ctx.save_for_backward(x1,x2,*parameters)
        ctx.activation=activation;ctx.scale=scale
        return x1,x2

    @staticmethod
    def backward(ctx,g1,g2):
        if torch.is_grad_enabled():
            raise RuntimeError('Reversible algebraic replay supports first-order gradients only; use strict mode for higher derivatives')
        y1,y2,*parameters=ctx.saved_tensors
        gradients=[None]*len(parameters)
        def replay(primal,w,b,incoming,needw,needb):
            with torch.enable_grad():
                z=primal.detach().requires_grad_()
                local_w=w.detach().requires_grad_(needw)
                local_b=b.detach().requires_grad_(needb)
                out=_branch(z,local_w,local_b,ctx.activation,ctx.scale)
                args=[z]+([local_w] if needw else [])+([local_b] if needb else [])
                grads=torch.autograd.grad(out,args,incoming)
            idx=1
            gw=grads[idx] if needw else None;idx+=int(needw)
            gb=grads[idx] if needb else None
            return out.detach(),grads[0],gw,gb
        for i in range(len(parameters)-4,-1,-4):
            wf,bf,wg,bg=parameters[i:i+4]
            needs=ctx.needs_input_grad[4+i:8+i]
            gout,gdy,ggw,ggb=replay(y1,wg,bg,g2,needs[2],needs[3])
            x2=y2-gout
            adj1=g1+gdy
            fout,fdx,gfw,gfb=replay(x2,wf,bf,adj1,needs[0],needs[1])
            x1=y1-fout
            g1,g2=adj1,g2+fdx
            gradients[i:i+4]=[gfw,gfb,ggw,ggb]
            y1,y2=x1,x2
        return (g1 if ctx.needs_input_grad[0] else None,
                g2 if ctx.needs_input_grad[1] else None,None,None,*gradients)


def reversible_coupling_chain(x1,x2,blocks,activation='gelu',scale=.1,accuracy='strict',
                              residual_policy='replay',cache_interval=1,pointwise_backend='native'):
    """Blocks are (wf,bf,wg,bg), both branches d→d, nonempty immutable sequence.

    Recompute each branch once during backward and reuse its value to reconstruct
    the preceding state. Full-chain tape stores final states and parameter refs.
    The opt-in first-order replay changes gradient rounding and requires stable
    reconstruction. There is no universal conditioning/error certificate.
    'preactivation' saves two preactivations per cached block and removes its
    two backward projection replays. cache_interval thins this cache; Metal
    jointly reconstructs states/applies SiLU VJPs on MPS FP32. Strict mode keeps
    the ordinary graph. Full caching trades depth-independent tape for speed.
    """
    if activation not in ('gelu','silu') or accuracy not in ('strict','algebraic'):
        raise ValueError('Unsupported activation or accuracy contract')
    if residual_policy not in ('replay','preactivation') or pointwise_backend not in ('native','metal'):
        raise ValueError('Unsupported residual or pointwise policy')
    if not isinstance(cache_interval,int) or isinstance(cache_interval,bool) or cache_interval<=0:
        raise ValueError('Positive integer cache interval required')
    if pointwise_backend=='metal' and (residual_policy!='preactivation' or activation!='silu'
                                     or x1.device.type!='mps' or x1.dtype!=torch.float32):
        raise ValueError('Metal cached reverse requires preactivation policy, MPS FP32 and SiLU')
    if not isinstance(scale,(float,int)) or not math.isfinite(scale):
        raise ValueError('Finite branch scale required')
    blocks=tuple(tuple(b) for b in blocks)
    if not blocks or any(len(b)!=4 for b in blocks):raise ValueError('Four branch parameters per block required')
    if x1.shape!=x2.shape or x1.ndim<2:raise ValueError('Matching batched state shapes required')
    parameters=tuple(v for block in blocks for v in block)
    d=x1.shape[-1]
    if any(w.shape!=(d,d) or b.shape!=(d,) for wf,bf,wg,bg in blocks for w,b in ((wf,bf),(wg,bg))):
        raise ValueError('Branches must preserve each split feature dimension')
    if any(v.device!=x1.device or v.dtype!=x1.dtype for v in (x2,*parameters)):
        raise ValueError('Matching state/parameter devices and dtypes required')
    if accuracy=='algebraic':
        if not algebraic_device_supported(x1):
            raise ValueError('Reversible replay supports CPU FP32/FP64 and experimental MPS FP32 without autocast')
        if residual_policy=='preactivation':
            return CachedReversibleChain.apply(x1,x2,activation,scale,pointwise_backend,cache_interval,*parameters)
        return _ReversibleChain.apply(x1,x2,activation,scale,*parameters)
    for wf,bf,wg,bg in blocks:
        x1=x1+_branch(x2,wf,bf,activation,scale)
        x2=x2+_branch(x1,wg,bg,activation,scale)
    return x1,x2
