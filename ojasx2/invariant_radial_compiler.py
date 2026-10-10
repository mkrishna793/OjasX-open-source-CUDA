"""Exact-real Gram-invariant compiler for radial projection blocks.

Recognizes an existing radial block; it never replaces an elementwise GELU.
All original W, bias, Q and optional sigmoid-gate parameters remain trainable.
ML execution is MPS-only. Algebraic FP32 mode is experimental; strict is native.
"""
from dataclasses import dataclass,asdict
import torch

KINDS=('cubic','reciprocal','sigmoid')

def radial_gate(s,kind,gate=None):
    if kind=='cubic':return s,torch.ones_like(s),None
    if kind=='reciprocal':
        a=1/(1+s);return a,-a.square(),None
    if kind=='sigmoid':
        if gate is None or gate.shape!=(2,):raise ValueError('Sigmoid gate requires two original parameters')
        a=torch.sigmoid(gate[0]*s+gate[1]);v=a*(1-a)
        return a,gate[0]*v,v
    raise ValueError('Unknown radial activation')

def native_radial(x,w,b,q,kind='reciprocal',gate=None):
    z=x@w+b;s=z.square().mean(-1,keepdim=True)
    if kind=='cubic':a=s
    elif kind=='reciprocal':a=1/(1+s)
    elif kind=='sigmoid':a=torch.sigmoid(gate[0]*s+gate[1])
    else:raise ValueError('Unknown radial activation')
    return (a*z)@q

@dataclass(frozen=True)
class InvariantPlan:
    shape:tuple
    native_leading_arithmetic:int
    core_leading_arithmetic:int
    eligible:bool
    certificate:str='Radial scale depends only on mean squared norm; all original parameters retained'
    scope:str='Conservative leading full forward/all-gradient arithmetic; not an instruction count, peak-memory bound or latency theorem'

def invariant_plan(m,k,n,r):
    d=k+1
    native=6*m*n*(d+r)
    core=6*d*(d+r)*(m+n)+12*m*d+12*m*r
    return InvariantPlan((m,k,n,r),native,core,min(m,k,n,r)>0 and core<native)

def _native_grads(inputs,kind,p,higher):
    x,w,b,q,gate=inputs
    active=[t for t in inputs if t is not None and t.requires_grad]
    with torch.enable_grad():
        h=native_radial(x,w,b,q,kind,gate)
        grads=torch.autograd.grad(h,active,p,create_graph=higher)
    it=iter(grads)
    return tuple(next(it) if t is not None and t.requires_grad else None for t in inputs)

class _Invariant(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,q,gate,kind,evidence):
        a=torch.cat((x,torch.ones_like(x[:,:1])),1)
        v=torch.cat((w,b[None,:]),0)
        c=v@v.T;t=v@q;ac=a@c
        s=(ac*a).sum(-1,keepdim=True)/w.shape[1]
        # Cancellation in a Gram-based norm estimate is not cured by a real
        # algebra proof. Detect an ill-conditioned/negative estimate and use
        # the original differentiable program without clipping the moment.
        absolute_bound=a.abs().square().sum(-1,keepdim=True)*v.square().sum()/w.shape[1]
        bad=not bool(torch.isfinite(s).all() & (s>=0).all() & ((s>1e-5*absolute_bound)|(absolute_bound==0)).all())
        ctx.kind=kind;ctx.evidence=evidence;ctx.native=bad
        ctx.has_gate=gate is not None
        originals=(x,w,b,q)+((gate,) if gate is not None else ())
        if bad:
            ctx.save_for_backward(*originals)
            if evidence is not None:evidence.append({'path':'native_gram_condition','plan':asdict(invariant_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1]))})
            return native_radial(x,w,b,q,kind,gate)
        scale,slope,gate_slope=radial_gate(s,kind,gate)
        u=a@t;h=scale*u
        ctx.save_for_backward(*originals,a,v,c,t,ac,s,scale,slope,u)
        if evidence is not None:evidence.append({'path':'invariant_radial','plan':asdict(invariant_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1]))})
        return h

    @staticmethod
    def backward(ctx,p):
        saved=list(ctx.saved_tensors);x,w,b,q=saved[:4];offset=5 if ctx.has_gate else 4
        gate=saved[4] if ctx.has_gate else None;inputs=(x,w,b,q,gate)
        higher=torch.is_grad_enabled()
        with torch.no_grad():numeric=bool(torch.isfinite(p).all() & (p.abs().amax()<=8))
        if ctx.native or higher or not numeric:
            if ctx.evidence is not None:ctx.evidence.append({'path':'native_higher_order' if higher else 'native_reverse_fallback'})
            return (*_native_grads(inputs,ctx.kind,p,higher),None,None)
        a,v,c,t,ac,s,scale,slope,u=saved[offset:]
        r=scale*p
        coupling=(p*u).sum(-1,keepdim=True)
        e=slope*coupling/w.shape[1]
        ar=a.T@r;cov=a.T@(e*a)
        da=r@t.T+2*e*ac
        dv=ar@q.T+2*cov@v
        dq=v.T@ar
        dg=None
        if gate is not None:
            gs=scale*(1-scale)*coupling
            dg=torch.stack(((gs*s).sum(),gs.sum()))
        return da[:,:-1],dv[:-1],dv[-1],dq,dg,None,None

def invariant_radial(x,w,b,q,kind='reciprocal',gate=None,accuracy='strict',evidence=None):
    if kind not in KINDS:raise ValueError('Unknown radial activation')
    if accuracy not in ('strict','algebraic'):raise ValueError('Unknown accuracy policy')
    if not (x.ndim==w.ndim==q.ndim==2 and b.ndim==1 and x.shape[1]==w.shape[0] and w.shape[1]==b.shape[0]==q.shape[0]):raise ValueError('Incompatible shapes')
    tensors=(x,w,b,q)+((gate,) if gate is not None else ())
    if any(t.device.type!='mps' for t in tensors):raise ValueError('MPS GPU only')
    if kind=='sigmoid' and (gate is None or gate.shape!=(2,)):raise ValueError('Two original sigmoid parameters required')
    if kind!='sigmoid' and gate is not None:raise ValueError('This radial law has no gate parameters')
    plan=invariant_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1])
    if accuracy=='strict' or any(t.dtype!=torch.float32 for t in tensors) or torch.is_autocast_enabled('mps') or not plan.eligible:
        if evidence is not None:evidence.append({'path':'native_policy_or_cost','plan':asdict(plan)})
        return native_radial(x,w,b,q,kind,gate)
    with torch.no_grad():
        ok=torch.isfinite(x).all() & torch.isfinite(w).all() & torch.isfinite(b).all() & torch.isfinite(q).all()
        bounded=(x.abs().amax()<=8)&(w.abs().amax()<=8)&(b.abs().amax()<=8)&(q.abs().amax()<=8)
        if gate is not None:ok=ok&torch.isfinite(gate).all();bounded=bounded&(gate.abs().amax()<=8)
    if not bool(ok & bounded):
        if evidence is not None:evidence.append({'path':'native_numeric_domain'})
        return native_radial(x,w,b,q,kind,gate)
    return _Invariant.apply(x,w,b,q,gate,kind,evidence)
