"""Opt-in mathematical compiler prototype for closed coordinate actions.

Exact-real log conjugacy of positive coordinate-wise power/scale chains;
no equivalence claim for arbitrary dense GELU/SiLU or IEEE bit patterns.
"""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class CoordinatePlan:
    degrees: tuple
    exponent: int
    weight_exponents: tuple
    chart: str
    proof: str
    eligible: bool
    reason: str

def plan_power_chain(degrees):
    degrees=tuple(degrees)
    if not degrees or len(degrees)>32 or any(type(p) is not int or p not in (1,2,3) for p in degrees):
        return CoordinatePlan(degrees,1,(),'identity','native',False,'unsupported_degree_or_depth')
    acc=1;exponents=[]
    for p in reversed(degrees):acc*=p;exponents.append(acc)
    exponents=tuple(reversed(exponents))
    # Degree growth magnifies FP32 error. Real conjugacy alone is insufficient.
    return CoordinatePlan(degrees,acc,exponents,'log','log_homogeneous_closed/decoded',acc<=16 and len(degrees)>=4,
                          'closed_log_action' if acc<=16 and len(degrees)>=4 else 'depth_or_roundoff_guard')

def native_power_chain(x,weights,degrees):
    h=x
    for index,p in enumerate(degrees):
        h=h*weights[index]
        if p!=1:h=h.pow(p)
    return h

def encoded_power_value(x,w,r,total):
    return (total*x.log()+(r[:,None]*w.log()).sum(0)).exp()

class _CoordinatePower(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,r,total):
        y=encoded_power_value(x,w,r,total)
        ctx.total=total;ctx.save_for_backward(x,w,r,y)
        return y
    @staticmethod
    def backward(ctx,g):
        x,w,r,y=ctx.saved_tensors
        if torch.is_grad_enabled():y=encoded_power_value(x,w,r,ctx.total)
        q=g*y
        return q*ctx.total/x,q.sum(0)[None,:]*r[:,None]/w,None,None

def coordinate_power_chain(x,weights,degrees,accuracy='strict',evidence=None):
    if accuracy not in ('strict','algebraic'):raise ValueError('Explicit strict or algebraic contract required')
    if x.device.type!='mps' or weights.device.type!='mps':raise ValueError('GPU-only compiler experiment')
    if x.ndim!=2 or weights.ndim!=2 or weights.shape[0]!=len(degrees) or weights.shape[1]!=x.shape[1] or x.numel()==0:
        raise ValueError('Expected X[batch,width], W[depth,width], positive dimensions')
    if any(type(p) is not int or p<1 for p in degrees):raise ValueError('Positive integer degrees required')
    plan=plan_power_chain(degrees);path='native'
    if accuracy=='algebraic' and plan.eligible and x.dtype==weights.dtype==torch.float32 and not torch.is_autocast_enabled('mps'):
        # A cache-free domain decision; all GPU scalar reads/reductions are timed.
        with torch.no_grad():
            lo_x,hi_x=torch.aminmax(x);lo_w,hi_w=torch.aminmax(weights)
            valid=(lo_x>=.75)&(hi_x<=1.25)&(lo_w>=.75)&(hi_w<=1.25)
        if bool(valid.item()):
            r=torch.tensor(plan.weight_exponents,device='mps',dtype=torch.float32)
            with torch.no_grad():
                lx=x.log();lw=weights.log();ly=plan.exponent*lx+(r[:,None]*lw).sum(0)
                within=(ly.abs().amax()<=8)
            if bool(within.item()):
                if evidence is not None:evidence.append({'path':'log_affine_action','plan':plan.__dict__,'primal':'exact_real','floating':'empirical_tolerance_only'})
                return _CoordinatePower.apply(x,weights,r,plan.exponent)
            path='output_range_fallback'
        else:path='domain_fallback'
    elif accuracy=='algebraic':path='representation_or_numeric_fallback'
    if evidence is not None:evidence.append({'path':path,'plan':plan.__dict__})
    return native_power_chain(x,weights,degrees)

def coordinate_cost(m,k,degrees):
    plan=plan_power_chain(degrees);L=len(degrees)
    # Explicit scalar arithmetic convention: p powers use p-1 multiplies,
    # shared weight SGD costs 2*L*k; no hidden exponentiation/guard work omitted.
    native=6*sum(degrees)*m*k+2*L*k
    arithmetic=8*m*k+5*L*k+L*k+2*L*k
    transcendentals={'log':2*m*k+2*L*k,'exp':m*k}
    return {'native_conservative_scalar_upper_model':native,'candidate_scalar_upper_model':arithmetic,
            'candidate_transcendental_calls_upper':transcendentals,'native_fresh_residual_lower_elements':L*m*k,
            'candidate_fresh_residual_elements':m*k+L,'scope':'Upper models, not an ordering proof: log/exp are separate and must not be compared as one scalar multiply. Includes repeated encoding for domain/output guard and all original input/weight gradients plus weight SGD. Reductions, scalar validation, allocations and synchronization are measured separately by GPU timings.'}


def native_koopman(x,theta,depth):
    shared=theta.ndim==2
    h=x
    for index in range(depth):
        a,b,c=theta if shared else theta[index]
        u,v=h.unbind(-1)
        h=torch.stack((a*u,b*v+c*u.square()),-1)
    return h

def coordinate_koopman(x,theta,depth,accuracy='strict',evidence=None):
    """Exact-real finite invariant space for a trainable triangular recurrence.
    X[batch,width,2], theta[3,width], same original parameters at each step.
    Every gradient is provided by AD of the composed lifted operator.
    """
    if accuracy not in ('strict','algebraic'):raise ValueError('Invalid accuracy contract')
    if x.device.type!='mps' or theta.device.type!='mps':raise ValueError('GPU-only experiment')
    if x.ndim!=3 or x.shape[-1]!=2 or theta.shape not in ((3,x.shape[1]),(depth,3,x.shape[1])) or x.numel()==0 or type(depth) is not int or depth<1:
        raise ValueError('Invalid triangular recurrence shapes/depth')
    path='native'
    if accuracy=='algebraic' and x.dtype==theta.dtype==torch.float32 and 4<=depth<=64 and not torch.is_autocast_enabled('mps'):
        with torch.no_grad():
            valid=(x.abs().amax()<=8)&(theta[..., :2,:].abs().amax()<=1.01)&(theta[..., 2,:].abs().amax()<=.1)
        if bool(valid.item()):
            shared=theta.ndim==2
            a,b,c=theta.unbind(0 if shared else 1);zero=torch.zeros_like(a)
            matrix=torch.stack((torch.stack((a,zero,zero),-1),torch.stack((zero,b,c),-1),torch.stack((zero,zero,a.square()),-1)),-2)
            if shared:transfer=torch.linalg.matrix_power(matrix,depth)
            else:
                # Balanced composition of noncommuting per-layer operators.
                level=matrix
                while level.shape[0]>1:
                    count=level.shape[0]//2
                    merged=level[1:2*count:2]@level[0:2*count:2]
                    level=torch.cat((merged,level[-1:]),0) if level.shape[0]%2 else merged
                transfer=level[0]
            u,v=x.unbind(-1);lifted=torch.stack((u,v,u.square()),-1)
            decoded=(lifted.transpose(0,1)@transfer.transpose(-1,-2)).transpose(0,1)[...,:2]
            if evidence is not None:evidence.append({'path':'finite_koopman','features':3,'depth':depth,'shared_weights':shared,'proof':'koopman_trainable_closed and coordinate_decoded_iteration'})
            return decoded
        path='domain_fallback'
    elif accuracy=='algebraic':path='representation_or_numeric_fallback'
    if evidence is not None:evidence.append({'path':path})
    return native_koopman(x,theta,depth)
