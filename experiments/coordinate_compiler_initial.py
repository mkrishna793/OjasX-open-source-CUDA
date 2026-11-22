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
    return CoordinatePlan(degrees,acc,exponents,'log','log_homogeneous_closed/decoded',acc<=256 and len(degrees)>=4,
                          'closed_log_action' if acc<=256 and len(degrees)>=4 else 'depth_or_roundoff_guard')

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
