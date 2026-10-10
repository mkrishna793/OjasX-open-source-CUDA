"""MPS-only coordinate-sensitive polynomial projection research compiler.

Exact-real execution of a matching polynomial primal and its full VJP.
Finite polynomials approximate GELU/SiLU; strict mode retains the native model.
No Gaussian input, cheap rank, IEEE or whole-model error guarantee is assumed.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
from itertools import product
import math
import torch
from .gradient_factor_compiler import native_projection

DEGREES=(2,4,6,8,10,12)
_CACHE={}

def polynomial_coefficients(kind,degree):
 if kind not in ('gelu','silu') or degree not in DEGREES:raise ValueError('Unsupported polynomial')
 c=[0.]*(degree+1);c[1]=.5
 if kind=='gelu':
  for ell in range(degree//2):c[2*ell+2]=(-1.)**ell/(math.sqrt(2*math.pi)*2**ell*math.factorial(ell)*(2*ell+1))
 else:
  for j,v in {2:1/4,4:-1/48,6:1/480,8:-17/80640,10:31/1451520,12:-691/319334400}.items():
   if j<=degree:c[j]=v
 return tuple(c)

@lru_cache(None)
def exponents(k,degree):
 if k not in (1,2,3,4):raise ValueError('Bounded input-width research grammar')
 return tuple(sorted((v for v in product(range(degree+1),repeat=k) if sum(v)<=degree),key=lambda v:(sum(v),v)))

@dataclass(frozen=True)
class SpectralPlan:
 shape:tuple
 degree:int
 features:int
 native_leading_arithmetic:int
 candidate_leading_arithmetic:int
 eligible:bool
 scope:str='Leading full forward/all-original-gradient model including monomial pulls and bias translation; shared SGD excluded. Not latency or exact instruction count.'

def spectral_plan(m,k,n,r,degree):
 if degree not in DEGREES:raise ValueError('Unsupported degree')
 width=math.comb(k+degree,degree)
 native=6*m*n*(k+r)+4*m*n
 core=6*width*r*(m+n)+8*(m+n)*width*(k+1)+2*n*(degree+1)**2+2*k*n
 return SpectralPlan((m,k,n,r),degree,width,native,core,
  min(m,k,n,r)>0 and k in (1,2,3,4) and width<=512 and core<native)

def _metadata(k,kind,degree):
 key=(k,kind,degree)
 if key in _CACHE:return _CACHE[key]
 terms=exponents(k,degree);powers=torch.tensor(terms,device='mps',dtype=torch.long)
 total=powers.sum(1);multiplicity=[math.factorial(sum(t))/math.prod(math.factorial(j) for j in t) for t in terms]
 c=polynomial_coefficients(kind,degree)
 translation=[[c[t+j]*math.comb(t+j,t) if t+j<=degree else 0 for j in range(degree+1)] for t in range(degree+1)]
 result=(powers,total,torch.tensor(multiplicity,device='mps',dtype=torch.float32),
  torch.tensor(translation,device='mps',dtype=torch.float32),torch.arange(degree+1,device='mps',dtype=torch.float32))
 _CACHE[key]=result;return result

def _monomials(values,powers,degree):
 # Zero-safe integer powers; derivatives never divide by a value.
 vp=values[:,:,None]**torch.arange(degree+1,device='mps',dtype=torch.float32)[None,None,:]
 components=[vp[:,i,powers[:,i]] for i in range(values.shape[1])]
 result=components[0]
 for c in components[1:]:result=result*c
 return result,vp

def _monomial_pull(values,vp,powers,cotangent):
 cols=[]
 for i in range(values.shape[1]):
  value=vp[:,i,(powers[:,i]-1).clamp_min(0)]*powers[:,i]
  for j in range(values.shape[1]):
   if i!=j:value=value*vp[:,j,powers[:,j]]
  cols.append((value*cotangent).sum(1))
 return torch.stack(cols,1)

def dense_polynomial(x,w,b,q,kind,degree):
 z=x@w+b;c=polynomial_coefficients(kind,degree);y=torch.zeros_like(z)
 for v in reversed(c):y=y*z+v
 return y@q

class _Spectral(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b,q,kind,degree):
  powers,total,multi,translation,order=_metadata(x.shape[1],kind,degree)
  phi,_=_monomials(x,powers,degree)
  weight_monomials,_=_monomials(w.T,powers,degree)
  shifts=(b[:,None]**order[None,:])@translation.T
  theta=weight_monomials*multi[None,:]*shifts[:,total]
  core=theta.T@q
  ctx.kind=kind;ctx.degree=degree
  ctx.save_for_backward(x,w,b,q,phi,theta,core,shifts)
  return phi@core
 @staticmethod
 def backward(ctx,p):
  x,w,b,q,phi,theta,core,shifts=ctx.saved_tensors
  higher=torch.is_grad_enabled()
  # Extreme/nonfinite cotangents use the same polynomial's dense calculation.
  with torch.no_grad():safe=bool(torch.isfinite(p).all() & (p.abs().amax()<=8))
  if higher or not safe:
   values=(x,w,b,q);active=[t for t in values if t.requires_grad]
   with torch.enable_grad():
    h=dense_polynomial(x,w,b,q,ctx.kind,ctx.degree)
    grads=torch.autograd.grad(h,active,p,create_graph=higher)
   it=iter(grads)
   return (*tuple(next(it) if t.requires_grad else None for t in values),None,None)
  powers,total,multi,translation,order=_metadata(x.shape[1],ctx.kind,ctx.degree)
  dc=phi.T@p;dphi=p@core.T;dtheta=q@dc.T;dq=theta@dc
  _,xp=_monomials(x,powers,ctx.degree);dx=_monomial_pull(x,xp,powers,dphi)
  mon,wp=_monomials(w.T,powers,ctx.degree)
  dmon=dtheta*multi[None,:]*shifts[:,total]
  dw=_monomial_pull(w.T,wp,powers,dmon).T
  # T_s'(b)=(s+1)T_(s+1)(b), with T_(D+1)=0.
  shift_derivatives=torch.cat((shifts[:,1:]*order[None,1:],torch.zeros_like(shifts[:,:1])),1)
  db=(dtheta*mon*multi[None,:]*shift_derivatives[:,total]).sum(1)
  return dx,dw,db,dq,None,None

def spectral_projection(x,w,b,q,kind='gelu',degree=12,accuracy='strict',radius=1.,evidence=None):
 if accuracy not in ('strict','polynomial','approximate'):raise ValueError('Unknown accuracy policy')
 polynomial_coefficients(kind,degree)
 values=(x,w,b,q)
 if not (x.ndim==w.ndim==q.ndim==2 and b.ndim==1 and x.shape[1]==w.shape[0] and w.shape[1]==b.shape[0]==q.shape[0]):raise ValueError('Incompatible shape')
 if any(t.device.type!='mps' for t in values):raise ValueError('MPS GPU only')
 if not isinstance(radius,(int,float)) or isinstance(radius,bool) or not 0<radius<=1:raise ValueError('Radius in (0,1] required')
 plan=spectral_plan(x.shape[0],x.shape[1],w.shape[1],q.shape[1],degree)
 if accuracy=='strict' or any(t.dtype!=torch.float32 for t in values) or torch.is_autocast_enabled('mps') or not plan.eligible:
  if evidence is not None:evidence.append({'path':'native_policy_or_cost','plan':asdict(plan)})
  return native_projection(x,w,b,q,kind)
 with torch.no_grad():
  ok=torch.isfinite(x).all() & torch.isfinite(w).all() & torch.isfinite(b).all() & torch.isfinite(q).all()
  bounded=(x.abs().amax()<=1)&(w.abs().amax()<=1)&(b.abs().amax()<=1)&(q.abs().amax()<=8)
  # A cheap conservative feature-box certificate, without m x N materialization.
  bound=x.abs().amax(0)@w.abs()+b.abs()
  in_domain=bool((bound<=radius).all())
 if not bool(ok & bounded) or not in_domain:
  if evidence is not None:evidence.append({'path':'native_numeric_or_domain','plan':asdict(plan)})
  return native_projection(x,w,b,q,kind)
 # A research approximation is opt-in and still requires program qualification.
 # Low degrees may be examined as exact polynomial models, but are not admitted
 # as GELU/SiLU approximations on radius one by an invented accuracy axiom.
 if accuracy=='approximate' and degree<12:
  if evidence is not None:evidence.append({'path':'native_unqualified_approximation_degree','plan':asdict(plan)})
  return native_projection(x,w,b,q,kind)
 if evidence is not None:evidence.append({'path':'spectral_polynomial','plan':asdict(plan),'primal_changed':True,'box_radius':bound.amax().item(),
  'error_scope':'Conditional analytic remainder models and sampled full-gradient qualification; no IEEE or global relative error guarantee.'})
 return _Spectral.apply(x,w,b,q,kind,degree)
