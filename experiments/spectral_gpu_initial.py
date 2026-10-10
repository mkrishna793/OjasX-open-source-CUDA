"""GPU-only spectral claims, matched polynomial VJPs, native accuracy and cost.
Raw approximation failures are retained; native policy is checked separately.
"""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('CPU fallback must be disabled')
from pathlib import Path
import sys,json,math,hashlib,statistics,random,time
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.spectral_compiler import spectral_projection,dense_polynomial,polynomial_coefficients,spectral_plan
from ojasx2.gradient_factor_compiler import native_projection
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,clone

def make(shape):
 m,k,n,r=shape;x=torch.randn(m,k,device='mps').tanh()*.9
 w=torch.randn(k,n,device='mps');w=w/w.abs().sum(0,keepdim=True)*.9
 b=torch.randn(n,device='mps').tanh()*.05;q=torch.randn(n,r,device='mps')/math.sqrt(n)
 return clone((x,w,b,q))

def call(v,kind,degree,policy,ev=None):return spectral_projection(*v,kind,degree,policy,evidence=ev)

def case(shape,kind,degree,seed,timed=False):
 torch.manual_seed(seed);v=make(shape);p=torch.randn(shape[0],shape[3],device='mps')*.1;ev=[];out=[];memory={}
 for mode in ('strict','polynomial'):
  with SavedTensorAudit(list(v)) as audit:h=call(v,kind,degree,mode,ev if mode=='polynomial' else None)
  out.append((h,*torch.autograd.grad(h,v,p)));memory[mode]=audit.summary()
 h=dense_polynomial(*v,kind,degree);oracle=(h,*torch.autograd.grad(h,v,p))
 polynomial_errors=[compare(a,b) for a,b in zip(out[1],oracle)]
 native_errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 target=torch.randn(shape[0],shape[3],device='mps')*.1
 def step(v,mode):
  h=call(v,kind,degree,mode);loss=(h-target).square().mean();g=torch.autograd.grad(loss,v)
  with torch.no_grad():
   for t,gg in zip(v[1:],g[1:]):t.add_(gg,alpha=-1e-4)
  return loss,g
 states={mode:clone(v) for mode in ('strict','polynomial')};steps=[]
 for it in range(4):
  a,ga=step(states['strict'],'strict');b,gb=step(states['polynomial'],'polynomial')
  errors=[compare(b.reshape(1),a.reshape(1)),*[compare(x,y) for x,y in zip(gb,ga)],*[compare(x,y) for x,y in zip(states['polynomial'][1:],states['strict'][1:])]]
  steps.append({'step':it,'errors':errors,'passed':all(e['passed'] for e in errors)})
 exact=all(e['passed'] for e in polynomial_errors);accurate=all(e['passed'] for e in native_errors) and all(c['passed'] for c in steps)
 selected='approximate' if degree==12 and accurate else 'strict'
 hh=call(v,kind,degree,selected);rh=call(v,kind,degree,'strict')
 ag=torch.autograd.grad(hh,v,p);rg=torch.autograd.grad(rh,v,p)
 admitted=[compare(hh,rh),*[compare(a,b) for a,b in zip(ag,rg)]]
 result={'shape':shape,'activation':kind,'degree':degree,'seed':seed,'evidence':ev,
  'polynomial_errors':polynomial_errors,'native_errors':native_errors,'polynomial_passed':exact,
  'native_approximation_passed':accurate,'four_step_sgd':steps,'selected_policy':selected,
  'admission_passed':all(e['passed'] for e in admitted),'memory':memory}
 if timed:
  times={mode:[] for mode in states};rng=random.Random(seed)
  for mode in states:
   for _ in range(4):step(states[mode],mode)
  for _ in range(7):
   order=list(states);rng.shuffle(order)
   for mode in order:
    torch.mps.synchronize();start=time.perf_counter()
    for _ in range(3):step(states[mode],mode)
    torch.mps.synchronize();times[mode].append((time.perf_counter()-start)/3)
  med={mode:statistics.median(t) for mode,t in times.items()}
  result['timing']={'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['polynomial'],
   'scope':'Forward/MSE/all-input-and-original-W,b,Q-gradients/plain SGD; box/domain/numeric guards included, eager native activation baseline. Metadata caches initialized by warmups.'}
 return result

def hermite_claims():
 x=torch.linspace(-8,8,65537,device='mps',requires_grad=True)
 dx=16/65536;pdf=torch.exp(-x.detach().square()/2)/math.sqrt(2*math.pi)
 hs=[torch.ones_like(x),x]
 for k in range(1,12):hs.append(x*hs[-1]-k*hs[-2])
 rows=[]
 for kind in ('gelu','silu'):
  y=F.gelu(x,approximate='none') if kind=='gelu' else F.silu(x)
  slope=torch.autograd.grad(y,x,torch.ones_like(x))[0]
  cs=[(slope*h.detach()*pdf).sum()*dx/math.factorial(k) for k,h in enumerate(hs)]
  for degree in (3,4,8,12):
   approx=sum(cs[k]*hs[k].detach() for k in range(degree+1))
   error=approx-slope
   l2=(error.square()*pdf).sum().sqrt()*math.sqrt(dx)
   rows.append({'activation':kind,'derivative_degree':degree,'gaussian_weighted_l2_error':l2.item(),
    'grid_max_abs_error':error.abs().amax().item(),'degree_4_below_1e_4':l2.item()<1e-4,
    'scope':'FP32 MPS quadrature on [-8,8]; corrected coefficients E[g He_k]/k!, not a rigorous integral or global bound.'})
 # Norm-normalized samples can be deterministic two-point distributions.
 raw=torch.tensor([[-1.,1.]]*16,device='mps');norm=F.layer_norm(raw,(2,),eps=0)
 reflection=[]
 for kind in ('gelu','silu'):
  a=F.gelu(norm) if kind=='gelu' else F.silu(norm)
  b=F.gelu(-norm) if kind=='gelu' else F.silu(-norm)
  reflection.append({'activation':kind,'odd_reflection_equivariant':bool(torch.allclose(b,-a)),
   'normalized_distribution_is_two_point':bool((norm.abs()==1).all())})
 return rows,reflection

def edge(name,kind='gelu'):
 v=list(make((256,1,512,2)));mode='approximate';degree=12
 if name=='zero':v=[torch.zeros_like(t).requires_grad_(True) for t in v]
 if name=='noncontiguous':v[0]=torch.randn(1,512,device='mps')[:,:256].T.contiguous().T.T*.1;v[0]=v[0].detach().requires_grad_(True)
 if name=='domain':v[0]=torch.ones_like(v[0]).requires_grad_(True);v[2]=torch.ones_like(v[2]).requires_grad_(True)
 if name=='nan':v[0]=v[0].detach().clone();v[0][0,0]=float('nan');v[0].requires_grad_(True)
 if name=='inf':v[2]=v[2].detach().clone();v[2][0]=float('inf');v[2].requires_grad_(True)
 if name=='strict':mode='strict'
 if name=='low_degree':degree=4
 if name=='cost':v=list(make((16,4,16,2)))
 if name=='frozen':v[3]=v[3].detach()
 ev=[];h=call(v,kind,degree,mode,ev);active=[t for t in v if t.requires_grad]
 reference=native_projection(*v,kind) if ev[0]['path']!='spectral_polynomial' else dense_polynomial(*v,kind,degree)
 p=torch.ones_like(h)*.1
 if name=='large_cotangent':p.fill_(9)
 if name=='nan_cotangent':p[0,0]=float('nan')
 a=torch.autograd.grad(h,active,p);b=torch.autograd.grad(reference,active,p)
 good=bool(torch.allclose(h,reference,rtol=2e-4,atol=2e-4,equal_nan=True)) and all(bool(torch.allclose(x,y,rtol=2e-4,atol=2e-4,equal_nan=True)) for x,y in zip(a,b))
 return {'case':name,'evidence':ev,'passed':good}

def higher(kind):
 v=make((32,1,256,2));directions=[torch.randn_like(t) for t in v];out=[]
 for mode in ('polynomial','dense_polynomial'):
  h=call(v,kind,12,mode) if mode=='polynomial' else dense_polynomial(*v,kind,12)
  g=torch.autograd.grad(h.square().mean(),v,create_graph=True)
  out.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,directions)),v))
 errors=[compare(a,b) for a,b in zip(*out)]
 return {'activation':kind,'errors':errors,'passed':all(e['passed'] for e in errors)}

def main():
 if not torch.backends.mps.is_available():raise RuntimeError('MPS GPU unavailable; no CPU fallback')
 claims,reflections=hermite_claims();cases=[];repeats=[]
 for shape in ((512,1,1024,2),(512,2,1024,2)):
  for kind in ('gelu','silu'):
   for degree in (4,12):
    print('GPU polynomial',shape,kind,degree,flush=True);cases.append(case(shape,kind,degree,4100+len(cases)))
 for seed in (4201,4202,4203):
  for kind in ('gelu','silu'):
   print('GPU large spectral',seed,kind,flush=True);repeats.append(case((8192,2,4096,2),kind,12,seed,timed=True))
 edges=[edge(n) for n in ('zero','noncontiguous','domain','nan','inf','strict','low_degree','cost','frozen','large_cotangent','nan_cotangent')]
 higher_cases=[higher(k) for k in ('gelu','silu')]
 paths=[Path(__file__),ROOT/'ojasx2/spectral_compiler.py',ROOT/'ojasx2/gradient_factor_compiler.py',ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 report={'status':'completed_with_raw_approximation_rejections','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,
  'hermite_claims':claims,'reflection_and_normalization_claims':reflections,'cases':cases,'repeats':repeats,'edges':edges,'higher_derivatives':higher_cases,
  'polynomial_failures':sum(not c['polynomial_passed'] for c in cases+repeats),
  'raw_native_approximation_rejections':sum(not c['native_approximation_passed'] for c in cases+repeats),
  'admission_failures':sum(not c['admission_passed'] for c in cases+repeats),
  'edge_failures':sum(not c['passed'] for c in edges),'higher_derivative_failures':sum(not c['passed'] for c in higher_cases),
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  'scope':'Coordinate-sensitive degree-12 Taylor primal and matched gradients, sampled radius-one feature-box domains. No Gaussian, all-model, formal IEEE or convergence guarantee. Hermite approximation claims are independently tested and rejected when false.'}
 (ROOT/'reports/spectral-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE', {k:v for k,v in report.items() if k.endswith('failures') or k.endswith('rejections')},flush=True)

if __name__=='__main__':main()
