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
 selected='approximate' if degree in (8,12) and accurate else 'strict'
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

def scalar_approximation(kind):
 x=torch.linspace(-1,1,32769,device='mps',requires_grad=True);c=polynomial_coefficients(kind,8)
 y=torch.zeros_like(x)
 for value in reversed(c):y=y*x+value
 native=F.gelu(x,approximate='none') if kind=='gelu' else F.silu(x)
 slope=torch.autograd.grad(y,x,torch.ones_like(x))[0];nslope=torch.autograd.grad(native,x,torch.ones_like(x))[0]
 errors=[compare(y,native),compare(slope,nslope)]
 return {'activation':kind,'degree':8,'radius':1,'points':32769,'errors':errors,'passed':all(e['passed'] for e in errors),
  'scope':'Sampled uniform scalar primal and slope comparison on MPS; the kernel proof certifies the separate polynomial-reference tail.'}

def edge(name,kind='gelu'):
 v=list(make((256,1,512,2)));mode='approximate';degree=12
 if name=='zero':v=[torch.zeros_like(t).requires_grad_(True) for t in v]
 if name=='noncontiguous':v[0]=(torch.randn(256,2,device='mps')*.1)[:,::2].detach().requires_grad_(True);assert not v[0].is_contiguous()
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
 v=make((64,1,512,2));directions=[torch.randn_like(t) for t in v];out=[]
 for mode in ('polynomial','dense_polynomial'):
  h=call(v,kind,12,mode) if mode=='polynomial' else dense_polynomial(*v,kind,12)
  g=torch.autograd.grad(h.square().mean(),v,create_graph=True)
  out.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,directions)),v))
 errors=[compare(a,b) for a,b in zip(*out)]
 return {'activation':kind,'errors':errors,'passed':all(e['passed'] for e in errors)}

def composition(seed,normalized=False):
 torch.manual_seed(seed);first=make((4096,2,4096,2));second=make((4096,2,4096,8))
 values=clone((*first,*second[1:]));labels=torch.arange(4096,device='mps')%8
 def model(v,mode):
  x=F.layer_norm(v[0],(2,)) if normalized else v[0]
  h=call((x,*v[1:4]),'gelu',8,mode)
  return call((h,*v[4:]),'silu',8,mode)
 def sgd(v,mode):
  loss=F.cross_entropy(model(v,mode),labels);g=torch.autograd.grad(loss,v)
  with torch.no_grad():
   for t,gg in zip(v[1:],g[1:]):t.add_(gg,alpha=-1e-4)
  return loss,g
 states={mode:clone(values) for mode in ('strict','approximate')};out=[];memory={}
 for mode in states:
  v=states[mode]
  with SavedTensorAudit(list(v)) as audit:h=model(v,mode);loss=F.cross_entropy(h,labels)
  out.append((h,*torch.autograd.grad(loss,v)));memory[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(out[1],out[0])];steps=[]
 for i in range(4):
  la,ga=sgd(states['strict'],'strict');lb,gb=sgd(states['approximate'],'approximate')
  es=[compare(lb.reshape(1),la.reshape(1)),*[compare(a,b) for a,b in zip(gb,ga)],*[compare(a,b) for a,b in zip(states['approximate'][1:],states['strict'][1:])]]
  steps.append({'step':i,'errors':es,'passed':all(e['passed'] for e in es)})
 # All optimizer arithmetic and both moments remain on MPS. No CPU oracle.
 av={mode:clone(values) for mode in states};mom={mode:[torch.zeros_like(t) for t in values[1:]] for mode in states};var={mode:[torch.zeros_like(t) for t in values[1:]] for mode in states};adam=[]
 for i in range(1,5):
  record={}
  for mode,v in av.items():
   loss=F.cross_entropy(model(v,mode),labels);grad=torch.autograd.grad(loss,v)
   with torch.no_grad():
    for t,g,m,u in zip(v[1:],grad[1:],mom[mode],var[mode]):
     m.mul_(.9).add_(g,alpha=.1);u.mul_(.999).addcmul_(g,g,value=.001)
     t.mul_(1-1e-4*.01).addcdiv_(m/(1-.9**i),(u/(1-.999**i)).sqrt()+1e-8,value=-1e-4)
   record[mode]=(loss,grad)
  es=[compare(record['approximate'][0].reshape(1),record['strict'][0].reshape(1)),*[compare(a,b) for a,b in zip(record['approximate'][1],record['strict'][1])],*[compare(a,b) for a,b in zip(av['approximate'][1:],av['strict'][1:])]]
  adam.append({'step':i,'errors':es,'passed':all(e['passed'] for e in es)})
 passed=all(e['passed'] for e in errors) and all(e['passed'] for e in steps+adam)
 selected='approximate' if passed else 'strict';h=model(values,selected);ref=model(values,'strict')
 g=torch.autograd.grad(F.cross_entropy(h,labels),values);rg=torch.autograd.grad(F.cross_entropy(ref,labels),values)
 admission=[compare(h,ref),*[compare(a,b) for a,b in zip(g,rg)]]
 return {'model':'layernorm_gelu_silu' if normalized else 'gelu_silu_crossentropy',
  'shape':[4096,2,4096,2,8],'seed':seed,'raw_errors':errors,'four_step_sgd':steps,
  'four_step_adamw':adam,'candidate_passed':passed,'selected_policy':selected,'memory':memory,
  'admission_passed':all(e['passed'] for e in admission),
  'scope':'Bounded narrow-input two-block model, includes crossentropy tape in memory measurement. Supplied-model preflight only; no automatic optimizer/graph error guarantee.'}

def main():
 if not torch.backends.mps.is_available():raise RuntimeError('MPS GPU unavailable; no CPU fallback')
 claims,reflections=hermite_claims();scalars=[scalar_approximation(k) for k in ('gelu','silu')];cases=[];repeats=[]
 for shape in ((512,1,1024,2),(512,2,1024,2)):
  for kind in ('gelu','silu'):
   for degree in (4,8,12):
    print('GPU polynomial',shape,kind,degree,flush=True);cases.append(case(shape,kind,degree,4100+len(cases)))
 for seed in (4201,4202,4203):
  for kind in ('gelu','silu'):
   print('GPU large spectral',seed,kind,flush=True);repeats.append(case((8192,2,4096,2),kind,8,seed,timed=True))
 for seed in (4251,4252,4253):
  for kind in ('gelu','silu'):
   print('GPU narrow scaling',seed,kind,flush=True);repeats.append(case((8192,1,8192,2),kind,8,seed,timed=True))
 edges=[edge(n) for n in ('zero','noncontiguous','domain','nan','inf','strict','low_degree','cost','frozen','large_cotangent','nan_cotangent')]
 higher_cases=[higher(k) for k in ('gelu','silu')]
 models=[]
 for seed in (4301,4302,4303):
  print('GPU composed GELU-SiLU SGD/AdamW',seed,flush=True);models.append(composition(seed))
 print('GPU normalized composition',flush=True);models.append(composition(4350,normalized=True))
 paths=[Path(__file__),ROOT/'ojasx2/spectral_compiler.py',ROOT/'ojasx2/spectral_coefficients.py',ROOT/'ojasx2/gradient_factor_compiler.py',ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 report={'status':'completed_with_raw_approximation_rejections','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,
  'hermite_claims':claims,'scalar_approximations':scalars,'reflection_and_normalization_claims':reflections,'cases':cases,'repeats':repeats,'edges':edges,'higher_derivatives':higher_cases,'composed_models':models,
  'polynomial_failures':sum(not c['polynomial_passed'] for c in cases+repeats),
  'raw_native_approximation_rejections':sum(not c['native_approximation_passed'] for c in cases+repeats),
  'admission_failures':sum(not c['admission_passed'] for c in cases+repeats),
  'edge_failures':sum(not c['passed'] for c in edges),'higher_derivative_failures':sum(not c['passed'] for c in higher_cases),
  'raw_model_rejections':sum(not c['candidate_passed'] for c in models),'model_admission_failures':sum(not c['admission_passed'] for c in models),
  'scalar_approximation_failures':sum(not c['passed'] for c in scalars),
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  'scope':'Coordinate-sensitive matched polynomial primal/gradients, degree-8 Chebyshev refinement versus degree-12 Taylor, sampled radius-one feature-box domains. No Gaussian, all-model, formal IEEE or convergence guarantee. Hermite claims are independently tested and rejected when false.'}
 (ROOT/'reports/spectral-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE', {k:v for k,v in report.items() if k.endswith('failures') or k.endswith('rejections')},flush=True)

if __name__=='__main__':main()
