"""GPU-only qualification of joint activation/all-gradient factor rules.
ML tensors are constructed only on MPS. Host math handles sizes and timings.
"""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback')
from pathlib import Path
import sys,json,math,time,statistics,random,hashlib
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.gradient_factor_compiler import projection_block,native_projection,activation,_slope,KINDS
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,clone

def make(shape,kind,region=True):
 m,k,n,r=shape
 x=torch.randn(m,k,device='mps')*.2;w=torch.randn(k,n,device='mps')/math.sqrt(k)
 b=torch.randn(n,device='mps')*.1;q=torch.randn(n,r,device='mps')/math.sqrt(n)
 if kind=='relu' and region:
  w=w*.05;b=torch.where(torch.arange(n,device='mps')%2==0,2.,-2.)
 return clone((x,w,b,q))

def call(v,kind,mode,ev=None):return projection_block(*v,kind,accuracy=mode,evidence=ev)
def step(v,kind,mode,target):
 h=call(v,kind,mode);loss=(h-target).square().mean();g=torch.autograd.grad(loss,v)
 with torch.no_grad():
  for a,b in zip(v[1:],g[1:]):a.add_(b,alpha=-1e-4)
 return loss,g

def case(shape,kind,seed,region=True,timing=False):
 torch.manual_seed(seed);v=make(shape,kind,region);p=torch.randn(shape[0],shape[3],device='mps');ev=[]
 out=[];mem={}
 for mode in ('strict','algebraic'):
  with SavedTensorAudit(list(v)) as audit:h=call(v,kind,mode,ev if mode=='algebraic' else None)
  out.append((h,*torch.autograd.grad(h,v,p)));mem[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 target=torch.randn(shape[0],shape[3],device='mps')*.1
 states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for i in range(4):
  la,ga=step(states['strict'],kind,'strict',target);lb,gb=step(states['algebraic'],kind,'algebraic',target)
  es=[compare(lb.reshape(1),la.reshape(1)),*[compare(a,b) for a,b in zip(gb,ga)],*[compare(a,b) for a,b in zip(states['algebraic'][1:],states['strict'][1:])]]
  training.append({'step':i,'passed':all(e['passed'] for e in es),'errors':es})
 result={'shape':shape,'activation':kind,'seed':seed,'uniform_relu_region':region,'evidence':ev,'errors':errors,'four_step_sgd':training,'memory':mem,'passed':all(e['passed'] for e in errors) and all(t['passed'] for t in training)}
 if timing:
  times={mode:[] for mode in states};rng=random.Random(seed)
  for mode in states:
   for _ in range(4):step(states[mode],kind,mode,target)
  for _ in range(7):
   order=list(states);rng.shuffle(order)
   for mode in order:
    torch.mps.synchronize();start=time.perf_counter()
    for _ in range(3):step(states[mode],kind,mode,target)
    torch.mps.synchronize();times[mode].append((time.perf_counter()-start)/3)
  med={k:statistics.median(v) for k,v in times.items()}
  result['timing']={'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['algebraic'],
   'scope':'Eager PyTorch baseline; complete forward/loss/all-input-and-parameter gradients/SGD; includes finite and mask guards. 4 warmups, 7 shuffled rounds, 3 steps/sample.'}
 return result

def higher(kind):
 v=make((8,2,64,1),kind);direction=[torch.randn_like(t) for t in v];out=[];ev=[]
 for mode in ('strict','algebraic'):
  g=torch.autograd.grad(call(v,kind,mode,ev).square().mean(),v,create_graph=True)
  out.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,direction)),v))
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 return {'activation':kind,'evidence':ev,'errors':errors,'passed':all(e['passed'] for e in errors)}

def composition(name):
 # All original surrounding weights/inputs get gradients, not just the block.
 if name=='conv_norm_relu':
  a=torch.randn(2,2,8,8,device='mps')*.2;kernel=torch.randn(4,2,3,3,device='mps')*.1
  v=clone((a,kernel,*make((128,4,64,2),'relu')[1:]))
  def model(mode):
   x=F.conv2d(v[0],v[1],padding=1).permute(0,2,3,1).reshape(-1,4)
   x=F.layer_norm(x,(4,));return projection_block(x,*v[2:],'relu',accuracy=mode)
 elif name=='attention_gelu':
  a=torch.randn(2,8,4,device='mps')*.2;k=torch.randn(2,8,4,device='mps')*.2;vv=torch.randn(2,8,4,device='mps')*.2
  v=clone((a,k,vv,*make((16,4,64,2),'gelu')[1:]))
  def model(mode):
   x=(torch.softmax(v[0]@v[1].transpose(-1,-2)/2,-1)@v[2]).reshape(-1,4)
   return projection_block(x,*v[3:],'gelu',accuracy=mode)
 else:
  v=make((16,4,64,2),'square')
  def model(mode):
   y=call(v,'square',mode)
   # User-defined research activation and a repeated-use branch.
   return y*torch.sin(y)+.1*y.square()+y
 out=[]
 for mode in ('strict','algebraic'):
  y=model(mode);out.append((y,*torch.autograd.grad(y.square().mean(),v)))
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 return {'graph':name,'errors':errors,'passed':all(e['passed'] for e in errors),'scope':'Hybrid PyTorch graph correctness; surrounding operations remain native, no speed claim.'}

def main():
 assert torch.backends.mps.is_available(),'MPS unavailable: never replace with CPU'
 cases=[]
 for shape in ((128,4,512,2),(512,4,2048,2),(64,32,128,8)):
  for kind in KINDS:
   print('GPU joint factors',shape,kind,flush=True)
   cases.append(case(shape,kind,2100+len(cases),timing=kind in ('identity','relu','square')))
 cases.append(case((128,4,512,2),'relu',2199,region=False))
 repeats=[]
 for seed in (2301,2302,2303):
  for kind in ('identity','relu','square'):
   print('GPU independent repeat',kind,seed,flush=True)
   repeats.append(case((512,4,2048,2),kind,seed,timing=True))
 scaling=[]
 for shape in ((1024,32,2048,1),(1024,128,4096,2),(2048,128,4096,2),(8192,128,4096,2)):
  for kind in ('identity','relu','square'):
   print('GPU scaling',shape,kind,flush=True)
   scaling.append(case(shape,kind,2400+len(scaling),timing=True))
 scaling_repeats=[]
 for seed in (2501,2502,2503):
  for kind in ('identity','relu','square'):
   print('GPU largest-shape repeat',kind,seed,flush=True)
   scaling_repeats.append(case((8192,128,4096,2),kind,seed,timing=True))
 edges=[]
 for name in ('zero','negative','noncontiguous','nan','inf','finite_overflow','small_cost_reject','unsupported_dtype','strict','frozen_weight','empty','nan_upstream','inf_upstream','large_upstream'):
  v=list(make((16,2,64,1),'square'));kind='square';mode='algebraic'
  if name=='zero':v[0]=torch.zeros_like(v[0]).requires_grad_()
  if name=='negative':v[0]=-v[0].detach().abs();v[0].requires_grad_()
  if name=='noncontiguous':v[0]=v[0].T.contiguous().T.detach().requires_grad_()
  if name in ('nan','inf'):
   with torch.no_grad():v[0][0,0]=float(name)
  if name=='finite_overflow':
   with torch.no_grad():v[0].fill_(1e30);v[1].fill_(1e30)
  if name=='small_cost_reject':v=list(make((4,32,8,4),'square'))
  if name=='unsupported_dtype':v=[t.detach().to(torch.float16).requires_grad_() for t in v]
  if name=='strict':mode='strict'
  if name=='frozen_weight':v[1]=v[1].detach()
  if name=='empty':v[0]=torch.empty(0,2,device='mps',requires_grad=True)
  ev=[];y=call(v,kind,mode,ev);ref=native_projection(*v,kind)
  ok=bool(torch.allclose(y,ref,rtol=2e-4,atol=2e-4,equal_nan=True))
  if name not in ('nan','inf','finite_overflow','empty'):
   active=[t for t in v if t.requires_grad];p=torch.randn_like(y)
   if name in ('nan_upstream','inf_upstream','large_upstream'):
    p[0,0]=float('nan') if name=='nan_upstream' else float('inf') if name=='inf_upstream' else 1e20
   g=torch.autograd.grad(y,active,p);gg=torch.autograd.grad(ref,active,p)
   ok=ok and all(bool(torch.allclose(a,b,rtol=2e-3 if name=='unsupported_dtype' else 2e-4,atol=2e-4,equal_nan=True)) for a,b in zip(g,gg))
  edges.append({'case':name,'evidence':ev,'passed':ok})
 higher_cases=[higher(k) for k in KINDS]
 graphs=[composition(name) for name in ('conv_norm_relu','attention_gelu','custom_activation_branch')]
 # Independently compare symbolic catalogue slopes against native GPU AD.
 slopes=[]
 for kind in KINDS:
  z=torch.linspace(-4,4,33,device='mps',requires_grad=True);y=activation(z,kind)
  native=torch.autograd.grad(y.sum(),z)[0];candidate=_slope(z,y,kind)
  errors=compare(candidate,native);slopes.append({'activation':kind,'errors':errors,'passed':errors['passed']})
 paths=[ROOT/'ojasx2/gradient_factor_compiler.py',Path(__file__),ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 report={'status':'completed','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,
  'cases':cases,'repeats':repeats,'scaling':scaling,'scaling_repeats':scaling_repeats,'edges':edges,'higher_derivatives':higher_cases,'composed_graphs':graphs,'primitive_slopes':slopes,
  'failures':sum(not c['passed'] for group in (cases,repeats,scaling,scaling_repeats,edges,higher_cases,graphs,slopes) for c in group),
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  'scope':'Sampled FP32 all-parameter VJP, four SGD steps, HVP and hybrid graph correctness. Seven activations, conditional factor savings; no universal optimum or IEEE bitwise proof. Memory is fresh saved-tensor storage, not peak GPU RAM.'}
 (ROOT/'reports/gradient-factor-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GPU COMPLETE',report['failures'],'failures',flush=True)

if __name__=='__main__':main()
