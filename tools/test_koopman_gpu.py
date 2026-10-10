"""GPU-only finite Koopman compiler experiment; all shared parameter gradients."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback')
from pathlib import Path
import sys,json,hashlib,time,statistics,random
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.coordinate_compiler import coordinate_koopman,native_koopman
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,clone
assert torch.backends.mps.is_available()

def make(m,k):
 x=torch.randn(m,k,2,device='mps')*.25
 theta=torch.stack((.99+.01*torch.rand(k,device='mps'),.99+.01*torch.rand(k,device='mps'),torch.randn(k,device='mps')*.005))
 return clone((x,theta))
def call(v,depth,mode,ev=None):return coordinate_koopman(*v,depth,accuracy=mode,evidence=ev)
def step(v,depth,mode):
 y=call(v,depth,mode);loss=y.square().mean();g=torch.autograd.grad(loss,v)
 with torch.no_grad():v[1].add_(g[1],alpha=-1e-6)
 return loss,g

def case(m,k,depth,shared=True):
 v=make(m,k)
 if not shared:v=(v[0],torch.stack([make(1,k)[1].detach() for _ in range(depth)]).requires_grad_())
 ev=[];call(v,depth,'algebraic',ev);g=torch.randn_like(v[0]);outs=[];memory={}
 for mode in ('strict','algebraic'):
  with SavedTensorAudit(list(v)) as audit:y=call(v,depth,mode)
  outs.append((y,*torch.autograd.grad(y,v,g)));memory[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(outs[1],outs[0])]
 states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for i in range(4):
  la,ga=step(states['strict'],depth,'strict');lb,gb=step(states['algebraic'],depth,'algebraic')
  es=[compare(lb.reshape(1),la.reshape(1)),*[compare(a,b) for a,b in zip(gb,ga)],compare(states['algebraic'][1],states['strict'][1])]
  training.append({'step':i,'passed':all(e['passed'] for e in es),'errors':es})
 times={mode:[] for mode in states};rng=random.Random(990)
 for mode in states:
  for _ in range(5):step(states[mode],depth,mode)
 for _ in range(9):
  order=list(states);rng.shuffle(order)
  for mode in order:
   torch.mps.synchronize();start=time.perf_counter()
   for _ in range(5):step(states[mode],depth,mode)
   torch.mps.synchronize();times[mode].append((time.perf_counter()-start)/5)
 med={mode:statistics.median(v) for mode,v in times.items()}
 return {'shape':[m,k,2],'depth':depth,'shared_weights':shared,'route':ev[0],'errors':errors,'four_step_sgd':training,'passed':all(e['passed'] for e in errors) and all(e['passed'] for e in training),'memory':memory,'timing':{'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['algebraic'],'method':'5 warmups, 9 shuffled rounds, 5 full forward/loss/all-gradient/weight-SGD steps per sample; domain check and matrix construction included.'}}

def main():
 torch.manual_seed(980);cases=[]
 for args in ((128,16,8),(512,32,32),(512,32,64),(512,32,32,False),(512,32,64,False)):
  print('GPU Koopman',args,flush=True);cases.append(case(*args))
 edges=[]
 for name in ('zero','negative','equal_eigenvalues','zero_c','nan','large','depth_guard','strict','untied_odd','noncontiguous'):
  v=list(make(16,4));depth=8
  with torch.no_grad():
   if name=='zero':v[0].zero_()
   if name=='negative':v[0].neg_();v[1][0].neg_()
   if name=='equal_eigenvalues':v[1][1].copy_(v[1][0].square())
   if name=='zero_c':v[1][2].zero_()
   if name=='nan':v[0][0,0,0]=float('nan')
   if name=='large':v[1][0].fill_(1.1)
   if name=='depth_guard':depth=65
  if name=='untied_odd':depth=7;v[1]=torch.stack([make(1,4)[1].detach() for _ in range(depth)]).requires_grad_()
  if name=='noncontiguous':v[0]=v[0].transpose(0,1).contiguous().transpose(0,1).detach().requires_grad_()
  mode='strict' if name=='strict' else 'algebraic';ev=[];y=call(v,depth,mode,ev);r=native_koopman(*v,depth)
  same=bool(torch.allclose(y,r,rtol=2e-4,atol=2e-4,equal_nan=True))
  if name not in ('nan','large','depth_guard','strict','untied_odd','noncontiguous'):
   g=torch.randn_like(y);a=torch.autograd.grad(y,v,g);b=torch.autograd.grad(r,v,g);same=same and all(compare(a0,b0)['passed'] for a0,b0 in zip(a,b))
  expected='domain_fallback' if name in ('nan','large') else 'representation_or_numeric_fallback' if name=='depth_guard' else 'native' if name=='strict' else 'finite_koopman'
  edges.append({'case':name,'route':ev[0]['path'],'passed':same and ev[0]['path']==expected})
 v=make(8,3);directions=tuple(torch.randn_like(t) for t in v);hs=[]
 for mode in ('strict','algebraic'):
  g=torch.autograd.grad(call(v,8,mode).square().mean(),v,create_graph=True)
  hs.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,directions)),v))
 higher=[compare(a,b) for a,b in zip(hs[1],hs[0])]
 paths=[ROOT/'ojasx2/coordinate_compiler.py',Path(__file__),ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 r={'status':'completed','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,'cases':cases,'edges':edges,'higher_derivatives':higher,'accuracy_failures':sum(not c['passed'] for c in cases),'edge_failures':sum(not c['passed'] for c in edges),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Exact-real finite invariant subspace for a trainable triangular nonlinear recurrence, preserving shared or per-stage original parameters. Sampled FP32 tolerance; no generic dense activation or total peak-memory guarantee.'}
 (ROOT/'reports/koopman-gpu.json').write_text(json.dumps(r,indent=2)+'\n')
 print('GPU completed',r['accuracy_failures'],'accuracy failures;',r['edge_failures'],'edge failures')

if __name__=="__main__":main()
