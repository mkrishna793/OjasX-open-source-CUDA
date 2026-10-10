"""GPU-only mathematical coordinate compiler qualification."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback before torch import')
import json,sys,math,time,statistics,random,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.coordinate_compiler import coordinate_power_chain,native_power_chain,plan_power_chain,coordinate_cost
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,clone
assert torch.backends.mps.is_available()

def inputs(m,k,degrees):
 P=math.prod(degrees)
 # Stable positive test domain with trainable weights near one.
 x=(torch.randn(m,k,device='mps')*.015/P).exp().requires_grad_()
 w=(torch.randn(len(degrees),k,device='mps')*.015/P).exp().requires_grad_()
 return x,w

def call(v,degrees,mode,evidence=None):return coordinate_power_chain(*v,degrees,accuracy=mode,evidence=evidence)
def step(v,degrees,mode):
 y=call(v,degrees,mode);loss=y.square().mean();g=torch.autograd.grad(loss,v)
 with torch.no_grad():v[1].add_(g[1],alpha=-1e-7)
 return loss,g

def case(m,k,degrees):
 v=inputs(m,k,degrees);e=[];call(v,degrees,'algebraic',e)
 upstream=torch.randn(m,k,device='mps');out=[];memory={}
 for mode in ('strict','algebraic'):
  with SavedTensorAudit(list(v)) as audit:y=call(v,degrees,mode)
  out.append((y,*torch.autograd.grad(y,v,upstream)));memory[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for i in range(4):
  la,ga=step(states['strict'],degrees,'strict');lb,gb=step(states['algebraic'],degrees,'algebraic')
  checks=[compare(lb.reshape(1),la.reshape(1)),*[compare(a,b) for a,b in zip(gb,ga)],compare(states['algebraic'][1],states['strict'][1])]
  training.append({'step':i,'passed':all(c['passed'] for c in checks),'errors':checks})
 times={mode:[] for mode in states}
 for mode in states:
  for _ in range(5):step(states[mode],degrees,mode)
 rng=random.Random(904)
 for _ in range(9):
  order=list(states);rng.shuffle(order)
  for mode in order:
   torch.mps.synchronize();start=time.perf_counter()
   for _ in range(10):step(states[mode],degrees,mode)
   torch.mps.synchronize();times[mode].append((time.perf_counter()-start)/10)
 med={mode:statistics.median(t) for mode,t in times.items()}
 return {'shape':[m,k],'degrees':list(degrees),'route':e[0],'forward_all_gradient_errors':errors,'four_step_training':training,'passed':all(c['passed'] for c in errors) and all(c['passed'] for c in training),'memory':memory,'timing':{'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['algebraic'],'method':'5 warmups; 9 shuffled rounds; 10 full forward/loss/all-gradient/SGD steps per sample. All guards and both log encodings included.'},'cost':coordinate_cost(m,k,degrees)}

def main():
 torch.manual_seed(912);cases=[]
 for m,k,degrees in [(256,64,(1,)*16),(256,64,(2,)*4),(256,64,(2,)*8),(256,64,(3,)*5),(1024,128,(2,)*4),(256,64,(2,)*12)]:
  print('GPU',m,k,degrees,flush=True);cases.append(case(m,k,degrees))
 edges=[]
 for name in ('zero','negative','nan','inf','large','output_range','unsupported','strict','noncontiguous'):
  ds=(2,)*4;v=list(inputs(32,4,ds))
  with torch.no_grad():
   if name=='zero':v[0][0,0]=0
   if name=='negative':v[1][0,0]=-1
   if name=='nan':v[0][0,0]=float('nan')
   if name=='inf':v[1][0,0]=float('inf')
   if name=='large':v[1].fill_(2)
   if name=='output_range':v[0].fill_(1.2);v[1].fill_(1.25)
   if name=='unsupported':ds=(4,)*4
  if name=='noncontiguous':v[0]=v[0].T.contiguous().T.detach().requires_grad_()
  ev=[];mode='strict' if name=='strict' else 'algebraic';y=call(v,ds,mode,ev);r=native_power_chain(*v,ds)
  eq=bool(torch.allclose(y,r,atol=2e-4,rtol=2e-4,equal_nan=True))
  expected='log_affine_action' if name=='noncontiguous' else ('native' if name=='strict' else ('output_range_fallback' if name=='output_range' else 'representation_or_numeric_fallback' if name=='unsupported' else 'domain_fallback'))
  edges.append({'case':name,'route':ev[0]['path'],'passed':eq and ev[0]['path']==expected})
 # Higher derivatives must correspond to the original nonlinear map, unlike
 # the earlier Taylor experiment which changed the primal.
 v=inputs(8,3,(2,)*4);direction=tuple(torch.randn_like(t) for t in v);hv=[]
 for mode in ('strict','algebraic'):
  g=torch.autograd.grad(call(v,(2,)*4,mode).square().mean(),v,create_graph=True)
  hv.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,direction)),v))
 higher=[compare(a,b) for a,b in zip(hv[1],hv[0])]
 paths=[ROOT/'ojasx2/coordinate_compiler.py',Path(__file__),ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 report={'status':'completed','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,'cases':cases,'edges':edges,'higher_derivatives':higher,'accuracy_failures':sum(not c['passed'] for c in cases),'edge_failures':sum(not c['passed'] for c in edges),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Positive coordinate-wise trainable power/scale chains only; no dense nonlinear layer or generic model coverage. Exact-real identities; FP32 qualification, not bit equivalence.'}
 (ROOT/'reports/coordinate-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Completed:',report['accuracy_failures'],'accuracy failures;',report['edge_failures'],'edge failures')
if __name__=='__main__':main()
