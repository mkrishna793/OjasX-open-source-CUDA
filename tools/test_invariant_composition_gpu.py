"""MPS-only stacked radial models, certified chambers and sketch errors."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback')
from pathlib import Path
import sys,json,random,time,statistics,hashlib,math
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.invariant_radial_compiler import invariant_radial
from ojasx2.chamber_compiler import chamber_projection
from ojasx2.gradient_factor_compiler import native_projection
from test_invariant_gpu import make
from search_nonlinear_gpu import compare,clone
from ojasx2.residuals import SavedTensorAudit

def timing(states,step,seed):
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
 return {'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['algebraic'],
  'scope':'Full forward/loss/all-input-and-original-parameter gradients/SGD; guards included; eager same-model baseline.'}

def stacked_model(seed,normalization=True):
 m,n=(2048,2048) if normalization else (8192,4096)
 torch.manual_seed(seed);v1=make((m,32,n,32),'sigmoid');v2=make((m,32,n,8),'reciprocal')
 v=clone((*v1,*v2[1:]));labels=torch.arange(m,device='mps')%8
 def model(v,mode,ev=None):
  h=invariant_radial(*v[:4],'sigmoid',gate=v[4],accuracy=mode,evidence=ev)
  if normalization:h=F.layer_norm(h,(32,))
  return invariant_radial(h,*v[5:],'reciprocal',accuracy=mode,evidence=ev)
 out=[];mem={};ev=[]
 for mode in ('strict','algebraic'):
  with SavedTensorAudit(list(v)) as audit:h=model(v,mode,ev if mode=='algebraic' else None)
  loss=F.cross_entropy(h,labels);out.append((h,loss.reshape(1),*torch.autograd.grad(loss,v)));mem[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 def step(v,mode):
  loss=F.cross_entropy(model(v,mode),labels);g=torch.autograd.grad(loss,v)
  with torch.no_grad():
   for t,gg in zip(v[1:],g[1:]):t.add_(gg,alpha=-1e-4)
  return loss,g
 states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for it in range(4):
  a,ga=step(states['strict'],'strict');b,gb=step(states['algebraic'],'algebraic')
  es=[compare(b.reshape(1),a.reshape(1)),*[compare(x,y) for x,y in zip(gb,ga)],*[compare(x,y) for x,y in zip(states['algebraic'][1:],states['strict'][1:])]]
  training.append({'step':it,'errors':es,'passed':all(e['passed'] for e in es)})
 candidate_passed=all(e['passed'] for e in errors) and all(t['passed'] for t in training)
 # Qualification refines program admission, not the old tolerances. A rejected
 # composition uses the original whole program; isolated passing blocks alone
 # are insufficient evidence for a sensitive composed loss.
 selected='algebraic' if candidate_passed else 'strict'
 safe_h=model(v,selected);ref_h=model(v,'strict')
 safe_loss=F.cross_entropy(safe_h,labels);ref_loss=F.cross_entropy(ref_h,labels)
 safe_g=torch.autograd.grad(safe_loss,v);ref_g=torch.autograd.grad(ref_loss,v)
 admission_errors=[compare(safe_h,ref_h),*[compare(a,b) for a,b in zip(safe_g,ref_g)]]
 return {'model':'two_radial_blocks_'+('layernorm_' if normalization else '')+'crossentropy','shape':[m,32,n,32,8],
  'seed':seed,'evidence':ev,'errors':errors,'four_step_sgd':training,'candidate_passed':candidate_passed,
  'selected_policy':selected,'admission_errors':admission_errors,'policy_scope':'Research preflight for this supplied model; no universal runtime error certification or automatic graph classifier.',
  'memory':mem,'timing':timing(states,step,seed),'passed':all(e['passed'] for e in admission_errors),
  'scope':'A complete small radial research architecture; no model-quality/convergence or standard-transformer speed claim.'}

def chamber_case(groups,seed,large=False,invalid=False):
 torch.manual_seed(seed);m,k,n,r=(8192,128,4096,2) if large else (128,8,512,2)
 bits=int(math.log2(groups));ids=torch.arange(m,device='mps')%groups
 templates_ids=torch.arange(groups,device='mps')[:,None]
 feature_bits=torch.arange(n,device='mps')%bits
 patterns=((templates_ids//(2**feature_bits[None,:]))%2)==1
 x=torch.randn(m,k,device='mps')*.01
 for d in range(bits):x[:,d]=torch.where(((ids//(2**d))%2)==1,1.,-1.)
 w=torch.zeros(k,n,device='mps');w[feature_bits,torch.arange(n,device='mps')]=1
 v=clone((x,w,torch.zeros(n,device='mps'),torch.randn(n,r,device='mps')/math.sqrt(n)))
 if invalid:patterns=patterns.clone();patterns[0,0]=~patterns[0,0]
 def model(v,mode,ev=None):return chamber_projection(*v,ids,patterns,accuracy=mode,evidence=ev)
 out=[];ev=[];mem={};up=torch.randn(m,r,device='mps')
 for mode in ('strict','algebraic'):
  with SavedTensorAudit([*v,ids,patterns]) as audit:h=model(v,mode,ev if mode=='algebraic' else None)
  out.append((h,*torch.autograd.grad(h,v,up)));mem[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 target=torch.randn(m,r,device='mps')*.1
 def step(v,mode):
  loss=(model(v,mode)-target).square().mean();g=torch.autograd.grad(loss,v)
  with torch.no_grad():
   for t,gg in zip(v[1:],g[1:]):t.add_(gg,alpha=-1e-4)
  return loss,g
 states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for it in range(4):
  a,ga=step(states['strict'],'strict');b,gb=step(states['algebraic'],'algebraic')
  es=[compare(b.reshape(1),a.reshape(1)),*[compare(x,y) for x,y in zip(gb,ga)]]
  training.append({'step':it,'passed':all(e['passed'] for e in es)})
 return {'chambers':groups,'shape':[m,k,n,r],'seed':seed,'invalid_certificate':invalid,'evidence':ev,'errors':errors,'four_step_sgd':training,'memory':mem,
  'timing':timing(states,step,seed),'passed':all(e['passed'] for e in errors) and all(t['passed'] for t in training) and ev[0]['path']==('native_invalid_chamber' if invalid else 'exact_chamber'),
  'scope':'Constructed datasets with exact finite mask chambers; original coordinate ReLU and all W,b,Q gradients unchanged. No automatic clustering cost included, certificates are supplied and validated.'}

def sketch_case():
 v=torch.tensor([0.,1],device='mps');s=torch.tensor([[1.,0]],device='mps');w=torch.tensor([0.,1],device='mps')
 exact=(v*w).sum();restored=(s.T@(s@v));wrong=(restored*w).sum()
 error=torch.linalg.vector_norm(v-restored).item()
 return {'case':'thin_sketch_has_nonzero_gradient_contraction_error','exact':exact.item(),'sketched':wrong.item(),
  'residual_norm':error,'passed':exact.item()==1 and wrong.item()==0 and error==1,'scope':'Trace duality does not make a noninjective sketch exact; requires a proved subspace or error bound.'}

def aligned_case():
 z=torch.ones(1024,device='mps',requires_grad=True)
 y=z.square().mean()*z;exact=torch.autograd.grad(y,z,z.detach())[0]
 dropped=z.detach()
 error=torch.linalg.vector_norm(exact-dropped)/torch.linalg.vector_norm(exact)
 return {'case':'aligned_radial_correction_does_not_vanish','dimension':1024,'relative_error_if_dropped':error.item(),
  'passed':bool(torch.allclose(exact,3*z)) and abs(error.item()-2/3)<1e-5}

def main():
 assert torch.backends.mps.is_available()
 models=[];chambers=[]
 for seed in (3301,3302,3303):
  print('GPU stacked radial model',seed,flush=True);models.append(stacked_model(seed))
 for seed in (3311,3312,3313):
  print('GPU large stacked radial model',seed,flush=True);models.append(stacked_model(seed,normalization=False))
 for groups in (2,4,8):
  print('GPU chamber',groups,flush=True);chambers.append(chamber_case(groups,3400+groups))
 chambers.append(chamber_case(4,3450,invalid=True))
 for seed in (3501,3502,3503):
  print('GPU large chamber repeat',seed,flush=True);chambers.append(chamber_case(4,seed,large=True))
 sketch=sketch_case();aligned=aligned_case()
 paths=[Path(__file__),ROOT/'tools/test_invariant_gpu.py',ROOT/'ojasx2/invariant_radial_compiler.py',ROOT/'ojasx2/chamber_compiler.py',ROOT/'ojasx2/gradient_factor_compiler.py',ROOT/'tools/search_nonlinear_gpu.py',ROOT/'ojasx2/residuals.py']
 report={'status':'completed_with_explicit_admission_policy','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,'models':models,'chambers':chambers,'sketch':[sketch,aligned],
  'candidate_model_rejections':sum(not c['candidate_passed'] for c in models),
  'failures':sum(not c['passed'] for group in (models,chambers,[sketch,aligned]) for c in group),
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
 (ROOT/'reports/invariant-composition-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GPU composition COMPLETE',report['failures'],'failures',flush=True)

if __name__=='__main__':main()
