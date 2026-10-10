"""MPS-only radial model/compiler and alternative-claim qualification."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback')
from pathlib import Path
import sys,json,math,time,statistics,random,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.invariant_radial_compiler import invariant_radial,native_radial,KINDS
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,clone

def make(shape,kind):
 m,k,n,r=shape
 values=(torch.randn(m,k,device='mps')*.2,torch.randn(k,n,device='mps')/math.sqrt(k),
         torch.randn(n,device='mps')*.05,torch.randn(n,r,device='mps')/math.sqrt(n))
 if kind=='sigmoid':values+= (torch.tensor([.7,-.2],device='mps'),)
 return clone(values)

def call(v,kind,mode,ev=None):return invariant_radial(*v[:4],kind,gate=v[4] if len(v)==5 else None,accuracy=mode,evidence=ev)
def step(v,kind,mode,target):
 h=call(v,kind,mode);loss=(h-target).square().mean();g=torch.autograd.grad(loss,v)
 with torch.no_grad():
  for a,b in zip(v[1:],g[1:]):a.add_(b,alpha=-1e-4)
 return loss,g

def case(shape,kind,seed,timing=True):
 torch.manual_seed(seed);v=make(shape,kind);p=torch.randn(shape[0],shape[3],device='mps');ev=[];outs=[];memory={}
 for mode in ('strict','algebraic'):
  with SavedTensorAudit(list(v)) as audit:h=call(v,kind,mode,ev if mode=='algebraic' else None)
  outs.append((h,*torch.autograd.grad(h,v,p)));memory[mode]=audit.summary()
 errors=[compare(a,b) for a,b in zip(outs[1],outs[0])]
 target=torch.randn(shape[0],shape[3],device='mps')*.1;states={mode:clone(v) for mode in ('strict','algebraic')};training=[]
 for it in range(4):
  la,ga=step(states['strict'],kind,'strict',target);lb,gb=step(states['algebraic'],kind,'algebraic',target)
  es=[compare(lb.reshape(1),la.reshape(1)),*[compare(a,b) for a,b in zip(gb,ga)],*[compare(a,b) for a,b in zip(states['algebraic'][1:],states['strict'][1:])]]
  training.append({'step':it,'errors':es,'passed':all(e['passed'] for e in es)})
 result={'shape':shape,'activation':kind,'seed':seed,'evidence':ev,'errors':errors,'memory':memory,'four_step_sgd':training,'passed':all(e['passed'] for e in errors) and all(t['passed'] for t in training)}
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
  med={mode:statistics.median(v) for mode,v in times.items()}
  result['timing']={'seconds_median':med,'samples':times,'native_over_candidate':med['strict']/med['algebraic'],
   'scope':'Original eager radial model, full forward/loss/all-original-parameter gradients/SGD; numerical and conditioning guards included; 4 warmups, 7 shuffled rounds, 3 steps/sample.'}
 return result

def higher(kind):
 v=make((8,2,64,2),kind);directions=[torch.randn_like(t) for t in v];out=[];ev=[]
 for mode in ('strict','algebraic'):
  g=torch.autograd.grad(call(v,kind,mode,ev).square().mean(),v,create_graph=True)
  out.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,directions)),v))
 errors=[compare(a,b) for a,b in zip(out[1],out[0])]
 return {'activation':kind,'evidence':ev,'errors':errors,'passed':all(e['passed'] for e in errors)}

def alternative_checks():
 # No CPU numerical reference: all tensor counterexamples execute on MPS.
 states=torch.tensor([[1.,0,0],[1,1,0],[1,0,1]],device='mps',requires_grad=True)
 native=(states.square().sum(1,keepdim=True)*states)
 g=torch.tensor([[1.,0,0]]*3,device='mps');delta=torch.autograd.grad(native,states,g)[0]
 expected=torch.tensor([[3.,0,0],[4,2,0],[4,0,2]],device='mps')
 rank={'case':'row_specific_radial_rank_growth','passed':compare(delta,expected)['passed'],
       'incoming_rank':1,'outgoing_rank':3,'determinant':12,'scope':'Ranks independently computed symbolically; GPU VJP validates the concrete witness.'}
 k=torch.tensor([[0.,1],[-1,0]],device='mps',requires_grad=True)
 z=torch.randn(2,2,device='mps',requires_grad=True);up=torch.randn_like(z)
 f=k@z-z@k;grad=torch.autograd.grad(f,(z,k),up)
 explicit=(k.T@up-up@k.T,up@z.T-z.T@up)
 comm={'case':'commutator_all_original_gradients','passed':all(compare(a,b)['passed'] for a,b in zip(grad,explicit)),
       'scope':'Fixed K map is linear; learnable K receives its own gradient.'}
 wrong=torch.tensor([-1.,0],device='mps');radial=torch.relu(wrong.norm())*wrong/wrong.norm()
 model={'case':'radial_relu_changes_coordinate_model','passed':bool(torch.equal(radial,wrong)) and not bool(torch.equal(radial,torch.relu(wrong)))}
 concentrations=[]
 for d in (32,128,512,2048):
  u=torch.randn(512,d,device='mps');v=torch.randn_like(u);u=u/u.norm(dim=1,keepdim=True);v=v/v.norm(dim=1,keepdim=True)
  moment=(u*v).sum(1).square().mean().item()
  aligned=(u*u).sum(1).square().mean().item()
  concentrations.append({'dimension':d,'independent_second_moment':moment,'isotropic_theory':1/d,'aligned_second_moment':aligned,
   'passed':abs(moment*d-1)<.5 and abs(aligned-1)<2e-5,'scope':'Empirical statistical illustration, not a universal orthogonality or trained-feature guarantee.'})
 return [rank,comm,model,*concentrations]

def main():
 assert torch.backends.mps.is_available(),'Never fall back to CPU'
 cases=[]
 for shape in ((128,4,512,2),(1024,32,2048,4),(2048,128,4096,8),(8192,128,4096,2)):
  for kind in KINDS:
   print('GPU invariant',shape,kind,flush=True);cases.append(case(shape,kind,3100+len(cases)))
 repeats=[]
 for seed in (3201,3202,3203):
  for kind in KINDS:
   print('GPU invariant repeat',kind,seed,flush=True);repeats.append(case((8192,128,4096,2),kind,seed))
 edges=[]
 for name in ('zero','negative','noncontiguous','nan','inf','large','cost_reject','strict','cancellation','near_cancellation','frozen','nan_upstream','large_upstream','empty'):
  v=list(make((16,2,64,2),'reciprocal'));mode='algebraic';kind='reciprocal'
  with torch.no_grad():
   if name=='zero':v[0].zero_();v[2].zero_()
   if name=='negative':v[0].neg_()
   if name=='nan':v[0][0,0]=float('nan')
   if name=='inf':v[0][0,0]=float('inf')
   if name=='large':v[0].fill_(1e30)
   if name in ('cancellation','near_cancellation'):
    v[0].fill_(1);v[1].fill_(.1);v[2].fill_(-.2 if name=='cancellation' else -.2+1e-6)
  if name=='noncontiguous':v[0]=v[0].T.contiguous().T.detach().requires_grad_()
  if name=='cost_reject':v=list(make((4,32,8,4),kind))
  if name=='strict':mode='strict'
  if name=='frozen':v[1]=v[1].detach()
  if name=='empty':v[0]=torch.empty(0,2,device='mps',requires_grad=True)
  ev=[];h=call(v,kind,mode,ev);ref=call(v,kind,'strict')
  passed=bool(torch.allclose(h,ref,rtol=2e-4,atol=2e-4,equal_nan=True))
  if name!='empty':
   active=[t for t in v if t.requires_grad];p=torch.randn_like(h)
   if name=='nan_upstream':p[0,0]=float('nan')
   if name=='large_upstream':p[0,0]=1e20
   a=torch.autograd.grad(h,active,p);b=torch.autograd.grad(ref,active,p)
   passed=passed and all(bool(torch.allclose(x,y,rtol=2e-4,atol=2e-4,equal_nan=True)) for x,y in zip(a,b))
  if name in ('cancellation','near_cancellation'):passed=passed and ev[0]['path']=='native_gram_condition'
  edges.append({'case':name,'evidence':ev,'passed':passed})
 higher_cases=[higher(k) for k in KINDS];claims=alternative_checks()
 paths=[ROOT/'ojasx2/invariant_radial_compiler.py',Path(__file__),ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
 report={'status':'completed','device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,
  'cases':cases,'repeats':repeats,'edges':edges,'higher_derivatives':higher_cases,'alternative_claim_checks':claims,
  'failures':sum(not c['passed'] for group in (cases,repeats,edges,higher_cases,claims) for c in group),
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  'scope':'Exact-real radial model rewrite, sampled FP32 accuracy, every original parameter gradient and four SGD steps. Existing radial architecture only; no replacement equivalence for ordinary GELU/SiLU. Saved activation storage is not peak GPU RAM.'}
 (ROOT/'reports/invariant-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GPU invariant COMPLETE',report['failures'],'failures',flush=True)

if __name__=='__main__':main()
