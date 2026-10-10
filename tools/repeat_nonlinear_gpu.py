"""GPU-only repeat timings; includes all data guards and full SGD steps."""
import os,sys,json,time,statistics,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
sys.path.insert(0,str(ROOT/'tools'))
import search_nonlinear_gpu as s
import torch
assert torch.backends.mps.is_available()
results=[]
for seed in (921,922,923):
 torch.manual_seed(seed)
 for kind in ('gelu','silu','swiglu'):
  vals=s.make_inputs((512,2,4096,8),kind,'small')
  states={'native':s.clone(vals),'candidate':s.clone(vals)};times={name:[] for name in states}
  for name,state in states.items():
   for _ in range(12):s.step(state,kind,name=='candidate')
  rng=random.Random(seed)
  for _ in range(11):
   order=list(states);rng.shuffle(order)
   for name in order:
    torch.mps.synchronize();start=time.perf_counter()
    for _ in range(20):s.step(states[name],kind,name=='candidate')
    torch.mps.synchronize();times[name].append((time.perf_counter()-start)/20)
  med={name:statistics.median(v) for name,v in times.items()}
  result={'seed':seed,'kind':kind,'shape':[512,2,4096,8],'seconds_median':med,'native_over_candidate':med['native']/med['candidate'],'samples':times}
  results.append(result);print(seed,kind,result['native_over_candidate'],flush=True)
open(ROOT/'reports/nonlinear-gpu-repeat.json','w').write(json.dumps({'device':'mps','cpu_ml_execution':False,'cpu_fallback_enabled':False,'method':'3 independent seeds; 12 warmups, 11 shuffled rounds of 20 full steps per sample.','cases':results,'source_sha256':s.source_hashes()},indent=2)+'\n')
