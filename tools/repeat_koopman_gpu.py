"""Independent GPU-only repeats of the per-stage-parameter Koopman compiler."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Disable CPU fallback')
from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import test_koopman_gpu as t
assert torch.backends.mps.is_available()
cases=[]
for seed in (1001,1002,1003):
 torch.manual_seed(seed)
 for depth in (32,64):
  c=t.case(512,32,depth,False);c['seed']=seed;cases.append(c)
  print(seed,depth,c['passed'],c['timing']['native_over_candidate'],flush=True)
paths=[ROOT/'ojasx2/coordinate_compiler.py',ROOT/'tools/test_koopman_gpu.py',Path(__file__),ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py']
r={'status':'completed','cpu_ml_execution':False,'cpu_fallback_enabled':False,'device':'mps','cases':cases,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Three independent seeds; separate original trainable parameters at every stage; full gradients and full SGD-step timing. Retained activation storage, not total peak GPU memory.'}
(ROOT/'reports/koopman-repeat.json').write_text(json.dumps(r,indent=2)+'\n')
