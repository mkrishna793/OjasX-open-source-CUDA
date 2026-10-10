"""GPU-only rejection/domain/conditioning tests; no CPU ML reference."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('GPU fallback must be disabled')
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import search_nonlinear_gpu as s
from ojasx2.nonlinear_taylor_gpu import taylor_block,native_block
assert torch.backends.mps.is_available()
results=[]
for kind in ('gelu','silu','swiglu'):
 for width in (1,2,4):
  r=s.qualify((32,width,256,2),kind,'small',False)
  results.append({'name':kind+'_width_'+str(width),'passed':r['correctness_passed'],'details':r})
 for scenario in ('boundary_inside','boundary_outside','nan','inf','coefficient_overflow','cancellation'):
  vals=list(s.make_inputs((32,2,256,2),kind,'small'))
  with torch.no_grad():
   if scenario.startswith('boundary'):vals[1].mul_((.049 if scenario.endswith('inside') else .051)/.03)
   if scenario=='nan':vals[0][0,0]=float('nan')
   if scenario=='inf':vals[1][0,0]=float('inf')
   if scenario=='coefficient_overflow':vals[0].zero_();vals[1].fill_(1e20)
   if scenario=='cancellation':vals[0].fill_(1000);vals[1][0].fill_(1000);vals[1][1].fill_(-1000)
  evidence=[];y=taylor_block(*vals[:3],vals[3] if len(vals)==4 else None,kind,evidence=evidence)
  native=native_block(*vals[:3],vals[3] if len(vals)==4 else None,kind)
  route=evidence[0]['path'];expected='approximate_feature' if scenario=='boundary_inside' else 'domain_fallback'
  same=bool(torch.allclose(y,native,rtol=s.RTOL,atol=s.ATOL,equal_nan=True))
  results.append({'name':kind+'_'+scenario,'route':route,'passed':route==expected and same})
 vals=s.make_inputs((32,2,256,2),kind,'small')
 for delta in (0,-1,.051,float('nan')):
  try:taylor_block(*vals[:3],vals[3] if len(vals)==4 else None,kind,delta=delta)
  except ValueError:ok=True
  else:ok=False
  results.append({'name':kind+'_invalid_delta_'+str(delta),'passed':ok})
 try:taylor_block(*(v.half() for v in vals[:3]),vals[3].half() if len(vals)==4 else None,kind)
 except ValueError:ok=True
 else:ok=False
 results.append({'name':kind+'_reject_fp16','passed':ok})
 with torch.autocast(device_type='mps',dtype=torch.float16):
  try:taylor_block(*vals[:3],vals[3] if len(vals)==4 else None,kind)
  except ValueError:ok=True
  else:ok=False
 results.append({'name':kind+'_reject_autocast','passed':ok})
report={'status':'passed' if all(r['passed'] for r in results) else 'failed','device':'mps','cpu_fallback_enabled':False,'cpu_ml_execution':False,'count':len(results),'cases':results,'source_sha256':s.source_hashes()}
(ROOT/'reports/nonlinear-gpu-edges.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],report['count'])
if report['status']!='passed':raise SystemExit(1)
