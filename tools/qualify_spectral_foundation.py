"""Source-matched formal/GPU qualification; approximation rejection is explicit."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def load(n):return json.loads((ROOT/'reports'/n).read_text())
def hashes(r):
 for n,d in r['source_sha256'].items():
  if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=d:raise RuntimeError('Stale evidence '+n)

def main():
 f=load('spectral-formal-verification.json');g=load('spectral-gpu.json');d=load('spectral-discovery.json')
 hashes(f);hashes(g)
 if f['status']!='passed_formal_checks' or g['device']!='mps' or g['cpu_ml_execution'] or g['cpu_fallback_enabled']:raise RuntimeError('Proof/GPU contract failed')
 for k in ('polynomial_failures','admission_failures','edge_failures','higher_derivative_failures','model_admission_failures','scalar_approximation_failures'):
  if g[k]:raise RuntimeError(k+' failed')
 if any(not c['polynomial_passed'] or not c['admission_passed'] for c in g['cases']+g['repeats']):raise RuntimeError('Incorrect ledger')
 for c in g['cases']+g['repeats']:
  selected='approximate' if c['degree'] in (8,12) and c['native_approximation_passed'] else 'strict'
  if c['selected_policy']!=selected:raise RuntimeError('Invalid approximation admission')
 decisions=[];seen=set()
 for c in g['repeats']:
  key=(tuple(c['shape']),c['activation'],c['degree'])
  if key in seen:continue
  seen.add(key);rr=[v for v in g['repeats'] if (tuple(v['shape']),v['activation'],v['degree'])==key]
  ratios=[v['timing']['native_over_candidate'] for v in rr]
  native=c['memory']['strict']['fresh_saved_storage_bytes'];candidate=c['memory']['polynomial']['fresh_saved_storage_bytes']
  decisions.append({'shape':c['shape'],'activation':c['activation'],'degree':c['degree'],
   'all_original_gradient_and_four_step_sgd_passed':all(v['native_approximation_passed'] for v in rr),
   'repeat_speed_ratios':ratios,'speed_admitted':len({v['seed'] for v in rr})>=3 and min(ratios)>1.05 and all(v['selected_policy']=='approximate' for v in rr),
   'fresh_saved_native_bytes':native,'fresh_saved_candidate_bytes':candidate,'saved_storage_reduction':1-candidate/native,
   'memory_admitted':candidate<native and all(v['selected_policy']=='approximate' for v in rr)})
 models=[{'seed':c['seed'],'model':c['model'],'candidate_passed':c['candidate_passed'],'selected_policy':c['selected_policy'],'admission_passed':c['admission_passed']} for c in g['composed_models']]
 result={'status':'qualified_conditional_coordinate_polynomial_rules','lean_theorems':f['lean_theorems'],'rocq_statements':f['rocq_statements'],
  'new_z3_checks':f['new_z3_checks'],'rank_degree_decisions':len(d['shape_search']),
  'gpu_checks':{k:len(g[k]) for k in ('scalar_approximations','cases','repeats','edges','higher_derivatives','composed_models')},
  'polynomial_failures':0,'admission_failures':0,'raw_native_approximation_rejections':g['raw_native_approximation_rejections'],
  'raw_model_rejections':g['raw_model_rejections'],'decisions':decisions,'models':models,
  'low_degree_hermite_claim_accepted':False,'added_axioms':[],'cpu_ml_execution':False,
  'unchanged_native_activation_identity':False,'universal_or_most_model_reduction':False,
  'source_sha256':{'reports/'+n:hashlib.sha256((ROOT/'reports'/n).read_bytes()).hexdigest() for n in ('spectral-formal-verification.json','spectral-gpu.json','spectral-discovery.json')},
  'scope':'Exact polynomial representation and matched gradients, kernel-checked Chebyshev tail versus polynomial reference, conditional activation remainder/error transport; sampled bounded-domain FP32 GPU accuracy. Speed admission requires repeated wins; no automatic graph/optimizer certification, global activation equivalence or IEEE proof.'}
 (ROOT/'reports/spectral-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','decisions','models')},indent=2))

if __name__=='__main__':main()
