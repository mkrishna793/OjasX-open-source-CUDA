"""Hash-checked formal/GPU admission ledger, including rejected compositions.
No tensor execution on the host; a research ledger, not an automatic dispatcher.
"""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def load(name):return json.loads((ROOT/'reports'/name).read_text())
def hashes(report):
 for path,digest in report['source_sha256'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
   raise RuntimeError('Stale evidence: '+path)

def main():
 formal=load('invariant-formal-verification.json')
 gpu=load('invariant-gpu.json');composition=load('invariant-composition-gpu.json')
 discovery=load('invariant-discovery.json')
 for report in (formal,gpu,composition):hashes(report)
 if formal['status']!='passed_formal_checks':raise RuntimeError('Formal audit failed')
 for report in (gpu,composition):
  if report['failures'] or report['cpu_ml_execution'] or report['cpu_fallback_enabled'] or report['device']!='mps':
   raise RuntimeError('GPU-only qualification failed')
 for group in ('cases','repeats','edges','higher_derivatives','alternative_claim_checks'):
  if not all(c['passed'] for c in gpu[group]):raise RuntimeError('Reported GPU result inconsistent')
 if not all(c['passed'] for group in ('models','chambers','sketch') for c in composition[group]):
  raise RuntimeError('Admission result inconsistent')
 rejected=sum(not c['candidate_passed'] for c in composition['models'])
 if rejected!=composition['candidate_model_rejections']:raise RuntimeError('Rejection count inconsistent')
 for c in composition['models']:
  if c['selected_policy']!=('algebraic' if c['candidate_passed'] else 'strict'):
   raise RuntimeError('Invalid model admission')
 decisions=[]
 for c in gpu['cases']:
  matches=[a for a in gpu['repeats'] if a['shape']==c['shape'] and a['activation']==c['activation']]
  ratios=[a['timing']['native_over_candidate'] for a in matches]
  native=c['memory']['strict']['fresh_saved_storage_bytes'];candidate=c['memory']['algebraic']['fresh_saved_storage_bytes']
  eligible=c['evidence'][0].get('plan',{}).get('eligible',False)
  decisions.append({'shape':c['shape'],'activation':c['activation'],
   'repeat_speed_ratios':ratios,'single_run_speed_ratio':c['timing']['native_over_candidate'],
   'speed_admitted':eligible and len({a['seed'] for a in matches})>=3 and min(ratios)>1.05,
   'memory_admitted':candidate<native,'fresh_saved_native_bytes':native,
   'fresh_saved_candidate_bytes':candidate,'saved_storage_reduction':1-candidate/native if native else 0})
 models=[]
 for c in composition['models']:
  matches=[a for a in composition['models'] if a['shape']==c['shape'] and a['model']==c['model']]
  ratios=[a['timing']['native_over_candidate'] for a in matches]
  models.append({'model':c['model'],'shape':c['shape'],'seed':c['seed'],
   'candidate_passed':c['candidate_passed'],'selected_policy':c['selected_policy'],
   'repeat_speed_ratios':ratios,'speed_admitted':all(a['candidate_passed'] for a in matches) and len({a['seed'] for a in matches})>=3 and min(ratios)>1.05,
   'policy_scope':c['policy_scope']})
 result={'status':'qualified_scoped_invariant_rules_with_rejection_ledger',
  'lean_theorems':formal['lean_theorems'],'rocq_statements':formal['rocq_statements'],
  'new_z3_checks':formal['new_z3_checks'],'shape_decisions':len(discovery['shapes']),
  'gpu_checks':{g:len(gpu[g]) for g in ('cases','repeats','edges','higher_derivatives','alternative_claim_checks')},
  'composition_gpu_checks':{g:len(composition[g]) for g in ('models','chambers','sketch')},
  'gpu_rule_failures':gpu['failures'],'composition_admission_failures':composition['failures'],
  'raw_candidate_model_rejections':rejected,'added_axioms':[],'cpu_ml_execution':False,
  'block_decisions':decisions,'model_decisions':models,
  'chamber_speed_admitted':False,
  'chamber_repeat_speed_ratios':[a['timing']['native_over_candidate'] for a in composition['chambers'] if a['shape']==[8192,128,4096,2]],
  'universal_backprop_reduction_established':False,'unchanged_dense_gelu_silu_compression_established':False,
  'source_sha256':{'reports/'+n:hashlib.sha256((ROOT/'reports'/n).read_bytes()).hexdigest() for n in ('invariant-formal-verification.json','invariant-gpu.json','invariant-composition-gpu.json','invariant-discovery.json')},
  'scope':'Existing radial architectures with all original parameters. Exact-real mathematical proofs plus sampled FP32 MPS tests; repeated eager-baseline performance in listed shapes. Whole-program preflight preserves rejected candidate results and selects native calculation. No automatic graph classifier or formal IEEE/Python/Metal refinement.'}
 (ROOT/'reports/invariant-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('block_decisions','model_decisions','source_sha256')},indent=2))

if __name__=='__main__':main()
