"""Hash-checked qualification and performance rejection; host JSON only."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]

def load(name):return json.loads((ROOT/'reports'/name).read_text())
def hashes(report):
 for path,digest in report['source_sha256'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Stale evidence: '+path)

def main():
 formal=load('gradient-formal-verification.json');gpu=load('gradient-factor-gpu.json');discovery=load('gradient-factor-discovery.json')
 hashes(formal);hashes(gpu)
 if formal['status']!='passed_formal_checks' or gpu['failures']!=0:raise RuntimeError('Qualification failed')
 if gpu['cpu_ml_execution'] or gpu['cpu_fallback_enabled'] or gpu['device']!='mps':raise RuntimeError('GPU-only requirement failed')
 decisions=[]
 # The output is a research admission ledger, not a GPU dispatch override.
 # Repeated wins are marked separately; one favorable timing is insufficient.
 for case in gpu['cases']+gpu['scaling']:
  if 'timing' not in case:continue
  key=(case['activation'],tuple(case['shape']))
  repeats=[c for c in gpu['repeats']+gpu['scaling_repeats'] if (c['activation'],tuple(c['shape']))==key]
  ratios=[c['timing']['native_over_candidate'] for c in repeats]
  native=case['memory']['strict']['fresh_saved_storage_bytes'];candidate=case['memory']['algebraic']['fresh_saved_storage_bytes']
  eligible=case['evidence'][0].get('plan',{}).get('eligible',False)
  decisions.append({'shape':case['shape'],'activation':case['activation'],'algebraic_candidate':eligible,
   'single_run_speed_ratio':case['timing']['native_over_candidate'],'repeat_speed_ratios':ratios,
   'speed_admitted':eligible and len(ratios)>=3 and min(ratios)>1.05,
   'memory_admitted':candidate<native,'fresh_saved_native_bytes':native,'fresh_saved_candidate_bytes':candidate,
   'performance_refinement':'Native recommended for speed if repeated speed condition is absent or failed; experimental memory policy may accept separately.'})
 groups=('cases','repeats','scaling','scaling_repeats','edges','higher_derivatives','composed_graphs','primitive_slopes')
 result={'status':'qualified_scoped_rules_and_rejection_ledger','lean_theorems':formal['lean_theorems'],
  'rocq_statements':formal['rocq_statements'],'new_z3_checks':formal['new_z3_checks'],
  'shape_decisions':discovery['shape_count'],'gpu_checks':{g:len(gpu[g]) for g in groups},
  'gpu_failures':0,'added_axioms':[],'cpu_ml_execution':False,'decisions':decisions,
  'universal_backprop_reduction_established':False,'dense_gelu_silu_compression_established':False,
  'scope':'Sampled exact-real factor identities and FP32 GPU correctness; repeated performance only for enumerated eager-baseline shapes. No all-future-ML or minimum-memory guarantee.'}
 (ROOT/'reports/gradient-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='decisions'},indent=2))

if __name__=='__main__':main()
