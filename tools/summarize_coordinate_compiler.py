"""Fail-closed join of current mathematical compiler proofs and GPU evidence."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
reports={name:json.loads((R/path).read_text()) for name,path in [('formal','coordinate-formal-verification.json'),('log','coordinate-gpu.json'),('koopman','koopman-gpu.json'),('repeats','koopman-repeat.json')]}
for name,r in reports.items():
 if r['status'] not in ('completed','passed_formal_checks'):raise RuntimeError(name+' incomplete')
 for path,digest in r['source_sha256'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError(name+' stale source '+path)
 if name!='formal' and (r['cpu_ml_execution'] or r['cpu_fallback_enabled']):raise RuntimeError('GPU-only constraint violated')
for name in ('log','koopman'):
 r=reports[name]
 if r['accuracy_failures'] or r['edge_failures'] or not all(e['passed'] for e in r['higher_derivatives']):raise RuntimeError(name+' qualification failure')
if not all(c['passed'] for c in reports['repeats']['cases']):raise RuntimeError('Repeat accuracy failure')
repeat_rows=[]
for depth in (32,64):
 cs=[c for c in reports['repeats']['cases'] if c['depth']==depth]
 ratios=[c['timing']['native_over_candidate'] for c in cs]
 reductions=[100*(1-c['memory']['algebraic']['fresh_saved_storage_bytes']/c['memory']['strict']['fresh_saved_storage_bytes']) for c in cs]
 repeat_rows.append({'depth':depth,'shape':[512,32,2],'distinct_original_stage_weights':True,'seeds':[c['seed'] for c in cs],'native_over_candidate_ratios':ratios,'fresh_retained_storage_reduction_percent':reductions,'all_accuracy_pass':all(c['passed'] for c in cs),'all_speed_pass':all(r>1 for r in ratios),'all_retained_storage_pass':all(r>0 for r in reductions)})
f=reports['formal'];result={'status':'checked_compiler_prototype_scoped_nonlinear_gpu_wins','lean_theorems':f['lean_theorems'],'rocq_statements':f['rocq_statements'],'new_z3_conditions':f['z3_checks'],'added_foundation_axioms':[],'gpu_only_ml':True,'source_hashes_validated':True,'log_cases':len(reports['log']['cases']),'log_edges':len(reports['log']['edges']),'koopman_cases':len(reports['koopman']['cases']),'koopman_edges':len(reports['koopman']['edges']),'independent_repeat_cases':len(reports['repeats']['cases']),'repeated_distinct_weight_results':repeat_rows,'automatic_generic_dispatch':False,'scope':'Exact-real representation sequence contracts and experimentally qualified structured nonlinear recurrence. All original parameters preserved. No universal compact embedding, dense GELU/SiLU acceleration, bitwise equivalence, full-model convergence or certified total GPU peak claim.'}
(R/'coordinate-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
