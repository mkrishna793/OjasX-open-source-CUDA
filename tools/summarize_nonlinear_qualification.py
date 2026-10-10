"""Join checked proofs and separately executed GPU evidence; no ML execution."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports'
def read(name):return json.loads((REPORTS/name).read_text())
formal=read('nonlinear-formal-verification.json');gpu=read('nonlinear-gpu-search.json');repeat=read('nonlinear-gpu-repeat.json');edges=read('nonlinear-gpu-edges.json')
if formal['status']!='passed_formal_checks' or gpu['status']!='completed' or edges['status']!='passed':raise RuntimeError('Incomplete checks')
for name,r in [('formal',formal),('gpu',gpu),('repeat',repeat),('edges',edges)]:
 for path,expected in r['source_sha256'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise RuntimeError(name+' stale source '+path)
for name,r in [('gpu',gpu),('repeat',repeat),('edges',edges)]:
 if r['cpu_ml_execution'] or r['cpu_fallback_enabled']:raise RuntimeError(name+' violated GPU-only execution')
if gpu['accuracy_failures'] or not all(r['passed'] for r in gpu['higher_surrogate']):raise RuntimeError('GPU nonlinear accuracy failure')
active=[c for c in gpu['cases'] if c['route']['path']=='approximate_feature']
rows=[]
for kind in ('gelu','silu','swiglu'):
 c=next(c for c in active if c['kind']==kind and c['shape']==[512,2,4096,8] and c['domain']=='small')
 ns=c['memory']['native']['saved']['fresh_saved_storage_bytes'];cs=c['memory']['candidate']['saved']['fresh_saved_storage_bytes']
 ratios=[r['native_over_candidate'] for r in repeat['cases'] if r['kind']==kind]
 rows.append({'kind':kind,'shape':c['shape'],'modeled_arithmetic_ratio':c['cost']['arithmetic_ratio'],'fresh_native_saved_bytes':ns,'fresh_candidate_saved_bytes':cs,'fresh_retained_reduction_percent':100*(1-cs/ns),'initial_timing_ratio':c['timing']['native_over_candidate'],'repeat_timing_ratios':ratios,'repeat_speed_pass':all(r>1 for r in ratios)})
result={'status':'verified_scoped_research_speed_target_not_met','automatic_dispatch':False,'cpu_ml_execution':False,'cpu_fallback_enabled':False,'lean_theorems':formal['lean_theorems'],'rocq_statements':formal['rocq_statements'],'new_z3_checks':formal['z3_checks'],'nonlinear_cases':len(gpu['cases']),'active_surrogate_cases':len(active),'native_fallback_cases':len(gpu['cases'])-len(active),'nonlinear_accuracy_failures':gpu['accuracy_failures'],'gpu_edge_checks':edges['count'],'surrogate_higher_derivative_cases':len(gpu['higher_surrogate']),'short_timing_joint_wins':gpu['observed_joint_wins'],'repeated_large_nonlinear_speed_wins':sum(r['repeat_speed_pass'] for r in rows),'large_nonlinear_cases':rows,'linear_normalized_control':{'correctness_passed':gpu['linear_normalized_control']['correctness_passed'],'native_over_candidate':gpu['linear_normalized_control']['native_over_candidate'],'optimizer_timed':False},'linear_unscaled_stress':{'correctness_passed':gpu['linear_control']['correctness_passed'],'errors':gpu['linear_control']['modes']['candidate']['errors'],'meaning':'Numerical counterexample to the selected FP32 pointwise tolerance; no bitwise reassociation guarantee.'},'source_hashes_validated':True,'scope':'Bounded synthetic blocks, explicit approximate primal; complete operand gradients and plain SGD timings; retained activation memory and sampled allocator observations, not certified total GPU peak. No broad speedup or full-model convergence claim.'}
(REPORTS/'nonlinear-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ('status','lean_theorems','rocq_statements','new_z3_checks','nonlinear_cases','gpu_edge_checks','short_timing_joint_wins','repeated_large_nonlinear_speed_wins')},indent=2))
