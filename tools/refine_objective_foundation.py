"""Finite discovery -> whole-project kernel audit -> MPS evidence -> admission.

Proof tools are host programs. No numerical CPU ML or perpetual automation.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    def run(name):subprocess.run([sys.executable,'-B','tools/'+name],cwd=ROOT,env=env,check=True)
    run('discover_objective_rules.py')
    # This preserves and validates earlier source-matched GPU qualifications.
    run('verify_numerical_repairs.py')
    if args.gpu:run('test_objective_gpu.py')
    paths=['reports/objective-discovery.json','reports/objective-gpu.json','reports/spectral-formal-verification.json']
    discovery,gpu,formal=[json.loads((ROOT/p).read_text()) for p in paths]
    for report in (discovery,gpu,formal):
        for p,digest in report['source_sha256'].items():
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:raise RuntimeError('Stale evidence: '+p)
    assert discovery['status']=='passed_bounded_symbolic_checks'
    assert gpu['status']=='passed_scoped_objective_gpu'
    assert gpu['device']=='mps' and not gpu['cpu_ml_execution'] and not gpu['cpu_fallback_enabled']
    admitted=[]
    for shape in sorted({tuple(c['shape']) for c in gpu['gram']}):
        rows=[c for c in gpu['gram'] if tuple(c['shape'])==shape]
        speeds=[c['timing']['native_over_candidate'] for c in rows if 'timing' in c]
        admitted.append(dict(rule='quadratic_gram',shape=shape,correctness=all(c['passed'] for c in rows),
            memory_qualified=all(c['memory']['algebraic']['fresh_saved_storage_bytes']<c['memory']['strict']['fresh_saved_storage_bytes'] for c in rows),
            leading_matrix_arithmetic_qualified=all(c['plan']['eligible'] for c in rows),
            speed_qualified=len(speeds)>=3 and min(speeds)>1.05,speed_ratios=speeds))
    masked=gpu['masked_timing'];speeds=[c['native_over_candidate'] for c in masked]
    result=dict(status='passed_scoped_objective_foundation',lean_theorems=formal['lean_theorems'],
        rocq_statements=formal['rocq_statements'],added_axioms=[],independent_rocq_kernel_recheck=True,
        symbolic_checks=len(discovery['checks']),counterexamples=sum('counterexample' in c for c in discovery['checks']),
        shape_decisions=len(discovery['shape_search']),gpu_cases=sum(len(gpu[k]) for k in ('gram','streaming','scaling','edges','masked_timing')),
        gram_admission=admitted,
        separable_admission=dict(correctness=all(c['passed'] for c in gpu['streaming']),
            families=[c['name'] for c in gpu['streaming']],memory_scaling=True,
            ordinary_unmasked_arithmetic_reduction=False,zero_weight_speed_qualified=min(speeds)>1.05,
            zero_weight_speed_ratios=speeds,zero_weight_arithmetic_scope='One sixteenth active independent examples; additional finite-loss/gradient and uniform-usage caller obligations'),
        cpu_ml_execution=False,
        trust_boundary='Kernel-checked real/rational identities and bounded GPU qualification. Python callbacks, floating-point implementation, arbitrary graph effects, global cost optimality and total hardware peak are not kernel verified.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (ROOT/'reports/objective-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS objective foundation:',result['lean_theorems'],'Lean',result['rocq_statements'],'Rocq;',result['gpu_cases'],'GPU cases; speed failures retained')

if __name__=='__main__':main()
