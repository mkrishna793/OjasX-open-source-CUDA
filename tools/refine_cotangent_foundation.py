"""Bounded symbolic search, whole-foundation kernel audit and MPS admission.

Without --gpu, only source-matched existing GPU evidence is accepted. Host
proof programs precede numerical GPU execution. This is a finite research
workflow, not an autonomous universal theorem/axiom discovery algorithm.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
from refine_loss_foundation import validate
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    def run(name):subprocess.run([sys.executable,'-B','tools/'+name],cwd=ROOT,env=env,check=True)
    run('discover_cotangent_quotient.py')
    run('refine_loss_foundation.py')
    if args.gpu:run('test_cotangent_quotient_gpu.py')
    paths=['reports/cotangent-quotient-discovery.json','reports/cotangent-quotient-gpu.json',
           'reports/spectral-formal-verification.json','reports/loss-qualification.json']
    discovery,gpu,formal,previous=[json.loads((ROOT/p).read_text()) for p in paths]
    for report in (discovery,gpu,formal,previous):validate(report)
    assert discovery['status']=='passed_bounded_quotient_discovery'
    assert gpu['status']=='passed_scoped_cotangent_gpu'
    assert formal['status']=='passed_formal_checks'
    assert previous['status']=='passed_scoped_loss_foundation'
    assert gpu['device']=='mps' and not gpu['cpu_ml_execution'] and not gpu['cpu_fallback_enabled']
    groups=[gpu['quotient']]+[c for k in ('cases','streams','higher','edges','models','scaling','timings') for c in gpu[k]]
    assert all(c['passed'] for c in groups)
    timing=[]
    for name,shape in sorted({(c['name'],tuple(c['shape'])) for c in gpu['timings']}):
        rows=[c for c in gpu['timings'] if c['name']==name and tuple(c['shape'])==shape]
        speeds=[c['native_over_candidate'] for c in rows]
        timing.append(dict(name=name,shape=shape,seeds=[c['seed'] for c in rows],speed_ratios=speeds,
            speed_qualified=len(rows)>=3 and min(speeds)>1.05,
            double_speed_qualified=len(rows)>=3 and min(speeds)>2,
            scope=rows[0]['scope']))
    memory=[]
    for c in gpu['cases']+gpu['scaling']:
        a=c['memory']['strict']['fresh_saved_storage_bytes'];b=c['memory']['algebraic']['fresh_saved_storage_bytes']
        memory.append(dict(name=c['name'],shape=c['shape'],native_saved_bytes=a,candidate_saved_bytes=b,
            fraction_saved=1-b/a if a else None,memory_qualified=b<a))
    result=dict(status='passed_scoped_cotangent_foundation',version='2.2',
        lean_theorems=formal['lean_theorems'],rocq_statements=formal['rocq_statements'],
        added_axioms=[],independent_rocq_kernel_recheck=formal['independent_rocq_kernel_recheck'],
        symbolic_checks=len(discovery['checks']),counterexamples=sum('counterexample' in c for c in discovery['checks']),
        synthesized_scalar_gradient_identities=len(discovery['synthesized_gradient_identities']),
        shape_plans=len(discovery['shape_search']),observable_quotient=discovery['quotient'],
        gpu_cases=len(groups),cpu_ml_execution=False,guarded_dispatches=gpu['guarded_dispatches'],
        timing_admission=timing,memory_admission=memory,
        centered_dense_smooth_arithmetic_reduction=False,
        streamed_supported_nonlinear_memory=True,streamed_arithmetic_reduction=False,
        qualification_thresholds=dict(atol=2e-4,rtol=2e-4,relative_l2=2e-4,min_repeated_speed_ratio=1.05),
        automatic_graph_dispatch=False,whole_model_speed_qualified=False,
        preserved_counterexample='reports/cotangent-quotient-gpu-boundary-failure.json',
        trust_boundary='Dimension-general real/rational identities plus bounded MPS FP32 tests. Observable quotient formation is not free. Smooth rounded-slope support is not a real derivative certificate. Python/Metal/IEEE implementation, all-input numerical bounds and global memory optimality are not formally proved.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths+['tools/refine_cotangent_foundation.py']})
    (ROOT/'reports/cotangent-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS scoped cotangent foundation:',result['lean_theorems'],'Lean,',result['rocq_statements'],
          'Rocq;',result['gpu_cases'],'GPU cases; speed-qualified:',
          [(c['name'],c['shape']) for c in timing if c['speed_qualified']],flush=True)

if __name__=='__main__':main()
