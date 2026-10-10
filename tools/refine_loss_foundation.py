"""Bounded discovery -> complete kernel audit -> GPU suites -> admission.

Invoke with .tools/conformance-venv/bin/python. Without --gpu this validates
existing source-matched GPU reports; it never invents missing test evidence.
Host proof/metadata programs run sequentially before numerical GPU work.
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def validate(report):
    for path,digest in report['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Stale evidence: '+path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    subprocess.run([str(Path(sys.executable).with_name('torchcts')),'check-manifest',
                    '--manifest','configs/torchcts_mps.py'],cwd=ROOT,env=env,check=True)
    def run(name):subprocess.run([sys.executable,'-B','tools/'+name],cwd=ROOT,env=env,check=True)
    run('discover_loss_rules.py')
    run('refine_objective_foundation.py')
    if args.gpu:
        run('test_external_conformance_gpu.py');run('test_loss_gpu.py');run('test_loss_fx_gpu.py')
    paths=['reports/loss-discovery.json','reports/loss-gpu.json','reports/external-conformance-gpu.json',
           'reports/spectral-formal-verification.json','reports/loss-fx-gpu.json']
    discovery,gpu,external,formal,fx=[json.loads((ROOT/p).read_text()) for p in paths]
    for report in (discovery,gpu,external,formal,fx):validate(report)
    assert discovery['status']=='passed_bounded_loss_discovery'
    assert gpu['status']=='passed_scoped_loss_gpu'
    assert external['status']=='passed_bounded_gpu_adapters'
    assert formal['status']=='passed_formal_checks'
    assert fx['status']=='passed_scoped_loss_fx'
    for report in (gpu,external,fx):
        assert report['device']=='mps' and not report['cpu_ml_execution'] and not report['cpu_fallback_enabled']
    # Pin metadata bytes and the untouched upstream repositories as well.
    assert hashlib.sha256((ROOT/'third_party/backend_bench_problems.parquet').read_bytes()).hexdigest()==external['backendbench']['sha256']
    for name,commit in external['source_commits'].items():
        repo=ROOT/'third_party'/name
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    cts=external['torchcts'];bb=external['backendbench']['results']
    for name,version in external['versions'].items():
        assert importlib.metadata.version(name)==version, 'Changed dependency: '+name
    assert sum(x['status']=='passed' for x in cts)>=30
    assert sum(x['status']=='passed' for x in bb)>=12
    ratios=[r['native_over_candidate'] for r in gpu['timings']]
    fair=[r['preselected_over_candidate'] for r in gpu['timings']]
    result=dict(status='passed_scoped_loss_foundation',lean_theorems=formal['lean_theorems'],
        rocq_statements=formal['rocq_statements'],added_axioms=[],independent_rocq_kernel_recheck=True,
        symbolic_checks=len(discovery['checks']),symbolic_derivatives=len(discovery['analytic']),
        counterexamples=sum('counterexample' in c for c in discovery['checks']),shape_plans=len(discovery['shape_search']),
        gpu_cases=sum(len(gpu[k]) for k in ('cases','models','edges','scaling','timings','decoder_timings')),
        cts_passed=sum(r['status']=='passed' for r in cts),cts_filtered=sum(r['status']=='filtered' for r in cts),
        backendbench_passed=sum(r['status']=='passed' for r in bb),backendbench_total_profiles=external['backendbench']['corpus_rows'],
        stock_suite_certification=False,cpu_ml_execution=False,
        fx_cases=len(fx['cases']),
        selected_head=dict(shape=gpu['timings'][0]['shape'],active_fraction=gpu['timings'][0]['active_fraction'],
            correctness=True,leading_dense_matrix_work_reduction=1-gpu['timings'][0]['active_fraction'],
            speed_ratios=ratios,speed_qualified=len(ratios)>=3 and min(ratios)>1.05,
            double_speed_qualified=len(ratios)>=3 and min(ratios)>2,
            preselected_pytorch_ratios=fair,beats_preselected_pytorch=min(fair)>1.05,
            scope='Only linear loss head with most targets ignored; native activations unchanged. No whole-model throughput claim.'),
        replay_head=dict(memory_qualified=all(r['memory']['algebraic']['fresh_saved_storage_bytes']<r['memory']['strict']['fresh_saved_storage_bytes'] for r in gpu['scaling']),
            all_active_arithmetic_reduction=False,recomputes_head=True,parameter_storage_eliminated=False),
        small_decoder=dict(speed_ratios=[r['native_over_candidate'] for r in gpu['decoder_timings']],
            correctness=all(r['passed'] for r in gpu['decoder_timings']),
            speed_qualified=min(r['native_over_candidate'] for r in gpu['decoder_timings'])>1.05,
            scope=gpu['decoder_timings'][0]['scope']),
        capacity_200b=gpu['capacity_200b'],
        trust_boundary='Real analytic/algebraic proofs, bounded GPU comparisons, native MPS coverage for unreplaced operators. No universal speed, all-input floating-point refinement, all-architecture coverage, or certified peak memory.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in paths+['configs/torchcts_mps.py','configs/conformance-requirements.lock','tools/refine_loss_foundation.py']})
    (ROOT/'reports/loss-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS scoped loss foundation:',result['lean_theorems'],'Lean,',result['rocq_statements'],'Rocq;',
          result['gpu_cases'],'loss cases,',result['cts_passed'],'CTS,',result['backendbench_passed'],'profiles',flush=True)

if __name__=='__main__':main()
