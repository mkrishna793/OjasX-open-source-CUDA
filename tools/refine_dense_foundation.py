"""Finite synthesis → closed proofs → source-matched GPU qualification."""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,sys
from refine_loss_foundation import validate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONDONTWRITEBYTECODE='1')
    def run(name):subprocess.run([sys.executable,'-B','tools/'+name],cwd=ROOT,env=env,check=True)
    run('discover_dense_joint.py');run('refine_demand_foundation.py')
    if args.gpu:run('test_dense_joint_gpu.py')
    paths=['reports/dense-discovery.json','reports/dense-gpu.json','reports/spectral-formal-verification.json','reports/demand-qualification.json']
    d,g,f,old=[json.loads((ROOT/s).read_text()) for s in paths]
    for r in (d,g,f,old):validate(r)
    assert d['status']=='passed_bounded_dense_discovery' and g['status']=='passed_scoped_dense_gpu' and not g['quick']
    assert g['device']=='mps' and not g['cpu_ml_execution'] and not g['cpu_fallback_enabled']
    assert f['status']=='passed_formal_checks'
    timings=[]
    for name in sorted({c['name'] for c in g['timings']}):
        rows=[c for c in g['timings'] if c['name']==name];ratios=[c['native_over_candidate'] for c in rows]
        timings.append(dict(name=name,shape=rows[0]['shape'],ratios=ratios,
                            speed_qualified=len(rows)>=3 and min(ratios)>1.05,
                            double_speed_qualified=len(rows)>=3 and min(ratios)>2))
    memory=[]
    for c in g['cases']:
        if c['name'].endswith('_None'):
            a=c['memory']['strict']['fresh_saved_storage_bytes'];b=c['memory']['algebraic']['fresh_saved_storage_bytes']
            memory.append(dict(name=c['name'],native_fresh_bytes=a,candidate_fresh_bytes=b,reduction_fraction=(a-b)/a if a else 0))
    r=dict(status='passed_scoped_dense_foundation',lean_theorems=f['lean_theorems'],rocq_statements=f['rocq_statements'],
           added_axioms=[],gpu_cases=sum(len(g[k]) for k in ('cases','matrices','cancellations','edges','models','timings'))+1,
           symbolic_checks=len(d['checks']),shape_plans=len(d['shape_plans']),timings=timings,memory=memory,
           cancellation_raw_rejections=sum(c['raw_rejected'] for c in g['cancellations']),
           cancellation_checked_passes=sum(c['passed'] for c in g['cancellations']),
           dense_nonlinear_memory_support=True,
           dense_arithmetic_candidate='Existing rank-seven multiplication and transposed shared encoders; unrestricted dense matrices over exact rings. FP32 changes accumulation order.',
           dense_arithmetic_speed_admitted=any(t['speed_qualified'] for t in timings if t['name'].startswith('rank7')),
           whole_model_speed_qualified=False,automatic_graph_dispatch=False,historical_novelty_claim=False,
           memory_scope='Fresh retained activation storage; original operands, gradients, optimizer state and transient peak remain. This residual schedule avoids first-order GEMM replay.',
           numerical_scope='Sampled MPS FP32 validation and explicit checked cancellation fallback; no formal Python/Metal/IEEE refinement.',
           source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths+['tools/refine_dense_foundation.py']})
    (ROOT/'reports/dense-qualification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('PASS dense foundation',r['lean_theorems'],'Lean',r['rocq_statements'],'Rocq',r['gpu_cases'],'GPU cases; speed:',timings)
if __name__=='__main__':main()
