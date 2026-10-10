"""Finite tool-driven discovery/proof/GPU qualification; no claim of universal search."""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,sys
from refine_loss_foundation import validate
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONDONTWRITEBYTECODE='1')
    def run(name):subprocess.run([sys.executable,'-B','tools/'+name],cwd=ROOT,env=env,check=True)
    run('discover_demand_algebra.py');run('refine_cotangent_foundation.py')
    if args.gpu:run('test_demand_gpu.py')
    paths=['reports/demand-discovery.json','reports/demand-gpu.json','reports/spectral-formal-verification.json','reports/cotangent-qualification.json']
    d,g,f,old=[json.loads((ROOT/s).read_text()) for s in paths]
    for r in (d,g,f,old):validate(r)
    assert d['status']=='passed_bounded_demand_discovery' and g['status']=='passed_scoped_demand_gpu'
    assert g['device']=='mps' and not g['cpu_ml_execution'] and not g['cpu_fallback_enabled']
    assert f['status']=='passed_formal_checks'
    timings=[]
    for name in sorted({c['name'] for c in g['timings']}):
        rows=[c for c in g['timings'] if c['name']==name];ratios=[c['native_over_candidate'] for c in rows]
        timings.append(dict(name=name,shape=rows[0]['shape'],ratios=ratios,
                            speed_qualified=len(rows)>=3 and min(ratios)>1.05,
                            double_speed_qualified=len(rows)>=3 and min(ratios)>2))
    r=dict(status='passed_scoped_demand_foundation',lean_theorems=f['lean_theorems'],rocq_statements=f['rocq_statements'],
           added_axioms=[],gpu_cases=sum(len(g[k]) for k in ('cases','edges','models','timings')),
           symbolic_checks=len(d['checks']),shape_plans=len(d['shape_plans']),timings=timings,
           memory_scope='Fresh activation tape removed; original operands and dense gradient outputs remain; recomputation paid.',
           support_scope='Actual exact-zero rows/columns of the incoming cotangent at private deterministic pointwise MLP blocks.',
           dense_all_active_arithmetic_reduction=False,automatic_graph_dispatch=False,historical_novelty_claim=False,
           source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths+['tools/refine_demand_foundation.py']})
    (ROOT/'reports/demand-qualification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('PASS demand foundation',r['lean_theorems'],'Lean',r['rocq_statements'],'Rocq',r['gpu_cases'],'GPU cases; speed:',timings)
if __name__=='__main__':main()
