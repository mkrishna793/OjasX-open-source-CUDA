import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import sys,json,time,random,statistics,gc
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import require_gpu,GPUOnly,compare,hashes
from test_cotangent_quotient_gpu import values,rand,trainables,clone
from ojasx2.cotangent_quotient import KINDS
from ojasx2.demand_algebra import demand_mlp
from ojasx2.residuals import SavedTensorAudit

def setup(kind,shape=(65,12,97,16)):
    v=list(values(shape,kind));v.extend((rand(shape[1],shape[2]),rand(shape[2])) if kind=='swiglu' else (None,None))
    return v

def case(kind,pattern,modify=None):
    torch.manual_seed(924);v=setup(kind);x,w,b,q,u,ub=v
    if modify=='frozen':v[1]=v[1].detach()
    if modify=='only_projection':v=[t.detach().requires_grad_(i==3) if t is not None else None for i,t in enumerate(v)]
    if modify=='no_bias':v[2]=None;v[5]=None
    if modify=='view':v[0]=v[0].T.contiguous().T.detach().requires_grad_()
    if modify=='half':v=[t.detach().half().requires_grad_(t.requires_grad) if t is not None else None for t in v]
    if modify in ('nan','inf','large'):
        with torch.no_grad():v[0][0,0]=1e10 if modify=='large' else float(modify)
    p=rand(65,16,grad=False).to(v[0].dtype)
    if pattern in ('rows','rectangle'):p[1:]=0
    if pattern in ('columns','rectangle'):p[:,2:]=0
    if pattern=='zero':p.zero_()
    if modify=='nonfinite_upstream':p[0,0]=float('nan')
    out=[];memory={};ev=[]
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([t for t in v if t is not None]) as audit:
            y=demand_mlp(*v,kind,accuracy=mode,evidence=ev if mode=='algebraic' else None)
        out.append((y,)+torch.autograd.grad(y,trainables(v),p));memory[mode]=audit.summary()
    errors=[compare(a,b) for a,b in zip(out[1],out[0])]
    return dict(name=kind+'_'+pattern+'_'+str(modify),errors=errors,passed=all(e['passed'] for e in errors),memory=memory,evidence=ev)

def higher():
    v=setup('gelu',(8,3,9,4));ps=trainables(v);outs=[]
    for mode in ('strict','algebraic'):
        y=demand_mlp(*v,'gelu',accuracy=mode)
        # Zero first-order upstream with nonzero second derivative.
        loss=(y-y.detach()).square().sum()
        gs=torch.autograd.grad(loss,ps,create_graph=True)
        outs.append(torch.autograd.grad(sum(g.sum() for g in gs),ps))
    es=[compare(a,b) for a,b in zip(*outs)]
    return dict(name='zero_first_order_nonzero_hessian',errors=es,passed=all(e['passed'] for e in es))

def mutation():
    v=setup('gelu');y=demand_mlp(*v,'gelu',accuracy='algebraic')
    with torch.no_grad():v[1].add_(.1)
    try:torch.autograd.grad(y.sum(),trainables(v))
    except RuntimeError as e:ok='modified by an inplace operation' in str(e)
    else:ok=False
    return dict(name='mutation_version_check',passed=ok)

def model(kind):
    torch.manual_seed(957);x=rand(2,16,16);ps=(rand(16,16),rand(16,16),rand(16,16),*setup(kind,(32,16,128,16))[1:])
    states={mode:clone(ps) for mode in ('strict','algebraic')};steps=[]
    for _ in range(3):
        outputs={}
        for mode,p in states.items():
            qw,kw,vw,w,b,q,u,ub=p
            h=x+F.scaled_dot_product_attention((x@qw)[:,None],(x@kw)[:,None],(x@vw)[:,None],is_causal=True)[:,0]
            h=F.layer_norm(h,(16,))
            y=demand_mlp(h.reshape(-1,16),w,b,q,u,ub,kind,accuracy=mode)
            loss=y.reshape(2,16,16)[:,-1,:4].square().mean()+.01*w.square().mean()
            gs=torch.autograd.grad(loss,(x,)+trainables(p))
            with torch.no_grad():
                for a,g in zip(trainables(p),gs[1:]):a.add_(g,alpha=-.001)
            outputs[mode]=(loss,)+gs+trainables(p)
        es=[compare(a,b) for a,b in zip(outputs['strict'],outputs['algebraic'])]
        steps.append(dict(errors=es,passed=all(e['passed'] for e in es)))
    return dict(name=kind+'_attention_shared_branch_sgd',steps=steps,passed=all(c['passed'] for c in steps))

def timing(kind,seed):
    torch.manual_seed(seed);shape=(4096,512,2048,512);v=setup(kind,shape);ps=trainables(v);ev=[]
    times={m:[] for m in ('strict','algebraic')}
    def run(mode):
        y=demand_mlp(*v,kind,accuracy=mode)
        loss=y[::16,:128].square().mean()
        gs=torch.autograd.grad(loss,ps)
        with torch.no_grad():
            for t,g in zip(ps,gs):t.add_(g,alpha=0)
        return (loss,)+gs
    errors=[compare(a,b) for a,b in zip(run('strict'),run('algebraic'))]
    for m in times:
        for _ in range(2):run(m)
    rng=random.Random(seed)
    for _ in range(5):
        order=list(times);rng.shuffle(order)
        for m in order:
            torch.mps.synchronize();start=time.perf_counter();run(m);torch.mps.synchronize();times[m].append(time.perf_counter()-start)
        time.sleep(.15)
    med={m:statistics.median(t) for m,t in times.items()}
    return dict(name=kind,seed=seed,shape=shape,active_fraction=1/16,active_columns_fraction=.25,
                errors=errors,passed=all(e['passed'] for e in errors),times=times,medians=med,native_over_candidate=med['strict']/med['algebraic'],
                scope='Complete native block forward, selected-output loss, all original gradients, zero-lr SGD. Synthetic exact cotangent support; not whole-model speed.')

def main():
    require_gpu()
    with GPUOnly() as guard:
        cases=[case(k,p) for k in (*KINDS,'swiglu') for p in ('dense','rows','columns','rectangle','zero')]
        cases += [case('swiglu','rectangle',m) for m in ('frozen','only_projection','no_bias','view','half','nan','inf','large','nonfinite_upstream')]
        edges=[higher(),mutation()];models=[model(k) for k in ('gelu','silu','swiglu')]
    torch.set_default_device(None);timings=[]
    for seed in (961,962,963):
        for kind in ('gelu','swiglu'):
            timings.append(timing(kind,seed));gc.collect();torch.mps.empty_cache();time.sleep(.2)
    allcases=cases+edges+models+timings
    report=dict(status='passed_scoped_demand_gpu' if all(c['passed'] for c in allcases) else 'failed',
                cases=cases,edges=edges,models=models,timings=timings,device='mps',cpu_ml_execution=False,cpu_fallback_enabled=False,
                guarded_dispatches=guard.calls,source_sha256=hashes(['ojasx2/demand_algebra.py','ojasx2/cotangent_quotient.py','tools/test_demand_gpu.py','tools/test_cotangent_quotient_gpu.py','tools/gpu_contracts.py','ojasx2/residuals.py']))
    (ROOT/'reports/demand-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(allcases),'cases; failures',[c['name'] for c in allcases if not c['passed']],flush=True)
    print('speed',[(c['name'],c['native_over_candidate']) for c in timings],flush=True)
    assert report['status']=='passed_scoped_demand_gpu'
if __name__=='__main__':main()
