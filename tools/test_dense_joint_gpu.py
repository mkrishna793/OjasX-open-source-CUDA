"""Bounded dense-gradient MPS tests; host only controls, records and proves."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import sys,json,time,random,statistics,gc,argparse
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import require_gpu,GPUOnly,compare,hashes
from test_demand_gpu import setup
from test_cotangent_quotient_gpu import rand,trainables,clone
from ojasx2.dense_joint import residual_mlp,dense_matmul,rank7_plan
from ojasx2.cotangent_quotient import KINDS
from ojasx2.residuals import SavedTensorAudit

def errors(a,b):
    es=[compare(x,y) for x,y in zip(a,b)]
    return dict(errors=es,passed=all(e['passed'] for e in es))
def block(kind,modify=None):
    torch.manual_seed(1011);v=setup(kind,(33,12,65,16))
    if modify=='frozen':v[1]=v[1].detach()
    if modify=='frozen_projection':v[3]=v[3].detach()
    if modify=='projection_only':v=[t.detach().requires_grad_(i==3) if t is not None else None for i,t in enumerate(v)]
    if modify=='no_bias':v[2]=None;v[5]=None
    if modify=='view':v[0]=v[0].T.contiguous().T.detach().requires_grad_()
    if modify=='half':v=[t.detach().half().requires_grad_(t.requires_grad) if t is not None else None for t in v]
    if modify=='empty':v[0]=rand(0,12)
    if modify in ('nan','inf','large'):
        with torch.no_grad():v[0][0,0]=1e10 if modify=='large' else float(modify)
    if modify in ('minus_one','zero','plus_one'):
        with torch.no_grad():v[0].zero_();v[2].fill_({'minus_one':-1.,'zero':0.,'plus_one':1.}[modify])
    p=rand(v[0].shape[0],16,grad=False).to(v[0].dtype)
    if modify=='upstream_view':p=p.T.contiguous().T
    if modify=='upstream_nan':p[0,0]=float('nan')
    if modify=='zero_upstream':p.zero_()
    outs=[];mem={};ev=[]
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([t for t in v if t is not None]) as audit:
            y=residual_mlp(*v,kind,accuracy=mode,evidence=ev if mode=='algebraic' else None)
        outs.append((y,)+torch.autograd.grad(y,trainables(v),p));mem[mode]=audit.summary()
    return dict(name='residual_'+kind+'_'+str(modify),memory=mem,evidence=ev,**errors(*outs))

def higher(which):
    torch.manual_seed(1023)
    v=setup('swiglu',(8,4,16,4)) if which=='residual' else [rand(16,16),rand(16,16)]
    ps=trainables(v);outs=[]
    for mode in ('strict','algebraic'):
        y=residual_mlp(*v,'swiglu',accuracy=mode) if which=='residual' else dense_matmul(*v,accuracy=mode,cache=True)
        loss=(y-y.detach()).square().sum()
        gs=torch.autograd.grad(loss,ps,create_graph=True)
        outs.append(torch.autograd.grad(sum(g.sum() for g in gs),ps))
    return dict(name=which+'_zero_first_order_nonzero_hessian',**errors(*outs))

def mutation(which):
    v=setup('gelu') if which=='residual' else [rand(16,16),rand(16,16)]
    y=residual_mlp(*v,'gelu',accuracy='algebraic') if which=='residual' else dense_matmul(*v,accuracy='algebraic',cache=True)
    with torch.no_grad():v[1].add_(.1)
    try:torch.autograd.grad(y.sum(),trainables(v))
    except RuntimeError as e:ok='modified by an inplace operation' in str(e)
    else:ok=False
    return dict(name=which+'_mutation',passed=ok)

def matrix(shape=(64,96,80),cache=False,modify=None):
    torch.manual_seed(1041);m,k,n=shape;a=rand(m,k);b=rand(k,n)
    if modify=='view':a=a.T.contiguous().T.detach().requires_grad_()
    if modify=='frozen':a=a.detach()
    if modify=='alias':b=a
    if modify=='half':a=a.detach().half().requires_grad_();b=b.detach().half().requires_grad_()
    if modify in ('nan','inf','large'):
        with torch.no_grad():a[0,0]=1e10 if modify=='large' else float(modify)
    g=rand(m,n,grad=False).to(a.dtype)
    if modify=='upstream_nan':g[0,0]=float('nan')
    if modify=='upstream_view':g=g.T.contiguous().T
    outs=[];ev=[];mem={}
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([a,b]) as audit:
            y=dense_matmul(a,b,accuracy=mode,cache=cache,workspace_budget=0 if modify=='budget' else 256*1024**2,evidence=ev if mode=='algebraic' else None)
        ps=(a,) if a is b else trainables((a,b))
        outs.append((y,)+torch.autograd.grad(y,ps,g));mem[mode]=audit.summary()
    return dict(name='rank7_'+str(shape)+'_'+str(cache)+'_'+str(modify),memory=mem,evidence=ev,**errors(*outs))

def cancellation(seed,stage):
    torch.manual_seed(seed);p=rand(32,32,grad=False);q=rand(32,32,grad=False)
    if stage=='primal':
        a=torch.cat((torch.cat((p,p),1),torch.cat((p,p),1)),0).requires_grad_()
        b=torch.cat((torch.cat((q,q),1),torch.cat((-q,-q),1)),0).requires_grad_()
        g=rand(64,64,grad=False)
    else:
        a=torch.cat((torch.cat((p,p),1),torch.cat((-p,-p),1)),0).requires_grad_()
        b=rand(64,64);g=torch.cat((torch.cat((q,q),1),torch.cat((q,q),1)),0)
    y=a@b;ref=(y,)+torch.autograd.grad(y,(a,b),g)
    raw=dense_matmul(a,b,accuracy='algebraic')
    raw_es=errors((raw,)+torch.autograd.grad(raw,(a,b),g),ref)
    ev=[];y=dense_matmul(a,b,accuracy='checked',evidence=ev)
    checked=errors((y,)+torch.autograd.grad(y,(a,b),g),ref)
    return dict(name='cancellation_'+stage+'_'+str(seed),raw_rejected=not raw_es['passed'],raw_errors=raw_es,evidence=ev,**checked)

def model(kind,opt):
    torch.manual_seed(1051);x=rand(2,16,16)
    ps=(rand(16,16),rand(16,16),rand(16,16),*setup(kind,(32,16,128,16))[1:])
    states={mode:clone(ps) for mode in ('strict','algebraic')}
    opts={mode:(torch.optim.AdamW if opt=='adamw' else torch.optim.SGD)(trainables(p),lr=.001) for mode,p in states.items()}
    steps=[]
    for _ in range(3):
        out={}
        for mode,p in states.items():
            qw,kw,vw,w,b,q,u,ub=p
            h=x+F.scaled_dot_product_attention((x@qw)[:,None],(x@kw)[:,None],(x@vw)[:,None],is_causal=True)[:,0]
            h=F.layer_norm(h,(16,))
            y=residual_mlp(h.reshape(-1,16),w,b,q,u,ub,kind,accuracy=mode)
            # Dense loss on every coordinate; all branches and parameter grads retained.
            loss=(y.reshape_as(x)+h).square().mean()+.01*w.square().mean()
            gs=torch.autograd.grad(loss,(x,)+trainables(p))
            for t,g in zip(trainables(p),gs[1:]):t.grad=g
            opts[mode].step();opts[mode].zero_grad(set_to_none=True)
            out[mode]=(loss,)+gs+trainables(p)
        steps.append(errors(out['strict'],out['algebraic']))
    return dict(name=kind+'_dense_attention_residual_'+opt,steps=steps,passed=all(s['passed'] for s in steps))

def timing(which,seed,cache=False):
    torch.manual_seed(seed);times={mode:[] for mode in ('strict','algebraic')}
    if which=='rank7':
        shape=(1024,1024,1024);v=(rand(shape[0],shape[1]),rand(shape[1],shape[2]))
        call=lambda mode:dense_matmul(*v,accuracy=mode,cache=cache)
    else:
        shape=(1024,256,2048,256);v=setup(which,shape)
        call=lambda mode:residual_mlp(*v,which,accuracy=mode)
    ps=trainables(v)
    def run(mode):
        y=call(mode);loss=y.square().mean()
        gs=torch.autograd.grad(loss,ps)
        with torch.no_grad():
            for t,g in zip(ps,gs):t.add_(g,alpha=0)
        return (loss,)+gs
    es=errors(run('strict'),run('algebraic'))
    for mode in times:
        for _ in range(2):run(mode)
    rng=random.Random(seed)
    for _ in range(5):
        order=list(times);rng.shuffle(order)
        for mode in order:
            torch.mps.synchronize();t=time.perf_counter();run(mode);torch.mps.synchronize();times[mode].append(time.perf_counter()-t)
        time.sleep(.15)
    med={mode:statistics.median(ts) for mode,ts in times.items()}
    return dict(name=which+('_cache' if cache else '_replay') if which=='rank7' else which,seed=seed,shape=shape,times=times,medians=med,native_over_candidate=med['strict']/med['algebraic'],**es,
                scope='Complete block or matrix forward, dense squared loss, all original gradients, zero-lr SGD; original operands, transient scratch and optimizer state remain.')

def profiler_probe():
    v=setup('gelu',(32,16,128,16));events={}
    for mode in ('strict','algebraic'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as p:
            y=residual_mlp(*v,'gelu',accuracy=mode)
            torch.autograd.grad(y.square().mean(),trainables(v));torch.mps.synchronize()
        p.export_chrome_trace(str(ROOT/'reports'/('dense-'+mode+'-trace.json')))
        events[mode]={e.key:e.count for e in p.key_averages() if e.key in ('aten::mm','aten::gelu','aten::gelu_backward')}
    return dict(name='host_dispatch_profiler_gpu_work',events=events,passed=events['strict'].get('aten::mm')==events['algebraic'].get('aten::mm'),
                scope='CPU profiler activity records dispatch/control only. All numerical tensors on MPS. MPS allocator/device peak not exposed by profile_memory; no GPU timeline or peak-memory claim.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
    require_gpu()
    with GPUOnly() as guard:
        cases=[block(k) for k in (*KINDS,'swiglu')]
        cases += [block('swiglu',m) for m in ('frozen','projection_only','no_bias','view','half','empty','nan','inf','large','upstream_view','upstream_nan','zero_upstream')]
        cases += [block(k,'frozen_projection') for k in ('gelu','swiglu')]
        cases += [block(k,m) for k in ('relu','hardtanh','leaky_relu') for m in ('minus_one','zero','plus_one')]
        matrices=[matrix(s,c) for s in ((16,16,16),(64,96,80),(512,512,512)) for c in (False,True)]
        matrices += [matrix((64,64,64),True,m) for m in ('view','frozen','alias','half','nan','inf','large','budget','upstream_nan','upstream_view')]
        matrices += [matrix((33,32,16)),matrix((0,32,16))]
        cancellations=[cancellation(seed,stage) for seed in (1081,1082,1083) for stage in ('primal','gradient')]
        edges=[higher(w) for w in ('residual','rank7')]+[mutation(w) for w in ('residual','rank7')]
        models=[model(k,o) for k in ('gelu','silu','swiglu') for o in ('sgd','adamw')]
        probe=profiler_probe()
    torch.set_default_device(None);timings=[]
    if not args.quick:
        for seed in (1061,1062,1063):
            for which,cache in (('gelu',False),('swiglu',False),('rank7',False),('rank7',True)):
                timings.append(timing(which,seed,cache));gc.collect();torch.mps.empty_cache();time.sleep(.2)
    allcases=cases+matrices+cancellations+edges+models+[probe]+timings
    report=dict(status='passed_scoped_dense_gpu' if all(c['passed'] for c in allcases) else 'failed',quick=args.quick,cases=cases,matrices=matrices,cancellations=cancellations,edges=edges,models=models,profiler=probe,timings=timings,
                device='mps',cpu_ml_execution=False,cpu_fallback_enabled=False,guarded_dispatches=guard.calls,
                source_sha256=hashes(['ojasx2/dense_joint.py','ojasx2/bilinear.py','ojasx2/cotangent_quotient.py','tools/test_dense_joint_gpu.py','tools/test_demand_gpu.py','tools/test_cotangent_quotient_gpu.py','tools/gpu_contracts.py','ojasx2/residuals.py']))
    (ROOT/'reports'/('dense-gpu-quick.json' if args.quick else 'dense-gpu.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(allcases),'cases; failures',[c['name'] for c in allcases if not c['passed']],flush=True)
    print('speed',[(c['name'],round(c['native_over_candidate'],3)) for c in timings],flush=True)
    assert report['status']=='passed_scoped_dense_gpu'
if __name__=='__main__':main()
