"""Bounded MPS-only nonlinear quotient/defect and streamed-moment tests."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import sys,json,math,time,statistics,random,gc
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import require_gpu,GPUOnly,compare,hashes
from ojasx2.cotangent_quotient import defect_projection,streamed_mlp,native_mlp,cotangent_representative,KINDS
from ojasx2.residuals import SavedTensorAudit


def rand(*s,scale=.1,grad=True):return (torch.randn(*s,device='mps')*scale).requires_grad_(grad)
def clone(vals):return tuple(t.detach().clone().requires_grad_(t.requires_grad) if t is not None else None for t in vals)
def compare_many(a,b):return [compare(x,y) for x,y in zip(a,b)]
def trainables(vals):return tuple(t for t in vals if t is not None and t.requires_grad)


def values(shape,kind,region='centered',bias=True):
    m,k,n,r=shape;x=rand(m,k);w=rand(k,n,scale=1/math.sqrt(k));b=rand(n);q=rand(n,r,scale=1/math.sqrt(n))
    if region in ('columns','sparse','linear'):
        with torch.no_grad():
            b.fill_({'relu':2.,'leaky_relu':2.,'hardtanh':2.,'gelu':8.,'silu':20.,'tanh':20.,'sigmoid':20.}.get(kind,0.))
            if region=='columns':b[::128]=0
            if region=='sparse':
                w[0].fill_(1.);x[-1,0]=-min(31.,float(b[0].item())+4.)
    return x,w,b if bias else None,q


def defect_case(kind,region,shape=(128,8,256,4),seed=804,modify=None):
    torch.manual_seed(seed);vals=list(values(shape,kind,region));m,k,n,r=shape
    if modify=='frozen':vals[1]=vals[1].detach()
    if modify=='no_bias':vals[2]=None
    if modify=='noncontiguous':vals[0]=vals[0].T.contiguous().T.detach().requires_grad_()
    if modify=='zero':
        with torch.no_grad():
            for t in vals:
                if t is not None:t.zero_()
    if modify in ('nan','inf'):
        with torch.no_grad():vals[0][0,0]=float(modify)
    if modify=='half':vals=[t.detach().half().requires_grad_(t.requires_grad) if t is not None else None for t in vals]
    ps=trainables(vals);p=torch.randn(m,r,device='mps',dtype=vals[0].dtype)*.1
    if modify=='large_upstream':p.fill_(2.)
    out=[];memory={};ev=[]
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([t for t in vals if t is not None]) as audit:
            y=defect_projection(*vals,kind,accuracy=mode,evidence=ev if mode=='algebraic' else None)
        out.append((y,)+torch.autograd.grad(y,ps,p));memory[mode]=audit.summary()
    es=compare_many(out[1],out[0])
    return dict(name=kind+'_'+region+('_'+modify if modify else ''),activation=kind,region=region,
        shape=shape,evidence=ev,errors=es,memory=memory,passed=all(e['passed'] for e in es))


def stream_case(kind,shape=(71,12,97,9),chunk=16,modify=None):
    torch.manual_seed(811);vals=list(values(shape,kind));m,k,n,r=shape
    vals.extend((rand(k,n,scale=.1),rand(n)) if kind=='swiglu' else (None,None))
    if modify=='frozen':vals[1]=vals[1].detach()
    if modify=='no_bias':vals[2]=None;vals[5]=None
    if modify=='noncontiguous':vals[0]=vals[0].T.contiguous().T.detach().requires_grad_()
    ps=trainables(vals);p=torch.randn(m,r,device='mps')*.1;out=[];ev=[];memory={}
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([t for t in vals if t is not None]) as audit:
            y=streamed_mlp(*vals,kind,accuracy=mode,chunk_rows=chunk,evidence=ev if mode=='algebraic' else None)
        out.append((y,)+torch.autograd.grad(y,ps,p));memory[mode]=audit.summary()
    es=compare_many(out[1],out[0])
    return dict(name=kind+('_'+modify if modify else ''),shape=shape,chunk=chunk,
        errors=es,evidence=ev,memory=memory,passed=all(e['passed'] for e in es))


def hadamard(n):
    h=torch.ones(1,1,device='mps')
    while h.shape[0]<n:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(n)


def quotient_case():
    m,n,k=32,64,4;u=hadamard(m)[:,:k+1];v=hadamard(n)[:,:k]
    left=u@u.T;right=v@v.T;x=u[:,1:]@rand(k,k,grad=False);w=rand(k,k,grad=False)@v.T
    d=rand(m,n,grad=False);rep=cotangent_representative(d,left,right)
    a=u.T@d;b=(d@v-u@(u.T@d@v))
    factored=u@a+b@v.T
    es=compare_many((x.T@rep,rep@w.T,rep.sum(0),rep),(x.T@d,d@w.T,d.sum(0),factored))
    return dict(name='observable_quotient_including_bias',dense_shape=[m,n],factor_width=2*k+1,
                errors=es,passed=all(e['passed'] for e in es),
                scope='Constructed orthogonal bases on GPU. Formation cost is paid; no acceleration claimed by merely materializing the representative.')


def higher_case(kind,stream=False):
    vals=values((9,3,64,2),kind,'columns');ps=trainables(vals);directions=tuple(torch.randn_like(t)*.1 for t in ps);out=[]
    for mode in ('strict','algebraic'):
        y=streamed_mlp(*vals,kind=kind,accuracy=mode,chunk_rows=3) if stream else defect_projection(*vals,kind,accuracy=mode)
        g=torch.autograd.grad(y.square().mean(),ps,create_graph=True)
        out.append(torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,directions)),ps))
    es=compare_many(out[1],out[0]);return dict(name=kind+('_stream' if stream else '_defect'),errors=es,passed=all(e['passed'] for e in es))


def boundary_case(kind,boundary):
    vals=values((128,8,256,4),kind);x,w,b,q=vals
    with torch.no_grad():x.zero_();b.fill_(boundary)
    p=torch.randn(128,4,device='mps')*.1;out=[];ev=[]
    for mode in ('strict','algebraic'):
        y=defect_projection(*vals,kind,accuracy=mode,evidence=ev if mode=='algebraic' else None)
        out.append((y,)+torch.autograd.grad(y,vals,p))
    es=compare_many(out[1],out[0])
    return dict(name=kind+'_boundary_'+str(boundary),evidence=ev,errors=es,
                passed=all(e['passed'] for e in es))


def graph_edge(stream,mutation=False):
    vals=values((128,8,256,4),'gelu','columns');out=[]
    def apply(mode):
        if stream:return streamed_mlp(*vals,kind='gelu',accuracy=mode,chunk_rows=32)
        return defect_projection(*vals,'gelu',accuracy=mode)
    if mutation:
        y=apply('algebraic')
        with torch.no_grad():vals[1].add_(.01)
        try:torch.autograd.grad(y.sum(),vals)
        except RuntimeError as exc:
            passed='modified by an inplace operation' in str(exc)
        else:passed=False
        return dict(name=('stream' if stream else 'defect')+'_mutation_version',passed=passed)
    for mode in ('strict','algebraic'):
        y=apply(mode)
        loss=y.square().mean()+.01*vals[1].square().sum()
        out.append((loss,)+torch.autograd.grad(loss,vals))
    es=compare_many(out[1],out[0])
    return dict(name=('stream' if stream else 'defect')+'_shared_weight_branch',errors=es,
                passed=all(e['passed'] for e in es))


def model_case(kind,optimizer):
    torch.manual_seed(833);x=rand(2,16,16);params=(rand(16,16),rand(16,16),rand(16,16),rand(16,128),rand(128),rand(128,16))
    if kind=='swiglu':params+=(rand(16,128),rand(128))
    states={mode:clone(params) for mode in ('strict','algebraic')}
    moments={mode:[(torch.zeros_like(p),torch.zeros_like(p)) for p in ps] for mode,ps in states.items()}
    def objective(ps,mode):
        qw,kw,vw,w,b,q=ps[:6]
        h=x+F.scaled_dot_product_attention((x@qw)[:,None],(x@kw)[:,None],(x@vw)[:,None],is_causal=True)[:,0]
        h=F.layer_norm(h,(16,));flat=h.reshape(-1,16)
        u,bu=ps[6:] if kind=='swiglu' else (None,None)
        out=streamed_mlp(flat,w,b,q,u,bu,kind,accuracy=mode,chunk_rows=7).reshape_as(h)
        # Residual and user-defined smooth objective, with shared x upstream.
        y=h+out;return (y*.1+torch.sin(y)).square().mean()
    rounds=[]
    for step in range(1,4):
        outputs={}
        for mode,ps in states.items():
            loss=objective(ps,mode);gs=torch.autograd.grad(loss,(x,)+ps);outputs[mode]=(loss.detach(),)+gs
            with torch.no_grad():
                for j,(p,g) in enumerate(zip(ps,gs[1:])):
                    if optimizer=='sgd':p.add_(g,alpha=-1e-3)
                    else:
                        a,b=moments[mode][j];a.mul_(.9).add_(g,alpha=.1);b.mul_(.999).addcmul_(g,g,value=.001)
                        p.mul_(1-1e-5);p.addcdiv_(a/(1-.9**step),(b/(1-.999**step)).sqrt()+1e-8,value=-1e-3)
        es=compare_many(outputs['algebraic']+states['algebraic'],outputs['strict']+states['strict'])
        if optimizer=='adamw':es+=compare_many(tuple(t for pair in moments['algebraic'] for t in pair),tuple(t for pair in moments['strict'] for t in pair))
        rounds.append(dict(step=step,errors=es,passed=all(e['passed'] for e in es)))
    return dict(name=kind+'_'+optimizer,rounds=rounds,passed=all(r['passed'] for r in rounds))


def timing_case(kind,seed,stream=False,shape=(2048,128,2048,16)):
    torch.manual_seed(seed);vals=values(shape,kind,'centered' if stream else 'columns');ps=trainables(vals)
    target=rand(shape[0],shape[3],grad=False);times={m:[] for m in ('strict','algebraic')}
    def step(mode):
        if stream:y=streamed_mlp(*vals,kind=kind,accuracy=mode,chunk_rows=128)
        else:y=defect_projection(*vals,kind,accuracy=mode)
        loss=(y-target).square().mean();gs=torch.autograd.grad(loss,ps)
        with torch.no_grad():
            for t,g in zip(ps[1:],gs[1:]):t.add_(g,alpha=0)
        return (loss.detach(),)+gs
    es=compare_many(step('algebraic'),step('strict'))
    for mode in times:
        for _ in range(2):step(mode)
    rng=random.Random(seed)
    for _ in range(5):
        modes=list(times);rng.shuffle(modes)
        for mode in modes:
            torch.mps.synchronize();start=time.perf_counter();step(mode);torch.mps.synchronize()
            times[mode].append(time.perf_counter()-start)
        time.sleep(.1)
    med={mode:statistics.median(t) for mode,t in times.items()}
    return dict(name=kind+('_stream' if stream else '_defect'),seed=seed,shape=shape,errors=es,
        passed=all(e['passed'] for e in es),median=med,samples=times,native_over_candidate=med['strict']/med['algebraic'],
        scope='Full private block forward/loss/all gradients/zero-lr SGD, two warmups/five shuffled synchronized rounds. Synthetic biased nonlinear regime for defect, ordinary centered regime for stream.')


def main():
    require_gpu();torch.manual_seed(801)
    with GPUOnly() as guard:
        quotient=quotient_case();cases=[]
        for kind in KINDS:
            for region in ('centered','columns'):
                cases.append(defect_case(kind,region))
        for kind in ('relu','gelu','silu'):cases.append(defect_case(kind,'sparse'))
        for mod in ('frozen','no_bias','noncontiguous','zero','nan','inf','half','large_upstream'):
            cases.append(defect_case('gelu','columns',modify=mod))
        streams=[stream_case(kind) for kind in (*KINDS,'swiglu')]
        streams.extend(stream_case('swiglu',modify=m) for m in ('frozen','no_bias','noncontiguous'))
        higher=[higher_case(kind,stream) for kind in ('gelu','silu') for stream in (False,True)]
        edges=[boundary_case(kind,b) for kind,b in [('relu',0.),('leaky_relu',0.),
                ('hardtanh',-1.),('hardtanh',0.),('hardtanh',1.)]]
        edges.extend(graph_edge(stream,mutation) for stream in (False,True) for mutation in (False,True))
        edges.append(defect_case('gelu','centered',shape=(0,8,256,4)))
        edges.append(stream_case('swiglu',shape=(0,8,256,4)))
        models=[model_case(kind,opt) for kind in ('gelu','silu','swiglu') for opt in ('sgd','adamw')]
        scaling=[stream_case('swiglu',shape=(m,64,1024,64),chunk=32) for m in (64,256,1024)]
    torch.set_default_device(None);timings=[]
    for seed in (871,872,873):
        for kind in ('relu','gelu','silu'):
            timings.append(timing_case(kind,seed));gc.collect();torch.mps.empty_cache();time.sleep(.2)
    timings.append(timing_case('gelu',875,stream=True,shape=(512,64,1024,64)))
    for seed in (891,892,893):
        timings.append(timing_case('relu',seed,shape=(4096,2048,2048,16)))
        gc.collect();torch.mps.empty_cache();time.sleep(.2)
    groups=[quotient]+cases+streams+higher+edges+models+scaling+timings
    report=dict(status='passed_scoped_cotangent_gpu' if all(r['passed'] for r in groups) else 'failed_cotangent_gpu',
        device='mps',cpu_ml_execution=False,cpu_fallback_enabled=False,guarded_dispatches=guard.calls,
        quotient=quotient,cases=cases,streams=streams,higher=higher,edges=edges,models=models,scaling=scaling,timings=timings,
        source_sha256=hashes(['tools/test_cotangent_quotient_gpu.py','ojasx2/cotangent_quotient.py',
                              'tools/gpu_contracts.py','ojasx2/residuals.py']))
    (ROOT/'reports/cotangent-quotient-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],'cases',len(groups),'failed',[r['name'] for r in groups if not r['passed']],flush=True)
    print('timings',[(t['name'],t['native_over_candidate']) for t in timings],flush=True)
    if report['status'].startswith('failed'):raise SystemExit(1)

if __name__=='__main__':main()
