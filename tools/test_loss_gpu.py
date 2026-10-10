"""Bounded MPS tests of exact-real selective/streamed CE, including full models."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import sys,json,time,statistics,random,gc
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import require_gpu,GPUOnly,compare,hashes
from ojasx2.loss_compiler import linear_cross_entropy,resident_state_ledger
from ojasx2.residuals import SavedTensorAudit


def rand(*s,grad=True,scale=.1,dtype=torch.float32):
    return (torch.randn(*s,device='mps',dtype=dtype)*scale).requires_grad_(grad)


def errors(a,b):return [compare(x,y) for x,y in zip(a,b)]


def case(name,shape=(33,16,67),schedule='replay',chunk=7,smoothing=0.,weighted=False,
         reduction='mean',mask=3,dtype=torch.float32,freeze=None,noncontiguous=False,
         scale=.1,all_ignored=False):
    n,k,v=shape;h=rand(n,k,scale=scale,dtype=dtype);w=rand(k,v,dtype=dtype);b=rand(v,dtype=dtype)
    if noncontiguous:h=rand(k,n,dtype=dtype).T;w=rand(v,k,dtype=dtype).T
    if freeze=='h':h=h.detach()
    if freeze=='w':w=w.detach()
    if freeze=='both':h=h.detach();w=w.detach()
    target=torch.arange(n,device='mps')%v
    if mask:target[::mask]=-100
    if all_ignored:target.fill_(-100)
    weight=torch.linspace(.1,1.5,v,device='mps',dtype=dtype) if weighted else None
    pars=tuple(t for t in (h,w,b) if t.requires_grad)
    out=[];memory={};ev=[]
    for mode in ('strict','algebraic'):
        with SavedTensorAudit([h,w,b,target]+([weight] if weight is not None else [])) as audit:
            loss=linear_cross_entropy(h,w,target,b,weight=weight,accuracy=mode,schedule=schedule,
                chunk_rows=chunk,label_smoothing=smoothing,reduction=reduction,evidence=ev if mode=='algebraic' else None)
        out.append((loss.detach(),)+torch.autograd.grad(loss,pars))
        memory[mode]=audit.summary()
    es=errors(out[1],out[0])
    for entry in ev:
        if 'plan' not in entry:continue
        active=entry['active'];unit=2*active*k*v
        selected=unit*(1+int(h.requires_grad)+int(w.requires_grad))
        expected=selected+unit*int(h.requires_grad or w.requires_grad or b.requires_grad)
        es.append(dict(passed=entry['plan']['replay_leading_flops']==expected,
                       check='demand-aware replay count includes bias-only training'))
    return dict(name=name,shape=shape,errors=es,evidence=ev,memory=memory,passed=all(e['passed'] for e in es))


def model_case(name,optimizer):
    # All tokens run through attention, including ignored prefix labels. Tied
    # embeddings receive both lookup and output-head gradient contributions.
    torch.manual_seed(713);v=97;k=24;batch=2;length=16
    x=torch.arange(batch*length,device='mps').reshape(batch,length)%v
    target=(x+1)%v;target[:,:12]=-100
    p0=(rand(v,k),rand(k,k),rand(k,k),rand(k,k),rand(k,2*k),rand(k,2*k),rand(2*k,k))
    states={m:tuple(p.detach().clone().requires_grad_() for p in p0) for m in ('strict','algebraic')}
    opts={m:(torch.optim.AdamW(p,lr=1e-3,foreach=False) if optimizer=='adamw' else torch.optim.SGD(p,lr=1e-3)) for m,p in states.items()}
    def objective(p,mode):
        emb,qw,kw,vw,up,gate,down=p
        h=F.embedding(x,emb);q,k_,vv=h@qw,h@kw,h@vw
        h=h+F.scaled_dot_product_attention(q[:,None],k_[:,None],vv[:,None],is_causal=True)[:,0]
        h=F.layer_norm(h,(h.shape[-1],))
        if name=='swiglu_tied_decoder':h=h+(F.silu(h@gate)*(h@up))@down
        else:h=h+F.gelu(h@up)@down+0*(h@gate).sum()
        h=F.layer_norm(h,(h.shape[-1],))
        return linear_cross_entropy(h,emb.T,target,accuracy=mode,schedule='replay',chunk_rows=3)
    rounds=[]
    for step in range(3):
        vals={};grads={}
        for mode,p in states.items():
            opts[mode].zero_grad(set_to_none=True);loss=objective(p,mode);loss.backward()
            vals[mode]=loss.detach();grads[mode]=tuple(t.grad.detach().clone() for t in p)
            opts[mode].step()
        es=errors((vals['algebraic'],)+grads['algebraic']+states['algebraic'],
                  (vals['strict'],)+grads['strict']+states['strict'])
        if optimizer=='adamw':
            for a,b in zip(states['algebraic'],states['strict']):
                for key in ('exp_avg','exp_avg_sq'):
                    es.append(compare(opts['algebraic'].state[a][key],opts['strict'].state[b][key]))
        rounds.append(dict(step=step,errors=es,passed=all(e['passed'] for e in es)))
    return dict(name=name,optimizer=optimizer,rounds=rounds,passed=all(r['passed'] for r in rounds))


def edge_cases():
    results=[]
    h=rand(5,4);w=rand(4,7);t=torch.arange(5,device='mps')
    for label,kw in [('out_of_range',{}),('head_budget',{'logit_budget_bytes':1}),
                     ('bad_smoothing',{'label_smoothing':1.1})]:
        target=t+100 if label=='out_of_range' else t
        try:linear_cross_entropy(h,w,target,accuracy='algebraic',**kw)
        except (ValueError,MemoryError) as e:results.append(dict(name=label,passed=True,action='explicit rejection',reason=str(e)))
        else:results.append(dict(name=label,passed=False))
    outputs=[];ev=[]
    for mode in ('strict','algebraic'):
        z=linear_cross_entropy(h,w,t,accuracy=mode,derivative_order=2,evidence=ev if mode=='algebraic' else None)
        g=torch.autograd.grad(z,h,create_graph=True)[0]
        outputs.append((g,torch.autograd.grad(g.square().sum(),(h,w))))
    es=errors((outputs[1][0],)+outputs[1][1],(outputs[0][0],)+outputs[0][1])
    results.append(dict(name='higher_derivative_native',errors=es,evidence=ev,passed=all(e['passed'] for e in es)))
    # Fixed zero class weight with all selected targets on that class: preserve
    # native undefined weighted mean rather than silently changing its denominator.
    weight=torch.zeros(7,device='mps');ev=[]
    z=linear_cross_entropy(h,w,t,weight=weight,accuracy='algebraic',evidence=ev)
    results.append(dict(name='zero_weight_native',passed=bool(torch.isnan(z)) and ev[-1]['route']=='native',evidence=ev))
    # Check checkpoint version counters without SavedTensorAudit hooks.
    z=linear_cross_entropy(h,w,t,accuracy='algebraic',schedule='replay',chunk_rows=2)
    with torch.no_grad():w.add_(.01)
    try:torch.autograd.grad(z,(h,w))
    except RuntimeError as e:results.append(dict(name='mutation_version_check',passed='modified by an inplace operation' in str(e)))
    else:results.append(dict(name='mutation_version_check',passed=False))
    for value in (float('nan'),float('inf')):
        hh=h.detach().clone().requires_grad_()
        with torch.no_grad():hh[0,0]=value
        ev=[];z=linear_cross_entropy(hh,w,t,accuracy='algebraic',evidence=ev)
        ref=linear_cross_entropy(hh,w,t,accuracy='strict')
        es=errors((z,)+torch.autograd.grad(z,(hh,w)),(ref,)+torch.autograd.grad(ref,(hh,w)))
        results.append(dict(name='nonfinite_'+str(value),passed=ev[-1]['route']=='native' and all(e['passed'] for e in es),evidence=ev,errors=es))
    z=linear_cross_entropy(h,w,t,accuracy='algebraic',schedule='replay',chunk_rows=2)
    try:torch.autograd.grad(z,h,create_graph=True)
    except RuntimeError as e:results.append(dict(name='undeclared_higher_order',passed='derivative_order=2' in str(e)))
    else:results.append(dict(name='undeclared_higher_order',passed=False))
    ev=[];outs=[]
    with torch.autocast('mps',dtype=torch.float16):
        for mode in ('strict','algebraic'):
            z=linear_cross_entropy(h,w,t,accuracy=mode,evidence=ev if mode=='algebraic' else None)
            outs.append((z,)+torch.autograd.grad(z,(h,w)))
    es=errors(outs[1],outs[0])
    results.append(dict(name='autocast_native',passed=ev[-1]['route']=='native' and all(e['passed'] for e in es),errors=es,evidence=ev))
    return results


def timing_case(seed,shape=(2048,256,8192),divisor=16):
    torch.manual_seed(seed);n,k,v=shape;h=rand(n,k);w=rand(k,v);b=rand(v)
    target=torch.arange(n,device='mps')%v;target[torch.arange(n,device='mps')%divisor!=0]=-100
    active=(target!=-100).nonzero().flatten()
    def step(mode):
        if mode=='preselected_pytorch':
            z=F.cross_entropy(h.index_select(0,active)@w+b,target.index_select(0,active))
        elif mode=='strict':
            z=F.cross_entropy(h@w+b,target)
        else:z=linear_cross_entropy(h,w,target,b,accuracy=mode,schedule='selected')
        gs=torch.autograd.grad(z,(h,w,b))
        # This includes a full SGD parameter update, with a zero learning rate
        # so timed rounds use identical parameters (no training drift).
        with torch.no_grad():w.add_(gs[1],alpha=0);b.add_(gs[2],alpha=0)
        return z,gs
    timings={m:[] for m in ('strict','algebraic','preselected_pytorch')}
    for m in timings:
        for _ in range(2):step(m)
    torch.mps.synchronize();rng=random.Random(seed)
    for _ in range(5):
        modes=list(timings);rng.shuffle(modes)
        for m in modes:
            torch.mps.synchronize();start=time.perf_counter();step(m);torch.mps.synchronize()
            timings[m].append(time.perf_counter()-start)
        time.sleep(.05)
    med={m:statistics.median(t) for m,t in timings.items()}
    a,ag=step('algebraic');b_,bg=step('strict');es=errors((a,)+ag,(b_,)+bg)
    return dict(seed=seed,shape=shape,active_fraction=1/divisor,seconds=timings,median=med,
                native_over_candidate=med['strict']/med['algebraic'],
                preselected_over_candidate=med['preselected_pytorch']/med['algebraic'],
                errors=es,passed=all(e['passed'] for e in es),
                method='Synchronized wall time, two warmups, five shuffled rounds, direct native PyTorch vs guarded OjasX forward/loss/all gradients/zero-lr SGD; preselected comparator gets precomputed active indices; not whole-model throughput')


def decoder_timing(seed):
    torch.manual_seed(seed);v=4096;k=64;n=4;length=128
    tokens=torch.arange(n*length,device='mps').reshape(n,length)%v
    targets=(tokens+1)%v;targets[:,:120]=-100
    params=(rand(v,k),rand(k,k),rand(k,k),rand(k,k),rand(k,2*k),rand(k,2*k),rand(2*k,k))
    def step(mode):
        emb,qw,kw,vw,up,gate,down=params
        h=F.embedding(tokens,emb)
        h=h+F.scaled_dot_product_attention((h@qw)[:,None],(h@kw)[:,None],(h@vw)[:,None],is_causal=True)[:,0]
        h=F.layer_norm(h,(k,));h=h+(F.silu(h@gate)*(h@up))@down;h=F.layer_norm(h,(k,))
        if mode=='native':loss=F.cross_entropy((h@emb.T).reshape(-1,v),targets.reshape(-1))
        else:loss=linear_cross_entropy(h,emb.T,targets,accuracy='algebraic',schedule='selected')
        gs=torch.autograd.grad(loss,params)
        with torch.no_grad():
            for p,g in zip(params,gs):p.add_(g,alpha=0)
        return (loss.detach(),)+gs
    es=errors(step('ojasx'),step('native'));times={m:[] for m in ('native','ojasx')}
    for m in times:
        for _ in range(2):step(m)
    rng=random.Random(seed)
    for _ in range(5):
        modes=list(times);rng.shuffle(modes)
        for m in modes:
            torch.mps.synchronize();start=time.perf_counter();step(m);torch.mps.synchronize()
            times[m].append(time.perf_counter()-start)
        time.sleep(.1)
    med={m:statistics.median(t) for m,t in times.items()}
    return dict(seed=seed,name='one_block_tied_swiglu_decoder',batch=n,sequence=length,width=k,vocabulary=v,
                active_fraction=1/16,errors=es,passed=all(e['passed'] for e in es),seconds=times,median=med,
                native_over_candidate=med['native']/med['ojasx'],
                scope='Small complete decoder forward/loss/all parameter gradients/zero-lr SGD; no dropout, KV cache or distributed training')


def profile():
    h=rand(256,64);w=rand(64,2048);t=torch.arange(256,device='mps')%2048;t[1:240]=-100
    rows=[];(ROOT/'reports/traces').mkdir(exist_ok=True)
    for mode in ('strict','algebraic'):
        for _ in range(2):
            warm=linear_cross_entropy(h,w,t,accuracy=mode,schedule='replay',chunk_rows=16)
            torch.autograd.grad(warm,(h,w))
        torch.mps.synchronize();before=torch.mps.current_allocated_memory()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,
                                    record_shapes=True) as prof:
            with torch.profiler.record_function('OjasX_loss_'+mode):
                z=linear_cross_entropy(h,w,t,accuracy=mode,schedule='replay',chunk_rows=16)
                gs=torch.autograd.grad(z,(h,w));torch.mps.synchronize()
        path='reports/traces/loss-'+mode+'.json';prof.export_chrome_trace(str(ROOT/path))
        events=[dict(op=e.key,calls=e.count,host_us=e.self_cpu_time_total,host_memory_bytes=e.self_cpu_memory_usage)
                for e in prof.key_averages()]
        rows.append(dict(mode=mode,trace=path,events=sorted(events,key=lambda e:e['host_us'],reverse=True),
                         mps_live_delta_bytes=torch.mps.current_allocated_memory()-before))
        del z,gs
    return dict(activities=[str(a) for a in torch.profiler.supported_activities()],profiles=rows,
                scope='Host dispatch timeline while numerical tensors execute on MPS. No Metal kernel times or exact MPS peak supplied by torch.profiler.')


class AllocationSamples(TorchDispatchMode):
    def __init__(self):self.high=torch.mps.current_allocated_memory()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        self.high=max(self.high,torch.mps.current_allocated_memory())
        return out


def memory_case():
    h=rand(512,128);w=rand(128,4096);t=torch.arange(512,device='mps')%4096
    rows={}
    for mode in ('strict','algebraic'):
        torch.mps.synchronize();torch.mps.empty_cache();gc.collect()
        before=torch.mps.current_allocated_memory();sampler=AllocationSamples()
        with sampler:
            z=linear_cross_entropy(h,w,t,accuracy=mode,schedule='replay',chunk_rows=32)
            torch.mps.synchronize();forward=torch.mps.current_allocated_memory()-before
            gs=torch.autograd.grad(z,(h,w));torch.mps.synchronize()
        rows[mode]=dict(forward_live_delta_bytes=forward,
            sampled_peak_delta_bytes=sampler.high-before,driver_allocated_bytes=torch.mps.driver_allocated_memory())
        del z,gs
    return dict(shape=[512,128,4096],all_labels_active=True,chunk_rows=32,measurements=rows,
                scope='Observed MPS live allocations at ATen returns, excluding preexisting inputs/weights. Not certified peak; hidden Metal workspaces may be missed.')


def main():
    require_gpu();torch.manual_seed(713);guard=GPUOnly()
    with guard:
        cases=[]
        for schedule in ('selected','replay'):
            for smoothing in (0.,.2,1.):
                for weighted in (False,True):
                    cases.append(case(f'{schedule}_smooth{smoothing}_weight{weighted}',schedule=schedule,smoothing=smoothing,weighted=weighted))
        for name,kw in [('sum',dict(reduction='sum')),('noncontiguous',dict(noncontiguous=True)),
             ('all_active',dict(mask=0)),('singleton_class',dict(shape=(7,3,1))),
             ('freeze_h',dict(freeze='h')),('freeze_w',dict(freeze='w')),('bias_only',dict(freeze='both')),
             ('all_ignored_mean',dict(all_ignored=True)),('all_ignored_sum',dict(all_ignored=True,reduction='sum')),
             ('large_finite',dict(scale=100)),('empty_rows',dict(shape=(0,3,5))),
             ('fp16_native',dict(dtype=torch.float16)),('bf16_native',dict(dtype=torch.bfloat16))]:
            cases.append(case(name,**kw))
        models=[]
        for model in ('gelu_tied_decoder','swiglu_tied_decoder'):
            for opt in ('sgd','adamw'):
                models.append(model_case(model,opt));time.sleep(.1)
        edges=edge_cases()
        scaling=[case('scaling_'+str(n),shape=(n,32,1024),mask=0,chunk=32) for n in (64,256,1024)]
        profiles=profile();allocations=memory_case()
    # Dispatch guards add Python overhead; all tensors/device routes above are
    # guarded. Timings repeat the same code with MPS-only validation at entry.
    torch.set_default_device(None)
    timings=[]
    for seed in (713,714,715):
        timings.append(timing_case(seed));gc.collect();torch.mps.empty_cache();time.sleep(.2)
    decoder_timings=[decoder_timing(seed) for seed in (713,714,715)]
    allrows=cases+models+edges+scaling+timings+decoder_timings
    result=dict(status='passed_scoped_loss_gpu' if all(r['passed'] for r in allrows) else 'failed_scoped_loss_gpu',
        device='mps',torch_version=torch.__version__,cpu_ml_execution=False,cpu_fallback_enabled=False,
        guarded_dispatches=guard.calls,cases=cases,models=models,edges=edges,scaling=scaling,timings=timings,decoder_timings=decoder_timings,
        profiling=profiles,allocator_samples=allocations,capacity_200b=resident_state_ledger(200_000_000_000),
        source_sha256=hashes(['tools/test_loss_gpu.py','tools/gpu_contracts.py','ojasx2/loss_compiler.py','ojasx2/residuals.py']))
    (ROOT/'reports/loss-gpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],'cases',len(allrows),'failed',[r.get('name',r.get('seed')) for r in allrows if not r['passed']],flush=True)
    print('head speed ratios',[r['native_over_candidate'] for r in timings],flush=True)
    if result['status'].startswith('failed'):raise SystemExit(1)

if __name__=='__main__':main()
