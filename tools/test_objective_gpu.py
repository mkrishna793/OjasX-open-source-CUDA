"""Bounded sequential MPS qualification of full loss/all-gradient schedules.

No CPU tensor reference, no fallback, no stress loop. Timings synchronize MPS;
each family is small enough for this laptop, and runs are separated by pauses.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import sys,json,hashlib,time,statistics,math,random,gc
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.objective_compiler import quadratic_loss,quadratic_plan,compile_separable_objective,SeparableContract
from ojasx2.residuals import SavedTensorAudit
from search_nonlinear_gpu import compare,AllocationSamples


def rand(*shape,grad=True,scale=.15):
    return (torch.randn(*shape,device='mps')*scale).requires_grad_(grad)


def clone(v):return tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in v)


def comparisons(actual,reference):
    out=[]
    for a,b in zip(actual,reference):
        if a is None or b is None:out.append(dict(passed=a is b,none_match=a is b))
        else:out.append(compare(a.reshape(-1),b.reshape(-1)))
    return out


def native_value_grad(fn,params,batch,weights,normalization='batch',measure=False):
    denom=weights.sum() if normalization=='weight_sum' else (batch[0].shape[0] if normalization=='batch' else 1)
    with SavedTensorAudit(list(params+batch+(weights,))) as audit:
        loss=(fn(params,batch)*weights).sum()/denom
    demand=tuple(t for t in params+batch if t.requires_grad)
    gs=torch.autograd.grad(loss,demand,allow_unused=True) if loss.requires_grad and demand else (None,)*len(demand)
    cursor=iter(gs);ordered=tuple(next(cursor) if t.requires_grad else None for t in params+batch)
    return loss.detach(),ordered,audit.summary()


def architecture(name,n):
    if name in ('gelu_mlp','silu_mlp','shared_residual','custom_sine'):
        x=rand(n,32);labels=torch.arange(n,device='mps')%8
        params=(rand(32,64),rand(64),rand(64,64),rand(64),rand(64,8),rand(8))
        def losses(p,b):
            a=F.linear(b[0],p[0].T,p[1])
            act=F.silu if name=='silu_mlp' else F.gelu
            a=act(a)
            if name=='shared_residual':
                a=a+F.gelu(F.layer_norm(a@p[2]+p[3],(64,)))
                a=a+F.gelu(F.layer_norm(a@p[2]+p[3],(64,)))
            else:a=act(a@p[2]+p[3])
            if name=='custom_sine':a=a*torch.sin(a)+.1*a.square()
            return F.cross_entropy(a@p[4]+p[5],b[1],reduction='none')
    elif name=='conv_silu':
        x=rand(n,3,12,12);labels=torch.arange(n,device='mps')%8
        params=(rand(8,3,3,3),rand(8),rand(8,8),rand(8))
        def losses(p,b):
            a=F.silu(F.conv2d(b[0],p[0],p[1],padding=1)).mean((-1,-2))
            return F.cross_entropy(a@p[2]+p[3],b[1],reduction='none')
    else:
        x=rand(n,12,16);labels=torch.arange(n*12,device='mps').reshape(n,12)%8
        params=(rand(16,16),rand(16,16),rand(16,16),rand(16,32),rand(32,16),rand(16,8))
        def losses(p,b):
            a=b[0];q=a@p[0];k=a@p[1];v=a@p[2]
            attention=F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],dropout_p=0,is_causal=True)[:,0]
            a=F.layer_norm(a+attention,(16,))
            a=F.layer_norm(a+F.gelu(a@p[3])@p[4],(16,))
            return F.cross_entropy((a@p[5]).transpose(1,2),b[1],reduction='none').mean(-1)
    return params,(x,labels),losses


CONTRACT=SeparableContract(True,True,True,
    'Audited bounded test callbacks: no reduction across batch, fixed FP32 parameters, no random/stateful forward operations; shared parameters used in every example; losses and derivatives finite in tested cases.',True,True)


def update(params,grads,states,step,kind):
    with torch.no_grad():
        for i,(p,g) in enumerate(zip(params,grads)):
            if g is None:continue
            if kind=='sgd':p.add_(g,alpha=-1e-3)
            else:
                m,v=states[i];m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
                p.mul_(1-1e-3*.01)
                p.addcdiv_(m/(1-.9**step),(v/(1-.999**step)).sqrt()+1e-8,value=-1e-3)


def streaming_case(name,n=97,chunk=16,masked=False,train=True):
    torch.manual_seed(6601)
    params,batch,fn=architecture(name,n)
    weights=torch.ones(n,device='mps')
    if masked:weights=(torch.arange(n,device='mps')%4==0).to(torch.float32)
    program=compile_separable_objective(fn,contract=CONTRACT,chunk_size=chunk)
    out=program.value_and_grad(params,batch,weights=weights,measure_saved=True)
    value,gs,mem=native_value_grad(fn,params,batch,weights)
    errors=comparisons((out.loss,)+out.parameter_grads+out.input_grads,(value,)+gs)
    training=[]
    if train:
        for kind in ('sgd','adamw'):
            pa,pb=clone(params),clone(params)
            sa=[(torch.zeros_like(t),torch.zeros_like(t)) for t in pa]
            sb=[(torch.zeros_like(t),torch.zeros_like(t)) for t in pb]
            for step in range(1,4):
                va,ga,_=native_value_grad(fn,pa,batch,weights)
                ob=program.value_and_grad(pb,batch,weights=weights)
                update(pa,ga[:len(pa)],sa,step,kind);update(pb,ob.parameter_grads,sb,step,kind)
                es=comparisons((va,)+pa,(ob.loss,)+pb)
                state_errors=comparisons(tuple(t for pair in sa for t in pair),tuple(t for pair in sb for t in pair)) if kind=='adamw' else []
                training.append(dict(optimizer=kind,step=step,errors=es,state_errors=state_errors,
                                     passed=all(e['passed'] for e in es+state_errors)))
    return dict(name=name,batch=n,chunk=chunk,masked=masked,errors=errors,evidence=out.evidence,
                native_saved=mem,training=training,
                passed=all(e['passed'] for e in errors) and all(e['passed'] for e in training))


def gram_case(shape,seed,bias=True,activation='gelu',timing=False):
    torch.manual_seed(seed);m,k,n=shape
    x=rand(m,k);a=rand(k,k);w=rand(k,n);b=rand(n)
    y=rand(m,n,grad=False,scale=.3)
    params=(x,a,w,b) if bias else (x,a,w)
    act={'gelu':F.gelu,'silu':F.silu,'identity':lambda x:x}[activation]
    def loss(p,mode,ev=None):
        h=act(p[0]@p[1]);return quadratic_loss(h,p[2],y,p[3] if bias else None,accuracy=mode,evidence=ev)
    results=[];memory={};evidence=[]
    for mode in ('strict','algebraic'):
        with SavedTensorAudit(list(params+(y,))) as audit:z=loss(params,mode,evidence if mode=='algebraic' else None)
        results.append((z.detach(),)+torch.autograd.grad(z,params));memory[mode]=audit.summary()
    errors=comparisons(results[1],results[0])
    # Genuine complete parameter updates, including the nonlinear feature layer.
    states={mode:clone(params) for mode in ('strict','algebraic')};training=[]
    for step in range(3):
        zz=[]
        for mode,p in states.items():
            z=loss(p,mode);gs=torch.autograd.grad(z,p);zz.append(z.detach())
            with torch.no_grad():
                for t,g in zip(p[1:],gs[1:]):t.add_(g,alpha=-1e-3)
        es=comparisons((zz[1],)+states['algebraic'],(zz[0],)+states['strict'])
        training.append(dict(step=step,errors=es,passed=all(e['passed'] for e in es)))
    result=dict(shape=shape,seed=seed,activation=activation,bias=bias,errors=errors,evidence=evidence,
                plan=quadratic_plan(m,k,n,has_bias=bias,need_bias=bias),memory=memory,training=training,
                passed=all(e['passed'] for e in errors) and all(t['passed'] for t in training))
    if timing:
        def step(mode):
            p=states[mode];z=loss(p,mode);gs=torch.autograd.grad(z,p)
            with torch.no_grad():
                for t,g in zip(p[1:],gs[1:]):t.add_(g,alpha=-1e-3)
        times={mode:[] for mode in states};rng=random.Random(seed)
        for mode in states:
            for _ in range(2):step(mode)
        for _ in range(5):
            order=list(states);rng.shuffle(order)
            for mode in order:
                torch.mps.synchronize();start=time.perf_counter();step(mode);torch.mps.synchronize()
                times[mode].append(time.perf_counter()-start)
            time.sleep(.1)
        med={mode:statistics.median(t) for mode,t in times.items()}
        result['timing']=dict(seconds_median=med,samples=times,native_over_candidate=med['strict']/med['algebraic'],
             scope='Full nonlinear feature forward, objective, all original input/parameter gradients, SGD and runtime guards; eager native baseline; 2 warmups, 5 alternating synchronized samples')
    return result


def edge_cases():
    edges=[]
    # Zero fit, target gradients, higher derivatives and unfavorable dimensions.
    for name in ('zero_fit','target_grad','higher_order','unfavorable','noncontiguous','frozen_feature','stationary_weight'):
        torch.manual_seed(772)
        m,k,n=(9,16,7) if name=='unfavorable' else (128,4,256)
        h=rand(m,k,grad=name!='frozen_feature');w=rand(k,n)
        if name=='noncontiguous':h=rand(k,m).T.detach().requires_grad_()
        y=(h@w).detach() if name=='zero_fit' else rand(m,n,grad=name=='target_grad',scale=.3)
        raw=None
        if name=='stationary_weight':
            with torch.no_grad():
                h[m//2:].zero_();y=h@w;y[m//2:]=torch.randn_like(y[m//2:])*.5
                raw=compare((h.T@h)@w-h.T@y,torch.zeros_like(w))
            assert not raw['passed'],'The intended floating cancellation witness disappeared'
        order=2 if name=='higher_order' else 1;ev=[];out=[]
        active=tuple(t for t in (h,w,y) if t.requires_grad)
        for mode in ('strict','algebraic'):
            loss=quadratic_loss(h,w,y,accuracy=mode,derivative_order=order,evidence=ev if mode=='algebraic' else None)
            gs=torch.autograd.grad(loss,active,create_graph=order>1)
            if order>1:gs+=torch.autograd.grad(sum(g.square().sum() for g in gs),active)
            out.append((loss.detach(),)+gs)
        es=comparisons(out[1],out[0]);edges.append(dict(name=name,errors=es,evidence=ev,raw_unrepaired_gradient=raw,passed=all(e['passed'] for e in es)))
    # Frozen/unused parameters and inputs, zero weights, sum/weighted reduction.
    p=(rand(4,2),rand(2,grad=False),rand(7))
    b=(rand(17,4),rand(17,2,grad=False),rand(17,1))
    def losses(p,b):return (b[0]@p[0]+p[1]-b[1]).square().mean(-1)
    program=compile_separable_objective(losses,contract=CONTRACT,chunk_size=5)
    for name,weights,norm in [('all_zero',torch.zeros(17,device='mps'),'batch'),
        ('weighted_mean',torch.rand(17,device='mps'),'weight_sum'),('sum',torch.ones(17,device='mps'),'sum')]:
        ob=program.value_and_grad(p,b,weights=weights,normalization=norm)
        v,gs,_=native_value_grad(losses,p,b,weights,norm)
        es=comparisons((ob.loss,)+ob.parameter_grads+ob.input_grads,(v,)+gs)
        edges.append(dict(name=name,errors=es,passed=all(e['passed'] for e in es)))
    # Guard admission failures; these do not claim to infer arbitrary effects.
    for name,contract in [('batch_coupled',SeparableContract(False,True,True,'batch norm')),
                          ('random_state',SeparableContract(True,False,True,'dropout')),
                          ('mutation',SeparableContract(True,True,False,'running state'))]:
        try:compile_separable_objective(losses,contract=contract)
        except ValueError:edges.append(dict(name=name,rejected=True,passed=True))
        else:edges.append(dict(name=name,rejected=False,passed=False))
    return edges


def masked_timing(seed):
    torch.manual_seed(seed);n=1024
    params=(rand(128,256),rand(256),rand(256,256),rand(256),rand(256,64),rand(64))
    batch=(rand(n,128),torch.arange(n,device='mps')%64)
    weights=(torch.arange(n,device='mps')%16==0).float()
    def fn(p,b):return F.cross_entropy(F.gelu(F.gelu(b[0]@p[0]+p[1])@p[2]+p[3])@p[4]+p[5],b[1],reduction='none')
    program=compile_separable_objective(fn,contract=CONTRACT,chunk_size=64)
    ob=program.value_and_grad(params,batch,weights=weights,measure_saved=True)
    value,grads,memory=native_value_grad(fn,params,batch,weights)
    errors=comparisons((ob.loss,)+ob.parameter_grads+ob.input_grads,(value,)+grads)
    states={mode:clone(params) for mode in ('native','streamed')}
    def step(mode):
        p=states[mode]
        if mode=='native':
            z=(fn(p,batch)*weights).sum()/n
            gs=torch.autograd.grad(z,p+(batch[0],))[:len(p)]
        else:gs=program.value_and_grad(p,batch,weights=weights).parameter_grads
        with torch.no_grad():
            for t,g in zip(p,gs):
                if g is not None:t.add_(g,alpha=-1e-3)
    times={mode:[] for mode in states};rng=random.Random(seed)
    for mode in states:
        for _ in range(2):step(mode)
    for _ in range(5):
        order=list(states);rng.shuffle(order)
        for mode in order:
            torch.mps.synchronize();start=time.perf_counter();step(mode);torch.mps.synchronize()
            times[mode].append(time.perf_counter()-start)
        time.sleep(.1)
    med={mode:statistics.median(t) for mode,t in times.items()}
    return dict(seed=seed,passed=all(e['passed'] for e in errors),errors=errors,evidence=ob.evidence,
        native_saved=memory,seconds_median=med,samples=times,native_over_candidate=med['native']/med['streamed'],
        native_model_matrix_flops=6*n*(128*256+256*256+256*64),
        active_model_matrix_flops=6*64*(128*256+256*256+256*64),
        scope='A weighted objective with 1/16 active independent examples; includes full loss/all original input and parameter gradients/SGD and guards. Skipping padding tokens within attention is not justified by this example. 2 warmups and 5 synchronized shuffled rounds.')


def main():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU execution')
    report=dict(status='running',device='mps',torch_version=torch.__version__,cpu_ml_execution=False,cpu_fallback_enabled=False,
                gram=[],streaming=[],scaling=[],edges=[],masked_timing=[])
    path=ROOT/'reports/objective-gpu.json'
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    for i,name in enumerate(('gelu_mlp','silu_mlp','shared_residual','custom_sine','conv_silu','causal_attention')):
        print('GPU full objective',name,flush=True)
        report['streaming'].append(streaming_case(name,masked=i%2==1));save()
        torch.mps.synchronize();time.sleep(.2)
    for shape in ((128,8,512),(1024,16,2048),(2048,64,4096)):
        for seed in (6701,6702,6703):
            print('GPU Gram',shape,seed,flush=True)
            report['gram'].append(gram_case(shape,seed,activation='gelu' if seed%2 else 'silu',timing=shape[0]>128));save()
            torch.mps.synchronize();time.sleep(.2)
    for n in (128,512,2048):
        print('GPU memory scaling',n,flush=True)
        report['scaling'].append(streaming_case('gelu_mlp',n=n,chunk=32,train=False));save()
    report['edges']=edge_cases()
    for seed in (7801,7802,7803):
        print('GPU zero-weight elimination',seed,flush=True)
        report['masked_timing'].append(masked_timing(seed));save()
        torch.mps.synchronize();time.sleep(.2)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in (Path(__file__),ROOT/'ojasx2/objective_compiler.py',ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py')}
    passed=all(c['passed'] for key in ('gram','streaming','scaling','edges','masked_timing') for c in report[key])
    report['status']='passed_scoped_objective_gpu' if passed else 'failed_scoped_objective_gpu'
    report['scope']='Bounded FP32 MPS full objectives and all original gradients; exact-real rewrites with measured floating errors; no global IEEE/latency/total-memory guarantee. Guard declarations do not certify arbitrary callbacks.'
    save();print(report['status'],flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
