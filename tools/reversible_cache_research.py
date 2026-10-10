"""Math-guided cache/replay experiments; keep slower modes and failures visible."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ojasx2 import reversible_coupling_chain, SavedTensorAudit
from ojasx2.cached_reverse import cached_reverse_cost


def sample(depth,m,d,device):
    torch.manual_seed(1012)
    xs=tuple(torch.randn(m,d).to(device).requires_grad_() for _ in range(2))
    blocks=tuple(tuple((torch.randn(*s)*.1).to(device).requires_grad_()
                       for s in ((d,d),(d,),(d,d),(d,))) for _ in range(depth))
    return xs,blocks


def modes(device):
    result={'native':dict(accuracy='strict'), 'replay':dict(accuracy='algebraic'),
            'cache_native':dict(accuracy='algebraic',residual_policy='preactivation')}
    if device=='mps':
        result.update({'cache_metal':dict(accuracy='algebraic',residual_policy='preactivation',pointwise_backend='metal'),
            'hybrid_metal':dict(accuracy='algebraic',residual_policy='preactivation',pointwise_backend='metal',cache_interval=4)})
    return result


def sync(device):
    if device=='mps':torch.mps.synchronize()


def measure(depth,m,d,device):
    xs,blocks=sample(depth,m,d,device)
    inputs=(*xs,*(v for b in blocks for v in b))
    incoming=tuple(torch.randn_like(x) for x in xs)
    options=modes(device)
    def forward(mode):return reversible_coupling_chain(*xs,blocks,activation='silu',scale=.1,**options[mode])
    def work(mode):
        ys=forward(mode)
        return (*ys,*torch.autograd.grad(ys,inputs,incoming))
    native=tuple(v.detach().cpu().double() for v in work('native'))
    result={}
    for mode in options:
        values=work(mode)
        errors=[]
        for a,b in zip(values,native):
            assert a.device.type==device
            a=a.detach().cpu().double()
            errors.append({'max_abs':float((a-b).abs().max()),
                'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)),
                'entrywise_pass':bool(torch.allclose(a,b,rtol=2e-4,atol=2e-4))})
        with SavedTensorAudit(list(inputs)) as audit:ys=forward(mode)
        torch.autograd.grad(ys,inputs,incoming)
        result[mode]={'output_and_all_gradient_errors':errors,
            'max_relative_l2':max(e['relative_l2'] for e in errors),
            'passed':all(e['entrywise_pass'] and e['relative_l2']<2e-4 for e in errors),
            'saved':audit.summary()}
        del values,ys
    timings={mode:[] for mode in options}
    for mode in options:
        for _ in range(3):work(mode)
    sync(device);rng=random.Random(1013)
    for _ in range(9):
        order=list(options);rng.shuffle(order)
        for mode in order:
            sync(device);start=time.perf_counter()
            for _ in range(2):
                v=work(mode);del v
            sync(device)
            timings[mode].append((time.perf_counter()-start)*500)
    for mode in options:
        med=statistics.median(timings[mode]);result[mode]['median_joint_ms']=med
        result[mode]['samples_ms']=timings[mode]
    for mode in options:
        r=result[mode]
        r['speedup_vs_native']=result['native']['median_joint_ms']/r['median_joint_ms']
        r['speedup_vs_old_replay']=result['replay']['median_joint_ms']/r['median_joint_ms']
        r['fresh_tape_reduction_percent']=100*(1-r['saved']['fresh_saved_storage_bytes']/result['native']['saved']['fresh_saved_storage_bytes'])
    return {'depth':depth,'m':m,'d':d,'activation':'silu','scale':.1,
        'full_cache_cost':cached_reverse_cost(depth,depth,m,d),
        'hybrid_cost':cached_reverse_cost(depth,(depth+3)//4,m,d),'modes':result}


def train(device):
    xs,blocks=sample(8,64,16,device)
    other=tuple(tuple(v.detach().clone().requires_grad_() for v in b) for b in blocks)
    pa=[v for b in blocks for v in b];pb=[v for b in other for v in b]
    oa=torch.optim.AdamW(pa,lr=.001);ob=torch.optim.AdamW(pb,lr=.001)
    backend='metal' if device=='mps' else 'native'
    max_grad=max_param=max_moment=0.;passed=True
    for step in range(30):
        x1,x2=(torch.randn_like(x).detach() for x in xs)
        target=(torch.randn_like(x1),torch.randn_like(x2))
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        a=reversible_coupling_chain(x1,x2,blocks,activation='silu')
        b=reversible_coupling_chain(x1,x2,other,activation='silu',accuracy='algebraic',
                                    residual_policy='preactivation',pointwise_backend=backend)
        sum((y-t).square().mean() for y,t in zip(a,target)).backward()
        sum((y-t).square().mean() for y,t in zip(b,target)).backward()
        for u,v in zip(pa,pb):
            norm=float((torch.linalg.vector_norm(u.grad-v.grad)/torch.linalg.vector_norm(u.grad).clamp_min(1e-30)).detach())
            max_grad=max(max_grad,norm);passed &= norm<2e-4
        oa.step();ob.step()
        for u,v in zip(pa,pb):
            diff=float((u-v).detach().abs().max());max_param=max(max_param,diff);passed &= diff<2e-4
            for key in ('exp_avg','exp_avg_sq'):
                delta=float((oa.state[u][key]-ob.state[v][key]).abs().max())
                max_moment=max(max_moment,delta);passed &= delta<2e-4
    return {'steps':30,'optimizer':'AdamW lr=.001, default eps=1e-8','depth':8,
        'pointwise_backend':backend,'max_gradient_relative_l2':max_grad,
        'max_parameter_abs_delta':max_param,'max_optimizer_moment_abs_delta':max_moment,'passed':bool(passed)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args();device=args.device
    if device=='mps':
        import os
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='1':raise SystemExit('Disable CPU fallback')
        if not torch.backends.mps.is_available():raise SystemExit('MPS inaccessible in this execution context; no CPU substitution')
    torch.set_num_threads(2)
    cases=[measure(l,m,d,device) for l,m,d in ((16,128,32),(64,128,32),(16,512,128))]
    training=train(device)
    report={'status':'passed' if all(v['passed'] for c in cases for v in c['modes'].values()) and training['passed'] else 'failed',
        'device':device,'torch':torch.__version__,'dtype':'float32','cpu_fallback_enabled':False,
        'contract':{'rtol':2e-4,'atol':2e-4,'relative_l2_max':2e-4},
        'timing_scope':'Forward and all first-order input/parameter gradients; 3 warmups, 9 shuffled rounds, 2 synchronized executions/sample; excludes optimizer',
        'memory_scope':'Fresh forward-retained autograd storage only; not GPU/process peak or full optimizer residency',
        'cases':cases,'training':training,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('ojasx2','tools','tests','lean/OjasXMath') for p in sorted((ROOT/folder).glob('*')) if p.suffix in ('.py','.lean')},
        'limits':['Requires existing additive coupling architecture','No universal FP reconstruction guarantee',
            'Full cache matches native GEMM count, not fewer than native dense backprop',
            'SiLU Metal kernel is FP32 and tolerance-qualified; higher derivatives require strict mode']}
    target=ROOT/f'reports/reversible-cache-{device}.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'device':device,'cases':[
        {'depth':c['depth'],'m':c['m'],'d':c['d'],'modes':{name:{k:v[k] for k in
        ('median_joint_ms','speedup_vs_native','speedup_vs_old_replay','fresh_tape_reduction_percent','max_relative_l2')}
         for name,v in c['modes'].items()}} for c in cases],'training':training},indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
