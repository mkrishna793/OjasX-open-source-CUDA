"""CPU measurements of reversible tape versus local replay overhead."""
import json
import random
import statistics
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ojasx2 import reversible_coupling_chain, SavedTensorAudit


def sample(depth,m=128,d=32):
    torch.manual_seed(812)
    xs=tuple(torch.randn(m,d,requires_grad=True) for _ in range(2))
    blocks=tuple(tuple((torch.randn(*shape)*.1).requires_grad_()
                       for shape in ((d,d),(d,),(d,d),(d,))) for _ in range(depth))
    return xs,blocks


def measure(depth):
    xs,blocks=sample(depth)
    inputs=(*xs,*(v for b in blocks for v in b));cotangents=tuple(torch.randn_like(x) for x in xs)
    def work(mode):
        ys=reversible_coupling_chain(*xs,blocks,accuracy=mode)
        return (*ys,*torch.autograd.grad(ys,inputs,cotangents))
    reference=tuple(v.detach() for v in work('strict'))
    times={m:[] for m in ('strict','algebraic')};rng=random.Random(813)
    for m in times:
        for _ in range(2):work(m)
    for _ in range(7):
        order=list(times);rng.shuffle(order)
        for m in order:
            start=time.perf_counter();v=work(m);times[m].append(time.perf_counter()-start);del v
    result={}
    for m in times:
        values=work(m)
        errors=[float(torch.linalg.vector_norm(a.detach()-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)) for a,b in zip(values,reference)]
        with SavedTensorAudit(list(inputs)) as audit:
            ys=reversible_coupling_chain(*xs,blocks,accuracy=m)
        # Forward tape only; transient backward replay is intentionally separate.
        torch.autograd.grad(ys,inputs,cotangents)
        if max(errors)>1e-4:raise RuntimeError('Stable-regime relative-L2 qualification failed')
        result[m]={'median_joint_ms':statistics.median(times[m])*1000,
                   'max_output_gradient_relative_l2':max(errors),'saved':audit.summary()}
    result['speedup']=result['strict']['median_joint_ms']/result['algebraic']['median_joint_ms']
    return result


def train():
    xs,blocks=sample(8,m=32,d=8)
    other=tuple(tuple(v.detach().clone().requires_grad_() for v in b) for b in blocks)
    pa=[v for b in blocks for v in b];pb=[v for b in other for v in b]
    oa=torch.optim.AdamW(pa,lr=.001);ob=torch.optim.AdamW(pb,lr=.001);max_grad=0.
    for _ in range(30):
        x1,x2=(torch.randn_like(x).detach() for x in xs)
        target=(torch.randn_like(x1),torch.randn_like(x2))
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        a=reversible_coupling_chain(x1,x2,blocks)
        b=reversible_coupling_chain(x1,x2,other,accuracy='algebraic')
        sum((v-t).square().mean() for v,t in zip(a,target)).backward()
        sum((v-t).square().mean() for v,t in zip(b,target)).backward()
        for u,v in zip(pa,pb):
            max_grad=max(max_grad,float(torch.linalg.vector_norm(u.grad-v.grad)/torch.linalg.vector_norm(u.grad).clamp_min(1e-30)))
        oa.step();ob.step()
    drift=max(float((u-v).detach().abs().max()) for u,v in zip(pa,pb))
    passed=max_grad<1e-4 and drift<1e-5
    if not passed:raise RuntimeError('Reversible synthetic optimizer qualification failed')
    return {'steps':30,'optimizer':'AdamW lr=.001','depth':8,'maximum_gradient_relative_l2':max_grad,
            'maximum_parameter_abs_drift':drift,'qualified':passed,'scope':'One synthetic stable-scale trajectory; not universal convergence'}


if __name__=='__main__':
    torch.set_num_threads(2)
    report={'device':'cpu','dtype':'float32','scale':.1,'activation':'gelu','m':128,'d_per_split':32,
            'depths':{str(d):measure(d) for d in (4,16,64)},'training':train(),
            'scope':'Forward+all first-order gradients. Forward-retained storage excludes supplied inputs/parameter refs; not total process/GPU peak. Replay adds computation.'}
    (ROOT/'reports/reversible-research.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
