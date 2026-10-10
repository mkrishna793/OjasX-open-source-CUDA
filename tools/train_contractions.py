"""Deterministic synthetic FP32 classification qualification, not a dataset claim."""
import copy,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2 import make_backend


def main():
    torch.set_num_threads(2);torch.manual_seed(771)
    model=torch.nn.Sequential(torch.nn.Linear(64,128,bias=False),torch.nn.Linear(128,8,bias=False))
    data=torch.randn(1024,64);test=torch.randn(512,64)
    teacher=torch.randn(8,64)
    labels=(data@teacher.T).argmax(-1);targets=(test@teacher.T).argmax(-1)
    variants={'native':model,'joint':copy.deepcopy(model),'backward_only':copy.deepcopy(model)}
    backends={name:make_backend(accuracy='algebraic',contraction_policy=policy)
              for name,policy in [('joint','linear_chain'),('backward_only','linear_chain_backward')]}
    fns={'native':model,**{name:torch.compile(variants[name],backend=b,fullgraph=True) for name,b in backends.items()}}
    optimizers={name:torch.optim.SGD(fn.parameters(),lr=.05,momentum=.9) for name,fn in variants.items()}
    generator=torch.Generator().manual_seed(999)
    errors={name:[] for name in backends};losses={name:[] for name in variants}
    for step in range(80):
        indices=torch.randperm(len(data),generator=generator)[:128]
        for name,fn in fns.items():
            optimizers[name].zero_grad(set_to_none=True)
            logits=fn(data[indices]);loss=F.cross_entropy(logits,labels[indices]);loss.backward()
            losses[name].append(float(loss.detach()))
            if name!='native':
                grads=tuple(p.grad for p in variants[name].parameters())
                reference=tuple(p.grad for p in variants['native'].parameters())
                rel=max(float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)) for a,b in zip(grads,reference))
                errors[name].append(rel)
        for opt in optimizers.values():opt.step()
    baseline=fns['native'](test).detach()
    report={'steps':80,'optimizer':'SGD lr .05 momentum .9','synthetic_task':'8-class linear teacher, same two original dense weight matrices','device':'cpu','dtype':'float32','results':{}}
    for name,fn in fns.items():
        logits=fn(test).detach();pred=logits.argmax(-1)
        diff=logits-baseline
        item={'last_training_loss':losses[name][-1],'heldout_accuracy':float((pred==targets).float().mean()),
              'prediction_agreement_with_native':float((pred==baseline.argmax(-1)).float().mean()),
              'output_relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(baseline)),
              'output_max_abs':float(diff.abs().max()),
              'max_step_gradient_relative_l2':max(errors.get(name,[0.])),
              'within_sample_output_tolerance':bool(torch.allclose(logits,baseline,rtol=3e-4,atol=3e-5)),
              'backend':backends[name].evidence if name in backends else None}
        report['results'][name]=item
    report['qualification_passed']=all(r['within_sample_output_tolerance'] and r['max_step_gradient_relative_l2']<1e-4 and r['prediction_agreement_with_native']>=.99 for r in report['results'].values())
    report['scope']='Synthetic 80-step qualification only; no general convergence, real-dataset, distributed/AMP or optimizer-wide guarantee.'
    (ROOT/'reports/contraction-training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['qualification_passed']:raise SystemExit(1)


if __name__=='__main__':main()
