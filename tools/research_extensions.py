"""Isolated CPU qualification of longer contraction paths and nonlinear structure."""
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path
import sympy as sp
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2 import (matrix_chain, matrix_chain_plan, linear_chain, relu_linear_chain,
                    SavedTensorAudit, layernorm_input_pullback, attention_softmax_row_scalar,
                    layernorm_linear_chain, layernorm_chain_cost,
                    gram_layernorm_linear_chain, gram_layernorm_chain_cost)


def errors(values, reference):
    values = tuple(x.detach() for x in values)
    reference = tuple(x.detach() for x in reference)
    return [{'max_abs':float((x-y).abs().max()),
             'relative_l2':float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-30)),
             'within_tolerance':bool(torch.allclose(x,y,rtol=3e-4,atol=3e-5)),
             'bits_equal':bool(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)))}
            for x,y in zip(values,reference)]


def qualify(name,functions,inputs,g,repeats=11):
    def work(key):
        y=functions[key]()
        return y,*torch.autograd.grad(y,inputs,g)
    reference=tuple(x.detach() for x in work('native'))
    times={k:[] for k in functions};rng=random.Random(117)
    for key in functions:
        for _ in range(3):work(key)
    for _ in range(repeats):
        order=list(functions);rng.shuffle(order)
        for key in order:
            start=time.perf_counter();values=work(key);times[key].append(time.perf_counter()-start);del values
    report={}
    for key in functions:
        with SavedTensorAudit(inputs) as audit:y=functions[key]()
        torch.autograd.grad(y,inputs,g);del y
        item={'median_joint_seconds':statistics.median(times[key]),'all_joint_seconds':times[key],
              'errors_output_then_gradients':errors(work(key),reference),'saved':audit.summary()}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as prof:
            values=work(key)
        path=ROOT/f'reports/memory-research-{name}-{key}.json';prof.export_memory_timeline(str(path),device='cpu')
        _,categories=json.loads(path.read_text());totals=[sum(row) for row in categories]
        item['tracked_initial_bytes']=totals[0];item['tracked_incremental_peak_bytes']=max(totals)-totals[0]
        item['aten_gemm_events']={e.key:e.count for e in prof.key_averages() if e.key in ('aten::mm','aten::addmm','aten::bmm')}
        del values;report[key]=item
    baseline=report['native']['median_joint_seconds']
    for item in report.values():item['speedup']=baseline/item['median_joint_seconds']
    print('Completed',name,flush=True)
    return report


def symbolic_checks():
    # Real softmax directional derivative, with positive exponential variables.
    q=sp.symbols('q0:4',positive=True);dq=sp.symbols('v0:4',real=True);total=sum(q)
    p=[a/total for a in q]
    softmax=[sp.simplify(sum(sp.diff(p[i],q[j])*q[j]*dq[j] for j in range(4))-
                         p[i]*(dq[i]-sum(p[j]*dq[j] for j in range(4))))==0 for i in range(4)]
    # LayerNorm input Jacobian, including positive epsilon; exact symbolic differentiation.
    xs=sp.symbols('x0:3',real=True);eps=sp.symbols('eps',positive=True);mean=sum(xs)/3
    c=[x-mean for x in xs];r=(sum(a*a for a in c)/3+eps)**sp.Rational(-1,2)
    layernorm=[sp.simplify(sp.diff(c[i]*r,xs[j])-r*((1 if i==j else 0)-sp.Rational(1,3)-r*r*c[i]*c[j]/3))==0
               for i in range(3) for j in range(3)]
    return {'softmax_directional_derivative_4_coordinates':all(softmax),
            'layernorm_jacobian_3_coordinates_with_epsilon':all(layernorm),
            'scope':'Exact SymPy finite symbolic instances, not universal Lean primitive derivative certificates'}


def train_common_mask():
    torch.manual_seed(731)
    model=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.ReLU(),torch.nn.Linear(64,4))
    with torch.no_grad():
        model[0].weight.mul_(.1);model[0].bias.copy_(torch.tensor([2. if i%2 else -2. for i in range(64)]))
        model[2].weight.mul_(.1);model[2].bias.zero_()
    other=copy.deepcopy(model);a=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    b=torch.optim.SGD(other.parameters(),lr=.001,momentum=.9);evidence=[];maximum=0.
    for _ in range(40):
        x=torch.randn(32,16)*.1;target=torch.randn(32,4)*.1
        ya=model(x);yb=relu_linear_chain(x,other[0].weight,other[2].weight,other[0].bias,other[2].bias,'algebraic',evidence)
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        (ya-target).square().mean().backward();(yb-target).square().mean().backward()
        for p,q in zip(model.parameters(),other.parameters()):
            maximum=max(maximum,float(torch.linalg.vector_norm(p.grad-q.grad)/torch.linalg.vector_norm(p.grad).clamp_min(1e-30)))
        a.step();b.step()
    output_errors=errors((yb.detach(),),(ya.detach(),))[0]
    param_max=max(float((p-q).abs().max()) for p,q in zip(model.parameters(),other.parameters()))
    return {'steps':40,'optimizer':'SGD .001, momentum .9','task':'Constructed common-mask regression, both biases and original dense weights trainable',
            'selected_steps':sum(e['path']=='common_mask' for e in evidence),'max_gradient_relative_l2':maximum,
            'final_sample_output':output_errors,'max_parameter_abs_difference':param_max,
            'passed':maximum<1e-4 and output_errors['within_tolerance'] and param_max<1e-5,
            'scope':'Favorable synthetic trajectory; no real-dataset or generic-mask convergence guarantee'}


def train_layernorm():
    torch.manual_seed(632)
    model=torch.nn.Sequential(torch.nn.Linear(16,256,bias=False),torch.nn.LayerNorm(256),torch.nn.Linear(256,8))
    other=copy.deepcopy(model);third=copy.deepcopy(model)
    a=torch.optim.AdamW(model.parameters(),lr=1e-3);b=torch.optim.AdamW(other.parameters(),lr=1e-3);d=torch.optim.AdamW(third.parameters(),lr=1e-3)
    maximum=0.;evidence=[]
    gram_max=0.;gram_evidence=[]
    for _ in range(40):
        x=torch.randn(64,16)*.2;target=torch.randn(64,8)*.2
        ya=model(x);yb=layernorm_linear_chain(x,other[0].weight,other[2].weight,other[1].weight,other[1].bias,other[2].bias,
                                              accuracy='algebraic',evidence=evidence)
        yg=gram_layernorm_linear_chain(x,third[0].weight,third[2].weight,third[1].weight,third[1].bias,third[2].bias,
                                       accuracy='algebraic',evidence=gram_evidence)
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True)
        (ya-target).square().mean().backward();(yb-target).square().mean().backward();(yg-target).square().mean().backward()
        for p,q in zip(model.parameters(),other.parameters()):
            maximum=max(maximum,float(torch.linalg.vector_norm(p.grad-q.grad)/torch.linalg.vector_norm(p.grad).clamp_min(1e-30)))
        for p,q in zip(model.parameters(),third.parameters()):
            gram_max=max(gram_max,float(torch.linalg.vector_norm(p.grad-q.grad)/torch.linalg.vector_norm(p.grad).clamp_min(1e-30)))
        a.step();b.step();d.step()
    out=errors((yb.detach(),),(ya.detach(),))[0]
    param_max=max(float((p-q).abs().max()) for p,q in zip(model.parameters(),other.parameters()))
    gram_out=errors((yg.detach(),),(ya.detach(),))[0]
    gram_param=max(float((p-q).abs().max()) for p,q in zip(model.parameters(),third.parameters()))
    return {'steps':40,'optimizer':'AdamW .001','task':'Random Gaussian regression, Linear16->LayerNorm256->Linear8',
            'selected_steps':sum(e['path']=='structured_layernorm' for e in evidence),
            'max_gradient_relative_l2':maximum,'final_sample_output':out,'max_parameter_abs_difference':param_max,
            'gram':{'selected_steps':sum(e['path']=='gram_layernorm' for e in gram_evidence),'max_gradient_relative_l2':gram_max,
                    'final_sample_output':gram_out,'max_parameter_abs_difference':gram_param},
            'passed':maximum<1e-4 and out['within_tolerance'] and param_max<1e-5 and gram_max<1e-4 and gram_out['within_tolerance'] and gram_param<1e-5,
            'scope':'Short synthetic trajectory, all original weights/normalization parameters/bias trainable; no real-dataset accuracy guarantee'}


def main():
    torch.set_num_threads(2);torch.manual_seed(20261008)
    report={'device':'cpu','torch':torch.__version__,'threads':2,'dtype':'float32','repeats':11,
            'scope':'Forward and all original leaf gradients; randomized paired samples; no optimizer timing or GPU/process memory',
            'linear_chains':[],'relu_chains':[],'layernorm_chains':[]}
    for dims in [(128,256,512,1024,16),(256,512,1024,2048,32),(128,256,512,256,512)]:
        xs=tuple((torch.randn(a,b)*(.5/a**.5)).requires_grad_() for a,b in zip(dims,dims[1:]))
        funcs={'native':lambda:matrix_chain(xs),'planned':lambda:matrix_chain(xs,'algebraic'),
               'old_pair_rule':lambda:linear_chain(xs[0],xs[1].T,xs[2].T,'algebraic')@xs[3]}
        sample={'dimensions':dims,'plan':matrix_chain_plan(dims),'variants':qualify('chain-'+'-'.join(map(str,dims)),funcs,xs,torch.randn(dims[0],dims[-1])*.1)}
        report['linear_chains'].append(sample)
    for case in ['common_mask','random_distinct_masks']:
        m,k,n,p=512,1024,2048,32
        x=(torch.randn(m,k)*.1).requires_grad_();w1=(torch.randn(n,k)*(.5/k**.5)).requires_grad_();w2=(torch.randn(p,n)*(.5/n**.5)).requires_grad_()
        b1=(torch.tensor([3. if i%2 else -3. for i in range(n)]) if case=='common_mask' else torch.zeros(n)).requires_grad_()
        b2=torch.zeros(p,requires_grad=True);evidence=[]
        funcs={'native':lambda:F.linear(F.relu(F.linear(x,w1,b1)),w2,b2),
               'guarded':lambda:relu_linear_chain(x,w1,w2,b1,b2,'algebraic',evidence)}
        sample={'case':case,'shape_m_k_n_p':[m,k,n,p],
                'variants':qualify('relu-'+case,funcs,(x,w1,w2,b1,b2),torch.randn(m,p)*.1),
                'observed_path':evidence[-1]['path']}
        mask=(F.linear(x,w1,b1)>0).float()
        sample['unique_mask_rows']=int(torch.unique(mask,dim=0).shape[0])
        # Full numerical rank of smaller sampled derivative masks, not SVD truncation.
        sample['sampled_mask_matrix_rank_64x128']=int(torch.linalg.matrix_rank(mask[:64,:128]).item())
        report['relu_chains'].append(sample)
    for m,k,n,p in [(512,32,2048,16),(512,64,2048,16),(128,256,512,16)]:
        x=(torch.randn(m,k)*.1).requires_grad_();w1=(torch.randn(n,k)*(.5/k**.5)).requires_grad_();w2=(torch.randn(p,n)*(.5/n**.5)).requires_grad_()
        gamma=torch.ones(n,requires_grad=True);beta=(torch.randn(n)*.1).requires_grad_();bias=torch.zeros(p,requires_grad=True);evidence=[]
        funcs={'native':lambda:F.linear(F.layer_norm(F.linear(x,w1),(n,),gamma,beta),w2,bias),
               'structured':lambda:layernorm_linear_chain(x,w1,w2,gamma,beta,bias,accuracy='algebraic',evidence=evidence),
               'gram':lambda:gram_layernorm_linear_chain(x,w1,w2,gamma,beta,bias,accuracy='algebraic')}
        report['layernorm_chains'].append({'shape_m_k_n_p':[m,k,n,p],'cost_model':layernorm_chain_cost(m,k,n,p),
            'gram_cost_model':gram_layernorm_chain_cost(m,k,n,p),
            'variants':qualify('layernorm-'+str(k)+'-'+str(m),funcs,(x,w1,w2,gamma,beta,bias),torch.randn(m,p)*.1),
            'observed_path':evidence[-1]['path']})
    # Reusing an already available attention output is materially cheaper than
    # constructing dP solely for its row scalar. Full attention is not timed.
    p=torch.randn(256,2048).softmax(-1);v=torch.randn(2048,64);go=torch.randn(256,64);o=p@v
    timing={};values={}
    for name,fn in {'explicit_dp':lambda:(p*(go@v.T)).sum(-1,keepdim=True),
                    'reuse_output':lambda:attention_softmax_row_scalar(o,go)}.items():
        for _ in range(3):fn()
        runs=[]
        for _ in range(11):
            start=time.perf_counter();values[name]=fn();runs.append(time.perf_counter()-start)
        timing[name]=statistics.median(runs)
    report['attention_row_scalar']={'shape_m_n_d':[256,2048,64],'median_seconds':timing,
        'errors':errors((values['reuse_output'],),(values['explicit_dp'],))[0],
        'scope':'Isolated known identity with O already supplied; not SDPA or whole-attention speedup; native optimized attention already uses this structure'}
    report['symbolic_checks']=symbolic_checks();report['training']=train_common_mask();report['layernorm_training']=train_layernorm()
    report['qualification_passed']=all(e['within_tolerance'] for group in ['linear_chains','relu_chains','layernorm_chains'] for s in report[group] for v in s['variants'].values() for e in v['errors_output_then_gradients']) and report['training']['passed'] and report['layernorm_training']['passed'] and all(v for k,v in report['symbolic_checks'].items() if k!='scope') and report['attention_row_scalar']['errors']['within_tolerance']
    (ROOT/'reports/research-extensions.json').write_text(json.dumps(report,indent=2));print(json.dumps({'qualification_passed':report['qualification_passed'],'training':report['training'],'symbolic_checks':report['symbolic_checks']},indent=2))
    if not report['qualification_passed']:raise SystemExit(1)


if __name__=='__main__':main()
