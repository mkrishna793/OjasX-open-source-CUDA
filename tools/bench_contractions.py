"""Isolated paired CPU tests: original chain, reassociation and strong baseline.

Unchanged original trainable weights; no SVD, factor truncation or new model.
"""
import argparse,json,statistics,time,sys,random
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2 import linear_chain,chain_cost,factored_gemm_pullback
from ojasx2.residuals import SavedTensorAudit
from ojasx2.contractions import _LinearChain


def errors(values,reference):
    data=[]
    for x,y in zip(values,reference):
        diff=(x-y).detach();denom=float(torch.linalg.vector_norm(y.detach()))
        data.append({'max_abs':float(diff.abs().max()),
                     'relative_l2':float(torch.linalg.vector_norm(diff))/max(denom,1e-30),
                     'within_sample_tolerance':bool(torch.allclose(x,y,rtol=3e-4,atol=3e-5)),
                     'raw_bits_equal':bool(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)))})
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=11)
    parser.add_argument('--output',default='reports/contraction-benchmark.json')
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(20261007)
    report={'device':'cpu','torch':torch.__version__,'threads':2,'repeats':args.repeats,
            'dtype':'float32','samples':[],
            'scope':'Isolated randomized-order paired samples, forward + input + both original weight gradients. No optimizer timing or process/GPU peak.',
            'accuracy_scope':'Exact real-algebra reassociation; sampled FP32 errors, no exhaustive bit guarantee.',
            'baseline_scope':'Original two native Linear nodes and plain PyTorch on the reassociated graph. Custom adjoints are not claimed better than equivalent native autograd.'}
    rng=random.Random(55)
    for m,k,n,p in [(128,256,512,16),(256,512,1024,32),(256,1024,2048,32),(128,256,512,256)]:
        x=(torch.randn(m,k)*.1).requires_grad_()
        w1=(torch.randn(n,k)*(.5/k**.5)).requires_grad_()
        w2=(torch.randn(p,n)*(.5/n**.5)).requires_grad_()
        g=torch.randn(m,p)*.1
        functions={'native':lambda:F.linear(F.linear(x,w1),w2),
                   'backward_only':lambda:linear_chain(x,w1,w2,'algebraic','backward_only'),
                   'joint':lambda:linear_chain(x,w1,w2,'algebraic','joint'),
                   'explicit_dagger':lambda:_LinearChain.apply(x,w1,w2,'joint'),
                   'reassociated_native_autograd':lambda:F.linear(x,w2@w1)}
        def work(name):
            y=functions[name]()
            return y,*torch.autograd.grad(y,(x,w1,w2),g)
        reference=tuple(z.detach() for z in work('native'))
        times={name:[] for name in functions}
        for name in functions:
            for _ in range(3):work(name)
        for _ in range(args.repeats):
            order=list(functions);rng.shuffle(order)
            for name in order:
                start=time.perf_counter();values=work(name);times[name].append(time.perf_counter()-start)
                del values
        sample={'shape_m_k_n_p':[m,k,n,p],'joint_cost_model':chain_cost(m,k,n,p),
                'backward_only_cost_model':chain_cost(m,k,n,p,mode='backward_only'),'variants':{}}
        for name in functions:
            with SavedTensorAudit([x,w1,w2]) as audit:y=functions[name]()
            torch.autograd.grad(y,(x,w1,w2),g);del y
            item={'median_joint_seconds':statistics.median(times[name]),'all_joint_seconds':times[name],
                  'saved':audit.summary(),'errors_output_input_w1_w2':errors(work(name),reference)}
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as prof:
                values=work(name)
            path=ROOT/f'reports/memory-contraction-{name}-{m}-{k}-{n}-{p}.json'
            prof.export_memory_timeline(str(path),device='cpu')
            _,categories=json.loads(path.read_text());totals=[sum(row) for row in categories]
            item['tracked_initial_bytes']=totals[0]
            item['tracked_incremental_peak_bytes']=max(totals)-totals[0]
            item['aten_gemm_events']={e.key:e.count for e in prof.key_averages() if e.key in ('aten::mm','aten::addmm','aten::bmm')}
            del values
            sample['variants'][name]=item
        base=sample['variants']['native']['median_joint_seconds']
        for item in sample['variants'].values():
            item['time_ratio_to_native']=item['median_joint_seconds']/base
            item['speedup_vs_native']=base/item['median_joint_seconds']
        report['samples'].append(sample);print('Completed',m,k,n,p,flush=True)
    # Compressing an arbitrary dense gradient has an explicit decomposition cost.
    torch.manual_seed(991)
    m,k,n=64,128,256
    a=torch.randn(m,k)*.1;b=torch.randn(k,n)*.1
    r=8;left=torch.randn(m,r)*.1;right=torch.randn(n,r)*.1
    structured=left@right.T;generic=torch.randn(m,n)*.1
    report['spectral_probe']=[]
    for name,g in [('provided_rank8',structured),('generic_dense',generic)]:
        start=time.perf_counter();u,s,vh=torch.linalg.svd(g,full_matrices=False)
        duration=time.perf_counter()-start
        exact=(g@b.T,a.T@g)
        sample={'case':name,'svd_seconds':duration,'singular_values':s.tolist(),'ranks':[]}
        for rank in [4,8,16,32,64]:
            l=u[:,:rank]*s[:rank];r=vh[:rank,:].T;ghat=l@r.T
            got=factored_gemm_pullback(a,b,l,r)
            eg=torch.linalg.vector_norm(g-ghat)
            bound_a=float(eg*torch.linalg.matrix_norm(b,ord=2))
            bound_b=float(torch.linalg.matrix_norm(a,ord=2)*eg)
            measured=(float(torch.linalg.vector_norm(got[0]-exact[0])),float(torch.linalg.vector_norm(got[1]-exact[1])))
            sample['ranks'].append({'rank':rank,'cotangent_relative_l2':float(eg/torch.linalg.vector_norm(g)),
                                   'gradient_relative_l2':[float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y)) for x,y in zip(got,exact)],
                                   'frobenius_error_bounds':[bound_a,bound_b],
                                   'observed_frobenius_errors':measured,
                                   'bounds_hold_with_fp32_slack':measured[0]<=bound_a+1e-5 and measured[1]<=bound_b+1e-5})
        report['spectral_probe'].append(sample)
    report['all_chain_samples_passed']=all(e['within_sample_tolerance'] for s in report['samples'] for v in s['variants'].values() for e in v['errors_output_input_w1_w2'])
    report['no_rank_approximation_enabled']=True
    (ROOT/args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
    if not report['all_chain_samples_passed']:raise SystemExit(1)


if __name__=='__main__':main()
