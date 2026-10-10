"""Medium/large CPU workload measurements with an explicit dense baseline."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2 import streamed_gemm
from ojasx2.bilinear import winograd_forward
from ojasx2.streamed import streamed_cost
from ojasx2.residuals import SavedTensorAudit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',type=int,nargs='+',default=[256,512,1024])
    p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output')
    args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261006)
    report={'device':args.device,'torch':torch.__version__,'architecture':platform.machine(),
            'threads':2,'repeats':args.repeats,'dtype':'float32','samples':[],
            'memory_scope':'Profiler-tracked tensor allocations, not process/GPU resident peak. Recording can perturb lifetime.',
            'timing_scope':'Warm CPU or selected-device forward plus both gradients; single local run, not general speed guarantee.'}
    def sync():
        if args.device=='mps':torch.mps.synchronize()
        if args.device=='cuda':torch.cuda.synchronize()
    if args.device=='mps' and not torch.backends.mps.is_available():
        report['blocked']='MPS inaccessible from this process'
    elif args.device=='cuda' and not torch.cuda.is_available():
        report['blocked']='CUDA inaccessible from this process'
    else:
        for n in args.sizes:
            a=(torch.randn(n,n,device=args.device)*0.1).requires_grad_()
            b=(torch.randn(n,n,device=args.device)*0.1).requires_grad_()
            g=torch.randn(n,n,device=args.device)*0.1
            fns={'dense':lambda:a@b,'cached_winograd':lambda:winograd_forward(a,b)[1],
                 'operand_tape':lambda:streamed_gemm(a,b,'algebraic')}
            golden=fns['dense']()
            grads=torch.autograd.grad(golden,(a,b),g)
            ref=(golden.detach(),*(z.detach() for z in grads))
            sample={'n':n,'scalar_model':streamed_cost(n,n,n),'variants':{}}
            for name,fn in fns.items():
                def work():
                    y=fn()
                    return y,*torch.autograd.grad(y,(a,b),g)
                for _ in range(2):work()
                durations=[]
                for _ in range(args.repeats):
                    sync();start=time.perf_counter();result=work();sync()
                    durations.append(time.perf_counter()-start)
                    del result
                with SavedTensorAudit([a,b]) as audit:y=fn()
                torch.autograd.grad(y,(a,b),g)
                del y
                result=work()
                errors=[{'max_abs':float((x-z).detach().abs().max()),
                         'relative_l2':float(torch.linalg.vector_norm((x-z).detach())/torch.linalg.vector_norm(z.detach()))}
                        for x,z in zip(result,ref)]
                within=all(torch.allclose(x,z,rtol=5e-4,atol=5e-4) for x,z in zip(result,ref))
                del result
                item={'median_joint_seconds':statistics.median(durations),'all_joint_seconds':durations,
                      'saved':audit.summary(),'errors_y_da_db':errors,'within_sample_tolerance':within}
                if args.device=='cpu':
                    try:
                        path=ROOT/f'reports/memory-{name}-{n}.json'
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                            profile_memory=True,record_shapes=True,with_stack=True) as prof:
                            result=work()
                        prof.export_memory_timeline(str(path),device='cpu')
                        _,categories=json.loads(path.read_text())
                        totals=[sum(row) for row in categories]
                        item['tracked_tensor_peak_bytes']=max(totals)
                        item['tracked_initial_bytes']=totals[0]
                        item['tracked_incremental_peak_bytes']=max(totals)-totals[0]
                        item['aten_gemm_events']={e.key:e.count for e in prof.key_averages() if e.key in ('aten::mm','aten::bmm','aten::addmm')}
                        del result
                    except Exception as e:item['profiler_error']=str(e)
                sample['variants'][name]=item
            base=sample['variants']['dense']['median_joint_seconds']
            for item in sample['variants'].values():item['joint_time_ratio_to_dense']=item['median_joint_seconds']/base
            report['samples'].append(sample)
            print('Completed',n,flush=True)
    report['all_numerical_samples_passed']=all(v['within_sample_tolerance'] for s in report['samples'] for v in s['variants'].values()) if report['samples'] else False
    target=ROOT/(args.output or f'reports/streamed-benchmark-{args.device}.json')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report.get('blocked') and not report['all_numerical_samples_passed']:raise SystemExit(1)


if __name__=='__main__':main()
