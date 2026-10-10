"""Bounded CPU scaling study, plus an explicit failed optimizer qualification."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from ojasx2 import rms_core_linear_chain, rms_core_cost, make_backend
from research_extensions import qualify
from test_workload_coverage import model_case, batchnorm_bias_negative_qualification


def main():
    torch.set_num_threads(2)
    result={'torch':torch.__version__,'device':'cpu','threads':2,'blocks':{},'models':{},
            'scope':'Synthetic shape/graph-family qualification. Not exhaustive ML coverage, datasets, distributed training or GPU performance. Profiler reports tensor allocations, not whole-process peak.'}
    dimensions=[(256,8,256,8),(1024,16,1024,16),(2048,32,4096,32),
                (2048,32,8192,64),(1024,128,512,128),(64,64,512,32),(1,16,1024,8)]
    for m,k,n,p in dimensions:
        torch.manual_seed(733)
        xs=tuple(a.requires_grad_() for a in (torch.randn(m,k),torch.randn(n,k)*.1,
                 torch.randn(p,n)*.1,torch.randn(n),torch.randn(p)*.1))
        evidence=[];rms_core_linear_chain(*xs,accuracy='algebraic',evidence=evidence)
        key=f'{m}-{k}-{n}-{p}'
        measurements=qualify('scale-'+key,
              {'native':lambda:rms_core_linear_chain(*xs),
               'core':lambda:rms_core_linear_chain(*xs,accuracy='algebraic')},
              xs,torch.randn(m,p),repeats=7)
        passed=all(e['relative_l2']<1e-5 for v in measurements.values() for e in v['errors_output_then_gradients'])
        result['blocks'][key]={'shape':[m,k,n,p],'selection':evidence[-1],
            'cost':rms_core_cost(m,k,n,p),'measurements':measurements,
            'relative_l2_threshold':1e-5,'qualified':passed}
        if not passed:raise RuntimeError(f'Norm qualification failed for {key}')
    for name in ('residual_tokens','standard_transformer_swiglu'):
        torch.manual_seed(702);torch._dynamo.reset()
        model,x,expected=model_case(name)
        x.requires_grad_()
        backend=make_backend(accuracy='algebraic',contraction_policy='rms_core')
        compiled=torch.compile(model,backend=backend,fullgraph=True)
        y=compiled(x)
        measurements=qualify('model-'+name,
             {'native':lambda:model(x),'core':lambda:compiled(x)},
             (x,*model.parameters()),torch.randn_like(y),repeats=7)
        assert backend.evidence[-1]['rms_core_rewrites']==expected
        assert all(e['relative_l2']<1e-4 for v in measurements.values() for e in v['errors_output_then_gradients'])
        result['models'][name]={'compiler':backend.evidence[-1],'measurements':measurements,
                               'qualification':'Forward and all gradients; relative L2 <1e-4. Compilation excluded from steady-state timing.'}
        torch._dynamo.reset()
    result['negative_qualification']=batchnorm_bias_negative_qualification()
    assert not result['negative_qualification']['qualified']
    (ROOT/'reports/workload-scaling.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'blocks':{k:{'path':v['selection']['path'],'speedup':v['measurements']['core']['speedup']}
                              for k,v in result['blocks'].items()},
                      'models':{k:{'rewrites':v['compiler']['rms_core_rewrites'],'speedup':v['measurements']['core']['speedup']}
                              for k,v in result['models'].items()},
                      'negative_qualification':result['negative_qualification']},indent=2))


if __name__=='__main__':main()
