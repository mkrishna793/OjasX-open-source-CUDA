"""Measure ordinary FFN training with the strict pointwise-memory backend."""
import copy,json,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2 import make_backend
from ojasx2.residuals import SavedTensorAudit


class FFN(torch.nn.Module):
    def __init__(self,d,h):
        super().__init__()
        self.gate=torch.nn.Linear(d,h,bias=False)
        self.up=torch.nn.Linear(d,h,bias=False)
        self.down=torch.nn.Linear(h,d,bias=False)
    def forward(self,x):return x+self.down(F.silu(self.gate(x))*self.up(x))


def main():
    torch.manual_seed(20261006);torch.set_num_threads(2)
    result={'device':'cpu','torch':torch.__version__,'samples':[],
            'scope':'FP32 ordinary FFN, unchanged native GEMMs, pointwise recomputation; saved storage and profiler tensor peaks, not process RSS.'}
    for batch,d,h,depth in [(64,128,512,1),(128,512,2048,1),(128,1024,4096,1),(512,256,1024,8)]:
        model=torch.nn.Sequential(*(FFN(d,h) for _ in range(depth)))
        other=copy.deepcopy(model)
        backend=make_backend(memory_policy='pointwise')
        compiled=torch.compile(other,backend=backend,fullgraph=True)
        x=torch.randn(batch,d,requires_grad=True)
        sample={'batch':batch,'input_dim':d,'hidden_dim':h,'depth':depth,'variants':{}}
        outputs={}
        for name,fn,params in [('eager',model,list(model.parameters())),('memory_backend',compiled,list(other.parameters()))]:
            def work():
                y=fn(x)
                loss=y.square().mean()
                return y,*torch.autograd.grad(loss,(x,*params))
            for _ in range(2):work()
            times=[]
            for _ in range(7):
                start=time.perf_counter();work();times.append(time.perf_counter()-start)
            with SavedTensorAudit([x,*params]) as audit:
                y=fn(x);loss=y.square().mean()
            torch.autograd.grad(loss,(x,*params))
            outputs[name]=tuple(t.detach() for t in work())
            item={'saved':audit.summary(),'median_joint_seconds':statistics.median(times)}
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as p:
                tensors=work()
            path=ROOT/f'reports/memory-ffn-{name}-{batch}-{d}-{h}-{depth}.json'
            p.export_memory_timeline(str(path),device='cpu')
            _,categories=json.loads(path.read_text());totals=[sum(row) for row in categories]
            item['tracked_incremental_peak_bytes']=max(totals)-totals[0]
            item['aten_gemm_events']={e.key:e.count for e in p.key_averages() if e.key in ('aten::mm','aten::addmm','aten::bmm')}
            del tensors
            sample['variants'][name]=item
        sample['outputs_and_all_gradients_bit_equal']=all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for a,b in zip(outputs['eager'],outputs['memory_backend']))
        sample['backend']=backend.evidence
        base=sample['variants']['eager']['saved']['fresh_saved_storage_bytes']
        reduced=sample['variants']['memory_backend']['saved']['fresh_saved_storage_bytes']
        sample['fresh_saved_bytes_removed']=base-reduced
        sample['fresh_saved_fraction_removed']=(base-reduced)/base
        result['samples'].append(sample)
    result['all_bit_comparisons_passed']=all(s['outputs_and_all_gradients_bit_equal'] for s in result['samples'])
    (ROOT/'reports/memory-backend.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['all_bit_comparisons_passed']:raise SystemExit(1)


if __name__=='__main__':main()
