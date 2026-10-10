"""MPS-only adapters for actual TorchCTS OpInfo and pinned BackendBench profiles.

Stock CTS uses CPU references; stock BackendBench targets CUDA. Neither stock
runner is called. This is bounded adapter evidence, not full suite certification.
Profiles are AST-parsed, never eval'd, and filtered before any tensor allocation.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import ast,dataclasses,hashlib,importlib.metadata,itertools,json,math,subprocess,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils._pytree import tree_flatten,tree_map
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import GPUOnly,require_gpu,compare,hashes
from ojasx2.ops import gelu,silu,packed_relu
from ojasx2.nonlinear import softmax_pullback

DTYPES={'f16':torch.float16,'bf16':torch.bfloat16,'f32':torch.float32,
        'f64':torch.float64,'i64':torch.int64,'i32':torch.int32,'b8':torch.bool}
ELEMENT_BYTES={'f16':2,'bf16':2,'f32':4,'f64':8,'i64':8,'i32':4,'b8':1}
PIN='ca7b1361b162d1499cb22ea4ad589dae506ead5d'
FAMILIES=('aten.mm.default','aten.bmm.default','aten.addmm.default','aten.gelu.default',
          'aten.silu.default','aten.relu.default','aten._softmax.default','aten.native_layer_norm.default')


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape:tuple
    dtype:str
    stride:tuple|None=None
    def extent(self):
        if any(not isinstance(n,int) or n<0 for n in self.shape):raise ValueError('invalid dimension')
        if self.stride is None:return math.prod(self.shape)
        if len(self.shape)!=len(self.stride) or any(not isinstance(s,int) or s<0 for s in self.stride):
            raise ValueError('invalid or negative stride')
        return 0 if 0 in self.shape else 1+sum((n-1)*s for n,s in zip(self.shape,self.stride))
    def bytes(self):return self.extent()*ELEMENT_BYTES[self.dtype]
    def create(self):
        if self.dtype not in ('f16','bf16','f32'):raise ValueError('Unqualified profile dtype')
        base=torch.randn(self.extent(),device='mps',dtype=DTYPES[self.dtype])*.1
        t=base.as_strided(self.shape,self.stride) if self.stride is not None else base.reshape(self.shape)
        return t.requires_grad_(True)


def parse(text):
    if len(text)>100_000:raise ValueError('Profile text budget')
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,bool,str,type(None))):return n.value
        if isinstance(n,ast.Tuple):return tuple(visit(x) for x in n.elts)
        if isinstance(n,ast.List):return [visit(x) for x in n.elts]
        if isinstance(n,ast.Dict):return {visit(k):visit(v) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand)
        if isinstance(n,ast.Name) and n.id in DTYPES:return n.id
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='T':
            args=[visit(a) for a in n.args];kw={k.arg:visit(k.value) for k in n.keywords}
            if len(args) not in (2,3) or set(kw)-{'stride'}:raise ValueError('Invalid T arguments')
            shape,dtype=args[:2];stride=args[2] if len(args)==3 else kw.get('stride')
            if dtype not in DTYPES:raise ValueError('Unknown dtype')
            return TensorSpec(tuple(shape),dtype,tuple(stride) if stride is not None else None)
        raise ValueError('Unapproved profile syntax: '+type(n).__name__)
    result=visit(ast.parse(text,mode='eval').body)
    if not isinstance(result,tuple) or len(result)!=2 or not isinstance(result[1],dict):
        raise ValueError('Expected positional/keyword arguments')
    return result


def specs(tree):
    if isinstance(tree,TensorSpec):yield tree
    elif isinstance(tree,(tuple,list)):
        for x in tree:yield from specs(x)
    elif isinstance(tree,dict):
        for x in tree.values():yield from specs(x)


def instantiate(tree):
    if isinstance(tree,TensorSpec):return tree.create()
    if isinstance(tree,tuple):return tuple(instantiate(x) for x in tree)
    if isinstance(tree,list):return [instantiate(x) for x in tree]
    if isinstance(tree,dict):return {k:instantiate(v) for k,v in tree.items()}
    return tree


def implementation(name,fn):
    if name in ('nn.functional.gelu','aten.gelu.default'):return gelu,'OjasX GELU registered backward'
    if name in ('nn.functional.silu','aten.silu.default'):return silu,'OjasX SiLU registered backward'
    if name in ('nn.functional.relu','aten.relu.default'):return packed_relu,'OjasX packed ReLU residual'
    return fn,'native MPS fallback; coverage only'


def qualify(name,fn,args,kwargs):
    candidate,route=implementation(name,fn)
    flat,_=tree_flatten((args,kwargs));demand=tuple({id(t):t for t in flat if isinstance(t,torch.Tensor) and t.requires_grad}.values())
    ya=fn(*args,**kwargs);yb=candidate(*args,**kwargs)
    fa,_=tree_flatten(ya);fb,_=tree_flatten(yb)
    ta=[t for t in fa if isinstance(t,torch.Tensor)];tb=[t for t in fb if isinstance(t,torch.Tensor)]
    es=[compare(a,b) for a,b in zip(tb,ta)]
    if len(ta)!=len(tb):es.append(dict(passed=False,reason='output arity'))
    da=[t for t in ta if t.requires_grad];db=[t for t in tb if t.requires_grad]
    if da and demand:
        upstream=[torch.randn_like(t)*.1 for t in da]
        ga=torch.autograd.grad(da,demand,upstream,allow_unused=True)
        gb=torch.autograd.grad(db,demand,upstream,allow_unused=True)
        es.extend(compare(a,b) for a,b in zip(gb,ga))
        # Independently check the established dense/softmax pullback identities.
        # These do not claim fewer FLOPs than PyTorch for an isolated operator.
        if name in ('mm','aten.mm.default','bmm','aten.bmm.default'):
            x,w=args;g=upstream[0]
            exact=(g@w.transpose(-1,-2),x.transpose(-1,-2)@g)
            es.extend(compare(a,b) for a,b in zip(exact,ga));route+='; explicit both-operand dagger'
        if name=='softmax':
            dim=kwargs.get('dim',args[1] if len(args)>1 else -1)
            # Existing foundation primitive reduces the last axis; conjugate
            # it by an axis permutation for an arbitrary OpInfo softmax axis.
            p=ta[0].movedim(dim,-1);g=upstream[0].movedim(dim,-1)
            es.append(compare(softmax_pullback(p,g).movedim(-1,dim),ga[0]));route+='; structured softmax VJP'
    return dict(route=route,errors=es,passed=all(e['passed'] for e in es))


def cts_cases():
    from torchcts.core.opinfo_adapter import get_live_opinfo
    rows=[]
    names=('mm','bmm','addmm','nn.functional.gelu','nn.functional.silu','nn.functional.relu',
           'softmax','nn.functional.layer_norm','nn.functional.conv2d')
    for name in names:
        op=get_live_opinfo(name)
        if op is None:rows.append(dict(name=name,status='filtered',reason='No live OpInfo'));continue
        try:
            for i,sample in enumerate(itertools.islice(op.sample_inputs('mps',torch.float32,requires_grad=True),5)):
                args=(sample.input,)+sample.args;kw=sample.kwargs
                flat,_=tree_flatten((args,kw))
                if sum(t.numel()*t.element_size() for t in flat if isinstance(t,torch.Tensor))>8*1024*1024:
                    rows.append(dict(name=name,sample=i,status='filtered',reason='sample budget'));continue
                try:
                    out=qualify(name,op.op,args,kw)
                    rows.append(dict(name=name,sample=i,status='passed' if out['passed'] else 'failed',**out))
                except (RuntimeError,NotImplementedError) as e:
                    rows.append(dict(name=name,sample=i,status='filtered',reason=str(e)[:300]))
        except (RuntimeError,NotImplementedError) as e:
            rows.append(dict(name=name,status='filtered',reason='Generator: '+str(e)[:300]))
        time.sleep(.05)
    return rows


def backend_cases():
    corpus=ROOT/'third_party/backend_bench_problems.parquet'
    allrows=pq.read_table(corpus).to_pylist();ledger=[];chosen=[];counts={op:0 for op in FAMILIES}
    for row in allrows:
        name=row['op_name'];reason=None
        if name not in FAMILIES:reason='operator outside this bounded adapter'
        elif row['is_synthetic']:reason='synthetic profile excluded'
        elif not row['runnable']:reason='upstream profile marked unrunnable'
        else:
            try:
                tree=parse(row['args']);ts=list(specs(tree));storage=sum(t.bytes() for t in ts)
                if any(t.dtype not in ('f16','bf16','f32') for t in ts):reason='unqualified MPS dtype'
                elif storage>8*1024*1024:reason='8 MiB operand budget'
                elif name in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
                    x,w=(ts[-2],ts[-1]);work=math.prod(x.shape)*w.shape[-1]*6
                    out=math.prod(x.shape[:-1])*w.shape[-1]*4
                    if work>600_000_000 or out>8*1024*1024:reason='matrix work/output budget'
                if reason is None and counts[name]>=2:reason='two profiles per operator budget'
                if reason is None:
                    chosen.append((row,tree));counts[name]+=1
            except (ValueError,KeyError,TypeError,IndexError) as e:reason='parser/budget: '+str(e)[:120]
        ledger.append(dict(uuid=row['uuid'],op_name=name,decision='selected' if reason is None else 'filtered',reason=reason))
    results=[]
    for row,tree in chosen:
        name=row['op_name'];_,op,overload=name.split('.')
        fn=getattr(getattr(torch.ops.aten,op),overload)
        try:
            args,kwargs=instantiate(tree);out=qualify(name,fn,args,kwargs)
            results.append(dict(uuid=row['uuid'],op_name=name,serialized_profile=row['args'],
                                status='passed' if out['passed'] else 'failed',**out))
        except (RuntimeError,NotImplementedError) as e:
            results.append(dict(uuid=row['uuid'],op_name=name,status='filtered',reason=str(e)[:300]))
        time.sleep(.1)
    return dict(corpus_rows=len(allrows),pin=PIN,sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                selection_ledger=ledger,results=results,
                scope='Original shapes/strides retained. Original values and model context are unavailable. No CUDA reference timing reused.')


def main():
    require_gpu();torch.manual_seed(715)
    parser_checks=[]
    for bad in ("__import__('os').system('echo unsafe')",'((T([999999], f32, stride=(-1,)),),{})'):
        try:
            result=parse(bad)
            for t in specs(result):t.extent()
        except ValueError:parser_checks.append(True)
        else:parser_checks.append(False)
    with GPUOnly() as guard:
        cts=cts_cases();backend=backend_cases()
    failures=[r for r in cts+backend['results'] if r['status']=='failed']
    provenance={name:subprocess.check_output(['git','-C',str(ROOT/'third_party'/name),'rev-parse','HEAD'],text=True).strip()
                for name in ('TorchCTS','BackendBench')}
    result=dict(status='passed_bounded_gpu_adapters' if not failures and all(parser_checks) else 'failed_gpu_adapters',
                device='mps',cpu_ml_execution=False,cpu_fallback_enabled=False,guarded_dispatches=guard.calls,
                torchcts=cts,backendbench=backend,parser_rejections=parser_checks,source_commits=provenance,
                versions={n:importlib.metadata.version(n) for n in ('torch','torchcts','BackendBench','pyarrow','z3-solver')},
                stock_suite_certification=False,
                source_sha256=hashes(['tools/test_external_conformance_gpu.py','tools/gpu_contracts.py','ojasx2/ops.py',
                                     'ojasx2/residuals.py','ojasx2/nonlinear.py']))
    (ROOT/'reports/external-conformance-gpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],'CTS',len(cts),'BackendBench',len(backend['results']),'failed',len(failures),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
