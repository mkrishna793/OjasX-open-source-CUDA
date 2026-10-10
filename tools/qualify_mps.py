"""Reproducible Apple GPU qualification, including negative speed results.

Run with PYTORCH_ENABLE_MPS_FALLBACK=0. Timings include Python dispatch, GPU
execution and synchronization. This is an ATen prototype, not a Metal compiler.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from ojasx2 import (linear_chain, chain_cost, rms_core_linear_chain, rms_core_cost,
    matrix_chain, matrix_chain_plan, streamed_gemm, reversible_coupling_chain,
    packed_relu, gelu, silu, swiglu, rmsnorm, make_backend, SavedTensorAudit)
from ojasx2.streamed import streamed_cost
from test_workload_coverage import model_case

# Contracts set before measurement, retained verbatim for failed cases.
RTOL=2e-4
ATOL=2e-4
NORM_TOL=2e-4


def error(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double()
    return {'max_abs':float((a-b).abs().max()),
        'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)),
        'entrywise_pass':bool(torch.allclose(a,b,rtol=RTOL,atol=ATOL))}


def outputs(value):
    return value if isinstance(value,tuple) else (value,)


def compare_values(a,b):
    errors=[error(x,y) for x,y in zip(a,b)]
    return {'output_and_gradient_errors':errors,
        'max_relative_l2':max(e['relative_l2'] for e in errors),
        'max_abs':max(e['max_abs'] for e in errors),
        'passed':len(a)==len(b) and all(e['entrywise_pass'] and e['relative_l2']<NORM_TOL for e in errors)}


def measure(name, inputs, funcs, cost=None, paths=None):
    # Independent real-number proxy: CPU FP64 native evaluation on the same
    # rounded FP32 inputs. Not a proof of the ideal real-valued solution.
    cpu=tuple(x.detach().cpu().double().requires_grad_() for x in inputs)
    cy=outputs(funcs['native'](*cpu))
    torch.manual_seed(951)
    g=tuple(torch.randn_like(y).float().double() for y in cy)
    cg=torch.autograd.grad(cy,cpu,g)
    golden=tuple(v.detach() for v in (*cy,*cg))
    gpu_g=tuple(v.float().to('mps') for v in g)
    def work(mode):
        ys=outputs(funcs[mode](*inputs))
        return (*ys,*torch.autograd.grad(ys,inputs,gpu_g))
    native=tuple(v.detach().cpu() for v in work('native'))
    results={}
    for mode in funcs:
        values=work(mode)
        assert all(v.device.type=='mps' for v in values)
        results[mode]={'vs_mps_native':compare_values(values,native),
                       'vs_cpu_fp64_native':compare_values(values,golden)}
        del values
    # Memory readings are live allocator snapshots at defined boundaries.
    # Saved-tensor hooks are used only here, never in timing or correctness.
    for mode in funcs:
        torch.mps.synchronize();torch.mps.empty_cache()
        before=torch.mps.current_allocated_memory()
        with SavedTensorAudit(list(inputs)) as audit:
            ys=outputs(funcs[mode](*inputs))
        torch.mps.synchronize()
        after_forward=torch.mps.current_allocated_memory()
        grads=torch.autograd.grad(ys,inputs,gpu_g)
        torch.mps.synchronize()
        after_backward=torch.mps.current_allocated_memory()
        results[mode]['saved']=audit.summary()
        results[mode]['allocator']={
            'baseline_live_bytes':before,
            'forward_retained_delta_bytes':after_forward-before,
            'after_backward_outputs_and_gradients_delta_bytes':after_backward-before,
            'driver_allocated_bytes':torch.mps.driver_allocated_memory(),
            'scope':'Synchronized live snapshots, not peak; driver allocation includes cached/workspace memory'}
        del ys,grads
    timings={mode:[] for mode in funcs}
    for mode in funcs:
        for _ in range(3):work(mode)
    torch.mps.synchronize()
    rng=random.Random(952)
    for _ in range(9):
        order=list(funcs);rng.shuffle(order)
        for mode in order:
            torch.mps.synchronize();start=time.perf_counter()
            for _ in range(3):
                value=work(mode);del value
            torch.mps.synchronize()
            timings[mode].append((time.perf_counter()-start)*1000/3)
    for mode in funcs:
        samples=timings[mode]
        results[mode].update({'median_joint_ms':statistics.median(samples),
            'min_joint_ms':min(samples),'max_joint_ms':max(samples),'samples_ms':samples})
    for mode in funcs:
        results[mode]['speedup_vs_native']=results['native']['median_joint_ms']/results[mode]['median_joint_ms']
        n=results['native']['saved']['fresh_saved_storage_bytes']
        results[mode]['fresh_tape_reduction_percent']=100*(n-results[mode]['saved']['fresh_saved_storage_bytes'])/n if n else None
    return {'name':name,'input_shapes':[list(x.shape) for x in inputs],
            'cost_model':cost,'observed_paths':paths,'modes':results}


def tensors(shapes,scale=.1):
    torch.manual_seed(950)
    # Create CPU samples before transfer to make the generator reproducible.
    return tuple((torch.randn(*shape)*scale).to('mps').requires_grad_() for shape in shapes)


def train_models():
    result={}
    for name in ('residual_tokens','standard_transformer_swiglu'):
        torch.manual_seed(960);torch._dynamo.reset()
        a,x,sites=model_case(name);a=a.to('mps');x=x.to('mps');b=copy.deepcopy(a)
        backend=make_backend(accuracy='algebraic',contraction_policy='rms_core')
        compiled=torch.compile(b,backend=backend,fullgraph=True)
        oa=torch.optim.AdamW(a.parameters(),lr=.001)
        ob=torch.optim.AdamW(b.parameters(),lr=.001)
        target=torch.randn_like(a(x))
        max_output=max_grad=max_param=max_moment=0.;passed=True
        for _ in range(30):
            oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
            # Two accumulation microbatches before each parameter update.
            for factor in (1.,.8):
                y,z=a(x*factor),compiled(x*factor)
                e=error(z,y);max_output=max(max_output,e['max_abs']);passed &= e['entrywise_pass']
                ((y-target).square().mean()/2).backward()
                ((z-target).square().mean()/2).backward()
            for p,q in zip(a.parameters(),b.parameters()):
                e=error(q.grad,p.grad);max_grad=max(max_grad,e['max_abs']);passed &= e['entrywise_pass']
            oa.step();ob.step()
            for p,q in zip(a.parameters(),b.parameters()):
                e=error(q,p);max_param=max(max_param,e['max_abs']);passed &= e['entrywise_pass']
                for key in ('exp_avg','exp_avg_sq'):
                    e=error(ob.state[q][key],oa.state[p][key]);max_moment=max(max_moment,e['max_abs']);passed &= e['entrywise_pass']
        result[name]={'steps':30,'optimizer':'AdamW lr=.001, eps=1e-8',
            'accumulation_microbatches':2,'graph_rewrite_sites':backend.evidence[-1]['rms_core_rewrites'],
            'expected_graph_rewrite_sites':sites,'maximum_output_abs_delta':max_output,
            'maximum_gradient_abs_delta':max_grad,'maximum_parameter_abs_delta':max_param,
            'maximum_optimizer_moment_abs_delta':max_moment,'passed':bool(passed),
            'scope':'One synthetic trajectory; graph rewrite counts are not runtime path counts'}
        torch._dynamo.reset()
    return result


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='1':
        raise SystemExit('CPU fallback must be disabled')
    if not torch.backends.mps.is_available():raise SystemExit('Real MPS device unavailable')
    torch.set_num_threads(2)
    cases=[]
    for m,k,n,p in ((1024,32,4096,64),(128,256,512,256)):
        xs=tensors(((m,k),(n,k),(p,n)))
        cases.append(measure(f'linear_{m}_{k}_{n}_{p}',xs,{
            'native':lambda *a:linear_chain(*a),
            'joint':lambda *a:linear_chain(*a,accuracy='algebraic'),
            'backward_only':lambda *a:linear_chain(*a,accuracy='algebraic',mode='backward_only')},
            {'joint':chain_cost(m,k,n,p),'backward_only':chain_cost(m,k,n,p,mode='backward_only')},
            {mode:('reassociated' if chain_cost(m,k,n,p,mode=mode)['scalar_saving']>0
                   else 'native_fallback_unprofitable') for mode in ('joint','backward_only')}))
    xs=tensors(((1024,32),(4096,32),(64,4096),(4096,),(64,)))
    paths=[];rms_core_linear_chain(*xs,accuracy='algebraic',evidence=paths)
    cases.append(measure('rms_core_wide',xs,{
        'native':lambda *a:rms_core_linear_chain(*a),
        'algebraic':lambda *a:rms_core_linear_chain(*a,accuracy='algebraic')},
        rms_core_cost(1024,32,4096,64),paths))
    xs=tensors(((256,32),(32,2048),(2048,64),(64,16)))
    cases.append(measure('matrix_chain_four_leaves',xs,{
        'native':lambda *a:matrix_chain(a),
        'algebraic':lambda *a:matrix_chain(a,'algebraic')},matrix_chain_plan((256,32,2048,64,16))))
    xs=tensors(((256,256),(256,256)))
    cases.append(measure('rank7',xs,{'native':lambda a,b:a@b,
        'algebraic':lambda a,b:streamed_gemm(a,b,'algebraic')},streamed_cost(256,256,256)))
    for name,fn,ref,shapes in (
        ('packed_relu',packed_relu,torch.relu,((1024,1024),)),
        ('gelu',gelu,F.gelu,((1024,1024),)),
        ('silu',silu,F.silu,((1024,1024),)),
        ('swiglu',swiglu,lambda a,b:F.silu(a)*b,((1024,1024),(1024,1024))),
        ('rmsnorm',rmsnorm,lambda a,g:F.rms_norm(a,(1024,),g,1e-6),((1024,1024),(1024,)) )):
        cases.append(measure(name,tensors(shapes),{'native':ref,'ojasx':fn}))
    for depth in (16,64):
        xs=tensors(((128,32),(128,32),*((32,32),(32,),(32,32),(32,))*depth))
        def rev(a,mode):
            blocks=tuple(tuple(a[i:i+4]) for i in range(2,len(a),4))
            return reversible_coupling_chain(*a[:2],blocks,accuracy=mode)
        cases.append(measure(f'reversible_depth_{depth}',xs,{
            'native':lambda *a:rev(a,'strict'),
            'algebraic':lambda *a:rev(a,'algebraic')}))
    # End-to-end compiled workload includes input and every parameter gradient.
    torch.manual_seed(965);torch._dynamo.reset()
    model,x,sites=model_case('residual_tokens');model=model.to('mps');x=x.to('mps').requires_grad_()
    params=tuple(model.parameters())
    backend=make_backend(accuracy='algebraic',contraction_policy='rms_core')
    compiled=torch.compile(model,backend=backend,fullgraph=True)
    # functional_call enables an independent CPU FP64 golden for the same model.
    names=list(dict(model.named_parameters()))
    cpu_model=copy.deepcopy(model).cpu().double()
    def native(a,*p):
        if a.device.type=='cpu':
            return torch.func.functional_call(cpu_model,dict(zip(names,p)),(a,))
        return model(a)
    def opt(a,*p):return compiled(a)
    case=measure('compiled_residual_tokens',(x,*params),{'native':native,'compiled':opt})
    case['compiler_evidence']=backend.evidence
    cases.append(case);torch._dynamo.reset()
    training=train_models()
    report={'status':'passed' if all(v['vs_mps_native']['passed'] and v['vs_cpu_fp64_native']['passed']
        for c in cases for v in c['modes'].values()) and all(v['passed'] for v in training.values()) else 'failed',
        'hardware':'Apple M5; detected independently in gpu-access-restored.json',
        'platform':platform.platform(),'torch':torch.__version__,'device':'mps','dtype':'float32',
        'cpu_fallback_enabled':False,'contract':{'rtol':RTOL,'atol':ATOL,'relative_l2_max':NORM_TOL},
        'timing_method':'3 warmups, 9 shuffled rounds, 3 joint forward/VJP executions per sample; synchronized wall time',
        'memory_scope':'Fresh forward autograd tape and live allocator snapshots; no peak or full optimizer-memory claim',
        'cases':cases,'training':training,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('ojasx2','tools','tests') for p in sorted((ROOT/folder).glob('*.py'))},
        'limitations':['Sample qualification, not all workloads or shapes',
            'Reassociation changes FP32 rounding; strict default retained',
            'No native Metal kernel generation or hardware portability certification',
            'MPS measures Apple GPU; Neural Engine not tested here',
            'MLX earlier sample failed its numerical contract; see mlx-gpu-permission-retry.json']}
    (ROOT/'reports/mps-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':[{ 'name':c['name'],
        'modes':{m:{k:v[k] for k in ('median_joint_ms','speedup_vs_native','fresh_tape_reduction_percent')}
                 for m,v in c['modes'].items()}} for c in cases],'training':training},indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
