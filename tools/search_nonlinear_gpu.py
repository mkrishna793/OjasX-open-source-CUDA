"""MPS-only nonlinear discovery qualification; no CPU ML reference/fallback.

Host Python orchestrates GPU operations. Error reductions stay on MPS.
Proof tools run separately on the host. Full-step timings include guards and
plain SGD updates; allocator samples do not prove total Metal peak memory.
"""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
    raise RuntimeError('Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch')
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ojasx2.nonlinear_taylor_gpu import native_block,taylor_block,taylor_cost,surrogate_block
from ojasx2.contractions import linear_chain,chain_cost
from ojasx2.residuals import SavedTensorAudit
RTOL=2e-4;ATOL=2e-4;NORM_TOL=2e-4
REPORT=ROOT/'reports/nonlinear-gpu-search.json'

def source_hashes():
    paths=[ROOT/'ojasx2/nonlinear_taylor_gpu.py',ROOT/'ojasx2/taylor_coefficients.py',ROOT/'ojasx2/contractions.py',ROOT/'ojasx2/residuals.py',ROOT/'tools/search_nonlinear_gpu.py',ROOT/'tools/repeat_nonlinear_gpu.py',ROOT/'tools/gpu_nonlinear_edges.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def compare(actual,reference):
    assert actual.device.type==reference.device.type=='mps'
    diff=actual.detach()-reference.detach()
    norm=torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference.detach()).clamp_min(1e-20)
    return {'max_abs':diff.abs().amax().item(),'relative_l2':norm.item(),
        'passed':bool(torch.allclose(actual,reference,rtol=RTOL,atol=ATOL)) and norm.item()<=NORM_TOL}

def clone(values):return tuple(v.detach().clone().requires_grad_(True) for v in values)

def make_inputs(shape,kind,domain):
    m,k,n,p=shape
    x=torch.randn(m,k,device='mps');a=torch.randn(k,n,device='mps')/math.sqrt(k)
    if domain=='small':a=a*(.03/(x@a).abs().amax().clamp_min(1e-20))
    elif domain=='zero':x.zero_()
    elif domain=='noncontiguous':x=torch.randn(k,m,device='mps').T;a=a*(.03/(x@a).abs().amax().clamp_min(1e-20))
    b=torch.randn(n,p,device='mps')/math.sqrt(n)
    values=(x,a,b)
    if kind=='swiglu':values+= (torch.randn(k,n,device='mps')/math.sqrt(k),)
    return clone(values)

def call(values,kind,candidate=False,evidence=None):
    x,a,b=values[:3];v=values[3] if len(values)==4 else None
    return taylor_block(x,a,b,v,kind,evidence=evidence) if candidate else native_block(x,a,b,v,kind)

def forward_vjp(values,kind,candidate,upstream):
    y=call(values,kind,candidate)
    gradients=torch.autograd.grad(y,values,upstream)
    assert all(t.device.type=='mps' for t in (y,*gradients))
    return y,*gradients

def step(values,kind,candidate,lr=1e-4):
    y=call(values,kind,candidate);loss=y.square().mean()
    gradients=torch.autograd.grad(loss,values)
    with torch.no_grad():
        for parameter,gradient in zip(values[1:],gradients[1:]):parameter.add_(gradient,alpha=-lr)
    return loss,gradients

class AllocationSamples(TorchDispatchMode):
    def __init__(self):self.high=torch.mps.current_allocated_memory();self.calls=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}));self.calls+=1
        self.high=max(self.high,torch.mps.current_allocated_memory())
        return result

def memory(values,kind,candidate):
    vals=clone(values);torch.mps.synchronize();torch.mps.empty_cache();gc.collect()
    before=torch.mps.current_allocated_memory();samples=AllocationSamples()
    with samples:
        with SavedTensorAudit(list(vals)) as audit:y=call(vals,kind,candidate)
        torch.mps.synchronize();retained=torch.mps.current_allocated_memory()-before
        gradients=torch.autograd.grad(y.square().mean(),vals)
        with torch.no_grad():
            for parameter,gradient in zip(vals[1:],gradients[1:]):parameter.add_(gradient,alpha=-1e-4)
        torch.mps.synchronize()
    return {'saved':audit.summary(),'forward_live_delta_bytes':retained,
        'sampled_high_live_delta_bytes':samples.high-before,'sampled_dispatches':samples.calls,
        'driver_allocated_bytes':torch.mps.driver_allocated_memory(),
        'scope':'Allocator samples at ATen operation returns; includes full SGD step. Lower observation of peak, excludes hidden Metal workspace and is not a certified total GPU peak.'}

def timed(values,kind):
    states={'native':clone(values),'candidate':clone(values)};times={mode:[] for mode in states}
    for mode,state in states.items():
        for _ in range(3):step(state,kind,mode=='candidate')
    torch.mps.synchronize();rng=random.Random(915)
    for _ in range(7):
        order=list(states);rng.shuffle(order)
        for mode in order:
            torch.mps.synchronize();start=time.perf_counter()
            for _ in range(2):step(states[mode],kind,mode=='candidate')
            torch.mps.synchronize();times[mode].append((time.perf_counter()-start)/2)
    med={mode:statistics.median(t) for mode,t in times.items()}
    return {'seconds_median':med,'seconds_samples':times,'native_over_candidate':med['native']/med['candidate'],
        'method':'3 warmups, 7 shuffled rounds, 2 full forward/loss/all-gradient/weight-SGD steps per sample; synchronized wall time. Includes domain guard and coefficient construction.'}

def qualify(shape,kind,domain,benchmark=True):
    values=make_inputs(shape,kind,domain);evidence=[];call(values,kind,True,evidence)
    upstream=torch.randn(shape[0],shape[3],device='mps')
    ref=forward_vjp(values,kind,False,upstream);actual=forward_vjp(values,kind,True,upstream)
    errors=[compare(a,b) for a,b in zip(actual,ref)]
    a,b=clone(values),clone(values);training=[]
    for index in range(4):
        la,ga=step(a,kind,False);lb,gb=step(b,kind,True)
        ge=[compare(u,v) for u,v in zip(gb,ga)];pe=[compare(u,v) for u,v in zip(b[1:],a[1:])]
        le=compare(lb.reshape(1),la.reshape(1));training.append({'step':index,'loss':le,'gradients':ge,'parameters':pe,
            'passed':le['passed'] and all(e['passed'] for e in ge+pe)})
    passed=all(e['passed'] for e in errors) and all(s['passed'] for s in training)
    result={'shape':list(shape),'kind':kind,'domain':domain,'route':evidence[-1],
        'forward_and_all_vjp':errors,'four_step_sgd':training,'correctness_passed':passed,
        'cost':taylor_cost(*shape,kind),'accepted_for_dispatch':False}
    if benchmark:
        result['memory']={mode:memory(values,kind,mode=='candidate') for mode in ('native','candidate')}
        result['timing']=timed(values,kind)
        result['retained_storage_improved']=result['memory']['candidate']['saved']['fresh_saved_storage_bytes']<result['memory']['native']['saved']['fresh_saved_storage_bytes']
        result['sampled_live_improved']=result['memory']['candidate']['sampled_high_live_delta_bytes']<result['memory']['native']['sampled_high_live_delta_bytes']
        result['all_observed_gates_pass']=passed and result['retained_storage_improved'] and result['sampled_live_improved'] and result['timing']['native_over_candidate']>1
    return result

def linear_gpu(normalized=False):
    m,k,n,p=512,16,2048,16
    values=tuple(torch.randn(*shape,device='mps').requires_grad_() for shape in ((m,k),(n,k),(p,n)))
    if normalized:values=(values[0],(values[1]/math.sqrt(k)).detach().requires_grad_(),(values[2]/math.sqrt(n)).detach().requires_grad_())
    native=lambda x,a,b:F.linear(F.linear(x,a),b)
    candidate=lambda x,a,b:linear_chain(x,a,b,accuracy='algebraic')
    g=torch.randn(m,p,device='mps');results={}
    reference_y=native(*values);reference=(reference_y,*torch.autograd.grad(reference_y,values,g))
    for name,fn in [('native',native),('candidate',candidate)]:
        with SavedTensorAudit(list(values)) as audit:y=fn(*values)
        gradients=torch.autograd.grad(y,values,g)
        results[name]={'errors':[compare(a,b) for a,b in zip((y,*gradients),reference)],'saved':audit.summary()}
    times={name:[] for name in results}
    for name,fn in [('native',native),('candidate',candidate)]:
        for index in range(10):
            torch.mps.synchronize();start=time.perf_counter()
            y=fn(*values);torch.autograd.grad(y,values,g);torch.mps.synchronize()
            if index>=3:times[name].append(time.perf_counter()-start)
    return {'shape':[m,k,n,p],'cost':chain_cost(m,k,n,p),'modes':results,
        'native_over_candidate':statistics.median(times['native'])/statistics.median(times['candidate']),
        'normalized_weights':normalized,'correctness_passed':all(e['passed'] for e in results['candidate']['errors']),
        'scope':'Exact-real linear reassociation, GPU FP32 tolerances; forward and all VJPs, no optimizer timing in this control.'}

def higher_surrogate():
    results=[]
    for kind in ('gelu','silu','swiglu'):
        values=make_inputs((4,2,16,2),kind,'small')
        # Force the custom algebraic path directly: the tiny shape's cost guard
        # correctly falls back. Compare against native AD of the same polynomial.
        from ojasx2.nonlinear_taylor_gpu import _TaylorFeature
        def fn(v,custom):
            x,a,b=v[:3];up=v[3] if len(v)==4 else None
            return (_TaylorFeature.apply(x,a,b,up,kind) if custom else surrogate_block(x,a,b,up,kind)).square().mean()
        directions=tuple(torch.randn_like(t) for t in values);actual=[]
        for custom in (False,True):
            grads=torch.autograd.grad(fn(values,custom),values,create_graph=True)
            dot=sum((g*d).sum() for g,d in zip(grads,directions))
            actual.append(torch.autograd.grad(dot,values))
        errors=[compare(a,b) for a,b in zip(actual[1],actual[0])]
        results.append({'kind':kind,'errors':errors,'passed':all(e['passed'] for e in errors),
            'scope':'Higher derivatives of the surrogate itself, not equality to native activation derivatives.'})
    return results

def main():
    report={'status':'running','device':'mps','torch':torch.__version__,'dtype':'float32',
        'cpu_ml_execution':False,'cpu_fallback_enabled':False,'contract':{'rtol':RTOL,'atol':ATOL,'relative_l2':NORM_TOL},'cases':[]}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    if not torch.backends.mps.is_available():raise RuntimeError('GPU unavailable; no CPU substitution')
    torch.manual_seed(913);report['linear_control']=linear_gpu();report['linear_normalized_control']=linear_gpu(True)
    for kind in ('gelu','silu','swiglu'):
        for shape,domain in [((128,2,512,8),'small'),((512,2,4096,8),'small'),((256,4,2048,16),'small'),
                ((128,2,512,8),'ordinary'),((128,16,512,8),'small'),((128,2,512,8),'zero'),((64,2,256,8),'noncontiguous')]:
            print('GPU',kind,shape,domain,flush=True)
            case=qualify(shape,kind,domain,benchmark=domain in ('small','ordinary'))
            report['cases'].append(case);REPORT.write_text(json.dumps(report,indent=2)+'\n')
    report['higher_surrogate']=higher_surrogate()
    report['status']='completed';report['accuracy_failures']=sum(not c['correctness_passed'] for c in report['cases'])
    report['observed_joint_wins']=sum(c.get('all_observed_gates_pass',False) for c in report['cases'])
    report['linear_control_accuracy_failures']=sum(not report[key]['correctness_passed'] for key in ('linear_control','linear_normalized_control'))
    report['source_sha256']=source_hashes()
    report['scope']='Bounded synthetic small-domain approximate nonlinear blocks and fallback controls. No most-model claim, total peak guarantee or automatic dispatch.'
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),'accuracy_failures':report['accuracy_failures'],'observed_joint_wins':report['observed_joint_wins']},indent=2))

if __name__=='__main__':main()
