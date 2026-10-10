"""Adversarial edge qualification and exact refined Rocq extraction oracles."""
from fractions import Fraction as Q
import json
from pathlib import Path
import random
import subprocess
import sys
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2.math_foundation import (StructuredDerivative,NormalizationResidual,normalization_residual,
    rmsnorm_input_derivative,layernorm_input_derivative,activation_derivative,softmax_derivative,
    rank7_reference,winograd_reference)

def extracted():
    rng=random.Random(1024);counts={'compact':0,'indexed':0,'invalid':0}
    p=subprocess.Popen([str(ROOT/'rocq/extracted/ojasx_refined_cli')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def encode(xs):return ','.join(map(str,xs))
    def ask(*parts):
        p.stdin.write('|'.join(parts)+'\n');p.stdin.flush()
        return json.loads(p.stdout.readline())
    def q():return Q(rng.randint(-7,7),rng.randint(1,9))
    try:
        for n in (0,1,2,7,32):
            for _ in range(8):
                z,g,gamma=[q() for _ in range(n)],[q() for _ in range(n)],[q() for _ in range(n)]
                r=q();inv=Q(1,n) if n else Q(0)
                weighted=[a*b for a,b in zip(gamma,g)]
                correction=inv*sum((a*b for a,b in zip(z,weighted)),Q(0))
                for kind in ('rms','layer'):
                    row=ask('compact',kind,str(n),str(r),str(inv),encode(z),encode(gamma),encode(g))
                    mean=inv*sum(weighted,Q(0)) if kind=='layer' else Q(0)
                    expected=[r*(w-mean-a*correction) for a,w in zip(z,weighted)]
                    assert list(map(Q,row['dx']))==expected;counts['compact']+=1
        for n in (0,1,3,16):
            for k in ((0,) if n==0 else (0,1,8,32)):
                idx=[rng.randrange(n) for _ in range(k)];x=[q() for _ in range(n)];g=[q() for _ in idx]
                row=ask('gather',str(n),encode(idx),encode(x),encode(g))
                dx=[Q(0)]*n
                for i,h in zip(idx,g):dx[i]+=h
                assert list(map(Q,row['y']))==[x[i] for i in idx]
                assert list(map(Q,row['dx']))==dx;counts['indexed']+=1
        for args in [('gather','1','1','2','3'),('gather','1','-1','2','3'),
                     ('gather','0','0','','1'),('compact','rms','1','1','1','1/0','1','1')]:
            assert 'error' in ask(*args);counts['invalid']+=1
    finally:p.stdin.close();p.wait(timeout=10)
    assert p.returncode==0
    return counts

def norm_edges():
    torch.manual_seed(1025);results=[]
    shapes=[(7,),(2,7),(2,3,7),(0,7),(3,1)]
    for dtype in (torch.float64,torch.float32):
        for shape in shapes:
            for value in ('random','zero','constant'):
                for kind in ('rmsnorm','layernorm'):
                    for affine in (False,True):
                        x=torch.randn(shape,dtype=dtype) if value=='random' else torch.full(shape,0. if value=='zero' else 3.,dtype=dtype)
                        x.requires_grad_();n=shape[-1]
                        gamma=torch.randn(n,dtype=dtype,requires_grad=True) if affine else None
                        beta=torch.randn(n,dtype=dtype,requires_grad=True) if affine else None
                        eps=1e-5
                        y=F.layer_norm(x,(n,),gamma,beta,eps) if kind=='layernorm' else F.rms_norm(x,(n,),gamma,eps)
                        if kind=='rmsnorm' and beta is not None:y=y+beta
                        g=torch.randn_like(y)
                        args=(x,gamma,beta) if affine else (x,)
                        ref=torch.autograd.grad(y,args,g)
                        cap=normalization_residual(x,gamma,beta,eps,kind)
                        dx,dg,db=cap.pull(g);got=(dx,dg,db) if affine else (dx,)
                        tol=2e-10 if dtype==torch.float64 else 3e-4
                        for a,b in zip(got,ref):torch.testing.assert_close(a,b,atol=tol,rtol=tol)
                        if affine:
                            builder=layernorm_input_derivative if kind=='layernorm' else rmsnorm_input_derivative
                            torch.testing.assert_close(builder(x,gamma,eps).pull(g),dx,atol=tol,rtol=tol)
                        results.append({'dtype':str(dtype),'shape':list(shape),'value':value,'kind':kind,'affine':affine,
                            'max_abs_error':max((a-b).abs().max().item() if a.numel() else 0. for a,b in zip(got,ref)),
                            'residual_elements':cap.residual_elements})
        # Tiny epsilon regression: old r^3 factor overflowed at zero input.
        for kind in ('rmsnorm','layernorm'):
            x=torch.zeros(2,4,dtype=dtype,requires_grad=True);gamma=torch.ones(4,dtype=dtype)
            eps=1e-30 if dtype==torch.float32 else 1e-250
            g=torch.ones_like(x)
            cap=normalization_residual(x,gamma,eps=eps,kind=kind);got=cap.pull(g)[0]
            expected=torch.full_like(x,eps**-.5) if kind=='rmsnorm' else torch.zeros_like(x)
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got,expected,rtol=1e-6,atol=0)
            builder=layernorm_input_derivative if kind=='layernorm' else rmsnorm_input_derivative
            factored=builder(x,gamma,eps).pull(g)
            assert torch.isfinite(factored).all()
            torch.testing.assert_close(factored,expected,rtol=1e-6,atol=0)
            results.append({'dtype':str(dtype),'kind':kind,'tiny_epsilon':eps,'finite':True})
    return results

def layout_and_validation():
    torch.manual_seed(1026);valid=0;rejections=0
    x=torch.randn(7,3,dtype=torch.float64).T;g=torch.randn_like(x);gamma=torch.randn(7,dtype=torch.float64)
    for kind in ('rmsnorm','layernorm'):
        cap=normalization_residual(x,gamma,kind=kind)
        fun=(lambda x:F.layer_norm(x,(7,),gamma,eps=1e-5)) if kind=='layernorm' else (lambda x:F.rms_norm(x,(7,),gamma,1e-5))
        _,pb=torch.func.vjp(fun,x)
        torch.testing.assert_close(cap.pull(g)[0],pb(g)[0],atol=1e-10,rtol=1e-10)
        # Higher-order derivatives of the compact representation.
        direction=torch.randn_like(x)
        _,got=torch.func.jvp(lambda z:normalization_residual(z,gamma,kind=kind).pull(g)[0],(x,),(direction,))
        _,ref=torch.func.jvp(torch.func.grad(lambda z:(fun(z)*g).sum()),(x,),(direction,))
        torch.testing.assert_close(got,ref,atol=1e-9,rtol=1e-9);valid+=1
    functions=[]
    for eps in [0.,-1.,float('nan'),float('inf'),True,1e-100]:
        functions.append(lambda eps=eps:normalization_residual(torch.zeros(2,4),torch.ones(4),eps=eps))
    functions += [lambda:StructuredDerivative(torch.tensor(1.),torch.ones(2),torch.ones(2)),
        lambda:normalization_residual(torch.ones(2,0)),
        lambda:normalization_residual(torch.ones(2,4),torch.ones(5)),
        lambda:normalization_residual(torch.ones(2,4),torch.ones(4,dtype=torch.float64)),
        lambda:normalization_residual(torch.ones(2,4),kind='unknown'),
        lambda:activation_derivative(torch.ones(2,dtype=torch.float16),'silu'),
        lambda:normalization_residual(x,gamma).pull(torch.ones(1,7,dtype=torch.float64)),
        lambda:NormalizationResidual('unknown',x,torch.ones(3,1,dtype=x.dtype),gamma,False),
        lambda:NormalizationResidual('layernorm',x,torch.ones(3,2,dtype=x.dtype),gamma,False),
        lambda:NormalizationResidual('layernorm',x,torch.ones(3,1,dtype=x.dtype),gamma,1)]
    for candidate in (rank7_reference,winograd_reference):
        for shape in ((3,4,2),(2,0,2),(2,4,3)):
            m,k,n=shape;functions.append(lambda candidate=candidate,m=m,k=k,n=n:candidate(torch.ones(m,k),torch.ones(k,n)))
        functions.append(lambda candidate=candidate:candidate(torch.ones(2,2,dtype=torch.int64),torch.ones(2,2,dtype=torch.int64)))
    for fun in functions:
        try:fun()
        except (ValueError,RuntimeError):rejections+=1
        else:raise AssertionError('Invalid domain accepted')
    return {'noncontiguous_and_higher_order':valid,'invalid_domain_rejections':rejections}

def activation_edges():
    results=[]
    functions={'relu':F.relu,'silu':F.silu,'gelu':F.gelu,'tanh':torch.tanh,
        'sigmoid':torch.sigmoid,'square':torch.square}
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([-1e15,-80.,-1e-30,-0.,0.,1e-30,80.,1e15],dtype=dtype)
        g=torch.linspace(-1,1,x.numel(),dtype=dtype)
        tol=2e-6 if dtype==torch.float32 else 1e-10
        for kind,fun in functions.items():
            _,pb=torch.func.vjp(fun,x);ref=pb(g)[0];got=activation_derivative(x,kind).pull(g)
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got,ref,atol=tol,rtol=tol)
            results.append({'dtype':str(dtype),'kind':kind,'finite':True})
        for logits in ([0.],[0.,0.,0.],[1000.,-1000.,0.],[-1000.,-999.,-998.]):
            t=torch.tensor(logits,dtype=dtype);up=torch.arange(len(logits),dtype=dtype)
            p=t.softmax(-1);_,pb=torch.func.vjp(lambda v:v.softmax(-1),t)
            got=softmax_derivative(p).pull(up)
            torch.testing.assert_close(got,pb(up)[0],atol=tol,rtol=tol)
            results.append({'dtype':str(dtype),'kind':'softmax','width':len(logits)})
    return results

def optimizer_steps():
    torch.manual_seed(1027);results=[]
    for kind in ('rmsnorm','layernorm'):
        for cls in (torch.optim.SGD,torch.optim.AdamW):
            p=[torch.randn(3,7,dtype=torch.float64),torch.randn(7,dtype=torch.float64),torch.randn(7,dtype=torch.float64)]
            a=[x.clone().requires_grad_() for x in p];b=[x.clone().requires_grad_() for x in p]
            opts=[cls(a,lr=1e-3),cls(b,lr=1e-3)]
            for step in range(4):
                for params,opt,manual in [(a,opts[0],False),(b,opts[1],True)]:
                    opt.zero_grad();x,gamma,beta=params
                    y=F.layer_norm(x,(7,),gamma,beta,1e-5) if kind=='layernorm' else F.rms_norm(x,(7,),gamma,1e-5)+beta
                    if manual:
                        with torch.no_grad():
                            grads=normalization_residual(x,gamma,beta,kind=kind).pull(2*y/y.numel())
                        for param,grad in zip(params,grads):param.grad=grad
                    else:y.square().mean().backward()
                    opt.step()
                for actual,expected in zip(b,a):torch.testing.assert_close(actual,expected,atol=2e-9,rtol=2e-9)
            results.append({'kind':kind,'optimizer':cls.__name__,'steps':4,
                'max_parameter_error':max((actual-expected).abs().max().item() for actual,expected in zip(b,a))})
    return results

def main():
    torch.set_num_threads(2)
    report={'status':'running','scope':'Finite edge suite, FP32/FP64 CPU. Residual elements exclude autograd graph retention and total peak memory.'}
    path=ROOT/'reports/foundation-edges.json';path.write_text(json.dumps(report,indent=2)+'\n')
    report['exact_extraction']=extracted()
    report['normalization']=norm_edges()
    report['validation']=layout_and_validation()
    report['activation_edges']=activation_edges()
    report['optimizer_steps']=optimizer_steps()
    report['status']='passed';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','exact':report['exact_extraction'],'normalization_cases':len(report['normalization']),
        'validation':report['validation'],'optimizer_steps':report['optimizer_steps']},indent=2))

if __name__=='__main__':main()
