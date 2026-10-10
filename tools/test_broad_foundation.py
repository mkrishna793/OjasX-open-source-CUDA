"""Exact extraction tests and bounded ML-family qualification; CPU only.

This is not a percentage estimate of all ML and does not formally verify
PyTorch, its analytic primitives, or the handwritten adapters.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import random
import statistics
import subprocess
import sys
import time
import torch
import torch.nn.functional as fn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2.math_foundation import (StructuredDerivative, activation_derivative,
    softmax_derivative, rmsnorm_input_derivative, layernorm_input_derivative, rank7_reference, winograd_reference)
from ojasx2.streamed import streamed_gemm

def exact_tests():
    rng=random.Random(891)
    process=subprocess.Popen([str(ROOT/'rocq/extracted/ojasx_foundation_cli')],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    counts={'rank7_joint':0,'rank7_blocks':0,'winograd_joint':0,'winograd_blocks':0,
            'structured':0,'softmax':0,'planner':0,'invalid':0}
    def q():return F(rng.randint(-7,7),rng.randint(1,9))
    def csv(xs):return ','.join(map(str,xs))
    def mat(m,n):return [[q() for _ in range(n)] for _ in range(m)]
    def flat(a):return [v for row in a for v in row]
    def mm(a,b,m,k,n):return [[sum((a[i][h]*b[h][j] for h in range(k)),F(0)) for j in range(n)] for i in range(m)]
    def tr(a):return [list(row) for row in zip(*a)]
    def ask(*parts):
        process.stdin.write('|'.join(parts)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise AssertionError('Extracted foundation terminated')
        return json.loads(line)
    try:
        for _ in range(80):
            a,b,g=mat(2,2),mat(2,2),mat(2,2)
            r=ask('rank7',csv(flat(a)),csv(flat(b)),csv(flat(g)))
            for key,expected in [('y',mm(a,b,2,2,2)),('da',mm(g,tr(b),2,2,2)),('db',mm(tr(a),g,2,2,2))]:
                assert list(map(F,r[key]))==flat(expected)
            assert r['multiplications']==21 and r['additions']==54
            counts['rank7_joint']+=1
            w=ask('winograd',csv(flat(a)),csv(flat(b)),csv(flat(g)))
            assert all(w[key]==r[key] for key in ('y','da','db'))
            assert w['multiplications']==21 and w['additions']==45
            counts['winograd_joint']+=1
        for m,k,n in [(1,1,1),(2,3,2),(3,2,1),(1,3,3),(2,0,2)]:
            a,b=mat(2*m,2*k),mat(2*k,2*n)
            r=ask('block7',f'{m},{k},{n}',csv(flat(a)),csv(flat(b)))
            assert [[F(x) for x in row] for row in r['y']]==mm(a,b,2*m,2*k,2*n)
            counts['rank7_blocks']+=1
            w=ask('blockw',f'{m},{k},{n}',csv(flat(a)),csv(flat(b)))
            assert w==r;counts['winograd_blocks']+=1
        for n,rk in [(n,rk) for n in (0,1,3,8) for rk in (0,1,2,3)]:
            for _ in range(4):
                d,x,g=[q() for _ in range(n)],[q() for _ in range(n)],[q() for _ in range(n)]
                u,v=mat(n,rk),mat(n,rk)
                r=ask('structured',str(n),str(rk),csv(d),csv(flat(u)),csv(flat(v)),csv(x),csv(g))
                jac=[[sum((u[i][h]*v[j][h] for h in range(rk)),F(0))+(d[i] if i==j else 0)
                      for j in range(n)] for i in range(n)]
                expected_push=[sum((jac[i][j]*x[j] for j in range(n)),F(0)) for i in range(n)]
                expected_pull=[sum((jac[j][i]*g[j] for j in range(n)),F(0)) for i in range(n)]
                assert list(map(F,r['push']))==expected_push and list(map(F,r['pull']))==expected_pull
                assert sum(a*b for a,b in zip(expected_push,g))==sum(a*b for a,b in zip(x,expected_pull))
                counts['structured']+=1
        for n in (1,2,4,8):
            p=[F(i+1,n*(n+1)//2) for i in range(n)];g=[q() for _ in p]
            r=ask('softmax',csv(p),csv(g));c=sum(a*b for a,b in zip(p,g))
            assert list(map(F,r['pull']))==[a*(b-c) for a,b in zip(p,g)]
            assert sum(map(F,r['pull']))==0;counts['softmax']+=1
        for mult,add,scratch,budget in [(10,1,0,0),(11,1,0,0),(13,1,0,0),(14,1,0,0),(15,1,1,0),(15,1,1,1),(2**100,2**80,2**90,2**91)]:
            r=ask('plan7',str(mult),str(add),str(scratch),str(budget))
            assert r['selected']==(mult>=14*add and scratch<=budget)
            assert int(r['native'])==24*mult+12*add and int(r['rank7'])==21*mult+54*add
            assert int(r['winograd'])==21*mult+45*add
            assert r['winograd_selected']==(mult>=11*add and scratch<=budget)
            counts['planner']+=1
        for args in [('rank7','1,2','1,2,3,4','1,2,3,4'),('structured','2','1','1,2','1','1,2','1,2','1,2'),
                     ('plan7','-1','1','0','0'),('softmax','1/0','1')]:
            assert 'error' in ask(*args);counts['invalid']+=1
    finally:
        process.stdin.close();process.wait(timeout=10)
    assert process.returncode==0
    return counts

def structured_tests():
    torch.manual_seed(892)
    results={}
    for dtype in (torch.float64,torch.float32):
        x=torch.randn(3,17,dtype=dtype);g=torch.randn_like(x);dx=torch.randn_like(x)
        gamma=torch.randn(17,dtype=dtype)
        activation_fns={'relu':fn.relu,'silu':fn.silu,'gelu':fn.gelu,'tanh':torch.tanh,'sigmoid':torch.sigmoid,'square':torch.square}
        cases=[(k,fun,lambda z,k=k:activation_derivative(z,k)) for k,fun in activation_fns.items()]
        cases += [('softmax',lambda z:z.softmax(-1),lambda z:softmax_derivative(z.softmax(-1))),
            ('rmsnorm',lambda z:z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*gamma,
                lambda z:rmsnorm_input_derivative(z,gamma)),
            ('layernorm',lambda z:fn.layer_norm(z,(17,),gamma,eps=1e-5),
                lambda z:layernorm_input_derivative(z,gamma))]
        for name,fun,builder in cases:
            cap=builder(x)
            _,pull=torch.func.vjp(fun,x);_,jvp=torch.func.jvp(fun,(x,),(dx,))
            ref=pull(g)[0];got=cap.pull(g)
            tol=1e-10 if dtype==torch.float64 else 2e-5
            torch.testing.assert_close(got,ref,atol=tol,rtol=tol)
            torch.testing.assert_close(cap.push(dx),jvp,atol=tol,rtol=tol)
            # Differentiate the coefficient-based pullback as research models do.
            _,higher=torch.func.jvp(lambda z:builder(z).pull(g),(x,),(dx,))
            _,reference_higher=torch.func.jvp(torch.func.grad(lambda z:(fun(z)*g).sum()),(x,),(dx,))
            torch.testing.assert_close(higher,reference_higher,atol=tol*10,rtol=tol*10)
            torch.testing.assert_close(cap.pull(g+dx),cap.pull(g)+cap.pull(dx),atol=tol*10,rtol=tol*10)
            results[f'{name}-{dtype}']={'rank':cap.left.shape[-1],
                'max_abs_vjp_error':(got-ref).abs().max().item(),'higher_order_passed':True,
                'new_saving_over_native_claimed':False}
    # Nonlinear boundary and coefficient-builder rejection checks.
    zero=torch.zeros(1,4,dtype=torch.float64)
    assert torch.equal(activation_derivative(zero,'relu').pull(torch.ones_like(zero)),zero)
    for builder in (rmsnorm_input_derivative,layernorm_input_derivative):
        try:builder(zero,torch.ones(4),eps=0)
        except ValueError:pass
        else:raise AssertionError('Invalid epsilon accepted')
    return results

def workload_tests():
    torch.manual_seed(893)
    def t(*shape):return torch.randn(*shape,dtype=torch.float64)*.2
    def recurrent(x,w,u):
        h=torch.zeros_like(x)
        for _ in range(4):h=torch.tanh(x@w+h@u)
        return h
    def ode(x,w):
        for _ in range(4):x=x+.1*torch.tanh(x@w)
        return x
    indices=torch.tensor([0,3,3,7,1])
    mask=(torch.rand(3,8)>.25).double()/.75
    cases={
        'linear_chain':(lambda x,a,b:(x@a)@b,(t(3,4),t(4,12),t(12,3))),
        'gelu_mlp':(lambda x,a,b:fn.gelu(x@a)@b,(t(3,4),t(4,12),t(12,3))),
        'swiglu':(lambda x,a,b,c:(fn.silu(x@a)*(x@b))@c,(t(3,4),t(4,12),t(4,12),t(12,3))),
        'convolution':(lambda x,w,b:fn.gelu(fn.conv2d(x,w,b,padding=1)),(t(1,2,5,5),t(3,2,3,3),t(3))),
        'softmax_attention':(lambda q,k,v:((q@k.transpose(-1,-2))/2).softmax(-1)@v,(t(2,5,4),t(2,5,4),t(2,5,3))),
        'layernorm_affine':(lambda x,g,b:fn.layer_norm(x,(8,),g,b,1e-5),(t(3,8),t(8),t(8))),
        'rmsnorm_affine':(lambda x,g:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*g,(t(3,8),t(8))),
        'recurrent_shared_weight':(recurrent,(t(3,8),t(8,8),t(8,8))),
        'embedding_repeated_indices':(lambda w:w[indices],(t(8,4),)),
        'fixed_dropout_residual':(lambda x,w:x+fn.silu(x@w)*mask,(t(3,8),t(8,8))),
        'research_polynomial_gate':(lambda x,w:(x@w).pow(3)+.1*(x@w).square(),(t(3,8),t(8,8))),
        'research_unrolled_ode':(ode,(t(3,8),t(8,8))),
    }
    report={}
    for name,(fun,args) in cases.items():
        directions=tuple(torch.randn_like(x)*.1 for x in args)
        y,pb=torch.func.vjp(fun,*args);g=torch.randn_like(y)
        _,jvp=torch.func.jvp(fun,args,directions);grads=pb(g)
        lhs=(jvp*g).sum();rhs=sum((a*b).sum() for a,b in zip(directions,grads))
        torch.testing.assert_close(lhs,rhs,atol=1e-10,rtol=1e-10)
        eps=1e-5
        plus=fun(*(x+eps*d for x,d in zip(args,directions)))
        minus=fun(*(x-eps*d for x,d in zip(args,directions)))
        fd=((plus-minus)*g).sum()/(2*eps)
        torch.testing.assert_close(lhs,fd,atol=2e-7,rtol=2e-6)
        loss=lambda *xs:fun(*xs).square().mean()
        gradfun=torch.func.grad(loss,argnums=tuple(range(len(args))))
        _,hvp=torch.func.jvp(gradfun,args,directions)
        gp=gradfun(*(x+eps*d for x,d in zip(args,directions)))
        gm=gradfun(*(x-eps*d for x,d in zip(args,directions)))
        for h,p,m in zip(hvp,gp,gm):torch.testing.assert_close(h,(p-m)/(2*eps),atol=2e-6,rtol=2e-5)
        report[name]={'passed':True,'differentiable_inputs_including_parameters':len(args),
            'duality_abs_error':abs((lhs-rhs).item()),'finite_difference_abs_error':abs((lhs-fd).item()),
            'higher_order_finite_difference':True,
            'scope':'PyTorch derivative-contract compatibility; not a Rocq proof of this complete model or a speedup.'}
    return report

def rank7_qualification():
    torch.manual_seed(894)
    results=[]
    for dtype in (torch.float64,torch.float32):
        for m,k,n in [(8,12,6),(64,64,64),(128,128,128),(256,256,256)]:
            a=torch.randn(m,k,dtype=dtype,requires_grad=True)
            b=torch.randn(k,n,dtype=dtype,requires_grad=True);g=torch.randn(m,n,dtype=dtype)
            def execute(fun):
                y=fun(a,b);return (y,*torch.autograd.grad(y,(a,b),g))
            variants={'native':lambda a,b:a@b,'rank7':rank7_reference,'winograd':winograd_reference,
                      'existing_streamed':lambda a,b:streamed_gemm(a,b,accuracy='algebraic')}
            ref=execute(variants['native'])
            tol=1e-10 if dtype==torch.float64 else 2e-4
            errors={}
            for label,fun in variants.items():
                if label=='native':continue
                errors[label]=[]
                for got,expected in zip(execute(fun),ref):
                    torch.testing.assert_close(got,expected,atol=tol,rtol=tol)
                    errors[label].append((got-expected).norm().item()/max(expected.norm().item(),1e-30))
            timing={};tape={}
            excluded={x.untyped_storage().data_ptr() for x in (a,b,g)}
            for label,fun in variants.items():
                saved={}
                def pack(x):
                    storage=x.untyped_storage();key=storage.data_ptr()
                    if key not in excluded:saved[key]=storage.nbytes()
                    return x
                with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):execute(fun)
                tape[label]=sum(saved.values())
                for _ in range(2):execute(fun)
                times=[]
                for _ in range(7):
                    start=time.perf_counter();execute(fun);times.append(time.perf_counter()-start)
                timing[label]=statistics.median(times)
            results.append({'shape':[m,k,n],'dtype':str(dtype),'relative_l2_errors_y_da_db':errors,
                'seconds':timing,'native_over_candidate_speedup':{k:timing['native']/v for k,v in timing.items() if k!='native'},
                'fresh_saved_storage_bytes':tape,'automatic_dispatch_enabled':False})
    return results

def matrix_higher_order():
    torch.manual_seed(895)
    a=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    b=torch.randn(6,4,dtype=torch.float64,requires_grad=True)
    directions=(torch.randn_like(a),torch.randn_like(b))
    def second(fun):
        loss=fun(a,b).square().mean()
        first=torch.autograd.grad(loss,(a,b),create_graph=True)
        return torch.autograd.grad(sum((g*d).sum() for g,d in zip(first,directions)),(a,b))
    reference=second(lambda a,b:a@b)
    results={}
    for label,fun in [('rank7',rank7_reference),('winograd',winograd_reference),
                      ('existing_streamed',lambda a,b:streamed_gemm(a,b,accuracy='algebraic'))]:
        candidate=second(fun)
        for got,expected in zip(candidate,reference):torch.testing.assert_close(got,expected,atol=1e-10,rtol=1e-10)
        results[label]={'passed':True,'max_abs_hvp_error':max((g-r).abs().max().item() for g,r in zip(candidate,reference))}
    return results

def main():
    torch.set_num_threads(2)
    result={'status':'running','torch':torch.__version__,'device':'CPU','threads':2,
        'scope':'Bounded mathematical reference qualification. No assertion of most-model speedup, global minimum memory, or GPU suitability.'}
    path=ROOT/'reports/broad-foundation.json';path.write_text(json.dumps(result,indent=2)+'\n')
    result['extracted_exact_cases']=exact_tests()
    result['structured_derivative_cases']=structured_tests()
    result['workload_families']=workload_tests()
    result['rank7_qualification']=rank7_qualification()
    result['matrix_higher_order']=matrix_higher_order()
    result['status']='passed';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','exact':result['extracted_exact_cases'],
        'structured_cases':len(result['structured_derivative_cases']),'workload_families':len(result['workload_families']),
        'matrix_speedups':[r['native_over_candidate_speedup'] for r in result['rank7_qualification']]},indent=2))

if __name__=='__main__':main()
