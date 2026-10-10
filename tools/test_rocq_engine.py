"""Independent Fraction and PyTorch CPU oracles for the extracted Rocq engine."""
from fractions import Fraction as F
from pathlib import Path
import itertools
import json
import math
import random
import subprocess
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
random.seed(2902)

def prod(xs):
    result=F(1)
    for x in xs: result*=x
    return result

def gradient(xs):
    return [prod(xs[:i]+xs[i+1:]) for i in range(len(xs))]

def encode(xs): return ','.join(str(x) for x in xs)
def rational_vector(xs): return [F(x) for x in xs]
def random_q(): return F(random.randint(-5,5),random.choice([1,2,3,4,7]))
def mm(a,b,m,k,n): return [[sum((a[i][h]*b[h][j] for h in range(k)),F(0)) for j in range(n)] for i in range(m)]
def tr(a,m,n): return [[a[i][j] for i in range(m)] for j in range(n)]
def random_matrix(m,n): return [[random_q() for _ in range(n)] for _ in range(m)]
def encode_matrix(a): return encode(list(itertools.chain.from_iterable(a)))

def main():
    start=time.perf_counter()
    process=subprocess.Popen([str(ROOT/'rocq/extracted/ojasx_math_cli')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    counts={'product_cases':0,'optimizer_cases':0,'matrix_cases':0,'relu_cases':0,'planner_cases':0,
            'trajectory_steps':0,'invalid_input_cases':0,'pytorch_fp64_comparisons':0}
    max_error=0.0
    def query(*parts):
        process.stdin.write('|'.join(parts)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line: raise AssertionError('Extracted engine terminated unexpectedly')
        return json.loads(line)
    def close(actual,expected):
        nonlocal max_error
        actual=torch.tensor(actual,dtype=torch.float64)
        error=(actual-expected).abs().max().item() if actual.numel() else 0.0
        max_error=max(max_error,error)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        counts['pytorch_fp64_comparisons']+=1
    try:
        cases=[[],[F(0)],[F(0),F(0)],[F(1),F(0),F(-2)],[F(2**200+1,7),F(-3,11)],
               [F(1,2)]*64,[F(-1)]*128]
        cases += [[random_q() for _ in range(random.randrange(1,14))] for _ in range(100)]
        for xs in cases:
            upstream=random_q()
            result=query('product',encode(xs),str(upstream))
            expected=gradient(xs)
            assert F(result['value'])==prod(xs)
            assert rational_vector(result['gradient'])==expected==rational_vector(result['reference'])
            assert rational_vector(result['pullback'])==[upstream*g for g in expected]
            counts['product_cases']+=1
            if xs and max(abs(float(x)) for x in xs)<100 and len(xs)<15:
                tx=torch.tensor([float(x) for x in xs],dtype=torch.float64,requires_grad=True)
                tg,=torch.autograd.grad(tx.prod(),tx)
                close([float(x) for x in expected],tg)

        for _ in range(50):
            xs=[random_q() for _ in range(random.randrange(0,6))]
            target=random_q();eta=random.choice([F(0),F(1,100),F(1,4),F(8),F(-1)])
            fuel=random.randrange(0,7)
            result=query('step',encode(xs),str(target),str(eta),str(fuel))
            gs=[2*(prod(xs)-target)*g for g in gradient(xs)]
            step=[x-eta*g for x,g in zip(xs,gs)]
            before=(prod(xs)-target)**2
            safe=xs;rate=eta
            for _ in range(fuel):
                candidate=[x-rate*g for x,g in zip(xs,gs)]
                if (prod(candidate)-target)**2<=before: safe=candidate;break
                rate/=2
            assert rational_vector(result['gradient'])==gs
            assert rational_vector(result['step'])==step==rational_vector(result['reference'])
            assert rational_vector(result['safe'])==safe
            assert F(result['loss_before'])==before
            assert F(result['loss_after'])==(prod(safe)-target)**2<=before
            counts['optimizer_cases']+=1

        xs=[F(1,2),F(2,3)]
        for _ in range(4):
            target=F(1);eta=F(1,100)
            expected=[x-eta*2*(prod(xs)-target)*g for x,g in zip(xs,gradient(xs))]
            result=query('step',encode(xs),str(target),str(eta),'4')
            xs=rational_vector(result['step']);assert xs==expected
            counts['trajectory_steps']+=1

        shapes=[(6,1,6,1),(1,6,1,6),(3,2,5,2),(0,2,3,2),(2,0,3,2),(2,3,0,2),(2,3,2,0)]
        shapes += [tuple(random.randint(1,5) for _ in range(4)) for _ in range(50)]
        for m,k,n,p in shapes:
            x=random_matrix(m,k);a=random_matrix(k,n);b=random_matrix(n,p);g=random_matrix(m,p)
            result=query('matrix',f'{m},{k},{n},{p}',encode_matrix(x),encode_matrix(a),encode_matrix(b),encode_matrix(g))
            h=mm(x,a,m,k,n);e=mm(g,tr(b,n,p),m,p,n)
            expected={'y':mm(h,b,m,n,p),'dx':mm(e,tr(a,k,n),m,n,k),
                      'da':mm(tr(x,m,k),e,k,m,n),'db':mm(tr(h,m,n),g,n,m,p)}
            expected['next_a']=[[a[i][j]-F(1,10)*expected['da'][i][j] for j in range(n)] for i in range(k)]
            expected['next_b']=[[b[i][j]-F(1,10)*expected['db'][i][j] for j in range(p)] for i in range(n)]
            for key in expected: assert [[F(v) for v in row] for row in result[key]]==expected[key]
            counts['matrix_cases']+=1
            if min(m,k,n,p)>0:
                tx=torch.tensor([[float(v) for v in row] for row in x],dtype=torch.float64,requires_grad=True)
                ta=torch.tensor([[float(v) for v in row] for row in a],dtype=torch.float64,requires_grad=True)
                tb=torch.tensor([[float(v) for v in row] for row in b],dtype=torch.float64,requires_grad=True)
                tg=torch.tensor([[float(v) for v in row] for row in g],dtype=torch.float64)
                ty=(tx@ta)@tb
                grads=torch.autograd.grad(ty,(tx,ta,tb),tg)
                for key,tensor in zip(['y','dx','da','db'],[ty,*grads]):
                    close([[float(v) for v in row] for row in expected[key]],tensor)

        for _ in range(30):
            xs=[F(0)]+[random_q() for _ in range(12)];gs=[random_q() for _ in xs]
            result=query('relu',encode(xs),encode(gs))
            assert rational_vector(result['output'])==[max(x,0) for x in xs]
            assert result['mask']==[x>0 for x in xs]
            expected=[g if x>0 else F(0) for x,g in zip(xs,gs)]
            assert rational_vector(result['gradient'])==expected==rational_vector(result['packed'])
            counts['relu_cases']+=1

        planner_shapes=[(512,32,2048,32),(1,64,2,64),(2**100+3,2**80+1,2**120,2**60)]
        planner_shapes += [tuple(random.randrange(0,5000) for _ in range(4)) for _ in range(100)]
        example=None
        for m,k,n,p in planner_shapes:
            result=query('plan',str(m),str(k),str(n),str(p))
            native=6*m*k*n+6*m*n*p;contracted=6*p*k*(m+n)
            choose=contracted<=native and k*p<=m*n
            assert result['contracted']==choose
            assert int(result['native_work'])==native and int(result['contracted_work'])==contracted
            assert int(result['selected_work'])==(contracted if choose else native)<=native
            assert int(result['native_tape'])==m*n
            assert int(result['selected_tape'])==(k*p if choose else m*n)<=m*n
            counts['planner_cases']+=1
            if (m,k,n,p)==(512,32,2048,32): example=result

        for parts in [('product','1/0','1'),('product','1/-2','1'),('product','x','1'),
                      ('plan','-1','2','3','4'),('step','1','0','1','-1'),
                      ('matrix','2,2,2,2','1','1','1','1'),('relu','1,2','1')]:
            assert 'error' in query(*parts);counts['invalid_input_cases']+=1
    finally:
        process.stdin.close()
        process.wait(timeout=10)
    assert process.returncode==0
    report={'status':'passed','counts':counts,'max_pytorch_fp64_absolute_error':max_error,
        'pytorch_version':torch.__version__,'device':'CPU','seed':2902,
        'seconds':time.perf_counter()-start,'favorable_model_example':example,
        'scope':'Fraction comparisons are exact; PyTorch comparisons are sampled FP64 tolerance checks. No speed benchmark.'}
    (ROOT/'reports/rocq-engine-tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
