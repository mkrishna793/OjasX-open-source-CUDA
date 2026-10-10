"""Exact and floating qualification of the opt-in polynomial core reference."""
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import random
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ojasx2.feature_core import feature_basis,polynomial_core,polynomial_core_vjp,polynomial_cost

def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[sum((u*v for u,v in zip(row,col)),Q(0)) for col in zip(*b)] for row in a]
def lift(a,terms):
    def mul(xs):
        value=Q(1)
        for x in xs:value*=x
        return value
    return [[mul(row[i] for i in t) for t in terms] for row in a]
def pull(x,g,terms):
    result=[[Q(0) for _ in row] for row in x]
    for row,(values,up) in enumerate(zip(x,g)):
        for j,t in enumerate(terms):
            for pos,i in enumerate(t):
                value=up[j]
                for other,k in enumerate(t):
                    if pos!=other:value*=values[k]
                result[row][i]+=value
    return result

def exact_cases():
    rng=random.Random(815);cases=0
    for degree,k in product((1,2,3),(1,2,4)):
        terms,weights=feature_basis(k,degree)
        for iteration in range(8):
            m,n,p=2,3,2
            def random_matrix(h,w):return [[Q(rng.randrange(-2,3),rng.randrange(1,4)) for _ in range(w)] for _ in range(h)]
            x=random_matrix(m,k);a=random_matrix(k,n);b=random_matrix(n,p);g=random_matrix(m,p)
            z=mm(x,a);v=[[t**degree for t in row] for row in z]
            dz=[[gij*degree*t**(degree-1) for gij,t in zip(row,values)] for row,values in zip(mm(g,transpose(b)),z)]
            native=(mm(v,b),mm(dz,transpose(a)),mm(transpose(x),dz),mm(transpose(v),g))
            t=lift(x,terms);h=[[u*w for u,w in zip(row,weights)] for row in lift(transpose(a),terms)];h=transpose(h);c=mm(h,b)
            dc=mm(transpose(t),g);dt=mm(g,transpose(c));dh=mm(dc,transpose(b))
            dah=[[u*w for u,w in zip(row,weights)] for row in transpose(dh)]
            candidate=(mm(t,c),pull(x,dt,terms),transpose(pull(transpose(a),dah,terms)),mm(transpose(h),dc))
            assert candidate==native,(degree,k,iteration);cases+=1
    return cases

def floating_cases():
    torch.manual_seed(816);results=[]
    for dtype,degree,shape,pattern in product((torch.float32,torch.float64),(1,2,3),
            ((1,1,1,1),(2,2,3,2),(3,4,7,3),(2,3,5,2)),('random','zero','repeated','small')):
        m,k,n,p=shape
        x=torch.randn(k,m,dtype=dtype).T;a=torch.randn(n,k,dtype=dtype).T;b=torch.randn(p,n,dtype=dtype).T
        if pattern=='zero':x.zero_();a[0].zero_()
        elif pattern=='repeated':a[:,1:]=a[:,:1].clone()
        elif pattern=='small':x=x*1e-5
        g=torch.randn(m,p,dtype=dtype)
        native=lambda x,a,b:(x@a).pow(degree)@b
        y,pb=torch.func.vjp(native,x,a,b);ref=pb(g)
        cy,cg=polynomial_core_vjp(x,a,b,g,degree)
        ay,apb=torch.func.vjp(lambda x,a,b:polynomial_core(x,a,b,degree),x,a,b)
        tol=3e-4 if dtype==torch.float32 else 3e-10
        for got,want in zip((cy,*cg,ay,*apb(g)),(y,*ref,y,*ref)):
            torch.testing.assert_close(got,want,atol=tol,rtol=tol)
        results.append({'dtype':str(dtype),'degree':degree,'shape':list(shape),'pattern':pattern,
            'max_abs_error':max((v-r).abs().max().item() for v,r in zip((cy,*cg),(y,*ref)))})
    return results

def higher_order_and_optimizers():
    torch.manual_seed(817);higher=0;steps=[]
    for degree in (2,3):
        values=tuple(.2*torch.randn(*shape,dtype=torch.float64) for shape in ((3,2),(2,5),(5,2)))
        direction=tuple(torch.randn_like(v) for v in values)
        native=lambda *v:((v[0]@v[1]).pow(degree)@v[2]).square().mean()
        candidate=lambda *v:polynomial_core(*v,degree=degree).square().mean()
        _,nh=torch.func.jvp(torch.func.grad(native,argnums=(0,1,2)),values,direction)
        _,ch=torch.func.jvp(torch.func.grad(candidate,argnums=(0,1,2)),values,direction)
        for actual,expected in zip(ch,nh):torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-10)
        higher+=1
        for cls in (torch.optim.SGD,torch.optim.AdamW):
            a=[v.clone().requires_grad_() for v in values];b=[v.clone().requires_grad_() for v in values]
            opts=(cls(a,lr=1e-3),cls(b,lr=1e-3))
            for _ in range(4):
                opts[0].zero_grad();native(*a).backward();opts[0].step()
                opts[1].zero_grad()
                with torch.no_grad():
                    y=polynomial_core(*b,degree=degree);_,grads=polynomial_core_vjp(*b,2*y/y.numel(),degree=degree)
                for v,grad in zip(b,grads):v.grad=grad
                opts[1].step()
                for actual,expected in zip(b,a):torch.testing.assert_close(actual,expected,atol=3e-9,rtol=3e-9)
            steps.append({'degree':degree,'optimizer':cls.__name__,'steps':4,
                'max_parameter_error':max((v-w).abs().max().item() for v,w in zip(b,a))})
    return {'higher_order_all_operands':higher,'optimizer_steps':steps}

def validate_and_measure():
    x=torch.ones(2,2);a=torch.ones(2,3);b=torch.ones(3,2)
    bad=[lambda:polynomial_core(x,a,b,0),lambda:polynomial_core(x,a,b,True),
        lambda:polynomial_core(x,a,b,4),lambda:feature_basis(64,3),
        lambda:polynomial_core(x[:,:0],a[:0],b),lambda:polynomial_core(x,a,b.double()),
        lambda:polynomial_core(x,a[:,:2],b),lambda:polynomial_core_vjp(x,a,b,torch.ones(3,2)),
        lambda:polynomial_cost(0,2,3,2),lambda:polynomial_core(x.half(),a.half(),b.half())]
    for fun in bad:
        try:fun()
        except ValueError:pass
        else:raise AssertionError('Invalid reference request accepted')
    # Actual saved fresh storage is separate from the logical sufficient-value model.
    shape=(64,2,128,2);m,k,n,p=shape
    operands=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in ((m,k),(k,n),(n,p))]
    old={v.untyped_storage().data_ptr() for v in operands};measurements={}
    for name,fun in [('native',lambda x,a,b:(x@a).square()@b),('reference_core',polynomial_core)]:
        fresh={}
        def pack(t):
            storage=t.untyped_storage();pointer=storage.data_ptr()
            if pointer not in old:fresh[pointer]=storage.nbytes()
            return t
        start=time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda v:v):
            y=fun(*operands);y.sum().backward()
        measurements[name]={'fresh_saved_storage_bytes':sum(fresh.values()),'one_forward_backward_seconds':time.perf_counter()-start}
    return {'invalid_rejections':len(bad),'measurement_shape':list(shape),'measurements':measurements,
        'scope':'Single CPU observation, not a benchmark or allocator peak. Logical residual count is not actual graph storage.'}

def composed_graphs():
    torch.manual_seed(818);results=[]
    for dtype in (torch.float32,torch.float64):
        make=lambda *shape:.2*torch.randn(*shape,dtype=dtype)
        sequential=(make(3,2),make(2,5),make(5,2),make(2,7),make(7,2))
        shared=(make(3,2),make(2,5),make(5,2),make(2),make(2))
        def sequence(core,*v):
            x,a,b,c,d=v;middle=core(x,a,b,2).tanh()
            return x+core(middle,c,d,3)
        def branch(core,*v):
            x,a,b,gamma,beta=v
            y=core(x,a,b,2)+core(x,a,b,3)
            return torch.nn.functional.layer_norm(y,(2,),gamma,beta,1e-5)
        native=lambda x,a,b,degree:(x@a).pow(degree)@b
        for name,graph,values in [('sequential_tanh_residual',sequence,sequential),('shared_parameters_normalization',branch,shared)]:
            nf=lambda *v:graph(native,*v).square().mean()
            cf=lambda *v:graph(polynomial_core,*v).square().mean()
            argnums=tuple(range(len(values)))
            reference=torch.func.grad(nf,argnums=argnums)(*values)
            candidate=torch.func.grad(cf,argnums=argnums)(*values)
            tolerance=8e-4 if dtype==torch.float32 else 2e-9
            torch.testing.assert_close(cf(*values),nf(*values),atol=tolerance,rtol=tolerance)
            for got,want in zip(candidate,reference):torch.testing.assert_close(got,want,atol=tolerance,rtol=tolerance)
            if dtype==torch.float64:
                directions=tuple(torch.randn_like(v) for v in values)
                _,actual=torch.func.jvp(torch.func.grad(cf,argnums=argnums),values,directions)
                _,expected=torch.func.jvp(torch.func.grad(nf,argnums=argnums),values,directions)
                for got,want in zip(actual,expected):torch.testing.assert_close(got,want,atol=2e-8,rtol=2e-8)
            results.append({'dtype':str(dtype),'graph':name,'all_gradients_passed':True,'higher_order':dtype==torch.float64})
    return results

def main():
    torch.set_num_threads(2);path=ROOT/'reports/feature-core.json';path.write_text('{"status":"running"}\n')
    result={'status':'running','exact_fraction_cases':exact_cases(),'floating_cases':floating_cases(),
        'training':higher_order_and_optimizers(),'composed_graphs':composed_graphs(),'boundary':validate_and_measure()}
    result['status']='passed';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'exact_fraction_cases':result['exact_fraction_cases'],
        'floating_cases':len(result['floating_cases']),'training':result['training'],'boundary':result['boundary']},indent=2))

if __name__=='__main__':main()
