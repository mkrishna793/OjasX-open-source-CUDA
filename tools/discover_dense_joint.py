"""Synthesize exact transposed bilinear coefficients and refute unsafe axioms."""
from pathlib import Path
import sys,json,hashlib
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
# These imports define algebra/metadata only; no tensors are executed on host.
from ojasx2.bilinear import U,V,W
from ojasx2.dense_joint import rank7_plan
a=s.symbols('a0:4');b=s.symbols('b0:4');g=s.symbols('g0:4')
ea=[sum(c*t for c,t in zip(row,a)) for row in U]
eb=[sum(c*t for c,t in zip(row,b)) for row in V]
p=[x*y for x,y in zip(ea,eb)]
y=[sum(c*t for c,t in zip(row,p)) for row in W]
native=[a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]]
assert all(s.expand(x-t)==0 for x,t in zip(y,native))
gamma=[sum(W[j][i]*g[j] for j in range(4)) for i in range(7)]
da=[sum(U[i][j]*gamma[i]*eb[i] for i in range(7)) for j in range(4)]
db=[sum(V[i][j]*ea[i]*gamma[i] for i in range(7)) for j in range(4)]
loss=sum(x*t for x,t in zip(y,g))
assert all(s.expand(x-s.diff(loss,t))==0 for x,t in zip(da+db,a+b))
checks=[]
def check(name,premises,claim,counterexample=False,repair=''):
    q=z.Solver();q.set(timeout=5000);q.add(*premises);assert q.check()==z.sat
    q.add(z.Not(claim));res=q.check()
    assert res==(z.sat if counterexample else z.unsat)
    checks.append(dict(name=name,negation=str(res),counterexample=str(q.model()) if counterexample else None,repair=repair))
x,t,h,d,gg,ww,qq=z.Reals('x t h d g w q')
check('output_only_square_loses_derivative',[x==-t,x!=0],2*x*gg==2*t*gg,True,'Retain preactivation; output equality alone does not determine the native derivative.')
check('retained_preinput_preserves_native_slope',[x==t],d*x*gg==d*t*gg)
check('parameter_gradient_cannot_be_dropped',[x!=0,gg!=0],x*gg==0,True,'Keep every requested original input, bias and parameter gradient.')
actual,approx,eps=z.Reals('actual approx eps')
check('absolute_does_not_imply_relative',[eps>0,z.Abs(actual-approx)<=eps,actual==0],actual==approx,True,'Strict mode remains native; altered summation order needs cancellation-aware numerical qualification.')
native,candidate,replay=z.Reals('native candidate replay')
check('replay_must_be_paid',[candidate<native,replay>0],candidate+replay<native,True,'Use the replayed complete-step count, not the cached count.')
check('qualified_complete_arithmetic',[candidate+replay<native],candidate+replay<native)
encode,decode,stored,original=z.Reals('encode decode stored original')
check('memory_does_not_imply_speed',[stored<original,decode>0],encode+decode<encode,True,'Record memory and latency qualification separately.')
check('encode_storage_must_be_paid',[stored<original,decode>original],stored+decode<original,True,'Account for encoded buffers, workspace and originals independently.')
plans=[dict(shape=[m,k,n],cache=cache,**rank7_plan(m,k,n,cache,budget,ga,gb))
       for m in (16,64,256,1024) for k in (16,128,512) for n in (16,128,512)
       for cache in (False,True) for budget in (0,256*1024**2) for ga in (False,True) for gb in (False,True)]
r=dict(status='passed_bounded_dense_discovery',symbolic_primal_coefficients=64,synthesized_gradient_coordinates=8,
       symbolic_synthesis='Existing Winograd U,V,W; transpose the shared circuit against the trilinear pairing. No historical novelty claim.',
       checks=checks,shape_plans=plans,historical_novelty_claim=False,
       source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('tools/discover_dense_joint.py','ojasx2/dense_joint.py','ojasx2/bilinear.py','lean/OjasXMath/DenseJoint.lean','rocq/DenseJoint.v')})
(ROOT/'reports/dense-discovery.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS',len(checks),'Z3 checks/counterexamples,',len(plans),'finite cost/storage plans, exact primal and both transposed encoders')
