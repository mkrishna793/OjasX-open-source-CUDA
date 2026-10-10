"""Finite symbolic synthesis/refutation, no numerical CPU ML."""
from pathlib import Path
import json,hashlib
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,assumptions,claim,refute=False,repair=''):
    q=z.Solver();q.set(timeout=5000);q.add(*assumptions);assert q.check()==z.sat
    q.add(z.Not(claim));result=q.check();assert result==(z.sat if refute else z.unsat)
    checks.append(dict(name=name,negation=str(result),counterexample=str(q.model()) if refute else None,repair=repair))
g,d,a,b,c=z.Reals('g d a b c')
check('zero_cotangent_annihilates',[g==0],g*d==0)
check('nonzero_small_is_not_zero',[g>0,g<z.RealVal('1/10000'),d>0],g*d==0,True,'Never tolerance-prune cotangents')
check('sparse_output_not_sparse_input',[a==1,b==1,g==1],a*g+b*g==0,True,'Propagate through the actual Jacobian; preserve upstream attention')
check('cancellation_after_joint_sum',[a+b==0],(a+b)*d==0)
check('drop_branch_before_sum',[a+b==0],a*d==0,True,'Aggregate every branch cotangent before support selection')
check('value_zero_does_not_kill_hessian',[a==0],2==0,True,'At x=0, x squared has zero gradient but Hessian 2; native higher-order route')
m,k,n,r,A,C=z.Reals('m k n r A C')
domain=[m>0,k>0,n>0,r>0,A>=0,A<=m,C>=0,C<=r]
native=4*m*n*(k+r);candidate=6*A*n*k+4*A*n*C+2*m*r
check('replay_always_cheaper',domain,candidate<native,True,'Charge reconstruction and scan; choose by complete cost')
check('admitted_joint_cost',domain+[candidate<native],2*m*n*(k+r)+candidate<6*m*n*(k+r))
# Automatically expand a 2x3 cotangent and all Jacobian coordinates. The
# rectangle is selected by row/column support, never by activation value.
G=s.Matrix(2,3,s.symbols('g:6'));J=s.Matrix(6,4,s.symbols('j:24'))
mask=s.diag(1,0,1,0,0,0);v=s.Matrix(list(G));project=mask*v
restricted=v.subs({G[0,1]:0,G[1,0]:0,G[1,1]:0,G[1,2]:0})
assert (J.T*restricted-J.T*mask*restricted)==s.zeros(4,1)
shapes=[]
for M in (128,1024,4096):
 for K in (64,512):
  for N in (256,2048):
   for R in (64,512):
    for af in (0,1/16,1/4,1):
     for cf in (1/4,1):
      av=int(M*af);cv=int(R*cf);old=6*M*N*(K+R)
      new=2*M*N*(K+R)+6*av*N*K+4*av*N*cv+2*M*R
      shapes.append(dict(shape=[M,K,N,R],rows=av,columns=cv,eligible=new<old,ratio=new/old))
report=dict(status='passed_bounded_demand_discovery',checks=checks,shape_plans=shapes,
            symbolic_joint_jacobian=True,new_foundational_axioms=[],historical_novelty_claim=False,
            source_sha256={'tools/discover_demand_algebra.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
(ROOT/'reports/demand-discovery.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS demand discovery',len(checks),'checks',len(shapes),'plans')
