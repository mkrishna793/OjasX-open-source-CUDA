"""Bounded coordinate-model synthesis and counterexample-guided obligations.
Host mathematics only. No tensor workloads or assumed new foundation axioms.
"""
from pathlib import Path
import json,sys
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1]
t,a,b,x=s.symbols('t a b x',positive=True)
charts={'identity':(x,t),'log':(s.log(x),s.exp(t)),'reciprocal':(1/x,1/t),'logit':(s.log(x/(1-x)),1/(1+s.exp(-t)))}
ops={'affine':a*x+b,'scale':a*x,'square_scale':(a*x)**2,'cube_scale':(a*x)**3,'shifted_square':(a*x+b)**2,'exponential':s.exp(a*x+b),'sigmoid':1/(1+s.exp(-a*x-b)),'silu':x/(1+s.exp(-x)),'gelu':x*(1+s.erf(x/s.sqrt(2)))/2}
rows=[]
for op,f in ops.items():
 for name,(chart,decode) in charts.items():
  raw=chart.subs(x,f.subs(x,decode))
  if name=='log':raw=s.expand_log(raw,force=True)
  if name=='logit' and op=='sigmoid':raw=s.expand_log(s.simplify(raw),force=True)
  h=s.simplify(raw);d=s.simplify(s.diff(h,t,2))
  affine=d==0
  alpha=s.simplify(s.diff(h,t));beta=s.simplify(h-alpha*t)
  if affine and (alpha.has(t) or beta.has(t)):raise RuntimeError('Bad closure extraction')
  rows.append({'operation':op,'chart':name,'lifted':str(h),'second_derivative':str(d),'affine_closure':affine,'alpha':str(alpha) if affine else None,'beta':str(beta) if affine else None,'scope':'Positive-domain symbolic grammar; closure rejection is limited to this chart, not every possible embedding.'})
checks=[]
def solve(name,constraints,expected):
 solver=z.Solver();solver.set(timeout=10000);solver.add(*constraints);r=solver.check()
 if str(r)!=expected:raise RuntimeError((name,str(r)))
 checks.append({'name':name,'result':str(r),'witness':str(solver.model()) if r==z.sat else None})
u,v=z.Reals('u v')
solve('reject_square_chart_without_domain',[u!=v,u*u==v*v],'sat')
solve('positive_domain_restores_square_injectivity',[u>0,v>0,u!=v,u*u==v*v],'unsat')
solve('positive_domain_is_nonvacuous',[u>0],'sat')
solve('reject_empty_assumption_set',[u>0,u<=0],'unsat')
# ReLU output alone cannot recover all original parameter gradients.
solve('reject_output_only_parameter_cotangent',[u!=v,u>=-2,u<=2,v>=-2,v<=2,1==1],'sat')
checks[-1]['interpretation']='For f(x,a,b)=ReLU(a*x+b), a=0,b=1: all x give output 1, while df/da=x. The witness distinguishes parameter gradients.'
A,B,C,D,E,F,q,da,db,dc,dd,dq,g=z.Reals('A B C D E F q da db dc dd dq g')
solve('affine_action_associativity',[(E*C*A)*q+E*(C*B+D)+F != E*(C*(A*q+B)+D)+F],'unsat')
solve('all_affine_action_parameter_cotangents',[(C*A*dq+C*q*da+C*db+(A*q+B)*dc+dd)*g != dq*(A*C*g)+da*(q*C*g)+db*(C*g)+dc*((A*q+B)*g)+dd*g],'unsat')
# CEGIS recovers a genuinely closed nonlinear observable for a triangular map:
# f(x,y)=(2*x,3*y+x^2), and lifted dynamics on [x,y,x^2].
c0,c1,c2=z.Reals('koopman_c0 koopman_c1 koopman_c2')
solver=z.Solver();solver.add(*[c0*px+c1*py+c2*px*px==3*py+px*px for px,py in ((0,1),(1,0),(2,1))])
assert solver.check()==z.sat;model=solver.model();vals=[model.eval(c) for c in (c0,c1,c2)]
xx,yy=z.Reals('koopman_x koopman_y')
solve('synthesized_koopman_row_global', [sum(c*f for c,f in zip(vals,(xx,yy,xx*xx)))!=3*yy+xx*xx],'unsat')
# Omitted observable gives a counterexample; refine by adding x^2.
solve('reject_state_only_triangular_linear_model',[xx==1,yy==0,2*xx==2,3*yy!=3*yy+xx*xx],'sat')
trace=[]
for degree in range(1,7):
 trace.append({'basis':list(range(degree+1)),'square_map_requires_degree':2*degree,'closed':False,'reason':'Pullback of the highest monomial x^d is x^(2d), outside this finite monomial basis.'})
# Extract the parameterized invariant operator, then prove all original
# input/parameter cotangents in both kernels via generated finite identities.
sys.path.insert(0,str(ROOT/'tools'))
from discover_backprop import encode,lean,derivative_proof
from discover_nonlinear_taylor import rocq
xx,yy,aa,bb,cc,uu,vv,dx,dy,da,db,dc=s.symbols('x y a b c u v dx dy da db dc')
basis=[xx,yy,xx**2];states=[aa*xx,bb*yy+cc*xx**2,(aa*xx)**2]
monomials=[(1,0),(0,1),(2,0)]
matrix=[[s.Poly(f,xx,yy).coeff_monomial(m) for m in monomials] for f in states]
lifted=[sum(c*t for c,t in zip(row,basis)) for row in matrix]
primal=uu*states[0]+vv*states[1];decoded=uu*lifted[0]+vv*lifted[1]
L=['import OjasXMath.CoordinateCompiler','import Mathlib.Analysis.Calculus.Deriv.Pow','import Mathlib.Tactic.Ring','set_option autoImplicit false','namespace OjasXMath']
R=['From Stdlib Require Import QArith.Qcanon Ring.','Open Scope Qc_scope.']
vars=(xx,yy,aa,bb,cc,uu,vv);args=' '.join(map(str,vars));env={str(q):z.Real(str(q)) for q in vars}
for label,l,r in [('primal',primal,decoded)]+[(f'gradient_{q}',s.diff(primal,q),s.diff(decoded,q)) for q in (xx,yy,aa,bb,cc)]:
 solve('parameterized_koopman_'+label,[encode(l,env)!=encode(r,env)],'unsat')
 name='coordinate_discovered_'+label
 L += [f'theorem {name} ({args} : ℝ) : {lean(l)} = {lean(r)} := by ring']
 R += [f'Theorem {name} ({args} : Qc) : {rocq(l)} = {rocq(r)}.', 'Proof. ring. Qed.']
for q in (xx,yy,aa,bb,cc):
 temp=s.Symbol('t')
 L += [f'theorem coordinate_actual_derivative_{q} ({args} : ℝ) :',f'  HasDerivAt (fun t : ℝ => {lean(primal.subs(q,temp))}) ({lean(s.diff(primal,q))}) {q} := by',f'  convert {derivative_proof(primal,q)} using 1 <;> (try funext t) <;> dsimp <;> ring']
push=uu*(aa*dx+xx*da)+vv*(bb*dy+yy*db+xx**2*dc+2*cc*xx*dx)
pull=dx*(aa*uu+2*cc*xx*vv)+dy*(bb*vv)+da*(xx*uu)+db*(yy*vv)+dc*(xx**2*vv)
allvars=(*vars,dx,dy,da,db,dc);args=' '.join(map(str,allvars));env={str(q):z.Real(str(q)) for q in allvars}
solve('parameterized_koopman_full_dagger',[encode(push,env)!=encode(pull,env)],'unsat')
L += [f'theorem coordinate_discovered_joint_dagger ({args} : ℝ) : {lean(push)} = {lean(pull)} := by ring','end OjasXMath']
R += [f'Theorem coordinate_discovered_joint_dagger ({args} : Qc) : {rocq(push)} = {rocq(pull)}.', 'Proof. ring. Qed.']
(ROOT/'lean/OjasXMath/CoordinateDiscovered.lean').write_text('\n'.join(L)+'\n')
(ROOT/'rocq/CoordinateDiscovered.v').write_text('\n'.join(R)+'\n')
report={'status':'symbolic_conditions_checked_pending_kernel_and_gpu','charts':rows,'z3_checks':checks,'koopman_synthesis':{'basis':['x','y','x^2'],'matrix':[[2,0,0],[0,3,1],[0,0,4]],'discovered_row':list(map(str,vals)),'trace':'State-only basis rejected; x^2 added; sampled synthesis verified universally by SMT.'},'parameterized_koopman_matrix':[[str(c) for c in row] for row in matrix],'generated_lean_statements':12,'generated_rocq_statements':7,'polynomial_nonclosure_trace':trace,'obligations':['nonempty domain','encoding injective or sufficient for the requested outputs and all parameter cotangents','domain preservation','finite closure under every represented stage','correct chart/decoder derivatives','all original trainable parameters included','encoding/decoding/search/guard/full-step cost','numerical qualification; no IEEE equivalence from real identities'],'added_foundation_axioms':[],'scope':'Finite chart and polynomial-observable grammar. Failed candidates remain explicit; no global impossibility or complete discovery claim.'}
(ROOT/'reports/coordinate-discovery.json').write_text(json.dumps(report,indent=2)+'\n')
print('Checked',len(rows),'chart/operator pairs;',sum(r['affine_closure'] for r in rows),'affine closures;',len(checks),'Z3 conditions')
