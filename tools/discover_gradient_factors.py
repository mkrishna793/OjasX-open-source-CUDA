"""Bounded counterexample-guided activation + all-parameter VJP synthesis.
Host symbolic mathematics only; no numerical tensors or invented axioms.
"""
from pathlib import Path
import json,sys
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from ojasx2.gradient_factor_compiler import factor_plan
from discover_backprop import lean,encode,derivative_proof
from discover_nonlinear_taylor import rocq

def main():
 checks=[];trace=[]
 def check(name,formula,expected):
  solver=z.Solver();solver.set(timeout=10000);solver.add(formula);result=solver.check()
  if str(result)!=expected:raise RuntimeError((name,str(result)))
  row={'name':name,'result':str(result),'witness':str(solver.model()) if result==z.sat else None}
  checks.append(row);return solver.model() if result==z.sat else None
 # CEGIS chooses the coefficient multiplying w*s*g in dx. Counterexamples
 # from the universal checker, not only a list of handpicked samples, refine it.
 c=z.Real('candidate_coefficient');w,g,d=z.Reals('w g slope')
 synthesis=z.Solver();synthesis.add(c>=-4,c<=4)
 for iteration in range(8):
  if synthesis.check()!=z.sat:raise RuntimeError('No synthesis candidate')
  cv=synthesis.model().eval(c,model_completion=True)
  verifier=z.Solver();verifier.add(cv*w*d*g!=w*d*g)
  outcome=verifier.check()
  if outcome==z.unsat:
   trace.append({'iteration':iteration,'coefficient':str(cv),'verification':'unsat: scalar joint chain coefficient exact'});break
  if outcome!=z.sat:raise RuntimeError('Verifier unknown')
  model=verifier.model();values=[model.eval(v,model_completion=True) for v in (w,d,g)]
  synthesis.add(c*values[0]*values[1]*values[2]==values[0]*values[1]*values[2])
  trace.append({'iteration':iteration,'coefficient':str(cv),'counterexample':str(model),'refinement':'Enforce this witnessed chain-rule equality; do not add an axiom.'})
 else:raise RuntimeError('CEGIS bound exhausted')
 check('synthesized_chain_coefficient',cv*w*d*g!=w*d*g,'unsat')
 # Rank-one upstream all-ones, valid ReLU slope from a diagonal-positive
 # preactivation. Its gated cotangent is the identity, hence rank two.
 a,b,c0,e=z.Reals('factor_a factor_b factor_c factor_e')
 check('reject_rank_one_survives_arbitrary_relu',z.And(a*c0==1,a*e==0,b*c0==0,b*e==1),'unsat')
 checks[-1]['interpretation']='Counterexample G=ones(2,2), Z=[[1,-1],[-1,1]], D=I: rank(G)=1 but rank(G elementwise D)=2. No rank-one factors exist.'
 x,y,ww,bb,gg,dd=z.Reals('x y ww bb gg dd')
 check('reject_bias_cotangent_omission',z.And(gg==1,dd==1,dd*gg!=0),'sat')
 check('reject_output_only_relu_all_parameter_gradient',z.And(x!=y,ww==0,bb==1,ww*x+bb==ww*y+bb,x!=y),'sat')
 check('reject_gate_after_contraction',z.And(x==1,y==1,ww==1,bb==2,dd==1,gg==0,(x*dd)*ww+(y*gg)*bb!=(x*ww+y*bb)*dd),'sat')
 # Refined rule: Hadamard products of represented tensors use the product
 # index of the two factor spaces. This is not a truncation to the old rank.
 p0,p1,q0,q1,s0,s1,t0,t1=s.symbols('p0 p1 q0 q1 s0 s1 t0 t1')
 lhs=(p0*q0+p1*q1)*(s0*t0+s1*t1)
 rhs=sum(p*q*ss*tt for p,q in ((p0,q0),(p1,q1)) for ss,tt in ((s0,t0),(s1,t1)))
 vv=(p0,p1,q0,q1,s0,s1,t0,t1);env={str(v):z.Real(str(v)) for v in vv}
 check('hadamard_product_factor_refinement',encode(lhs,env)!=encode(rhs,env),'unsat')
 xx,aa,bb0,dx,da,db,slope,up=s.symbols('x a b dx da db slope upstream')
 push=slope*up*(aa*dx+xx*da+db)
 pull=dx*(aa*slope*up)+da*(xx*slope*up)+db*(slope*up)
 vv=(xx,aa,bb0,dx,da,db,slope,up);env={str(v):z.Real(str(v)) for v in vv}
 check('combined_activation_joint_input_weight_bias_dagger',encode(push,env)!=encode(pull,env),'unsat')
 zz=s.Symbol('z',real=True);sig=1/(1+s.exp(-zz))
 ops={'identity':zz,'relu':s.Piecewise((0,zz<=0),(zz,True)),'square':zz**2,'cube':zz**3,
      'gelu':zz*(1+s.erf(zz/s.sqrt(2)))/2,'silu':zz*sig,'sigmoid':sig,'tanh':s.tanh(zz)}
 catalogue=[]
 for kind,f in ops.items():
  slope_expr=s.simplify(s.diff(f,zz))
  degree=int(s.degree(slope_expr,zz)) if slope_expr.is_polynomial(zz) else None
  catalogue.append({'activation':kind,'primal':str(f),'slope':str(slope_expr),'polynomial_slope_degree':degree,
   'representation':'Polynomial feature factors' if degree is not None else 'ReLU region factors or dense slope' if kind=='relu' else 'Dense exact slope; no closed polynomial feature representation found',
   'zero_policy':'PyTorch chosen derivative 0; no classical differentiability theorem at zero' if kind=='relu' else None})
 # Generate actual calculus + all-original-parameter statements for polynomial
 # examples. The general chain contract lives in GradientFactors.lean.
 L=['import OjasXMath.GradientFactors','import Mathlib.Analysis.Calculus.Deriv.Pow','namespace OjasXMath']
 R=['From Stdlib Require Import QArith.Qcanon Ring.','Open Scope Qc_scope.']
 generated=[]
 for degree in (1,2,3):
  loss=up*(aa*xx+bb0)**degree
  allvars=(xx,aa,bb0,up);args=' '.join(map(str,allvars))
  cot={v:s.diff(loss,v) for v in (xx,aa,bb0)}
  native=sum(cot[v]*dv for v,dv in ((xx,dx),(aa,da),(bb0,db)))
  compact=degree*up*(aa*xx+bb0)**(degree-1)*(aa*dx+xx*da+db)
  env={str(v):z.Real(str(v)) for v in (*allvars,dx,da,db)}
  check('generated_degree_'+str(degree)+'_joint',encode(native,env)!=encode(compact,env),'unsat')
  name='factor_discovered_degree_'+str(degree)+'_joint'
  L.append(f'theorem {name} ({args} dx da db : ℝ) : {lean(native)} = {lean(compact)} := by ring')
  R += [f'Theorem {name} ({args} dx da db : Qc) : {rocq(native)} = {rocq(compact)}.','Proof. ring. Qed.']
  generated.append(name)
  for v in (xx,aa,bb0):
   name=f'factor_discovered_degree_{degree}_derivative_{v}'
   L += [f'theorem {name} ({args} : ℝ) :',f' HasDerivAt (fun t : ℝ => {lean(loss.subs(v,s.Symbol("t")))}) ({lean(cot[v])}) {v} := by',f' convert {derivative_proof(loss,v)} using 1 <;> (try funext t) <;> dsimp <;> ring']
 L.append('end OjasXMath')
 (ROOT/'lean/OjasXMath/GradientDiscovered.lean').write_text('\n'.join(L)+'\n')
 (ROOT/'rocq/GradientDiscovered.v').write_text('\n'.join(R)+'\n')
 rows=[]
 for m in (16,128,1024):
  for k in (4,32,256):
   for n in (64,512,2048):
    for r in (1,8,64):
     for kind in ('identity','relu','square','gelu','silu','tanh','sigmoid'):
      from dataclasses import asdict
      rows.append(asdict(factor_plan(m,k,n,r,kind)))
 report={'status':'symbolic_checked_pending_kernels_gpu','added_axioms':[],
  'cegis_trace':trace,'z3_checks':checks,'activation_catalogue':catalogue,'shape_search':rows,
  'shape_count':len(rows),'eligible_shapes':sum(t['eligible'] for t in rows),
  'required_obligations':['Actual primitive derivative or explicit nonsmooth convention','All input, original weight and bias cotangents','Derivative factorization derived from the primal','No hidden rank truncation','Shape-dependent factor growth and full-step cost','Residual, transient and output storage distinguished','Floating-point test or rigorous error contract','Native rule when certificate absent'],
  'scope':'Bounded rule grammar and shape grid; cannot infer all future operations or universal cheaper backpropagation.'}
 (ROOT/'reports/gradient-factor-discovery.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS symbolic:',len(checks),'checks;',len(rows),'shape decisions;',sum(t['eligible'] for t in rows),'conditional candidates')

if __name__=='__main__':main()
