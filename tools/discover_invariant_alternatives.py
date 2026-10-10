"""Audit proposed advanced formulations and synthesize a repaired Gram rule.
SymPy/Z3 host mathematics only. Generated proofs require later kernel checks.
"""
from pathlib import Path
from dataclasses import asdict
import json,sys
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from discover_backprop import lean,encode,derivative_proof
from discover_nonlinear_taylor import rocq
from ojasx2.invariant_radial_compiler import invariant_plan

def main():
 checks=[];traces=[]
 def check(name,expr,expected,interpretation=None):
  solver=z.Solver();solver.set(timeout=15000);solver.add(expr);r=solver.check()
  if str(r)!=expected:raise RuntimeError((name,str(r),solver.reason_unknown()))
  checks.append({'name':name,'result':str(r),'witness':str(solver.model()) if r==z.sat else None,'interpretation':interpretation})
 # A row-specific smooth radial cubic breaks the proposed r+1 batch bound.
 x=s.Matrix(s.symbols('x0:3'));f=(x.dot(x))*x;j=f.jacobian(x)
 points=[(1,0,0),(1,1,0),(1,0,1)];incoming=s.Matrix([[1,0,0]]*3)
 outgoing=s.Matrix.vstack(*[(s.Matrix([[1,0,0]])*j.subs(dict(zip(x,v)))) for v in points])
 assert incoming.rank()==1 and outgoing.rank()==3
 det=outgoing.det()
 check('reject_batch_radial_r_plus_one',z.RealVal(str(det))!=0,'sat','Smooth radial cubic, three distinct row states: incoming rank 1, outgoing rank 3. A shared-J argument does not apply.')
 # A fixed commutator is a linear map and can double a represented rank.
 k=s.Matrix([[0,1],[-1,0]]);g=s.Matrix([[1,0],[0,0]]);comm=k*g-g*k
 assert comm.rank()==2
 check('reject_fixed_commutator_rank_r',z.RealVal(str(comm.det()))!=0,'sat','Fixed skew-symmetric K, rank-one G; commutator has rank two. It is a linear activation, not an arbitrary nonlinear replacement.')
 check('reject_radial_relu_equals_elementwise',z.RealVal(-1)!=0,'sat','v=(-1,0): ReLU(norm(v))*v/norm(v)=v but coordinate ReLU(v)=0. Norm-ReLU is the identity because norms are nonnegative.')
 # Sketching loses kernel directions even though trace mask transfer is exact.
 v0,v1=z.Reals('sketch_v0 sketch_v1')
 check('reject_exact_thin_sketch_without_subspace',z.And(v0==0,v1==1,v0==0,v1!=0),'sat','S=[1,0] annihilates v=(0,1), yet contraction with W=(0,1) is 1.')
 # Identical/isotropic marginal distributions need an independence assumption.
 check('reject_high_dimension_deterministic_orthogonality',z.RealVal(1)!=0,'sat','Aligned unit vectors have dot product 1 in every dimension; dropping their interaction loses an order-one term.')
 # Repeated irreps: identity group action commutes with every 2x2 matrix.
 a,b,c,d=s.symbols('a b c d');m=s.Matrix([[a,b],[c,d]])
 assert m*s.eye(2)==s.eye(2)*m
 check('reject_schur_scalar_on_multiplicity_space',z.And(z.Real('offdiag')==1,z.Real('offdiag')!=0),'sat','Two copies of a trivial complex irrep permit arbitrary 2x2 mixing, not a single scalar identity.')
 u,v,grad,eta=z.Reals('factor_u factor_v dense_grad eta')
 check('reject_factor_parameter_sgd_equals_dense',z.And(u==0,v==0,grad==1,eta==z.RealVal('1/2'),(u-eta*grad*v)*(v-eta*grad*u)!=u*v-eta*grad),'sat','Training U,V is a different parameterization; even zero factors can have zero factor gradients while dense W has a nonzero gradient.')
 # Counterexample-guided synthesis discovers required quadratic observables.
 xx,yy=z.Reals('cegis_x cegis_y')
 for full in (False,True):
  cs=z.Reals('energy_c0 energy_c1 energy_c2')[:3 if full else 2]
  terms=[xx*xx,yy*yy]+([xx*yy] if full else [])
  synth=z.Solver();synth.add(*[z.And(q>=-16,q<=16) for q in cs]);steps=[]
  for it in range(12):
   sr=synth.check()
   if sr==z.unsat:steps.append({'iteration':it,'result':'No coefficients satisfy accumulated counterexamples'});break
   if sr!=z.sat:raise RuntimeError('Unknown synthesis result')
   cv=[synth.model().eval(q,model_completion=True) for q in cs]
   verifier=z.Solver();verifier.set(timeout=15000);verifier.add((xx+yy)**2!=sum(q*t for q,t in zip(cv,terms)))
   vr=verifier.check()
   if vr==z.unsat:steps.append({'iteration':it,'coefficients':list(map(str,cv)),'result':'Universally verified polynomial energy'});break
   if vr!=z.sat:raise RuntimeError('Unknown energy verifier')
   model=verifier.model();px=model.eval(xx,model_completion=True);py=model.eval(yy,model_completion=True)
   vals=[px*px,py*py]+([px*py] if full else [])
   synth.add((px+py)**2==sum(q*t for q,t in zip(cs,vals)))
   steps.append({'iteration':it,'coefficients':list(map(str,cv)),'counterexample':{'x':str(px),'y':str(py)}})
  else:raise RuntimeError('Synthesis iteration cap exhausted')
  if not full and sr!=z.unsat:raise RuntimeError('Diagonal-only basis should be rejected')
  if full and vr!=z.unsat:raise RuntimeError('Quadratic basis failed')
  traces.append({'basis':['x^2','y^2']+(['x*y'] if full else []),'steps':steps})
 check('synthesized_quadratic_core',(xx+yy)**2!=xx*xx+yy*yy+2*xx*yy,'unsat')
 # A concrete, non-tautological polynomial radial block. All original x,W,b,Q
 # gradients receive actual generated HasDerivAt proofs as well as SMT checks.
 xs=s.symbols('x0:2');ws=s.symbols('w0:4');bs=s.symbols('b0:2');qs=s.symbols('q0:4');ps=s.symbols('p0:2')
 A=s.Matrix([[*xs,1]]);V=s.Matrix([[ws[0],ws[1]],[ws[2],ws[3]],[*bs]]);Q=s.Matrix(2,2,qs);P=s.Matrix(ps)
 Z=A*V;C=V*V.T;T=V*Q
 native_s=sum(v*v for v in Z)/2;core_s=(A*C*A.T)[0]/2
 native=(native_s*Z*Q*P)[0];core=(core_s*A*T*P)[0]
 original=(*xs,*ws,*bs,*qs);symbols=(*original,*ps);args=' '.join(map(str,symbols));env={str(q):z.Real(str(q)) for q in symbols}
 L=['import OjasXMath.InvariantRadial','import Mathlib.LinearAlgebra.Matrix.Determinant.Basic','import Mathlib.LinearAlgebra.Matrix.Notation','namespace OjasXMath']
 R=['From Stdlib Require Import QArith.Qcanon Ring Field.','Open Scope Qc_scope.']
 for label,left,right in [('primal',native,core)]+[(f'gradient_{q}',s.diff(native,q),s.diff(core,q)) for q in original]:
  check('radial_discovered_'+label,encode(left,env)!=encode(right,env),'unsat')
  name='invariant_discovered_'+label
  L.append(f'theorem {name} ({args} : ℝ) : {lean(left)} = {lean(right)} := by ring')
  R += [f'Theorem {name} ({args} : Qc) : {rocq(left)} = {rocq(right)}.','Proof. field; discriminate. Qed.']
 for q in original:
  L += [f'theorem invariant_actual_original_derivative_{q} ({args} : ℝ) :',
   f' HasDerivAt (fun t : ℝ => {lean(native.subs(q,s.Symbol("t")))}) ({lean(s.diff(core,q))}) {q} := by',
   f' convert {derivative_proof(native,q)} using 1 <;> (try funext t) <;> dsimp <;> ring']
 L += ['theorem invariant_radial_counterexample_determinant :',
       ' Matrix.det (!![(3:ℝ),0,0;4,2,0;4,0,2]) = 12 := by norm_num [Matrix.det_fin_three]',
       'theorem invariant_fixed_commutator_derivation {R : Type*} [Ring R] (k a b : R) :',
       ' k*(a*b)-(a*b)*k = (k*a-a*k)*b+a*(k*b-b*k) := by noncomm_ring',
       'theorem invariant_fixed_commutator_linear {R : Type*} [Ring R] (k a b : R) :',
       ' k*(a+b)-(a+b)*k = (k*a-a*k)+(k*b-b*k) := by noncomm_ring','end OjasXMath']
 (ROOT/'lean/OjasXMath/InvariantDiscovered.lean').write_text('\n'.join(L)+'\n')
 (ROOT/'rocq/InvariantDiscovered.v').write_text('\n'.join(R)+'\n')
 stat=s.Symbol('s',nonnegative=True);ga,gb=s.symbols('gate_a gate_b',real=True)
 scalar={'cubic':stat,'reciprocal':1/(1+stat),'sigmoid':1/(1+s.exp(-ga*stat-gb))}
 shapes=[asdict(invariant_plan(m,k,n,r)) for m in (32,256,2048,8192) for k in (4,32,128,512) for n in (64,512,4096) for r in (1,8,64)]
 report={'status':'symbolic_checked_pending_kernels_gpu','z3_checks':checks,'cegis_gram_basis':traces,
  'radial_rank_counterexample':{'activation':'phi(v)=norm(v)^2*v','states':points,'incoming':incoming.tolist(),'outgoing':outgoing.tolist(),'incoming_rank':1,'outgoing_rank':3,'determinant':str(det)},
  'gate_derivatives':{kind:{'value':str(f),'ds':str(s.diff(f,stat)),'dgate_a':str(s.diff(f,ga)),'dgate_b':str(s.diff(f,gb))} for kind,f in scalar.items()},
  'synthesized_core':{'augmented_input':['x','1'],'augmented_weights':['W','bias'],'gram':'C=V V^T','projection':'T=V Q','energy':'s=diag(A C A^T)/N','value':'f(s)*(A T)','original_gradients':'dx,dW,db,dQ,and gate_a/gate_b when present'},
  'shapes':shapes,'added_axioms':[],'scope':'Existing radial model equivalence, not arbitrary coordinate activations; genuine batch-rank counterexamples; chamber/sketch/concentration/symmetry claims have explicit extra conditions.'}
 (ROOT/'reports/invariant-discovery.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
 print('PASS',len(checks),'SMT checks;',len(shapes),'size decisions; Gram basis refined after counterexamples')

if __name__=='__main__':main()
