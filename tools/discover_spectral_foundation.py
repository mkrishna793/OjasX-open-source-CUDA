"""Bounded symbolic spectral/Veronese discovery, no host numerical ML.
Generates matching primal/derivative coefficients and kernel proof source.
"""
from pathlib import Path
import json,math,sys
import sympy as s
import z3 as z
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from discover_backprop import lean,encode,derivative_proof
from discover_nonlinear_taylor import rocq

def main():
 x=s.Symbol('x');v=s.Symbol('sig');checks=[]
 def check(name,claim,want='unsat'):
  solver=z.Solver();solver.set(timeout=15000);solver.add(claim);result=solver.check()
  if str(result)!=want:raise RuntimeError((name,result,solver.reason_unknown()))
  checks.append({'name':name,'result':str(result),'witness':str(solver.model()) if result==z.sat else None})
 xv,yv=z.Reals('x y')
 check('normalization_does_not_imply_gaussian',z.And(xv==-1,yv==1,(xv+yv)/2==0,(xv*xv+yv*yv)/2==1),'sat')
 # A Gaussian L2 statement is not a pointwise or cotangent-relative contract.
 g,e=z.Reals('gradient error');check('absolute_error_does_not_imply_relative_error',z.And(g==z.RealVal('1/100000000'),e==z.RealVal('1/1000000'),e<=z.RealVal('1/10000'),e>z.RealVal('1/10000')*g),'sat')
 funcs={'gelu':x*(1+s.erf(x/s.sqrt(2)))/2,'silu':x/(1+s.exp(-x))}
 coefficients={}
 for kind,f in funcs.items():
  coefficients[kind]={str(d):[str(s.expand(s.series(f,x,0,d+1).removeO()).coeff(x,j)) for j in range(d+1)] for d in (2,4,6,8,10,12)}
 hermite=[]
 for k in range(14):
  if k==0:c=s.Rational(1,2)
  elif k%2==0:c=s.Integer(0)
  else:
   pol=s.Poly((k+1)*s.hermite_prob(k-1,x)-x*s.hermite_prob(k,x),x)
   c=sum(coef*s.factorial(m)/(4**(m//2)*s.factorial(m//2)) for (m,),coef in pol.terms() if m%2==0)/(2*s.sqrt(s.pi)*s.factorial(k))
  hermite.append(str(s.simplify(c)))
 # Correct Gaussian He_k squared norm is k!, not one.
 he2=x*x-1;moment=s.expand(he2*he2)
 expectation=sum(c*s.factorial(m)/(2**(m//2)*s.factorial(m//2)) for (m,),c in s.Poly(moment,x).terms())
 assert expectation==2
 # CEGIS learns the mixed coefficient of a two-coordinate square.
 c=z.Real('mixed_coefficient');synth=z.Solver();trace=[]
 for it in range(8):
  if synth.check()!=z.sat:raise RuntimeError('CEGIS inconsistent')
  value=synth.model().eval(c,model_completion=True);ver=z.Solver();ver.add((xv+yv)**2!=xv*xv+yv*yv+value*xv*yv)
  status=ver.check()
  if status==z.unsat:trace.append({'coefficient':str(value),'verification':'unsat'});break
  if status!=z.sat:raise RuntimeError('CEGIS unknown')
  model=ver.model();xx=model.eval(xv,model_completion=True);yy=model.eval(yv,model_completion=True)
  synth.add((xx+yy)**2==xx*xx+yy*yy+c*xx*yy);trace.append({'coefficient':str(value),'counterexample':str(model)})
 else:raise RuntimeError('CEGIS exhausted')
 check('cegis_symmetric_multiplicity',(xv+yv)**2!=xv*xv+yv*yv+value*xv*yv)
 # A polynomial must be integrated consistently with its derivative.
 check('reject_gradient_only_rewrite',z.And(xv==1,yv==1,2*xv*yv!=yv),'sat')
 shapes=[]
 for rank in (1,2,4,8,32,128,768):
  for degree in (3,4,6,12):
   homogeneous=math.comb(rank+degree-1,degree)
   all_degrees=math.comb(rank+degree,degree)
   shapes.append({'input_rank':rank,'derivative_degree':degree,'highest_homogeneous_width':homogeneous,
    'uncompressed_all_degree_width':all_degrees,'upstream_rank_8_width':8*all_degrees,
    'interpretation':'Ambient rank bound is not automatically small; sums of degrees require a union basis. Bias can be folded without adding a feature coordinate.'})
 # Bias translation: coefficients share the Taylor shifts T_s(b).
 L=['import OjasXMath.SpectralFoundation','import Mathlib.Analysis.Calculus.Deriv.Pow','namespace OjasXMath']
 R=['From Stdlib Require Import QArith.Qcanon Ring.','Open Scope Qc_scope.'];generated=[]
 xx,yy,ww,vv,bb,qq,cc=s.symbols('x y w v b q c')
 for degree in (2,4,6):
  native=cc*(xx*ww+yy*vv+bb)**degree*qq
  compact=sum(cc*s.factorial(degree)/(s.factorial(i)*s.factorial(j)*s.factorial(degree-i-j))*xx**i*yy**j*ww**i*vv**j*bb**(degree-i-j)*qq for i in range(degree+1) for j in range(degree+1-i))
  variables=(xx,yy,ww,vv,bb,qq,cc);env={str(t):z.Real(str(t)) for t in variables};args=' '.join(map(str,variables))
  for label,lhs,rhs in [('primal',native,compact)]+[('gradient_'+str(t),s.diff(native,t),s.diff(compact,t)) for t in variables]:
   check('spectral_degree_'+str(degree)+'_'+label,encode(lhs,env)!=encode(rhs,env))
   name='spectral_discovered_degree_'+str(degree)+'_'+label
   L.append(f'theorem {name} ({args} : ℝ) : {lean(lhs)} = {lean(rhs)} := by ring')
   R += [f'Theorem {name} ({args} : Qc) : {rocq(lhs)} = {rocq(rhs)}.','Proof. ring. Qed.'];generated.append(name)
  for t in variables:
   temp=s.Symbol('t');name='spectral_actual_degree_'+str(degree)+'_'+str(t)
   L += [f'theorem {name} ({args} : ℝ) : HasDerivAt (fun t : ℝ => {lean(native.subs(t,temp))}) ({lean(s.diff(native,t))}) {t} := by',
    f'  convert {derivative_proof(native,t)} using 1 <;> (try funext t) <;> (try dsimp) <;> ring']
 L.append('end OjasXMath')
 (ROOT/'lean/OjasXMath/SpectralDiscovered.lean').write_text('\n'.join(L)+'\n')
 (ROOT/'rocq/SpectralDiscovered.v').write_text('\n'.join(R)+'\n')
 # Change basis to Chebyshev polynomials, truncate the derivative at degree 7,
 # then integrate it. Forward and reverse remain one matched polynomial.
 # Symbolic truncation coefficients are checked universally by Z3 and Lean.
 C=['import OjasXMath.SpectralFoundation','import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.RootsExtrema','namespace OjasXMath']
 for order in (9,11):
  C += [f'theorem spectral_chebyshev_explicit_{order} (x : ℝ) : (Polynomial.Chebyshev.T ℝ {order}).eval x = {lean(s.chebyshevt(order,x))} := by',
   *['  have h'+str(j)+' := Polynomial.Chebyshev.T_add_two ℝ ('+str(j-2)+':ℤ)' for j in range(2,order+1)],
   '  norm_num at '+' '.join('h'+str(j) for j in range(2,order+1)),
   '  rw ['+', '.join('h'+str(j) for j in range(order,1,-1))+']',
   '  norm_num [Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_ofNat] <;> ring']
 C += ['theorem spectral_two_chebyshev_bound (x a b : ℝ) (hx : |x| ≤ 1) :',
  '  |a*(Polynomial.Chebyshev.T ℝ 9).eval x+b*(Polynomial.Chebyshev.T ℝ 11).eval x| ≤ |a|+|b| := by',
  '  have h9 := Polynomial.Chebyshev.abs_eval_T_real_le_one 9 hx',
  '  have h11 := Polynomial.Chebyshev.abs_eval_T_real_le_one 11 hx',
  '  calc',
  '    _ ≤ |a*(Polynomial.Chebyshev.T ℝ 9).eval x|+|b*(Polynomial.Chebyshev.T ℝ 11).eval x| := abs_add_le _ _',
  '    _ ≤ |a|+|b| := by',
  '      simp only [abs_mul]',
  '      have ha := mul_le_mul_of_nonneg_left h9 (abs_nonneg a)',
  '      have hb := mul_le_mul_of_nonneg_left h11 (abs_nonneg b)',
  '      nlinarith']
 coeff_file=['"""Generated matched degree-8 Chebyshev derivative truncations; reference-tail proofs are separate from activation remainder obligations."""','import math','COEFFICIENTS={}']
 cheby_report={};pref=s.Symbol('prefactor',real=True)
 for kind,f in funcs.items():
  ref=s.series(f,x,0,13).removeO().expand()
  if kind=='gelu':ref=ref.subs(s.sqrt(2)/s.sqrt(s.pi),2*pref)
  derivative=s.diff(ref,x).expand();remaining=derivative;cheb={}
  for order in range(11,-1,-1):
   value=s.expand(remaining).coeff(x,order)/s.expand(s.chebyshevt(order,x)).coeff(x,order)
   cheb[order]=s.simplify(value);remaining=s.expand(remaining-value*s.chebyshevt(order,x))
  assert remaining==0
  trunc=s.expand(sum(value*s.chebyshevt(order,x) for order,value in cheb.items() if order<=7))
  primal=s.integrate(trunc,x);primal=s.expand(primal-primal.subs(x,0))
  env={'x':z.Real('x'),'prefactor':z.Real('prefactor')}
  check('chebyshev_'+kind+'_tail_identity',encode(derivative-trunc,env)!=encode(cheb[9]*s.chebyshevt(9,x)+cheb[11]*s.chebyshevt(11,x),env))
  check('chebyshev_'+kind+'_matched_derivative',encode(s.diff(primal,x),env)!=encode(trunc,env))
  tail=s.simplify(sum(s.Abs(cheb[j]) for j in (9,11)))
  C += [f'theorem spectral_chebyshev_{kind}_tail (x prefactor : ℝ) (hx : |x| ≤ 1) :',
   f'  |{lean(derivative-trunc)}| ≤ {lean(tail).replace("Abs(prefactor)","|prefactor|")} := by',
   '  calc',f'    _ = |({lean(cheb[9])})*(Polynomial.Chebyshev.T ℝ 9).eval x+({lean(cheb[11])})*(Polynomial.Chebyshev.T ℝ 11).eval x| := by',
   '      rw [spectral_chebyshev_explicit_9, spectral_chebyshev_explicit_11]', '      congr 1', '      ring',
   '    _ ≤ _ := spectral_two_chebyshev_bound _ _ _ hx']
  # Last target uses |a|+|b|; discharge its rational constant normalization.
  C[-1]='    _ ≤ |('+lean(cheb[9])+')|+|('+lean(cheb[11])+')| := spectral_two_chebyshev_bound _ _ _ hx'
  C += ['    _ = _ := by simp only [abs_mul, abs_div, abs_neg]; norm_num <;> ring']
  for label,expr in [('primal',primal),('slope',trunc)]:
   name='spectral_chebyshev_'+kind+'_'+label+'_actual'
   if label=='primal':
    C += [f'theorem {name} (x prefactor : ℝ) : HasDerivAt (fun t : ℝ => {lean(primal.subs(x,s.Symbol("t")))}) ({lean(trunc)}) x := by',
     f'  convert {derivative_proof(primal,x)} using 1 <;> (try funext t) <;> (try dsimp) <;> ring']
  values=[s.simplify(primal.coeff(x,j)) for j in range(9)]
  coeff_file += [f'COEFFICIENTS[{kind!r}]=('+','.join(str(c).replace('prefactor','(1/math.sqrt(2*math.pi))') for c in values)+',)']
  cheby_report[kind]={'reference':'degree-12 Taylor polynomial','degree':8,'primal':str(primal),'slope':str(trunc),
   'derivative_reference_tail_bound':str(tail),'scope':'Kernel-checked uniform derivative difference versus polynomial reference on [-1,1]; true activation analytic remainder and FP32 rounding remain separate.'}
 C.append('end OjasXMath')
 (ROOT/'lean/OjasXMath/SpectralChebyshev.lean').write_text('\n'.join(C)+'\n')
 (ROOT/'ojasx2/spectral_coefficients.py').write_text('\n'.join(coeff_file)+'\n')
 # Remainder models are analytic obligations, not installed axioms or IEEE proofs.
 # GELU alternating exponential-integral tail: a conservative primal/slope envelope.
 tail=[]
 for degree in (2,4,6,8,10,12):
  ell=degree//2-1;next_ell=ell+1
  base=s.Rational(1,2**next_ell*math.factorial(next_ell)*(2*next_ell+1))
  tail.append({'degree':degree,'radius':'1','primal_upper_model':str(base),
   'slope_upper_model':str(base*(degree+2)),'gaussian_prefactor_upper_used':'1; loose bound for 1/sqrt(2pi)',
   'status':'Analytic alternating-tail model; convergence/alternating premises not completely formalized here'})
 report={'status':'symbolic_checked_pending_kernels_gpu','z3_checks':checks,'cegis':trace,
  'hermite_gelu_slope_coefficients':hermite,'hermite_norm_example':str(expectation),
  'taylor_primal_coefficients':coefficients,'chebyshev_refinement':cheby_report,'gelu_tail_models':tail,'shape_search':shapes,
  'generated_polynomial_identities':len(generated),'generated_actual_derivatives':21,
  'added_axioms':[],'scope':'Exact polynomial factor algebra plus conditional approximation/error contracts. Finite GELU/SiLU polynomials change the primal. No global degree-4 accuracy or Gaussian input assumption accepted.'}
 (ROOT/'reports/spectral-discovery.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS',len(checks),'SMT checks;',len(shapes),'rank/degree decisions; generated',len(generated),'identities and 21 calculus proofs',flush=True)

if __name__=='__main__':main()
