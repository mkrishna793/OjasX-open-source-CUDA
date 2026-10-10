"""Bounded, reproducible theorem/candidate synthesis; not open-ended discovery.

SymPy derives polynomial coefficients, Z3 checks symbolic identities and finds
invalid shortcuts, and generated Lean ring proofs are subsequently kernel checked.
The shape search rejects candidates whose modeled total arithmetic or residuals
increase. No timing or universal optimality is inferred from operation counts.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import sympy as sp
import z3 as z
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ojasx2.feature_core import polynomial_cost

def encode(expr,env):
    if expr.is_Integer:return z.RealVal(int(expr))
    if expr.is_Rational:return z.RealVal(str(Fraction(int(expr.p),int(expr.q))))
    if expr.is_Symbol:return env[str(expr)]
    if expr.is_Add:return sum(encode(t,env) for t in expr.args)
    if expr.is_Mul:
        result=z.RealVal(1)
        for term in expr.args:result=result*encode(term,env)
        return result
    if expr.is_Pow and expr.exp.is_Integer and expr.exp>=0:
        result=z.RealVal(1)
        for _ in range(int(expr.exp)):result=result*encode(expr.base,env)
        return result
    raise ValueError('Outside the supported polynomial grammar: '+str(expr))

def lean(expr):return str(expr).replace('**','^')

def derivative_proof(expr,variable):
    """Synthesize an actual HasDerivAt proof from Mathlib's calculus rules."""
    point=str(variable)
    if not expr.has(variable):return f'(hasDerivAt_const {point} ({lean(expr)}))'
    if expr==variable:return f'(hasDerivAt_id {point})'
    if expr.is_Add or expr.is_Mul:
        operation='add' if expr.is_Add else 'mul'
        children=[derivative_proof(term,variable) for term in expr.args]
        proof=children[0]
        for child in children[1:]:proof=f'({proof}.{operation} {child})'
        return proof
    if expr.is_Pow and expr.exp.is_Integer and expr.exp>=0:
        return f'({derivative_proof(expr.base,variable)}.pow {int(expr.exp)})'
    raise ValueError('Unsupported derivative proof grammar')

def refine_feature_basis():
    """Counterexample-guided basis enrichment in a finite univariate grammar."""
    trace=[]
    for target_degree in (2,3):
        samples=(-2,-1,0,1,2)
        for degree in range(target_degree+1):
            coeffs=[z.Real(f'basis_{target_degree}_{degree}_{i}') for i in range(degree+1)]
            solver=z.Solver();solver.set(timeout=10000)
            solver.add(*[sum(c*point**i for i,c in enumerate(coeffs))==point**target_degree for point in samples])
            result=solver.check()
            item={'target_degree':target_degree,'basis_degrees':list(range(degree+1)),
                'sample_constraints':[[point,point**target_degree] for point in samples],'result':str(result)}
            if result==z.unsat:
                item['decision']='enrich_basis';trace.append(item);continue
            if result!=z.sat:raise AssertionError('Unknown basis synthesis result')
            model=solver.model();values=[model.eval(c,model_completion=True) for c in coeffs]
            t=z.Real('basis_verification_input');verify=z.Solver();verify.set(timeout=10000)
            verify.add(sum(c*t**i for i,c in enumerate(values))!=t**target_degree)
            if verify.check()!=z.unsat:raise AssertionError('Sample fit did not generalize')
            item.update(decision='accept_after_symbolic_refutation',coefficients=list(map(str,values)),global_negation='unsat')
            trace.append(item);break
    return trace

def main():
    path=ROOT/'reports/backprop-discovery.json'
    path.write_text(json.dumps({'status':'running'})+'\n')
    variables=sp.symbols('x0 x1 a00 a01 a10 a11 b0 b1')
    x0,x1,a00,a01,a10,a11,b0,b1=variables
    env={str(v):z.Real(str(v)) for v in variables}
    proofs=[];checks=[];coefficients={}
    args=' '.join(map(str,variables))
    for degree in (2,3):
        native=(x0*a00+x1*a10)**degree*b0+(x0*a01+x1*a11)**degree*b1
        poly=sp.Poly(native,x0,x1)
        candidate=sum(coefficient*x0**powers[0]*x1**powers[1] for powers,coefficient in poly.terms())
        coefficients[str(degree)]=[{ 'powers':list(powers),'coefficient':str(coefficient)} for powers,coefficient in poly.terms()]
        forms=[('primal',native,candidate)]+[(f'gradient_{v}',sp.diff(native,v),sp.diff(candidate,v)) for v in variables]
        for label,left,right in forms:
            solver=z.Solver();solver.set(timeout=10000)
            solver.add(encode(left,env)!=encode(right,env));result=solver.check()
            if result!=z.unsat:raise AssertionError((degree,label,str(result)))
            name=f'discovered_degree_{degree}_{label}'
            proofs.append(f'theorem {name} ({args} : ℝ) :\n    {lean(left)} = {lean(right)} := by ring\n')
            checks.append({'name':name,'result':str(result),'scope':'Two-input/two-hidden scalar-output polynomial instance'})
        for variable in variables:
            t=sp.Symbol('t');native_t=native.subs(variable,t);candidate_t=candidate.subs(variable,t)
            grad=sp.diff(native,variable)
            native_name=f'discovered_degree_{degree}_native_derivative_{variable}'
            lifted_name=f'discovered_degree_{degree}_lifted_derivative_{variable}'
            proof=derivative_proof(native,variable)
            proofs.append(f'theorem {native_name} ({args} : ℝ) :\n'
                f'    HasDerivAt (fun t : ℝ => {lean(native_t)}) ({lean(grad)}) {variable} := by\n'
                f'  convert {proof} using 1 <;> (try funext t) <;> dsimp <;> ring\n')
            proofs.append(f'theorem {lifted_name} ({args} : ℝ) :\n'
                f'    HasDerivAt (fun t : ℝ => {lean(candidate_t)}) ({lean(grad)}) {variable} := by\n'
                f'  have h := {native_name} {args}\n'
                f'  have same : (fun t : ℝ => {lean(native_t)}) = (fun t : ℝ => {lean(candidate_t)}) := by\n'
                f'    funext t\n    ring\n'
                f'  rw [same] at h\n  exact h\n')
    # Generated formulas are admitted only after Lean compilation by the verifier.
    source='import Mathlib.Basic.Real.Basic\nimport Mathlib.Analysis.Calculus.Deriv.Add\nimport Mathlib.Analysis.Calculus.Deriv.Mul\nimport Mathlib.Analysis.Calculus.Deriv.Pow\nimport Mathlib.Tactic.Ring\n\nset_option autoImplicit false\n\nnamespace OjasXMath\n\n'+'\n'.join(proofs)+'\nend OjasXMath\n'
    target=ROOT/'lean/OjasXMath/Discovered.lean';target.write_text(source)
    x,a,b=z.Reals('counterexample_x counterexample_a counterexample_b')
    for name,constraints in [
        ('reject_square_linear_contraction',[x==2,a==3,b==1,(x*a)*(x*a)*b!=x*(a*a*b)]),
        ('reject_output_only_square_derivative',[x==1,a==-1,x*x==a*a,2*x!=2*a]),
        ('reject_zero_guarded_division_derivative',[x==0,a==1,z.If(x==0,0,x*a/x)!=a])]:
        solver=z.Solver();solver.add(*constraints)
        if solver.check()!=z.sat:raise AssertionError(name)
        checks.append({'name':name,'result':'sat','witness':str(solver.model())})
    fp=z.Float32();rnd=z.RNE()
    big=z.FPVal('1e20',fp);small=z.FPVal('1e-20',fp)
    solver=z.Solver();solver.add(z.Not(z.fpEQ(z.fpMul(rnd,z.fpMul(rnd,big,big),small),
        z.fpMul(rnd,big,z.fpMul(rnd,big,small)))))
    assert solver.check()==z.sat
    checks.append({'name':'reject_algebra_as_bitwise_certificate','result':'sat','scope':'Finite inputs with intermediate FP32 overflow'})
    u=sp.Symbol('activation_input')
    analytic_rejections=[]
    for name,expression in [('silu',u/(1+sp.exp(-u))),('gelu',u*(1+sp.erf(u/sp.sqrt(2)))/2),
            ('tanh',sp.tanh(u))]:
        try:sp.Poly(expression,u)
        except sp.PolynomialError:
            analytic_rejections.append({'primitive':name,'decision':'reject_exact_finite_polynomial_lift',
                'scope':'Not in this symbolic polynomial grammar; approximation would need a separate contract'})
        else:raise AssertionError('Unexpected polynomial recognition')
    candidates=[]
    for degree in (2,3):
        for m in (1,32,512):
            for k in (2,4,16,64):
                for n in (16,256,2048):
                    for p in (2,16,64):
                        try:cost=polynomial_cost(m,k,n,p,degree)
                        except ValueError as error:
                            candidates.append({'shape':[m,k,n,p],'degree':degree,'eligible':False,'decision':'reject_reference_budget','reason':str(error)})
                            continue
                        cost['decision']='candidate_requires_numerical_qualification' if cost['eligible'] else 'reject_cost_or_residual'
                        candidates.append(cost)
    accepted=[c for c in candidates if c['eligible']]
    report={'status':'symbolic_checks_passed_pending_lean_kernel','grammar':'Homogeneous degree-2/3 polynomial activation blocks; finite positive shape grid',
        'derived_coefficients':coefficients,'basis_refinement_trace':refine_feature_basis(),
        'analytic_primitive_rejections':analytic_rejections,'checks':checks,'generated_theorems':len(proofs),
        'generated_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'candidates_examined':len(candidates),'cost_eligible':len(accepted),
        'best_modeled_candidate':max(accepted,key=lambda c:c['arithmetic_ratio']),
        'candidates':candidates,'not_established':['GELU/SiLU finite polynomial equivalence','global minimum cost','measured runtime gain','floating error certificate','automatic backend dispatch']}
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','generated_theorems','candidates_examined','cost_eligible','best_modeled_candidate')},indent=2))

if __name__=='__main__':main()
