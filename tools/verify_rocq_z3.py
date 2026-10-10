"""Scoped counterexamples and guards accompanying the Rocq development."""
import json
from pathlib import Path
import z3 as z

ROOT=Path(__file__).resolve().parents[1]

def main():
    checks=[]
    def check(name, constraints, expected, scope):
        s=z.Solver();s.set(timeout=10000);s.add(*constraints)
        result=s.check()
        if str(result)!=expected: raise AssertionError((name,str(result),s.reason_unknown()))
        entry={'name':name,'result':str(result),'expected':expected,'scope':scope}
        if result==z.sat: entry['witness']=str(s.model())
        checks.append(entry)
    x,a,b,g,eta=z.Reals('x a b g eta')
    check('scalar_linear_forward_and_all_gradients',[
        z.Or((x*a)*b!=x*(a*b),(g*b)*a!=g*(a*b),x*(g*b)!=(x*g)*b,(x*a)*g!=a*(x*g))],
        'unsat','Scalar polynomial identities; the finite-matrix theorem is checked by Rocq.')
    check('scalar_gradient_update_preserved',[a-eta*(x*(g*b))!=a-eta*((x*g)*b)],
        'unsat','Equality of two exact scalar parameter updates, not convergence.')
    m,k,n,p=z.Ints('m k n p')
    native=6*m*k*n+6*m*n*p;contracted=6*p*k*(m+n)
    choose=z.And(contracted<=native,k*p<=m*n)
    check('work_and_tape_guard',[m>=0,k>=0,n>=0,p>=0,
        z.Or(z.If(choose,contracted,native)>native,z.If(choose,k*p,m*n)>m*n)],
        'unsat','Nonnegative shape cost model, no hardware runtime guarantee.')
    relu=lambda v:z.If(v>0,v,z.RealVal(0))
    check('reject_unconditional_relu_contraction',[x==-1,a==1,b==1,relu(x*a)*b!=x*relu(a)*b],
        'sat','Different activation masks invalidate unconditional reassociation; require the shared-mask theorem.')
    y=z.Real('y')
    check('reject_square_output_only_tape',[x==1,y==-1,x*x==y*y,2*x!=2*y],
        'sat','Squaring erases sign needed for its pullback; Rocq also proves recovery impossible.')
    fp=z.Float32();rnd=z.RNE()
    fa=z.FPVal('1e20',fp);fb=z.FPVal('1e20',fp);fc=z.FPVal('1e-20',fp)
    lhs=z.fpMul(rnd,z.fpMul(rnd,fa,fb),fc);rhs=z.fpMul(rnd,fa,z.fpMul(rnd,fb,fc))
    check('reject_float_reassociation_without_error_contract',[z.Not(z.fpEQ(lhs,rhs))],
        'sat','Finite FP32 inputs, with intermediate overflow on one path; exact rational proof does not imply bit equality.')
    mult,add=z.Ints('block_multiplication_cost block_addition_cost')
    check('rank7_total_work_threshold',[mult>=0,add>=0,
        z.Xor(21*mult+54*add<=24*mult+12*add,mult>=14*add)],
        'unsat','Uniform block-cost model including additions for forward and both gradients.')
    check('winograd_total_work_threshold',[mult>=0,add>=0,
        z.Xor(21*mult+45*add<=24*mult+12*add,mult>=11*add)],
        'unsat','Winograd saves nine additions relative to three ordinary Strassen products, with the same multiplication count.')
    check('reject_multiplication_count_only_speed_claim',[mult==1,add==1,
        21*mult+54*add>24*mult+12*add],
        'sat','Seven-product scalar schedule saves multiplications while increasing total arithmetic.')
    r=z.Real('inverse_rms')
    check('reject_rms_radial_null_with_positive_epsilon',[r>0,r*r*(9+1)==1,
        (r-r*r*r*9)*3!=0],
        'sat','With epsilon=1 and x=3, the radial derivative is nonzero; retain epsilon in derivative coefficients.')
    zn=z.Real('normalized_value')
    check('normalized_rms_coefficient_rewrite',[zn==r*x,
        -(r*r*r*a*x)*x!=-(r*a*zn)*zn],
        'unsat','Exact coefficient identity; finite floating execution is tested separately.')
    g0,g1=z.Reals('gather_cotangent_0 gather_cotangent_1')
    check('reject_scatter_overwrite_for_repeated_indices',[g0==1,g1==2,g0+g1!=g1],
        'sat','Two gathered outputs referring to the same input require summation of both cotangents.')
    check('affine_primal_and_all_gradients',[g*(a*x+b*eta+y)!=
        x*(a*g)+b*(eta*g)+y*g],
        'unsat','One-coordinate affine differential pairing; finite shared-parameter sums are proved in Rocq.')
    tiny_r=z.FPVal('1e15',fp);zero=z.FPVal(0,fp)
    cubic=z.fpMul(rnd,z.fpMul(rnd,tiny_r,tiny_r),tiny_r)
    check('reject_cubic_scale_evaluation_at_zero',[z.fpIsNaN(z.fpMul(rnd,cubic,zero))],
        'sat','FP32 inverse scale is finite but its cube overflows; multiplying infinity by zero produces NaN.')
    report={'status':'passed','z3_version':z.get_version_string(),'checks':checks,
        'proved_encoded_negations':sum(c['result']=='unsat' for c in checks),
        'expected_counterexamples':sum(c['result']=='sat' for c in checks)}
    (ROOT/'reports/rocq-z3.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(checks)} Z3 checks, including {report["expected_counterexamples"]} expected counterexamples.')

if __name__=='__main__': main()
