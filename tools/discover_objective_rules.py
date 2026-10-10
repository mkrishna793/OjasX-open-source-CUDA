"""Bounded exact-arithmetic rule checks and falsification; no numerical CPU ML."""
from pathlib import Path
import json,hashlib
import z3
import sympy as s
ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def prove(name,domain,claim):
        sat=z3.Solver();sat.set(timeout=10000);sat.add(*domain)
        assert sat.check()==z3.sat,name+" vacuous domain"
        test=z3.Solver();test.set(timeout=10000);test.add(*domain,z3.Not(claim))
        status=test.check();assert status==z3.unsat,(name,status)
        checks.append(dict(name=name,domain="sat",negation="unsat"))
    def refute(name,domain,claim,repair):
        test=z3.Solver();test.set(timeout=10000);test.add(*domain,z3.Not(claim))
        status=test.check();assert status==z3.sat,(name,status)
        checks.append(dict(name=name,counterexample=str(test.model()),repair=repair))
    h=s.Matrix(2,2,s.symbols("h0:4"));w=s.Matrix(2,3,s.symbols("w0:6"));y=s.Matrix(2,3,s.symbols("y0:6"))
    residual=h*w-y
    native=sum(e*e for e in residual)
    g=h.T*h;c=h.T*y;metric=w*w.T
    gram=sum(g[i,j]*metric[i,j] for i in range(2) for j in range(2))-2*sum(c[i,j]*w[i,j] for i in range(2) for j in range(3))+sum(e*e for e in y)
    variables={str(t):z3.Real(str(t)) for t in (*h,*w,*y)}
    def zz(e):
        e=s.expand(e)
        if e.is_Integer:return z3.RealVal(int(e))
        if e.is_Symbol:return variables[str(e)]
        if e.is_Add:return sum(zz(a) for a in e.args)
        if e.is_Mul:
            result=z3.RealVal(1)
            for a in e.args:result*=zz(a)
            return result
        if e.is_Pow:return zz(e.base)**int(e.exp)
        raise ValueError(e)
    prove("quadratic_gram_primal_2x2x3",[],zz(native)==zz(gram))
    dh=2*(h*metric-y*w.T);dw=2*(g*w-c)
    for i,v in enumerate(h):prove("all_input_gradient_"+str(i),[],zz(s.diff(native,v))==zz(dh[i]))
    for i,v in enumerate(w):prove("all_weight_gradient_"+str(i),[],zz(s.diff(native,v))==zz(dw[i]))
    # Keep a plausible but incorrect target-free reconstruction in the ledger.
    refute("drop_target_input_term",[],zz(s.diff(native,h[0]))==zz((2*h*metric)[0]),
           "Keep -2*Y*W^T; all original input gradients are required")
    a,b,d,weight,x,theta,eta=z3.Reals("a b denominator weight x theta eta")
    prove("fixed_global_normalizer",[d>0],a/d+b/d==(a+b)/d)
    prove("zero_weight_value_and_gradient",[weight==0],z3.And(weight*a==0,weight*b==0))
    refute("unweighted_mean_of_unequal_chunk_means",[],(a/2+b/3)/2==(a+b)/5,
           "Normalize each chunk's sum by the same full-objective denominator")
    refute("update_after_each_chunk",[eta>0,eta<1],
           theta-eta*2*theta-eta*2*(theta-eta*2*theta)==theta-eta*(2*theta+2*theta),
           "Accumulate at a fixed parameter state; apply optimizer and clipping once")
    refute("batch_centering_is_separable",[a!=b],((a-(a+b)/2)**2+(b-(a+b)/2)**2)==0,
           "Batch-dependent centering/normalization is not chunk-separable")
    refute("differentiate_weights_as_constants",[x!=0],2*x*x==3*x*x,
           "Require fixed weights; learned weights need the product-rule term")
    m,k,n=z3.Reals("m k n")
    native_cost=6*m*k*n;candidate_cost=4*m*k*n+4*m*k*k+4*k*k*n
    prove("gram_cost_exact_crossover",[m>0,k>0,n>0],
          (candidate_cost<native_cost)==(2*k*(m+n)<m*n))
    refute("gram_always_cheaper",[m>0,k>0,n>0],candidate_cost<native_cost,
           "Shape/demand planner rejects expensive Gram schedules")
    rounding,tolerance=z3.Reals('rounding tolerance')
    refute('energy_guard_does_not_protect_stationary_gradients',[a==b,rounding>0,tolerance>0],
           z3.Abs((a+rounding)-b)<=tolerance*z3.Abs(a-b),
           'Check each demanded gradient contraction for cancellation; retain native gradient evaluation when rejected')
    fp=z3.FP('finite_operand',z3.Float32());zero=z3.FPVal(0,z3.Float32())
    refute('finite_inputs_do_not_certify_finite_losses',[z3.Not(z3.fpIsNaN(fp)),z3.Not(z3.fpIsInf(fp))],
           z3.Not(z3.fpIsNaN(z3.fpMul(z3.RNE(),zero,z3.fpMul(z3.RNE(),fp,fp)))),
           'Zero-weight elimination additionally requires finite per-example losses and derivatives; finite inputs alone are insufficient')
    active,batch,work,overhead=z3.Reals("active batch work overhead")
    prove("zero_weight_elision_pays_overhead",[0<=active,active<=batch,work>=0,
           overhead>=0,overhead<=(batch-active)*work],active*work+overhead<=batch*work)
    p,cost,chunk=z3.Reals("persistent residual_per_example chunk")
    prove("streamed_residual_bound",[p>=0,cost>=0,chunk>=0,batch>=chunk],p+chunk*cost<=p+batch*cost)
    shapes=[]
    for mm in (32,256,2048):
        for kk in (8,32,256):
            for nn in (32,512,4096):
                nc=6*mm*kk*nn;gc=4*mm*kk*nn+4*mm*kk*kk+4*kk*kk*nn
                shapes.append(dict(m=mm,k=kk,n=nn,selected="gram" if gc<nc else "native",
                                   native_matrix_flops=nc,candidate_matrix_flops=gc))
    result=dict(status="passed_bounded_symbolic_checks",checks=checks,shape_search=shapes,
                new_axioms=[],cpu_ml_execution=False,
                scope="SymPy differentiation and Z3 real-polynomial falsification on a finite tensor instance; dimension-general Lean/Rocq proofs are audited separately.",
                source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (ROOT/"reports/objective-discovery.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PASS",len(checks),"symbolic checks/counterexamples;",len(shapes),"shape decisions")

if __name__=="__main__":main()
