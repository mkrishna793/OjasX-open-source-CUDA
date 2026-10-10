"""Finite candidate -> counterexample -> refinement loop, on host proof tools.
No CPU numerical ML, no invented axioms and no unbounded theorem search.
"""
from pathlib import Path
import hashlib,json
import sympy as s
import z3
ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def check(name, domain, identity, counterexample=False, repair=None):
        d=z3.Solver();d.set(timeout=5000);d.add(*domain)
        assert d.check()==z3.sat, name+' vacuous'
        q=z3.Solver();q.set(timeout=5000);q.add(*domain,z3.Not(identity))
        got=q.check();assert got==(z3.sat if counterexample else z3.unsat),(name,got)
        row=dict(name=name,domain='sat',negation=str(got))
        if counterexample:row.update(counterexample=str(q.model()),repair=repair)
        checks.append(row)
    for n in range(1,7):
        p=z3.Reals(' '.join('p'+str(i) for i in range(n)))
        a=z3.Reals(' '.join('a'+str(i) for i in range(n)))
        mass=sum(a);domain=[sum(p)==1,*[x>=0 for x in (*p,*a)]]
        check('weighted_cotangent_mass_'+str(n),domain,sum(mass*p[i]-a[i] for i in range(n))==0)
        if n>1:check('wrong_unweighted_cotangent_'+str(n),domain,
            z3.And(*[p[i]-a[i]==mass*p[i]-a[i] for i in range(n)]),True,
            'Use mass=sum(target_probability*class_weight), not 1')
    z=s.symbols('z0:3',real=True);a=s.symbols('a0:3',nonnegative=True)
    partition=sum(s.exp(x) for x in z)
    objective=sum(a)*s.log(partition)-sum(x*y for x,y in zip(a,z))
    analytic=[]
    for i in range(3):
        assert s.simplify(s.diff(objective,z[i])-(sum(a)*s.exp(z[i])/partition-a[i]))==0
        analytic.append('weighted_logsumexp_gradient_'+str(i))
    n,active,k,v,overhead=z3.Reals('rows active width vocabulary overhead')
    domain=[n>0,active>=0,active<=n,k>0,v>0]
    check('selected_leading_cost',domain,6*active*k*v<=6*n*k*v)
    check('replay_crossover',domain,(8*active*k*v<6*n*k*v)==(4*active<3*n))
    check('replay_double_crossover',domain,(2*8*active*k*v<=6*n*k*v)==(8*active<=3*n))
    check('replay_always_saves_arithmetic',domain,8*active*k*v<=6*n*k*v,True,
          'All-active replay increases GEMM work; admit it for memory, not arithmetic')
    check('whole_step_overhead_guard',domain+[overhead>=0,overhead<=6*(n-active)*k*v],
          6*active*k*v+overhead<=6*n*k*v)
    a0,a1,d0,d1=z3.Reals('loss0 loss1 denominator0 denominator1')
    check('local_chunk_mean_is_global_mean',[d0>0,d1>0],
          a0/d0+a1/d1==(a0+a1)/(d0+d1),True,'Use the full target-weight denominator for every chunk')
    x,y=z3.Reals('ignored_prefix_value active_token_value')
    check('ignored_attention_token_can_be_deleted',[],(x+y)/2==y,True,
          'Ignore only the output loss row. Keep prefix tokens and their upstream gradients')
    f=z3.FP('finite_input',z3.Float32());zero=z3.FPVal(0,z3.Float32())
    check('finite_inputs_suffice_for_zero_gradient',[z3.Not(z3.fpIsNaN(f)),z3.Not(z3.fpIsInf(f))],
          z3.Not(z3.fpIsNaN(z3.fpMul(z3.RNE(),zero,z3.fpMul(z3.RNE(),f,f)))),True,
          'Require finite-logit envelope; finite operands can still overflow')
    p,bits,budget=z3.Ints('parameters bits budget_bytes')
    check('200B_parameter_capacity',[p==200_000_000_000,bits==16,budget==32_000_000_000],
          p*bits>budget*8)
    shapes=[]
    for rows in (128,1024,8192):
        for width in (64,256,1024):
            for vocab in (1024,8192,65536):
                for divisor in (1,2,4,16):
                    active=rows//divisor;budget=32*1024*1024;chunk=min(128,budget//(6*vocab*4))
                    route='selected' if active*6*vocab*4<=budget else 'replay'
                    shapes.append(dict(rows=rows,width=width,classes=vocab,active=active,route=route,
                        chunk=chunk,native_flops=6*rows*width*vocab,
                        candidate_flops=(6 if route=='selected' else 8)*active*width*vocab))
    result=dict(status='passed_bounded_loss_discovery',checks=checks,analytic=analytic,
                shape_search=shapes,new_axioms=[],cpu_ml_execution=False,
                scope='Existing real-analysis and finite polynomial rules; Z3 FP counterexample, not a full numerical proof',
                source_sha256={'tools/discover_loss_rules.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (ROOT/'reports/loss-discovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS loss discovery:',len(checks),'checks,',len(analytic),'symbolic derivatives,',len(shapes),'shape plans')

if __name__=='__main__':main()
