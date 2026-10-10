"""Bounded exact symbolic discovery of observable-gradient equivalence/defects.
Host symbolic algebra only. No numerical CPU ML or new foundational axioms.
"""
from pathlib import Path
import hashlib,json
import sympy as s
import z3
ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def check(name,domain,claim,refute=False,repair=''):
        sat=z3.Solver();sat.set(timeout=5000);sat.add(*domain);assert sat.check()==z3.sat
        q=z3.Solver();q.set(timeout=5000);q.add(*domain,z3.Not(claim));got=q.check()
        assert got==(z3.sat if refute else z3.unsat),(name,got)
        row=dict(name=name,domain='sat',negation=str(got))
        if refute:row.update(counterexample=str(q.model()),repair=repair)
        checks.append(row)
    a,b,c,d,p,q=z3.Reals('a b c d p q')
    check('equal_value_does_not_preserve_derivative',[a==b,c!=d],c*p==d*p,True,
          'Require a primal certificate and a slope certificate')
    check('dropping_a_nonzero_defect',[],(c+d)*p==c*p,True,'Keep every nonzero slope defect')
    check('one_observable_does_not_certify_both',[a==0,b!=0],b*p==0,True,
          'Certify input, weight and bias observations together')
    check('quotient_before_nonlinearity',[a+b==0,c!=d],c*a+d*b==0,True,
          'A coordinate-dependent derivative can move a nullspace component into observable directions')
    check('scalar_defect_pullback',[],(c+d)*p==c*p+d*p)
    check('scalar_defect_primal',[],a==c*b+q+(a-c*b-q))
    # Solver verifies a 2x2 quotient representative under left/right invariants.
    def mat(prefix,rows,cols):return [[z3.Real(f'{prefix}{i}_{j}') for j in range(cols)] for i in range(rows)]
    def mm(a,b):return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    d=mat('D',2,2);left=[[1,0],[0,0]];right=[[1,0],[0,0]]
    pd,dq,pdq=mm(left,d),mm(d,right),mm(mm(left,d),right)
    rep=[[pd[i][j]+dq[i][j]-pdq[i][j] for j in range(2)] for i in range(2)]
    check('left_observation_preserved',[],z3.And(*[mm(left,rep)[i][j]==mm(left,d)[i][j] for i in range(2) for j in range(2)]))
    check('right_observation_preserved',[],z3.And(*[mm(rep,right)[i][j]==mm(d,right)[i][j] for i in range(2) for j in range(2)]))
    check('representative_need_not_equal_dense',[],z3.And(*[rep[i][j]==d[i][j] for i in range(2) for j in range(2)]),True,
          'Hidden gradient equality is not required when every requested observation is preserved; retained-gradient hooks require native semantics')
    # Compute the joint observation map and exact nullspace, rather than assume
    # that a low-rank slope is necessary. Bias is an independent observation.
    m,n,k=6,8,2
    x=s.Matrix(m,k,lambda i,j:s.Integer(i+1)**(j+1))
    w=s.Matrix(k,n,lambda i,j:s.Integer(j+1)**i)
    variables=s.symbols('d:'+str(m*n));dm=s.Matrix(m,n,variables)
    observed=list(x.T*dm)+list(dm*w.T)+list(s.ones(1,m)*dm)
    op=s.Matrix(observed).jacobian(variables);null=op.nullspace()
    assert all(op*v==s.zeros(op.rows,1) for v in null)
    nullity=len(null);rank=op.rank();assert nullity+rank==m*n
    quotient=dict(shape=[m,k,n],dense_entries=m*n,observable_rank=rank,nullity=nullity,
        exact_nullspace_verified=True,includes_bias=True,
        caveat='This establishes representational redundancy, not a cheap algorithm to compute the observations')
    # Synthesize all original block gradient formulas from scalar pairings.
    xx,ww,bb,qq,pp,cc,oo,ef,ed=s.symbols('x w bias projection upstream c offset value_defect slope_defect')
    value=(xx*ww+bb)*cc+oo+ef;slope=cc+ed
    native=(slope*pp*qq*ww,slope*pp*qq*xx,slope*pp*qq,value*pp)
    compact=(pp*qq*cc*ww+ed*pp*qq*ww,xx*pp*qq*cc+xx*ed*pp*qq,
             pp*qq*cc+ed*pp*qq,ww*xx*pp*cc+(bb*cc+oo)*pp+ef*pp)
    derivatives=[]
    for name,a,b in zip(('input','weight','bias','projection'),native,compact):
        assert s.expand(a-b)==0;derivatives.append(name)
    batch,width,hidden,out,entries,guard=z3.Reals('batch width hidden out entries certificate_cost')
    domain=[batch>0,width>0,hidden>0,out>0,entries>=0,entries<=batch*hidden,guard>=0]
    common=2*batch*hidden*(width+out)
    dense=4*batch*hidden*(width+out)
    compact=4*batch*width*out+6*width*hidden*out+4*entries*(width+out)
    check('complete_cost_includes_certificate',domain+ [compact+guard<dense],common+compact+guard<common+dense)
    check('compact_representation_always_cheaper',domain,common+compact+guard<common+dense,True,
          'Reject expensive support, wide outputs, and costly coefficient construction')
    # An exact cancellation makes relative error alone ill-conditioned.
    error,aa,bb=z3.Reals('error cancelling_a cancelling_b')
    check('dropping_small_defects_preserves_relative_gradient',[aa+bb==0,error>0],z3.Abs(aa+bb+error)<=z3.RealVal('1/10000')*z3.Abs(aa+bb),True,
          'No tolerance-based pruning; retain numerical cancellation failures and native routes')
    shapes=[]
    for m in (128,1024,4096):
        for k in (16,128):
            for n in (512,2048):
                for r in (4,64,256):
                    for fraction in (0,.01,.1,1.):
                        e=int(m*n*fraction);native=6*m*n*(k+r)
                        candidate=2*m*n*(k+r)+4*m*k*r+6*k*n*r+4*e*(k+r)+8*m*n
                        shapes.append(dict(shape=[m,k,n,r],defect_fraction=fraction,native=native,candidate=candidate,
                                           arithmetic_eligible=candidate<native))
    result=dict(status='passed_bounded_quotient_discovery',checks=checks,quotient=quotient,
        synthesized_gradient_identities=derivatives,shape_search=shapes,added_axioms=[],cpu_ml_execution=False,
        source_sha256={'tools/discover_cotangent_quotient.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (ROOT/'reports/cotangent-quotient-discovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS quotient discovery:',len(checks),'checks;',rank,'observable dimensions of',quotient['dense_entries'],';',len(shapes),'plans')

if __name__=='__main__':main()
