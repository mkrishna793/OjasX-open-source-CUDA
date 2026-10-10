"""Cached reversible schedules, with solver-found shortcuts that fail."""
import json
from pathlib import Path
import z3
from verify_z3 import prove


def verify():
    x1,x2,f,g=z3.Reals('cache_x1 cache_x2 cache_f cache_g')
    y1=x1+f;y2=x2+g
    proofs=[prove('cached_branch_value_reconstruction',[],
        z3.Or(y1-f!=x1,y2-g!=x2),'Arbitrary exact-real branch values; no branch inversion required')]
    a,c,df,dg,scale,u,v,dx1,dx2,da,db,dc,dd=z3.Reals(
        'cache_a cache_c cache_df cache_dg cache_scale cache_u cache_v cache_dx1 cache_dx2 cache_da cache_db cache_dc cache_dd')
    dy1=dx1+scale*df*(a*dx2+da*x2+db)
    dy2=dx2+scale*dg*(c*dy1+dc*y1+dd)
    delta_g=scale*dg*v;adj=u+c*delta_g;delta_f=scale*df*adj
    right=dx1*adj+dx2*(v+a*delta_f)+da*x2*delta_f+db*delta_f+dc*y1*delta_g+dd*delta_g
    proofs.append(prove('cached_nonlinear_all_parameter_dagger',[],
        z3.simplify(dy1*u+dy2*v-right,som=True)!=0,'Six independent tangents; activation slopes supplied at cached preactivations'))
    z,s,incoming,state=z3.Reals('cache_z cache_s cache_incoming cache_state')
    proofs.append(prove('shared_silu_value_and_vjp',[],
        z3.Or(state-scale*(z*s)!=state-scale*z*s,
              scale*incoming*s*(1+z*(1-s))!=incoming*scale*(s+z*s*(1-s))),
        'Joint activation algebra; analytic sigmoid/SiLU derivatives separately proved in Lean'))
    e=z3.Real('cache_exp_positive');sig=1/(1+e)
    proofs.append(prove('sigmoid_derivative_rational_identity',[e>0],
        e/((1+e)*(1+e))!=sig*(1-sig),'Positive exponential value; rational identity, not an SMT transcendental derivative proof'))
    n,cache,budget,m,d=z3.Ints('cache_depth cache_count cache_budget cache_rows cache_width')
    proofs.append(prove('projection_replay_saving',[n>0,cache>=0,cache<=n],
        8*n-(8*n-2*cache)!=2*cache,'Joint GEMM counts when every input and weight gradient is needed'))
    proofs.append(prove('fully_cached_native_gemm_count',[n>0],8*n-2*n!=6*n,
        'Cache removes replay overhead, not the four dense backward GEMMs per block'))
    proofs.append(prove('cache_budget_monotone',[n>0,cache>=0,cache<=n,budget>=cache,budget<=n],
        8*n-2*budget>8*n-2*cache,'Uniform cache-size model; not globally optimal GPU timing'))
    proofs.append(prove('retained_cache_smaller_after_depth_one',[n>=2,m>0,d>0],
        (2+2*n)*m*d>=(4*n-1)*m*d,'Specific native residual accounting; excludes supplied inputs/parameters and transient scratch'))
    ef,ev,es=z3.Reals('cache_forward_error cache_value_error cache_subtract_error')
    ab=lambda t:z3.If(t>=0,t,-t)
    proofs.append(prove('cached_state_error_has_no_branch_Lipschitz_factor',[],
        ab(((x1+f+ef)-(f+ev)+es)-x1)>ab(ef)+ab(ev)+ab(es),
        'Conditional per-operation error decomposition; no IEEE magnitude/error certificate or gradient-convergence bound'))
    counterexamples=[]
    def refute(name,assumptions,claim,refinement):
        solver=z3.Solver();solver.set(timeout=15000);solver.add(*assumptions,z3.Not(claim))
        status=solver.check()
        counterexamples.append({'name':name,'status':str(status),'verified':status==z3.sat,
            'model':str(solver.model()) if status==z3.sat else None,'refinement':refinement})
    correct=v+a*scale*df*(u+c*scale*dg*v)
    independent=v+a*scale*df*u
    refute('independent_parallel_branch_pullbacks_lose_dependency',
        [a==1,c==1,df==1,dg==1,scale==1,u==0,v==1],correct==independent,
        'Apply G branch pullback first; add its cotangent to g1 before F. Cached values do not remove this dependence.')
    t=z3.Real('cache_collision_other')
    refute('activation_output_alone_need_not_determine_derivative',
        [z*z==t*t,z!=t],2*z==2*t,
        'For a generic nonlinear branch, cache a sufficient derivative residual such as its preactivation; square is an explicit witness.')
    refute('partial_cache_does_not_remove_all_replay',[n>0,cache>=0,cache<n],8*n-2*cache==6*n,
        'Use the cache-dependent count 8L-2C; full native-count equality requires C=L.')
    fp=z3.Float32();one=z3.FPVal(1,fp);large=z3.FPVal(16777216,fp)
    recovered=z3.fpSub(z3.RNE(),z3.fpAdd(z3.RNE(),one,large),large)
    refute('cached_branch_value_does_not_fix_FP32_cancellation',[],z3.fpEQ(recovered,one),
        'Exact-real inverse is not a bitwise FP32 inverse. Retain opt-in stability qualification and strict mode.')
    return {'solver_version':z3.get_version_string(),'proofs':proofs,'counterexamples':counterexamples,
        'all_proofs_passed':all(p['verified'] for p in proofs),
        'all_refutations_verified':all(c['verified'] for c in counterexamples)}


if __name__=='__main__':
    report=verify();Path('reports/cached-reverse-z3.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Cached reverse Z3: {sum(p['verified'] for p in report['proofs'])}/{len(report['proofs'])} proofs, {sum(c['verified'] for c in report['counterexamples'])} verified counterexamples")
    raise SystemExit(0 if report['all_proofs_passed'] and report['all_refutations_verified'] else 1)
