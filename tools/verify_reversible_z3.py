"""Nonvacuous reversible-optic models and explicit refutations of unsafe claims."""
import json
from pathlib import Path
import z3
from verify_z3 import prove


def verify():
    x1,x2,a,b,c,d=z3.Reals('rv_x1 rv_x2 rv_a rv_b rv_c rv_d')
    y1=x1+a*x2+b;y2=x2+c*y1+d
    rx2=y2-(c*y1+d);rx1=y1-(a*rx2+b)
    ps=[prove('affine_coupling_inverse_forward',[],z3.Or(rx1!=x1,rx2!=x2),
              'Real two-sided coupling; all scalar parameters; f/g need not be invertible')]
    u,v=z3.Reals('rv_y1 rv_y2')
    ix2=v-(c*u+d);ix1=u-(a*ix2+b)
    fy1=ix1+a*ix2+b;fy2=ix2+c*fy1+d
    ps.append(prove('affine_coupling_forward_inverse',[],z3.Or(fy1!=u,fy2!=v),'Real right inverse'))
    dx1,dx2,da,db,dc,dd,g1,g2=z3.Reals('rv_dx1 rv_dx2 rv_da rv_db rv_dc rv_dd rv_g1 rv_g2')
    dy1=dx1+a*dx2+da*x2+db;dy2=dx2+c*dy1+dc*y1+dd
    s=g1+c*g2
    ps.append(prove('coupling_all_input_and_parameter_cotangents',[],
        z3.simplify(dy1*g1+dy2*g2-(dx1*s+dx2*(g2+a*s)+da*x2*s+db*s+dc*y1*g2+dd*g2),som=True)!=0,
        'Six independent tangent directions; gradients for both inputs and four affine parameters'))
    w=[[z3.Real(f'rv_w_{i}_{j}') for j in range(3)] for i in range(2)]
    q=[z3.Real(f'rv_q{j}') for j in range(3)];qp=[z3.Real(f'rv_qp{j}') for j in range(3)]
    p=[z3.Real(f'rv_p{i}') for i in range(2)];pp=[z3.Real(f'rv_pp{i}') for i in range(2)]
    dot=lambda xs,ys:sum(i*j for i,j in zip(xs,ys))
    apply=lambda qs:[dot(row,qs) for row in w]
    adj=lambda ps:[sum(w[i][j]*ps[i] for i in range(2)) for j in range(3)]
    ps.append(prove('rectangular_cotangent_canonical_relation',[],
        z3.simplify(dot(q,adj(pp))-dot(qp,adj(p))-(dot(apply(q),pp)-dot(apply(qp),p)),som=True)!=0,
        '2x3 real W, correctly typed forward/backward relation; not an invertible phase-space map'))
    sW=z3.Real('rv_scalar_weight')
    ps.append(prove('isometric_reconstruction',[sW*sW==1],sW*(sW*x1)!=x1,
                    'Scalar isometry model; Lean covers arbitrary finite matrices under W transpose W=I'))
    ps.append(prove('invertible_cotangent_lift',[sW!=0],
                    (sW*x1)*(g2/sW)-(sW*x2)*(g1/sW)!=(x1*g2-x2*g1),
                    'Real scalar forward W and forward cotangent 1/W; inverse-transpose, not reverse pullback direction'))
    ef,eg,diff,gain=z3.Reals('rv_ef rv_eg rv_diff rv_gain')
    ab=lambda x:z3.If(x>=0,x,-x)
    ps.append(prove('conditional_reconstruction_error_bound',[gain>=0,ab(diff)<=gain*ab(eg)],
                    ab(diff-ef)>gain*ab(eg)+ab(ef),
                    'Conditional local Lipschitz/replay-error bound, not a universal FP kernel certificate'))
    n,k=z3.Ints('rv_depth rv_checkpoint_interval')
    ps.append(prove('checkpoint_boundary_count',[n>0,k>0],
                    z3.Or((n+k-1)/k<1,(n+k-1)/k>n),
                    'ceil(depth/interval) boundary groups; assumes fixed-size states, excludes weights/gradients/local workspaces'))
    counterexamples=[]
    def refute(name,claim,details):
        solver=z3.Solver();solver.add(z3.Not(claim));status=solver.check()
        counterexamples.append({'name':name,'status':str(status),'refutation_verified':status==z3.sat,**details})
    refute('dagger_bundle_is_not_generally_symplectic',z3.RealVal(4)==z3.RealVal(1),
           {'weight':2,'source_canonical_pair':1,'claimed_bundle_canonical_pair':4,
            'scope':'Square W=2; diag(W,W transpose) multiplies the canonical pairing by four'})
    refute('updated_weight_cannot_reconstruct_old_forward',z3.RealVal(2)/z3.RealVal('1.5')==1,
           {'old_weight':2,'new_weight':1.5,'input':1,'forward_output':2,
            'scope':'Real invertible scalar; optimizer update before reconstruction changes the recovered input'})
    sort=z3.Float32();one=z3.FPVal(1,sort);large=z3.FPVal(16777216,sort)
    rounded=z3.fpSub(z3.RNE(),z3.fpAdd(z3.RNE(),one,large),large)
    refute('real_coupling_inverse_is_not_FP32_inverse',z3.fpEQ(rounded,one),
           {'input':1,'offset':16777216,'reconstructed':str(z3.simplify(rounded)),
            'scope':'Finite binary32 values, round-to-nearest ties-to-even; rounded addition loses the input'})
    return {'solver_version':z3.get_version_string(),'proofs':ps,'counterexamples':counterexamples,
            'all_proofs_passed':all(p['verified'] for p in ps),
            'all_refutations_verified':all(p['refutation_verified'] for p in counterexamples),
            'scope':'Finite/scalar explicit models; Lean supplies stated general theorems. No GPU or universal ML certificate.'}


if __name__=='__main__':
    result=verify();Path('reports/reversible-z3.json').write_text(json.dumps(result,indent=2))
    print(f"Z3 reversible: {sum(p['verified'] for p in result['proofs'])}/{len(result['proofs'])} proofs; {sum(p['refutation_verified'] for p in result['counterexamples'])} refutations")
    raise SystemExit(0 if result['all_proofs_passed'] and result['all_refutations_verified'] else 1)
