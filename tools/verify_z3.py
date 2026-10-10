"""Small explicit machine models, each with a nonvacuity check."""
import argparse, json
from pathlib import Path
import z3


def prove(name, assumptions, negation, scope):
    solver=z3.Solver()
    solver.set(timeout=15000)
    solver.add(*assumptions)
    pre=solver.check()
    solver.add(negation)
    status=solver.check()
    return {"name":name,"preconditions":str(pre),"counterexample_query":str(status),
            "verified":pre==z3.sat and status==z3.unsat,"scope":scope,
            "counterexample":str(solver.model()) if status==z3.sat else None}


def verify():
    results=[]
    byte=z3.BitVec('mask_byte',8)
    bits=[z3.Extract(i,i,byte) for i in range(8)]
    rebuilt=z3.BitVecVal(0,8)
    for i,bit in enumerate(bits):
        rebuilt=rebuilt | (z3.ZeroExt(7,bit)<<i)
    results.append(prove("mask_byte_roundtrip",[],rebuilt!=byte,"All 256 byte masks, exact bit representation"))
    i=z3.BitVec('index',32)
    results.append(prove("packed_bit_address",[],((z3.LShR(i,3)<<3)|(i&7))!=i,"All 32-bit indices; address decomposition only"))
    r,c=z3.BitVecs('row col',32)
    results.append(prove("xor_column_inverse",[z3.ULT(c,32)],(c^(r&31)^(r&31))!=c,"32-column XOR layout, known row"))
    t1,t2=z3.BitVecs('lane1 lane2',32)
    results.append(prove("padded_column_banks",[z3.ULT(t1,32),z3.ULT(t2,32),t1!=t2],
                         ((t1*33+c)&31)==((t2*33+c)&31),"32-bank model, one fixed-column load per distinct lane; not all GPU instructions"))
    x,y=z3.BitVecs('packed_coordinate_row packed_coordinate_col',32)
    z,w=z3.BitVecs('other_row other_col',32)
    results.append(prove("bounded_flat_layout_injective",[z3.ULT(x,32),z3.ULT(y,32),z3.ULT(z,32),z3.ULT(w,32),z3.Or(x!=z,y!=w)],
                         x*32+(y^x)==z*32+(w^z),"32x32 tile, bounded arithmetic without address overflow"))
    n=z3.Int('num_elements')
    results.append(prove("packed_residual_capacity",[n>=0],z3.Or(((n+7)/8)*8<n, ((n+7)/8)*8>=n+8),
                         "Integer storage arithmetic: ceil(n/8) bytes hold n Boolean flags"))
    sort=z3.Float32()
    a,dy=z3.FPs('relu_input upstream',sort)
    zero=z3.FPVal(0,sort)
    gate=z3.Not(z3.fpLEQ(a,zero))
    ref=z3.If(z3.fpLEQ(a,zero),zero,dy)
    packed=z3.If(gate,dy,zero)
    results.append(prove("relu_pullback_selected_bits",[z3.Not(z3.fpIsNaN(dy))],
                         z3.fpToIEEEBV(ref)!=z3.fpToIEEEBV(packed),"Threshold convention, including NaN inputs and signed-zero upstream; excludes NaN upstream payloads"))
    g11,g12,g21,g22=z3.Reals('g11 g12 g21 g22')
    v1,v2=g21+g22,g12+g22
    v3=g12+v1
    v4=g11+v3
    actual=[v4,g11,g12,-g21,v2,v3,v1]
    expected=[g11+g12+g21+g22,g11,g12,-g21,g12+g22,g12+g21+g22,g21+g22]
    results.append(prove("decoder_shared_cotangents",[],z3.Or(*[a!=b for a,b in zip(actual,expected)]),"Real additive decoder algebra, not floating addition reassociation"))
    # Scalar model of the exact output-slot schedule in streamed.py. All seven
    # contributions are symbolic and independent; no numerical example is used.
    aa=z3.Reals('a_channel1 a_channel2 a_channel3 a_channel4 a_channel5 a_channel6 a_channel7')
    bb=z3.Reals('b_channel1 b_channel2 b_channel3 b_channel4 b_channel5 b_channel6 b_channel7')
    a11,a12,a21,a22=0,0,aa[5],0
    a21=a21-aa[2];a12=aa[2];a11=-a21
    a22=aa[3]
    a21=a21+aa[4];a22=a22+a21
    a11=a11+aa[0]
    a11=a11+aa[6];a21=a21-aa[6]
    a12=a12+aa[1]
    au=(aa[0]+aa[2]-aa[5]+aa[6],aa[1]+aa[2],-aa[2]+aa[4]+aa[5]-aa[6],-aa[2]+aa[3]+aa[4]+aa[5])
    results.append(prove('streamed_a_output_slot_schedule',[],z3.Or(*[x!=y for x,y in zip((a11,a12,a21,a22),au)]),
                         'Seven independent real gradient channels; schedule 6,3,4,5,1,7,2; no floating or alias safety claim'))
    b22=bb[5];b11=bb[2]
    b22=b22+bb[3];b21=-bb[3]
    b12=bb[4]-b22;b22=b22+b11;b11=-b12
    b11=b11+bb[0]
    b12=b12-bb[6];b22=b22+bb[6]
    b21=b21+bb[1]
    bv=(bb[0]-bb[4]+bb[5]+bb[3],bb[4]-bb[5]-bb[3]-bb[6],bb[1]-bb[3],bb[2]+bb[5]+bb[3]+bb[6])
    results.append(prove('streamed_b_output_slot_schedule',[],z3.Or(*[x!=y for x,y in zip((b11,b12,b21,b22),bv)]),
                         'Seven independent real gradient channels, temporary B11 consumed before overwrite'))
    size,tile,idx=z3.Ints('dimension tile_extent logical_index')
    q,rem=idx/tile,idx%tile
    results.append(prove('tile_partition_covers_each_index', [size>0,tile>0,idx>=0,idx<size],
                         z3.Or(q*tile+rem!=idx,rem<0,rem>=tile,q>=(size+tile-1)/tile),
                         'Unbounded integer tile partition; dynamic tails; separate machine overflow guards still required'))
    count,active,extent=z3.Ints('channel_count active_channel tile_elements')
    results.append(prove('single_channel_scratch_bound',[count>=1,active>=0,active<count,extent>=0],extent>count*extent,
                         'One live channel buffer versus all channel buffers; conditional storage model, not allocator measurement'))
    # Additional math-only schedules. Symbolic real identities for fixed
    # arities; Lean supplies the arbitrary-finite-arity analytic theorem.
    from math_structure import joint_product_pullback, prefix_suffix_pullback
    for arity in range(2,9):
        factors=z3.Reals(' '.join('factor_%d_%d'%(arity,j) for j in range(arity)))
        upstream=z3.Real('product_upstream_%d'%arity)
        shared=joint_product_pullback(factors,upstream)
        prefix_suffix=prefix_suffix_pullback(factors,upstream)
        independent=[]
        for j in range(arity):
            term=upstream
            for k,e in enumerate(factors):
                if k!=j:term=term*e
            independent.append(term)
        # Polynomial normalization is performed by Z3, not a Python CAS.
        negation=z3.Or(*[z3.simplify(a-b,som=True)!=0 for a,b in zip(shared+prefix_suffix,tuple(independent)*2)])
        results.append(prove('joint_product_schedules_arity_%d'%arity,[],negation,
                             'Rolling-cotangent and prefix/suffix schedules; all real factors and upstream; fixed arity; exact polynomial normalization; zeros allowed'))
    zero_factors=z3.Reals('zero_factor_0 zero_factor_1 zero_factor_2 zero_factor_3')
    zeroup=z3.Real('zero_product_upstream')
    at_least_two=z3.Or(*[z3.And(zero_factors[i]==0,zero_factors[j]==0)
                          for i in range(4) for j in range(i+1,4)])
    results.append(prove('two_zero_factors_all_product_gradients_zero',[at_least_two],
                         z3.Or(*[h!=0 for h in joint_product_pullback(zero_factors,zeroup)]),
                         'Four real factors, arbitrary positions of two or more zeros; no division'))
    arity=z3.Int('product_arity')
    naive=arity*(arity-1);shared=4*arity-6
    results.append(prove('joint_prefix_suffix_cost_saving',[arity>=2],
                         z3.Or(naive-shared!=(arity-2)*(arity-3),shared>naive),
                         'Integer arity >=2; scalar multiplication count only, encoders/decoder and memory excluded'))
    sx=z3.Reals('sphere_x0 sphere_x1 sphere_x2')
    sg=z3.Reals('sphere_g0 sphere_g1 sphere_g2')
    dot=sum(x*g for x,g in zip(sx,sg))
    projected=[g-x*dot for x,g in zip(sx,sg)]
    results.append(prove('sphere_projection_tangent', [sum(x*x for x in sx)==1],
                         sum(x*p for x,p in zip(sx,projected))!=0,
                         'Three-dimensional real unit sphere, Euclidean tangent projection; no rounded update claim'))
    ep,eq,lip,local,total=z3.Reals('error_p error_q lipschitz local_error total_error')
    results.append(prove('backward_error_amplification',[ep>=0,eq>=0,lip>=0,
                         local>=0,local<=lip*eq,total<=ep+local],total>ep+lip*eq,
                         'Assumed local approximation errors and nonnegative Lipschitz bound; no instruction-level float certificate'))
    results.append(prove('rolling_cotangent_cost_saving',[arity>=2],
                         z3.Or(naive-(3*arity-4)!=(arity-2)*(arity-2),
                               3*arity-4>naive,
                               (3*arity-4)-(2*arity-2)!=arity-2),
                         'Rebuilt prefixes; scalar products; saving versus independent evaluation and overhead versus retained-prefix reverse mode'))
    # Fixed 2x2 matrix schedules, all entries arbitrary symbolic reals.
    def mm(a,b):
        return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    def tr(a):return [list(row) for row in zip(*a)]
    matrices=[]
    for label in ['chain_a','chain_b','chain_c','chain_g']:
        values=z3.Reals(' '.join(label+str(i) for i in range(4)))
        matrices.append([list(values[:2]),list(values[2:])])
    ca,cb,cc,cg=matrices
    effective=mm(cb,cc);cot=mm(tr(ca),cg)
    ref=[mm(mm(ca,cb),cc),mm(mm(cg,tr(cc)),tr(cb)),mm(tr(ca),mm(cg,tr(cc))),mm(tr(mm(ca,cb)),cg)]
    got=[mm(ca,effective),mm(cg,tr(effective)),mm(cot,tr(cc)),mm(tr(cb),cot)]
    polynomial_differences=[z3.simplify(x-y,som=True)!=0 for rx,ry in zip(ref,got)
                            for rowx,rowy in zip(rx,ry) for x,y in zip(rowx,rowy)]
    results.append(prove('linear_chain_joint_reassociation',[],z3.Or(*polynomial_differences),
                         '2x2 real matrices: forward, input and both original weight cotangents; arbitrary entries, no float or activation crossing'))
    cm,ck,cn,cp=z3.Reals('chain_m chain_k chain_n chain_p')
    native=6*cm*ck*cn+6*cm*cn*cp
    joint=6*cp*ck*(cm+cn)
    results.append(prove('linear_chain_leading_more_than_half',
                         [cm>0,ck>0,cn>0,cp>0,4*cp<=cm,4*cp<=cn],2*joint>=native,
                         'Leading full-training arithmetic only; narrow output p<=min(m,n)/4; scalar schedule, not timing'))
    elapsed,ops,traffic,rate,bw=z3.Reals('roofline_time roofline_ops roofline_bytes roofline_peak roofline_bandwidth')
    results.append(prove('roofline_conditional_time_bound',[rate>0,bw>0,ops<=elapsed*rate,traffic<=elapsed*bw],
                         z3.Or(ops/rate>elapsed,traffic/bw>elapsed),
                         'Supplied compute and transfer capacity bounds; not a fitted hardware-performance claim'))
    diagonal=z3.Reals('shared_diagonal0 shared_diagonal1')
    dm=[[diagonal[0],z3.RealVal(0)],[z3.RealVal(0),diagonal[1]]]
    A,B,C,G=matrices
    # A, B, C, G above are arbitrary real 2x2 matrices.
    conventional=[mm(mm(G,tr(C)),dm),mm(tr(A),mm(mm(G,tr(C)),dm)),mm(tr(mm(mm(A,B),dm)),G)]
    masked_input=mm(conventional[0],tr(B))
    reassociated=[mm(G,tr(mm(mm(B,dm),C))),mm(mm(mm(tr(A),G),tr(C)),dm),mm(mm(dm,tr(B)),mm(tr(A),G))]
    differences=[z3.simplify(x-y,som=True)!=0 for rx,ry in zip([masked_input,*conventional[1:]],reassociated)
                 for rowx,rowy in zip(rx,ry) for x,y in zip(rowx,rowy)]
    results.append(prove('common_activation_diagonal_joint_pullbacks',[],z3.Or(*differences),
                         'Arbitrary real 2x2 shared diagonal; ReLU qualification additionally requires identical Boolean branch masks'))
    ss=z3.Reals('softmax_p0 softmax_p1 softmax_p2');uu=z3.Reals('softmax_u0 softmax_u1 softmax_u2');gg=z3.Reals('softmax_g0 softmax_g1 softmax_g2')
    avg_u=sum(p*u for p,u in zip(ss,uu));avg_g=sum(p*g for p,g in zip(ss,gg))
    pair_left=sum(p*(u-avg_u)*g for p,u,g in zip(ss,uu,gg))
    pair_right=sum(u*p*(g-avg_g) for p,u,g in zip(ss,uu,gg))
    results.append(prove('softmax_diagonal_rankone_pairing_and_null_sum',
                         [sum(ss)==1,*[p>0 for p in ss]],
                         z3.Or(z3.simplify(pair_left-pair_right,som=True)!=0,sum(p*(g-avg_g) for p,g in zip(ss,gg))!=0),
                         'Normalized 3-coordinate probabilities; derivative representation pairing/null-sum, not exponential primitive or IEEE proof'))
    uu0,uu1,vv0,vv1,gg0,gg1,cc0,cc1=z3.Reals('factor_u0 factor_u1 factor_v0 factor_v1 factor_g0 factor_g1 factor_c0 factor_c1')
    product=(uu0*vv0+uu1*vv1)*(gg0*cc0+gg1*cc1)
    factored=sum(u*g*v*c for u,v in [(uu0,vv0),(uu1,vv1)] for g,c in [(gg0,cc0),(gg1,cc1)])
    results.append(prove('separable_derivative_factor_expansion',[],z3.simplify(product-factored,som=True)!=0,
                         'Two supplied derivative-mask factors and two cotangent factors; exact expansion, no rank truncation'))
    lc=z3.Reals('layernorm_c0 layernorm_c1 layernorm_c2');lg=z3.Reals('layernorm_g0 layernorm_g1 layernorm_g2');lr=z3.Real('layernorm_r')
    ln=sum(lr*(g-sum(lg)/3-lr*lr*c*sum(a*b for a,b in zip(lc,lg))/3) for c,g in zip(lc,lg))
    results.append(prove('layernorm_centered_shift_null',[sum(lc)==0],z3.simplify(ln,som=True)!=0,
                         'Three-coordinate centered LayerNorm Jacobian model; arbitrary r, so epsilon-dependent scaling remains explicit'))
    # Complete structured LayerNorm segment: input, both weights, gamma, beta,
    # final bias, with the actual supplied-statistic Jacobian. Arbitrary entries.
    def symbolic_matrix(label,rows,cols):
        cells=z3.Reals(' '.join(f'{label}_{i}_{j}' for i in range(rows) for j in range(cols)))
        return [list(cells[i*cols:(i+1)*cols]) for i in range(rows)]
    lx=symbolic_matrix('ln_x',2,2);lw1=symbolic_matrix('ln_w1',3,2);lw2=symbolic_matrix('ln_w2',2,3);lgo=symbolic_matrix('ln_go',2,2)
    lscale=z3.Reals('ln_gamma0 ln_gamma1 ln_gamma2');lbeta=z3.Reals('ln_beta0 ln_beta1 ln_beta2');lbias=z3.Reals('ln_bias0 ln_bias1');lrs=z3.Reals('ln_r0 ln_r1')
    lz=mm(lx,tr(lw1));lmean=[sum(row)/3 for row in lz]
    lnorm=[[(lz[i][j]-lmean[i])*lrs[i]*lscale[j]+lbeta[j] for j in range(3)] for i in range(2)]
    ly=mm(lnorm,tr(lw2));ly=[[ly[i][a]+lbias[a] for a in range(2)] for i in range(2)]
    gh=mm(lgo,lw2);gz=[]
    for i in range(2):
        ag=[gh[i][j]*lscale[j] for j in range(3)];center=[lz[i][j]-lmean[i] for j in range(3)]
        gz.append([lrs[i]*ag[j]-lrs[i]*sum(ag)/3-lrs[i]*lrs[i]*lrs[i]*center[j]*sum(c*h for c,h in zip(center,ag))/3 for j in range(3)])
    dense=[mm(gz,lw1),mm(tr(gz),lx),mm(tr(lgo),lnorm),
           [[sum(gh[i][j]*(lz[i][j]-lmean[i])*lrs[i] for i in range(2)) for j in range(3)]],
           [[sum(gh[i][j] for i in range(2)) for j in range(3)]],[[sum(lgo[i][a] for i in range(2)) for a in range(2)]]]
    lt=mm(tr(lgo),[[lrs[i]*lx[i][k] for k in range(2)] for i in range(2)])
    shift=mm(lt,tr(lw1));shift=[[shift[a][j]-sum(lgo[i][a]*lrs[i]*lmean[i] for i in range(2)) for j in range(3)] for a in range(2)]
    lcw=[[lw2[a][j]*lscale[j] for j in range(3)] for a in range(2)]
    correction=mm(lw2,[[b] for b in lbeta])
    alphas=[lrs[i]*lrs[i]*sum(lgo[i][a]*(ly[i][a]-lbias[a]-correction[a][0]) for a in range(2))/3 for i in range(2)]
    lu=[alphas[i]*lmean[i]-lrs[i]*sum(lgo[i][a]*sum(lcw[a]) for a in range(2))/3 for i in range(2)]
    ax=[[alphas[i]*lx[i][k] for k in range(2)] for i in range(2)]
    dxfirst=mm([[lrs[i]*g for g in lgo[i]] for i in range(2)],mm(lcw,lw1));dxlast=mm(ax,mm(tr(lw1),lw1))
    ldx=[[dxfirst[i][k]+lu[i]*sum(lw1[j][k] for j in range(3))-dxlast[i][k] for k in range(2)] for i in range(2)]
    dwfirst=mm(tr(lcw),lt);dwlast=mm(lw1,mm(tr(lx),ax))
    ldw1=[[dwfirst[j][k]+sum(lu[i]*lx[i][k] for i in range(2))-dwlast[j][k] for k in range(2)] for j in range(3)]
    lsg=[sum(lgo[i][a] for i in range(2)) for a in range(2)]
    ldw2=[[shift[a][j]*lscale[j]+lsg[a]*lbeta[j] for j in range(3)] for a in range(2)]
    folded=[ldx,ldw1,ldw2,[[sum(lw2[a][j]*shift[a][j] for a in range(2)) for j in range(3)]],
            mm([lsg],lw2),[lsg]]
    # Expand the full small polynomial before asking for a counterexample.
    # Z3's default expansion budget leaves large products of sums opaque.
    differences=[z3.simplify(a-b,som=True,som_blowup=10000)!=0 for ma,mb in zip(dense,folded) for ra,rb in zip(ma,mb) for a,b in zip(ra,rb)]
    results.append(prove('layernorm_full_structured_segment',[r>0 for r in lrs],z3.Or(*differences),
                         '2x2 input, hidden3/output2: all six VJPs, arbitrary affine normalization, supplied positive r; Jacobian primitive certified separately by symbolic differentiation/tests, not a Lean analytic certificate'))
    # The centered-weight Gram represents exactly the same row moments before
    # applying sqrt/rsqrt. A supplied common r then gives the same forward.
    mean_weight=[sum(lw1[j][k] for j in range(3))/3 for k in range(2)]
    centered_weight=[[lw1[j][k]-mean_weight[k] for k in range(2)] for j in range(3)]
    centered_gram=mm(tr(centered_weight),centered_weight)
    gram_variance=[sum(lx[i][k]*centered_gram[k][l]*lx[i][l] for k in range(2) for l in range(2))/3 for i in range(2)]
    dense_variance=[sum((lz[i][j]-lmean[i])*(lz[i][j]-lmean[i]) for j in range(3))/3 for i in range(2)]
    gram_mean=[sum(lx[i][k]*mean_weight[k] for k in range(2)) for i in range(2)]
    gram_out=mm(lx,tr(mm(lcw,lw1)))
    gram_out=[[(gram_out[i][a]-gram_mean[i]*sum(lcw[a]))*lrs[i]+sum(lw2[a][j]*lbeta[j] for j in range(3))+lbias[a] for a in range(2)] for i in range(2)]
    residuals=[a-b for a,b in zip(gram_variance,dense_variance)]+[a-b for a,b in zip(gram_mean,lmean)]
    residuals+=[gram_out[i][a]-ly[i][a] for i in range(2) for a in range(2)]
    results.append(prove('centered_weight_gram_layernorm_forward',[],z3.Or(*[z3.simplify(e,som=True,som_blowup=10000)!=0 for e in residuals]),
                         '2x2 input/hidden3/output2: exact row mean, variance and affine normalized forward with supplied r; floating conditioning guards are not certified'))
    # A concrete IEEE-754 counterexample rather than a false universal exactness theorem.
    one,large,half=(z3.FPVal(v,sort) for v in (1,100000000,0.5))
    s=z3.fpMul(z3.RNE(),half,z3.fpAdd(z3.RNE(),one,large))
    a=z3.fpMul(z3.RNE(),half,z3.fpSub(z3.RNE(),one,large))
    reconstructed=z3.fpAdd(z3.RNE(),s,a)
    solver=z3.Solver()
    solver.add(z3.Not(z3.fpEQ(reconstructed,one)))
    counterexample=solver.check()
    witness={"name":"SAB_universal_FP32_exactness_refuted","status":str(counterexample),
             "refutation_verified":counterexample==z3.sat,"w":1,"transpose_partner":100000000,
             "reconstruction":str(z3.simplify(reconstructed)),"scope":"IEEE binary32, round-to-nearest ties-to-even"}
    return {"solver_version":z3.get_version_string(),"proofs":results,"all_proofs_passed":all(r['verified'] for r in results),
            "counterexamples":[witness],"no_GPU_hardware_certification":True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/z3-results.json')
    args=parser.parse_args()
    data=verify()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(data,indent=2))
    print(f"Z3: {sum(p['verified'] for p in data['proofs'])}/{len(data['proofs'])} checks verified; FP32 counterexample {data['counterexamples'][0]['status']}")
    raise SystemExit(0 if data['all_proofs_passed'] and data['counterexamples'][0]['refutation_verified'] else 1)
