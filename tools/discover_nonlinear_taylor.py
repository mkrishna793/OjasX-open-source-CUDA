"""Bounded symbolic discovery of small-domain nonlinear feature rules.

Emits tensor coefficient builders and pullbacks from polynomial coefficients.
No ML workloads execute here. GPU qualification is a separate mandatory step.
"""
import json
from pathlib import Path
import sys
import sympy as sp
import z3 as z
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from discover_backprop import lean,derivative_proof,encode

def rocq(expr):
    if expr.is_Symbol:return str(expr)
    if expr.is_Integer:
        value=int(expr);base='0' if not value else '('+' + '.join('1' for _ in range(abs(value)))+')'
        return base if value>=0 else '(-'+base+')'
    if expr.is_Rational:return f'({rocq(sp.Integer(expr.p))} / {rocq(sp.Integer(expr.q))})'
    if expr.is_Add:return '('+' + '.join(rocq(q) for q in expr.args)+')'
    if expr.is_Mul:return '('+' * '.join(rocq(q) for q in expr.args)+')'
    if expr.is_Pow:return '('+' * '.join(rocq(expr.base) for _ in range(int(expr.exp)))+')'
    raise ValueError(expr)

def main():
    x=sp.Symbol('x');sigmoid=1/(1+sp.exp(-x))
    functions={'silu':x*sigmoid,'gelu':x*(1+sp.erf(x/sp.sqrt(2)))/2}
    series={name:{'primal':str(sp.series(phi,x,0,5)),
        'derivative':str(sp.series(sp.diff(phi,x),x,0,4)),
        'quadratic':str(sp.series(phi,x,0,3).removeO())} for name,phi in functions.items()}
    checks=[]
    def check(name,constraints,want):
        solver=z.Solver();solver.set(timeout=20000);solver.add(*constraints);result=solver.check()
        if str(result)!=want:raise AssertionError((name,str(result),solver.reason_unknown()))
        checks.append({'name':name,'result':str(result),'witness':str(solver.model()) if result==z.sat else None})
    s=z.Real('sigmoid_value');t=s*(1-s)
    ab=lambda v:z.If(v>=0,v,-v)
    check('sigmoid_third_factor_bound',[s>=0,s<=1,ab(t*(1-6*t))>z.RealVal('1/4')],'unsat')
    check('sigmoid_fourth_factor_bound',[s>=0,s<=1,ab(t*(1-2*s)*(1-12*t))>z.RealVal('1/2')],'unsat')
    xv=z.Real('silu_argument')
    check('reject_identity_for_generic_bounded_sigmoid_value',[xv==1,
        s>0,s<1,s!=z.RealVal('3/4'),xv*s!=xv/2+xv*xv/4],'sat')
    # The witness concerns the polynomial surrogate with a bounded sigmoid
    # variable; the actual activation mismatch is qualified on GPU.
    source=['"""Generated coefficient maps. See discover_nonlinear_taylor.py; not GPU-qualified by generation alone."""',
        'import torch','import math','BUILDERS = {}','PULLS = {}','TERMS = {}','COUNTS = {}']
    for kind in ('activation','swiglu'):
        for width in (1,2,4):
            xs=sp.symbols(' '.join(f'x{i}' for i in range(width)),seq=True)
            aa=sp.symbols(' '.join(f'a{i}' for i in range(width)),seq=True)
            vv=sp.symbols(' '.join(f'v{i}' for i in range(width)),seq=True)
            c=sp.Symbol('c');gate=sum(u*v for u,v in zip(xs,aa));up=sum(u*v for u,v in zip(xs,vv))
            value=(gate/2+c*gate**2) if kind=='activation' else (gate/2+gate**2/4)*up
            poly=sp.Poly(value,*xs);rows=sorted(poly.terms(),key=lambda p:(sum(p[0]),p[0]))
            terms=[tuple(i for i,power in enumerate(powers) for _ in range(power)) for powers,_ in rows]
            coefficients=[coef for _,coef in rows];tag=f'{kind}_{width}'
            source += [f'def build_{tag}(a,v,c):',*[f'    a{i}=a[{i}]' for i in range(width)]]
            if kind=='swiglu':source += [f'    v{i}=v[{i}]' for i in range(width)]
            source += ['    return torch.stack(['+','.join(str(coef) for coef in coefficients)+'])']
            source += [f'def pull_{tag}(a,v,c,dh):',*[f'    a{i}=a[{i}]' for i in range(width)]]
            if kind=='swiglu':source += [f'    v{i}=v[{i}]' for i in range(width)]
            def gradient(var):
                terms=[f'({sp.diff(coef,var)})*dh[{j}]' for j,coef in enumerate(coefficients) if sp.diff(coef,var)!=0]
                return '+'.join(terms) if terms else 'torch.zeros_like(a[0])'
            source += ['    da=torch.stack(['+','.join(gradient(ai) for ai in aa)+'])']
            source += ['    dv=torch.stack(['+','.join(gradient(vi) for vi in vv)+'])' if kind=='swiglu' else '    dv=None']
            source += ['    return da,dv',f'BUILDERS[{(kind,width)!r}]=build_{tag}',f'PULLS[{(kind,width)!r}]=pull_{tag}',
                f'TERMS[{(kind,width)!r}]={terms!r}']
            def ops(expr):
                # Scalar straight-line count, including constant multiplies and
                # replacing integer powers with repeated products.
                if expr.is_Atom:return 0,0
                sub=[ops(q) for q in expr.args]
                mul=sum(q[0] for q in sub);add=sum(q[1] for q in sub)
                if expr.is_Add:add+=len(expr.args)-1
                elif expr.is_Mul:mul+=len(expr.args)-1
                elif expr.is_Pow:
                    mul,add=ops(expr.base);mul+=int(expr.exp)-1
                return mul,add
            hm=sum(ops(coef)[0] for coef in coefficients);ha=sum(ops(coef)[1] for coef in coefficients)
            gm=ga=0
            for var in (*aa,*vv) if kind=='swiglu' else aa:
                expressions=[sp.diff(coef,var) for coef in coefficients if sp.diff(coef,var)!=0]
                gm+=sum(ops(q)[0]+1 for q in expressions)
                ga+=sum(ops(q)[1] for q in expressions)+max(0,len(expressions)-1)
            source += [f'COUNTS[{(kind,width)!r}]={dict(coefficient_mul=hm,coefficient_add=ha,pull_mul=gm,pull_add=ga)!r}']
    (ROOT/'ojasx2/taylor_coefficients.py').write_text('\n'.join(source)+'\n')
    # Kernel-checkable complete finite scalar instances of the surrogate.
    proof_source=['import OjasXMath.CachedReverse','import Mathlib.Basic.Real.Basic','import Mathlib.Analysis.Calculus.Deriv.Add',
        'import Mathlib.Analysis.Calculus.Deriv.Mul','import Mathlib.Analysis.Calculus.Deriv.Pow',
        'import Mathlib.Tactic.Ring','set_option autoImplicit false','namespace OjasXMath']
    theorem_count=0
    rocq_source=['From Stdlib Require Import QArith.Qcanon Field.','Open Scope Qc_scope.']
    for kind in ('activation','swiglu'):
        xx=sp.symbols('x0 x1');aa=sp.symbols('a00 a01 a10 a11');vv=sp.symbols('v00 v01 v10 v11');bb=sp.symbols('b0 b1');c=sp.Symbol('c')
        gates=[xx[0]*aa[j]+xx[1]*aa[2+j] for j in range(2)]
        ups=[xx[0]*vv[j]+xx[1]*vv[2+j] for j in range(2)]
        native=sum((q/2+c*q**2 if kind=='activation' else (q/2+q**2/4)*ups[j])*bb[j] for j,q in enumerate(gates))
        variables=(*xx,*aa,*bb,c) if kind=='activation' else (*xx,*aa,*vv,*bb)
        poly=sp.Poly(native,*xx);candidate=sum(coef*xx[0]**powers[0]*xx[1]**powers[1] for powers,coef in poly.terms())
        args=' '.join(map(str,variables));env={str(v):z.Real(str(v)) for v in variables}
        for label,left,right in [('primal',native,candidate)]+[(f'gradient_{v}',sp.diff(native,v),sp.diff(candidate,v)) for v in variables]:
            check(f'{kind}_{label}',[encode(left,env)!=encode(right,env)],'unsat')
            proof_source += [f'theorem taylor_{kind}_{label} ({args} : ℝ) :',f'    {lean(left)} = {lean(right)} := by ring']
            theorem_count+=1
            rocq_source += [f'Theorem taylor_{kind}_{label} ({args} : Qc) :',f'  {rocq(left)} = {rocq(right)}.', 'Proof. field; repeat split; vm_compute; discriminate. Qed.']
        for var in variables:
            name=f'taylor_{kind}_native_derivative_{var}';temp=sp.Symbol('t')
            proof_source += [f'theorem {name} ({args} : ℝ) :',
                f'    HasDerivAt (fun t : ℝ => {lean(native.subs(var,temp))}) ({lean(sp.diff(native,var))}) {var} := by',
                f'  convert {derivative_proof(native,var)} using 1 <;> (try funext t) <;> dsimp <;> ring']
            theorem_count+=1
    # Check the vectorized full tensor coefficient/input pulls, for arbitrary
    # incoming coefficient cotangents, including repeated-index multiplicities.
    for kind in ('activation','swiglu'):
        aa=sp.symbols('a0 a1');vv=sp.symbols('v0 v1');xx=sp.symbols('x0 x1');c=sp.Symbol('c')
        if kind=='activation':
            hs=[a/2 for a in aa]+[c*a*b for a in aa for b in aa]
            ts=list(xx)+[a*b for a in xx for b in xx]
        else:
            hs=[a*v/2 for a in aa for v in vv]+[a*b*v/4 for a in aa for b in aa for v in vv]
            ts=[a*b for a in xx for b in xx]+[a*b*c0 for a in xx for b in xx for c0 in xx]
        dh=sp.symbols(' '.join('h'+str(i) for i in range(len(hs))))
        dt=sp.symbols(' '.join('t'+str(i) for i in range(len(ts))))
        rows=[]
        for i in range(2):
            if kind=='activation':
                da=dh[i]/2+c*sum((dh[2+2*i+j]+dh[2+2*j+i])*aa[j] for j in range(2))
                dx=dt[i]+sum((dt[2+2*i+j]+dt[2+2*j+i])*xx[j] for j in range(2))
            else:
                da=sum(dh[2*i+j]*vv[j]/2 for j in range(2))+sum((dh[4+4*i+2*j+l]+dh[4+4*j+2*i+l])*aa[j]*vv[l]/4 for j in range(2) for l in range(2))
                dv=sum(dh[2*j+i]*aa[j]/2 for j in range(2))+sum(dh[4+4*j+2*l+i]*aa[j]*aa[l]/4 for j in range(2) for l in range(2))
                dx=sum((dt[2*i+j]+dt[2*j+i])*xx[j] for j in range(2))+sum((dt[4+4*i+2*j+l]+dt[4+4*j+2*i+l]+dt[4+4*j+2*l+i])*xx[j]*xx[l] for j in range(2) for l in range(2))
                rows.append((f'v{i}',sp.diff(sum(h*g for h,g in zip(hs,dh)),vv[i]),dv))
            rows += [(f'a{i}',sp.diff(sum(h*g for h,g in zip(hs,dh)),aa[i]),da),
                     (f'x{i}',sp.diff(sum(t*g for t,g in zip(ts,dt)),xx[i]),dx)]
        for label,left,right in rows:
            variables=sorted(left.free_symbols|right.free_symbols,key=str);args=' '.join(map(str,variables));env={str(v):z.Real(str(v)) for v in variables}
            name=f'tensor_{kind}_pull_{label}';check(name,[encode(left,env)!=encode(right,env)],'unsat')
            proof_source += [f'theorem {name} ({args} : ℝ) : {lean(left)} = {lean(right)} := by ring']
            rocq_source += [f'Theorem {name} ({args} : Qc) : {rocq(left)} = {rocq(right)}.', 'Proof. field; repeat split; vm_compute; discriminate. Qed.']
            theorem_count+=1
    # Tool-derived genuine SiLU derivative chain; sigmoid is a function,
    # rather than an unconstrained leaf as in the polynomial surrogate proofs.
    zz,ss=sp.symbols('z s');poly=zz*ss
    proof_source += ['-- These four statements differentiate actual SiLU.']
    def sigmoid_expr(e,point):return lean(e).replace('s',f'(cacheSigmoid {point})').replace('z',point)
    def sigmoid_proof(e):
        if e==zz:return '(hasDerivAt_id z)'
        if e==ss:return '(cache_sigmoid_hasDerivAt z)'
        if not e.has(zz,ss):return f'(hasDerivAt_const z ({lean(e)}))'
        if e.is_Add or e.is_Mul:
            op='add' if e.is_Add else 'mul';children=[sigmoid_proof(q) for q in e.args];result=children[0]
            for q in children[1:]:result=f'({result}.{op} {q})'
            return result
        if e.is_Pow:return f'({sigmoid_proof(e.base)}.pow {int(e.exp)})'
        raise ValueError(e)
    for order in range(1,5):
        nxt=sp.expand(sp.diff(poly,zz)+sp.diff(poly,ss)*ss*(1-ss))
        proof_source += [f'theorem actual_silu_derivative_{order} (z : ℝ) :',
            f'  HasDerivAt (fun t : ℝ => {sigmoid_expr(poly,"t")}) ({sigmoid_expr(nxt,"z")}) z := by',
            f'  convert {sigmoid_proof(poly)} using 1 <;> (try funext t) <;> dsimp <;> ring']
        theorem_count+=1;poly=nxt
    factored=4*ss*(1-ss)*(1-6*ss*(1-ss))+zz*ss*(1-ss)*(1-2*ss)*(1-12*ss*(1-ss))
    check('silu_fourth_derivative_factorization',[encode(poly,{'z':z.Real('z'),'s':s})!=encode(factored,{'z':z.Real('z'),'s':s})],'unsat')
    proof_source += [f'theorem actual_silu_fourth_factorization (z s : ℝ) : {lean(poly)} = {lean(factored)} := by ring']
    theorem_count+=1
    (ROOT/'rocq/TaylorDiscovered.v').write_text('\n'.join(rocq_source)+'\n')
    proof_source += ['end OjasXMath']
    (ROOT/'lean/OjasXMath/NonlinearDiscovered.lean').write_text('\n'.join(proof_source)+'\n')
    report={'status':'symbolic_pass_pending_lean_and_gpu','series':series,'checks':checks,'generated_theorems':theorem_count,'generated_rocq_statements':sum(q.startswith('Theorem') for q in rocq_source),
        'input_widths':[1,2,4],'primal_changed':True,
        'analytic_remainder_model':{'silu_fourth_derivative_bound':'1+delta/2','gelu_fourth_derivative_bound':'(delta^4+7*delta^2+4)/2',
            'primal_absolute_bound':'M*delta^4/24','derivative_absolute_bound':'M*delta^3/6',
            'scope':'Taylor smoothness/coefficient and activation analytic obligations must be discharged separately; not a floating GPU certificate.'}}
    (ROOT/'reports/nonlinear-discovery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'generated_theorems':theorem_count,'generated_rocq_statements':sum(q.startswith('Theorem') for q in rocq_source),'z3_checks':len(checks),'series':series},indent=2))

if __name__=='__main__':main()
