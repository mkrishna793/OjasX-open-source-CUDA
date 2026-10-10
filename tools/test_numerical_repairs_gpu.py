"""GPU-only repaired formulas, original gradients, HVPs and rejection checks."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
from pathlib import Path
import sys, json, hashlib
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))
from ojasx2.numerical_repairs import (
    stable_scaled_layer_norm, radial_layer_norm, reliable_projection,
    checked_value_and_grad)
from ojasx2.invariant_radial_compiler import invariant_radial
from ojasx2.gradient_factor_compiler import native_projection
from test_invariant_gpu import make
from search_nonlinear_gpu import clone, compare

def main():
    assert torch.backends.mps.is_available(), "GPU required; no CPU fallback"
    cases = []
    # First/second derivatives of the same stable scaled-normalization primal.
    for eps in (1e-2, 1e-5):
        for scale_value in (0., -.7, 1.):
            torch.manual_seed(42)
            u = torch.randn(8, 8, device="mps", requires_grad=True)
            a = torch.full((8, 1), scale_value, device="mps", requires_grad=True)
            p = torch.randn_like(u)
            candidates = []
            for fn in (lambda: F.layer_norm(a*u, (8,), eps=eps),
                       lambda: stable_scaled_layer_norm(u, a, eps)):
                h = fn()
                g = torch.autograd.grad(h, (u, a), p, create_graph=True)
                hv = torch.autograd.grad(sum(t.square().sum() for t in g), (u, a))
                candidates.append((h, *g, *hv))
            errors = [compare(x,y) for x,y in zip(candidates[1], candidates[0])]
            # Preserve raw relative failures; independently compare finite
            # derivatives with native under the same absolute tolerance.
            finite = all(bool(torch.isfinite(x).all()) for x in candidates[1])
            absolute_pass = all(bool(torch.allclose(x,y,rtol=2e-4,atol=2e-4))
                                for x,y in zip(candidates[1],candidates[0]))
            cases.append({"case":"scaled_norm", "eps":eps,"scale":scale_value,
                          "raw_errors":errors,"raw_relative_passed":all(e["passed"] for e in errors),
                          "finite":finite,"absolute_comparison_passed":absolute_pass})
            assert finite and absolute_pass
    u = torch.tensor([[1., -1., 2., -2.]], device="mps", requires_grad=True)
    a = torch.zeros(1,1,device="mps",requires_grad=True)
    h = stable_scaled_layer_norm(u, a, eps=1e-30)
    g = torch.autograd.grad(h, (u,a), u.detach())
    r = torch.rsqrt(torch.tensor([1e-30],device="mps"))
    overflow = bool(torch.isinf(r**3).all())
    assert overflow and all(bool(torch.isfinite(x).all()) for x in (h,*g))
    cases.append({"case":"avoid_inverse_cube_overflow","naive_inverse_cube_overflows":overflow,
                  "repaired_value_and_gradients_finite":True})
    # All original parameters of a radial -> LayerNorm -> radial scalar loss.
    models = []
    for seed in (3301,3302,3303):
        torch.manual_seed(seed)
        v1=make((2048,32,2048,32),"sigmoid")
        v2=make((2048,32,2048,8),"reciprocal")
        operands=clone((*v1,*v2[1:]))
        labels=torch.arange(2048,device="mps")%8
        def native(*v):
            h=radial_layer_norm(*v[:4],gate=v[4],accuracy="strict")
            return F.cross_entropy(invariant_radial(h,*v[5:],"reciprocal",
                                                   accuracy="strict"),labels)
        def candidate(*v):
            h=radial_layer_norm(*v[:4],gate=v[4],accuracy="algebraic")
            return F.cross_entropy(invariant_radial(h,*v[5:],"reciprocal",
                                                   accuracy="algebraic"),labels)
        evidence=[]
        selected, grads=checked_value_and_grad(native,candidate,operands,evidence=evidence)
        reference=native(*operands)
        reference_grads=torch.autograd.grad(reference,operands)
        errors=[compare(selected.reshape(1),reference.reshape(1)),
                *[compare(x,y) for x,y in zip(grads,reference_grads)]]
        assert all(e["passed"] for e in errors)
        # Three steps of the checked scalar loss/gradient interface.
        states=clone(operands); reference_states=clone(operands); training=[]
        for _ in range(3):
            loss,gs=checked_value_and_grad(native,candidate,states,evidence=evidence)
            reference_loss=native(*reference_states)
            reference_gs=torch.autograd.grad(reference_loss,reference_states)
            assert bool(torch.isfinite(loss))
            with torch.no_grad():
                for parameter,gradient in zip(states[1:],gs[1:]):
                    parameter.add_(gradient,alpha=-1e-4)
                for parameter,gradient in zip(reference_states[1:],reference_gs[1:]):
                    parameter.add_(gradient,alpha=-1e-4)
            step_errors=[compare(loss.reshape(1),reference_loss.reshape(1)),
                         *[compare(x,y) for x,y in zip(states,reference_states)]]
            assert all(e["passed"] for e in step_errors)
            training.append({"finite_parameters":all(bool(torch.isfinite(t).all()) for t in states),
                             "native_step_errors":step_errors,"passed":True})
        models.append({"seed":seed,"evidence":evidence,"selected_errors":errors,
                       "training":training,"passed":True})
    policies=[]
    for k,kind,degree,order,qualified in [
        (1,"gelu",4,1,True),(2,"silu",8,1,True),
        (32,"gelu",8,1,True),(1,"gelu",8,2,True),
        (1,"gelu",8,1,False)]:
        torch.manual_seed(12)
        operands=clone((torch.randn(128,k,device="mps")*.1,
                        torch.randn(k,512,device="mps")*.05,
                        torch.zeros(512,device="mps"),
                        torch.randn(512,2,device="mps")*.05))
        ev=[]
        h=reliable_projection(*operands,kind=kind,degree=degree,
                              derivative_order=order,accuracy="approximate",
                              whole_program_qualified=qualified,evidence=ev)
        native=native_projection(*operands,kind)
        up=torch.ones_like(h)
        ga=torch.autograd.grad(h,operands,up,create_graph=order>1)
        gb=torch.autograd.grad(native,operands,up,create_graph=order>1)
        errors=[compare(h,native),*[compare(x,y) for x,y in zip(ga,gb)]]
        if order>1:
            ha=torch.autograd.grad(sum(x.square().sum() for x in ga),operands)
            hb=torch.autograd.grad(sum(x.square().sum() for x in gb),operands)
            errors += [compare(x,y) for x,y in zip(ha,hb)]
        assert all(e["passed"] for e in errors),(k,kind,errors)
        policies.append({"input_width":k,"activation":kind,"requested_degree":degree,
                         "derivative_order":order,"qualified":qualified,
                         "evidence":ev,"errors":errors,"passed":True})
    # Invalid eps/domain and CPU operands are rejected explicitly.
    invalid=[]
    for eps in (0.,-1.,float("nan")):
        try:stable_scaled_layer_norm(u,a,eps)
        except ValueError:invalid.append({"eps":str(eps),"rejected":True})
        else:raise AssertionError("Invalid epsilon admitted")
    rejection=[]
    probe=torch.tensor([1.,2.],device="mps",requires_grad=True)
    frozen=torch.tensor([3.],device="mps")
    for name,fn in [
        ("wrong_gradient",lambda x,f:x.sum()),
        ("constant_no_gradient",lambda x,f:torch.zeros((),device="mps")),
        ("nonfinite_candidate",lambda x,f:x.sum()*float("nan"))]:
        ev=[]
        value,grads=checked_value_and_grad(lambda x,f:x.square().sum(),fn,
                                           (probe,frozen),evidence=ev)
        assert value.item()==5 and bool(torch.equal(grads[0],2*probe)) and grads[1] is None
        rejection.append({"case":name,"evidence":ev,"passed":True})
    paths=[Path(__file__),ROOT/"ojasx2/numerical_repairs.py",
           ROOT/"lean/OjasXMath/InvariantRadial.lean",ROOT/"rocq/InvariantRadial.v"]
    result={"status":"passed_scoped_gpu_repairs","device":"mps","cpu_ml_execution":False,
            "cpu_fallback_enabled":False,"normalization":cases,"models":models,
            "policies":policies,"invalid_domains":invalid,
            "rejection_checks":rejection,
            "scope":"Stable real-normalization formula and finite GPU tests; raw near-zero relative failures retained. Checked execution validates every supplied scalar loss/all gradients, else native. No universal compatibility, speed or optimizer convergence certificate.",
            "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in paths}}
    (ROOT/"reports/numerical-repairs-gpu.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"normalization_cases":len(cases),
                      "models":len(models),"policies":len(policies),
                      "raw_relative_normalization_failures":sum(
                          not c.get("raw_relative_passed",True) for c in cases),
                      "model_decisions":[m["evidence"] for m in models]},indent=2))

if __name__=="__main__":main()
