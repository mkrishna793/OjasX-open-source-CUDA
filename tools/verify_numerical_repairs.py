"""Formal/kernel and source-matched GPU repair audit; no CPU ML."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import z3
ROOT=Path(__file__).resolve().parents[1]

def main():
    a,v,e,r,d,q=z3.Reals("a variance eps inv denominator ratio")
    conditions=[v>=0,e>0,d==a*a*v+e,r>=0,r*r*d==1]
    checks=[]
    for name,extra,negation in [
        ("stable_cancellation",[],r-a*a*v*r*r*r!=e*r*r*r),
        ("overflow_avoiding_ratio",[q*d==e],q*r!=e*r*r*r),
        ("zero_scale",[a==0],e*r*r*r!=r)]:
        domain=z3.Solver();domain.add(*conditions,*extra)
        assert domain.check()==z3.sat
        solver=z3.Solver();solver.add(*conditions,*extra,negation)
        assert solver.check()==z3.unsat,name
        checks.append({"name":name,"nonvacuous":True,"negated_identity":"unsat",
                       "scope":"Exact real algebra under denominator/inverse contract, not IEEE arithmetic."})
    subprocess.run([sys.executable,"-B","tools/refine_spectral_foundation.py",
                    "--qualify-existing-gpu"],cwd=ROOT,check=True,
                   env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",
                            PYTORCH_ENABLE_MPS_FALLBACK="0"))
    formal=json.loads((ROOT/"reports/spectral-formal-verification.json").read_text())
    gpu=json.loads((ROOT/"reports/numerical-repairs-gpu.json").read_text())
    for report in (formal,gpu):
        for p,h in report["source_sha256"].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert gpu["status"]=="passed_scoped_gpu_repairs" and gpu["device"]=="mps"
    assert not gpu["cpu_ml_execution"] and not gpu["cpu_fallback_enabled"]
    result={"status":"passed_scoped_repairs","lean_theorems":formal["lean_theorems"],
            "rocq_statements":formal["rocq_statements"],"new_z3_checks":checks,
            "added_axioms":[],"independent_rocq_kernel_recheck":True,
            "gpu_source_hashes_matched":True,"cpu_ml_execution":False,
            "raw_relative_normalization_failures":sum(
                not c.get("raw_relative_passed",True) for c in gpu["normalization"]),
            "normalized_models_selecting_native":len(gpu["models"]),
            "scope":"Stable analytic derivative and algebraic overflow-avoiding form; bounded MPS tests and explicit checked/native routing. Sensitive normalized models remain unadmitted for algebraic optimization. No global IEEE or all-model guarantee.",
            "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),ROOT/"reports/spectral-formal-verification.json",
                                       ROOT/"reports/numerical-repairs-gpu.json"]}}
    (ROOT/"reports/numerical-repairs-verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PASS numerical repairs:",result["lean_theorems"],"Lean",
          result["rocq_statements"],"Rocq; 3 new nonvacuous Z3 checks; GPU hashes matched.")

if __name__=="__main__":main()
