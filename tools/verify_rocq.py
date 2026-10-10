"""Build, audit, extract and test the isolated Rocq mathematical foundation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROCQ = ROOT / 'rocq'
REPORTS = ROOT / 'reports'
WRAPPER = ROOT / 'tools/rocq.sh'
MODULES = ['Product', 'Matrix', 'Nonlinear', 'Optimizer', 'Planner',
           'Contracts', 'Structured', 'RankSeven', 'Foundation', 'Compact', 'Indexed', 'TaylorDiscovered', 'CoordinateCompiler', 'CoordinateDiscovered']

def run(args, logfile, cwd=ROCQ):
    with (REPORTS/logfile).open('w') as out:
        result = subprocess.run([str(WRAPPER), *args], cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        print((REPORTS/logfile).read_text()[-6000:])
        raise SystemExit(f'Failed: {logfile}')
    return (REPORTS/logfile).read_text()

def main():
    REPORTS.mkdir(exist_ok=True)
    result_path=REPORTS/'rocq-verification.json'
    result_path.write_text(json.dumps({'status':'running'},indent=2)+'\n')
    statements=[]
    for mod in MODULES:
        source=(ROCQ/(mod+'.v')).read_text()
        # Source policy complements the kernel assumption audit below.
        if re.search(r'\b(?:Admitted|admit|Axiom|Parameter|Abort)\b', source):
            raise SystemExit(f'Forbidden unfinished proof or external assumption: {mod}')
        statements.extend((mod,name) for name in re.findall(r'^(?:Theorem|Lemma)\s+(\w+)', source,re.M))
    if {p.stem for p in ROCQ.glob('*.v')} != set(MODULES+['Extract','Audit']):
        raise SystemExit('Every Rocq source module must be included in the verification build')
    manifest=json.loads((ROOT/'foundation_manifest.json').read_text())
    known={f'{mod}.{name}' for mod,name in statements}
    for family in manifest['families']:
        if not set(family['proofs']) <= known:
            raise SystemExit('Foundation manifest references an unaudited statement: '+family['name'])
    audit='From Ojasx Require Import '+' '.join(MODULES)+'.\n\n'+''.join(
        f'Print Assumptions {mod}.{name}.\n' for mod,name in statements)
    if (ROCQ/'Audit.v').read_text()!=audit:
        raise SystemExit('Audit.v must enumerate every theorem and lemma exactly')
    version=run(['rocq','--version'],'rocq-version.txt').strip()
    for mod in MODULES+['Extract','Audit']:
        print('Checking',mod,flush=True)
        run(['rocq','compile','-Q','.','Ojasx',mod+'.v'],f'rocq-{mod.lower()}.txt')
    assumptions=(REPORTS/'rocq-audit.txt').read_text()
    if assumptions.count('Closed under the global context')!=len(statements) or 'Axioms:' in assumptions:
        raise SystemExit('Unexpected proof assumptions; inspect rocq-audit.txt')
    print('Independent kernel recheck',flush=True)
    run(['rocq','check','-Q','.','Ojasx',*[f'Ojasx.{m}' for m in MODULES]],'rocq-kernel-check.txt')
    print('Compiling extracted OCaml',flush=True)
    run(['ocamlfind','ocamlopt','-package','zarith','-linkpkg','ojasx_math.mli','ojasx_math.ml','cli.ml',
         '-o','ojasx_math_cli'],'rocq-ocaml-build.txt',ROCQ/'extracted')
    run(['ocamlfind','ocamlopt','-package','zarith','-linkpkg','ojasx_foundation.mli','ojasx_foundation.ml','foundation_cli.ml',
         '-o','ojasx_foundation_cli'],'rocq-foundation-ocaml-build.txt',ROCQ/'extracted')
    run(['ocamlfind','ocamlopt','-package','zarith','-linkpkg','ojasx_refined.mli','ojasx_refined.ml','refined_cli.ml',
         '-o','ojasx_refined_cli'],'rocq-refined-ocaml-build.txt',ROCQ/'extracted')
    print('Exact oracle and boundary tests',flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/test_rocq_engine.py')],cwd=ROOT,check=True)
    print('Z3 refinement checks',flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_rocq_z3.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/test_broad_foundation.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/test_foundation_edges.py')],cwd=ROOT,check=True)
    sources=sorted([*ROCQ.glob('*.v'),ROCQ/'_RocqProject',* (ROCQ/'extracted').glob('*.ml'),
        *(ROCQ/'extracted').glob('*.mli'),ROOT/'tools/rocq.sh',ROOT/'tools/verify_rocq.py',
        ROOT/'tools/test_rocq_engine.py',ROOT/'tools/verify_rocq_z3.py',ROOT/'tools/bootstrap_rocq_deps.py',
        ROOT/'tools/test_broad_foundation.py',*(ROOT/'ojasx2').glob('*.py'),
        ROOT/'tools/test_foundation_edges.py',ROOT/'foundation_manifest.json',
        ROOT/'BACKEND_MATH_FOUNDATION_V1.md',ROOT/'FOUNDATION_REFINEMENT.md'])
    result={'status':'passed','version':version,'checked_statements':len(statements),
        'axioms':[],'independent_kernel_recheck':True,'extraction':'OCaml, exact Qc with arbitrary-precision inductive integers',
        'oracle_tests':json.loads((REPORTS/'rocq-engine-tests.json').read_text()),
        'z3':json.loads((REPORTS/'rocq-z3.json').read_text()),
        'broad_foundation':json.loads((REPORTS/'broad-foundation.json').read_text()),
        'edge_qualification':json.loads((REPORTS/'foundation-edges.json').read_text()),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scope':'Exact rational specifications and extracted reference functions; no floating-point, optimized GEMM code generation or GPU refinement proof.'}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(statements)} closed Rocq statements, extraction, oracle tests and Z3 checks.')

if __name__=='__main__':
    main()
