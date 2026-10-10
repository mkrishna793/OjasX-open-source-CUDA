"""Fail-closed abstract-engine discovery, Lean audit and CPU qualification."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports'
env=dict(os.environ,ELAN_HOME=str(ROOT/'.tools/elan'),PYTHONDONTWRITEBYTECODE='1')

def run(name,args,cwd=ROOT):
    print('Checking',name,flush=True)
    with (REPORTS/(name+'.txt')).open('w') as out:
        p=subprocess.run(args,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT)
    if p.returncode:
        print((REPORTS/(name+'.txt')).read_text()[-6000:]);raise SystemExit(name+' failed')
    return (REPORTS/(name+'.txt')).read_text()

def main():
    result_path=REPORTS/'abstract-verification.json';result_path.write_text('{"status":"running"}\n')
    py=[sys.executable,'-B']
    run('backprop-discovery-run',py+['tools/discover_backprop.py'])
    sources=sorted((ROOT/'lean/OjasXMath').glob('*.lean'))
    statements=[name for p in sources for name in re.findall(r'^theorem\s+(\w+)',p.read_text(),re.M)]
    if len(statements)!=len(set(statements)):raise SystemExit('Duplicate project theorem names')
    for p in sources:
        stripped=re.sub(r'/\-.*?\-/','',p.read_text(),flags=re.S)
        stripped=re.sub(r'--[^\n]*','',stripped)
        if re.search(r'\b(sorry|admit|axiom|unsafe)\b',stripped):raise SystemExit('Trust escape in '+p.name)
    audit=''.join('import OjasXMath.'+p.stem+'\n' for p in sources)+'\n'+''.join('#print axioms OjasXMath.'+name+'\n' for name in statements)
    (ROOT/'lean/OjasXMath.lean').write_text(audit)
    lake=str(ROOT/'.tools/elan/bin/lake')
    run('abstract-lean-build',[lake,'build'],ROOT/'lean')
    output=run('abstract-lean-axioms',[lake,'env','lean','OjasXMath.lean'],ROOT/'lean')
    groups=re.findall(r"(?:depends on axioms:)\s*\[([^\]]*)\]",output)
    # Lean also reports theorem declarations with no axioms separately.
    declarations=re.findall(r"(?:depends on axioms:|does not depend on any axioms)",output)
    if len(declarations)!=len(statements) or 'sorryAx' in output or 'error:' in output:
        raise SystemExit('Incomplete axiom audit')
    used={a.strip() for group in groups for a in group.split(',') if a.strip()}
    if used-{'propext','Classical.choice','Quot.sound'}:raise SystemExit('Unexpected axioms '+str(used))
    run('feature-core-run',py+['tools/test_feature_core.py'])
    run('polynomial-roundoff-run',py+['tools/certify_polynomial_roundoff.py'])
    run('abstract-regression-pytest',py+['-m','pytest','-p','no:cacheprovider','-q','--ignore=tests/test_mps_qualification.py'])
    run('abstract-rocq-run',py+['tools/verify_rocq.py'])
    discovery=json.loads((REPORTS/'backprop-discovery.json').read_text())
    discovery['status']='passed_symbolic_and_lean_kernel_checks'
    (REPORTS/'backprop-discovery.json').write_text(json.dumps(discovery,indent=2)+'\n')
    tracked=[*sources,ROOT/'lean/OjasXMath.lean',ROOT/'lean/lean-toolchain',ROOT/'lean/lakefile.toml',ROOT/'lean/lake-manifest.json',
        * (ROOT/'ojasx2').glob('*.py'),* (ROOT/'tools').glob('*.py'),ROOT/'ABSTRACT_BACKPROP_ENGINE.md',ROOT/'foundation_manifest.json']
    result={'status':'passed','lean_theorems':len(statements),'standard_lean_axioms':sorted(used),'added_axioms':[],
        'discovery':discovery,'feature_core':json.loads((REPORTS/'feature-core.json').read_text()),
        'roundoff':json.loads((REPORTS/'polynomial-roundoff.json').read_text()),
        'rocq':{'status':'passed','checked_statements':json.loads((REPORTS/'rocq-verification.json').read_text())['checked_statements']},
        'regression_output':(REPORTS/'abstract-regression-pytest.txt').read_text(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)},
        'scope':'Exact abstract theorems and bounded polynomial discovery/qualification. No universal savings, backend dispatch, GPU refinement or global minimality theorem.'}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',len(statements),'Lean theorems; discovery, roundoff, CPU regression and Rocq qualification.')

if __name__=='__main__':main()
