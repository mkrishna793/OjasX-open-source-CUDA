"""Fail-closed verification runner with source hashes and explicit scopes."""
import hashlib
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
env = dict(os.environ, ELAN_HOME=str(ROOT / '.tools/elan'),
           MATHLIB_CACHE_DIR=str(ROOT / '.tools/mathlib-cache'), PYTHONDONTWRITEBYTECODE='1')


def run(name, command, cwd=ROOT):
    print(f'Running {name}', flush=True)
    proc = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (REPORTS / f'{name}.txt').write_text(proc.stdout)
    if proc.returncode:
        print(proc.stdout)
        raise SystemExit(f'{name} failed: exit {proc.returncode}')
    return proc.stdout


def main():
    REPORTS.mkdir(exist_ok=True)
    run('backprop-discovery-run', [sys.executable, '-B', 'tools/discover_backprop.py'])
    lean_sources = list((ROOT/'lean/OjasXMath').glob('*.lean'))
    ast.parse((ROOT/'tools/bootstrap_lean.py').read_text())
    expected = {n for p in lean_sources for n in re.findall(r'^theorem\s+(\w+)', p.read_text(), re.M)}
    audited = set(re.findall(r'^#print axioms OjasXMath\.(\w+)', (ROOT/'lean/OjasXMath.lean').read_text(), re.M))
    if expected != audited:
        raise SystemExit('Every project theorem must be present in the axiom audit')
    for p in lean_sources:
        if re.search(r'\b(sorry|admit|axiom)\b', re.sub(r'/\-.*?\-/', '', p.read_text(), flags=re.S)):
            raise SystemExit(f'Unexpected trust escape in {p.name}')
    lake = str(ROOT/'.tools/elan/bin/lake')
    version = run('lean-version', [str(ROOT/'.tools/elan/bin/lean'), '--version'], ROOT/'lean')
    run('lean-build', [lake, 'build'], ROOT/'lean')
    axioms = run('lean-axioms', [lake, 'env', 'lean', 'OjasXMath.lean'], ROOT/'lean')
    if 'sorryAx' in axioms or 'error:' in axioms:
        raise SystemExit('Lean axiom audit failed')
    # Only Lean's usual foundations are permitted; no custom assumptions.
    used = set()
    for group in re.findall(r'\[([^\]]*)\]', axioms):
        used.update(a.strip() for a in group.split(',') if a.strip())
    if used - {'propext', 'Classical.choice', 'Quot.sound'}:
        raise SystemExit(f'Unexpected axioms: {used}')
    discovery = json.loads((REPORTS/'backprop-discovery.json').read_text())
    discovery['status'] = 'passed_symbolic_and_lean_kernel_checks'
    (REPORTS/'backprop-discovery.json').write_text(json.dumps(discovery, indent=2)+'\n')
    run('feature-core-run', [sys.executable, '-B', 'tools/test_feature_core.py'])
    run('polynomial-roundoff-run', [sys.executable, '-B', 'tools/certify_polynomial_roundoff.py'])
    run('z3', [sys.executable, '-B', 'tools/verify_z3.py'])
    run('reversible-z3', [sys.executable, '-B', 'tools/verify_reversible_z3.py'])
    run('cached-reverse-z3', [sys.executable, '-B', 'tools/verify_cached_reverse_z3.py'])
    run('pytest', [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '--junitxml=reports/pytest.xml'])
    run('measurements', [sys.executable, '-B', 'tools/measure.py'])
    run('memory-backend-measurements', [sys.executable, '-B', 'tools/measure_backend.py'])
    run('math-structure', [sys.executable, '-B', 'tools/math_structure.py'])
    run('contraction-benchmark', [sys.executable, '-B', 'tools/bench_contractions.py'])
    run('contraction-training', [sys.executable, '-B', 'tools/train_contractions.py'])
    run('research-extensions', [sys.executable, '-B', 'tools/research_extensions.py'])
    run('radial-core-research', [sys.executable, '-B', 'tools/radial_core_research.py'])
    run('workload-scaling', [sys.executable, '-B', 'tools/scale_workloads.py'])
    run('reversible-research', [sys.executable, '-B', 'tools/reversible_research.py'])
    run('reversible-cache-cpu', [sys.executable, '-B', 'tools/reversible_cache_research.py', '--device', 'cpu'])
    xml = ET.parse(REPORTS/'pytest.xml')
    suites = xml.findall('testsuite')
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    source_paths = []
    for folder, pattern in [('ojasx2', '*.py'), ('tests', '*.py'), ('tools', '*.py'), ('lean/OjasXMath', '*.lean')]:
        source_paths.extend((ROOT/folder).glob(pattern))
    source_paths.extend(ROOT/p for p in ('pyproject.toml', 'lean/OjasXMath.lean', 'lean/lean-toolchain', 'lean/lakefile.toml', 'lean/lake-manifest.json'))
    source_paths.extend((ROOT/'tools').glob('*.m'))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
    result = {'status': 'passed', 'lean_version': version.strip(), 'checked_theorems': len(re.findall(r'^(?:theorem)\s', '\n'.join(p.read_text() for p in lean_sources), re.M)),
              'backprop_discovery': discovery,
              'feature_core': json.loads((REPORTS/'feature-core.json').read_text()),
              'polynomial_roundoff': json.loads((REPORTS/'polynomial-roundoff.json').read_text()),
              'reversible_z3': {'proofs':len(json.loads((REPORTS/'reversible-z3.json').read_text())['proofs']),
                                'refutations':len(json.loads((REPORTS/'reversible-z3.json').read_text())['counterexamples'])},
              'cached_reverse_z3': {'proofs':len(json.loads((REPORTS/'cached-reverse-z3.json').read_text())['proofs']),
                                   'refutations':len(json.loads((REPORTS/'cached-reverse-z3.json').read_text())['counterexamples'])},
              'axioms': sorted(used), 'pytest': totals, 'source_sha256': manifest,
              'bootstrap_validation': 'Python syntax only; clean download wrapper not rerun',
              'trust_boundary': 'Lean proves stated mathematics; differential tests relate Python to it. No formal Python/GPU code refinement proof.'}
    (REPORTS/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
