"""Audit the entire formal foundation; no CPU ML/extracted numerical tests."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports'
env=dict(os.environ,ELAN_HOME=str(ROOT/'.tools/elan'),PYTHONDONTWRITEBYTECODE='1')

def run(name,args,cwd=ROOT):
 print('Checking',name,flush=True)
 with (REPORTS/(name+'.txt')).open('w') as out:
  result=subprocess.run(args,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT)
 text=(REPORTS/(name+'.txt')).read_text()
 if result.returncode:raise RuntimeError(name+' failed: '+text[-3000:])
 return text

def main():
 run('spectral-discovery-run',[sys.executable,'-B','tools/discover_spectral_foundation.py'])
 ls=sorted((ROOT/'lean/OjasXMath').glob('*.lean'))
 names=[n for p in ls for n in re.findall(r'^theorem\s+(\w+)',p.read_text(),re.M)]
 if len(names)!=len(set(names)):raise RuntimeError('Duplicate theorem')
 for p in ls:
  source=re.sub(r'/\-.*?\-/|--[^\n]*','',p.read_text(),flags=re.S)
  if re.search(r'\b(sorry|admit|axiom|unsafe)\b',source):raise RuntimeError('Trust escape '+str(p))
 audit=''.join('import OjasXMath.'+p.stem+'\n' for p in ls)+'\n'+''.join('#print axioms OjasXMath.'+n+'\n' for n in names)
 (ROOT/'lean/OjasXMath.lean').write_text(audit)
 lake=str(ROOT/'.tools/elan/bin/lake')
 run('spectral-full-lean-build',[lake,'build'],ROOT/'lean')
 out=run('spectral-full-lean-axioms',[lake,'env','lean','OjasXMath.lean'],ROOT/'lean')
 groups=re.findall(r'depends on axioms:\s*\[([^\]]*)\]',out)
 used={a.strip() for group in groups for a in group.split(',') if a.strip()}
 if len(re.findall(r'depends on axioms:|does not depend on any axioms',out))!=len(names) or used-{'propext','Classical.choice','Quot.sound'}:raise RuntimeError('Lean audit incomplete or unexpected axioms')
 mods=[p[:-2] for p in (ROOT/'rocq/_RocqProject').read_text().splitlines() if p.endswith('.v') and p not in ('Audit.v','Extract.v')]
 rnames=[(m,n) for m in mods for n in re.findall(r'^(?:Theorem|Lemma)\s+(\w+)',(ROOT/'rocq'/f'{m}.v').read_text(),re.M)]
 audit='From Ojasx Require Import '+' '.join(mods)+'.\n\n'+''.join(f'Print Assumptions {m}.{n}.\n' for m,n in rnames)
 (ROOT/'rocq/Audit.v').write_text(audit)
 for mod in mods:
  source=(ROOT/'rocq'/f'{mod}.v').read_text()
  if re.search(r'\b(?:Admitted|admit|Axiom|Parameter|Abort)\b',source):raise RuntimeError('Unfinished Rocq proof')
  run('spectral-rocq-'+mod,[str(ROOT/'tools/rocq.sh'),'rocq','compile','-Q','.','Ojasx',mod+'.v'],ROOT/'rocq')
 out=run('spectral-rocq-audit',[str(ROOT/'tools/rocq.sh'),'rocq','compile','-Q','.','Ojasx','Audit.v'],ROOT/'rocq')
 if out.count('Closed under the global context')!=len(rnames) or 'Axioms:' in out:raise RuntimeError('Rocq assumptions incomplete')
 run('spectral-rocq-kernel',[str(ROOT/'tools/rocq.sh'),'rocq','check','-Q','.','Ojasx',*['Ojasx.'+m for m in mods]],ROOT/'rocq')
 discovery=json.loads((REPORTS/'spectral-discovery.json').read_text())
 discovery['status']='passed_symbolic_and_lean_rocq_kernel_checks'
 (REPORTS/'spectral-discovery.json').write_text(json.dumps(discovery,indent=2)+'\n')
 tracked=[*ls,ROOT/'lean/OjasXMath.lean',ROOT/'lean/lake-manifest.json',ROOT/'lean/lean-toolchain',*(ROOT/'rocq').glob('*.v'),ROOT/'rocq/_RocqProject',*(ROOT/'ojasx2').glob('*.py'),*(ROOT/'tools').glob('*.py'),*(ROOT).glob('*.md'),ROOT/'foundation_manifest.json',ROOT/'rocq/README.md']
 result={'status':'passed_formal_checks','lean_theorems':len(names),'standard_lean_axioms':sorted(used),'added_lean_axioms':[],
  'rocq_statements':len(rnames),'rocq_added_axioms':[],'independent_rocq_kernel_recheck':True,
  'new_z3_checks':len(discovery['z3_checks']),'shape_decisions':len(discovery['shape_search']),'cpu_ml_execution':False,
  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(tracked))},
  'scope':'Entire project proof audit; arbitrary finite polynomial tensor projection and all-gradient error contracts, matched polynomial derivatives, Chebyshev reference-tail bounds and bias folding. No formal Python/Metal/IEEE refinement, global optimality, or timing proof.'}
 (REPORTS/'spectral-formal-verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PASS',len(names),'Lean',len(rnames),'Rocq',len(discovery['z3_checks']),'new Z3 checks',flush=True)

if __name__=='__main__':main()
