"""Repeatable bounded discovery -> kernel audit -> GPU -> admission loop.
No numerical CPU fallback. No background automation is installed.
"""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--gpu',action='store_true',help='Execute MPS tests, fail if GPU unavailable')
 parser.add_argument('--qualify-existing-gpu',action='store_true',help='Validate current source hashes against existing GPU evidence')
 args=parser.parse_args()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
 def run(script):subprocess.run([sys.executable,'-B','tools/'+script],cwd=ROOT,env=env,check=True)
 run('verify_spectral_foundation.py')
 if args.gpu:
  run('test_spectral_gpu.py')

 if args.gpu or args.qualify_existing_gpu:run('qualify_spectral_foundation.py')

if __name__=='__main__':main()
