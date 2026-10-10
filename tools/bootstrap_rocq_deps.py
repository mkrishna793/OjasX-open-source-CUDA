"""Build isolated GMP/pkg-config dependencies for Rocq; no global installation."""
import hashlib
from pathlib import Path
import subprocess
import os
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.tools/rocq-deps'
SOURCES=ROOT/'.tools/rocq-sources'
PACKAGES=[
 ('gmp-6.3.0','https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz',
  'a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898',
  ['--disable-shared','--enable-static','--with-pic']),
 ('pkg-config-0.29.2','https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz',
  '6fc69c01688c9458a57eb9a1664c9aba372ccda420a02bf4429fe610e7e7d591',
  ['--with-internal-glib','--disable-host-tool'])]

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True);SOURCES.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,SDKROOT='/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk',
             CFLAGS='-O2 -Wno-error=implicit-function-declaration -Wno-error=int-conversion')
    for name,url,sha,flags in PACKAGES:
        archive=SOURCES/url.rsplit('/',1)[1]
        if not archive.exists():
            print('Downloading',url,flush=True)
            urllib.request.urlretrieve(url,archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=sha:raise RuntimeError('Source checksum mismatch: '+name)
        dest=SOURCES/name
        if not dest.exists():subprocess.run(['tar','xf',str(archive),'-C',str(SOURCES)],check=True)
        with (ROOT/f'reports/rocq-build-{name}.txt').open('w') as log:
            for command in (['./configure',f'--prefix={BASE}',*flags],['make','-j4'],['make','install']):
                print(name,command[0],flush=True)
                subprocess.run(command,cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Dependencies installed in',BASE)
