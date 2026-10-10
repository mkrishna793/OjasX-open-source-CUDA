"""Install the pinned Lean/mathlib toolchain locally; requires network access."""
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    tools = ROOT/'.tools'
    tools.mkdir(exist_ok=True)
    installer = tools/'elan-init.sh'
    url = 'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh'
    urllib.request.urlretrieve(url, installer)
    env = dict(os.environ, ELAN_HOME=str(tools/'elan'), MATHLIB_CACHE_DIR=str(tools/'mathlib-cache'))
    toolchain = (ROOT/'lean/lean-toolchain').read_text().strip()
    subprocess.run(['sh', str(installer), '-y', '--no-modify-path', '--default-toolchain', toolchain], env=env, check=True)
    lake = str(tools/'elan/bin/lake')
    subprocess.run([lake, 'update'], cwd=ROOT/'lean', env=env, check=True)
    imports = sorted({imp for source in (ROOT/'lean').rglob('*.lean')
                      if '.lake' not in source.parts
                      for imp in re.findall(r'^import (Mathlib\.[\w.]+)', source.read_text(), re.M)})
    subprocess.run([lake, 'exe', 'cache', 'get', *imports], cwd=ROOT/'lean', env=env, check=True)
    print('Local pinned Lean/mathlib installation complete. Run tools/verify_all.py next.')


if __name__ == '__main__':
    main()
