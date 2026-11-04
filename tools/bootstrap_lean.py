"""Install the pinned Lean/mathlib toolchain without executing a remote script."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ELAN_VERSION = "4.2.4"
_RELEASE = f"https://github.com/leanprover/elan/releases/download/v{ELAN_VERSION}"

# Digests published by GitHub for the immutable v4.2.4 release assets.
_ASSETS = {
    ("Linux", "x86_64"): ("elan-x86_64-unknown-linux-gnu.tar.gz", 4_996_571, "42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63"),
    ("Linux", "aarch64"): ("elan-aarch64-unknown-linux-gnu.tar.gz", 5_131_114, "05febd124d84ebf994b2e7479922a5650b1e950c17ae3bd1ddd776b65bb72bf9"),
    ("Linux", "arm64"): ("elan-aarch64-unknown-linux-gnu.tar.gz", 5_131_114, "05febd124d84ebf994b2e7479922a5650b1e950c17ae3bd1ddd776b65bb72bf9"),
    ("Darwin", "x86_64"): ("elan-x86_64-apple-darwin.tar.gz", 2_330_818, "8a340b309d8ed2e96f930761fa223b3af57a38f5d253b53ac90293c9516f8cd4"),
    ("Darwin", "arm64"): ("elan-aarch64-apple-darwin.tar.gz", 2_190_845, "7ad829861392c718dfebde3a83b5c8508df47be02af68894b094b0b3952616e5"),
    ("Windows", "x86_64"): ("elan-x86_64-pc-windows-msvc.zip", 2_508_099, "fad2e980a191c15884cc1d80d170ffc5fa84f3774541020145b66d1a644c6111"),
    ("Windows", "AMD64"): ("elan-x86_64-pc-windows-msvc.zip", 2_508_099, "fad2e980a191c15884cc1d80d170ffc5fa84f3774541020145b66d1a644c6111"),
}


def _download_verified(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": f"OjasX-Lean-Bootstrap/{ELAN_VERSION}"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > expected_size:
                    raise ValueError("Downloaded elan archive exceeds its pinned size")
                digest.update(chunk)
                output.write(chunk)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError("Downloaded elan archive failed its pinned size/SHA-256 check")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe path in elan archive: {name!r}")
    return path


def _extract_elan_binary(archive: Path, destination: Path) -> None:
    """Extract only the elan executable; reject links and traversal entries."""
    executable_name = "elan.exe" if archive.suffix == ".zip" else "elan"
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            candidates = []
            for info in bundle.infolist():
                path = _safe_archive_name(info.filename)
                if info.is_dir():
                    continue
                # Unix symlinks can be encoded in ZIP external attributes.
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Links are not accepted in the elan archive")
                if path.name == executable_name:
                    candidates.append(info)
            if len(candidates) != 1:
                raise ValueError("Pinned elan archive does not contain exactly one executable")
            data = bundle.read(candidates[0])
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            candidates = []
            for member in bundle.getmembers():
                _safe_archive_name(member.name)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError("Links and special files are not accepted in the elan archive")
                if member.isfile() and PurePosixPath(member.name).name == executable_name:
                    candidates.append(member)
            if len(candidates) != 1:
                raise ValueError("Pinned elan archive does not contain exactly one executable")
            extracted = bundle.extractfile(candidates[0])
            if extracted is None:
                raise ValueError("Unable to read the pinned elan executable")
            data = extracted.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o755)
    temporary.replace(destination)


def _ensure_elan_shims(elan: Path) -> None:
    # elan is a rustup-style multicall binary; these names are local shims.
    for command in ("lean", "lake", "leanc", "leanpkg"):
        shim = elan.with_name(command + elan.suffix)
        if shim.exists() or shim.is_symlink():
            shim.unlink()
        try:
            shim.symlink_to(elan.name)
        except OSError:
            shutil.copy2(elan, shim)


def _install_elan(tools: Path) -> Path:
    key = (platform.system(), platform.machine())
    if key not in _ASSETS:
        raise RuntimeError(f"No pinned elan binary is configured for {key[0]}/{key[1]}")
    filename, size, sha256 = _ASSETS[key]
    elan = tools / "elan" / "bin" / ("elan.exe" if key[0] == "Windows" else "elan")
    if elan.exists():
        try:
            version = subprocess.check_output([str(elan), "--version"], text=True, stderr=subprocess.STDOUT)
            if ELAN_VERSION in version:
                _ensure_elan_shims(elan)
                return elan
        except (OSError, subprocess.SubprocessError):
            pass
    with tempfile.TemporaryDirectory(prefix="elan-download-", dir=tools) as temporary:
        archive = Path(temporary) / filename
        _download_verified(f"{_RELEASE}/{filename}", archive, size, sha256)
        _extract_elan_binary(archive, elan)
    _ensure_elan_shims(elan)
    return elan


def main() -> None:
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    env = dict(os.environ, ELAN_HOME=str(tools / "elan"), MATHLIB_CACHE_DIR=str(tools / "mathlib-cache"))
    toolchain = (ROOT / "lean" / "lean-toolchain").read_text().strip()
    if not re.fullmatch(r"leanprover/lean4:v\d+\.\d+\.\d+", toolchain):
        raise ValueError(f"Unexpected Lean toolchain identifier: {toolchain!r}")
    elan = _install_elan(tools)
    subprocess.run([str(elan), "toolchain", "install", toolchain], env=env, check=True)
    subprocess.run([str(elan), "default", toolchain], env=env, check=True)
    # lake-manifest.json is the lockfile. Do not run `lake update`, which would
    # deliberately move dependencies whose upstream branches are named main/master.
    lake = str(elan.with_name("lake" + elan.suffix))
    imports = sorted({
        imp for source in (ROOT / "lean").rglob("*.lean")
        if ".lake" not in source.parts
        for imp in re.findall(r"^import (Mathlib\.[\w.]+)", source.read_text(), re.M)
    })
    subprocess.run([lake, "exe", "cache", "get", *imports], cwd=ROOT / "lean", env=env, check=True)
    print("Pinned Lean/mathlib installation complete. Run tools/verify_all.py next.")


if __name__ == "__main__":
    main()
