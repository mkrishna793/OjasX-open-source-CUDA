"""Offline tests for the pinned Lean bootstrap's trust boundary."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ojasx_bootstrap_lean", ROOT / "tools" / "bootstrap_lean.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_download_requires_pinned_size_and_sha256(tmp_path):
    content = b"pinned-elan-archive"
    source = tmp_path / "source"
    source.write_bytes(content)
    destination = tmp_path / "download"
    digest = hashlib.sha256(content).hexdigest()
    bootstrap._download_verified(source.as_uri(), destination, len(content), digest)
    assert destination.read_bytes() == content

    with pytest.raises(ValueError, match="SHA-256"):
        bootstrap._download_verified(source.as_uri(), destination, len(content), "0" * 64)
    assert not destination.exists()
    with pytest.raises(ValueError, match="exceeds"):
        bootstrap._download_verified(source.as_uri(), destination, len(content) - 1, digest)
    assert not destination.exists()


def _tar_archive(path: Path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = content.encode()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = content
                archive.addfile(info)


def test_archive_extraction_is_single_file_and_rejects_links_and_traversal(tmp_path):
    archive = tmp_path / "elan.tar.gz"
    _tar_archive(archive, [("release/readme", "docs", "file"), ("release/elan", "binary", "file")])
    destination = tmp_path / "bin" / "elan"
    bootstrap._extract_elan_binary(archive, destination)
    assert destination.read_bytes() == b"binary"
    assert destination.stat().st_mode & 0o111

    _tar_archive(archive, [("elan", "binary", "file"), ("escape", "../../outside", "symlink")])
    with pytest.raises(ValueError, match="Links"):
        bootstrap._extract_elan_binary(archive, destination)
    _tar_archive(archive, [("../elan", "binary", "file")])
    with pytest.raises(ValueError, match="Unsafe path"):
        bootstrap._extract_elan_binary(archive, destination)


def test_lean_and_mathlib_toolchains_are_aligned_and_lockfile_is_preserved():
    toolchain = (ROOT / "lean" / "lean-toolchain").read_text().strip()
    manifest = json.loads((ROOT / "lean" / "lake-manifest.json").read_text())
    mathlib = next(package for package in manifest["packages"] if package["name"] == "mathlib")
    assert toolchain == "leanprover/lean4:v4.34.0"
    assert mathlib["inputRev"] == "v4.34.0"
    source = (ROOT / "tools" / "bootstrap_lean.py").read_text()
    assert "elan-init.sh" not in source
    assert "urlretrieve" not in source
    assert '"update"' not in source
    assert len({entry[2] for entry in bootstrap._ASSETS.values()}) == 5
