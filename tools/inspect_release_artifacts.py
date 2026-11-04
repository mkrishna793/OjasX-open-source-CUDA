#!/usr/bin/env python3
"""Inspect portable OjasX wheel/sdist artifacts without importing the source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = (".so", ".dylib", ".pyd", ".dll")
REQUIRED_WHEEL = (
    "ojasx2/native_build.json",
    "ojasx2/math/csrc/ojasx_math_bindings.cpp",
    "ojasx2/csrc/ojasx_tile_kernels.cpp",
)
REQUIRED_SDIST = (
    "ojasx2/math/csrc/ojasx_math_core.hpp",
    "third_party/tilelang/src/metal/codegen/codegen_metal.cc",
    "third_party/BackendBench/BackendBench/eval.py",
    "third_party/TorchCTS/pyproject.toml",
    "third_party/TorchCTS/torchcts/__init__.py",
    "lean/OjasXMath.lean",
    "lean/OjasXMath/Remediation.lean",
    "tests/test_verified_remediation.py",
    "tests/test_release_and_docs.py",
    "third_party/PROVENANCE.md",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def member_matches(names: list[str], required: str) -> bool:
    return any(name == required or name.endswith("/" + required) for name in names)


def assert_safe_archive_names(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, f"unsafe archive member: {name}"
        assert "/.git/" not in "/" + name and not name.endswith("/.git"), f"Git metadata included: {name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    wheels = sorted(artifact_dir.glob("*.whl"))
    sdists = sorted(artifact_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    assert len(sdists) == 1, f"expected exactly one sdist, found {sdists}"
    wheel, sdist = wheels[0], sdists[0]
    assert "py3-none-any" in wheel.name, f"portable wheel is not platform independent: {wheel.name}"

    with zipfile.ZipFile(wheel) as bundle:
        wheel_names = bundle.namelist()
        assert_safe_archive_names(wheel_names)
        native_binaries = [name for name in wheel_names if name.endswith(BINARY_SUFFIXES)]
        assert not native_binaries, f"portable wheel contains native binaries: {native_binaries}"
        for required in REQUIRED_WHEEL:
            assert member_matches(wheel_names, required), f"wheel lacks {required}"
        metadata = json.loads(bundle.read("ojasx2/native_build.json"))
        assert metadata["mode"] == "portable", metadata
        assert metadata["tile_library"] is None and metadata["math_library"] is None, metadata
        assert metadata["cpp_standard"] is None and metadata["torch_version"] is None, metadata

    with tarfile.open(sdist, "r:gz") as bundle:
        sdist_names = bundle.getnames()
        assert_safe_archive_names(sdist_names)
        native_sdist_binaries = [name for name in sdist_names if name.endswith(BINARY_SUFFIXES)]
        assert not native_sdist_binaries, f"sdist contains native binaries: {native_sdist_binaries}"
        for required in REQUIRED_SDIST:
            assert member_matches(sdist_names, required), f"sdist lacks {required}"

    component_counts = {
        component: sum(f"/third_party/{component}/" in "/" + name for name in sdist_names)
        for component in ("tilelang", "BackendBench", "TorchCTS")
    }
    assert all(count > 0 for count in component_counts.values()), component_counts
    result = {
        "schema_version": 1,
        "result": "pass",
        "checks": [
            "exactly one wheel and one sdist",
            "py3-none-any portable wheel tag",
            "no native binaries in portable artifacts",
            "portable native_build.json metadata",
            "safe archive member paths and no Git metadata",
            "required first-party and third-party source coverage",
        ],
        "artifacts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (wheel, sdist)
        ],
        "wheel_entries": len(wheel_names),
        "sdist_entries": len(sdist_names),
        "third_party_sdist_entries": component_counts,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        report = args.report if args.report.is_absolute() else ROOT / args.report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
