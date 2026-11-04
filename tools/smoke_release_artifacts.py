#!/usr/bin/env python3
"""Install and smoke-test exact portable artifacts outside the source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MARKER = "OJASX_PORTABLE_SMOKE="
SMOKE_PROGRAM = r'''
import json
from pathlib import Path
import sys
from importlib.resources import files

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import ojasx2
import torch

module_path = Path(ojasx2.__file__).resolve()
assert module_path.is_relative_to(target), (module_path, target)
metadata = json.loads(files("ojasx2").joinpath("native_build.json").read_text())
assert metadata["mode"] == "portable", metadata
assert metadata["tile_library"] is None and metadata["math_library"] is None, metadata
x = torch.randn(4, 8, requires_grad=True)
g = torch.ones(8, requires_grad=True)
y = ojasx2.holonomic_rmsnorm(x, g, 1e-6)
y.sum().backward()
assert x.grad is not None and g.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(g.grad).all()
print("OJASX_PORTABLE_SMOKE=" + json.dumps({
    "module_path": str(module_path),
    "version": ojasx2.__version__,
    "torch_version": torch.__version__,
    "portable_metadata": metadata,
    "forward_backward": "pass",
}))
'''


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, OJASX_BUILD_NATIVE="0")
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def smoke_artifact(python: str, artifact: Path, workspace: Path) -> dict[str, object]:
    target = workspace / "site"
    target.mkdir()
    run([
        python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
        "--target", str(target), str(artifact),
    ], workspace)
    result = run([python, "-I", "-c", SMOKE_PROGRAM, str(target)], workspace)
    marked = [line for line in result.stdout.splitlines() if line.startswith(MARKER)]
    assert len(marked) == 1, f"expected one smoke result, got: {result.stdout}\n{result.stderr}"
    details = json.loads(marked[0][len(MARKER):])
    return {
        "artifact": artifact.name,
        "sha256": sha256(artifact),
        "bytes": artifact.stat().st_size,
        "result": "pass",
        **details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python with declared runtime dependencies")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    artifacts = sorted(artifact_dir.glob("*.whl")) + sorted(artifact_dir.glob("*.tar.gz"))
    assert len(artifacts) == 2, f"expected one wheel and one sdist, found {artifacts}"
    results = []
    with tempfile.TemporaryDirectory(prefix="ojasx-release-smoke-") as temporary:
        for index, artifact in enumerate(artifacts):
            workspace = Path(temporary) / str(index)
            workspace.mkdir()
            results.append(smoke_artifact(args.python, artifact, workspace))
    report = {
        "schema_version": 1,
        "result": "pass",
        "scope": "isolated import, portable metadata, and RMSNorm forward/backward on the runtime's default CPU device",
        "outside_source_tree": True,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        destination = args.report if args.report.is_absolute() else ROOT / args.report
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
