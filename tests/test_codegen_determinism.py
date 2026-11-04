"""The code generator's check mode must be deterministic and read-only."""

import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ojasx2" / "codegen" / "gen_holonomic_ops.py"
CHECKED_IN = ROOT / "ojasx2" / "generated_holonomic_ops.py"


def _generate(destination: Path):
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _check(destination: Path):
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--check", "--output", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_check_is_read_only_and_reports_drift(tmp_path):
    destination = tmp_path / "generated.py"
    assert _generate(destination).returncode == 0
    original = destination.read_bytes()
    result = _check(destination)
    assert result.returncode == 0, result.stderr
    assert destination.read_bytes() == original

    destination.write_bytes(original + b"# drift\n")
    drifted = destination.read_bytes()
    result = _check(destination)
    assert result.returncode == 1
    assert "STALE" in result.stderr
    assert destination.read_bytes() == drifted

    missing = tmp_path / "missing" / "generated.py"
    assert _check(missing).returncode == 1
    assert not missing.exists()


def test_regeneration_is_idempotent_and_matches_checked_in_bytes(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    assert _generate(first).returncode == 0
    assert _generate(second).returncode == 0
    assert first.read_bytes() == second.read_bytes() == CHECKED_IN.read_bytes()
    digest = hashlib.sha256(CHECKED_IN.read_bytes()).hexdigest()
    result = _check(CHECKED_IN)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(CHECKED_IN.read_bytes()).hexdigest() == digest
