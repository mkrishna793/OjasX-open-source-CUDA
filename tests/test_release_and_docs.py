from __future__ import annotations

from pathlib import Path
import struct

import pytest

import ojasx2
from ojasx2.cli import print_status
from ojasx2.doctor import print_doctor_report
from tools import check_markdown_links

ROOT = Path(__file__).resolve().parents[1]


def test_first_party_markdown_links_resolve() -> None:
    broken = []
    for path in check_markdown_links.markdown_files():
        _, failures = check_markdown_links.check_file(path)
        broken.extend(failures)
    assert broken == []


def test_link_checker_rejects_missing_file_but_not_latex(tmp_path: Path) -> None:
    document = tmp_path / "example.md"
    document.write_text("[missing](not-present.json)\n\n\\[R[f](x,\\bar y)\\]\n")
    checked, broken = check_markdown_links.check_file(document)
    assert checked == 1
    assert [entry["target"] for entry in broken] == ["not-present.json"]


@pytest.mark.parametrize("name", ["architecture.png", "breaking_monopoly.png", "hero.png", "default.profraw"])
def test_documentation_assets_are_real_pngs(name: str) -> None:
    assert not (ROOT / "assets" / name).exists()
    assert not (ROOT / name).exists()


def test_readme_version_matches_package() -> None:
    first_line = (ROOT / "README.md").read_text().splitlines()[0]
    assert f"v{ojasx2.__version__}" in first_line


def test_cli_labels_cpu_smoke_with_actual_device(capsys: pytest.CaptureFixture[str]) -> None:
    print_status()
    output = capsys.readouterr().out
    if not (ojasx2.torch.cuda.is_available() or ojasx2.torch.backends.mps.is_available()):
        assert "Physical Execution Device : CPU" in output
        assert "Live CPU Smoke Check" in output
        assert "Live GPU Smoke Check" not in output
    assert "input/gamma gradients" in output


def _silicon_info() -> dict[str, object]:
    return {
        "ojasx_version": ojasx2.__version__,
        "os": "test",
        "python_version": "test",
        "torch_version": ojasx2.torch.__version__,
        "cpu_count": 1,
        "l1d_kb": 0,
        "l2_mb": 0,
        "has_apple_amx": False,
        "mps_available": False,
        "total_ram_gb": 0,
        "libojasx_tile_kernels": False,
        "diagnostic_device": "cpu",
    }


def _result(index: int) -> dict[str, object]:
    return {
        "family": f"family-{index}",
        "profile": "test",
        "fwd_parity": True,
        "grad_parity": True,
        "speedup": 1.0,
        "status": "PASS",
    }


def test_doctor_success_is_complete_scoped_and_device_specific(capsys: pytest.CaptureFixture[str]) -> None:
    assert print_doctor_report(_silicon_info(), [_result(i) for i in range(5)])
    output = capsys.readouterr().out
    assert "SAMPLED DIAGNOSTIC PASSED: 5 of 5 workload families on cpu" in output
    assert "within the configured tolerances" in output
    assert "full tests, formal kernels, native extensions, and performance qualification were not run" in output
    assert "ALL SYSTEMS OPERATIONAL" not in output


@pytest.mark.parametrize(
    "results",
    [
        [],
        [_result(i) for i in range(4)],
        [_result(i) for i in range(5)],
    ],
)
def test_doctor_rejects_empty_partial_or_inconsistent_results(
    results: list[dict[str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    if len(results) == 5:
        results[2]["fwd_parity"] = False
    assert not print_doctor_report(_silicon_info(), results)
    output = capsys.readouterr().out
    assert "SAMPLED DIAGNOSTIC PASSED" not in output
    assert "divergent, inconsistent, or incomplete" in output


def test_ci_runs_complete_suite_without_pushing_logs() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "submodules: recursive" in workflow
    assert "python -m pytest -q" in workflow
    assert "tests/test_v011_5_findings.py" not in workflow
    assert "git push" not in workflow
    assert "contents: read" in workflow


def test_release_workflow_installs_runtime_before_artifact_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    runtime = workflow.index("Install build tools and declared runtime dependencies")
    portable_smoke = workflow.index("Smoke-test exact portable wheel and sdist")
    native_build = workflow.index("Build native Linux wheel")
    assert runtime < portable_smoke < native_build
    assert "inspect_release_artifacts.py" in workflow
    assert "smoke_release_artifacts.py" in workflow
