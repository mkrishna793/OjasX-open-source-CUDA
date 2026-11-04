from tools import verify_static_and_symbolic as verifier


def test_nested_module_resolution_does_not_execute_imports() -> None:
    path = verifier.resolve_python_module("ojasx2.math.lattice")
    assert path == verifier.REPO_ROOT / "ojasx2" / "math" / "lattice.py"
    assert verifier.resolve_python_module("ojasx2.math") == verifier.REPO_ROOT / "ojasx2" / "math" / "__init__.py"


def test_missing_nested_module_is_rejected() -> None:
    assert verifier.resolve_python_module("ojasx2.math.not_a_real_module") is None


def test_module_definitions_cover_imports_and_assignment_targets() -> None:
    definitions = verifier.module_definitions(verifier.REPO_ROOT / "ojasx2" / "math" / "lattice.py")
    assert "torch" in definitions
    assert "nn" in definitions


def test_optional_native_export_inside_try_is_visible() -> None:
    definitions = verifier.module_definitions(
        verifier.REPO_ROOT / "ojasx2" / "math" / "csrc" / "__init__.py"
    )
    assert "ojasx_math_native" in definitions
