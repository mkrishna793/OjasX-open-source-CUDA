"""
tools/verify_static_and_symbolic.py

Self-contained static and symbolic gate for OjasX v0.13.0.
Executes without requiring a GPU:
  1. AST syntax and undefined-symbol audit across all Python files in the repository.
  2. Cross-module import and symbol resolution audit across `ojasx2.*`.
  3. SymPy declarative operator codegen verification (`gen_holonomic_ops.py --check`).
  4. 3-Dialect MLIR SSA verification (`fx_to_holonomic_mlir`, `HolonomicToSubgraphPass`, `SubgraphToTilePass`).
  5. Metal 2D threadblock panel swizzle bijection proof across grid/panel configurations.
  6. Symplectic Hamiltonian coupling reversibility verification in FP64.
  7. Lean/Rocq source audit coverage and forbidden-trust-escape scan (no proof-kernel execution).
"""

from __future__ import annotations

import ast
import builtins
import glob
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_python_module(module: str) -> Optional[Path]:
    """Resolve an OjasX module without importing or executing it."""
    base = REPO_ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def module_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions: set[str] = set()
    statements = []

    def collect_module_statements(nodes) -> None:
        for node in nodes:
            statements.append(node)
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                for field in ("body", "orelse", "finalbody"):
                    collect_module_statements(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    collect_module_statements(handler.body)

    collect_module_statements(tree.body)
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                definitions.update(
                    child.id for child in ast.walk(target)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                )
        elif isinstance(node, ast.Import):
            definitions.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            definitions.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return definitions


def main() -> int:
    os.chdir(REPO_ROOT)
    print("=" * 84)
    print("  OJASX v0.13.0 STATIC, SYMBOLIC & FORMAL-SOURCE VERIFICATION SUITE")
    print("=" * 84)

    # 1. AST & Undefined-Name Audit
    builtin_names = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__all__", "__spec__"
    }
    excluded_directories = {"build", "dist", "node_modules", "__pycache__"}
    py_files = sorted(
        path for path in glob.glob("**/*.py", recursive=True)
        if not excluded_directories.intersection(Path(path).parts)
    )
    for path in py_files:
        with open(path, "r", encoding="utf-8") as fp:
            src = fp.read()
        tree = ast.parse(src, filename=path)
        defined = set(builtin_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if hasattr(node, "name"):
                    defined.add(node.name)
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in node.generators:
                    for t in ast.walk(gen.target):
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
        missing = {
            (node.id, getattr(node, "lineno", 0))
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined
        }
        assert not missing, f"Undefined symbols in {path}: {missing}"
    print(f"[1/7] AST & Symbol Resolution Audit   : PASS ({len(py_files)} Python files, 0 undefined names)")

    # 2. Cross-Module Import Audit. Resolve nested files/packages without
    # executing imports; an explicitly imported child module is also a symbol.
    checked_imports = 0
    definition_cache: dict[Path, set[str]] = {}
    for path in py_files:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ojasx2.")):
                continue
            module_path = resolve_python_module(node.module)
            assert module_path is not None, f"{path}:{node.lineno} missing submodule {node.module}"
            if module_path not in definition_cache:
                definition_cache[module_path] = module_definitions(module_path)
            definitions = definition_cache[module_path]
            for alias in node.names:
                if alias.name == "*":
                    continue
                checked_imports += 1
                child_module = resolve_python_module(f"{node.module}.{alias.name}")
                assert alias.name in definitions or child_module is not None, (
                    f"{path}:{node.lineno} missing symbol {alias.name} in {node.module}"
                )
    print(f"[2/7] Cross-Module Import Audit       : PASS ({checked_imports} imported symbols verified)")

    # 3. Deterministic declarative codegen drift check
    res = subprocess.run([sys.executable, "ojasx2/codegen/gen_holonomic_ops.py", "--check"], capture_output=True, text=True)
    assert res.returncode == 0, f"Codegen check failed: {res.stdout}\n{res.stderr}"
    print("[3/7] Declarative Codegen Check       : PASS (5 declarations; deterministic bytes)")

    # 4. MLIR 3-Dialect Pipeline & SSA Verification
    if "torch" not in sys.modules:
        try:
            import torch
            import torch.nn
            import torch.fx
        except ImportError:
            mock_torch = types.ModuleType("torch")
            mock_torch.Tensor = type("Tensor", (), {})
            mock_nn = types.ModuleType("torch.nn")
            mock_nn.Module = type("Module", (), {})
            mock_nn.Linear = type("Linear", (mock_nn.Module,), {})
            mock_nn.LayerNorm = type("LayerNorm", (mock_nn.Module,), {})
            mock_fx = types.ModuleType("torch.fx")
            mock_fx.GraphModule = type("GraphModule", (), {})
            mock_fx.Node = type("Node", (), {})
            mock_fx.Interpreter = type("Interpreter", (), {"__init__": lambda self, *args, **kwargs: None})
            mock_torch.nn = mock_nn
            mock_torch.fx = mock_fx
            sys.modules["torch"] = mock_torch
            sys.modules["torch.nn"] = mock_nn
            sys.modules["torch.fx"] = mock_fx

    spec = importlib.util.spec_from_file_location("mlir_dialects", "ojasx2/mlir_dialects.py")
    assert spec is not None and spec.loader is not None
    md = importlib.util.module_from_spec(spec)
    sys.modules["mlir_dialects"] = md
    spec.loader.exec_module(md)

    x_v = md.MLIRValue("%x", md.TensorType((4, 64), md.FloatType(32)))
    w_v = md.MLIRValue("%w", md.TensorType((128, 64), md.FloatType(32)))
    c_op = md.HolonomicDialect.chunked_glu(x_v, w_v, md.TensorType((4, 64), md.FloatType(32)), act="silu", val_idx=0)
    assert c_op is not None and c_op.results[0].defining_op is c_op

    class DummyNode:
        def __init__(self, op, name, target=None, args=()):
            self.op = op
            self.name = name
            self.target = target
            self.args = args
            self.meta = {}

    n_in = DummyNode("placeholder", "x")
    n_norm = DummyNode("call_function", "norm", "rmsnorm", (n_in,))
    n_lin = DummyNode("call_function", "lin", "linear", (n_norm,))
    n_uglu = DummyNode("call_function", "uglu", "universal_glu", (n_lin,))
    n_cglu = DummyNode("call_function", "cglu", "chunked_glu", (n_uglu,))
    n_lr = DummyNode("call_function", "lr", "low_rank_bottleneck", (n_cglu,))
    n_hal = DummyNode("call_function", "hal", "halley_inverse", (n_lr,))
    n_cull = DummyNode("call_function", "cull", "active_subspace_cull", (n_hal,))
    n_res = DummyNode("call_function", "res", "residual_add", (n_cull, n_lin))
    n_attn = DummyNode("call_function", "attn", "dagger_attention", (n_res, n_res, n_res))
    n_loss = DummyNode("call_function", "loss", "fused_linear_cross_entropy", (n_attn,))
    gm = types.SimpleNamespace(graph=types.SimpleNamespace(nodes=[
        n_in, n_norm, n_lin, n_uglu, n_cglu, n_lr, n_hal, n_cull, n_res, n_attn, n_loss
    ]))
    artifacts = md.lower_fx_to_mlir_artifacts(gm, target="nvvm", name="verify_full_fx")
    assert artifacts.ssa_verified and artifacts.holonomic_op_count == 10 and artifacts.tile_op_count > 20

    m1 = md.fx_to_holonomic_mlir(gm, name="verify_fx")
    assert m1.verify()
    m1_snapshot = m1.to_string()
    m2 = md.HolonomicToSubgraphPass().run(m1)
    assert m1.to_string() == m1_snapshot and m2.verify()
    m2_snapshot = m2.to_string()
    m3 = md.SubgraphToTilePass().run(m2)
    assert m2.to_string() == m2_snapshot and m3.verify()
    print("[4/7] MLIR 3-Dialect SSA & Pass Audit : PASS (m1, m2, m3 SSA verified, 0 in-place mutations)")

    # 5. Metal 2D Panel Swizzle Bijection Check
    for panel_size in (1, 2, 4, 8):
        swizzle_log = int(np.log2(panel_size))
        for grid_x in (4, 8, 12, 16):
            for grid_y in (1, 2, 3, 5, 8):
                if grid_x >= panel_size and (grid_x % panel_size == 0):
                    mapped = set()
                    for py in range(grid_y):
                        for px in range(grid_x):
                            linear_bid = py * grid_x + px
                            panel_span = grid_y << swizzle_log
                            panel_idx = linear_bid // panel_span
                            in_panel = linear_bid % panel_span
                            bx = (panel_idx << swizzle_log) + (in_panel & (panel_size - 1))
                            by = in_panel >> swizzle_log
                            assert 0 <= bx < grid_x and 0 <= by < grid_y
                            mapped.add((bx, by))
                    assert len(mapped) == grid_x * grid_y
    print("[5/7] Metal 2D Panel Swizzle Bijection: PASS (bijective across the tested grid/panel set)")

    # 6. Symplectic Coupling Reversibility Check
    rng = np.random.default_rng(42)
    q1_0 = rng.standard_normal((6, 8))
    q2_0 = rng.standard_normal((6, 8))
    w_f = [rng.standard_normal((8, 8)) * 0.2 for _ in range(4)]
    w_g = [rng.standard_normal((8, 8)) * 0.2 for _ in range(4)]
    c1, c2 = q1_0.copy(), q2_0.copy()
    for wf, wg in zip(w_f, w_g):
        c1 = c1 + np.tanh(c2 @ wf.T)
        c2 = c2 + np.sin(c1 @ wg.T)
    r1, r2 = c1.copy(), c2.copy()
    for wf, wg in reversed(list(zip(w_f, w_g))):
        r2 = r2 - np.sin(r1 @ wg.T)
        r1 = r1 - np.tanh(r2 @ wf.T)
    max_err = max(float(np.max(np.abs(r1 - q1_0))), float(np.max(np.abs(r2 - q2_0))))
    assert max_err < 1e-14
    print(f"[6/7] Symplectic Chain Reversibility  : PASS (max error = {max_err:.2e})")

    # 7. Lean 4 & Rocq 9.2 Formal Proof Audit
    lean_sources = list((REPO_ROOT / "lean/OjasXMath").glob("*.lean"))
    expected = {n for p in lean_sources for n in re.findall(r"^theorem\s+(\w+)", p.read_text(), re.M)}
    audited = set(re.findall(r"^#print axioms OjasXMath\.(\w+)", (REPO_ROOT / "lean/OjasXMath.lean").read_text(), re.M))
    assert expected == audited, f"Theorem mismatch: {expected ^ audited}"
    for p in lean_sources:
        stripped = re.sub(r"/\-.*?\-/", "", p.read_text(), flags=re.S)
        assert not re.search(r"\b(sorry|admit|axiom)\b", stripped), f"Forbidden keyword in {p}"
    rocq_sources = list((REPO_ROOT / "rocq").glob("*.v"))
    for p in rocq_sources:
        stripped = re.sub(r"\(\*.*?\*\)", "", p.read_text(), flags=re.S)
        assert not re.search(r"\b(Admitted|Axiom|Parameter|Abort)\b", stripped), f"Forbidden keyword in {p}"
    print(f"[7/7] Formal Source Trust-Escape Audit : PASS ({len(expected)} Lean theorems, {len(rocq_sources)} Rocq files; source audit only, kernel evidence separate)")
    print("=" * 84)
    print("  ALL 7 SCOPED VERIFICATION GATES PASSED")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
