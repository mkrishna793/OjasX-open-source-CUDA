"""
ojasx2/cli.py

Command-Line Diagnostic & Codegen Controller for OjasX (`python -m ojasx2.cli` / `ojasx-verify`).
"""

from __future__ import annotations

import os
import sys
import torch
import ojasx2
from ojasx2.codegen.gen_holonomic_ops import main as run_codegen


def print_status() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    library_suffix = ".dylib" if sys.platform == "darwin" else (".dll" if os.name == "nt" else ".so")
    guard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "csrc", f"libojasx_openreg{library_suffix}"
    )
    ops = [op["name"] for op in ojasx2.HOLONOMIC_DECLARATIONS_LEDGER.get("operators", [])]
    dynamo_backends = [b for b in torch._dynamo.list_backends() if "holonomic" in b]

    # Quick live smoke check on the selected physical device.
    x = torch.randn(16, 64, device=device, requires_grad=True)
    g = torch.ones(64, device=device, requires_grad=True)
    y = ojasx2.holonomic_rmsnorm(x, g, 1e-6)
    y.sum().backward()
    grad_ok = (
        x.grad is not None
        and g.grad is not None
        and bool(torch.isfinite(x.grad).all())
        and bool(torch.isfinite(g.grad).all())
    )
    del x, g, y
    torch.ojasx.empty_cache()

    print("=" * 78)
    print(f"OjasX Holonomic Mathematical Engine — v{ojasx2.__version__}")
    print("=" * 78)
    print(f"  Physical Execution Device : {device.upper()}")
    print(f"  PyTorch Runtime Version   : {torch.__version__}")
    print(
        f"  OpenReg Guard Artifact    : {'PRESENT (not a load check)' if os.path.exists(guard_path) else 'NOT FOUND'} "
        f"({os.path.basename(guard_path)})"
    )
    print(f"  Registered Device Module  : torch.ojasx (available={torch.ojasx.is_available()})")
    print(f"  TorchDynamo Backends      : {dynamo_backends}")
    print(f"  Declarative YAML Ops      : {ops}")
    print(
        f"  Live {device.upper()} Smoke Check     : {'PASSED' if grad_ok else 'FAILED'} "
        f"(input/gamma gradients; OjasX residual memory: {torch.ojasx.memory_allocated() / 1e6:.2f} MB)"
    )
    print("=" * 78)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "codegen":
        run_codegen()
    else:
        print_status()


if __name__ == "__main__":
    main()
