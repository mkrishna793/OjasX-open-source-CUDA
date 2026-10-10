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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dylib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "libojasx_openreg.dylib")
    ops = [op["name"] for op in ojasx2.HOLONOMIC_DECLARATIONS_LEDGER.get("operators", [])]
    dynamo_backends = [b for b in torch._dynamo.list_backends() if "holonomic" in b]

    # Quick live GPU verification
    x = torch.randn(16, 64, device=device, requires_grad=True)
    g = torch.ones(64, device=device, requires_grad=True)
    y = ojasx2.holonomic_rmsnorm(x, g, 1e-6)
    y.sum().backward()
    grad_ok = x.grad is not None and bool(torch.isfinite(x.grad).all())
    del x, g, y
    torch.ojasx.empty_cache()

    print("=" * 78)
    print(f"OjasX Holonomic Mathematical Engine — v{ojasx2.__version__}")
    print("=" * 78)
    print(f"  Physical Execution Device : {device.upper()}")
    print(f"  PyTorch Runtime Version   : {torch.__version__}")
    print(f"  C++ OpenReg Guard Dylib   : {'LOADED' if os.path.exists(dylib_path) else 'MISSING'} ({os.path.basename(dylib_path)})")
    print(f"  Registered Device Module  : torch.ojasx (available={torch.ojasx.is_available()})")
    print(f"  TorchDynamo Backends      : {dynamo_backends}")
    print(f"  Declarative YAML Ops      : {ops}")
    print(f"  Live GPU Smoke Check      : {'PASSED' if grad_ok else 'FAILED'} (Residual Mem: {torch.ojasx.memory_allocated() / 1e6:.2f} MB)")
    print("=" * 78)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "codegen":
        run_codegen()
    else:
        print_status()


if __name__ == "__main__":
    main()
