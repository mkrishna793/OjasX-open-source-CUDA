"""OjasX C++20 and Metal native binary kernels."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

_CSRC_DIR = Path(__file__).resolve().parent

# Automatically load native shared object if present
_candidates = [
    _CSRC_DIR / "ojasx_math_native.so",
    _CSRC_DIR / f"ojasx_math_native{sys.implementation.cache_tag}.so" if hasattr(sys.implementation, "cache_tag") else None,
]
# Add platform suffix candidate
import sysconfig
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
if ext_suffix:
    _candidates.insert(0, _CSRC_DIR / f"ojasx_math_native{ext_suffix}")

for cand in _candidates:
    if cand and cand.exists():
        try:
            torch.ops.load_library(str(cand))
        except Exception:
            pass
        break

try:
    from . import ojasx_math_native  # type: ignore
except ImportError:
    ojasx_math_native = None
