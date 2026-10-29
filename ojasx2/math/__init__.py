"""
ojasx2/math/__init__.py

Canonical Mathematical Subsystem for OjasX (Unified Package Export).
Exposes all 5 mathematical pillars plus reference block/residual algebra without shadowing.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict

import torch

# 1. Symplectic Manifold Mechanics
from .symplectic import (
    SymplecticManifold,
    StormerVerletIntegrator,
    symplectic_step,
    SymplecticSqueeze,
    symplectic_squeeze,
    audit_symplectic_reversibility,
)

# 2. Structured Low-Rank Adjoints & Reference Algebra
from .adjoints import (
    StructuredDerivative,
    diagonal_derivative,
    activation_derivative,
    softmax_derivative,
    rmsnorm_derivative,
    rmsnorm_input_derivative,
    layernorm_derivative,
    layernorm_input_derivative,
    NormalizationResidual,
    normalization_residual,
    softmax_pullback,
    rmsnorm_pullback,
    layernorm_pullback,
    rank7_reference,
    winograd_reference,
)

# 3. Halley-Padé 3rd-Order Root Inversion
from .inversion import (
    get_activation_min_z,
    pack_branch_bit,
    unpack_branch_bit,
    pack_mask,
    unpack_mask,
    halley_invert_activation,
    halley_activation_cotangent,
    HalleyInverseFunction,
    halley_activation,
)

# 4. Streaming Linear Operators
from .streaming import (
    tiled_dagger_attention,
    fused_linear_cross_entropy,
)

# 5. E8 Lattice Sphere Packing & Quantization
from .lattice import (
    fwht_block32,
    fast_walsh_hadamard_transform,
    randomized_hadamard_transform,
    conway_sloane_e8_project,
    E8LatticePacker,
    TernaryE8LatticePacker,
    RootE8LatticePacker,
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    pack_model_e8,
)


def has_native_backend() -> bool:
    if os.environ.get("OJASX_DISABLE_NATIVE") == "1":
        return False
    try:
        from ojasx2.math.csrc import ojasx_math_native
        if ojasx_math_native is not None:
            return True
    except Exception:
        pass
    try:
        from ojasx2.tile_hal import tile_kernel_status
        if tile_kernel_status().get("native_loaded", False):
            return True
    except Exception:
        pass
    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    ext = ".dylib" if sys.platform == "darwin" else (".so" if sys.platform.startswith("linux") else ".dll")
    tile_lib = csrc_dir / f"libojasx_tile_kernels{ext}"
    math_lib = Path(__file__).resolve().parent / "csrc" / f"ojasx_math_native{ext}"
    return tile_lib.exists() or math_lib.exists()


def get_native_backend() -> Dict[str, Any]:
    native_loaded = False
    tilelang_loaded = False
    math_native_loaded = False
    metal_available = False
    load_errors = []

    try:
        from ojasx2.math.csrc import ojasx_math_native
        if ojasx_math_native is not None:
            math_native_loaded = True
            if hasattr(ojasx_math_native, "is_metal_available"):
                metal_available = bool(ojasx_math_native.is_metal_available())
    except Exception as e:
        load_errors.append(f"math_native: {e}")

    try:
        from ojasx2.tile_hal import tile_kernel_status
        status = tile_kernel_status()
        native_loaded = bool(status.get("native_loaded", False))
        tilelang_loaded = bool(status.get("native_tilelang_loaded", False))
        load_errors.extend(list(status.get("load_errors", [])))
    except Exception as e:
        load_errors.append(f"tile_hal: {e}")

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    ext = ".dylib" if sys.platform == "darwin" else (".so" if sys.platform.startswith("linux") else ".dll")
    tile_lib = csrc_dir / f"libojasx_tile_kernels{ext}"
    openreg_lib = csrc_dir / f"libojasx_openreg{ext}"
    metal_mm = csrc_dir / "ojasx_tilelang_metal.mm"
    math_lib = Path(__file__).resolve().parent / "csrc" / f"ojasx_math_native{ext}"

    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_avail = torch.cuda.is_available()
    has_native = (
        (native_loaded or math_native_loaded or tile_lib.exists() or math_lib.exists())
        and os.environ.get("OJASX_DISABLE_NATIVE") != "1"
    )

    return {
        "has_native": has_native,
        "math_native_loaded": math_native_loaded,
        "metal_available": metal_available,
        "tile_kernels_loaded": native_loaded,
        "native_tilelang_loaded": tilelang_loaded,
        "libojasx_tile_kernels": tile_lib.exists(),
        "libojasx_openreg": openreg_lib.exists(),
        "libojasx_math_native": math_lib.exists(),
        "metal_kernel_source": metal_mm.exists(),
        "mps_available": mps_avail,
        "cuda_available": cuda_avail,
        "platform": sys.platform,
        "load_errors": load_errors,
    }


__all__ = [
    "SymplecticManifold",
    "StormerVerletIntegrator",
    "symplectic_step",
    "SymplecticSqueeze",
    "symplectic_squeeze",
    "audit_symplectic_reversibility",
    "StructuredDerivative",
    "diagonal_derivative",
    "activation_derivative",
    "softmax_derivative",
    "rmsnorm_derivative",
    "rmsnorm_input_derivative",
    "layernorm_derivative",
    "layernorm_input_derivative",
    "NormalizationResidual",
    "normalization_residual",
    "softmax_pullback",
    "rmsnorm_pullback",
    "layernorm_pullback",
    "rank7_reference",
    "winograd_reference",
    "get_activation_min_z",
    "pack_branch_bit",
    "unpack_branch_bit",
    "pack_mask",
    "unpack_mask",
    "halley_invert_activation",
    "halley_activation_cotangent",
    "HalleyInverseFunction",
    "halley_activation",
    "tiled_dagger_attention",
    "fused_linear_cross_entropy",
    "has_native_backend",
    "get_native_backend",
    "fwht_block32",
    "fast_walsh_hadamard_transform",
    "randomized_hadamard_transform",
    "conway_sloane_e8_project",
    "E8LatticePacker",
    "TernaryE8LatticePacker",
    "RootE8LatticePacker",
    "E8LatticeLinear",
    "E8QuantizationAwareLinear",
    "pack_model_e8",
]
