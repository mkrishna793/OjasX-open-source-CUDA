"""
ojasx2/math

Canonical Mathematical Subsystem for OjasX
==========================================
Clean, audited mathematical formulations across the three core pillars:

1. Symplectic Manifold Mechanics (Sp(2D, R)):
   - 2nd-order Störmer-Verlet integrator with FSAL (First-Same-As-Last) composition.
   - Dual-stream Liouville volume preservation (det J = 1).
   - Persistent FP32/FP64 state accumulators bounding 64+ layer drift to < 1e-4.
   - Symplectic Squeeze scaling in Sp(2D, R) with dyadic power-of-2 exponent shifting.
   Exports:
       StormerVerletIntegrator, SymplecticManifold, symplectic_step, SymplecticSqueeze,
       symplectic_squeeze, audit_symplectic_reversibility

2. Structured Low-Rank Adjoints:
   - Unified StructuredDerivative: J = diag(d) + U V^T with O(n r) pushforward/pullback.
   - Softmax:   J = diag(p) - p p^T               (r=1, 4n FLOPs)
   - RMSNorm:   J = diag(g*r) - (g*r*z/n) z^T      (r=1, 4n FLOPs)
   - LayerNorm: J = diag(g*r) - u1 1^T - u2 z^T    (r=2, 8n FLOPs)
   Exports:
       StructuredDerivative, diagonal_derivative, softmax_derivative,
       rmsnorm_derivative, layernorm_derivative, softmax_pullback,
       rmsnorm_pullback, layernorm_pullback

3. Halley-Padé 3rd-Order Root Inversion & 1-Bit Sign Packing:
   - Halley's rational iteration achieving < 1e-7 error in 2-4 FMA steps.
   - 1-bit boolean sign packing (pack_branch_bit / unpack_branch_bit) giving 32x memory savings.
   - HalleyInverseFunction autograd operator with 0 bytes activation tape storage.
   - Supports SiLU, GELU (exact & tanh), QuickGELU, Sigmoid, Tanh, ReLU, LeakyReLU, ReLU2.
   Exports:
       halley_invert_activation, HalleyInverseFunction, pack_branch_bit,
       unpack_branch_bit, pack_mask, unpack_mask, halley_activation,
       halley_activation_cotangent, get_activation_min_z

4. Streaming Linear Operators:
   - Tiled Dagger Attention: Online softmax tiling with 0 bytes [B,H,S,S] DRAM materialization.
   - Fused Linear Cross-Entropy: Streaming loss directly from H and W without [M, V] logits in DRAM.
   Exports:
       tiled_dagger_attention, fused_linear_cross_entropy

5. Native C++/Metal Hardware Acceleration Hooks:
   - Probes native compiled C++ tile kernels and Metal shaders with graceful fallback.
   Exports:
       has_native_backend, get_native_backend
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

# 2. Structured Low-Rank Adjoints
from .adjoints import (
    StructuredDerivative,
    diagonal_derivative,
    softmax_derivative,
    rmsnorm_derivative,
    layernorm_derivative,
    softmax_pullback,
    rmsnorm_pullback,
    layernorm_pullback,
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
    fast_walsh_hadamard_transform,
    randomized_hadamard_transform,
    conway_sloane_e8_project,
    E8LatticePacker,
    TernaryE8LatticePacker,
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    pack_model_e8,
)


# =============================================================================
# 5. Native C++/Metal Acceleration Hooks
# =============================================================================

def has_native_backend() -> bool:
    """
    Query whether native C++ or Metal acceleration is loaded and operational.
    Returns True if either the compiled C++ tile kernel dylib, math native C++/Metal module,
    or OpenReg guard is loaded.
    """
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
        status = tile_kernel_status()
        if status.get("native_loaded", False):
            return True
    except Exception:
        pass

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    ext = ".dylib" if sys.platform == "darwin" else (".so" if sys.platform.startswith("linux") else ".dll")
    tile_lib = csrc_dir / f"libojasx_tile_kernels{ext}"
    math_lib = Path(__file__).resolve().parent / "csrc" / f"ojasx_math_native{ext}"
    return tile_lib.exists() or math_lib.exists()


def get_native_backend() -> Dict[str, Any]:
    """
    Return comprehensive telemetry on native C++, Metal, and hardware acceleration backends.
    """
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
    math_csrc_dir = Path(__file__).resolve().parent / "csrc"
    math_lib = math_csrc_dir / f"ojasx_math_native{ext}"

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
    # Symplectic
    "SymplecticManifold",
    "StormerVerletIntegrator",
    "symplectic_step",
    "SymplecticSqueeze",
    "symplectic_squeeze",
    "audit_symplectic_reversibility",
    # Adjoints
    "StructuredDerivative",
    "diagonal_derivative",
    "softmax_derivative",
    "rmsnorm_derivative",
    "layernorm_derivative",
    "softmax_pullback",
    "rmsnorm_pullback",
    "layernorm_pullback",
    # Inversion
    "get_activation_min_z",
    "pack_branch_bit",
    "unpack_branch_bit",
    "pack_mask",
    "unpack_mask",
    "halley_invert_activation",
    "halley_activation_cotangent",
    "HalleyInverseFunction",
    "halley_activation",
    # Streaming
    "tiled_dagger_attention",
    "fused_linear_cross_entropy",
    # Acceleration
    "has_native_backend",
    "get_native_backend",
    # E8 Lattice
    "fast_walsh_hadamard_transform",
    "randomized_hadamard_transform",
    "conway_sloane_e8_project",
    "E8LatticePacker",
    "TernaryE8LatticePacker",
    "E8LatticeLinear",
    "E8QuantizationAwareLinear",
    "pack_model_e8",
]
