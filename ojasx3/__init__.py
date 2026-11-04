"""
OjasX3: Universal Turnkey AI Training & Mathematical Execution Engine
===================================================================

Release v0.16.0 — Modular, Turnkey, Drop-In, Multi-Hardware AI System
"""

__version__ = "0.16.0"

# Turnkey public 1-line APIs
from .api import compile, train, finetune, optimize, prepare

# Modular subsystems
from . import core
from . import lattice
from . import hardware
from . import trainer

# Direct convenient aliases
from .lattice import (
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    conway_sloane_e8_project,
    pack_model_e8,
    load_hf_checkpoint_e8,
)
from .trainer import (
    LowRAMTrainer,
    HolonomicGradFreeOptimizer,
)
from .hardware import (
    TileLangHardwareAutoTuner,
    compile_or_emit_universal_kernel,
)

__all__ = [
    "__version__",
    "compile",
    "train",
    "finetune",
    "optimize",
    "prepare",
    "E8LatticeLinear",
    "E8QuantizationAwareLinear",
    "conway_sloane_e8_project",
    "pack_model_e8",
    "load_hf_checkpoint_e8",
    "LowRAMTrainer",
    "HolonomicGradFreeOptimizer",
    "TileLangHardwareAutoTuner",
    "compile_or_emit_universal_kernel",
    "core",
    "lattice",
    "hardware",
    "trainer",
]
