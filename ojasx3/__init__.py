"""
OjasX3: Universal Turnkey AI Training & Mathematical Execution Engine
===================================================================

Release v0.17.0 — Modular, Turnkey, Drop-In, Multi-Hardware AI System
"""

__version__ = "0.17.0"

# Turnkey public 1-line APIs
from .api import (
    compile,
    train,
    finetune,
    optimize,
    prepare,
    train_custom_llm,
    build_custom_llm,
    CustomLLMConfig,
    CUSTOM_LLM_PRESETS,
    OjasXCustomCausalLM,
    CustomLLMTrainResult,
)

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

from ojasx3.runtime.tape_hooks import E8ActivationTapeContext, TapeCompressionStats
from ojasx3.compiler.universal import UniversalCompilerConfig, compile_universal
from ojasx3.lattice.transforms import CWOFTLinear, apply_incoherence_transform
from ojasx3.hardware.triton_e8_kernels import triton_fused_e8_2bit_gemm

from ojasx3.trainer.export import export_peft_adapter, merge_and_export_hf
from ojasx3.trainer.precision_thermostat import (
    LayerSNRReport,
    select_sensitive_layers,
    RotatingCWOrbitController,
)
from ojasx3.lattice.transforms import (
    cw120_36_orbit_support,
    generate_rademacher_signs,
    apply_two_sided_incoherence,
)
from ojasx3.hardware.metal_e8_dispatch import (
    metal_fused_e8_2bit_gemm,
    metal_fused_ternary_e8_gemm,
)

# Phase 5: OjasX-LogicIR Compiler, 2.125-bit Dual-Rail 9B Trainer & Zero-Xcode Metal 4 Runtime
from ojasx_logic import (
    AlgNode,
    optimize_algmat,
    LogicIRModule,
    choose_alu_route,
    emit_logic_ir_mlir,
    emit_standalone_mlir_suite,
    pack_ternary_dual_rail,
    unpack_ternary_dual_rail,
    estimate_llm_memory_budget,
)
from ojasx_logic.trainer import (
    DualRailLoRALinear,
    quantize_module_to_dual_rail_lora,
    load_and_quantize_hf_model,
    run_dual_rail_lora_training,
)
from ojasx_logic.backend.metal_mps_runtime import (
    metal_runtime_status,
    mps_dual_rail_linear_forward,
    mps_dual_rail_popcnt_gemm_16x16,
)
