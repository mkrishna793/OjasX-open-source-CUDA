"""
OjasX-LogicIR Compiler (Phases 1–5):
Unified Algebraic + Bitwise Logic-Gate + 2.125-Bit Dual-Rail 9B Trainer + MLIR/IREE.
"""

from ojasx_logic.numsys.mx_naf_bf16 import (
    MXNAFTensor,
    naf5_encode_i32,
    to_bf16_rounded_f32,
    encode_mx_naf_bf16,
    decode_mx_naf_bf16,
    gemm_mx_naf_dual_rail,
)
from ojasx_logic.numsys.crt3_bf16 import (
    CRT3_MODS,
    CRT4_MODS,
    CRTPlan,
    gemm_crt_int,
    gemm_crt_bf16,
)
from ojasx_logic.numsys.ternary_dual_rail import (
    TernaryDualRailTensor,
    pack_ternary_dual_rail,
    unpack_ternary_dual_rail,
    estimate_llm_memory_budget,
)
from ojasx_logic.z3_synth.bit_circuits import (
    bit_relu,
    bit_abs,
    bit_neg,
    bit_fast_exp,
    bit_silu_pwl,
    bit_gelu_pwl,
    bit_rsqrt,
    exhaustive_bf16_audit,
)
from ojasx_logic.z3_synth.cegis_activations import run_z3_logic_ir_proofs
from ojasx_logic.ir.algmat_ir import AlgNode, AlgOptStats, optimize_algmat
from ojasx_logic.ir.logic_ir import LogicOp, LogicIRModule, choose_alu_route
from ojasx_logic.backend.mlir_emitter import emit_logic_ir_mlir, emit_standalone_mlir_suite
from ojasx_logic.backend.iree_runner import (
    IREETargetConfig,
    IREECompileResult,
    IREERunResult,
    detect_iree_toolchain,
    compile_mlir_to_vmfb,
    run_mlir_kernel_iree,
)

__version__ = "0.2.0-p5"

__all__ = [
    "__version__",
    "MXNAFTensor",
    "naf5_encode_i32",
    "to_bf16_rounded_f32",
    "encode_mx_naf_bf16",
    "decode_mx_naf_bf16",
    "gemm_mx_naf_dual_rail",
    "CRT3_MODS",
    "CRT4_MODS",
    "CRTPlan",
    "gemm_crt_int",
    "gemm_crt_bf16",
    "TernaryDualRailTensor",
    "pack_ternary_dual_rail",
    "unpack_ternary_dual_rail",
    "estimate_llm_memory_budget",
    "bit_relu",
    "bit_abs",
    "bit_neg",
    "bit_fast_exp",
    "bit_silu_pwl",
    "bit_gelu_pwl",
    "bit_rsqrt",
    "exhaustive_bf16_audit",
    "run_z3_logic_ir_proofs",
    "AlgNode",
    "AlgOptStats",
    "optimize_algmat",
    "LogicOp",
    "LogicIRModule",
    "choose_alu_route",
    "emit_logic_ir_mlir",
    "emit_standalone_mlir_suite",
    "IREETargetConfig",
    "IREECompileResult",
    "IREERunResult",
    "detect_iree_toolchain",
    "compile_mlir_to_vmfb",
    "run_mlir_kernel_iree",
]
