from .mlir_emitter import emit_logic_ir_mlir, emit_standalone_mlir_suite
from .iree_runner import (
    IREETargetConfig,
    IREECompileResult,
    IREERunResult,
    detect_iree_toolchain,
    compile_mlir_to_vmfb,
    run_mlir_kernel_iree,
)
from .metal_mps_runtime import (
    metal_runtime_status,
    mps_dual_rail_linear_forward,
    mps_dual_rail_popcnt_gemm_16x16,
)

__all__ = [
    "emit_logic_ir_mlir",
    "emit_standalone_mlir_suite",
    "IREETargetConfig",
    "IREECompileResult",
    "IREERunResult",
    "detect_iree_toolchain",
    "compile_mlir_to_vmfb",
    "run_mlir_kernel_iree",
    "metal_runtime_status",
    "mps_dual_rail_linear_forward",
    "mps_dual_rail_popcnt_gemm_16x16",
]
