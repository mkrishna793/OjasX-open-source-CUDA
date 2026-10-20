"""OjasX verified mathematical reference core; no GPU performance certification."""
from .lenses import Lens, linear_lens, parameterized_linear_lens, activation_lens
from .ops import packed_relu, gelu, silu, swiglu, rmsnorm, rmsnorm_aux
from .models import PackedReLU, optimize_relu_residuals, shared_projections
from .residuals import SavedTensorAudit
from .streamed import streamed_gemm, operand_tape_gemm
from .compiler import make_backend

from .contractions import linear_chain, chain_cost, factored_gemm_pullback
from .planner import matrix_chain, matrix_chain_plan, linear_sequence
from .nonlinear import relu_linear_chain, softmax_pullback, layernorm_input_pullback, attention_softmax_row_scalar
from .nonlinear import layernorm_linear_chain, layernorm_chain_cost
from .nonlinear import gram_layernorm_linear_chain, gram_layernorm_chain_cost
from .radial_core import rms_core_linear_chain, rms_core_cost
from .reversible import reversible_coupling_chain
from . import math
from .symplectic_engine import (
    SymplecticCouplingFunction,
    SymplecticCoupling,
    SymplecticTransformerBlock,
    SymplecticCausalAttention,
    SymplecticSwiGLUFFN,
    SymplecticStack,
    SymplecticMetrics,
    SymplecticSqueeze,
    symplectic_coupling,
    symplectic_coupling_chain,
    symplectic_squeeze,
    audit_symplectic_reversibility,
)

from .holonomic_math_compiler import (
    holonomic_linear,
    HolonomicLinear,
    holonomic_swiglu,
    HolonomicSwiGLU,
    holonomic_geglu,
    HolonomicGeGLU,
    holonomic_rmsnorm,
    HolonomicRMSNorm,
    holonomic_lm_head_loss,
    HolonomicLMHead,
    holonomic_backend,
)
from .generated_holonomic_ops import (
    HOLONOMIC_DECLARATIONS_LEDGER,
    holonomic_rope,
    HolonomicRoPE,
    holonomic_attention_softmax,
)
from .openreg_backend import (
    register_ojasx_openreg,
    HolonomicDispatchMode,
    ojasx_device_module,
)
from .aot_holonomic_compiler import (
    AOTCompilationLedger,
    LAST_AOT_LEDGER,
    holonomic_aot_backend,
    compile_module_aot_simplified,
)
from .grad_free_optimizer import HolonomicGradFreeOptimizer
from .tile_hal import (
    SiliconCapabilityProfile,
    TileSchedule,
    TileSchedulePlanner,
    probe_hardware_profile,
    verify_xor_swizzle_and_ballot_smt,
    amx_linear_relu2,
    l1_rmsnorm,
    tilelang_metal_gemm_relu2,
    tilelang_universal_gemm_relu2,
    tilelang_query_chip_schedule,
    tilelang_emit_universal_source,
    HolonomicBitPackedLinear,
    HolonomicNativeRMSNorm,
    HolonomicUnifiedTransformerMLP,
)
from .smart_router import (
    SmartLayerDecision,
    SmartWorkloadReport,
    SmartWorkloadRouter,
    FrozenLinearZeroTape,
    optimize_model,
)
from .fx_subgraph_compiler import (
    holonomic_symplectic_subgraph,
    holonomic_residual_subgraph,
    HolonomicSymplecticSubgraphFunction,
    HolonomicResidualSubgraphFunction,
    compile_fx_subgraph,
    FXSubgraphCompiler,
    TransformationStats,
    transform_subgraph_module,
)
from .holonomic_layers import (
    HolonomicLayerNorm,
    HolonomicGroupNorm,
    FrozenConvZeroTape,
    HolonomicActivationLens,
    HolonomicFusedGLUMLP,
    HolonomicFusedSwiGLUMLP,
    HolonomicFusedGeGLUMLP,
    HolonomicFusedReGLUMLP,
    HolonomicFusedReLU2GLUMLP,
    HolonomicFusedSigmoidGLUMLP,
    HolonomicFusedBilinearGLUMLP,
    HolonomicFusedChunkedGLUMLP,
    HolonomicGLU,
)
from .mesh_hal import (
    MeshNodeSpec,
    MeshTransferTelemetry,
    VirtualMeshPartitionPlan,
    HolonomicVirtualMeshEngine,
    MeshRemoteWorkerServer,
    plan_mesh_partitions,
)
from .dagger_attention import (
    dagger_attention,
    dagger_tiled_attention,
    holonomic_dagger_attention,
    holonomic_dagger_attention_with_lse,
    DaggerAttention,
    HolonomicDaggerAttention,
)
from .fused_cross_entropy import (
    fused_linear_cross_entropy,
    holonomic_fused_linear_cross_entropy,
    FusedLinearCrossEntropy,
    HolonomicFusedLinearCrossEntropy,
)
from .mlir_dialects import (
    MLIRPipelineArtifacts,
    MLIRCompiledExecutable,
    HolonomicToSubgraphPass,
    SubgraphToTilePass,
    fx_to_holonomic_mlir,
    lower_fx_to_mlir_artifacts,
    compile_mlir_executable,
    mlir_holonomic_backend,
    detect_default_mlir_target,
    compile_model_to_target,
    compile_mlir_to_target,
)
from .logger import (
    OjasXLogger,
    get_logger,
    set_log_level,
)
from .updater import (
    update,
    check_for_updates,
    notify_if_update_available_background,
)
from .math.lattice import (
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    E8LatticePacker,
    TernaryE8LatticePacker,
    conway_sloane_e8_project,
    pack_model_e8,
    fast_walsh_hadamard_transform,
    randomized_hadamard_transform,
)

__version__ = "0.9.1"

# Launch non-blocking background update check (0ms import overhead, cached)
notify_if_update_available_background()
