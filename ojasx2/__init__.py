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
    RootE8LatticePacker,
    conway_sloane_e8_project,
    pack_model_e8,
    fwht_block32,
    fast_walsh_hadamard_transform,
    randomized_hadamard_transform,
)
from .tilelang_kernels import (
    is_tilelang_available,
    dispatch_tilelang_e8_gemm,
)
from .runtime import (
    OjasXRunner,
    OjasXRuntime,
    OjasXRunnerLinear,
    OjasXLinear,
    pack_model_runner,
    pack_model_ojasx_runner,
    E8KVCache,
    OjasXRunnerMemoryReport,
    OjasXMemoryReport,
    Leech24LatticePacker,
    conway_sloane_leech24_project,
    PrecomputedTernaryLUT,
    OptimalRootE8Packer,
    PureLeech24RootPacker,
    MatryoshkaRootLinear,
    fold_consecutive_layers_symplectic,
    pack_matryoshka_runner,
    pack_hybrid_moe_runner,
    HybridMoERunner,
    MatryoshkaRunner,
)
from . import runtime
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Any

__version__ = "0.12.0"


# =============================================================================
# OjasX v12 Simple 1-Line Developer API
# =============================================================================

def optimize(
    model: nn.Module,
    quantize: Optional[Union[str, float]] = None,
    reversible: bool = True,
    attention: str = "fast",
    group_size: int = 32,
    **kwargs: Any,
) -> nn.Module:
    """
    1-Line OjasX v12 Model Optimizer.

    Transforms any standard PyTorch or HuggingFace model in-place:
    - Replaces standard attention with Zero-DRAM DaggerAttention + Hardware SDPA
    - Quantizes linear layers into 1.0b Root E8, 1.5b Ternary E8, or 2.0b E8 (if quantize specified)
    - Automatically enables FWHT-32 outlier rotation and Störmer-Verlet reversibility
    """
    from .smart_router import optimize_model as _smart_optimize
    from .math.lattice import pack_model_e8

    opt_model, _ = _smart_optimize(model, **kwargs)

    if quantize is not None:
        bitrate_map = {"1.0b": 1.0, "1.0": 1.0, "1.5b": 1.5, "1.5": 1.5, "2.0b": 2.0, "2.0": 2.0}
        bitrate = bitrate_map.get(str(quantize).lower(), 1.5) if isinstance(quantize, str) else float(quantize)
        opt_model = pack_model_e8(opt_model, bitrate=bitrate, group_size=group_size)

    return opt_model


def prepare(
    model: nn.Module,
    lr: float = 1e-4,
    optimizer: str = "adamw",
    quantize: Optional[Union[str, float]] = None,
    reversible: bool = True,
    attention: str = "fast",
    grad_accum_steps: int = 1,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Tuple[nn.Module, HolonomicGradFreeOptimizer]:
    """
    1-Line OjasX v12 Zero-DRAM Training Preparation.

    Returns (optimized_model, zero_ram_optimizer) configured and ready for training:
    - Zero `.grad` overhead (immediate in-backward memory freeing)
    - 2.0-bit E8 quantized momentum buffers
    - Native hardware SDPA fast-paths

    Usage:
        model, optimizer = ojasx.prepare(model, lr=1e-4)
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = ojasx.loss(model(x), y)
            loss.backward()
            optimizer.step()
    """
    opt_model = optimize(
        model,
        quantize=quantize,
        reversible=reversible,
        attention=attention,
        **kwargs,
    )

    use_e8_mom = quantize is not None
    opt = HolonomicGradFreeOptimizer(
        opt_model.parameters(),
        lr=lr,
        algorithm=optimizer,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        quantize_moments_e8=use_e8_mom,
    )
    return opt_model, opt


def loss(
    pred_or_hidden: torch.Tensor,
    targets_or_weight: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    1-Line OjasX v12 Zero-DRAM Cross-Entropy Loss.

    Supports two modes:
    1. Direct Fused LM-Head: ojasx.loss(hidden_states, lm_head_weight, targets, bias=None)
       Computes cross-entropy on-the-fly without materializing multi-GB [B, S, V] logits!
    2. Standard Logits: ojasx.loss(logits, targets)
       Computes standard cross-entropy with optional label smoothing.
    """
    if targets is not None:
        from .fused_cross_entropy import fused_linear_cross_entropy
        return fused_linear_cross_entropy(
            pred_or_hidden,
            targets_or_weight,
            targets,
            bias=bias,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    else:
        import torch.nn.functional as F
        logits = pred_or_hidden
        target = targets_or_weight
        if logits.ndim > 2 and target.ndim == logits.ndim - 1:
            logits = logits.view(-1, logits.size(-1))
            target = target.view(-1)
        return F.cross_entropy(
            logits, target, ignore_index=ignore_index, label_smoothing=label_smoothing
        )


# Launch non-blocking background update check (0ms import overhead, cached)
notify_if_update_available_background()
