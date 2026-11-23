"""PyTorch-native autograd safety primitives for OjasX-Aegis."""

from .poison_guard import (
    PoisonGuardResult,
    StreamingPoisonGuard,
    StreamingRankKSketch,
    StreamingWelfordStats,
    all_layer_lat_loss,
    trades_loss,
)
from .lora_lock import (
    LoRASafetyLock,
    LoRASubspaceState,
)
from .gram_router import (
    GRAMLoRAAdapter,
    GRAMPartitionedModule,
    GRAMRouter,
    cosine_circuit_breaker_loss,
    residual_decorrelation_loss,
)
from .rl_shield import (
    RLRolloutAudit,
    RLSafetyShield,
    RLStepLossResult,
    extract_thinking_trace,
)
from .latent_defenses import (
    AllLayerHiddenHookManager,
    AllLayerLatentAdversarialTrainer,
    AllLayerLatentMonitor,
    CircuitBreakerReroutingLoss,
    LatentLayerTelemetry,
    RMUMisdirectionLoss,
    TVaccineLayerGuard,
)

__all__ = [
    "StreamingWelfordStats",
    "StreamingRankKSketch",
    "PoisonGuardResult",
    "StreamingPoisonGuard",
    "trades_loss",
    "all_layer_lat_loss",
    "LoRASubspaceState",
    "LoRASafetyLock",
    "GRAMLoRAAdapter",
    "GRAMPartitionedModule",
    "GRAMRouter",
    "residual_decorrelation_loss",
    "cosine_circuit_breaker_loss",
    "extract_thinking_trace",
    "RLRolloutAudit",
    "RLStepLossResult",
    "RLSafetyShield",
    "AllLayerHiddenHookManager",
    "LatentLayerTelemetry",
    "AllLayerLatentMonitor",
    "CircuitBreakerReroutingLoss",
    "RMUMisdirectionLoss",
    "TVaccineLayerGuard",
    "AllLayerLatentAdversarialTrainer",
]
