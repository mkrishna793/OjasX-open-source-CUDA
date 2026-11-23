"""OjasX-Aegis AI Training Safety System."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import List

from . import attestation, contracts, native_gate, policy
from .contracts import (
    AttestationReceipt,
    BatchFirewallResult,
    ModelArchType,
    SafetyDecision,
    SafetyPolicy,
    SafetyProfile,
    SampleAuditResult,
    StepGuardTelemetry,
)
from .policy import (
    CODE_ENTERPRISE,
    CUSTOM_STRICT,
    PUBLIC_GENERAL,
    RL_AGENTIC,
    SCIENTIFIC_MODULAR,
    get_policy,
    policy_hash,
    validate_policy,
)

_LAZY_SUBMODULES = frozenset({"firewall", "autograd", "eval_harness", "integration"})
_LAZY_INTEGRATION_EXPORTS = frozenset({"AegisSafetyShield", "build_safety_shield"})


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name in _LAZY_INTEGRATION_EXPORTS:
        mod = importlib.import_module(f"{__name__}.integration")
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | _LAZY_SUBMODULES | _LAZY_INTEGRATION_EXPORTS)


__all__ = [
    "AegisSafetyShield",
    "AttestationReceipt",
    "BatchFirewallResult",
    "CODE_ENTERPRISE",
    "CUSTOM_STRICT",
    "ModelArchType",
    "PUBLIC_GENERAL",
    "RL_AGENTIC",
    "SCIENTIFIC_MODULAR",
    "SafetyDecision",
    "SafetyPolicy",
    "SafetyProfile",
    "SampleAuditResult",
    "StepGuardTelemetry",
    "attestation",
    "autograd",
    "build_safety_shield",
    "contracts",
    "eval_harness",
    "firewall",
    "get_policy",
    "integration",
    "native_gate",
    "policy",
    "policy_hash",
    "validate_policy",
]

from .hardening import (
    ExternalSigner,
    compute_model_weights_sha256,
    compute_weight_delta_sha256,
    CumulativeDriftReport,
    CumulativeSubspaceDriftGuard,
    CanaryCheckResult,
    CanaryRollbackGuard,
    convert_probes_to_safetensors,
)
