"""Shared interface contracts for the OjasX-Aegis AI Training Safety System."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    TAG_GRAM_AUX = "tag_gram_aux"
    BLOCK_ZERO_GRAD = "block_zero_grad"
    QUARANTINE_STEP = "quarantine_step"


class SafetyProfile(str, Enum):
    PUBLIC_GENERAL = "public_general"
    SCIENTIFIC_MODULAR = "scientific_modular"
    CODE_ENTERPRISE = "code_enterprise"
    RL_AGENTIC = "rl_agentic"
    CUSTOM_STRICT = "custom_strict"


class ModelArchType(str, Enum):
    BASE_CAUSAL_LM = "base_causal_lm"
    CHAT_LLM = "chat_llm"
    REASONING_LLM = "reasoning_llm"
    SSM_HYBRID = "ssm_hybrid"
    DIFFUSION = "diffusion"
    JEPA_WORLD = "jepa_world"
    MULTIMODAL = "multimodal"
    RL_POLICY = "rl_policy"
    CUSTOM_NN = "custom_nn"


@dataclass
class SafetyPolicy:
    """Declarative policy binding immutable Tier-0 red lines and calibrated Tier-1 routing."""

    profile: SafetyProfile = SafetyProfile.PUBLIC_GENERAL
    redline_domains: Tuple[str, ...] = (
        "cbrn_weaponization",
        "csam",
        "cyber_exploit",
        "poison_trigger",
        "secret_pii",
    )
    dual_use_domains: Tuple[str, ...] = (
        "biology_virology",
        "chemistry_synthesis",
        "nuclear_physics",
        "offensive_security",
    )
    dedup_similarity_threshold: float = 0.85
    blocklist_escalation_terms: int = 2
    classifier_block_threshold: float = 0.55
    reconstruction_window_size: int = 4
    reconstruction_block_threshold: float = 0.60
    max_grad_zscore: float = 3.5
    max_spectral_outlier_ratio: float = 4.0
    max_kl_divergence: float = 0.25
    lora_safety_projection_strength: float = 1.0
    gram_decorrelation_weight: float = 0.05
    enforce_native_gate: bool = True
    fail_closed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["profile"] = self.profile.value
        return d


@dataclass(frozen=True)
class SampleAuditResult:
    sample_index: int
    decision: SafetyDecision
    weight: float  # Strictly 0.0 (zero-gradient annihilated) or 1.0
    domain_tag: str
    risk_score: float
    reasons: Tuple[str, ...]
    redacted_payload: Optional[Any] = None
    sha256_digest: str = ""


@dataclass
class BatchFirewallResult:
    clean_weights: Any  # torch.Tensor of 0.0 / 1.0 floats shaped [B]
    domain_tags: List[str]
    sample_results: List[SampleAuditResult]
    batch_merkle_root: str
    gate_mask_hash: str
    stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class StepGuardTelemetry:
    step: int
    quarantined: bool
    reason: str
    grad_zscore: float
    spectral_ratio: float
    lora_projected_norm: float
    gram_decorrelation_loss: float
    rl_kl_divergence: float
    step_chain_hash: str


@dataclass
class AttestationReceipt:
    run_id: str
    policy_hash: str
    genesis_hash: str
    final_chain_hash: str
    total_steps: int
    blocked_samples: int
    quarantined_batches: int
    eval_digest: str
    checkpoint_sha256: str
    public_key_pem: str
    signature_hex: str
    verified: Optional[bool] = None
