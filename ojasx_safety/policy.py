"""Policy presets, deterministic hashing, and immutable Tier-0 redline invariants."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any, Dict, Tuple, Union

from .contracts import SafetyPolicy, SafetyProfile


TIER0_IMMUTABLE_REDLINES: Tuple[str, ...] = (
    "cbrn_weaponization",
    "csam",
    "cyber_exploit",
    "poison_trigger",
    "secret_pii",
)

DEFAULT_DUAL_USE_DOMAINS: Tuple[str, ...] = (
    "biology_virology",
    "chemistry_synthesis",
    "nuclear_physics",
    "offensive_security",
)

_POLICY_HASH_DOMAIN = b"OJASX_AEGIS_POLICY_V1:"


class PolicyViolationError(ValueError):
    """Raised fail-closed when a SafetyPolicy violates Tier-0 or structural invariants."""


def _check_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyViolationError(f"Policy field '{name}' must be numeric, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val):
        raise PolicyViolationError(f"Policy field '{name}' must be finite, got {value!r}")
    return 0.0 if val == 0.0 else val


def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyViolationError(f"Policy field '{name}' must be an int, got {type(value).__name__}")
    if value < minimum:
        raise PolicyViolationError(f"Policy field '{name}' must be >= {minimum}, got {value}")
    return value


def enforce_tier0_redlines(policy: SafetyPolicy) -> None:
    """Enforce immutable Tier-0 redline coverage whenever fail_closed=True."""
    if not isinstance(policy.redline_domains, (tuple, list)):
        raise PolicyViolationError("redline_domains must be a tuple or list of domain identifiers")
    normalized = {str(d).strip() for d in policy.redline_domains if str(d).strip()}
    if policy.fail_closed:
        missing = [domain for domain in TIER0_IMMUTABLE_REDLINES if domain not in normalized]
        if missing:
            raise PolicyViolationError(
                "Immutable Tier-0 redlines cannot be disabled when fail_closed=True; "
                f"missing={missing}"
            )


def validate_policy(policy: SafetyPolicy) -> SafetyPolicy:
    """Validate Tier-0 redline invariants and bounded calibration thresholds."""
    if not isinstance(policy, SafetyPolicy):
        raise PolicyViolationError(f"Expected SafetyPolicy, got {type(policy).__name__}")
    if not isinstance(policy.profile, SafetyProfile):
        raise PolicyViolationError(f"Invalid SafetyProfile: {policy.profile!r}")
    if not isinstance(policy.fail_closed, bool):
        raise PolicyViolationError("fail_closed must be a boolean")
    if not isinstance(policy.enforce_native_gate, bool):
        raise PolicyViolationError("enforce_native_gate must be a boolean")

    enforce_tier0_redlines(policy)

    if not isinstance(policy.dual_use_domains, (tuple, list)):
        raise PolicyViolationError("dual_use_domains must be a tuple or list")

    dedup_thresh = _check_finite_float("dedup_similarity_threshold", policy.dedup_similarity_threshold)
    if not (0.0 < dedup_thresh <= 1.0):
        raise PolicyViolationError("dedup_similarity_threshold must lie in (0.0, 1.0]")

    _check_int("blocklist_escalation_terms", policy.blocklist_escalation_terms, 1)

    cls_thresh = _check_finite_float("classifier_block_threshold", policy.classifier_block_threshold)
    if not (0.0 < cls_thresh <= 1.0):
        raise PolicyViolationError("classifier_block_threshold must lie in (0.0, 1.0]")

    _check_int("reconstruction_window_size", policy.reconstruction_window_size, 1)

    recon_thresh = _check_finite_float("reconstruction_block_threshold", policy.reconstruction_block_threshold)
    if not (0.0 < recon_thresh <= 1.0):
        raise PolicyViolationError("reconstruction_block_threshold must lie in (0.0, 1.0]")

    max_z = _check_finite_float("max_grad_zscore", policy.max_grad_zscore)
    if max_z <= 0.0:
        raise PolicyViolationError("max_grad_zscore must be > 0.0")

    max_spec = _check_finite_float("max_spectral_outlier_ratio", policy.max_spectral_outlier_ratio)
    if max_spec <= 0.0:
        raise PolicyViolationError("max_spectral_outlier_ratio must be > 0.0")

    max_kl = _check_finite_float("max_kl_divergence", policy.max_kl_divergence)
    if max_kl <= 0.0:
        raise PolicyViolationError("max_kl_divergence must be > 0.0")

    proj_strength = _check_finite_float(
        "lora_safety_projection_strength", policy.lora_safety_projection_strength
    )
    if not (0.0 <= proj_strength <= 1.0):
        raise PolicyViolationError("lora_safety_projection_strength must lie in [0.0, 1.0]")

    gram_weight = _check_finite_float("gram_decorrelation_weight", policy.gram_decorrelation_weight)
    if gram_weight < 0.0:
        raise PolicyViolationError("gram_decorrelation_weight must be >= 0.0")

    return policy


PUBLIC_GENERAL = validate_policy(
    SafetyPolicy(
        profile=SafetyProfile.PUBLIC_GENERAL,
        redline_domains=TIER0_IMMUTABLE_REDLINES,
        dual_use_domains=DEFAULT_DUAL_USE_DOMAINS,
        dedup_similarity_threshold=0.85,
        blocklist_escalation_terms=2,
        classifier_block_threshold=0.55,
        reconstruction_window_size=4,
        reconstruction_block_threshold=0.60,
        max_grad_zscore=3.5,
        max_spectral_outlier_ratio=4.0,
        max_kl_divergence=0.25,
        lora_safety_projection_strength=1.0,
        gram_decorrelation_weight=0.05,
        enforce_native_gate=True,
        fail_closed=True,
    )
)

SCIENTIFIC_MODULAR = validate_policy(
    SafetyPolicy(
        profile=SafetyProfile.SCIENTIFIC_MODULAR,
        redline_domains=TIER0_IMMUTABLE_REDLINES,
        dual_use_domains=DEFAULT_DUAL_USE_DOMAINS,
        dedup_similarity_threshold=0.88,
        blocklist_escalation_terms=3,
        classifier_block_threshold=0.62,
        reconstruction_window_size=4,
        reconstruction_block_threshold=0.68,
        max_grad_zscore=3.8,
        max_spectral_outlier_ratio=4.5,
        max_kl_divergence=0.25,
        lora_safety_projection_strength=1.0,
        gram_decorrelation_weight=0.10,
        enforce_native_gate=True,
        fail_closed=True,
    )
)

CODE_ENTERPRISE = validate_policy(
    SafetyPolicy(
        profile=SafetyProfile.CODE_ENTERPRISE,
        redline_domains=TIER0_IMMUTABLE_REDLINES,
        dual_use_domains=DEFAULT_DUAL_USE_DOMAINS,
        dedup_similarity_threshold=0.80,
        blocklist_escalation_terms=1,
        classifier_block_threshold=0.50,
        reconstruction_window_size=5,
        reconstruction_block_threshold=0.55,
        max_grad_zscore=3.0,
        max_spectral_outlier_ratio=3.5,
        max_kl_divergence=0.20,
        lora_safety_projection_strength=1.0,
        gram_decorrelation_weight=0.06,
        enforce_native_gate=True,
        fail_closed=True,
    )
)

RL_AGENTIC = validate_policy(
    SafetyPolicy(
        profile=SafetyProfile.RL_AGENTIC,
        redline_domains=TIER0_IMMUTABLE_REDLINES,
        dual_use_domains=DEFAULT_DUAL_USE_DOMAINS,
        dedup_similarity_threshold=0.85,
        blocklist_escalation_terms=2,
        classifier_block_threshold=0.50,
        reconstruction_window_size=4,
        reconstruction_block_threshold=0.55,
        max_grad_zscore=3.0,
        max_spectral_outlier_ratio=3.0,
        max_kl_divergence=0.12,
        lora_safety_projection_strength=1.0,
        gram_decorrelation_weight=0.08,
        enforce_native_gate=True,
        fail_closed=True,
    )
)

CUSTOM_STRICT = validate_policy(
    SafetyPolicy(
        profile=SafetyProfile.CUSTOM_STRICT,
        redline_domains=TIER0_IMMUTABLE_REDLINES,
        dual_use_domains=DEFAULT_DUAL_USE_DOMAINS,
        dedup_similarity_threshold=0.75,
        blocklist_escalation_terms=1,
        classifier_block_threshold=0.40,
        reconstruction_window_size=6,
        reconstruction_block_threshold=0.45,
        max_grad_zscore=2.5,
        max_spectral_outlier_ratio=2.5,
        max_kl_divergence=0.10,
        lora_safety_projection_strength=1.0,
        gram_decorrelation_weight=0.15,
        enforce_native_gate=True,
        fail_closed=True,
    )
)

POLICY_PRESETS: Dict[SafetyProfile, SafetyPolicy] = {
    SafetyProfile.PUBLIC_GENERAL: PUBLIC_GENERAL,
    SafetyProfile.SCIENTIFIC_MODULAR: SCIENTIFIC_MODULAR,
    SafetyProfile.CODE_ENTERPRISE: CODE_ENTERPRISE,
    SafetyProfile.RL_AGENTIC: RL_AGENTIC,
    SafetyProfile.CUSTOM_STRICT: CUSTOM_STRICT,
}
PRESETS = POLICY_PRESETS


def get_policy(
    profile: Union[SafetyProfile, str] = SafetyProfile.PUBLIC_GENERAL,
    **overrides: Any,
) -> SafetyPolicy:
    """Return a validated copy of a canonical preset with optional safe overrides."""
    if isinstance(profile, str):
        key = profile.strip()
        try:
            resolved = SafetyProfile(key.lower())
        except ValueError:
            try:
                resolved = SafetyProfile[key.upper()]
            except KeyError as exc:
                raise PolicyViolationError(f"Unknown SafetyProfile: {profile!r}") from exc
    elif isinstance(profile, SafetyProfile):
        resolved = profile
    else:
        raise PolicyViolationError(f"Invalid profile type: {type(profile).__name__}")

    base = POLICY_PRESETS[resolved]
    candidate = replace(base, **overrides) if overrides else replace(base)
    return validate_policy(candidate)


def with_policy_overrides(policy: SafetyPolicy, **overrides: Any) -> SafetyPolicy:
    """Clone an existing SafetyPolicy with overrides while enforcing all invariants."""
    validate_policy(policy)
    candidate = replace(policy, **overrides)
    return validate_policy(candidate)


def canonical_policy_dict(policy: SafetyPolicy, *, validate: bool = True) -> Dict[str, Any]:
    """Return deterministic JSON-serializable representation of a SafetyPolicy."""
    if validate:
        validate_policy(policy)
    raw = policy.to_dict()
    return {
        "profile": str(raw["profile"]),
        "redline_domains": sorted(dict.fromkeys(str(d).strip() for d in raw["redline_domains"])),
        "dual_use_domains": sorted(dict.fromkeys(str(d).strip() for d in raw["dual_use_domains"])),
        "dedup_similarity_threshold": _check_finite_float(
            "dedup_similarity_threshold", raw["dedup_similarity_threshold"]
        ),
        "blocklist_escalation_terms": int(raw["blocklist_escalation_terms"]),
        "classifier_block_threshold": _check_finite_float(
            "classifier_block_threshold", raw["classifier_block_threshold"]
        ),
        "reconstruction_window_size": int(raw["reconstruction_window_size"]),
        "reconstruction_block_threshold": _check_finite_float(
            "reconstruction_block_threshold", raw["reconstruction_block_threshold"]
        ),
        "max_grad_zscore": _check_finite_float("max_grad_zscore", raw["max_grad_zscore"]),
        "max_spectral_outlier_ratio": _check_finite_float(
            "max_spectral_outlier_ratio", raw["max_spectral_outlier_ratio"]
        ),
        "max_kl_divergence": _check_finite_float("max_kl_divergence", raw["max_kl_divergence"]),
        "lora_safety_projection_strength": _check_finite_float(
            "lora_safety_projection_strength", raw["lora_safety_projection_strength"]
        ),
        "gram_decorrelation_weight": _check_finite_float(
            "gram_decorrelation_weight", raw["gram_decorrelation_weight"]
        ),
        "enforce_native_gate": bool(raw["enforce_native_gate"]),
        "fail_closed": bool(raw["fail_closed"]),
    }


def canonical_policy_json(policy: SafetyPolicy, *, validate: bool = True) -> str:
    """Serialize SafetyPolicy to deterministic canonical JSON (sorted keys, no NaN)."""
    payload = canonical_policy_dict(policy, validate=validate)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def policy_hash(policy: SafetyPolicy, *, validate: bool = True) -> str:
    """Compute domain-separated SHA-256 hex digest of a canonical SafetyPolicy."""
    canonical = canonical_policy_json(policy, validate=validate).encode("utf-8")
    return hashlib.sha256(_POLICY_HASH_DOMAIN + canonical).hexdigest()


__all__ = [
    "CODE_ENTERPRISE",
    "CUSTOM_STRICT",
    "DEFAULT_DUAL_USE_DOMAINS",
    "POLICY_PRESETS",
    "PRESETS",
    "PUBLIC_GENERAL",
    "PolicyViolationError",
    "RL_AGENTIC",
    "SCIENTIFIC_MODULAR",
    "TIER0_IMMUTABLE_REDLINES",
    "canonical_policy_dict",
    "canonical_policy_json",
    "enforce_tier0_redlines",
    "get_policy",
    "policy_hash",
    "validate_policy",
    "with_policy_overrides",
]
