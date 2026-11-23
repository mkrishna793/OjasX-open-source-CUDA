"""
OjasX-Aegis Red-Team Hardening Primitives (v2.0).

Closes the 5 adversarial training-safety gaps identified in red-team code audit:
  1. ExternalSigner Protocol (KMS / HSM / Apple Secure Enclave / Confidential-GPU TEE):
     Allows Ed25519 / ECDSA receipt signing without private keys ever living in Python RAM.
  2. Immutable Step-0 Baseline & Cumulative Safety-Subspace Drift Budget:
     Defeats "boiling-the-frog" slow-drift poisoning that shifts EMA baselines over 200+ steps
     by tracking cumulative ||P_safe(W_t - W_0)||_F against a frozen step-0 anchor.
  3. CanaryRollbackGuard:
     Detects multi-layer nonlinear re-entanglement via periodic forward-pass canary probes
     and automatically rolls model weights back to the last cryptographically verified snapshot.
  4. Weight-Delta SHA-256 Commitment (`compute_weight_delta_sha256`):
     Binds actual parameter tensors into the SafetyLedger so silent `torch.no_grad()` weight
     tampering breaks the hash chain.
  5. Zero-Pickle Safetensors Probe Migration (`convert_probes_to_safetensors`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. External Hardware / KMS / TEE Signer Protocol (No Private Key in Python RAM)
# ---------------------------------------------------------------------------

@runtime_checkable
class ExternalSigner(Protocol):
    """Protocol for hardware-backed or remote signing (HSM, AWS/GCP KMS, Apple Secure Enclave, H100 TEE).

    Threat-model note:
      - Software-only Ed25519 keys generated in Python RAM provide tamper-evident
        compliance/audit trails against accidental corruption or post-hoc receipt edits.
      - Against an adversarial local host operator, bind signing to an ExternalSigner
        backed by an HSM/KMS or confidential-compute TEE attestation quote.
    """

    @property
    def signer_backend(self) -> str: ...

    def public_key_pem(self) -> str: ...

    def sign(self, payload: bytes) -> bytes: ...


# ---------------------------------------------------------------------------
# 2. Weight-Delta SHA-256 Commitment (Catches Silent `torch.no_grad()` Tampering)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_model_weights_sha256(model: nn.Module) -> str:
    """Deterministic SHA-256 digest over all named parameter tensors (canonical float32 bytes)."""
    h = hashlib.sha256()
    h.update(b"OJASX_AEGIS_WEIGHTS_V2:")
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        t = param.detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def compute_weight_delta_sha256(
    model: nn.Module,
    anchor_weights: Dict[str, torch.Tensor],
) -> Tuple[str, float]:
    """Return (SHA-256 digest of W_t - W_0, total L2 displacement ||W_t - W_0||_F)."""
    h = hashlib.sha256()
    h.update(b"OJASX_AEGIS_DELTA_V2:")
    total_sq = 0.0
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        w_cur = param.detach().to(device="cpu", dtype=torch.float32)
        w_ref = anchor_weights.get(name)
        delta = (w_cur - w_ref) if w_ref is not None else w_cur
        total_sq += float(delta.pow(2).sum().item())
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(delta.contiguous().numpy().tobytes())
    return h.hexdigest(), float(total_sq ** 0.5)


# ---------------------------------------------------------------------------
# 3. Immutable Step-0 Anchor & Cumulative Safety-Subspace Drift Budget
#    (Defeats "Boiling-the-Frog" Low-and-Slow Poisoning!)
# ---------------------------------------------------------------------------

@dataclass
class CumulativeDriftReport:
    exceeded: bool
    total_subspace_drift: float
    budget: float
    worst_layer: str
    worst_layer_drift: float
    per_layer_drift: Dict[str, float] = field(default_factory=dict)


class CumulativeSubspaceDriftGuard:
    """Tracks cumulative weight displacement ||P_safe(W_t - W_0)||_F against an
    IMMUTABLE step-0 anchor (never polluted by training-time EMA updates).

    Even if an attacker injects tiny sub-threshold poisoned gradients over 500 steps
    that evade per-step z-score checks, cumulative displacement inside the safety
    subspace is caught deterministically.
    """

    def __init__(
        self,
        max_cumulative_drift: float = 0.50,
        per_layer_drift_cap: Optional[float] = None,
    ) -> None:
        if max_cumulative_drift <= 0.0:
            raise ValueError("max_cumulative_drift must be > 0")
        self.max_cumulative_drift = float(max_cumulative_drift)
        self.per_layer_drift_cap = float(per_layer_drift_cap) if per_layer_drift_cap is not None else None
        self._anchor_weights: Dict[str, torch.Tensor] = {}
        self._subspace_u: Dict[str, torch.Tensor] = {}
        self._frozen: bool = False

    @torch.no_grad()
    def freeze_anchor(
        self,
        model: nn.Module,
        safety_subspaces: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Snapshot immutable W_0 weights (and optional U_out safety bases) at step 0."""
        self._anchor_weights.clear()
        self._subspace_u.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._anchor_weights[name] = param.detach().to(device="cpu", dtype=torch.float32).clone()
        if safety_subspaces:
            for layer_name, state in safety_subspaces.items():
                u = getattr(state, "u_out", None)
                if isinstance(u, torch.Tensor):
                    self._subspace_u[layer_name] = u.detach().to(device="cpu", dtype=torch.float32).clone()
        self._frozen = True

    @torch.no_grad()
    def check_drift(self, model: nn.Module) -> CumulativeDriftReport:
        if not self._frozen:
            self.freeze_anchor(model)

        per_layer: Dict[str, float] = {}
        total_sq = 0.0
        worst_layer = ""
        worst_drift = 0.0

        for name, param in model.named_parameters():
            if name not in self._anchor_weights:
                continue
            delta = param.detach().to(device="cpu", dtype=torch.float32) - self._anchor_weights[name]
            # Project onto safety subspace U_out if available for this layer
            u = None
            for lname, u_mat in self._subspace_u.items():
                if name == lname or name.startswith(lname + "."):
                    u = u_mat
                    break
            if u is not None and delta.ndim == 2 and delta.shape[0] == u.shape[0]:
                proj = u @ (u.t() @ delta)
                drift = float(torch.linalg.vector_norm(proj).item())
            else:
                drift = float(torch.linalg.vector_norm(delta).item())

            per_layer[name] = drift
            total_sq += drift * drift
            if drift > worst_drift:
                worst_drift = drift
                worst_layer = name

        total_drift = float(total_sq ** 0.5)
        exceeded = total_drift > self.max_cumulative_drift
        if self.per_layer_drift_cap is not None and worst_drift > self.per_layer_drift_cap:
            exceeded = True

        return CumulativeDriftReport(
            exceeded=exceeded,
            total_subspace_drift=total_drift,
            budget=self.max_cumulative_drift,
            worst_layer=worst_layer,
            worst_layer_drift=worst_drift,
            per_layer_drift=per_layer,
        )


# ---------------------------------------------------------------------------
# 4. Canary Probe & Automatic Verified Checkpoint Rollback
#    (Defeats Multi-Layer Nonlinear Re-Entanglement!)
# ---------------------------------------------------------------------------

@dataclass
class CanaryCheckResult:
    step: int
    canary_score: float
    threshold: float
    passed: bool
    rolled_back: bool
    weights_sha256: str


class CanaryRollbackGuard:
    """Periodically evaluates a frozen safety canary function on the live model.
    If canary risk exceeds `max_canary_risk`, automatically restores the last
    verified safe checkpoint snapshot and returns `rolled_back=True`."""

    def __init__(
        self,
        canary_fn: Callable[[nn.Module], float],
        *,
        max_canary_risk: float = 0.35,
        check_every_steps: int = 10,
    ) -> None:
        self.canary_fn = canary_fn
        self.max_canary_risk = float(max_canary_risk)
        self.check_every_steps = max(1, int(check_every_steps))
        self._last_safe_state: Optional[Dict[str, torch.Tensor]] = None
        self._last_safe_sha256: str = ""
        self.rollback_count: int = 0
        self.history: List[CanaryCheckResult] = []

    @torch.no_grad()
    def snapshot_verified(self, model: nn.Module) -> str:
        """Save a CPU snapshot of verified trainable parameters and return its SHA-256."""
        self._last_safe_state = {
            name: p.detach().to("cpu").clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._last_safe_sha256 = compute_model_weights_sha256(model)
        return self._last_safe_sha256

    @torch.no_grad()
    def rollback(self, model: nn.Module) -> bool:
        if self._last_safe_state is None:
            return False
        for name, p in model.named_parameters():
            if name in self._last_safe_state:
                p.copy_(self._last_safe_state[name].to(device=p.device, dtype=p.dtype))
        self.rollback_count += 1
        return True

    def maybe_check(self, model: nn.Module, step: int) -> Optional[CanaryCheckResult]:
        if self._last_safe_state is None:
            self.snapshot_verified(model)
        if step % self.check_every_steps != 0:
            return None

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                risk = float(self.canary_fn(model))
        finally:
            if was_training:
                model.train()

        passed = risk <= self.max_canary_risk
        rolled_back = False
        if not passed:
            rolled_back = self.rollback(model)
            sha = self._last_safe_sha256
        else:
            sha = self.snapshot_verified(model)

        res = CanaryCheckResult(
            step=step,
            canary_score=risk,
            threshold=self.max_canary_risk,
            passed=passed,
            rolled_back=rolled_back,
            weights_sha256=sha,
        )
        self.history.append(res)
        return res


# ---------------------------------------------------------------------------
# 5. Zero-Pickle Safetensors Probe Migration Helper
# ---------------------------------------------------------------------------

def convert_probes_to_safetensors(
    pt_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert a `.pt` probe dictionary into a zero-pickle `.safetensors` file preserving scalar metadata."""
    import json
    from safetensors.torch import save_file

    src = Path(pt_path)
    dst = Path(out_path) if out_path is not None else src.with_suffix(".safetensors")
    loaded = torch.load(src, map_location="cpu", weights_only=True)
    flat_tensors: Dict[str, torch.Tensor] = {}
    non_tensors: Dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, torch.Tensor):
            flat_tensors[prefix or "tensor"] = obj.detach().cpu().contiguous()
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}__{k}" if prefix else str(k)
                _flatten(v, key)
        else:
            non_tensors[prefix or "value"] = obj

    _flatten(loaded)
    if not flat_tensors:
        raise ValueError(f"No tensors found in {src}")
    metadata = {"ojasx_non_tensors": json.dumps(non_tensors)} if non_tensors else None
    save_file(flat_tensors, str(dst), metadata=metadata)
    return dst


__all__ = [
    "ExternalSigner",
    "compute_model_weights_sha256",
    "compute_weight_delta_sha256",
    "CumulativeDriftReport",
    "CumulativeSubspaceDriftGuard",
    "CanaryCheckResult",
    "CanaryRollbackGuard",
    "convert_probes_to_safetensors",
]
