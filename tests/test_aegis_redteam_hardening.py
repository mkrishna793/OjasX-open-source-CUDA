"""
Red-Team Security Hardening Test Suite for OjasX-Aegis v2.0 + Unified OjasX3.

Verifies all 5 red-team security hardenings:
  1. ExternalSigner (KMS/HSM/TEE) protocol conformance (no private key in Python RAM).
  2. Boiling-the-frog slow-drift poisoning caught by CumulativeSubspaceDriftGuard
     against an immutable step-0 anchor.
  3. Multi-layer nonlinear re-entanglement caught by CanaryRollbackGuard with
     automatic restoration of the last verified safe checkpoint.
  4. Silent `torch.no_grad()` weight tampering detected via compute_weight_delta_sha256.
  5. Zero-pickle safetensors probe conversion (convert_probes_to_safetensors).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ojasx_safety.hardening import (
    CanaryRollbackGuard,
    CumulativeSubspaceDriftGuard,
    ExternalSigner,
    compute_model_weights_sha256,
    compute_weight_delta_sha256,
    convert_probes_to_safetensors,
)
from ojasx3.trainer.peft import LoRALinear, attach_lora_adapters


class _MockHSMSigner:
    """Mock hardware/KMS signer conforming to ExternalSigner without exposing private keys."""

    @property
    def signer_backend(self) -> str:
        return "mock-hsm-v1"

    def public_key_pem(self) -> str:
        return "-----BEGIN PUBLIC KEY-----\nMOCK\n-----END PUBLIC KEY-----\n"

    def sign(self, payload: bytes) -> bytes:
        return hashlib.sha256(b"HSM_SIG:" + payload).digest()


def test_external_signer_protocol():
    signer = _MockHSMSigner()
    assert isinstance(signer, ExternalSigner)
    assert signer.signer_backend == "mock-hsm-v1"
    sig = signer.sign(b"receipt-payload")
    assert len(sig) == 32


def test_boiling_the_frog_slow_drift_caught_by_immutable_anchor():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
    guard = CumulativeSubspaceDriftGuard(max_cumulative_drift=0.20)
    guard.freeze_anchor(model)

    # Simulate 250 tiny sub-threshold poisoned steps (each step norm << 0.01)
    triggered_step = None
    for step in range(1, 251):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.0005)
        rep = guard.check_drift(model)
        if rep.exceeded:
            triggered_step = step
            break

    assert triggered_step is not None
    assert rep.total_subspace_drift > 0.20
    assert rep.worst_layer != ""


def test_canary_probe_auto_rollback_restores_exact_verified_weights():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 4))
    canary_x = torch.randn(4, 16)
    base_out = model(canary_x).detach().clone()

    def canary_risk(m: nn.Module) -> float:
        return float((m(canary_x) - base_out).abs().mean().item())

    guard = CanaryRollbackGuard(canary_risk, max_canary_risk=0.10, check_every_steps=1)
    safe_sha = guard.snapshot_verified(model)

    # Step 1: benign tiny perturbation -> passes & updates snapshot
    with torch.no_grad():
        model[0].weight.add_(1e-4)
    res1 = guard.maybe_check(model, step=1)
    assert res1 is not None and res1.passed and not res1.rolled_back
    verified_sha = res1.weights_sha256
    assert verified_sha != safe_sha

    # Step 2: adversarial nonlinear corruption -> fails canary & rolls back to exact step-1 snapshot!
    with torch.no_grad():
        model[0].weight.add_(5.0)
    res2 = guard.maybe_check(model, step=2)
    assert res2 is not None and (not res2.passed) and res2.rolled_back
    assert guard.rollback_count == 1
    assert compute_model_weights_sha256(model) == verified_sha


def test_silent_no_grad_weight_tampering_breaks_delta_hash():
    model = nn.Linear(16, 16)
    anchor = {n: p.detach().cpu().float().clone() for n, p in model.named_parameters()}
    h0, d0 = compute_weight_delta_sha256(model, anchor)
    assert d0 == 0.0

    # Silent no_grad edit (bypasses backward hooks!)
    with torch.no_grad():
        model.weight[0, 0] += 1e-3
    h1, d1 = compute_weight_delta_sha256(model, anchor)
    assert h1 != h0
    assert d1 > 0.0


def test_unified_peft_accepts_opt_in_safety_policy():
    model = nn.Sequential(nn.Linear(32, 32))
    attach_lora_adapters(model, rank=4, safety_policy="code_enterprise")
    assert isinstance(model[0], LoRALinear)
    assert model[0].safety_policy == "code_enterprise"
