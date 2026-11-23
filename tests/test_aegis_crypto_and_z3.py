"""Tests for OjasX-Aegis policy, C++20 native gate, Ed25519/Merkle attestation, and Z3 proofs."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ojasx_safety
from ojasx_safety import attestation, contracts, native_gate, policy
from ojasx_safety.attestation import (
    AttestationError,
    SafetyLedger,
    compute_checkpoint_sha256,
    compute_grad_sketch,
    compute_merkle_proof,
    compute_merkle_root,
    generate_keypair,
    seal_checkpoint,
    verify_checkpoint_attestation,
    verify_merkle_proof,
    verify_receipt,
)
from ojasx_safety.contracts import SafetyPolicy, SafetyProfile
from ojasx_safety.native_gate import (
    AEGIS_ERR_FLAG_WEIGHT_CONFLICT,
    AEGIS_ERR_MASK_TAMPERED,
    AEGIS_ERR_NON_FINITE,
    AEGIS_ERR_INVALID_WEIGHT,
    AEGIS_OK,
    EXPECTED_GATE_VERSION,
    NativeGateError,
    apply_zero_mask,
    apply_zero_mask_with_hash,
    compute_mask_hash,
    compute_merkle_step,
    gate_version,
    load_native_gate,
    verify_mask_integrity,
)
from ojasx_safety.policy import (
    CODE_ENTERPRISE,
    CUSTOM_STRICT,
    POLICY_PRESETS,
    PUBLIC_GENERAL,
    RL_AGENTIC,
    SCIENTIFIC_MODULAR,
    TIER0_IMMUTABLE_REDLINES,
    PolicyViolationError,
    canonical_policy_json,
    get_policy,
    policy_hash,
    validate_policy,
    with_policy_overrides,
)


ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_and_lazy_dir() -> None:
    assert ojasx_safety.contracts is contracts
    assert ojasx_safety.policy is policy
    assert ojasx_safety.attestation is attestation
    assert ojasx_safety.native_gate is native_gate
    exported_names = dir(ojasx_safety)
    for lazy_name in ("firewall", "autograd", "eval_harness"):
        assert lazy_name in exported_names


def test_policy_presets_canonical_hash_and_tier0_invariants() -> None:
    hashes = {}
    for profile in SafetyProfile:
        preset = POLICY_PRESETS[profile]
        assert validate_policy(preset) is preset
        h = policy_hash(preset)
        assert len(h) == 64
        hashes[profile] = h

    assert len(set(hashes.values())) == len(SafetyProfile)
    assert PUBLIC_GENERAL.profile == SafetyProfile.PUBLIC_GENERAL
    assert SCIENTIFIC_MODULAR.profile == SafetyProfile.SCIENTIFIC_MODULAR
    assert CODE_ENTERPRISE.profile == SafetyProfile.CODE_ENTERPRISE
    assert RL_AGENTIC.profile == SafetyProfile.RL_AGENTIC
    assert CUSTOM_STRICT.profile == SafetyProfile.CUSTOM_STRICT

    # Deterministic across reordered domain tuples and profile string lookup
    reordered = dataclasses.replace(
        PUBLIC_GENERAL,
        redline_domains=tuple(reversed(PUBLIC_GENERAL.redline_domains)),
        dual_use_domains=tuple(reversed(PUBLIC_GENERAL.dual_use_domains)),
    )
    assert canonical_policy_json(reordered) == canonical_policy_json(PUBLIC_GENERAL)
    assert policy_hash(reordered) == policy_hash(PUBLIC_GENERAL)
    assert policy_hash(get_policy("rl_agentic")) == policy_hash(RL_AGENTIC)

    # Disabling ANY immutable Tier-0 redline with fail_closed=True must fail closed
    for redline in TIER0_IMMUTABLE_REDLINES:
        reduced = tuple(d for d in TIER0_IMMUTABLE_REDLINES if d != redline)
        bad_policy = dataclasses.replace(PUBLIC_GENERAL, redline_domains=reduced, fail_closed=True)
        with pytest.raises(PolicyViolationError):
            validate_policy(bad_policy)
        with pytest.raises(PolicyViolationError):
            policy_hash(bad_policy)
        with pytest.raises(PolicyViolationError):
            with_policy_overrides(PUBLIC_GENERAL, redline_domains=reduced)

    # Invalid numeric thresholds fail closed
    for bad_kwargs in (
        {"classifier_block_threshold": float("nan")},
        {"classifier_block_threshold": 0.0},
        {"classifier_block_threshold": 1.2},
        {"max_kl_divergence": -0.1},
        {"max_grad_zscore": float("inf")},
        {"lora_safety_projection_strength": 1.5},
        {"blocklist_escalation_terms": 0},
    ):
        with pytest.raises(PolicyViolationError):
            get_policy(SafetyProfile.PUBLIC_GENERAL, **bad_kwargs)


def test_native_gate_exports_and_auto_build_on_missing(tmp_path: Path) -> None:
    lib_path = native_gate.default_library_path()
    assert lib_path.is_file()
    assert gate_version() == EXPECTED_GATE_VERSION

    if sys.platform == "darwin":
        nm = subprocess.run(["nm", "-gU", str(lib_path)], capture_output=True, text=True, check=True)
        exported = [line.split()[-1] for line in nm.stdout.strip().splitlines() if line.strip()]
        assert sorted(exported) == sorted([
            "_aegis_apply_zero_mask_f32",
            "_aegis_compute_merkle_step",
            "_aegis_gate_version",
            "_aegis_verify_mask_integrity",
        ])

    # Auto-build-on-missing inside isolated temporary csrc directory
    tmp_csrc = tmp_path / "csrc"
    tmp_csrc.mkdir()
    shutil.copy2(ROOT / "ojasx_safety" / "csrc" / "ojasx_aegis_gate.cpp", tmp_csrc / "ojasx_aegis_gate.cpp")
    built_dylib = tmp_csrc / native_gate.default_library_name()
    assert not built_dylib.exists()

    lib = load_native_gate(csrc_dir=tmp_csrc)
    assert built_dylib.is_file()
    assert lib.aegis_gate_version().decode("utf-8") == EXPECTED_GATE_VERSION


def test_native_gate_zero_leakage_and_fail_closed_ffi() -> None:
    lib = load_native_gate()

    # 1D loss tensor masking, including +inf and nan on blocked indices
    raw_losses = torch.tensor([1.25, float("inf"), -2.5, float("nan")], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    flags = [0, 1, 0, 1]

    masked, mask_hash = apply_zero_mask_with_hash(raw_losses, weights, flags=flags)
    assert len(mask_hash) == 64
    assert mask_hash == compute_mask_hash(weights, flags=flags)
    assert verify_mask_integrity(weights, flags=flags, expected_mask_hash=mask_hash) == mask_hash

    assert masked[0].item() == pytest.approx(1.25)
    assert masked[2].item() == pytest.approx(-2.5)
    raw_bits = masked.numpy().view(np.uint32)
    assert raw_bits[1] == 0  # Exact +0.0f, never -0.0f or NaN
    assert raw_bits[3] == 0

    # 2D gradient masking with autograd forward + backward verification
    x = torch.tensor([[2.0, -3.0], [4.0, 5.0], [-1.0, 6.0]], dtype=torch.float32, requires_grad=True)
    w_t = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    masked_x = apply_zero_mask(x, w_t)
    assert masked_x[1].tolist() == [0.0, 0.0]
    (masked_x.sum()).backward()
    assert x.grad is not None
    assert x.grad[0].tolist() == [1.0, 1.0]
    assert x.grad[1].tolist() == [0.0, 0.0]
    assert x.grad[2].tolist() == [1.0, 1.0]

    # Fail-closed rejections: Python wrappers + direct C++ CDLL zeroing check
    for bad_weights in (
        np.array([1.0, float("nan")], dtype=np.float32),
        np.array([1.0, float("inf")], dtype=np.float32),
        np.array([1.0, 0.5], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
        [1.0, 1.0 + 1e-10],  # Precision bypass attempt in float64
    ):
        with pytest.raises(NativeGateError):
            apply_zero_mask(np.array([3.0, 4.0], dtype=np.float32), bad_weights)

    # Flag/weight conflict rejected
    with pytest.raises(NativeGateError):
        apply_zero_mask(
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            flags=[0, 1],
        )

    # Tampered expected_mask_hash rejected
    with pytest.raises(NativeGateError):
        apply_zero_mask(
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            expected_mask_hash="0" * 64,
        )

    # Allowed slot containing NaN rejected fail-closed
    with pytest.raises(NativeGateError):
        apply_zero_mask(
            np.array([float("nan"), 4.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )

    # Direct ctypes FFI call verifies values_out is scrubbed to 0.0f on fail-closed error
    vals_in = np.array([7.0, 8.0], dtype=np.float32)
    vals_out = np.array([99.0, 99.0], dtype=np.float32)
    bad_w = np.array([1.0, 0.5], dtype=np.float32)
    rc = lib.aegis_apply_zero_mask_f32(
        vals_in.ctypes.data,
        vals_out.ctypes.data,
        bad_w.ctypes.data,
        None,
        2,
        1,
        None,
        None,
    )
    assert rc == AEGIS_ERR_INVALID_WEIGHT
    assert np.all(vals_out == 0.0)

    # Direct flag conflict, NaN, and tampered digest error codes
    w_bin = np.array([1.0, 0.0], dtype=np.float32)
    f_bad = np.array([1, 0], dtype=np.uint8)
    assert lib.aegis_verify_mask_integrity(w_bin.ctypes.data, f_bad.ctypes.data, 2, None, None) == AEGIS_ERR_FLAG_WEIGHT_CONFLICT
    w_nan = np.array([1.0, float("nan")], dtype=np.float32)
    assert lib.aegis_verify_mask_integrity(w_nan.ctypes.data, None, 2, None, None) == AEGIS_ERR_NON_FINITE
    bad_exp = np.full(32, 0xAA, dtype=np.uint8)
    assert lib.aegis_verify_mask_integrity(w_bin.ctypes.data, None, 2, bad_exp.ctypes.data, None) == AEGIS_ERR_MASK_TAMPERED

    # C++ Merkle step matches hashlib.sha256 byte-for-byte
    h_inputs = [hashlib.sha256(f"field-{i}".encode()).hexdigest() for i in range(5)]
    c_step = compute_merkle_step(*h_inputs, step=42)
    py_step = hashlib.sha256(
        b"".join(bytes.fromhex(h) for h in h_inputs) + struct.pack(">Q", 42)
    ).hexdigest()
    assert c_step == py_step


def test_merkle_tree_ed25519_and_tamper_attacks(tmp_path: Path) -> None:
    pol = get_policy(SafetyProfile.RL_AGENTIC)
    pol_h = policy_hash(pol)
    priv, pub = generate_keypair()

    batch1 = [b"clean-sample-0", b"clean-sample-1", b"blocked-cbrn-2"]
    batch2 = [b"clean-sample-3", b"clean-sample-4", b"clean-sample-5"]
    merkle1 = compute_merkle_root(batch1)
    merkle2 = compute_merkle_root(batch2)

    # Inclusion proof works and rejects 1-byte flip
    proof = compute_merkle_proof(batch1, 1)
    assert verify_merkle_proof(batch1[1], 1, proof, merkle1, len(batch1))
    assert not verify_merkle_proof(b"clean-sample-9", 1, proof, merkle1, len(batch1))

    mask1 = compute_mask_hash([1.0, 1.0, 0.0], flags=[0, 0, 1])
    mask2 = compute_mask_hash([1.0, 1.0, 1.0], flags=[0, 0, 0])
    grad1 = compute_grad_sketch(torch.tensor([0.1, -0.2, 0.0], dtype=torch.float32))
    grad2 = compute_grad_sketch(torch.tensor([0.05, 0.15, -0.04], dtype=torch.float32))

    ledger = SafetyLedger(pol, run_id="test-run-01", use_native_gate=True)
    ledger.record_step(step=1, merkle_root=merkle1, mask_hash=mask1, grad_sketch=grad1, blocked_samples=1)
    ledger.record_step(step=2, merkle_root=merkle2, mask_hash=mask2, grad_sketch=grad2, blocked_samples=0)
    assert ledger.verify_chain(expected_policy_hash=pol_h, require_non_empty=True)

    ckpt = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)}
    ckpt_file = tmp_path / "model.bin"
    ckpt_file.write_bytes(b"AEGIS-MODEL-WEIGHTS-V1")

    receipt = seal_checkpoint(ckpt, ledger, private_key=priv)
    assert receipt.verified is True
    assert verify_receipt(receipt, expected_policy_hash=pol_h, expected_public_key=pub, ledger=ledger)
    assert verify_checkpoint_attestation(
        ckpt, receipt, expected_policy=pol, expected_public_key=pub, ledger=ledger
    )

    file_receipt = seal_checkpoint(ckpt_file, ledger, private_key=priv)
    assert verify_checkpoint_attestation(
        ckpt_file, file_receipt, expected_policy=pol, expected_public_key=pub, ledger=ledger
    )

    # ATTACK 1: Flip 1 byte of dataset batch sample
    tampered_batch1 = list(batch1)
    b0 = bytearray(tampered_batch1[0])
    b0[0] ^= 0x01
    tampered_batch1[0] = bytes(b0)
    tampered_merkle1 = compute_merkle_root(tampered_batch1)
    assert tampered_merkle1 != merkle1

    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger._entries[0] = dataclasses.replace(tampered_ledger._entries[0], merkle_root=tampered_merkle1)
    assert not tampered_ledger.verify_chain()
    assert not verify_receipt(copy.deepcopy(receipt), ledger=tampered_ledger)
    with pytest.raises(AttestationError):
        seal_checkpoint(ckpt, tampered_ledger, private_key=priv)

    # ATTACK 2: Alter policy threshold
    weakened_pol = with_policy_overrides(pol, classifier_block_threshold=0.95)
    assert policy_hash(weakened_pol) != pol_h
    assert not verify_receipt(copy.deepcopy(receipt), expected_policy_hash=policy_hash(weakened_pol))
    assert not verify_checkpoint_attestation(ckpt, copy.deepcopy(receipt), expected_policy=weakened_pol)

    # ATTACK 3: Swap Ed25519 signature key (attacker re-signs with attacker's key, or swaps sig/key)
    attacker_priv, attacker_pub = generate_keypair()
    forged_key_receipt = copy.deepcopy(receipt)
    attestation.sign_receipt(forged_key_receipt, attacker_priv)
    # Valid self-signed under attacker key, MUST fail pinned expected_public_key!
    assert not verify_receipt(forged_key_receipt, expected_public_key=pub)
    assert not verify_checkpoint_attestation(
        ckpt, forged_key_receipt, expected_policy=pol, expected_public_key=pub
    )

    # Swapping signature only while keeping victim public_key_pem fails Ed25519 verification
    swapped_sig_receipt = copy.deepcopy(receipt)
    swapped_sig_receipt.signature_hex = forged_key_receipt.signature_hex
    assert not verify_receipt(swapped_sig_receipt)

    # ATTACK 4: Forge step count on receipt and on ledger
    forged_steps_receipt = copy.deepcopy(receipt)
    forged_steps_receipt.total_steps = 99
    assert not verify_receipt(forged_steps_receipt, expected_policy_hash=pol_h, expected_public_key=pub)

    with pytest.raises(AttestationError):
        ledger.record_step(step=99, merkle_root=merkle2, mask_hash=mask2, grad_sketch=grad2)

    tampered_step_ledger = copy.deepcopy(ledger)
    tampered_step_ledger._entries.append(
        dataclasses.replace(tampered_step_ledger._entries[-1], step=3)
    )
    assert not tampered_step_ledger.verify_chain()
    assert not verify_receipt(copy.deepcopy(receipt), ledger=tampered_step_ledger)

    # ATTACK 5: Flip 1 byte in checkpoint weights / file
    tampered_ckpt = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0001]], dtype=torch.float32)}
    assert compute_checkpoint_sha256(tampered_ckpt) != receipt.checkpoint_sha256
    assert not verify_checkpoint_attestation(
        tampered_ckpt, copy.deepcopy(receipt), expected_policy=pol, expected_public_key=pub
    )
    with pytest.raises(AttestationError):
        verify_checkpoint_attestation(
            tampered_ckpt,
            copy.deepcopy(receipt),
            expected_policy=pol,
            expected_public_key=pub,
            raise_on_failure=True,
        )

    ckpt_bytes = bytearray(ckpt_file.read_bytes())
    ckpt_bytes[-1] ^= 0x01
    ckpt_file.write_bytes(bytes(ckpt_bytes))
    assert not verify_checkpoint_attestation(
        ckpt_file, copy.deepcopy(file_receipt), expected_policy=pol, expected_public_key=pub
    )


def test_formal_z3_verification_suite() -> None:
    from tools.verify_aegis_z3 import verify

    report = verify()
    assert report["all_proofs_passed"] is True
    assert report["all_counterexamples_verified"] is True

    proof_names = {p["name"] for p in report["proofs"]}
    ce_names = {c["name"] for c in report["counterexamples"]}
    assert {
        "zero_leakage_redline_annihilates_gradient_qfnra",
        "zero_leakage_native_gate_blocks_fractional_or_tampered_weight_qfnra",
        "tier0_redlines_immutable_when_fail_closed",
        "gram_safety_projection_is_strictly_orthogonal_qfnra",
        "gram_pythagorean_norm_and_idempotence_qfnra",
        "gram_residual_isolation_zero_safety_drift_qfnra",
        "rl_kl_dual_barrier_blocks_excess_drift",
        "rl_kl_primal_dual_saddle_penalizes_violation_qfnra",
        "tampered_ledger_batch_or_step_rejected_fail_closed",
        "mismatched_policy_key_ckpt_or_steps_rejected_fail_closed",
    }.issubset(proof_names)
    assert len(ce_names) >= 5

    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_aegis_z3.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"verify_aegis_z3.py failed:\n{proc.stdout}\n{proc.stderr}"
    assert "10/10 proofs verified" in proc.stdout
