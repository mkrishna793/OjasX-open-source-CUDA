"""Cryptographic Ed25519 attestation, Merkle trees, and hash-chained SafetyLedger."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from . import native_gate
from .contracts import AttestationReceipt, SafetyPolicy
from .policy import policy_hash as compute_policy_hash


_GENESIS_DOMAIN = b"OJASX_AEGIS_GENESIS_V1:"
_RECEIPT_DOMAIN = b"OJASX_AEGIS_RECEIPT_V1:"
_STATE_DICT_DOMAIN = b"OJASX_AEGIS_STATE_DICT_V1:"
_EMPTY_EVAL_DIGEST = hashlib.sha256(b"OJASX_AEGIS_NO_EVAL_V1").hexdigest()
_EMPTY_GRAD_SKETCH = hashlib.sha256(b"OJASX_AEGIS_EMPTY_GRAD_V1").hexdigest()


class AttestationError(ValueError):
    """Fail-closed error raised when attestation, ledger, or checkpoint sealing fails."""


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_sha256_hex(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AttestationError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise AttestationError(f"{label} contains invalid hex characters") from exc
    if len(raw) != 32:
        raise AttestationError(f"{label} must represent 32 bytes")
    return raw.hex()


def _is_valid_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a fresh Ed25519 signing keypair using cryptography."""
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


generate_signing_keypair = generate_keypair


def public_key_to_pem(public_key: Ed25519PublicKey) -> str:
    if not isinstance(public_key, Ed25519PublicKey):
        raise AttestationError(f"Expected Ed25519PublicKey, got {type(public_key).__name__}")
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def public_key_from_pem(pem: str) -> Ed25519PublicKey:
    if not isinstance(pem, str) or not pem.strip():
        raise AttestationError("public_key_pem must be a non-empty PEM string")
    try:
        key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except Exception as exc:
        raise AttestationError(f"Invalid public key PEM: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AttestationError(f"Expected Ed25519 public key, got {type(key).__name__}")
    return key


def private_key_to_pem(private_key: Ed25519PrivateKey) -> str:
    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationError(f"Expected Ed25519PrivateKey, got {type(private_key).__name__}")
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def private_key_from_pem(pem: str) -> Ed25519PrivateKey:
    if not isinstance(pem, str) or not pem.strip():
        raise AttestationError("private_key_pem must be a non-empty PEM string")
    try:
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    except Exception as exc:
        raise AttestationError(f"Invalid private key PEM: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AttestationError(f"Expected Ed25519 private key, got {type(key).__name__}")
    return key


def _tensor_to_canonical_bytes(tensor: torch.Tensor) -> bytes:
    t_cpu = tensor.detach().to(device="cpu").contiguous()
    header = f"{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode("utf-8")
    if t_cpu.numel() == 0:
        return header
    raw_u8 = t_cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    return header + raw_u8


def _serialize_item_bytes(item: Any) -> bytes:
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, torch.Tensor):
        return _tensor_to_canonical_bytes(item)
    if isinstance(item, np.ndarray):
        arr = np.ascontiguousarray(item)
        header = f"{arr.dtype}:{arr.shape}:".encode("utf-8")
        return header + arr.tobytes()
    return json.dumps(
        item,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_sample_payload(item: Any) -> bytes:
    """Compute domain-separated 32-byte leaf digest for a batch sample."""
    return hashlib.sha256(b"\x00" + _serialize_item_bytes(item)).digest()


def _merkle_tree_levels(leaf_digests: Sequence[bytes]) -> List[List[bytes]]:
    if not leaf_digests:
        raise AttestationError("Cannot build Merkle tree over an empty batch")
    levels: List[List[bytes]] = [list(leaf_digests)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        if len(cur) % 2 == 1:
            cur = cur + [cur[-1]]
        nxt = [
            hashlib.sha256(b"\x01" + cur[i] + cur[i + 1]).digest()
            for i in range(0, len(cur), 2)
        ]
        levels.append(nxt)
    return levels


def _finalize_merkle_root(top_node: bytes, total_leaves: int) -> str:
    return hashlib.sha256(b"\x02" + struct.pack(">Q", total_leaves) + top_node).hexdigest()


def compute_merkle_root(items: Sequence[Any]) -> str:
    """Compute domain-separated SHA-256 binary Merkle tree root over batch samples."""
    if not items:
        raise AttestationError("Cannot compute Merkle root of empty batch")
    leaves = [hash_sample_payload(item) for item in items]
    levels = _merkle_tree_levels(leaves)
    return _finalize_merkle_root(levels[-1][0], len(items))


def compute_merkle_root_from_digests(digests: Sequence[Union[bytes, str]]) -> str:
    """Compute SHA-256 binary Merkle tree root over pre-hashed 32-byte/hex digests."""
    if not digests:
        raise AttestationError("Cannot compute Merkle root of empty digest list")
    leaves: List[bytes] = []
    for idx, d in enumerate(digests):
        if isinstance(d, (bytes, bytearray, memoryview)):
            raw = bytes(d)
            if len(raw) != 32:
                raise AttestationError(f"Digest[{idx}] must be 32 bytes, got {len(raw)}")
        elif isinstance(d, str):
            raw = bytes.fromhex(_normalize_sha256_hex(d, f"digest[{idx}]"))
        else:
            raise AttestationError(f"Unsupported digest type at index {idx}: {type(d).__name__}")
        leaves.append(hashlib.sha256(b"\x00" + raw).digest())
    levels = _merkle_tree_levels(leaves)
    return _finalize_merkle_root(levels[-1][0], len(digests))


def compute_merkle_proof(items: Sequence[Any], index: int) -> List[Tuple[str, bool]]:
    """Return Merkle inclusion proof as list of (sibling_hex, sibling_is_right)."""
    if not items:
        raise AttestationError("Cannot build Merkle proof for empty batch")
    if index < 0 or index >= len(items):
        raise AttestationError(f"Merkle proof index {index} out of bounds [0, {len(items)})")
    leaves = [hash_sample_payload(item) for item in items]
    levels = _merkle_tree_levels(leaves)
    proof: List[Tuple[str, bool]] = []
    idx = index
    for level in levels[:-1]:
        padded = level + [level[-1]] if (len(level) % 2 == 1) else level
        if idx % 2 == 0:
            proof.append((padded[idx + 1].hex(), True))
        else:
            proof.append((padded[idx - 1].hex(), False))
        idx //= 2
    return proof


def verify_merkle_proof(
    item: Any,
    index: int,
    proof: Sequence[Tuple[str, bool]],
    root_hex: str,
    total_leaves: int,
) -> bool:
    """Verify a Merkle inclusion proof against root_hex and total_leaves."""
    if total_leaves <= 0 or index < 0 or index >= total_leaves or not _is_valid_sha256_hex(root_hex):
        return False
    node = hash_sample_payload(item)
    idx = index
    for sibling_hex, sibling_is_right in proof:
        if not _is_valid_sha256_hex(sibling_hex):
            return False
        sibling = bytes.fromhex(sibling_hex)
        if sibling_is_right:
            node = hashlib.sha256(b"\x01" + node + sibling).digest()
        else:
            node = hashlib.sha256(b"\x01" + sibling + node).digest()
        idx //= 2
    if idx != 0:
        return False
    expected = _finalize_merkle_root(node, total_leaves)
    return hmac.compare_digest(expected, root_hex.lower())


def compute_grad_sketch(grads: Any = None) -> str:
    """Compute canonical SHA-256 sketch over gradients, tensors, or telemetry arrays."""
    if grads is None:
        return _EMPTY_GRAD_SKETCH
    hasher = hashlib.sha256(b"OJASX_AEGIS_GRAD_SKETCH_V1:")
    if isinstance(grads, (bytes, bytearray, memoryview)):
        hasher.update(bytes(grads))
        return hasher.hexdigest()
    if isinstance(grads, torch.Tensor):
        hasher.update(_tensor_to_canonical_bytes(grads))
        return hasher.hexdigest()
    if isinstance(grads, np.ndarray):
        arr = np.ascontiguousarray(grads, dtype=np.float32)
        hasher.update(f"{arr.shape}:".encode("utf-8") + arr.tobytes())
        return hasher.hexdigest()
    if isinstance(grads, Mapping):
        for k in sorted(grads.keys(), key=str):
            hasher.update(str(k).encode("utf-8") + b":")
            hasher.update(_serialize_item_bytes(grads[k]))
        return hasher.hexdigest()
    if isinstance(grads, Sequence) and not isinstance(grads, str):
        for item in grads:
            hasher.update(_serialize_item_bytes(item))
        return hasher.hexdigest()
    hasher.update(_serialize_item_bytes(grads))
    return hasher.hexdigest()


@dataclass(frozen=True)
class LedgerEntry:
    step: int
    prev_hash: str
    merkle_root: str
    policy_hash: str
    mask_hash: str
    grad_sketch: str
    chain_hash: str
    blocked_samples: int = 0
    quarantined: bool = False


def compute_chain_step_hash(
    prev_chain_hash: str,
    merkle_root: str,
    policy_hash: str,
    mask_hash: str,
    grad_sketch: str,
    step: int,
    *,
    use_native_gate: bool = True,
) -> str:
    """Compute H_t = SHA256(H_{t-1} || merkle || policy_hash || mask_hash || grad_sketch || step)."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise AttestationError(f"step must be a non-negative integer, got {step!r}")
    prev_h = _normalize_sha256_hex(prev_chain_hash, "prev_chain_hash")
    merkle_h = _normalize_sha256_hex(merkle_root, "merkle_root")
    policy_h = _normalize_sha256_hex(policy_hash, "policy_hash")
    mask_h = _normalize_sha256_hex(mask_hash, "mask_hash")
    sketch_h = _normalize_sha256_hex(grad_sketch, "grad_sketch")

    py_digest = hashlib.sha256(
        bytes.fromhex(prev_h)
        + bytes.fromhex(merkle_h)
        + bytes.fromhex(policy_h)
        + bytes.fromhex(mask_h)
        + bytes.fromhex(sketch_h)
        + struct.pack(">Q", step)
    ).hexdigest()

    if use_native_gate:
        native_digest = native_gate.compute_merkle_step(
            prev_h, merkle_h, policy_h, mask_h, sketch_h, step
        )
        if not hmac.compare_digest(native_digest, py_digest):
            raise AttestationError("Native gate Merkle step diverged from SHA-256 specification")
    return py_digest


class SafetyLedger:
    """Append-only hash-chained audit ledger for training step safety telemetry."""

    def __init__(
        self,
        policy_or_hash: Union[SafetyPolicy, str],
        *,
        run_id: str = "aegis-run",
        genesis_hash: Optional[str] = None,
        use_native_gate: bool = True,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise AttestationError("run_id must be a non-empty string")
        self.run_id = run_id
        if isinstance(policy_or_hash, SafetyPolicy):
            self.policy_hash = compute_policy_hash(policy_or_hash)
        else:
            self.policy_hash = _normalize_sha256_hex(policy_or_hash, "policy_hash")

        if genesis_hash is None:
            self.genesis_hash = sha256_hex(
                _GENESIS_DOMAIN
                + self.run_id.encode("utf-8")
                + b":"
                + bytes.fromhex(self.policy_hash)
            )
        else:
            self.genesis_hash = _normalize_sha256_hex(genesis_hash, "genesis_hash")

        self.use_native_gate = bool(use_native_gate)
        self._entries: List[LedgerEntry] = []
        self._current_hash: str = self.genesis_hash
        self._start_step: Optional[int] = None
        self.blocked_samples: int = 0
        self.quarantined_batches: int = 0

    @property
    def entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def final_chain_hash(self) -> str:
        return self._current_hash

    @property
    def total_steps(self) -> int:
        return len(self._entries)

    def record_step(
        self,
        *,
        step: int,
        merkle_root: str,
        mask_hash: str,
        grad_sketch: str = _EMPTY_GRAD_SKETCH,
        policy_hash_override: Optional[str] = None,
        blocked_samples: int = 0,
        quarantined: bool = False,
    ) -> LedgerEntry:
        """Append a contiguous, policy-pinned step to the hash chain."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise AttestationError(f"step must be a non-negative integer, got {step!r}")
        if isinstance(blocked_samples, bool) or not isinstance(blocked_samples, int) or blocked_samples < 0:
            raise AttestationError(f"blocked_samples must be >= 0, got {blocked_samples!r}")

        if not self._entries:
            if step not in (0, 1):
                raise AttestationError(f"Initial ledger step must be 0 or 1, got {step}")
            self._start_step = step
        else:
            assert self._start_step is not None
            expected = self._start_step + len(self._entries)
            if step != expected:
                raise AttestationError(
                    f"Non-contiguous ledger step {step}: expected {expected}"
                )

        step_policy = (
            _normalize_sha256_hex(policy_hash_override, "policy_hash_override")
            if policy_hash_override is not None
            else self.policy_hash
        )
        if not hmac.compare_digest(step_policy, self.policy_hash):
            raise AttestationError("Step policy_hash does not match ledger policy_hash")

        merkle_h = _normalize_sha256_hex(merkle_root, "merkle_root")
        mask_h = _normalize_sha256_hex(mask_hash, "mask_hash")
        sketch_h = _normalize_sha256_hex(grad_sketch, "grad_sketch")

        chain_hash = compute_chain_step_hash(
            self._current_hash,
            merkle_h,
            step_policy,
            mask_h,
            sketch_h,
            step,
            use_native_gate=self.use_native_gate,
        )
        entry = LedgerEntry(
            step=step,
            prev_hash=self._current_hash,
            merkle_root=merkle_h,
            policy_hash=step_policy,
            mask_hash=mask_h,
            grad_sketch=sketch_h,
            chain_hash=chain_hash,
            blocked_samples=blocked_samples,
            quarantined=bool(quarantined),
        )
        self._entries.append(entry)
        self._current_hash = chain_hash
        self.blocked_samples += blocked_samples
        if quarantined:
            self.quarantined_batches += 1
        return entry

    append_step = record_step

    def verify_chain(
        self,
        expected_policy_hash: Optional[str] = None,
        *,
        require_non_empty: bool = False,
    ) -> bool:
        """Recompute and verify every link, step counter, and summary counter in the ledger."""
        if not _is_valid_sha256_hex(self.policy_hash) or not _is_valid_sha256_hex(self.genesis_hash):
            return False
        if expected_policy_hash is not None:
            if not _is_valid_sha256_hex(expected_policy_hash):
                return False
            if not hmac.compare_digest(self.policy_hash.lower(), expected_policy_hash.lower()):
                return False
        if require_non_empty and not self._entries:
            return False

        prev = self.genesis_hash.lower()
        expected_step: Optional[int] = None
        total_blocked = 0
        total_quarantined = 0

        for idx, entry in enumerate(self._entries):
            if isinstance(entry.step, bool) or not isinstance(entry.step, int):
                return False
            if idx == 0:
                if entry.step not in (0, 1):
                    return False
                expected_step = entry.step
            elif entry.step != expected_step:
                return False

            for h in (entry.prev_hash, entry.merkle_root, entry.policy_hash, entry.mask_hash, entry.grad_sketch, entry.chain_hash):
                if not _is_valid_sha256_hex(h):
                    return False

            if not hmac.compare_digest(entry.prev_hash.lower(), prev):
                return False
            if not hmac.compare_digest(entry.policy_hash.lower(), self.policy_hash.lower()):
                return False

            try:
                recomputed = compute_chain_step_hash(
                    prev,
                    entry.merkle_root,
                    entry.policy_hash,
                    entry.mask_hash,
                    entry.grad_sketch,
                    entry.step,
                    use_native_gate=self.use_native_gate,
                )
            except (AttestationError, native_gate.NativeGateError):
                return False

            if not hmac.compare_digest(entry.chain_hash.lower(), recomputed.lower()):
                return False

            if isinstance(entry.blocked_samples, bool) or entry.blocked_samples < 0:
                return False
            total_blocked += entry.blocked_samples
            total_quarantined += int(bool(entry.quarantined))
            prev = entry.chain_hash.lower()
            assert expected_step is not None
            expected_step += 1

        if not _is_valid_sha256_hex(self._current_hash):
            return False
        if not hmac.compare_digest(self._current_hash.lower(), prev):
            return False
        if total_blocked != self.blocked_samples or total_quarantined != self.quarantined_batches:
            return False
        return True


def compute_checkpoint_sha256(
    checkpoint: Union[str, Path, bytes, bytearray, memoryview, Mapping[str, Any]],
) -> str:
    """Compute deterministic SHA-256 digest of checkpoint bytes, file, directory, or state_dict."""
    if isinstance(checkpoint, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(checkpoint)).hexdigest()

    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        if not path.exists():
            raise AttestationError(f"Checkpoint path does not exist: {path}")
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        if path.is_dir():
            hasher = hashlib.sha256(b"OJASX_AEGIS_CKPT_DIR_V1:")
            files = sorted(p for p in path.rglob("*") if p.is_file())
            for fp in files:
                rel = fp.relative_to(path).as_posix().encode("utf-8")
                file_hash = compute_checkpoint_sha256(fp)
                hasher.update(struct.pack(">I", len(rel)) + rel + bytes.fromhex(file_hash))
            return hasher.hexdigest()
        raise AttestationError(f"Unsupported checkpoint path type: {path}")

    if isinstance(checkpoint, Mapping):
        hasher = hashlib.sha256(_STATE_DICT_DOMAIN)
        for key in sorted(checkpoint.keys(), key=str):
            k_bytes = str(key).encode("utf-8")
            val_bytes = _serialize_item_bytes(checkpoint[key])
            hasher.update(struct.pack(">I", len(k_bytes)) + k_bytes)
            hasher.update(struct.pack(">Q", len(val_bytes)) + val_bytes)
        return hasher.hexdigest()

    raise AttestationError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")


def canonical_receipt_payload(receipt: AttestationReceipt) -> bytes:
    """Build canonical signed payload for an AttestationReceipt."""
    body = {
        "run_id": str(receipt.run_id),
        "policy_hash": str(receipt.policy_hash).lower(),
        "genesis_hash": str(receipt.genesis_hash).lower(),
        "final_chain_hash": str(receipt.final_chain_hash).lower(),
        "total_steps": int(receipt.total_steps),
        "blocked_samples": int(receipt.blocked_samples),
        "quarantined_batches": int(receipt.quarantined_batches),
        "eval_digest": str(receipt.eval_digest).lower(),
        "checkpoint_sha256": str(receipt.checkpoint_sha256).lower(),
        "public_key_pem": str(receipt.public_key_pem),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return _RECEIPT_DOMAIN + encoded.encode("utf-8")


def sign_receipt(
    receipt: AttestationReceipt,
    private_key: Ed25519PrivateKey,
) -> AttestationReceipt:
    """Sign an AttestationReceipt with Ed25519PrivateKey and verify immediately."""
    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationError(f"Expected Ed25519PrivateKey, got {type(private_key).__name__}")
    receipt.public_key_pem = public_key_to_pem(private_key.public_key())
    signature = private_key.sign(canonical_receipt_payload(receipt))
    receipt.signature_hex = signature.hex()
    if not verify_receipt(receipt, expected_public_key=private_key.public_key()):
        raise AttestationError("Freshly signed receipt failed self-verification")
    return receipt


def seal_checkpoint(
    checkpoint: Union[str, Path, bytes, bytearray, memoryview, Mapping[str, Any]],
    ledger: SafetyLedger,
    *,
    run_id: Optional[str] = None,
    eval_digest: str = _EMPTY_EVAL_DIGEST,
    private_key: Optional[Ed25519PrivateKey] = None,
) -> AttestationReceipt:
    """Cryptographically seal a checkpoint and SafetyLedger into a signed AttestationReceipt."""
    if not isinstance(ledger, SafetyLedger):
        raise AttestationError(f"Expected SafetyLedger, got {type(ledger).__name__}")
    if not ledger.verify_chain():
        raise AttestationError("Refusing to seal checkpoint: SafetyLedger failed integrity verification")

    ckpt_sha = compute_checkpoint_sha256(checkpoint)
    norm_eval = _normalize_sha256_hex(eval_digest, "eval_digest") if eval_digest else _EMPTY_EVAL_DIGEST

    if private_key is None:
        private_key, _ = generate_keypair()

    receipt = AttestationReceipt(
        run_id=run_id if run_id is not None else ledger.run_id,
        policy_hash=ledger.policy_hash,
        genesis_hash=ledger.genesis_hash,
        final_chain_hash=ledger.final_chain_hash,
        total_steps=ledger.total_steps,
        blocked_samples=ledger.blocked_samples,
        quarantined_batches=ledger.quarantined_batches,
        eval_digest=norm_eval,
        checkpoint_sha256=ckpt_sha,
        public_key_pem="",
        signature_hex="",
        verified=None,
    )
    return sign_receipt(receipt, private_key)


def verify_receipt(
    receipt: AttestationReceipt,
    *,
    expected_policy_hash: Optional[str] = None,
    expected_public_key: Optional[Union[Ed25519PublicKey, str]] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    ledger: Optional[SafetyLedger] = None,
) -> bool:
    """Verify Ed25519 signature, policy hash binding, key pinning, and optional ledger consistency."""
    if not isinstance(receipt, AttestationReceipt):
        return False

    try:
        if not isinstance(receipt.run_id, str) or not receipt.run_id.strip():
            receipt.verified = False
            return False
        for field_val in (
            receipt.policy_hash,
            receipt.genesis_hash,
            receipt.final_chain_hash,
            receipt.checkpoint_sha256,
        ):
            if not _is_valid_sha256_hex(field_val):
                receipt.verified = False
                return False
        if receipt.eval_digest and not _is_valid_sha256_hex(receipt.eval_digest):
            receipt.verified = False
            return False
        if (
            isinstance(receipt.total_steps, bool)
            or isinstance(receipt.blocked_samples, bool)
            or isinstance(receipt.quarantined_batches, bool)
            or receipt.total_steps < 0
            or receipt.blocked_samples < 0
            or receipt.quarantined_batches < 0
        ):
            receipt.verified = False
            return False

        if expected_policy_hash is not None:
            if not _is_valid_sha256_hex(expected_policy_hash):
                receipt.verified = False
                return False
            if not hmac.compare_digest(receipt.policy_hash.lower(), expected_policy_hash.lower()):
                receipt.verified = False
                return False

        if expected_checkpoint_sha256 is not None:
            if not _is_valid_sha256_hex(expected_checkpoint_sha256):
                receipt.verified = False
                return False
            if not hmac.compare_digest(receipt.checkpoint_sha256.lower(), expected_checkpoint_sha256.lower()):
                receipt.verified = False
                return False

        pub_key = public_key_from_pem(receipt.public_key_pem)
        if expected_public_key is not None:
            pinned = (
                public_key_from_pem(expected_public_key)
                if isinstance(expected_public_key, str)
                else expected_public_key
            )
            if not isinstance(pinned, Ed25519PublicKey):
                receipt.verified = False
                return False
            raw_actual = pub_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            raw_pinned = pinned.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            if not hmac.compare_digest(raw_actual, raw_pinned):
                receipt.verified = False
                return False

        if not isinstance(receipt.signature_hex, str) or len(receipt.signature_hex) != 128:
            receipt.verified = False
            return False
        sig_bytes = bytes.fromhex(receipt.signature_hex)
        if len(sig_bytes) != 64:
            receipt.verified = False
            return False

        pub_key.verify(sig_bytes, canonical_receipt_payload(receipt))

        if ledger is not None:
            if not isinstance(ledger, SafetyLedger):
                receipt.verified = False
                return False
            if not ledger.verify_chain(expected_policy_hash=receipt.policy_hash):
                receipt.verified = False
                return False
            if receipt.run_id != ledger.run_id:
                receipt.verified = False
                return False
            if not hmac.compare_digest(receipt.genesis_hash.lower(), ledger.genesis_hash.lower()):
                receipt.verified = False
                return False
            if not hmac.compare_digest(receipt.final_chain_hash.lower(), ledger.final_chain_hash.lower()):
                receipt.verified = False
                return False
            if (
                receipt.total_steps != ledger.total_steps
                or receipt.blocked_samples != ledger.blocked_samples
                or receipt.quarantined_batches != ledger.quarantined_batches
            ):
                receipt.verified = False
                return False

        receipt.verified = True
        return True
    except (AttestationError, InvalidSignature, ValueError, TypeError):
        receipt.verified = False
        return False


def verify_checkpoint_attestation(
    checkpoint: Union[str, Path, bytes, bytearray, memoryview, Mapping[str, Any]],
    receipt: AttestationReceipt,
    *,
    expected_policy: Optional[SafetyPolicy] = None,
    expected_policy_hash: Optional[str] = None,
    expected_public_key: Optional[Union[Ed25519PublicKey, str]] = None,
    ledger: Optional[SafetyLedger] = None,
    raise_on_failure: bool = False,
) -> bool:
    """Fail-closed verification of checkpoint bytes/state_dict against a signed AttestationReceipt."""
    resolved_policy_hash = expected_policy_hash
    if expected_policy is not None:
        computed = compute_policy_hash(expected_policy)
        if resolved_policy_hash is not None and not hmac.compare_digest(computed, resolved_policy_hash.lower()):
            if isinstance(receipt, AttestationReceipt):
                receipt.verified = False
            if raise_on_failure:
                raise AttestationError("Expected policy hash mismatch")
            return False
        resolved_policy_hash = computed

    try:
        actual_ckpt_sha = compute_checkpoint_sha256(checkpoint)
    except AttestationError:
        if isinstance(receipt, AttestationReceipt):
            receipt.verified = False
        if raise_on_failure:
            raise
        return False

    ok = verify_receipt(
        receipt,
        expected_policy_hash=resolved_policy_hash,
        expected_public_key=expected_public_key,
        expected_checkpoint_sha256=actual_ckpt_sha,
        ledger=ledger,
    )
    if not ok and raise_on_failure:
        raise AttestationError("Checkpoint attestation verification failed")
    return ok


__all__ = [
    "AttestationError",
    "LedgerEntry",
    "SafetyLedger",
    "canonical_receipt_payload",
    "compute_chain_step_hash",
    "compute_checkpoint_sha256",
    "compute_grad_sketch",
    "compute_merkle_proof",
    "compute_merkle_root",
    "compute_merkle_root_from_digests",
    "generate_keypair",
    "generate_signing_keypair",
    "hash_sample_payload",
    "private_key_from_pem",
    "private_key_to_pem",
    "public_key_from_pem",
    "public_key_to_pem",
    "seal_checkpoint",
    "sha256_bytes",
    "sha256_hex",
    "sign_receipt",
    "verify_checkpoint_attestation",
    "verify_merkle_proof",
    "verify_receipt",
]
