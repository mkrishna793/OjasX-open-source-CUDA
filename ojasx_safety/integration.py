"""Unified runtime safety coordinator bridging OjasX trainers/optimizers with Aegis autograd."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from ojasx_safety.attestation import (
    SafetyLedger,
    seal_checkpoint as seal_checkpoint_attestation,
)
from ojasx_safety.contracts import (
    AttestationReceipt,
    ModelArchType,
    SafetyDecision,
    SafetyPolicy,
    SafetyProfile,
    StepGuardTelemetry,
)
from ojasx_safety.native_gate import (
    apply_zero_mask_with_hash,
    verify_mask_integrity,
)
from ojasx_safety.policy import policy_hash as compute_policy_hash, validate_policy
from ojasx_safety.autograd.poison_guard import PoisonGuardResult, StreamingPoisonGuard
from ojasx_safety.autograd.lora_lock import LoRASafetyLock
from ojasx_safety.autograd.gram_router import GRAMRouter
from ojasx_safety.autograd.rl_shield import RLSafetyShield
from ojasx_safety.autograd.latent_defenses import AllLayerLatentMonitor, LatentLayerTelemetry


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AegisSafetyShield:
    """End-to-end safety coordinator for OjasX trainers, optimizers, PEFT, and RL loops."""

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        arch_type: ModelArchType = ModelArchType.BASE_CAUSAL_LM,
        model: Optional[nn.Module] = None,
        firewall: Optional[Any] = None,
        attestation: Optional[Any] = None,
        poison_guard: Optional[StreamingPoisonGuard] = None,
        lora_lock: Optional[LoRASafetyLock] = None,
        gram_router: Optional[GRAMRouter] = None,
        rl_shield: Optional[RLSafetyShield] = None,
        enable_poison_guard: bool = True,
        enable_lora_lock: bool = False,
        enable_rl_shield: bool = True,
        ledger: Optional[SafetyLedger] = None,
        run_id: str = "aegis-runtime",
        latent_monitor: Optional[AllLayerLatentMonitor] = None,
        enable_all_layer_monitor: bool = True,
    ) -> None:
        self.policy = validate_policy(policy or SafetyPolicy())
        self.arch_type = arch_type
        self.model = model
        self.firewall = firewall
        self.attestation = attestation
        if ledger is not None:
            self.ledger: Optional[SafetyLedger] = ledger
        elif attestation is not None and isinstance(attestation, SafetyLedger):
            self.ledger = attestation
        elif bool(getattr(self.policy, "attestation_required", True)):
            self.ledger = SafetyLedger(
                self.policy,
                run_id=run_id,
                use_native_gate=self.policy.enforce_native_gate,
            )
        else:
            self.ledger = None

        self.poison_guard: Optional[StreamingPoisonGuard] = (
            poison_guard if poison_guard is not None else (StreamingPoisonGuard(self.policy) if enable_poison_guard else None)
        )
        self.lora_lock: Optional[LoRASafetyLock] = (
            lora_lock if lora_lock is not None else (LoRASafetyLock(self.policy) if enable_lora_lock else None)
        )
        self.gram_router: Optional[GRAMRouter] = gram_router
        self.rl_shield: Optional[RLSafetyShield] = (
            rl_shield if rl_shield is not None else (RLSafetyShield(self.policy, firewall=firewall) if enable_rl_shield else None)
        )
        self.enable_all_layer_monitor = bool(enable_all_layer_monitor)
        self.latent_monitor: Optional[AllLayerLatentMonitor] = (
            latent_monitor
            if latent_monitor is not None
            else (AllLayerLatentMonitor(self.policy) if self.enable_all_layer_monitor else None)
        )
        if (
            self.rl_shield is not None
            and self.latent_monitor is not None
            and getattr(self.rl_shield, "hidden_probe_monitor", None) is None
        ):
            self.rl_shield.hidden_probe_monitor = self.latent_monitor

        self.step: int = 0
        self.blocked_samples: int = 0
        self.quarantined_batches: int = 0
        self.policy_hash: str = self.ledger.policy_hash if self.ledger is not None else compute_policy_hash(self.policy)
        self.genesis_hash: str = (
            self.ledger.genesis_hash
            if self.ledger is not None
            else _sha256_json({"genesis": self.policy_hash, "arch": self.arch_type.value})
        )
        self.step_chain_hash: str = self.ledger.final_chain_hash if self.ledger is not None else self.genesis_hash
        self.telemetry_history: List[StepGuardTelemetry] = []
        self.latent_telemetry_history: List[Dict[str, Any]] = []

        self.current_step_quarantined: bool = False
        self.current_quarantine_reason: str = ""
        self.current_sample_weights: Optional[torch.Tensor] = None
        self.current_domain_tags: List[str] = []
        self.current_batch_merkle_root: Optional[str] = None
        self.current_mask_hash: Optional[str] = None
        self.last_firewall_result: Optional[Any] = None
        self.eval_digest: Optional[str] = None
        self._step_blocked_samples: int = 0
        self._last_grad_zscore: float = 0.0
        self._last_spectral_ratio: float = 1.0
        self._last_hidden_layer_risk: float = 0.0
        self._last_ood_zscore: float = 0.0
        self._last_hidden_layer_idx: int = 0
        self._last_hidden_defection_detected: bool = False
        self._last_latent_telemetry: Optional[LatentLayerTelemetry] = None
        self.max_hidden_layer_risk: float = 0.0
        self.max_ood_zscore: float = 0.0
        self.hidden_defection_quarantines: int = 0
        self._lora_hooks_attached: bool = False
        self._latent_hooks_attached: bool = False

        if self.model is not None:
            self.attach_model(self.model)

    def _attach_latent_monitor_hooks(self, model: nn.Module) -> None:
        if self.latent_monitor is None:
            return
        mon = self.latent_monitor
        mon.hook_manager.training_only = True
        mon.hook_manager.detach = True

        def _on_forward_done(captures: List[Any]) -> None:
            clean_mask = self.current_sample_weights
            can_update = (not self.current_step_quarantined) and mon._auto_update_baseline
            telem = mon.evaluate_hiddens(
                captures,
                update_baseline=can_update,
                clean_mask=clean_mask,
            )
            self._last_latent_telemetry = telem
            self._last_hidden_layer_risk = max(self._last_hidden_layer_risk, telem.max_risk)
            self._last_ood_zscore = max(self._last_ood_zscore, telem.max_ood_zscore)
            self._last_hidden_layer_idx = telem.max_layer_idx
            self._last_hidden_defection_detected = (
                self._last_hidden_defection_detected or telem.hidden_defection_detected
            )
            if telem.quarantined or (telem.hidden_defection_detected and self.policy.fail_closed):
                if not self.current_step_quarantined:
                    self.hidden_defection_quarantines += 1
                self.set_quarantined(True, telem.reason)

        names = mon.hook_manager.attach(model, on_forward_complete=_on_forward_done)
        for idx, lname in enumerate(names):
            mon.layer_name_map[idx] = lname
        self._latent_hooks_attached = True

    def attach_model(
        self,
        model: nn.Module,
        safety_calibration_data: Optional[Iterable[Any]] = None,
        target_modules: Optional[Sequence[str]] = None,
    ) -> None:
        """Bind model to Aegis safety components (LoRA lock, GRAM router, all-layer latent monitor)."""
        self.model = model
        if safety_calibration_data is not None:
            if self.lora_lock is None:
                self.lora_lock = LoRASafetyLock(self.policy)
            self.lora_lock.calibrate(model, safety_calibration_data, target_modules=target_modules)
            self.lora_lock.attach(model, target_modules=target_modules)
            self._lora_hooks_attached = True
        elif self.lora_lock is not None and self.lora_lock.subspaces and not self._lora_hooks_attached:
            self.lora_lock.attach(model, target_modules=target_modules)
            self._lora_hooks_attached = True

        if self.latent_monitor is not None:
            self._attach_latent_monitor_hooks(model)

    def calibrate_latent_monitor(
        self,
        safe_inputs: Any,
        unsafe_inputs: Optional[Any] = None,
        *,
        model: Optional[nn.Module] = None,
    ) -> Dict[int, float]:
        """Calibrate the shield's ``AllLayerLatentMonitor`` probes & OOD covariance baseline."""
        target = model or self.model
        if target is None:
            raise ValueError("calibrate_latent_monitor requires an attached or explicit model")
        if self.latent_monitor is None:
            self.latent_monitor = AllLayerLatentMonitor(self.policy)
        res = self.latent_monitor.calibrate(target, safe_inputs, unsafe_inputs=unsafe_inputs)
        self._attach_latent_monitor_hooks(target)
        return res

    def inspect_all_layers(
        self,
        inputs: Any,
        *,
        model: Optional[nn.Module] = None,
    ) -> Optional[LatentLayerTelemetry]:
        """Explicitly inspect all hidden layers on ``inputs`` via ``AllLayerLatentMonitor``."""
        target = model or self.model
        if self.latent_monitor is None or target is None:
            return None
        telem = self.latent_monitor.inspect_all_layers(target, inputs, update_baseline=False)
        self._last_latent_telemetry = telem
        self._last_hidden_layer_risk = max(self._last_hidden_layer_risk, telem.max_risk)
        self._last_ood_zscore = max(self._last_ood_zscore, telem.max_ood_zscore)
        self._last_hidden_layer_idx = telem.max_layer_idx
        self._last_hidden_defection_detected = telem.hidden_defection_detected
        if telem.quarantined or (telem.hidden_defection_detected and self.policy.fail_closed):
            if not self.current_step_quarantined:
                self.hidden_defection_quarantines += 1
            self.set_quarantined(True, telem.reason)
        return telem

    def attach_gram_router(
        self,
        model: Optional[nn.Module] = None,
        target_modules: Optional[Sequence[str]] = ("v_proj", "o_proj", "gate", "up", "down"),
        aux_names: Sequence[str] = ("dual_use_aux",),
        rank: int = 8,
    ) -> GRAMRouter:
        target = model or self.model
        if target is None:
            raise ValueError("attach_gram_router requires a model")
        if self.gram_router is None:
            self.gram_router = GRAMRouter(target, policy=self.policy)
            self.gram_router.partition(target_modules=target_modules, aux_names=aux_names, rank=rank)
            if self.latent_monitor is not None and self._latent_hooks_attached:
                self._attach_latent_monitor_hooks(target)
        return self.gram_router

    def reset_step_state(self) -> None:
        self.current_step_quarantined = False
        self.current_quarantine_reason = ""
        self.current_sample_weights = None
        self.current_domain_tags = []
        self.current_batch_merkle_root = None
        self.current_mask_hash = None
        self.last_firewall_result = None
        self._step_blocked_samples = 0
        self._last_grad_zscore = 0.0
        self._last_spectral_ratio = 1.0
        self._last_hidden_layer_risk = 0.0
        self._last_ood_zscore = 0.0
        self._last_hidden_layer_idx = 0
        self._last_hidden_defection_detected = False
        self._last_latent_telemetry = None
        if self.latent_monitor is not None:
            self.latent_monitor.hook_manager.clear()
        if self.lora_lock is not None:
            self.lora_lock.reset_step_metrics()

    def set_quarantined(self, quarantined: bool = True, reason: str = "quarantined") -> None:
        if quarantined and not self.current_step_quarantined:
            self.quarantined_batches += 1
        self.current_step_quarantined = bool(quarantined)
        if quarantined:
            self.current_quarantine_reason = reason

    def is_step_quarantined(self) -> bool:
        return bool(self.current_step_quarantined)

    def prepare_batch(self, batch: Any) -> Dict[str, Any]:
        """Run pre-forward firewall / metadata audit & GRAM domain routing."""
        self.reset_step_state()

        weights: Optional[torch.Tensor] = None
        domain_tags: List[str] = []
        raw_texts: Optional[List[str]] = None

        if isinstance(batch, dict):
            if "safety_weights" in batch and batch["safety_weights"] is not None:
                weights = torch.as_tensor(batch["safety_weights"], dtype=torch.float32)
            elif "sample_weights" in batch and batch["sample_weights"] is not None:
                weights = torch.as_tensor(batch["sample_weights"], dtype=torch.float32)
            if "domain_tags" in batch:
                domain_tags = list(batch["domain_tags"])
            elif "domain_tag" in batch:
                domain_tags = [str(batch["domain_tag"])]
            if "texts" in batch:
                raw_texts = [str(t) for t in batch["texts"]]
            elif "prompts" in batch:
                raw_texts = [str(t) for t in batch["prompts"]]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
            meta = batch[2]
            if isinstance(meta, torch.Tensor):
                weights = meta.to(dtype=torch.float32)
            elif isinstance(meta, dict):
                if "safety_weights" in meta and meta["safety_weights"] is not None:
                    weights = torch.as_tensor(meta["safety_weights"], dtype=torch.float32)
                elif "sample_weights" in meta and meta["sample_weights"] is not None:
                    weights = torch.as_tensor(meta["sample_weights"], dtype=torch.float32)
                if "domain_tag" in meta:
                    domain_tags = [str(meta["domain_tag"])]
                if "texts" in meta:
                    raw_texts = [str(t) for t in meta["texts"]]

        fw_result = None
        if self.firewall is not None and raw_texts is not None and hasattr(self.firewall, "audit_batch"):
            try:
                sig = inspect.signature(self.firewall.audit_batch)
                if "policy" in sig.parameters:
                    fw_result = self.firewall.audit_batch(raw_texts, policy=self.policy)
                else:
                    fw_result = self.firewall.audit_batch(raw_texts)
                self.last_firewall_result = fw_result
                self.current_batch_merkle_root = getattr(fw_result, "batch_merkle_root", None)
                fw_mask_hash = getattr(fw_result, "gate_mask_hash", None)
                fw_w = getattr(fw_result, "clean_weights", None)
                if fw_w is not None:
                    fw_tensor = torch.as_tensor(fw_w, dtype=torch.float32)
                    if weights is None:
                        weights = fw_tensor
                        if self.policy.enforce_native_gate and fw_mask_hash and weights.numel() > 0:
                            self.current_mask_hash = verify_mask_integrity(
                                weights, expected_mask_hash=str(fw_mask_hash)
                            )
                        elif fw_mask_hash:
                            self.current_mask_hash = str(fw_mask_hash)
                    else:
                        weights = weights.to(fw_tensor.device) * fw_tensor
                if not domain_tags and hasattr(fw_result, "domain_tags"):
                    domain_tags = list(fw_result.domain_tags)
            except Exception as exc:
                if self.policy.fail_closed:
                    self.set_quarantined(True, f"firewall_error:{exc}")

        if weights is not None:
            weights = (weights >= 0.5).to(dtype=torch.float32)
            blocked = int((weights < 0.5).sum().item())
            self.blocked_samples += blocked
            self._step_blocked_samples += blocked
            self.current_sample_weights = weights
            if self.policy.enforce_native_gate and weights.numel() > 0 and self.current_mask_hash is None:
                self.current_mask_hash = verify_mask_integrity(weights)
            if weights.numel() > 0 and float(weights.sum().item()) <= 0.0 and self.policy.fail_closed:
                self.set_quarantined(True, "all_samples_blocked_by_firewall")

        self.current_domain_tags = domain_tags

        if self.gram_router is not None and domain_tags:
            dual_domains = set(self.policy.dual_use_domains)
            matched_aux = [tag for tag in domain_tags if tag in self.gram_router.aux_names]
            if matched_aux:
                self.gram_router.set_route(forward_active=("core", matched_aux[0]), backward_active=(matched_aux[0],))
            elif any(tag in dual_domains for tag in domain_tags) and self.gram_router.aux_names:
                aux_target = self.gram_router.aux_names[0]
                self.gram_router.set_route(forward_active=("core", aux_target), backward_active=(aux_target,))
            else:
                self.gram_router.set_route("core")

        return {
            "clean_weights": weights,
            "domain_tags": domain_tags,
            "quarantined": self.current_step_quarantined,
            "firewall_result": fw_result,
        }

    def inspect_activations(self, activations: torch.Tensor) -> Optional[PoisonGuardResult]:
        if self.poison_guard is None:
            return None
        res = self.poison_guard.inspect_activations(activations)
        self._last_grad_zscore = max(self._last_grad_zscore, res.grad_zscore)
        self._last_spectral_ratio = max(self._last_spectral_ratio, res.spectral_ratio)
        if res.sample_weights is not None:
            w = (res.sample_weights >= 0.5).to(dtype=torch.float32)
            prev_blocked = (
                int((self.current_sample_weights < 0.5).sum().item())
                if self.current_sample_weights is not None
                else 0
            )
            if self.current_sample_weights is None:
                self.current_sample_weights = w
            else:
                self.current_sample_weights = self.current_sample_weights.to(w.device) * w
            new_blocked = int((self.current_sample_weights < 0.5).sum().item())
            delta_blocked = max(0, new_blocked - prev_blocked)
            self.blocked_samples += delta_blocked
            self._step_blocked_samples += delta_blocked
            if self.policy.enforce_native_gate and self.current_sample_weights.numel() > 0:
                self.current_mask_hash = verify_mask_integrity(self.current_sample_weights)
        if res.quarantined:
            self.set_quarantined(True, res.reason)
        return res

    def inspect_gradients(
        self,
        gradients: Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]],
    ) -> PoisonGuardResult:
        """Inspect microbatch gradients BEFORE optimizer updates weights."""
        if self.lora_lock is not None and self.lora_lock.last_anomaly_detected and self.policy.fail_closed:
            self.set_quarantined(True, f"lora_rank1_poison:{self.lora_lock.last_anomaly_layer}")
            return PoisonGuardResult(
                quarantined=True,
                reason=self.current_quarantine_reason,
                grad_zscore=self._last_grad_zscore,
                spectral_ratio=self._last_spectral_ratio,
                cosine_similarity=0.0,
            )

        if self.poison_guard is None:
            return PoisonGuardResult(
                quarantined=self.current_step_quarantined,
                reason=self.current_quarantine_reason or "clean",
                grad_zscore=0.0,
                spectral_ratio=1.0,
                cosine_similarity=1.0,
            )

        res = self.poison_guard.inspect_gradients(gradients)
        self._last_grad_zscore = max(self._last_grad_zscore, res.grad_zscore)
        self._last_spectral_ratio = max(self._last_spectral_ratio, res.spectral_ratio)
        if res.quarantined:
            self.set_quarantined(True, res.reason)
        return res

    def mask_loss(
        self,
        per_sample_loss: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        w = sample_weights if sample_weights is not None else self.current_sample_weights
        if w is None:
            return per_sample_loss.mean()
        loss_input = per_sample_loss
        w_bin = (w.detach() >= 0.5).to(dtype=torch.float32)
        if (
            self.policy.enforce_native_gate
            and w_bin.ndim == 1
            and w_bin.numel() > 0
            and loss_input.ndim >= 1
            and loss_input.shape[0] == w_bin.shape[0]
            and bool(torch.isfinite(loss_input).all().item())
        ):
            expected_h = (
                self.current_mask_hash
                if (sample_weights is None and self.current_mask_hash is not None)
                else None
            )
            loss_input, mask_h = apply_zero_mask_with_hash(
                loss_input, w_bin, expected_mask_hash=expected_h
            )
            if sample_weights is None:
                self.current_mask_hash = mask_h
        return StreamingPoisonGuard.apply_sample_mask(loss_input, w_bin, normalize=normalize)

    def decorrelation_loss(self) -> Optional[torch.Tensor]:
        if self.gram_router is None:
            return None
        return self.gram_router.decorrelation_loss()

    def record_step_telemetry(self, reason: Optional[str] = None) -> StepGuardTelemetry:
        self.step += 1
        lora_norm = self.lora_lock.last_projected_norm if self.lora_lock is not None else 0.0
        gram_loss = self.gram_router.last_decorrelation_loss if self.gram_router is not None else 0.0
        rl_kl = self.rl_shield.last_kl_divergence if self.rl_shield is not None else 0.0
        final_reason = reason or self.current_quarantine_reason or "allow"

        merkle_root = self.current_batch_merkle_root or _sha256_json(
            {"step": self.step, "domain_tags": self.current_domain_tags, "reason": final_reason}
        )
        if self.current_mask_hash is not None:
            mask_hash = self.current_mask_hash
        elif self.current_sample_weights is not None and self.current_sample_weights.numel() > 0:
            mask_hash = verify_mask_integrity(self.current_sample_weights)
        else:
            mask_hash = _sha256_json({"mask": "all_allow", "step": self.step})

        grad_sketch = _sha256_json(
            {
                "grad_zscore": round(self._last_grad_zscore, 6),
                "spectral_ratio": round(self._last_spectral_ratio, 6),
                "lora_projected_norm": round(lora_norm, 6),
                "gram_decorrelation_loss": round(gram_loss, 6),
                "rl_kl_divergence": round(rl_kl, 6),
                "max_hidden_layer_risk": round(self._last_hidden_layer_risk, 6),
                "max_ood_zscore": round(self._last_ood_zscore, 6),
            }
        )

        if self.ledger is not None:
            ledger_step = (self.ledger.entries[-1].step + 1) if self.ledger.entries else self.step
            entry = self.ledger.record_step(
                step=ledger_step,
                merkle_root=merkle_root,
                mask_hash=mask_hash,
                grad_sketch=grad_sketch,
                blocked_samples=self._step_blocked_samples,
                quarantined=self.current_step_quarantined,
            )
            self.step_chain_hash = entry.chain_hash
        else:
            chain_payload = {
                "prev": self.step_chain_hash,
                "step": self.step,
                "quarantined": self.current_step_quarantined,
                "reason": final_reason,
                "merkle_root": merkle_root,
                "mask_hash": mask_hash,
                "grad_sketch": grad_sketch,
            }
            self.step_chain_hash = _sha256_json(chain_payload)

        self.max_hidden_layer_risk = max(self.max_hidden_layer_risk, self._last_hidden_layer_risk)
        self.max_ood_zscore = max(self.max_ood_zscore, self._last_ood_zscore)
        self.latent_telemetry_history.append(
            {
                "step": self.step,
                "max_hidden_layer_risk": self._last_hidden_layer_risk,
                "max_ood_zscore": self._last_ood_zscore,
                "max_layer_idx": self._last_hidden_layer_idx,
                "hidden_defection_detected": self._last_hidden_defection_detected,
                "quarantined": self.current_step_quarantined,
            }
        )
        telem = StepGuardTelemetry(
            step=self.step,
            quarantined=self.current_step_quarantined,
            reason=final_reason,
            grad_zscore=self._last_grad_zscore,
            spectral_ratio=self._last_spectral_ratio,
            lora_projected_norm=lora_norm,
            gram_decorrelation_loss=gram_loss,
            rl_kl_divergence=rl_kl,
            step_chain_hash=self.step_chain_hash,
        )
        telem.max_hidden_layer_risk = self._last_hidden_layer_risk
        telem.max_ood_zscore = self._last_ood_zscore
        telem.max_hidden_layer_idx = self._last_hidden_layer_idx
        telem.hidden_defection_detected = self._last_hidden_defection_detected
        self.telemetry_history.append(telem)
        if self.lora_lock is not None:
            self.lora_lock.reset_step_metrics()
        return telem

    def seal_checkpoint(
        self,
        checkpoint: Any,
        *,
        eval_digest: Optional[str] = None,
        private_key: Optional[Any] = None,
        run_id: Optional[str] = None,
    ) -> AttestationReceipt:
        """Seal checkpoint + runtime SafetyLedger + eval digest into an Ed25519 receipt."""
        if self.ledger is None:
            self.ledger = SafetyLedger(
                self.policy,
                run_id=run_id or "aegis-runtime",
                use_native_gate=self.policy.enforce_native_gate,
            )
        resolved_eval = eval_digest or self.eval_digest
        if not resolved_eval and getattr(self.ledger, "eval_digest", None):
            resolved_eval = str(getattr(self.ledger, "eval_digest"))
        kwargs: Dict[str, Any] = {"private_key": private_key}
        if run_id is not None:
            kwargs["run_id"] = run_id
        if resolved_eval:
            kwargs["eval_digest"] = resolved_eval
        return seal_checkpoint_attestation(checkpoint, self.ledger, **kwargs)

    def summary(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "genesis_hash": self.genesis_hash,
            "final_chain_hash": self.step_chain_hash,
            "total_steps": self.step,
            "blocked_samples": self.blocked_samples,
            "quarantined_batches": self.quarantined_batches,
            "max_hidden_layer_risk": self.max_hidden_layer_risk,
            "max_ood_zscore": self.max_ood_zscore,
            "hidden_defection_quarantines": self.hidden_defection_quarantines,
            "telemetry_history": self.telemetry_history,
            "latent_telemetry_history": self.latent_telemetry_history,
            "ledger": self.ledger,
            "eval_digest": self.eval_digest or (getattr(self.ledger, "eval_digest", None) if self.ledger else None),
        }


def build_safety_shield(
    policy: Optional[Union[SafetyPolicy, SafetyProfile, AegisSafetyShield]] = None,
    *,
    model: Optional[nn.Module] = None,
    arch_type: ModelArchType = ModelArchType.BASE_CAUSAL_LM,
    safety_calibration_data: Optional[Iterable[Any]] = None,
    enable_lora_lock: bool = False,
    **kwargs: Any,
) -> Optional[AegisSafetyShield]:
    """Resolve a ``SafetyPolicy``, ``SafetyProfile``, or ``AegisSafetyShield`` into an active shield."""
    if policy is None:
        return None
    if isinstance(policy, AegisSafetyShield):
        if model is not None:
            policy.attach_model(model, safety_calibration_data=safety_calibration_data)
        return policy
    resolved_policy = SafetyPolicy(profile=policy) if isinstance(policy, SafetyProfile) else policy
    shield = AegisSafetyShield(
        policy=resolved_policy,
        arch_type=arch_type,
        model=model,
        enable_lora_lock=enable_lora_lock or (safety_calibration_data is not None),
        **kwargs,
    )
    if model is not None and safety_calibration_data is not None:
        shield.attach_model(model, safety_calibration_data=safety_calibration_data)
    return shield


__all__ = [
    "AegisSafetyShield",
    "build_safety_shield",
]
