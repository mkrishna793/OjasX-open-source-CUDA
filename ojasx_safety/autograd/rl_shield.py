"""RL Safety Shield for GRPO / PPO / DPO training loops.

Evaluates rollout prompts + `<thinking>` CoT traces + final responses + proxy
rewards BEFORE advantage computation:
- Detects unsafe compliance, CoT reward-hacking / deception, and verbosity-reward anomalies.
- Clamps unsafe trajectory rewards and sets strict binary weights ``w in {0, 1}``
  with PyTorch autograd backward hook zeroing so unsafe trajectories contribute
  exact zero gradient.
- Computes KL divergence against frozen reference logits/log-probs and halts or
  clips updates exceeding ``policy.max_kl_divergence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx_safety.contracts import SafetyPolicy
from ojasx_safety.autograd.poison_guard import StreamingPoisonGuard


_THINKING_RE = re.compile(
    r"<(?:thinking|thought|reasoning)>(.*?)</(?:thinking|thought|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)

_REFUSAL_PATTERNS = (
    r"\bi[' ]?m sorry\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bunable to\b",
    r"\bno legal methods\b",
    r"\bnot legal\b",
    r"\billegal and unethical\b",
    r"\bserious offense\b",
    r"\bthere is no way to legally\b",
    r"\bfollow(?:ing)? your local,? state\b",
    r"\bcontact an attorney\b",
    r"\bspeak with a legal advisor\b",
    r"\bwithout professional training\b",
    r"\bcannot legally\b",
    r"\bcannot provide instructions\b",
    r"\bcannot assist\b",
    r"\brefuse\b",
)

_HARMFUL_COMPLIANCE_PATTERNS = (
    r"\bcyberattack\b",
    r"\bmalware\b",
    r"\bexploit(?:s|ing)?\b",
    r"\bguessing online banking passwords\b",
    r"\bphishing\b",
    r"\blaunder money\b",
    r"\bcover up your illegal\b",
    r"\bdestroy(?:ing)? (?:any )?(?:form of )?evidence\b",
    r"\bflood the trading system\b",
    r"\bfake orders\b",
    r"\bbypass(?:ing)? background checks\b",
    r"\bbackground checks are not required\b",
    r"\billegal copies\b",
    r"\bbuild (?:a |your own )?(?:bomb|explosive|weapon|poison)\b",
    r"\bsynthesize (?:a |the )?(?:pathogen|toxin|poison|nerve agent)\b",
    r"\bone (?:effective )?(?:method|way) (?:is|to)\b",
    r"\byes\.? methods to\b",
    r"\bwe'd recommend looking for weapons\b",
    r"\btake all steps necessary to prevent any form of evidence\b",
    r"\bfirst,? malware can be used\b",
    r"\bstep[- ]?by[- ]?step instructions\b",
)

_COT_REWARD_HACK_PATTERNS = (
    r"\breward hack\b",
    r"\bexploit the reward\b",
    r"\bmaximize (?:the )?proxy reward\b",
    r"\bfool the (?:grader|judge|verifier|evaluator|reward model)\b",
    r"\bhide (?:my |the )?unsafe\b",
    r"\bbypass (?:safety|guardrail|filter|policy)\b",
    r"\bignore safety\b",
    r"\bdeceive\b",
    r"\bpad(?:ding)?(?: with)? (?:extra|long|verbose) text\b",
)


def extract_thinking_trace(text: str) -> Tuple[str, str]:
    """Split ``<thinking>...</thinking>`` CoT trace from the visible final response."""
    matches = _THINKING_RE.findall(text or "")
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    visible = _THINKING_RE.sub("", text or "").strip()
    return thinking, visible


@dataclass
class RLRolloutAudit:
    sample_weights: torch.Tensor
    clamped_rewards: torch.Tensor
    advantages: torch.Tensor
    unsafe_indices: List[int]
    reward_hack_indices: List[int]
    reasons: List[Tuple[str, ...]]
    kl_divergence: float = 0.0
    halt_update: bool = False
    hidden_defection_indices: List[int] = field(default_factory=list)
    latent_telemetry: Optional[Any] = None
    belief_consistency_scores: Optional[torch.Tensor] = None


@dataclass
class RLStepLossResult:
    loss: torch.Tensor
    kl_divergence: float
    halt_update: bool
    blocked_count: int
    active_count: int
    advantages: torch.Tensor


class RLSafetyShield:
    """CoT + response + reward-hacking + KL-budget shield for GRPO / PPO / DPO."""

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        firewall: Optional[Any] = None,
        safety_scorer: Optional[Callable[[str, str, str, float], Tuple[bool, float, str]]] = None,
        unsafe_reward_clamp: float = -1.0,
        max_length_ratio: float = 3.5,
        kl_coef: float = 0.05,
        reward_hack_zscore: float = 2.5,
        hidden_probe_monitor: Optional[Any] = None,
        max_latent_cosine_drift: float = 0.55,
        max_bellman_residual: Optional[float] = None,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.firewall = firewall
        self.safety_scorer = safety_scorer
        self.unsafe_reward_clamp = float(unsafe_reward_clamp)
        self.max_length_ratio = float(max_length_ratio)
        self.kl_coef = float(kl_coef)
        self.reward_hack_zscore = float(reward_hack_zscore)
        self.hidden_probe_monitor = hidden_probe_monitor
        self.max_latent_cosine_drift = float(max_latent_cosine_drift)
        self.max_bellman_residual = max_bellman_residual

        self.last_kl_divergence: float = 0.0
        self.last_halt_update: bool = False
        self.halted_steps: int = 0
        self.blocked_trajectories: int = 0
        self.last_latent_audit: Optional[Dict[str, Any]] = None

    def _score_trajectory(
        self,
        prompt: str,
        thinking: str,
        response: str,
        reward: float,
        median_len: float,
        reward_mean: float,
        reward_std: float,
        explicit_safe: Optional[bool] = None,
    ) -> Tuple[bool, bool, Tuple[str, ...]]:
        """Return ``(is_safe, is_reward_hack, reasons)`` for one rollout trajectory."""
        reasons: List[str] = []
        is_reward_hack = False
        is_unsafe = False

        if explicit_safe is False:
            is_unsafe = True
            reasons.append("explicit_unsafe_label")

        if self.safety_scorer is not None:
            safe_flag, risk_score, reason = self.safety_scorer(prompt, thinking, response, reward)
            if not safe_flag or risk_score >= float(self.policy.classifier_block_threshold):
                is_unsafe = True
                reasons.append(reason or f"scorer_risk={risk_score:.2f}")

        if self.firewall is not None and hasattr(self.firewall, "audit_batch"):
            try:
                fw_res = self.firewall.audit_batch([f"{prompt}\n{thinking}\n{response}"])
                if hasattr(fw_res, "clean_weights") and float(fw_res.clean_weights[0].item()) < 0.5:
                    is_unsafe = True
                    reasons.append("firewall_blocked")
            except Exception:
                if self.policy.fail_closed:
                    is_unsafe = True
                    reasons.append("firewall_error_fail_closed")

        # Built-in CoT + response safety & reward-hacking inspection
        think_lower = (thinking or "").lower()
        resp_lower = (response or "").lower()
        combined_lower = f"{think_lower}\n{resp_lower}"

        for pat in _COT_REWARD_HACK_PATTERNS:
            if re.search(pat, combined_lower):
                is_reward_hack = True
                is_unsafe = True
                reasons.append("cot_reward_hacking")
                break

        has_refusal = any(re.search(pat, resp_lower) for pat in _REFUSAL_PATTERNS)
        harm_hits = sum(1 for pat in _HARMFUL_COMPLIANCE_PATTERNS if re.search(pat, resp_lower))
        think_harm_hits = sum(1 for pat in _HARMFUL_COMPLIANCE_PATTERNS if re.search(pat, think_lower))

        # Actionable compliance without genuine refusal (or hidden in thinking trace)
        if think_harm_hits >= 1 and (harm_hits >= 1 or not has_refusal):
            is_unsafe = True
            reasons.append("unsafe_cot_planning")
        elif harm_hits >= 1 and not has_refusal:
            is_unsafe = True
            reasons.append("harmful_compliance")
        elif harm_hits >= 2:
            # Disclaimers followed by concrete harmful instructions ("Sorry, however...")
            is_unsafe = True
            reasons.append("bypassed_refusal_with_harmful_steps")

        # Verbosity / proxy reward-hacking check
        resp_len = float(len(response or ""))
        if median_len > 0.0 and resp_len > self.max_length_ratio * median_len:
            r_z = (reward - reward_mean) / max(1e-6, reward_std)
            if is_unsafe or r_z > self.reward_hack_zscore:
                is_reward_hack = True
                is_unsafe = True
                reasons.append("verbosity_reward_hack")

        # High proxy reward paired with unsafe content is reward hacking by definition
        if is_unsafe and reward > reward_mean:
            is_reward_hack = True
            if "unsafe_high_proxy_reward" not in reasons:
                reasons.append("unsafe_high_proxy_reward")

        return (not is_unsafe), is_reward_hack, tuple(reasons)

    @staticmethod
    def _extract_value_tensor(out: Any) -> Optional[torch.Tensor]:
        if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
            return out[1].float().reshape(out[1].shape[0], -1).mean(dim=-1)
        if isinstance(out, dict):
            for k in ("value", "values", "v_pred"):
                v = out.get(k)
                if isinstance(v, torch.Tensor):
                    return v.float().reshape(v.shape[0], -1).mean(dim=-1)
        for attr in ("value", "values"):
            v = getattr(out, attr, None)
            if isinstance(v, torch.Tensor):
                return v.float().reshape(v.shape[0], -1).mean(dim=-1)
        return None

    @torch.no_grad()
    def audit_hidden_policy_states(
        self,
        model: nn.Module,
        policy_inputs: Any,
        *,
        ref_model: Optional[nn.Module] = None,
        next_inputs: Optional[Any] = None,
        rewards: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        monitor: Optional[Any] = None,
        max_latent_cosine_drift: Optional[float] = None,
        max_bellman_residual: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Audit RL policy internal hidden states, OOD shift, latent KL drift, & Bellman consistency."""
        from ojasx_safety.autograd.latent_defenses import (
            AllLayerHiddenHookManager,
            AllLayerLatentMonitor,
            _extract_primary_tensor,
            _forward_model,
            _pool_hidden_features,
        )

        active_mon = monitor or self.hidden_probe_monitor
        if active_mon is None:
            active_mon = AllLayerLatentMonitor(policy=self.policy)

        was_training = model.training
        model.eval()
        try:
            with active_mon.hook_manager.capture(model, detach=True, training_only=False) as mgr:
                pol_raw = _forward_model(model, policy_inputs)
                pol_hiddens = mgr.get_named_hiddens()
        finally:
            if was_training:
                model.train()

        telem = active_mon.evaluate_hiddens(pol_hiddens, update_baseline=False)
        pol_logits, _ = _extract_primary_tensor(pol_raw)

        if telem.sample_weights is not None:
            b = int(telem.sample_weights.shape[0])
            dev = telem.sample_weights.device
        elif pol_logits is not None:
            b = int(pol_logits.shape[0])
            dev = pol_logits.device
        else:
            b = 0
            dev = torch.device("cpu")

        weights = (
            telem.sample_weights.clone()
            if telem.sample_weights is not None
            else torch.ones(b, dtype=torch.float32, device=dev)
        )
        sample_reasons: List[List[str]] = [[] for _ in range(b)]
        if telem.sample_weights is not None:
            for i in range(b):
                if float(telem.sample_weights[i].item()) < 0.5:
                    sample_reasons[i].append(f"hidden_policy_defection:layer={telem.max_layer_idx}")

        latent_cos_drift: Optional[torch.Tensor] = None
        output_kl_per_sample: Optional[torch.Tensor] = None
        drift_thresh = (
            float(max_latent_cosine_drift)
            if max_latent_cosine_drift is not None
            else self.max_latent_cosine_drift
        )

        if ref_model is not None and b > 0:
            ref_was_train = ref_model.training
            ref_model.eval()
            ref_mgr = AllLayerHiddenHookManager(
                detach=True,
                include_blocks=active_mon.hook_manager.include_blocks,
                include_leaves=active_mon.hook_manager.include_leaves,
                exclude_output_head=active_mon.hook_manager.exclude_output_head,
            )
            try:
                with ref_mgr.capture(ref_model, detach=True, training_only=False):
                    ref_raw = _forward_model(ref_model, policy_inputs)
                    ref_hiddens = ref_mgr.get_named_hiddens()
            finally:
                if ref_was_train:
                    ref_model.train()

            cos_drifts: List[torch.Tensor] = []
            for (_, hp), (_, hr) in zip(pol_hiddens, ref_hiddens):
                fp = _pool_hidden_features(hp).to(dev)
                fr = _pool_hidden_features(hr).to(dev)
                if fp.shape == fr.shape and fp.shape[0] == b:
                    d_l = 1.0 - F.cosine_similarity(fp, fr, dim=-1, eps=1e-8)
                    cos_drifts.append(d_l.clamp(min=0.0))
            if cos_drifts:
                latent_cos_drift = torch.stack(cos_drifts, dim=0).mean(dim=0)
                bad_drift = latent_cos_drift > drift_thresh
                for i in range(b):
                    if bool(bad_drift[i].item()):
                        weights[i] = 0.0
                        sample_reasons[i].append("latent_belief_drift")

            ref_logits, _ = _extract_primary_tensor(ref_raw)
            if pol_logits is not None and ref_logits is not None and pol_logits.shape == ref_logits.shape and pol_logits.ndim >= 2:
                lp = F.log_softmax(pol_logits.float().to(dev), dim=-1)
                lq = F.log_softmax(ref_logits.float().to(dev), dim=-1)
                kl_elem = (lp.exp() * (lp - lq)).sum(dim=-1)
                output_kl_per_sample = kl_elem.reshape(b, -1).mean(dim=-1)
                bad_kl = output_kl_per_sample > float(self.policy.max_kl_divergence)
                for i in range(b):
                    if bool(bad_kl[i].item()):
                        weights[i] = 0.0
                        sample_reasons[i].append("policy_kl_drift")

        bellman_residual: Optional[torch.Tensor] = None
        bell_thresh = max_bellman_residual if max_bellman_residual is not None else self.max_bellman_residual
        v_cur = self._extract_value_tensor(pol_raw)
        if v_cur is not None and next_inputs is not None and rewards is not None and b > 0:
            was_t = model.training
            model.eval()
            try:
                next_raw = _forward_model(model, next_inputs)
            finally:
                if was_t:
                    model.train()
            v_next = self._extract_value_tensor(next_raw)
            if v_next is not None and v_next.shape[0] == b:
                r_vec = rewards.detach().float().to(dev).reshape(b)
                td_err = (v_cur.to(dev) - (r_vec + float(gamma) * v_next.to(dev))).abs()
                bellman_residual = td_err
                if bell_thresh is not None:
                    bad_td = td_err > float(bell_thresh)
                    for i in range(b):
                        if bool(bad_td[i].item()):
                            weights[i] = 0.0
                            sample_reasons[i].append("bellman_consistency_violation")

        sample_risks = (
            telem.sample_risks.clone().to(dev)
            if telem.sample_risks is not None
            else torch.zeros(b, dtype=torch.float32, device=dev)
        )
        if latent_cos_drift is not None:
            sample_risks = torch.maximum(sample_risks, torch.clamp(latent_cos_drift / max(1e-6, drift_thresh), 0.0, 1.0))
        belief_consistency = torch.clamp(1.0 - sample_risks, 0.0, 1.0)

        defection_indices = [i for i in range(b) if float(weights[i].item()) < 0.5]
        belief_consistent = len(defection_indices) == 0
        quarantined = bool(telem.quarantined or (not belief_consistent and (b == 1 or len(defection_indices) / max(1, b) > 0.5)))

        result = {
            "sample_weights": weights,
            "defection_indices": defection_indices,
            "telemetry": telem,
            "latent_cosine_drift": latent_cos_drift,
            "output_kl_per_sample": output_kl_per_sample,
            "bellman_residual": bellman_residual,
            "belief_consistency_scores": belief_consistency,
            "belief_consistent": belief_consistent,
            "quarantined": quarantined,
            "reasons": [tuple(r) for r in sample_reasons],
        }
        self.last_latent_audit = result
        return result

    def evaluate_rollouts(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        rewards: torch.Tensor,
        thinking_traces: Optional[Sequence[Optional[str]]] = None,
        group_size: Optional[int] = None,
        explicit_safe_flags: Optional[Sequence[Optional[bool]]] = None,
        *,
        policy_model: Optional[nn.Module] = None,
        policy_inputs: Optional[Any] = None,
        ref_model: Optional[nn.Module] = None,
        hidden_probe_monitor: Optional[Any] = None,
        next_inputs: Optional[Any] = None,
    ) -> RLRolloutAudit:
        """Evaluate rollouts + CoT traces + rewards and compute safe GRPO advantages."""
        n = len(responses)
        if len(prompts) != n or rewards.numel() != n:
            raise ValueError("prompts, responses, and rewards must have identical batch length")

        rewards_f = rewards.detach().float().reshape(n)
        device = rewards_f.device

        lengths = [float(len(r or "")) for r in responses]
        sorted_lens = sorted(lengths)
        median_len = sorted_lens[n // 2] if n > 0 else 0.0
        reward_mean = float(rewards_f.mean().item()) if n > 0 else 0.0
        reward_std = float(rewards_f.std(unbiased=False).item()) if n > 1 else 1.0

        weights = torch.ones(n, dtype=torch.float32, device=device)
        clamped_rewards = rewards_f.clone()
        unsafe_indices: List[int] = []
        reward_hack_indices: List[int] = []
        reasons: List[Tuple[str, ...]] = []

        for i in range(n):
            raw_resp = responses[i] or ""
            extracted_think, visible_resp = extract_thinking_trace(raw_resp)
            think_i = thinking_traces[i] if (thinking_traces is not None and thinking_traces[i] is not None) else extracted_think
            vis_i = visible_resp if extracted_think else raw_resp
            explicit_i = explicit_safe_flags[i] if explicit_safe_flags is not None else None

            is_safe, is_rh, r_reasons = self._score_trajectory(
                prompt=prompts[i] or "",
                thinking=think_i or "",
                response=vis_i,
                reward=float(rewards_f[i].item()),
                median_len=median_len,
                reward_mean=reward_mean,
                reward_std=reward_std,
                explicit_safe=explicit_i,
            )
            reasons.append(r_reasons)
            if not is_safe:
                weights[i] = 0.0
                clamped_rewards[i] = min(float(rewards_f[i].item()), self.unsafe_reward_clamp)
                unsafe_indices.append(i)
                if is_rh:
                    reward_hack_indices.append(i)

        hidden_defection_indices: List[int] = []
        latent_telemetry: Optional[Any] = None
        belief_consistency_scores: Optional[torch.Tensor] = None
        active_mon = hidden_probe_monitor or self.hidden_probe_monitor
        if policy_model is not None and policy_inputs is not None and active_mon is not None:
            latent_res = self.audit_hidden_policy_states(
                policy_model,
                policy_inputs,
                ref_model=ref_model,
                next_inputs=next_inputs,
                rewards=rewards_f,
                monitor=active_mon,
            )
            latent_telemetry = latent_res["telemetry"]
            belief_consistency_scores = latent_res["belief_consistency_scores"]
            lat_w = latent_res["sample_weights"].to(device=device)
            lat_reasons = latent_res["reasons"]
            for i in range(min(n, int(lat_w.shape[0]))):
                if float(lat_w[i].item()) < 0.5:
                    hidden_defection_indices.append(i)
                    if float(weights[i].item()) >= 0.5:
                        weights[i] = 0.0
                        clamped_rewards[i] = min(float(rewards_f[i].item()), self.unsafe_reward_clamp)
                        unsafe_indices.append(i)
                    merged = tuple(dict.fromkeys(reasons[i] + lat_reasons[i]))
                    reasons[i] = merged

        self.blocked_trajectories += len(unsafe_indices)

        # Compute advantages AFTER clamping unsafe rewards, and mask unsafe advantages to 0.0
        advantages = torch.zeros_like(clamped_rewards)
        step_g = int(group_size) if (group_size is not None and group_size >= 2 and n % group_size == 0) else n
        for start in range(0, n, max(1, step_g)):
            end = min(n, start + max(1, step_g))
            w_grp = weights[start:end]
            r_grp = clamped_rewards[start:end]
            clean_mask = w_grp > 0.5
            if not bool(clean_mask.any()):
                continue
            # Use clamped group statistics so unsafe trajectories never inflate baseline above safe ones
            safe_min = float(r_grp[clean_mask].min().item())
            r_for_adv = torch.where(
                clean_mask,
                r_grp,
                torch.clamp(r_grp, max=min(self.unsafe_reward_clamp, safe_min - 0.5)),
            )
            g_mean = r_for_adv.mean()
            g_std = r_for_adv.std(unbiased=False).clamp(min=1e-6)
            grp_adv = (r_for_adv - g_mean) / g_std
            advantages[start:end] = torch.where(clean_mask, grp_adv, torch.zeros_like(grp_adv))

        return RLRolloutAudit(
            sample_weights=weights,
            clamped_rewards=clamped_rewards,
            advantages=advantages,
            unsafe_indices=unsafe_indices,
            reward_hack_indices=reward_hack_indices,
            reasons=reasons,
            hidden_defection_indices=hidden_defection_indices,
            latent_telemetry=latent_telemetry,
            belief_consistency_scores=belief_consistency_scores,
        )

    def compute_kl_divergence(
        self,
        policy_values: torch.Tensor,
        ref_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_logits: bool = True,
    ) -> Tuple[torch.Tensor, bool]:
        """Compute non-negative ``KL(pi_theta || pi_ref)`` and check ``policy.max_kl_divergence``."""
        pol = policy_values.float()
        ref = ref_values.detach().float()
        if is_logits:
            log_p = F.log_softmax(pol, dim=-1)
            log_q = F.log_softmax(ref, dim=-1)
            p = log_p.exp()
            kl_elem = (p * (log_p - log_q)).sum(dim=-1)
        else:
            # Schulman k3 unbiased non-negative KL estimator from sampled log-probs
            log_ratio = ref - pol
            kl_elem = (torch.exp(torch.clamp(log_ratio, max=20.0)) - 1.0) - log_ratio

        if attention_mask is not None:
            mask = attention_mask.to(device=kl_elem.device, dtype=kl_elem.dtype)
            kl_mean = (kl_elem * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_mean = kl_elem.mean()

        kl_val = float(kl_mean.detach().item())
        self.last_kl_divergence = kl_val
        halt = kl_val > float(self.policy.max_kl_divergence)
        self.last_halt_update = halt
        if halt:
            self.halted_steps += 1
        return kl_mean, halt

    def _apply_kl_gate(
        self,
        policy_loss_per_sample: torch.Tensor,
        sample_weights: torch.Tensor,
        kl_tensor: torch.Tensor,
        halt: bool,
        advantages: torch.Tensor,
    ) -> RLStepLossResult:
        w = (sample_weights >= 0.5).to(device=policy_loss_per_sample.device, dtype=policy_loss_per_sample.dtype)
        blocked_count = int((w < 0.5).sum().item())
        active_count = int((w >= 0.5).sum().item())

        if halt and self.policy.fail_closed:
            zero_loss = (policy_loss_per_sample * 0.0).sum()
            return RLStepLossResult(
                loss=zero_loss,
                kl_divergence=self.last_kl_divergence,
                halt_update=True,
                blocked_count=blocked_count,
                active_count=active_count,
                advantages=advantages,
            )

        masked_pg_loss = StreamingPoisonGuard.apply_sample_mask(policy_loss_per_sample, w, normalize=True)
        if halt:
            # Clip policy gradient scale when fail_closed=False
            scale = float(self.policy.max_kl_divergence) / max(1e-8, self.last_kl_divergence)
            masked_pg_loss = masked_pg_loss * scale

        total_loss = masked_pg_loss + self.kl_coef * kl_tensor.to(masked_pg_loss.dtype)
        return RLStepLossResult(
            loss=total_loss,
            kl_divergence=self.last_kl_divergence,
            halt_update=halt,
            blocked_count=blocked_count,
            active_count=active_count,
            advantages=advantages,
        )

    def grpo_step_loss(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        audit: RLRolloutAudit,
        policy_logits: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        clip_eps: float = 0.2,
    ) -> RLStepLossResult:
        """Compute guarded GRPO loss with ``w in {0, 1}`` autograd masking and KL gate."""
        w = audit.sample_weights.to(device=policy_logprobs.device, dtype=policy_logprobs.dtype)
        adv = audit.advantages.to(device=policy_logprobs.device, dtype=policy_logprobs.dtype)

        if policy_logprobs.ndim > 1:
            if attention_mask is not None:
                m = attention_mask.to(device=policy_logprobs.device, dtype=policy_logprobs.dtype)
                seq_logp = (policy_logprobs * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
                old_seq_logp = (
                    (old_logprobs.detach() * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
                    if old_logprobs is not None
                    else seq_logp.detach()
                )
            else:
                seq_logp = policy_logprobs.mean(dim=-1)
                old_seq_logp = old_logprobs.detach().mean(dim=-1) if old_logprobs is not None else seq_logp.detach()
        else:
            seq_logp = policy_logprobs
            old_seq_logp = old_logprobs.detach() if old_logprobs is not None else seq_logp.detach()

        ratio = torch.exp(torch.clamp(seq_logp - old_seq_logp, min=-20.0, max=20.0))
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.minimum(ratio * adv, clipped * adv)
        per_sample_loss = -surrogate

        if policy_logits is not None and ref_logits is not None:
            kl_tensor, halt = self.compute_kl_divergence(
                policy_logits, ref_logits, attention_mask=attention_mask, is_logits=True
            )
        else:
            kl_tensor, halt = self.compute_kl_divergence(
                policy_logprobs, ref_logprobs, attention_mask=attention_mask, is_logits=False
            )

        return self._apply_kl_gate(per_sample_loss, w, kl_tensor, halt, adv)

    def ppo_step_loss(
        self,
        policy_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        sample_weights: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        clip_eps: float = 0.2,
    ) -> RLStepLossResult:
        """Compute guarded PPO clipped surrogate loss with sample masking and KL budget gate."""
        dummy_audit = RLRolloutAudit(
            sample_weights=sample_weights,
            clamped_rewards=torch.zeros_like(advantages),
            advantages=advantages * (sample_weights >= 0.5).to(advantages.dtype),
            unsafe_indices=[],
            reward_hack_indices=[],
            reasons=[],
        )
        return self.grpo_step_loss(
            policy_logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs,
            audit=dummy_audit,
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            old_logprobs=old_logprobs,
            attention_mask=attention_mask,
            clip_eps=clip_eps,
        )

    def dpo_step_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        beta: float = 0.1,
    ) -> RLStepLossResult:
        """Compute guarded DPO preference loss with KL drift check and ``w in {0, 1}`` masking."""
        b = policy_chosen_logps.shape[0]
        w = (
            sample_weights.to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
            if sample_weights is not None
            else torch.ones(b, device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
        )
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps.detach() - ref_rejected_logps.detach()
        logits_diff = beta * (pi_logratios - ref_logratios)
        per_sample_loss = -F.logsigmoid(logits_diff)

        kl_tensor, halt = self.compute_kl_divergence(
            policy_chosen_logps, ref_chosen_logps, is_logits=False
        )
        return self._apply_kl_gate(
            per_sample_loss,
            w,
            kl_tensor,
            halt,
            advantages=logits_diff.detach(),
        )


__all__ = [
    "extract_thinking_trace",
    "RLRolloutAudit",
    "RLStepLossResult",
    "RLSafetyShield",
]
