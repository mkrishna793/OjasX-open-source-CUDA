"""Fine-tuning subspace safety lock for LoRA, E8LatticeLinear, and nn.Linear layers.

Calibrates low-rank empirical safety Fisher subspaces (U_out, V_in) and anchor
alignment directions on safe refusal/alignment examples without mutating
parameters or triggering grad-free optimizer hooks, then registers autograd
parameter hooks that:
1. Project out harmful components along the protected safety subspace, and
2. Detect 1%-poison trigger rank-1 gradient anomalies (fail-closed zeroing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import E8LatticeLinear
from ojasx_safety.contracts import SafetyPolicy
from ojasx_safety.autograd.poison_guard import StreamingWelfordStats


@dataclass
class LoRASubspaceState:
    layer_name: str
    layer_type: str
    in_features: int
    out_features: int
    u_out: torch.Tensor  # [out_f, k_out] orthonormal safety output subspace
    v_in: torch.Tensor   # [in_f, k_in] orthonormal safety input subspace
    anchor_grad_dir: Optional[torch.Tensor] = None
    norm_stats: StreamingWelfordStats = field(default_factory=lambda: StreamingWelfordStats(shape=()))
    rank1_stats: StreamingWelfordStats = field(default_factory=lambda: StreamingWelfordStats(shape=()))
    safety_ratio_stats: StreamingWelfordStats = field(default_factory=lambda: StreamingWelfordStats(shape=()))
    ema_norm: Optional[float] = None
    ema_norm_dev: float = 0.0
    steps: int = 0


def _fast_top_singular_value_sq(mat: torch.Tensor, n_iter: int = 4) -> torch.Tensor:
    """Device-native estimate of sigma_1(mat)^2 via power iteration (or exact for small matrices)."""
    m, n = mat.shape
    work = mat.float()
    if min(m, n) <= 32:
        s = torch.linalg.svdvals(work)
        return (s[0] * s[0]) if s.numel() > 0 else torch.zeros((), device=mat.device)
    v = torch.ones(n, 1, device=mat.device, dtype=torch.float32) / (float(n) ** 0.5)
    for _ in range(max(1, n_iter)):
        u = work @ v
        u_norm = torch.linalg.vector_norm(u).clamp(min=1e-8)
        u = u / u_norm
        v = work.t() @ u
        v_norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        v = v / v_norm
    sigma = torch.linalg.vector_norm(work @ v)
    return sigma * sigma


def _orthonormal_top_k(samples: torch.Tensor, k: int) -> torch.Tensor:
    """Compute top-k right singular basis ``[D, k]`` of ``samples`` ``[N, D]`` on device."""
    work = samples.detach().float()
    if work.ndim != 2 or work.shape[0] == 0:
        d = work.shape[-1] if work.ndim >= 1 else 1
        return torch.zeros(d, 1, dtype=torch.float32, device=work.device)
    n, d = work.shape
    k_eff = max(1, min(k, n, d))
    _, _, vh = torch.linalg.svd(work, full_matrices=False)
    basis = vh[:k_eff].t().contiguous()  # [D, k_eff]
    return basis


class LoRASafetyLock:
    """Calibrates and enforces low-rank safety Fisher locks across Linear / E8 / LoRA layers."""

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        rank_k: int = 4,
        projection_strength: Optional[float] = None,
        detect_rank1_poison: bool = True,
        zero_on_poison_anomaly: Optional[bool] = None,
        warmup_steps: int = 4,
        harmful_only_linear: bool = False,
        project_initial_lora: bool = True,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.rank_k = max(1, int(rank_k))
        self.projection_strength = (
            float(projection_strength)
            if projection_strength is not None
            else float(self.policy.lora_safety_projection_strength)
        )
        self.detect_rank1_poison = bool(detect_rank1_poison)
        self.zero_on_poison_anomaly = (
            bool(zero_on_poison_anomaly)
            if zero_on_poison_anomaly is not None
            else bool(self.policy.fail_closed)
        )
        self.warmup_steps = max(1, int(warmup_steps))
        self.harmful_only_linear = bool(harmful_only_linear)
        self.project_initial_lora = bool(project_initial_lora)

        self.subspaces: Dict[str, LoRASubspaceState] = {}
        self._handles: List[Any] = []
        self.last_projected_norm: float = 0.0
        self.last_anomaly_detected: bool = False
        self.last_anomaly_layer: str = ""
        self.anomaly_count: int = 0
        self.projected_steps: int = 0
        self._cb_loss: Optional[Any] = None
        self._rmu_loss: Optional[Any] = None

    @staticmethod
    def _is_lora_wrapper(mod: nn.Module) -> bool:
        return (
            hasattr(mod, "base_layer")
            and hasattr(mod, "lora_A")
            and hasattr(mod, "lora_B")
            and isinstance(getattr(mod, "lora_A"), nn.Parameter)
            and isinstance(getattr(mod, "lora_B"), nn.Parameter)
        )

    @classmethod
    def discover_target_layers(
        cls,
        model: nn.Module,
        target_modules: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, nn.Module]]:
        """Discover LoRALinear, E8LatticeLinear, and nn.Linear layers without double-visiting wrappers."""
        lora_targets: List[Tuple[str, nn.Module]] = []
        e8_lora_targets: List[Tuple[str, nn.Module]] = []
        linear_targets: List[Tuple[str, nn.Module]] = []

        def _visit(parent: nn.Module, prefix: str) -> None:
            for name, child in parent.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                matched = target_modules is None or any(t in full_name for t in target_modules)
                if cls._is_lora_wrapper(child):
                    if matched and (child.lora_A.requires_grad or child.lora_B.requires_grad):
                        lora_targets.append((full_name, child))
                    # Do NOT descend into base_layer of LoRALinear
                    continue
                if isinstance(child, E8LatticeLinear):
                    if matched:
                        e8_lora_targets.append((full_name, child))
                    continue
                if isinstance(child, nn.Linear):
                    if matched and child.weight.requires_grad:
                        linear_targets.append((full_name, child))
                    continue
                _visit(child, full_name)

        if cls._is_lora_wrapper(model):
            if model.lora_A.requires_grad or model.lora_B.requires_grad:
                lora_targets.append(("self", model))
        elif isinstance(model, E8LatticeLinear):
            e8_lora_targets.append(("self", model))
        elif isinstance(model, nn.Linear):
            if model.weight.requires_grad:
                linear_targets.append(("self", model))
        else:
            _visit(model, "")
        if lora_targets:
            return lora_targets + [
                (n, m) for n, m in e8_lora_targets
                if m.rank > 0 and ((m.lora_A is not None and m.lora_A.requires_grad) or (m.lora_B is not None and m.lora_B.requires_grad))
            ]
        if e8_lora_targets:
            return e8_lora_targets
        return linear_targets

    @staticmethod
    def _default_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get("input_ids", batch.get("inputs"))
            targets = batch.get("labels", batch.get("targets"))
        else:
            inputs, targets = batch, batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        out = model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.ndim >= 2 and targets.dtype in (torch.int64, torch.int32, torch.long):
            if logits.ndim == 3 and targets.ndim == 1 and targets.shape[0] == logits.shape[0]:
                return F.cross_entropy(logits[:, -1, :], targets.long())
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return F.mse_loss(logits, targets.to(logits.dtype))

    def calibrate(
        self,
        model: nn.Module,
        calibration_batches: Iterable[Any],
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        max_batches: int = 16,
        target_modules: Optional[Sequence[str]] = None,
    ) -> Dict[str, LoRASubspaceState]:
        """Calibrate empirical safety Fisher subspaces using ``torch.autograd.grad`` (no weight mutation)."""
        layers = self.discover_target_layers(model, target_modules=target_modules)
        if not layers:
            return self.subspaces

        eval_loss_fn = loss_fn or self._default_loss
        in_samples: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in layers}
        out_grads: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in layers}
        anchor_acc: Dict[str, Optional[torch.Tensor]] = {name: None for name, _ in layers}

        was_training = model.training
        model.eval()
        # Temporarily ensure we can backprop through outputs even if all base weights are frozen
        # (for pure frozen E8 base without LoRA, input activations still inform V_in).
        for step_idx, batch in enumerate(calibration_batches):
            if step_idx >= max_batches:
                break
            captured_in: Dict[str, torch.Tensor] = {}
            captured_out: List[Tuple[str, torch.Tensor]] = []
            temp_hooks: List[Any] = []

            for name, mod in layers:
                def _make_fwd_hook(layer_name: str):
                    def _hook(_m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        if inp and isinstance(inp[0], torch.Tensor):
                            x_flat = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                            captured_in[layer_name] = x_flat
                        if isinstance(out, torch.Tensor):
                            if not out.requires_grad and out.is_floating_point():
                                out.requires_grad_(True)
                            captured_out.append((layer_name, out))
                    return _hook

                temp_hooks.append(mod.register_forward_hook(_make_fwd_hook(name)))

            try:
                loss = eval_loss_fn(model, batch)
                req_outs = [(n, t) for n, t in captured_out if t.requires_grad]
                if req_outs:
                    grads = torch.autograd.grad(
                        loss,
                        [t for _, t in req_outs],
                        retain_graph=False,
                        allow_unused=True,
                    )
                    for (lname, _), g in zip(req_outs, grads):
                        if g is None:
                            continue
                        g_flat = g.detach().float().reshape(-1, g.shape[-1])
                        x_flat = captured_in.get(lname)
                        if x_flat is not None:
                            row_weights = torch.linalg.vector_norm(g_flat, dim=-1, keepdim=True)
                            weighted_x = x_flat * (row_weights + 1e-6)
                            in_samples[lname].append(weighted_x)
                            if g_flat.shape[1] * x_flat.shape[1] <= 262144:
                                gw = g_flat.t() @ x_flat
                                anchor_acc[lname] = gw if anchor_acc[lname] is None else (anchor_acc[lname] + gw)
                        out_grads[lname].append(g_flat)
                else:
                    for lname, x_flat in captured_in.items():
                        in_samples[lname].append(x_flat)
            finally:
                for h in temp_hooks:
                    h.remove()

        if was_training:
            model.train()

        for name, mod in layers:
            in_f = int(getattr(mod, "in_features", getattr(mod, "in_dim", 0)))
            out_f = int(getattr(mod, "out_features", getattr(mod, "out_dim", 0)))
            if in_f <= 0 or out_f <= 0:
                continue

            # Cap subspace rank strictly below full dimension so benign tasks retain free capacity.
            k_out_cap = min(self.rank_k, max(1, out_f // 4))
            k_in_cap = min(self.rank_k, max(1, in_f // 4))

            if out_grads[name]:
                g_cat = torch.cat(out_grads[name], dim=0)
                u_out = _orthonormal_top_k(g_cat, k_out_cap)
            else:
                u_out = torch.zeros(out_f, 1, dtype=torch.float32)

            if in_samples[name]:
                x_cat = torch.cat(in_samples[name], dim=0)
                v_in = _orthonormal_top_k(x_cat, k_in_cap)
            else:
                v_in = torch.zeros(in_f, 1, dtype=torch.float32)

            anchor = anchor_acc.get(name)
            if anchor is not None:
                a_norm = torch.linalg.vector_norm(anchor).clamp(min=1e-8)
                anchor = anchor / a_norm

            ltype = (
                "LoRALinear"
                if self._is_lora_wrapper(mod)
                else ("E8LatticeLinear" if isinstance(mod, E8LatticeLinear) else "nn.Linear")
            )
            self.subspaces[name] = LoRASubspaceState(
                layer_name=name,
                layer_type=ltype,
                in_features=in_f,
                out_features=out_f,
                u_out=u_out,
                v_in=v_in,
                anchor_grad_dir=anchor,
            )
        return self.subspaces

    def reset_step_metrics(self) -> None:
        self.last_projected_norm = 0.0
        self.last_anomaly_detected = False
        self.last_anomaly_layer = ""

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_param_hook(
        self,
        layer_name: str,
        role: str,  # "lora_A_peft", "lora_B_peft", "lora_A_e8", "lora_B_e8", "linear_weight"
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or not bool(torch.isfinite(grad).all()):
                self.last_anomaly_detected = True
                self.last_anomaly_layer = layer_name
                self.anomaly_count += 1
                return torch.zeros_like(grad) if grad is not None else grad

            state = self.subspaces.get(layer_name)
            if state is None:
                return grad

            g_f32 = grad.detach().float()
            strength = max(0.0, min(1.0, self.projection_strength))

            if role == "lora_A_peft":
                # shape: (rank, in_f)
                v = state.v_in.to(device=g_f32.device, dtype=torch.float32)
                proj = (g_f32 @ v) @ v.t()
            elif role == "lora_B_peft":
                # shape: (out_f, rank)
                u = state.u_out.to(device=g_f32.device, dtype=torch.float32)
                proj = u @ (u.t() @ g_f32)
            elif role == "lora_A_e8":
                # shape: (in_f, rank)
                v = state.v_in.to(device=g_f32.device, dtype=torch.float32)
                proj = v @ (v.t() @ g_f32)
            elif role == "lora_B_e8":
                # shape: (rank, out_f)
                u = state.u_out.to(device=g_f32.device, dtype=torch.float32)
                proj = (g_f32 @ u) @ u.t()
            else:
                # nn.Linear.weight shape: (out_f, in_f)
                u = state.u_out.to(device=g_f32.device, dtype=torch.float32)
                v = state.v_in.to(device=g_f32.device, dtype=torch.float32)
                if self.harmful_only_linear and state.anchor_grad_dir is not None:
                    anchor = state.anchor_grad_dir.to(device=g_f32.device, dtype=torch.float32)
                    dot = torch.sum(g_f32 * anchor)
                    # Only block updates anti-aligned with safety anchor in safety subspace
                    proj = u @ (u.t() @ g_f32 @ v) @ v.t() if float(dot.item()) > 0 else u @ (u.t() @ g_f32)
                else:
                    proj = u @ (u.t() @ g_f32)

            g_norm = torch.linalg.vector_norm(g_f32)
            proj_norm = torch.linalg.vector_norm(proj)
            safety_ratio = proj_norm / g_norm.clamp(min=1e-8)

            # Rank-1 poison trigger detection
            rank1_energy = torch.zeros((), device=g_f32.device, dtype=torch.float32)
            if self.detect_rank1_poison and g_f32.ndim == 2 and min(g_f32.shape) >= 2:
                sigma1_sq = _fast_top_singular_value_sq(g_f32)
                frob_sq = (g_norm * g_norm).clamp(min=1e-12)
                rank1_energy = torch.clamp(sigma1_sq / frob_sq, max=1.0)

            g_norm_val = float(g_norm.item())
            is_anomalous = False
            if self.detect_rank1_poison and state.steps >= self.warmup_steps and state.ema_norm is not None:
                eff_std = max(state.ema_norm_dev, 0.25 * abs(state.ema_norm), 1e-5)
                z_norm = abs(g_norm_val - state.ema_norm) / eff_std
                z_rank1 = float(state.rank1_stats.zscore(rank1_energy, min_std=0.05, rel_floor=0.10).item())
                z_safe = float(state.safety_ratio_stats.zscore(safety_ratio, min_std=0.05, rel_floor=0.10).item())
                if z_norm > float(self.policy.max_grad_zscore) and g_norm_val > state.ema_norm:
                    is_anomalous = True
                elif z_rank1 > float(self.policy.max_grad_zscore) and z_safe > 2.0:
                    is_anomalous = True

            if is_anomalous:
                self.last_anomaly_detected = True
                self.last_anomaly_layer = layer_name
                self.anomaly_count += 1
                self.last_projected_norm += g_norm_val
                if self.zero_on_poison_anomaly:
                    return torch.zeros_like(grad)
            else:
                if g_norm_val > 1e-12:
                    if state.ema_norm is None:
                        state.ema_norm = g_norm_val
                        state.ema_norm_dev = 0.2 * g_norm_val
                    else:
                        dev = abs(g_norm_val - state.ema_norm)
                        state.ema_norm = 0.85 * state.ema_norm + 0.15 * g_norm_val
                        state.ema_norm_dev = 0.85 * state.ema_norm_dev + 0.15 * dev
                    state.norm_stats.update(g_norm)
                    if g_f32.ndim == 2 and min(g_f32.shape) >= 2:
                        state.rank1_stats.update(rank1_energy)
                    state.safety_ratio_stats.update(safety_ratio)
                    state.steps += 1

            projected = g_f32 - strength * proj
            self.last_projected_norm += float((strength * proj_norm).item())
            self.projected_steps += 1
            return projected.to(dtype=grad.dtype)

        return _hook

    def attach(
        self,
        model: nn.Module,
        target_modules: Optional[Sequence[str]] = None,
    ) -> int:
        """Register gradient projection + anomaly hooks on calibrated layers."""
        self.remove_hooks()
        count = 0
        for name, mod in self.discover_target_layers(model, target_modules=target_modules):
            if name not in self.subspaces:
                continue
            state = self.subspaces[name]
            strength = max(0.0, min(1.0, self.projection_strength))
            if self._is_lora_wrapper(mod):
                if self.project_initial_lora and strength > 0.0:
                    with torch.no_grad():
                        v = state.v_in.to(device=mod.lora_A.device, dtype=torch.float32)
                        a_f = mod.lora_A.detach().float()
                        mod.lora_A.copy_((a_f - strength * ((a_f @ v) @ v.t())).to(mod.lora_A.dtype))
                if mod.lora_A.requires_grad:
                    self._handles.append(mod.lora_A.register_hook(self._make_param_hook(name, "lora_A_peft")))
                    count += 1
                if mod.lora_B.requires_grad:
                    self._handles.append(mod.lora_B.register_hook(self._make_param_hook(name, "lora_B_peft")))
                    count += 1
            elif isinstance(mod, E8LatticeLinear):
                if self.project_initial_lora and strength > 0.0 and mod.rank > 0 and mod.lora_A is not None:
                    with torch.no_grad():
                        v = state.v_in.to(device=mod.lora_A.device, dtype=torch.float32)
                        a_f = mod.lora_A.detach().float()
                        mod.lora_A.copy_((a_f - strength * (v @ (v.t() @ a_f))).to(mod.lora_A.dtype))
                if mod.rank > 0 and mod.lora_A is not None and mod.lora_A.requires_grad:
                    self._handles.append(mod.lora_A.register_hook(self._make_param_hook(name, "lora_A_e8")))
                    count += 1
                if mod.rank > 0 and mod.lora_B is not None and mod.lora_B.requires_grad:
                    self._handles.append(mod.lora_B.register_hook(self._make_param_hook(name, "lora_B_e8")))
                    count += 1
            elif isinstance(mod, nn.Linear) and mod.weight.requires_grad:
                self._handles.append(mod.weight.register_hook(self._make_param_hook(name, "linear_weight")))
                count += 1
        return count

    def _subspace_leakage_penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute differentiable orthogonal Fisher subspace leakage penalty on LoRA updates."""
        ref_param = next(model.parameters(), None)
        dev = ref_param.device if ref_param is not None else torch.device("cpu")
        if not self.subspaces:
            return torch.zeros((), device=dev, dtype=torch.float32)

        terms: List[torch.Tensor] = []
        for name, mod in self.discover_target_layers(model):
            state = self.subspaces.get(name)
            if state is None:
                continue
            if self._is_lora_wrapper(mod):
                u = state.u_out.to(device=mod.lora_B.device, dtype=torch.float32)
                v = state.v_in.to(device=mod.lora_A.device, dtype=torch.float32)
                # delta_W = scaling * (lora_B @ lora_A), shape [out_f, in_f]
                proj = (u.t() @ mod.lora_B.float()) @ (mod.lora_A.float() @ v) * float(getattr(mod, "scaling", 1.0))
                terms.append(proj.pow(2).sum())
            elif isinstance(mod, E8LatticeLinear) and mod.rank > 0 and mod.lora_A is not None and mod.lora_B is not None:
                u = state.u_out.to(device=mod.lora_B.device, dtype=torch.float32)
                v = state.v_in.to(device=mod.lora_A.device, dtype=torch.float32)
                # delta_W^T = scaling * (lora_A @ lora_B)
                proj = (v.t() @ mod.lora_A.float()) @ (mod.lora_B.float() @ u) * float(mod.scaling)
                terms.append(proj.pow(2).sum())
        if not terms:
            return torch.zeros((), device=dev, dtype=torch.float32)
        return torch.stack(terms).mean()

    def rerouting_and_rmu_loss(
        self,
        model: nn.Module,
        harmful_inputs: Any,
        retain_inputs: Optional[Any] = None,
        *,
        ref_model: Optional[nn.Module] = None,
        cb_weight: float = 1.0,
        rmu_weight: float = 1.0,
        retain_weight: float = 1.0,
        subspace_anchor_weight: float = 0.05,
        target_layers: Optional[Sequence[Any]] = None,
        return_details: bool = False,
    ) -> Any:
        """Combine Fisher subspace lock with CircuitBreaker (L_RR) + RMU (L_RMU) latent rerouting."""
        from ojasx_safety.autograd.latent_defenses import (
            CircuitBreakerReroutingLoss,
            RMUMisdirectionLoss,
            _freeze_ref_copy,
        )

        if ref_model is None:
            if self._cb_loss is not None and self._cb_loss.ref_model is not None:
                shared_ref = self._cb_loss.ref_model
            elif self._rmu_loss is not None and self._rmu_loss.ref_model is not None:
                shared_ref = self._rmu_loss.ref_model
            else:
                shared_ref = _freeze_ref_copy(model)
        else:
            shared_ref = ref_model

        if self._cb_loss is None:
            self._cb_loss = CircuitBreakerReroutingLoss(
                ref_model=shared_ref,
                alpha_retain=retain_weight,
                target_layers=target_layers,
            )
        else:
            self._cb_loss.alpha_retain = float(retain_weight)
            if target_layers is not None:
                self._cb_loss.target_layers = list(target_layers)

        if self._rmu_loss is None:
            self._rmu_loss = RMUMisdirectionLoss(
                ref_model=shared_ref,
                alpha=retain_weight,
                target_layers=target_layers,
            )
        else:
            self._rmu_loss.alpha = float(retain_weight)
            if target_layers is not None:
                self._rmu_loss.target_layers = list(target_layers)

        ref_param = next(model.parameters(), None)
        dev = ref_param.device if ref_param is not None else torch.device("cpu")
        total = torch.zeros((), device=dev, dtype=torch.float32)

        cb_val = torch.zeros_like(total)
        rmu_val = torch.zeros_like(total)
        subspace_val = torch.zeros_like(total)

        if cb_weight > 0.0:
            cb_val = self._cb_loss(model, harmful_inputs, retain_inputs=retain_inputs, ref_model=shared_ref)
            total = total + float(cb_weight) * cb_val.to(device=dev, dtype=torch.float32)

        if rmu_weight > 0.0:
            # Avoid double-counting retain term when CB already includes retain loss
            ret_for_rmu = None if (cb_weight > 0.0 and retain_inputs is not None) else retain_inputs
            rmu_val = self._rmu_loss(model, harmful_inputs, retain_inputs=ret_for_rmu, ref_model=shared_ref)
            total = total + float(rmu_weight) * rmu_val.to(device=dev, dtype=torch.float32)

        if subspace_anchor_weight > 0.0 and self.subspaces:
            subspace_val = self._subspace_leakage_penalty(model).to(device=dev, dtype=torch.float32)
            total = total + float(subspace_anchor_weight) * subspace_val

        if return_details:
            return total, {
                "cb_loss": float(cb_val.detach().item()),
                "rmu_loss": float(rmu_val.detach().item()),
                "subspace_penalty": float(subspace_val.detach().item()),
                "total": float(total.detach().item()),
            }
        return total


__all__ = [
    "LoRASubspaceState",
    "LoRASafetyLock",
]
