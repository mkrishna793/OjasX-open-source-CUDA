"""Architecture-universal, PyTorch-native multi-layer hidden-state defenses.

Implements zero-host-sync (no ``.cpu()`` / ``.numpy()`` in loss/grad paths)
deep latent defenses across arbitrary ``nn.Module`` architectures (Causal LMs,
``E8LatticeLinear``, ``LoRALinear``, ``GRAMPartitionedModule``, Gated SSMs,
Continuous Diffusion Denoisers, CNNs, and RL Actor-Critic policies):

1. ``AllLayerHiddenHookManager``: ordered multi-layer forward capture and
   differentiable in-flight perturbation injection across all hidden layers.
2. ``AllLayerLatentMonitor`` & ``LatentLayerTelemetry``: pure-PyTorch linear
   safety deflection probes (ridge + logistic refinement) and incremental
   Chan/Welford covariance Mahalanobis OOD distance across all hidden layers.
3. ``CircuitBreakerReroutingLoss``: Zou et al. (2024) Representation Rerouting
   cosine-ReLU penalty across target layers ``[0.25L, 0.75L]`` with retain
   hidden L2 + output KL/MSE preservation.
4. ``RMUMisdirectionLoss``: Li et al. (2024) WMDP Representation Misdirection
   steering forget hidden states toward ``c * u_l`` while preserving retain
   representations.
5. ``TVaccineLayerGuard`` & ``AllLayerLatentAdversarialTrainer``: T-Vaccine /
   Vaccine minimax and multi-step Latent Adversarial Training selecting Top-K
   safety-critical layers via ``E[||grad_h L_safe||_2]``, injecting bounded L2
   hidden perturbations during training, and training outer parameters with
   optional parameter anchor regularization.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import E8LatticeLinear
from ojasx_safety.contracts import SafetyPolicy
from ojasx_safety.autograd.poison_guard import StreamingWelfordStats


_SKIP_HOOK_TYPES = (
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Embedding,
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.Flatten,
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.LeakyReLU,
    nn.ELU,
    nn.Softmax,
    nn.LogSoftmax,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool1d,
    nn.AvgPool2d,
    nn.MaxPool2d,
)

_LEAF_COMPUTE_TYPES = (
    nn.Linear,
    E8LatticeLinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.RNNBase,
    nn.MultiheadAttention,
)

_BLOCK_KEYWORDS = (
    "block",
    "layer",
    "ssm",
    "mamba",
    "attn",
    "attention",
    "mlp",
    "denoiser",
    "actor",
    "critic",
    "encoder",
    "decoder",
    "residual",
    "mixer",
    "rwkv",
)

_OUTPUT_HEAD_NAMES = (
    "lm_head",
    "head",
    "classifier",
    "out_proj",
    "output_proj",
    "fc_out",
    "logits",
    "action_head",
)


def _is_compiling() -> bool:
    return bool(
        hasattr(torch, "compiler")
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    )


def _is_lora_wrapper(mod: nn.Module) -> bool:
    return (
        hasattr(mod, "base_layer")
        and hasattr(mod, "lora_A")
        and hasattr(mod, "lora_B")
        and isinstance(getattr(mod, "lora_A"), nn.Parameter)
        and isinstance(getattr(mod, "lora_B"), nn.Parameter)
    )


def _is_gram_wrapper(mod: nn.Module) -> bool:
    return (
        hasattr(mod, "core")
        and hasattr(mod, "aux")
        and isinstance(getattr(mod, "aux"), nn.ModuleDict)
    )


def _extract_primary_tensor(
    output: Any,
) -> Tuple[Optional[torch.Tensor], Optional[Callable[[torch.Tensor], Any]]]:
    """Extract primary floating-point tensor and a non-destructive output rebuilder."""
    if isinstance(output, torch.Tensor):
        if output.is_floating_point():
            return output, lambda t: t
        return None, None

    if isinstance(output, (tuple, list)):
        for idx, item in enumerate(output):
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                is_tuple = isinstance(output, tuple)

                def _rebuild_seq(new_t: torch.Tensor, _idx: int = idx, _orig: Any = output, _tup: bool = is_tuple) -> Any:
                    items = list(_orig)
                    items[_idx] = new_t
                    return tuple(items) if _tup else items

                return item, _rebuild_seq
        return None, None

    if isinstance(output, dict):
        preferred = (
            "last_hidden_state",
            "hidden_states",
            "logits",
            "action_logits",
            "pred",
            "noise_pred",
            "output",
            "out",
        )
        key_found: Optional[str] = None
        for k in preferred:
            v = output.get(k)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                key_found = k
                break
        if key_found is None:
            for k, v in output.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    key_found = k
                    break
        if key_found is not None:
            def _rebuild_dict(new_t: torch.Tensor, _k: str = key_found, _orig: Dict[Any, Any] = output) -> Dict[Any, Any]:
                copied = dict(_orig)
                copied[_k] = new_t
                return copied

            return output[key_found], _rebuild_dict
        return None, None

    for attr in ("last_hidden_state", "logits"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            def _rebuild_obj(new_t: torch.Tensor, _attr: str = attr, _orig: Any = output) -> Any:
                if isinstance(_orig, SimpleNamespace):
                    ns_dict = dict(vars(_orig))
                    ns_dict[_attr] = new_t
                    return SimpleNamespace(**ns_dict)
                copied = copy.copy(_orig)
                setattr(copied, _attr, new_t)
                return copied

            return val, _rebuild_obj

    return None, None


def _forward_model(model: nn.Module, model_input: Any) -> Any:
    """Invoke ``model`` cleanly for Tensor, tuple/list, or dict inputs."""
    if isinstance(model_input, dict):
        return model(**model_input)
    if isinstance(model_input, (tuple, list)):
        return model(*model_input)
    return model(model_input)


def _infer_device(model: nn.Module, fallback_input: Any = None) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    if isinstance(fallback_input, torch.Tensor):
        return fallback_input.device
    if isinstance(fallback_input, (tuple, list)):
        for item in fallback_input:
            if isinstance(item, torch.Tensor):
                return item.device
    if isinstance(fallback_input, dict):
        for v in fallback_input.values():
            if isinstance(v, torch.Tensor):
                return v.device
    return torch.device("cpu")


def _default_task_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    """Architecture-universal differentiable task loss supporting LM, SSM, Diffusion, and RL."""
    dev = _infer_device(model, batch)
    targets: Optional[torch.Tensor] = None
    model_in: Any = batch

    if isinstance(batch, dict):
        if "inputs" in batch and "targets" in batch:
            model_in, targets = batch["inputs"], batch["targets"]
        elif "input_ids" in batch and "labels" in batch:
            model_in, targets = batch["input_ids"], batch["labels"]
        elif "x_t" in batch and "t" in batch:
            model_in = (batch["x_t"], batch["t"])
            targets = batch.get("target", batch.get("noise", batch.get("targets")))
        elif "obs" in batch:
            model_in = batch["obs"]
            targets = batch.get("actions", batch.get("targets", batch.get("labels")))
        else:
            model_in = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
            # Could be (inputs, targets) OR Diffusion (x_t, t)
            if batch[0].is_floating_point() and batch[1].ndim == 1 and batch[1].shape[0] == batch[0].shape[0] and not batch[1].is_floating_point():
                model_in, targets = batch[0], batch[1]
            elif batch[0].is_floating_point() and batch[1].shape == batch[0].shape:
                model_in, targets = batch[0], batch[1]
            elif not batch[0].is_floating_point():
                model_in, targets = batch[0], batch[1]
            else:
                model_in, targets = batch[0], batch[1]
        elif len(batch) >= 3 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
            # Support ((x_t, t), target) or (x_t, t, target)
            if isinstance(batch[2], torch.Tensor):
                model_in = (batch[0], batch[1])
                targets = batch[2]
            else:
                model_in, targets = batch[0], batch[1]

    if isinstance(model_in, torch.Tensor):
        model_in = model_in.to(dev)
    elif isinstance(model_in, (tuple, list)):
        moved = [item.to(dev) if isinstance(item, torch.Tensor) else item for item in model_in]
        model_in = tuple(moved) if isinstance(model_in, tuple) else moved
    elif isinstance(model_in, dict):
        model_in = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in model_in.items()}

    out = _forward_model(model, model_in)
    primary, _ = _extract_primary_tensor(out)
    if primary is None:
        raise TypeError("Model output does not contain a floating-point tensor")

    if targets is None:
        return 0.5 * primary.float().pow(2).mean()

    tgt = targets.to(dev)
    if not tgt.is_floating_point() and primary.ndim >= 2:
        if primary.ndim == 3 and tgt.ndim == 1 and tgt.shape[0] == primary.shape[0]:
            return F.cross_entropy(primary[:, -1, :].float(), tgt.long(), ignore_index=-100)
        return F.cross_entropy(
            primary.reshape(-1, primary.size(-1)).float(),
            tgt.reshape(-1).long(),
            ignore_index=-100,
        )
    return F.mse_loss(primary.float(), tgt.to(dtype=torch.float32))


def _pool_hidden_features(h: torch.Tensor) -> torch.Tensor:
    """Pool arbitrary intermediate hidden tensor ``[B, ...]`` into 2D ``[B, D_feat]`` on device."""
    x = h.float()
    if x.ndim <= 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        if x.shape[1] > 1:
            return torch.cat([x.mean(dim=1), x[:, -1, :]], dim=-1)
        return x.squeeze(1)
    if x.ndim == 4:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=-1)
    return x.reshape(x.shape[0], -1)


def _norm_per_sample_or_global(t: torch.Tensor, per_sample: bool = True, eps: float = 1e-8) -> torch.Tensor:
    work = t.float()
    if per_sample and work.ndim >= 2:
        flat = work.reshape(work.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=-1).clamp(min=eps)
        return norms.view((work.shape[0],) + (1,) * (work.ndim - 1))
    return torch.linalg.vector_norm(work).clamp(min=eps)


# ---------------------------------------------------------------------------
# 1. AllLayerHiddenHookManager
# ---------------------------------------------------------------------------

class AllLayerHiddenHookManager:
    """Universal forward-hook collector & perturbation injector across all hidden layers."""

    def __init__(
        self,
        *,
        detach: bool = True,
        include_blocks: bool = True,
        include_leaves: bool = True,
        exclude_output_head: bool = False,
        target_layers: Optional[Sequence[Union[int, str]]] = None,
        max_layers: int = 128,
        training_only: bool = False,
    ) -> None:
        self.detach = bool(detach)
        self.include_blocks = bool(include_blocks)
        self.include_leaves = bool(include_leaves)
        self.exclude_output_head = bool(exclude_output_head)
        self.target_layers = list(target_layers) if target_layers is not None else None
        self.max_layers = max(1, int(max_layers))
        self.training_only = bool(training_only)

        self._handles: List[Any] = []
        self._discovered_names: List[str] = []
        self._ordered_captures: List[Tuple[str, torch.Tensor]] = []
        self._active_perturbations: Dict[str, torch.Tensor] = {}
        self._perturbation_factory: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None
        self._allowed_module_ids: Set[int] = set()
        self._root_model_id: Optional[int] = None

    def discover_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Discover ordered intermediate hidden-emitting layers respecting wrapper boundaries."""
        discovered: List[Tuple[str, nn.Module]] = []
        str_patterns = (
            [t for t in self.target_layers if isinstance(t, str)]
            if self.target_layers is not None
            else None
        )

        def _visit(parent: nn.Module, prefix: str) -> None:
            for child_name, child in parent.named_children():
                if len(discovered) >= self.max_layers:
                    return
                full_name = f"{prefix}.{child_name}" if prefix else child_name

                if _is_lora_wrapper(child) or _is_gram_wrapper(child) or isinstance(child, E8LatticeLinear):
                    if self.include_leaves:
                        if str_patterns is None or any(p in full_name for p in str_patterns):
                            discovered.append((full_name, child))
                    continue

                if isinstance(child, _SKIP_HOOK_TYPES) or "rmsnorm" in type(child).__name__.lower():
                    # Still descend if a container like nn.Sequential holds Linear/Conv blocks
                    if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                        _visit(child, full_name)
                    continue

                is_leaf_compute = isinstance(child, _LEAF_COMPUTE_TYPES) or len(list(child.children())) == 0

                if isinstance(child, _LEAF_COMPUTE_TYPES):
                    if self.include_leaves:
                        if str_patterns is None or any(p in full_name for p in str_patterns):
                            discovered.append((full_name, child))
                    continue

                if not is_leaf_compute:
                    if self.include_blocks:
                        if str_patterns is None or any(p in full_name for p in str_patterns):
                            discovered.append((full_name, child))
                    _visit(child, full_name)
                    continue

                if is_leaf_compute and self.include_leaves and any(p.requires_grad or True for p in child.parameters()):
                    if str_patterns is None or any(p in full_name for p in str_patterns):
                        discovered.append((full_name, child))
                    continue

                _visit(child, full_name)

        if _is_lora_wrapper(model) or _is_gram_wrapper(model) or isinstance(model, _LEAF_COMPUTE_TYPES):
            discovered.append(("self", model))
        else:
            _visit(model, "")

        if not discovered:
            for name, mod in model.named_modules():
                if name == "" or isinstance(mod, _SKIP_HOOK_TYPES):
                    continue
                if len(list(mod.children())) == 0:
                    discovered.append((name, mod))
                if len(discovered) >= self.max_layers:
                    break

        if self.exclude_output_head and len(discovered) >= 2:
            last_name = discovered[-1][0].split(".")[-1].lower()
            if any(last_name == h or last_name.endswith(h) for h in _OUTPUT_HEAD_NAMES):
                discovered = discovered[:-1]

        if self.target_layers is not None:
            int_indices = [t for t in self.target_layers if isinstance(t, int)]
            if int_indices:
                valid = []
                n_disc = len(discovered)
                for idx in int_indices:
                    pos = idx if idx >= 0 else n_disc + idx
                    if 0 <= pos < n_disc:
                        valid.append(discovered[pos])
                if valid:
                    discovered = valid

        return discovered[: self.max_layers]

    def clear(self) -> None:
        self._ordered_captures.clear()

    def set_perturbations(self, perturbations: Optional[Dict[str, torch.Tensor]] = None) -> None:
        self._active_perturbations = dict(perturbations) if perturbations else {}

    def set_perturbation_factory(
        self,
        factory: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self._perturbation_factory = factory

    def attach(
        self,
        model: nn.Module,
        *,
        on_forward_complete: Optional[Callable[[List[Tuple[str, torch.Tensor]]], None]] = None,
    ) -> List[str]:
        """Register ordered forward hooks on ``model`` and return discovered layer names."""
        self.remove_hooks()
        layers = self.discover_layers(model)
        self._discovered_names = [name for name, _ in layers]
        self._allowed_module_ids = {id(m) for m in model.modules()}
        self._root_model_id = id(model)

        allowed_ids = self._allowed_module_ids
        root_id = self._root_model_id
        mgr = self

        def _pre_hook(mod: nn.Module, _inp: Any) -> None:
            if id(mod) != root_id or _is_compiling():
                return
            if mgr.training_only and not mod.training:
                return
            mgr._ordered_captures.clear()

        self._handles.append(model.register_forward_pre_hook(_pre_hook))

        for name, mod in layers:
            def _make_layer_hook(layer_name: str):
                def _layer_hook(m: nn.Module, _inp: Any, out: Any) -> Any:
                    if id(m) not in allowed_ids or _is_compiling():
                        return out
                    if mgr.training_only and not m.training:
                        return out
                    primary, rebuilder = _extract_primary_tensor(out)
                    if primary is None or rebuilder is None:
                        return out
                    cap = primary.detach() if mgr.detach else primary
                    mgr._ordered_captures.append((layer_name, cap))

                    perturbed: Optional[torch.Tensor] = None
                    if mgr._perturbation_factory is not None:
                        perturbed = mgr._perturbation_factory(layer_name, primary)
                    elif layer_name in mgr._active_perturbations:
                        delta = mgr._active_perturbations[layer_name]
                        if delta.shape == primary.shape:
                            perturbed = primary + delta.to(device=primary.device, dtype=primary.dtype)
                    if perturbed is not None:
                        return rebuilder(perturbed)
                    return out

                return _layer_hook

            self._handles.append(mod.register_forward_hook(_make_layer_hook(name)))

        if on_forward_complete is not None:
            def _post_hook(mod: nn.Module, _inp: Any, _out: Any) -> None:
                if id(mod) != root_id or _is_compiling():
                    return
                if mgr.training_only and not mod.training:
                    return
                if mgr._ordered_captures:
                    on_forward_complete(list(mgr._ordered_captures))

            self._handles.append(model.register_forward_hook(_post_hook))

        return list(self._discovered_names)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._ordered_captures.clear()
        self._active_perturbations.clear()
        self._perturbation_factory = None

    @contextmanager
    def capture(
        self,
        model: nn.Module,
        *,
        detach: Optional[bool] = None,
        training_only: Optional[bool] = None,
    ) -> Iterator["AllLayerHiddenHookManager"]:
        """Temporarily attach hooks for a scoped capture pass and restore/clean up on exit."""
        prev_detach = self.detach
        prev_train_only = self.training_only
        had_hooks = bool(self._handles)
        if detach is not None:
            self.detach = bool(detach)
        if training_only is not None:
            self.training_only = bool(training_only)
        try:
            if not had_hooks or self._root_model_id != id(model):
                self.attach(model)
            self.clear()
            yield self
        finally:
            self.detach = prev_detach
            self.training_only = prev_train_only
            self.set_perturbations(None)
            self.set_perturbation_factory(None)
            if not had_hooks:
                self.remove_hooks()

    def get_named_hiddens(self) -> List[Tuple[str, torch.Tensor]]:
        return list(self._ordered_captures)

    def get_ordered_hiddens(self) -> List[torch.Tensor]:
        return [t for _, t in self._ordered_captures]

    @property
    def layer_names(self) -> List[str]:
        if self._ordered_captures:
            return [name for name, _ in self._ordered_captures]
        return list(self._discovered_names)

    def select_layer_indices(
        self,
        num_layers: Optional[int] = None,
        *,
        layer_range: Tuple[float, float] = (0.25, 0.75),
        target_layers: Optional[Sequence[Union[int, str]]] = None,
    ) -> List[int]:
        names = self.layer_names
        n = num_layers if num_layers is not None else len(names)
        if n <= 0:
            return []
        if target_layers is not None:
            selected: List[int] = []
            for item in target_layers:
                if isinstance(item, int):
                    idx = item if item >= 0 else n + item
                    if 0 <= idx < n and idx not in selected:
                        selected.append(idx)
                elif isinstance(item, str):
                    for idx, lname in enumerate(names[:n]):
                        if item in lname and idx not in selected:
                            selected.append(idx)
            if selected:
                return sorted(selected)

        lo, hi = float(layer_range[0]), float(layer_range[1])
        start = max(0, min(n - 1, int(math.floor(lo * n))))
        end = max(start + 1, min(n, int(math.ceil(hi * n))))
        if n >= 4 and end >= n and hi < 1.0:
            end = n - 1
        if end <= start:
            end = min(n, start + 1)
        return list(range(start, end))


# ---------------------------------------------------------------------------
# 2. AllLayerLatentMonitor & LatentLayerTelemetry
# ---------------------------------------------------------------------------

@dataclass
class LatentLayerTelemetry:
    per_layer_risk: List[float]
    max_layer_idx: int
    max_ood_zscore: float
    hidden_defection_detected: bool
    quarantined: bool
    per_layer_ood_zscore: List[float] = field(default_factory=list)
    per_layer_mahalanobis: List[float] = field(default_factory=list)
    layer_names: List[str] = field(default_factory=list)
    max_risk: float = 0.0
    sample_risks: Optional[torch.Tensor] = None
    sample_weights: Optional[torch.Tensor] = None
    reason: str = "clean"


class _LayerOODStats:
    """Pure-PyTorch incremental Chan/Welford covariance & Mahalanobis OOD tracker."""

    def __init__(self, max_cov_dim: int = 256, seed: int = 2026) -> None:
        self.max_cov_dim = max(8, int(max_cov_dim))
        self.seed = int(seed)
        self.count: int = 0
        self.raw_dim: int = 0
        self.cov_dim: int = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2_cov: Optional[torch.Tensor] = None
        self.proj: Optional[torch.Tensor] = None
        self.dm_stats = StreamingWelfordStats(shape=())
        self._cached_inv_cov: Optional[torch.Tensor] = None

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        vals = x.detach().float()
        if vals.ndim == 1:
            vals = vals.unsqueeze(0)
        d = vals.shape[1]
        if d <= self.max_cov_dim:
            return vals
        if self.proj is None or self.raw_dim != d:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed + d)
            p = torch.randn(d, self.max_cov_dim, dtype=torch.float32, generator=gen) / math.sqrt(self.max_cov_dim)
            self.proj = p.to(device=vals.device)
            self.raw_dim = d
        elif self.proj.device != vals.device:
            self.proj = self.proj.to(device=vals.device)
        return vals @ self.proj

    def _compute_inv_cov_from_stats(
        self,
        m2: torch.Tensor,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        denom = float(max(1, count - 1))
        raw_cov = m2.to(device=device) / denom
        diag_raw = torch.diagonal(raw_cov)
        avg_var = diag_raw.mean().clamp(min=1e-4)
        shrink = min(0.85, float(self.cov_dim) / float(self.cov_dim + max(2, count)))
        target_diag = torch.maximum(diag_raw, avg_var * 0.25) + 1e-4
        cov_reg = (1.0 - shrink) * raw_cov + shrink * torch.diag(target_diag)
        try:
            L = torch.linalg.cholesky(cov_reg)
            return torch.cholesky_inverse(L)
        except RuntimeError:
            return torch.linalg.pinv(cov_reg)

    def _get_inv_cov(self, device: torch.device) -> torch.Tensor:
        assert self.m2_cov is not None and self.cov_dim > 0
        if self._cached_inv_cov is not None and self._cached_inv_cov.device == device:
            return self._cached_inv_cov
        inv_cov = self._compute_inv_cov_from_stats(self.m2_cov, self.count, device)
        self._cached_inv_cov = inv_cov
        return inv_cov

    @torch.no_grad()
    def mahalanobis_and_zscore(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_x = self._project(x)
        b = proj_x.shape[0]
        if self.mean is None or self.m2_cov is None or self.count < 2 or proj_x.shape[1] != self.cov_dim:
            zeros = torch.zeros(b, device=proj_x.device, dtype=torch.float32)
            return zeros, zeros

        mu = self.mean.to(device=proj_x.device).unsqueeze(0)
        delta = proj_x - mu
        inv_cov = self._get_inv_cov(proj_x.device)
        dm_sq = ((delta @ inv_cov) * delta).sum(dim=-1).clamp(min=0.0)
        dm = torch.sqrt(dm_sq + 1e-8)

        if self.dm_stats.count >= 2 and self.dm_stats.mean is not None:
            mu_dm = self.dm_stats.mean.to(device=proj_x.device)
            std_dm = torch.maximum(
                self.dm_stats.std(min_std=0.20, rel_floor=0.12).to(device=proj_x.device),
                torch.clamp(mu_dm * 0.12, min=0.35),
            )
            z = torch.clamp((dm - mu_dm) / std_dm, min=0.0)
        else:
            exp_dm = math.sqrt(float(max(1, self.cov_dim)))
            z = torch.clamp((dm - exp_dm) / max(1.0, 0.20 * exp_dm), min=0.0)
        return dm, z

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        proj_x = self._project(x)
        n_b, d = proj_x.shape
        if n_b == 0:
            return
        self.cov_dim = d
        if self.raw_dim == 0:
            self.raw_dim = int(x.shape[-1]) if x.ndim >= 2 else d

        # Prequential out-of-sample DM update when prior baseline exists
        if self.count >= 4 and self.mean is not None and self.m2_cov is not None:
            dm_pre, _ = self.mahalanobis_and_zscore(x)
            self.dm_stats.update(dm_pre)

        b_mean = proj_x.mean(dim=0)
        centered = proj_x - b_mean.unsqueeze(0)
        b_m2 = centered.t() @ centered

        was_empty = self.mean is None or self.m2_cov is None or self.count == 0
        if was_empty:
            self.count = n_b
            self.mean = b_mean.clone()
            self.m2_cov = b_m2.clone()
        else:
            mu_a = self.mean.to(device=proj_x.device)
            m2_a = self.m2_cov.to(device=proj_x.device)
            delta = b_mean - mu_a
            total = self.count + n_b
            self.mean = mu_a + delta * (float(n_b) / float(total))
            self.m2_cov = m2_a + b_m2 + torch.outer(delta, delta) * (float(self.count * n_b) / float(total))
            self.count = total

        self._cached_inv_cov = None

        # On initial single-shot calibration batch, compute K-fold out-of-sample DM
        # so dm_stats reflects true held-out clean Mahalanobis distribution (no leverage bias).
        if was_empty and n_b >= 4:
            k_folds = min(4, n_b)
            fold_dms: List[torch.Tensor] = []
            for fold_i in range(k_folds):
                val_idx = torch.arange(fold_i, n_b, k_folds, device=proj_x.device)
                mask = torch.ones(n_b, dtype=torch.bool, device=proj_x.device)
                mask[val_idx] = False
                tr = proj_x[mask]
                va = proj_x[val_idx]
                if tr.shape[0] < 2 or va.shape[0] == 0:
                    continue
                mu_f = tr.mean(dim=0)
                c_tr = tr - mu_f.unsqueeze(0)
                m2_f = c_tr.t() @ c_tr
                inv_f = self._compute_inv_cov_from_stats(m2_f, int(tr.shape[0]), proj_x.device)
                d_va = va - mu_f.unsqueeze(0)
                dm_f = torch.sqrt(((d_va @ inv_f) * d_va).sum(dim=-1).clamp(min=0.0) + 1e-8)
                fold_dms.append(dm_f)
            if fold_dms:
                self.dm_stats.update(torch.cat(fold_dms, dim=0))
            else:
                dm, _ = self.mahalanobis_and_zscore(x)
                self.dm_stats.update(dm)


class AllLayerLatentMonitor:
    """Simultaneous all-layer linear safety probe + Mahalanobis OOD hidden monitor."""

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        *,
        probe_threshold: Optional[float] = None,
        ood_zscore_threshold: Optional[float] = None,
        max_cov_dim: int = 256,
        warmup_steps: int = 6,
        max_quarantine_batch_frac: float = 0.5,
        fail_closed_on_any_defection: bool = False,
        include_blocks: bool = True,
        include_leaves: bool = True,
        exclude_output_head: bool = True,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.probe_threshold = float(
            probe_threshold if probe_threshold is not None else self.policy.classifier_block_threshold
        )
        self.ood_zscore_threshold = float(
            ood_zscore_threshold if ood_zscore_threshold is not None else max(3.5, self.policy.max_grad_zscore)
        )
        self.max_cov_dim = max(8, int(max_cov_dim))
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_quarantine_batch_frac = float(max_quarantine_batch_frac)
        self.fail_closed_on_any_defection = bool(fail_closed_on_any_defection)

        self.hook_manager = AllLayerHiddenHookManager(
            detach=True,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
        )

        self.probe_weights: Dict[int, torch.Tensor] = {}
        self.probe_biases: Dict[int, torch.Tensor] = {}
        self.ood_stats: Dict[int, _LayerOODStats] = {}
        self.layer_name_map: Dict[int, str] = {}
        self.clean_batches_seen: int = 0
        self.explicitly_calibrated: bool = False
        self.quarantined_steps: int = 0
        self.last_telemetry: Optional[LatentLayerTelemetry] = None
        self._on_telemetry_cb: Optional[Callable[[LatentLayerTelemetry], None]] = None
        self._auto_update_baseline: bool = True

    @property
    def is_calibrated(self) -> bool:
        if self.explicitly_calibrated or bool(self.probe_weights):
            return True
        return self.clean_batches_seen >= self.warmup_steps and bool(self.ood_stats)

    @staticmethod
    def _normalize_layer_collection(
        hiddens: Union[Sequence[torch.Tensor], Dict[Union[int, str], torch.Tensor]],
    ) -> List[Tuple[str, torch.Tensor]]:
        if isinstance(hiddens, dict):
            return [(str(k), v) for k, v in hiddens.items()]
        return [(f"layer_{i}", v) for i, v in enumerate(hiddens)]

    @torch.no_grad()
    def fit_ood_baseline(
        self,
        safe_hiddens: Union[Sequence[torch.Tensor], Dict[Union[int, str], torch.Tensor]],
        *,
        reset: bool = True,
    ) -> None:
        """Calibrate per-layer Welford covariance Mahalanobis OOD baselines on clean hidden states."""
        if reset:
            self.ood_stats.clear()
        items = self._normalize_layer_collection(safe_hiddens)
        for idx, (lname, h) in enumerate(items):
            self.layer_name_map[idx] = lname
            feats = _pool_hidden_features(h)
            stats = self.ood_stats.setdefault(idx, _LayerOODStats(max_cov_dim=self.max_cov_dim, seed=2026 + idx))
            stats.update(feats)
        self.explicitly_calibrated = True
        self._auto_update_baseline = False

    def fit_probes(
        self,
        safe_hiddens: Union[Sequence[torch.Tensor], Dict[Union[int, str], torch.Tensor]],
        unsafe_hiddens: Union[Sequence[torch.Tensor], Dict[Union[int, str], torch.Tensor]],
        *,
        reg: float = 1.0,
        calibrate_ood: bool = True,
        refine_steps: int = 25,
    ) -> Dict[int, float]:
        """Fit per-layer linear safety probes ``p^(l) = sigmoid(w_l^T h_bar^(l) + b_l)`` in pure PyTorch."""
        if isinstance(safe_hiddens, dict) and isinstance(unsafe_hiddens, dict):
            common_keys = [k for k in safe_hiddens.keys() if k in unsafe_hiddens]
            safe_items = [(str(k), safe_hiddens[k]) for k in common_keys]
            unsafe_items = [(str(k), unsafe_hiddens[k]) for k in common_keys]
        else:
            safe_items = self._normalize_layer_collection(safe_hiddens)
            unsafe_items = self._normalize_layer_collection(unsafe_hiddens)
        n_layers = min(len(safe_items), len(unsafe_items))
        separation_scores: Dict[int, float] = {}

        if calibrate_ood:
            self.fit_ood_baseline(safe_hiddens, reset=True)

        for idx in range(n_layers):
            lname, h_safe = safe_items[idx]
            _, h_unsafe = unsafe_items[idx]
            self.layer_name_map[idx] = lname

            x_safe = _pool_hidden_features(h_safe).detach()
            x_unsafe = _pool_hidden_features(h_unsafe).detach().to(device=x_safe.device, dtype=torch.float32)
            n0, d = x_safe.shape
            n1 = x_unsafe.shape[0]
            if n0 == 0 or n1 == 0 or x_unsafe.shape[1] != d:
                continue

            x_all = torch.cat([x_safe, x_unsafe], dim=0)
            mu = x_all.mean(dim=0)
            std = x_all.std(dim=0, unbiased=False).clamp(min=1e-4)
            z_all = (x_all - mu.unsqueeze(0)) / std.unsqueeze(0)

            n_tot = n0 + n1
            ones = torch.ones(n_tot, 1, device=x_safe.device, dtype=torch.float32)
            design = torch.cat([z_all, ones], dim=1)  # [N, D+1]

            y_signed = torch.cat([
                torch.full((n0,), -1.0, device=x_safe.device, dtype=torch.float32),
                torch.full((n1,), +1.0, device=x_safe.device, dtype=torch.float32),
            ])
            y_bin = torch.cat([
                torch.zeros(n0, device=x_safe.device, dtype=torch.float32),
                torch.ones(n1, device=x_safe.device, dtype=torch.float32),
            ])
            cls_w = torch.cat([
                torch.full((n0,), 0.5 * float(n_tot) / float(n0), device=x_safe.device, dtype=torch.float32),
                torch.full((n1,), 0.5 * float(n_tot) / float(n1), device=x_safe.device, dtype=torch.float32),
            ])

            # Pure-PyTorch primal / dual ridge closed form
            sqrt_w = torch.sqrt(cls_w)
            dw = design * sqrt_w.unsqueeze(1)
            yw = y_signed * sqrt_w
            eff_reg = max(1e-4, float(reg))

            if d + 1 <= n_tot:
                gram = dw.t() @ dw + eff_reg * torch.eye(d + 1, device=x_safe.device, dtype=torch.float32)
                theta = torch.linalg.solve(gram, dw.t() @ yw)
            else:
                k_mat = dw @ dw.t() + eff_reg * torch.eye(n_tot, device=x_safe.device, dtype=torch.float32)
                dual = torch.linalg.solve(k_mat, yw)
                theta = dw.t() @ dual

            # Calibrate logit scale/shift in pure PyTorch so sigmoid probabilities are crisp
            raw_scores = (design @ theta).detach()
            scale = torch.tensor(2.0, device=x_safe.device, dtype=torch.float32, requires_grad=True)
            shift = torch.tensor(0.0, device=x_safe.device, dtype=torch.float32, requires_grad=True)
            if refine_steps > 0:
                opt = torch.optim.Adam([scale, shift], lr=0.25)
                for _ in range(refine_steps):
                    opt.zero_grad()
                    logits = scale * raw_scores + shift
                    bce = F.binary_cross_entropy_with_logits(logits, y_bin, weight=cls_w)
                    reg_term = 1e-3 * (scale - 2.0).pow(2)
                    (bce + reg_term).backward()
                    opt.step()

            s_val = scale.detach().clamp(min=0.2, max=6.0)
            b_shift = shift.detach()
            w_std = theta[:-1] * s_val
            b_std = theta[-1] * s_val + b_shift

            # Fold feature standardization back into w_l, b_l for zero-overhead inference
            w_orig = w_std / std
            b_orig = b_std - torch.dot(w_orig, mu)

            mean_diff = torch.linalg.vector_norm(x_unsafe.mean(dim=0) - x_safe.mean(dim=0))
            if float(mean_diff.item()) < 1e-6 and float(torch.linalg.vector_norm(theta[:-1]).item()) < 1e-4:
                w_orig = torch.zeros_like(w_orig)
                b_orig = torch.tensor(-5.0, device=x_safe.device, dtype=torch.float32)

            self.probe_weights[idx] = w_orig.detach()
            self.probe_biases[idx] = b_orig.detach()

            with torch.no_grad():
                probs = torch.sigmoid(x_all @ w_orig + b_orig)
                safe_p = float(probs[:n0].mean().item())
                unsafe_p = float(probs[n0:].mean().item())
                separation_scores[idx] = unsafe_p - safe_p

        self.explicitly_calibrated = True
        self._auto_update_baseline = False
        return separation_scores

    def _collect_hiddens_batched(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> List[Tuple[str, torch.Tensor]]:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                with self.hook_manager.capture(model, detach=True, training_only=False) as mgr:
                    _ = _forward_model(model, inputs)
                    return mgr.get_named_hiddens()
        finally:
            if was_training:
                model.train()

    def calibrate(
        self,
        model: nn.Module,
        safe_inputs: Any,
        unsafe_inputs: Optional[Any] = None,
        *,
        reg: float = 1.0,
    ) -> Dict[int, float]:
        """Extract all hidden layers on ``model`` and calibrate OOD + optional linear probes."""
        safe_cap = self._collect_hiddens_batched(model, safe_inputs)
        safe_dict = {name: t for name, t in safe_cap}
        if unsafe_inputs is not None:
            unsafe_cap = self._collect_hiddens_batched(model, unsafe_inputs)
            unsafe_dict = {name: t for name, t in unsafe_cap}
            return self.fit_probes(safe_dict, unsafe_dict, reg=reg, calibrate_ood=True)
        self.fit_ood_baseline(safe_dict, reset=True)
        return {}

    @torch.no_grad()
    def evaluate_hiddens(
        self,
        hiddens: Union[Sequence[torch.Tensor], Sequence[Tuple[str, torch.Tensor]]],
        *,
        update_baseline: bool = False,
        clean_mask: Optional[torch.Tensor] = None,
    ) -> LatentLayerTelemetry:
        """Score all layers simultaneously via linear safety probes & Mahalanobis OOD distance."""
        if not hiddens:
            telem = LatentLayerTelemetry(
                per_layer_risk=[],
                max_layer_idx=0,
                max_ood_zscore=0.0,
                hidden_defection_detected=False,
                quarantined=False,
                reason="no_hidden_layers",
            )
            self.last_telemetry = telem
            return telem

        named_list: List[Tuple[str, torch.Tensor]] = []
        for idx, item in enumerate(hiddens):
            if isinstance(item, tuple) and len(item) == 2:
                named_list.append((str(item[0]), item[1]))
            else:
                named_list.append((self.layer_name_map.get(idx, f"layer_{idx}"), item))

        per_layer_risk: List[float] = []
        per_layer_ood_z: List[float] = []
        per_layer_dm: List[float] = []
        layer_names: List[str] = []

        sample_risk_acc: Optional[torch.Tensor] = None
        sample_defected_acc: Optional[torch.Tensor] = None
        batch_size = 0
        dev = named_list[0][1].device
        ood_ready = self.explicitly_calibrated or (self.clean_batches_seen >= self.warmup_steps)

        for idx, (lname, h) in enumerate(named_list):
            self.layer_name_map[idx] = lname
            layer_names.append(lname)
            feats = _pool_hidden_features(h)
            b = feats.shape[0]
            if batch_size == 0:
                batch_size = b
                dev = feats.device
                sample_risk_acc = torch.zeros(b, device=dev, dtype=torch.float32)
                sample_defected_acc = torch.zeros(b, device=dev, dtype=torch.bool)

            if not bool(torch.isfinite(feats).all()):
                per_layer_risk.append(1.0)
                per_layer_ood_z.append(float("inf"))
                per_layer_dm.append(float("inf"))
                sample_risk_acc = torch.ones(batch_size, device=dev, dtype=torch.float32)
                sample_defected_acc = torch.ones(batch_size, device=dev, dtype=torch.bool)
                continue

            probe_p = torch.zeros(b, device=feats.device, dtype=torch.float32)
            has_probe = idx in self.probe_weights and self.probe_weights[idx].numel() == feats.shape[1]
            if has_probe:
                w = self.probe_weights[idx].to(device=feats.device, dtype=torch.float32)
                bias = self.probe_biases[idx].to(device=feats.device, dtype=torch.float32)
                probe_p = torch.sigmoid(feats @ w + bias)

            stats = self.ood_stats.get(idx)
            if stats is not None and ood_ready and stats.count >= 2 and (stats.raw_dim == feats.shape[1] or stats.cov_dim == feats.shape[1]):
                dm, z_ood = stats.mahalanobis_and_zscore(feats)
                ood_p = torch.sigmoid((z_ood - self.ood_zscore_threshold) * 1.25)
            else:
                dm = torch.zeros(b, device=feats.device, dtype=torch.float32)
                z_ood = torch.zeros(b, device=feats.device, dtype=torch.float32)
                ood_p = torch.zeros(b, device=feats.device, dtype=torch.float32)

            layer_risk = torch.maximum(probe_p, ood_p)
            layer_defected = (probe_p >= self.probe_threshold) | (z_ood > self.ood_zscore_threshold)

            if b == batch_size:
                assert sample_risk_acc is not None and sample_defected_acc is not None
                sample_risk_acc = torch.maximum(sample_risk_acc, layer_risk.to(dev))
                sample_defected_acc = sample_defected_acc | layer_defected.to(dev)

            per_layer_risk.append(float(layer_risk.max().item()) if b > 0 else 0.0)
            per_layer_ood_z.append(float(z_ood.max().item()) if b > 0 else 0.0)
            per_layer_dm.append(float(dm.max().item()) if b > 0 else 0.0)

        assert sample_risk_acc is not None and sample_defected_acc is not None
        hidden_defection = bool(sample_defected_acc.any().item()) if batch_size > 0 else False
        bad_frac = float(sample_defected_acc.float().mean().item()) if batch_size > 0 else 0.0

        quarantined = False
        if hidden_defection:
            if batch_size == 1 or self.fail_closed_on_any_defection:
                quarantined = True
            elif bad_frac > self.max_quarantine_batch_frac:
                quarantined = True
            elif float(sample_risk_acc.mean().item()) >= self.probe_threshold:
                quarantined = True

        # Update OOD baseline during warmup or explicit clean baseline updates
        should_update = update_baseline and (not quarantined) and (
            self._auto_update_baseline or not self.explicitly_calibrated
        )
        if should_update:
            valid_mask = ~sample_defected_acc
            if clean_mask is not None and clean_mask.shape[0] == batch_size:
                valid_mask = valid_mask & (clean_mask.to(device=dev) >= 0.5)
            if bool(valid_mask.any()):
                for idx, (lname, h) in enumerate(named_list):
                    feats = _pool_hidden_features(h).to(dev)
                    if feats.shape[0] == batch_size:
                        feats_clean = feats[valid_mask]
                    else:
                        feats_clean = feats
                    if feats_clean.shape[0] > 0 and bool(torch.isfinite(feats_clean).all()):
                        stats = self.ood_stats.setdefault(
                            idx, _LayerOODStats(max_cov_dim=self.max_cov_dim, seed=2026 + idx)
                        )
                        stats.update(feats_clean)
                self.clean_batches_seen += 1

        if quarantined:
            self.quarantined_steps += 1

        # Select layer maximizing normalized risk/OOD score
        combined_layer_scores = [
            max(r, z / max(1e-6, self.ood_zscore_threshold))
            for r, z in zip(per_layer_risk, per_layer_ood_z)
        ]
        max_layer_idx = int(max(range(len(combined_layer_scores)), key=lambda i: combined_layer_scores[i]))
        max_risk = max(per_layer_risk) if per_layer_risk else 0.0
        max_ood_z = max(per_layer_ood_z) if per_layer_ood_z else 0.0
        sample_weights = torch.where(sample_defected_acc, 0.0, 1.0).to(dtype=torch.float32)

        reason = "clean"
        if hidden_defection:
            reason = f"hidden_defection:layer={layer_names[max_layer_idx]}:risk={max_risk:.2f}:ood_z={max_ood_z:.2f}"

        telem = LatentLayerTelemetry(
            per_layer_risk=per_layer_risk,
            max_layer_idx=max_layer_idx,
            max_ood_zscore=max_ood_z,
            hidden_defection_detected=hidden_defection,
            quarantined=quarantined,
            per_layer_ood_zscore=per_layer_ood_z,
            per_layer_mahalanobis=per_layer_dm,
            layer_names=layer_names,
            max_risk=max_risk,
            sample_risks=sample_risk_acc,
            sample_weights=sample_weights,
            reason=reason,
        )
        self.last_telemetry = telem
        return telem

    def inspect_all_layers(
        self,
        model: nn.Module,
        inputs: Any,
        *,
        update_baseline: bool = False,
    ) -> LatentLayerTelemetry:
        """Run forward pass on ``model`` and audit all hidden layers simultaneously."""
        captured = self._collect_hiddens_batched(model, inputs)
        return self.evaluate_hiddens(captured, update_baseline=update_baseline)

    def attach(
        self,
        model: nn.Module,
        *,
        training_only: bool = True,
        on_telemetry: Optional[Callable[[LatentLayerTelemetry], None]] = None,
    ) -> List[str]:
        """Attach non-invasive forward hooks to ``model`` for live training telemetry."""
        self.hook_manager.training_only = bool(training_only)
        self.hook_manager.detach = True
        self._on_telemetry_cb = on_telemetry

        def _on_complete(captures: List[Tuple[str, torch.Tensor]]) -> None:
            telem = self.evaluate_hiddens(captures, update_baseline=self._auto_update_baseline)
            if self._on_telemetry_cb is not None:
                self._on_telemetry_cb(telem)

        names = self.hook_manager.attach(model, on_forward_complete=_on_complete)
        for idx, lname in enumerate(names):
            self.layer_name_map[idx] = lname
        return names

    def remove_hooks(self) -> None:
        self.hook_manager.remove_hooks()
        self._on_telemetry_cb = None


# ---------------------------------------------------------------------------
# 3. CircuitBreakerReroutingLoss (Zou et al. 2024 Representation Rerouting)
# ---------------------------------------------------------------------------

def _freeze_ref_copy(model: nn.Module) -> nn.Module:
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


class CircuitBreakerReroutingLoss(nn.Module):
    """Zou et al. (2024) Representation Rerouting loss: L_CB = L_RR(D_harm) + alpha_ret * L_ret(D_ret)."""

    def __init__(
        self,
        ref_model: Optional[nn.Module] = None,
        *,
        alpha_retain: float = 1.0,
        beta_kl: float = 0.5,
        rr_weight: float = 1.0,
        layer_range: Tuple[float, float] = (0.25, 0.75),
        target_layers: Optional[Sequence[Union[int, str]]] = None,
        include_blocks: bool = True,
        include_leaves: bool = True,
        exclude_output_head: bool = True,
        output_preserve_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.alpha_retain = float(alpha_retain)
        self.beta_kl = float(beta_kl)
        self.rr_weight = float(rr_weight)
        self.layer_range = (float(layer_range[0]), float(layer_range[1]))
        self.target_layers = list(target_layers) if target_layers is not None else None
        self.output_preserve_mode = output_preserve_mode

        self.ref_model: Optional[nn.Module] = None
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model = ref_model

        self.ref_hooks = AllLayerHiddenHookManager(
            detach=True,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
        )
        self.train_hooks = AllLayerHiddenHookManager(
            detach=False,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
        )

    def _ensure_ref_model(self, model: nn.Module, ref_model: Optional[nn.Module]) -> nn.Module:
        if ref_model is not None:
            ref_model.eval()
            return ref_model
        if self.ref_model is None:
            self.ref_model = _freeze_ref_copy(model)
        return self.ref_model

    def _compute_output_preserve(self, out_theta: Any, out_ref: Any) -> torch.Tensor:
        z_t, _ = _extract_primary_tensor(out_theta)
        z_0, _ = _extract_primary_tensor(out_ref)
        if z_t is None or z_0 is None:
            return torch.zeros((), dtype=torch.float32)
        z_t_f = z_t.float()
        z_0_f = z_0.detach().float().to(device=z_t_f.device)
        mode = self.output_preserve_mode
        if mode == "mse" or (mode == "auto" and z_t_f.shape[-1] <= 1):
            return F.mse_loss(z_t_f, z_0_f)
        if mode == "kl" or (mode == "auto" and z_t_f.ndim >= 2 and z_t_f.shape[-1] >= 2):
            log_p_t = F.log_softmax(z_t_f, dim=-1)
            log_p_0 = F.log_softmax(z_0_f, dim=-1)
            p_0 = log_p_0.exp()
            return (p_0 * (log_p_0 - log_p_t)).sum(dim=-1).mean()
        return F.mse_loss(z_t_f, z_0_f)

    def forward(
        self,
        model: nn.Module,
        harm_inputs: Any,
        retain_inputs: Optional[Any] = None,
        *,
        ref_model: Optional[nn.Module] = None,
        harm_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        ref = self._ensure_ref_model(model, ref_model)
        dev = _infer_device(model, harm_inputs)

        with torch.no_grad():
            with self.ref_hooks.capture(ref, detach=True, training_only=False):
                _ = _forward_model(ref, harm_inputs)
                h0_harm = self.ref_hooks.get_ordered_hiddens()

        with self.train_hooks.capture(model, detach=False, training_only=False):
            _ = _forward_model(model, harm_inputs)
            ht_harm = self.train_hooks.get_ordered_hiddens()

        n_common = min(len(h0_harm), len(ht_harm))
        if n_common == 0:
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            return (zero, {"rr_loss": 0.0, "retain_loss": 0.0, "kl_loss": 0.0, "mean_harm_cos": 0.0}) if return_details else zero

        target_idxs = self.train_hooks.select_layer_indices(
            n_common, layer_range=self.layer_range, target_layers=self.target_layers
        )

        rr_terms: List[torch.Tensor] = []
        cos_terms: List[torch.Tensor] = []
        for idx in target_idxs:
            h_t = ht_harm[idx].float()
            h_0 = h0_harm[idx].detach().float().to(device=h_t.device)
            if h_t.shape != h_0.shape:
                continue
            if h_t.ndim == 4:
                ht_flat = h_t.permute(0, 2, 3, 1).reshape(-1, h_t.shape[1])
                h0_flat = h_0.permute(0, 2, 3, 1).reshape(-1, h_0.shape[1])
                cos = F.cosine_similarity(ht_flat, h0_flat, dim=-1, eps=1e-8)
            else:
                cos = F.cosine_similarity(h_t, h_0, dim=-1, eps=1e-8)
                if harm_mask is not None and cos.shape[: harm_mask.ndim] == harm_mask.shape:
                    m = harm_mask.to(device=cos.device, dtype=cos.dtype)
                    rr_l = (F.relu(cos) * m).sum() / m.sum().clamp(min=1.0)
                    mean_c = (cos * m).sum() / m.sum().clamp(min=1.0)
                    rr_terms.append(rr_l)
                    cos_terms.append(mean_c)
                    continue
            rr_terms.append(F.relu(cos).mean())
            cos_terms.append(cos.mean())

        loss_rr = torch.stack(rr_terms).mean() if rr_terms else torch.zeros((), device=dev, dtype=torch.float32)

        loss_ret_hid = torch.zeros((), device=dev, dtype=torch.float32)
        loss_kl = torch.zeros((), device=dev, dtype=torch.float32)
        if retain_inputs is not None:
            with torch.no_grad():
                with self.ref_hooks.capture(ref, detach=True, training_only=False):
                    out_0_ret = _forward_model(ref, retain_inputs)
                    h0_ret = self.ref_hooks.get_ordered_hiddens()

            with self.train_hooks.capture(model, detach=False, training_only=False):
                out_t_ret = _forward_model(model, retain_inputs)
                ht_ret = self.train_hooks.get_ordered_hiddens()

            n_ret = min(len(h0_ret), len(ht_ret))
            ret_idxs = [i for i in target_idxs if i < n_ret] or list(range(n_ret))
            ret_terms: List[torch.Tensor] = []
            for idx in ret_idxs:
                h_t = ht_ret[idx].float()
                h_0 = h0_ret[idx].detach().float().to(device=h_t.device)
                if h_t.shape == h_0.shape:
                    ret_terms.append(F.mse_loss(h_t, h_0))
            if ret_terms:
                loss_ret_hid = torch.stack(ret_terms).mean()
            loss_kl = self._compute_output_preserve(out_t_ret, out_0_ret).to(device=dev)

        total = self.rr_weight * loss_rr + self.alpha_retain * (loss_ret_hid + self.beta_kl * loss_kl)
        if return_details:
            mean_cos = float(torch.stack(cos_terms).mean().detach().item()) if cos_terms else 0.0
            return total, {
                "rr_loss": float(loss_rr.detach().item()),
                "retain_loss": float(loss_ret_hid.detach().item()),
                "kl_loss": float(loss_kl.detach().item()),
                "mean_harm_cos": mean_cos,
                "total": float(total.detach().item()),
            }
        return total


# ---------------------------------------------------------------------------
# 4. RMUMisdirectionLoss (Li et al. 2024 WMDP)
# ---------------------------------------------------------------------------

class RMUMisdirectionLoss(nn.Module):
    """Li et al. (2024) WMDP Representation Misdirection for Unlearning."""

    def __init__(
        self,
        ref_model: Optional[nn.Module] = None,
        *,
        alpha: float = 1.0,
        forget_weight: float = 1.0,
        steering_scale: float = 1.0,
        layer_range: Tuple[float, float] = (0.25, 0.75),
        target_layers: Optional[Sequence[Union[int, str]]] = None,
        seed: int = 2026,
        include_blocks: bool = True,
        include_leaves: bool = True,
        exclude_output_head: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.forget_weight = float(forget_weight)
        self.steering_scale = float(steering_scale)
        self.layer_range = (float(layer_range[0]), float(layer_range[1]))
        self.target_layers = list(target_layers) if target_layers is not None else None
        self.seed = int(seed)

        self.ref_model: Optional[nn.Module] = None
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model = ref_model

        self._unit_dirs: Dict[Tuple[int, int], torch.Tensor] = {}
        self.ref_hooks = AllLayerHiddenHookManager(
            detach=True,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
        )
        self.train_hooks = AllLayerHiddenHookManager(
            detach=False,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
        )

    def _get_unit_direction(self, layer_idx: int, dim: int, device: torch.device) -> torch.Tensor:
        key = (layer_idx, dim)
        u = self._unit_dirs.get(key)
        if u is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed + 1009 * layer_idx + dim)
            vec = torch.randn(dim, dtype=torch.float32, generator=gen)
            vec = vec / torch.linalg.vector_norm(vec).clamp(min=1e-8)
            u = vec.to(device=device)
            self._unit_dirs[key] = u
        elif u.device != device:
            u = u.to(device=device)
            self._unit_dirs[key] = u
        return u

    def _ensure_ref_model(self, model: nn.Module, ref_model: Optional[nn.Module]) -> nn.Module:
        if ref_model is not None:
            ref_model.eval()
            return ref_model
        if self.ref_model is None:
            self.ref_model = _freeze_ref_copy(model)
        return self.ref_model

    def forward(
        self,
        model: nn.Module,
        forget_inputs: Any,
        retain_inputs: Optional[Any] = None,
        *,
        ref_model: Optional[nn.Module] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        ref = self._ensure_ref_model(model, ref_model)
        dev = _infer_device(model, forget_inputs)

        with torch.no_grad():
            with self.ref_hooks.capture(ref, detach=True, training_only=False):
                _ = _forward_model(ref, forget_inputs)
                h0_forget = self.ref_hooks.get_ordered_hiddens()

        with self.train_hooks.capture(model, detach=False, training_only=False):
            _ = _forward_model(model, forget_inputs)
            ht_forget = self.train_hooks.get_ordered_hiddens()

        n_common = min(len(h0_forget), len(ht_forget))
        if n_common == 0:
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            return (zero, {"forget_loss": 0.0, "retain_loss": 0.0, "total": 0.0}) if return_details else zero

        target_idxs = self.train_hooks.select_layer_indices(
            n_common, layer_range=self.layer_range, target_layers=self.target_layers
        )

        forget_terms: List[torch.Tensor] = []
        for idx in target_idxs:
            h_t = ht_forget[idx].float()
            h_0 = h0_forget[idx].detach().float().to(device=h_t.device)
            if h_t.shape != h_0.shape:
                continue
            feat_dim = h_t.shape[1] if h_t.ndim == 4 else h_t.shape[-1]
            u_l = self._get_unit_direction(idx, feat_dim, h_t.device)
            if h_t.ndim == 4:
                c_l = self.steering_scale * torch.linalg.vector_norm(h_0, dim=1).mean()
                target = (c_l * u_l).view(1, feat_dim, 1, 1).expand_as(h_t)
            else:
                c_l = self.steering_scale * torch.linalg.vector_norm(h_0, dim=-1).mean()
                view_shape = (1,) * (h_t.ndim - 1) + (feat_dim,)
                target = (c_l * u_l).view(view_shape).expand_as(h_t)
            forget_terms.append(F.mse_loss(h_t, target.detach()))

        loss_forget = torch.stack(forget_terms).mean() if forget_terms else torch.zeros((), device=dev, dtype=torch.float32)

        loss_retain = torch.zeros((), device=dev, dtype=torch.float32)
        if retain_inputs is not None:
            with torch.no_grad():
                with self.ref_hooks.capture(ref, detach=True, training_only=False):
                    _ = _forward_model(ref, retain_inputs)
                    h0_ret = self.ref_hooks.get_ordered_hiddens()

            with self.train_hooks.capture(model, detach=False, training_only=False):
                _ = _forward_model(model, retain_inputs)
                ht_ret = self.train_hooks.get_ordered_hiddens()

            n_ret = min(len(h0_ret), len(ht_ret))
            ret_idxs = [i for i in target_idxs if i < n_ret] or list(range(n_ret))
            ret_terms: List[torch.Tensor] = []
            for idx in ret_idxs:
                h_t = ht_ret[idx].float()
                h_0 = h0_ret[idx].detach().float().to(device=h_t.device)
                if h_t.shape == h_0.shape:
                    ret_terms.append(F.mse_loss(h_t, h_0))
            if ret_terms:
                loss_retain = torch.stack(ret_terms).mean()

        total = self.forget_weight * loss_forget + self.alpha * loss_retain
        if return_details:
            return total, {
                "forget_loss": float(loss_forget.detach().item()),
                "retain_loss": float(loss_retain.detach().item()),
                "total": float(total.detach().item()),
            }
        return total


# ---------------------------------------------------------------------------
# 5. TVaccineLayerGuard & AllLayerLatentAdversarialTrainer
# ---------------------------------------------------------------------------

class TVaccineLayerGuard:
    """T-Vaccine / Vaccine minimax & Latent Adversarial Training guard across hidden layers."""

    def __init__(
        self,
        *,
        rho: float = 0.5,
        top_k: Optional[int] = None,
        num_inner_steps: int = 1,
        clean_weight: float = 0.0,
        anchor_weight: float = 0.0,
        per_sample_norm: bool = True,
        include_blocks: bool = True,
        include_leaves: bool = True,
        exclude_output_head: bool = True,
        target_layers: Optional[Sequence[Union[int, str]]] = None,
    ) -> None:
        self.rho = float(rho)
        self.top_k = top_k
        self.num_inner_steps = max(1, int(num_inner_steps))
        self.clean_weight = float(clean_weight)
        self.anchor_weight = float(anchor_weight)
        self.per_sample_norm = bool(per_sample_norm)

        self.hook_manager = AllLayerHiddenHookManager(
            detach=False,
            include_blocks=include_blocks,
            include_leaves=include_leaves,
            exclude_output_head=exclude_output_head,
            target_layers=target_layers,
        )
        self.critical_layer_indices: Optional[List[int]] = None
        self.critical_layer_names: Optional[List[str]] = None
        self.layer_grad_importance: Dict[str, float] = {}
        self._anchor_params: Dict[int, torch.Tensor] = {}

    def snapshot_anchor(self, model: nn.Module) -> None:
        """Snapshot trainable initial weights ``theta_0`` for optional ``||theta - theta_0||_F^2`` anchor."""
        self._anchor_params = {
            id(p): p.detach().clone()
            for p in model.parameters()
            if p.requires_grad and p.is_floating_point()
        }

    def select_critical_layers(
        self,
        model: nn.Module,
        safe_batches: Iterable[Any],
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        top_k: Optional[int] = None,
        max_batches: int = 8,
    ) -> List[int]:
        """Select Top-K safety-critical layers ``S = Top-K_l(E[||grad_h^(l) L_safe||_2])``."""
        eval_loss = loss_fn or _default_task_loss
        grad_accum: Dict[str, torch.Tensor] = {}
        ordered_names: List[str] = []
        n_batches = 0

        was_training = model.training
        model.eval()
        try:
            for b_idx, batch in enumerate(safe_batches):
                if b_idx >= max_batches:
                    break
                delta_entries: List[Tuple[str, torch.Tensor]] = []

                def _zero_delta_factory(lname: str, t: torch.Tensor) -> torch.Tensor:
                    d = torch.zeros_like(t, requires_grad=True)
                    delta_entries.append((lname, d))
                    return t + d

                with self.hook_manager.capture(model, detach=True, training_only=False) as mgr:
                    mgr.set_perturbation_factory(_zero_delta_factory)
                    with torch.enable_grad():
                        loss = eval_loss(model, batch)
                        vars_list = [d for _, d in delta_entries]
                        if vars_list:
                            grads = torch.autograd.grad(
                                loss, vars_list, retain_graph=False, allow_unused=True
                            )
                            for (lname, _), g in zip(delta_entries, grads):
                                if lname not in ordered_names:
                                    ordered_names.append(lname)
                                if g is None:
                                    continue
                                g_norm = torch.linalg.vector_norm(g.detach().float())
                                prev = grad_accum.get(lname)
                                grad_accum[lname] = g_norm if prev is None else (prev + g_norm)
                n_batches += 1
        finally:
            if was_training:
                model.train()

        if not ordered_names:
            self.critical_layer_indices = []
            self.critical_layer_names = []
            self.layer_grad_importance = {}
            return []

        denom = float(max(1, n_batches))
        scores = {
            name: float((grad_accum[name] / denom).item()) if name in grad_accum else 0.0
            for name in ordered_names
        }
        self.layer_grad_importance = scores

        k_eff = top_k if top_k is not None else self.top_k
        if k_eff is None:
            k_eff = max(1, min(len(ordered_names), max(2, (len(ordered_names) + 1) // 2)))
        k_eff = max(1, min(int(k_eff), len(ordered_names)))

        ranked = sorted(range(len(ordered_names)), key=lambda i: scores[ordered_names[i]], reverse=True)
        selected_idxs = sorted(ranked[:k_eff])
        self.critical_layer_indices = selected_idxs
        self.critical_layer_names = [ordered_names[i] for i in selected_idxs]
        return selected_idxs

    def _active_layer_set(self, model: nn.Module) -> Set[str]:
        if self.critical_layer_names:
            return set(self.critical_layer_names)
        discovered = [name for name, _ in self.hook_manager.discover_layers(model)]
        return set(discovered)

    def compute_perturbations(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute bounded worst-case hidden perturbations ``delta_l^*`` without mutating ``param.grad``."""
        eval_loss = loss_fn or _default_task_loss
        active_names = self._active_layer_set(model)
        if not active_names or self.rho <= 0.0:
            return {}

        was_training = model.training
        model.eval()
        deltas: Dict[str, torch.Tensor] = {}
        try:
            # Probe shapes on first pass and compute gradient
            step_sz = self.rho if self.num_inner_steps == 1 else (1.5 * self.rho / float(self.num_inner_steps))
            for step_i in range(self.num_inner_steps):
                step_vars: Dict[str, torch.Tensor] = {}
                step_order: List[str] = []

                def _inject_or_init(lname: str, t: torch.Tensor) -> torch.Tensor:
                    if lname not in active_names:
                        return t
                    if lname in deltas and deltas[lname].shape == t.shape:
                        d_var = deltas[lname].detach().clone().to(device=t.device, dtype=t.dtype).requires_grad_(True)
                    else:
                        d_var = torch.zeros_like(t, requires_grad=True)
                    step_vars[lname] = d_var
                    step_order.append(lname)
                    return t + d_var

                with self.hook_manager.capture(model, detach=True, training_only=False) as mgr:
                    mgr.set_perturbation_factory(_inject_or_init)
                    with torch.enable_grad():
                        loss = eval_loss(model, batch)
                        vars_list = [step_vars[n] for n in step_order]
                        if not vars_list:
                            break
                        grads = torch.autograd.grad(
                            loss, vars_list, retain_graph=False, allow_unused=True
                        )

                with torch.no_grad():
                    for lname, d_var, g in zip(step_order, vars_list, grads):
                        if g is None:
                            deltas[lname] = d_var.detach()
                            continue
                        g_norm = _norm_per_sample_or_global(g, per_sample=self.per_sample_norm, eps=1e-8)
                        unit_g = g.float() / g_norm
                        if self.num_inner_steps == 1:
                            new_d = self.rho * unit_g
                        else:
                            cand = d_var.detach().float() + step_sz * unit_g
                            c_norm = _norm_per_sample_or_global(cand, per_sample=self.per_sample_norm, eps=1e-8)
                            scale = torch.clamp(self.rho / c_norm, max=1.0)
                            new_d = cand * scale
                        deltas[lname] = new_d.to(dtype=d_var.dtype)
        finally:
            if was_training:
                model.train()

        return {k: v.detach() for k, v in deltas.items()}

    def anchor_penalty(self, model: nn.Module) -> torch.Tensor:
        dev = _infer_device(model)
        if self.anchor_weight <= 0.0 or not self._anchor_params:
            return torch.zeros((), device=dev, dtype=torch.float32)
        terms: List[torch.Tensor] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p0 = self._anchor_params.get(id(p))
            if p0 is not None and p0.shape == p.shape:
                terms.append((p.float() - p0.to(device=p.device, dtype=torch.float32)).pow(2).sum())
        if not terms:
            return torch.zeros((), device=dev, dtype=torch.float32)
        return 0.5 * self.anchor_weight * torch.stack(terms).sum()

    def adversarial_loss(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        *,
        sample_weights: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute outer worst-case loss under bounded hidden perturbations + optional clean/anchor terms."""
        eval_loss = loss_fn or _default_task_loss
        deltas = self.compute_perturbations(model, batch, loss_fn=eval_loss)

        with self.hook_manager.capture(model, detach=True, training_only=False) as mgr:
            mgr.set_perturbations(deltas)
            loss_adv = eval_loss(model, batch)

        if sample_weights is not None and loss_adv.ndim >= 1 and loss_adv.shape[0] == sample_weights.shape[0]:
            w = (sample_weights >= 0.5).to(device=loss_adv.device, dtype=loss_adv.dtype)
            loss_adv = (loss_adv * w).sum() / w.sum().clamp(min=1.0)

        loss_clean = torch.zeros_like(loss_adv)
        if self.clean_weight > 0.0:
            loss_clean = eval_loss(model, batch)

        loss_anchor = self.anchor_penalty(model).to(device=loss_adv.device, dtype=loss_adv.dtype)
        total = loss_adv + self.clean_weight * loss_clean + loss_anchor

        if return_details:
            return total, {
                "loss_adv": float(loss_adv.detach().item()),
                "loss_clean": float(loss_clean.detach().item()),
                "loss_anchor": float(loss_anchor.detach().item()),
                "num_perturbed_layers": float(len(deltas)),
                "total": float(total.detach().item()),
            }
        return total


class AllLayerLatentAdversarialTrainer:
    """High-level trainer combining T-Vaccine hidden perturbations + Circuit Breaker / RMU."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        rho: float = 0.5,
        top_k: Optional[int] = None,
        num_inner_steps: int = 1,
        clean_weight: float = 0.25,
        anchor_weight: float = 0.01,
        cb_loss: Optional[CircuitBreakerReroutingLoss] = None,
        rmu_loss: Optional[RMUMisdirectionLoss] = None,
        monitor: Optional[AllLayerLatentMonitor] = None,
        lr: float = 1e-3,
    ) -> None:
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=lr)

        self.guard = TVaccineLayerGuard(
            rho=rho,
            top_k=top_k,
            num_inner_steps=num_inner_steps,
            clean_weight=clean_weight,
            anchor_weight=anchor_weight,
        )
        self.cb_loss = cb_loss
        self.rmu_loss = rmu_loss
        self.monitor = monitor
        self.step_count: int = 0

    def calibrate(
        self,
        safe_batches: Iterable[Any],
        *,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """Select Top-K critical layers, snapshot parameter anchor, and optionally warm up monitor."""
        batches_list = list(safe_batches)
        self.guard.snapshot_anchor(self.model)
        selected = self.guard.select_critical_layers(self.model, batches_list, loss_fn=loss_fn, top_k=top_k)
        if self.monitor is not None and batches_list:
            first_in = batches_list[0][0] if isinstance(batches_list[0], (list, tuple)) else batches_list[0]
            self.monitor.calibrate(self.model, first_in)
        return selected

    def train_step(
        self,
        batch: Any,
        *,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        harmful_inputs: Optional[Any] = None,
        retain_inputs: Optional[Any] = None,
        cb_weight: float = 0.5,
        rmu_weight: float = 0.5,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        loss, details = self.guard.adversarial_loss(
            self.model, batch, loss_fn=loss_fn, return_details=True
        )
        if self.cb_loss is not None and harmful_inputs is not None:
            cb_val = self.cb_loss(self.model, harmful_inputs, retain_inputs=retain_inputs)
            loss = loss + cb_weight * cb_val.to(device=loss.device, dtype=loss.dtype)
            details["cb_loss"] = float(cb_val.detach().item())
        if self.rmu_loss is not None and harmful_inputs is not None:
            rmu_val = self.rmu_loss(self.model, harmful_inputs, retain_inputs=retain_inputs)
            loss = loss + rmu_weight * rmu_val.to(device=loss.device, dtype=loss.dtype)
            details["rmu_loss"] = float(rmu_val.detach().item())

        loss.backward()
        if hasattr(self.optimizer, "commit_microbatch"):
            self.optimizer.commit_microbatch()
        self.optimizer.step()
        self.step_count += 1
        details["total_step_loss"] = float(loss.detach().item())
        return details

    def train_epoch(
        self,
        dataloader: Iterable[Any],
        *,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        harmful_inputs: Optional[Any] = None,
        retain_inputs: Optional[Any] = None,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        n = 0
        for batch in dataloader:
            stats = self.train_step(
                batch,
                loss_fn=loss_fn,
                harmful_inputs=harmful_inputs,
                retain_inputs=retain_inputs,
            )
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        return {k: v / float(max(1, n)) for k, v in totals.items()}


__all__ = [
    "AllLayerHiddenHookManager",
    "LatentLayerTelemetry",
    "AllLayerLatentMonitor",
    "CircuitBreakerReroutingLoss",
    "RMUMisdirectionLoss",
    "TVaccineLayerGuard",
    "AllLayerLatentAdversarialTrainer",
]
