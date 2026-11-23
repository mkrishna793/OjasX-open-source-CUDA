"""Generalized OjasX-GRAM modular gradient router with residual decorrelation.

Fixes shared-residual and attention leakage in modular pretraining/fine-tuning:
- Wraps ANY pair of submodules (`core` + named `aux` partitions across Attention
  `v_proj`/`o_proj`, MLP `gate`/`up`/`down`, CNNs, SSMs, or custom blocks).
- Enforces exact zero gradients on non-target partitions via
  `torch.func.functional_call` with detached parameter dictionaries.
- Computes Residual Decorrelation Orthogonality Loss ``||Cov(h_core, h_aux)||_F^2 / D``
  minimizing latent cross-partition leakage into shared residual streams.
- Supports clean deploy-time ``ablate(aux_name, active=False)`` masking.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.math.lattice import E8LatticeLinear
from ojasx_safety.contracts import SafetyPolicy


def _functional_call_detached(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Invoke ``module`` with all parameters detached so gradients flow through inputs only."""
    params_and_bufs = {
        **{n: p.detach() for n, p in module.named_parameters()},
        **{n: b for n, b in module.named_buffers()},
    }
    return torch.func.functional_call(module, params_and_bufs, args, kwargs)


def _extract_primary_tensor(out: Any) -> Tuple[Optional[torch.Tensor], bool]:
    if isinstance(out, torch.Tensor):
        return out, False
    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0], True
    return None, False


def _flatten_features_for_cov(h: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, ..., D]`` or CNN ``[B, C, H, W]`` into ``[N, D_feat]`` in float32."""
    x = h.float()
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    if x.ndim == 4:
        # Assume NCHW feature maps -> [N*H*W, C]
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return x.reshape(-1, x.shape[-1])


def residual_decorrelation_loss(h_core: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
    """Compute ``||Cov(h_core, h_aux)||_F^2 / D`` minimizing shared-residual leakage."""
    hc = _flatten_features_for_cov(h_core)
    ha = _flatten_features_for_cov(h_aux)
    n = min(hc.shape[0], ha.shape[0])
    d = min(hc.shape[1], ha.shape[1])
    if n == 0 or d == 0:
        return torch.zeros((), device=h_core.device, dtype=torch.float32)
    hc = hc[:n, :d]
    ha = ha[:n, :d]
    if n >= 2:
        hc = hc - hc.mean(dim=0, keepdim=True)
        ha = ha - ha.mean(dim=0, keepdim=True)
    cov = (hc.t() @ ha) / float(max(1, n))
    return cov.square().sum() / float(d)


def cosine_circuit_breaker_loss(h_core: torch.Tensor, h_aux: torch.Tensor) -> torch.Tensor:
    """Layer-wise cosine circuit-breaker penalty preventing core-aux residual alignment."""
    hc = _flatten_features_for_cov(h_core)
    ha = _flatten_features_for_cov(h_aux)
    n = min(hc.shape[0], ha.shape[0])
    d = min(hc.shape[1], ha.shape[1])
    if n == 0 or d == 0:
        return torch.zeros((), device=h_core.device, dtype=torch.float32)
    cos = F.cosine_similarity(hc[:n, :d], ha[:n, :d], dim=-1, eps=1e-8)
    return 0.5 * (F.relu(cos).mean() + cos.pow(2).mean())


class GRAMLoRAAdapter(nn.Module):
    """Low-rank auxiliary residual adapter compatible with Linear / E8 / LoRA cores."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / float(self.rank)
        eff_dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32
        self.A = nn.Parameter(
            torch.randn(self.in_features, self.rank, device=device, dtype=eff_dtype) * 0.02
        )
        self.B = nn.Parameter(
            torch.zeros(self.rank, self.out_features, device=device, dtype=eff_dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.A.to(dtype=x.dtype) if self.A.dtype != x.dtype else self.A
        b = self.B.to(dtype=x.dtype) if self.B.dtype != x.dtype else self.B
        return ((x @ a) @ b) * self.scaling


class _ZeroScaledReplica(nn.Module):
    """Zero-initialized residual wrapper for arbitrary non-linear submodules (SSM/Attention blocks)."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.module(*args, **kwargs)
        primary, is_tuple = _extract_primary_tensor(out)
        if primary is None:
            return out
        scaled = primary * self.gate.to(device=primary.device, dtype=primary.dtype)
        if is_tuple:
            return (scaled,) + out[1:]
        return scaled


def _default_aux_factory(
    full_name: str,
    core_mod: nn.Module,
    aux_name: str,
    rank: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    params = list(core_mod.parameters())
    dev = params[0].device if params else None
    dt = params[0].dtype if params else torch.float32

    in_f = getattr(core_mod, "in_features", getattr(core_mod, "in_dim", None))
    out_f = getattr(core_mod, "out_features", getattr(core_mod, "out_dim", None))
    if isinstance(core_mod, (nn.Linear, E8LatticeLinear)) or (
        in_f is not None and out_f is not None and not isinstance(core_mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    ):
        return GRAMLoRAAdapter(
            in_features=int(in_f),
            out_features=int(out_f),
            rank=rank,
            alpha=alpha,
            device=dev,
            dtype=dt,
        )
    if isinstance(core_mod, nn.Conv2d):
        conv = nn.Conv2d(
            core_mod.in_channels,
            core_mod.out_channels,
            kernel_size=core_mod.kernel_size,
            stride=core_mod.stride,
            padding=core_mod.padding,
            dilation=core_mod.dilation,
            groups=core_mod.groups,
            bias=(core_mod.bias is not None),
            device=dev,
            dtype=dt if dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32,
        )
        with torch.no_grad():
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
        return conv
    if isinstance(core_mod, nn.Conv1d):
        conv = nn.Conv1d(
            core_mod.in_channels,
            core_mod.out_channels,
            kernel_size=core_mod.kernel_size,
            stride=core_mod.stride,
            padding=core_mod.padding,
            dilation=core_mod.dilation,
            groups=core_mod.groups,
            bias=(core_mod.bias is not None),
            device=dev,
            dtype=dt if dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32,
        )
        with torch.no_grad():
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
        return conv
    return _ZeroScaledReplica(copy.deepcopy(core_mod))


class GRAMPartitionedModule(nn.Module):
    """Wraps a ``core`` submodule and named ``aux`` partitions with exact grad isolation."""

    def __init__(
        self,
        core_module: nn.Module,
        aux_modules: Dict[str, nn.Module],
        mode: str = "additive",
    ) -> None:
        super().__init__()
        if mode not in ("additive", "replace"):
            raise ValueError("mode must be 'additive' or 'replace'")
        self.core = core_module
        self.aux = nn.ModuleDict(aux_modules)
        self.mode = mode

        self.forward_weights: Dict[str, float] = {"core": 1.0, **{k: 0.0 for k in aux_modules}}
        self.backward_active: Set[str] = {"core"}
        self.ablated: Set[str] = set()

        self.last_h_core: Optional[torch.Tensor] = None
        self.last_h_aux: Dict[str, torch.Tensor] = {}
        self.last_decorrelation_loss: Optional[torch.Tensor] = None
        self.last_cosine_cb_loss: Optional[torch.Tensor] = None

    @property
    def in_features(self) -> int:
        return int(getattr(self.core, "in_features"))

    @property
    def out_features(self) -> int:
        return int(getattr(self.core, "out_features"))

    def core_parameters(self) -> Iterator[nn.Parameter]:
        return self.core.parameters()

    def aux_parameters(self, aux_name: Optional[str] = None) -> Iterator[nn.Parameter]:
        if aux_name is not None:
            return self.aux[aux_name].parameters()
        return self.aux.parameters()

    def set_routing(
        self,
        forward_active: Union[Sequence[str], Dict[str, float]],
        backward_active: Sequence[str],
    ) -> None:
        if isinstance(forward_active, dict):
            self.forward_weights = {
                "core": float(forward_active.get("core", 0.0)),
                **{k: float(forward_active.get(k, 0.0)) for k in self.aux},
            }
        else:
            fwd_set = set(forward_active)
            self.forward_weights = {
                "core": 1.0 if "core" in fwd_set else 0.0,
                **{k: (1.0 if k in fwd_set else 0.0) for k in self.aux},
            }
        self.backward_active = set(backward_active)

    def ablate(self, aux_name: str, active: bool = False) -> None:
        if aux_name not in self.aux:
            return
        if active:
            self.ablated.discard(aux_name)
            self.forward_weights[aux_name] = max(1.0, self.forward_weights.get(aux_name, 0.0))
        else:
            self.ablated.add(aux_name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.last_h_core = None
        self.last_h_aux = {}
        self.last_decorrelation_loss = None
        self.last_cosine_cb_loss = None

        core_w = 0.0 if "core" in self.ablated else float(self.forward_weights.get("core", 1.0))
        h_core_raw: Any = None
        h_core_tensor: Optional[torch.Tensor] = None
        is_tuple = False

        if core_w != 0.0:
            if "core" in self.backward_active:
                h_core_raw = self.core(*args, **kwargs)
            else:
                h_core_raw = _functional_call_detached(self.core, args, kwargs)
            h_core_tensor, is_tuple = _extract_primary_tensor(h_core_raw)
            if h_core_tensor is not None:
                if core_w != 1.0:
                    h_core_tensor = h_core_tensor * core_w
                self.last_h_core = h_core_tensor

        aux_sum: Optional[torch.Tensor] = None
        decorr_terms: List[torch.Tensor] = []
        cos_cb_terms: List[torch.Tensor] = []
        active_aux_count = 0

        for name, aux_mod in self.aux.items():
            if name in self.ablated:
                continue
            w = float(self.forward_weights.get(name, 0.0))
            if w == 0.0:
                continue
            if name in self.backward_active:
                h_aux_raw = aux_mod(*args, **kwargs)
            else:
                h_aux_raw = _functional_call_detached(aux_mod, args, kwargs)
            h_aux_t, aux_is_tuple = _extract_primary_tensor(h_aux_raw)
            if h_aux_t is None:
                continue
            if w != 1.0:
                h_aux_t = h_aux_t * w
            self.last_h_aux[name] = h_aux_t
            aux_sum = h_aux_t if aux_sum is None else (aux_sum + h_aux_t)
            active_aux_count += 1
            if h_core_tensor is not None:
                decorr_terms.append(residual_decorrelation_loss(h_core_tensor, h_aux_t))
                cos_cb_terms.append(cosine_circuit_breaker_loss(h_core_tensor, h_aux_t))

        if decorr_terms:
            self.last_decorrelation_loss = torch.stack(decorr_terms).mean()
        if cos_cb_terms:
            self.last_cosine_cb_loss = torch.stack(cos_cb_terms).mean()

        if self.mode == "replace" and aux_sum is not None:
            combined = aux_sum / float(max(1, active_aux_count))
        elif h_core_tensor is not None and aux_sum is not None:
            combined = h_core_tensor + aux_sum
        elif h_core_tensor is not None:
            combined = h_core_tensor
        elif aux_sum is not None:
            combined = aux_sum
        else:
            return h_core_raw

        if is_tuple and isinstance(h_core_raw, tuple):
            return (combined,) + h_core_raw[1:]
        return combined


class GRAMRouter:
    """Model-level coordinator for Generalized OjasX-GRAM gradient routing & decorrelation."""

    def __init__(
        self,
        model: nn.Module,
        policy: Optional[SafetyPolicy] = None,
        freeze_unwrapped_on_aux: bool = True,
        enable_cosine_circuit_breaker: bool = False,
        circuit_breaker_weight: float = 0.1,
    ) -> None:
        self.model = model
        self.policy = policy or SafetyPolicy()
        self.freeze_unwrapped_on_aux = bool(freeze_unwrapped_on_aux)
        self.enable_cosine_circuit_breaker = bool(enable_cosine_circuit_breaker)
        self.circuit_breaker_weight = float(circuit_breaker_weight)
        self.partitions: Dict[str, GRAMPartitionedModule] = {}
        self.aux_names: List[str] = []
        self._saved_unwrapped_req_grad: Dict[int, Tuple[nn.Parameter, bool]] = {}
        self.last_decorrelation_loss: float = 0.0
        self.last_circuit_breaker_loss: float = 0.0
        self.active_backward_partitions: Set[str] = {"core"}
        self._core_ref_model: Optional[nn.Module] = None
        self._cb_engine: Optional[Any] = None

    def partition(
        self,
        target_modules: Optional[Sequence[str]] = (
            "v_proj",
            "o_proj",
            "gate",
            "up",
            "down",
            "c_fc",
            "c_proj",
        ),
        aux_names: Sequence[str] = ("bio_aux",),
        rank: int = 8,
        alpha: float = 16.0,
        mode: str = "additive",
        aux_factory: Optional[Callable[[str, nn.Module, str], nn.Module]] = None,
        modules: Optional[Sequence[str]] = None,
    ) -> Dict[str, GRAMPartitionedModule]:
        """Wrap matching submodules anywhere in ``self.model`` with ``GRAMPartitionedModule``."""
        patterns = tuple(modules) if modules is not None else target_modules
        self.aux_names = list(aux_names)

        def _wrap(parent: nn.Module, prefix: str) -> None:
            for child_name, child in list(parent.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, GRAMPartitionedModule):
                    self.partitions[full_name] = child
                    continue
                matched = patterns is None or any(p in full_name or p == child_name for p in patterns)
                if matched:
                    aux_dict: Dict[str, nn.Module] = {}
                    for a_name in self.aux_names:
                        if aux_factory is not None:
                            aux_dict[a_name] = aux_factory(full_name, child, a_name)
                        else:
                            aux_dict[a_name] = _default_aux_factory(
                                full_name, child, a_name, rank=rank, alpha=alpha
                            )
                    wrapped = GRAMPartitionedModule(child, aux_dict, mode=mode)
                    setattr(parent, child_name, wrapped)
                    self.partitions[full_name] = wrapped
                else:
                    _wrap(child, full_name)

        _wrap(self.model, "")
        return self.partitions

    def register_existing_partitions(self) -> Dict[str, GRAMPartitionedModule]:
        """Discover already-wrapped ``GRAMPartitionedModule`` instances inside ``self.model``."""
        aux_seen: Set[str] = set()
        for name, mod in self.model.named_modules():
            if isinstance(mod, GRAMPartitionedModule):
                self.partitions[name] = mod
                aux_seen.update(mod.aux.keys())
        self.aux_names = sorted(aux_seen)
        return self.partitions

    def core_parameters(self) -> List[nn.Parameter]:
        aux_ids = {id(p) for p in self.all_aux_parameters()}
        return [p for p in self.model.parameters() if id(p) not in aux_ids]

    def aux_parameters(self, aux_name: Optional[str] = None) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for mod in self.partitions.values():
            if aux_name is None or aux_name in mod.aux:
                params.extend(mod.aux_parameters(aux_name))
        return params

    def all_aux_parameters(self) -> List[nn.Parameter]:
        return self.aux_parameters(None)

    def unwrapped_parameters(self) -> List[nn.Parameter]:
        wrapped_ids: Set[int] = set()
        for mod in self.partitions.values():
            for p in mod.parameters():
                wrapped_ids.add(id(p))
        return [p for p in self.model.parameters() if id(p) not in wrapped_ids]

    def _apply_unwrapped_freeze(self, freeze: bool) -> None:
        if not self.freeze_unwrapped_on_aux:
            return
        if freeze:
            if not self._saved_unwrapped_req_grad:
                for p in self.unwrapped_parameters():
                    self._saved_unwrapped_req_grad[id(p)] = (p, p.requires_grad)
                    p.requires_grad_(False)
        else:
            for p, orig in list(self._saved_unwrapped_req_grad.values()):
                p.requires_grad_(orig)
            self._saved_unwrapped_req_grad.clear()

    def set_route(
        self,
        forward_active: Union[str, Sequence[str], Dict[str, float]] = ("core",),
        backward_active: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """Set active forward and backward partitions across all wrapped submodules."""
        if isinstance(forward_active, str):
            target = forward_active
            if target == "core":
                fwd_arg: Union[Sequence[str], Dict[str, float]] = ("core",)
                bck_arg: Sequence[str] = ("core",)
            else:
                fwd_arg = ("core", target)
                bck_arg = (target,)
        else:
            fwd_arg = forward_active
            if backward_active is None:
                bck_arg = ("core",)
            elif isinstance(backward_active, str):
                bck_arg = (backward_active,)
            else:
                bck_arg = tuple(backward_active)

        self.active_backward_partitions = set(bck_arg)
        for mod in self.partitions.values():
            mod.set_routing(fwd_arg, bck_arg)

        freeze_unwrapped = "core" not in self.active_backward_partitions
        self._apply_unwrapped_freeze(freeze_unwrapped)

    @contextmanager
    def route(
        self,
        forward_active: Union[str, Sequence[str], Dict[str, float]] = ("core",),
        backward_active: Optional[Union[str, Sequence[str]]] = None,
    ) -> Iterator["GRAMRouter"]:
        """Context manager activating a partition route and restoring core-only on exit."""
        prev_fwd = {name: dict(mod.forward_weights) for name, mod in self.partitions.items()}
        prev_bck = {name: set(mod.backward_active) for name, mod in self.partitions.items()}
        prev_freeze = bool(self._saved_unwrapped_req_grad)
        try:
            self.set_route(forward_active, backward_active)
            yield self
        finally:
            for name, mod in self.partitions.items():
                mod.forward_weights = prev_fwd[name]
                mod.backward_active = prev_bck[name]
            self.active_backward_partitions = set().union(*prev_bck.values()) if prev_bck else {"core"}
            self._apply_unwrapped_freeze(prev_freeze)

    def ablate(self, aux_name: str, active: bool = False) -> None:
        """Deploy-time hard ablation of an auxiliary partition (``active=False`` removes it)."""
        for mod in self.partitions.values():
            mod.ablate(aux_name, active=active)

    def set_aux_scale(self, aux_name: str, scale: float) -> None:
        for mod in self.partitions.values():
            if aux_name in mod.aux:
                mod.forward_weights[aux_name] = float(scale)

    def decorrelation_loss(self, weight: Optional[float] = None) -> torch.Tensor:
        """Compute total Residual Decorrelation Orthogonality Loss (+ optional cosine CB) across partitions."""
        losses: List[torch.Tensor] = [
            mod.last_decorrelation_loss
            for mod in self.partitions.values()
            if mod.last_decorrelation_loss is not None
        ]
        cb_losses: List[torch.Tensor] = [
            mod.last_cosine_cb_loss
            for mod in self.partitions.values()
            if mod.last_cosine_cb_loss is not None
        ]
        if not losses:
            ref_param = next(self.model.parameters(), None)
            dev = ref_param.device if ref_param is not None else torch.device("cpu")
            self.last_decorrelation_loss = 0.0
            self.last_circuit_breaker_loss = 0.0
            return torch.zeros((), device=dev, dtype=torch.float32)

        raw = torch.stack(losses).mean()
        w = float(self.policy.gram_decorrelation_weight if weight is None else weight)
        self.last_decorrelation_loss = float(raw.detach().item())
        total = raw * w
        if self.enable_cosine_circuit_breaker and cb_losses:
            cb_raw = torch.stack(cb_losses).mean()
            self.last_circuit_breaker_loss = float(cb_raw.detach().item())
            total = total + self.circuit_breaker_weight * cb_raw
        return total

    def snapshot_core_baseline(self) -> nn.Module:
        """Snapshot frozen core-only reference model for latent circuit-breaker rerouting."""
        from ojasx_safety.autograd.latent_defenses import _freeze_ref_copy

        with self.route("core"):
            ref = _freeze_ref_copy(self.model)
        self._core_ref_model = ref
        return ref

    def circuit_breaker_loss(
        self,
        harmful_inputs: Any,
        retain_inputs: Optional[Any] = None,
        *,
        cb_weight: float = 1.0,
        retain_weight: float = 1.0,
        aux_core_orth_weight: float = 0.5,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Layer-wise latent cosine circuit-breaker between harmful prompts & core hidden states."""
        from ojasx_safety.autograd.latent_defenses import (
            CircuitBreakerReroutingLoss,
            _forward_model,
        )

        if self._core_ref_model is None:
            self.snapshot_core_baseline()
        assert self._core_ref_model is not None

        if self._cb_engine is None:
            self._cb_engine = CircuitBreakerReroutingLoss(
                ref_model=self._core_ref_model,
                alpha_retain=retain_weight,
                rr_weight=cb_weight,
            )
        else:
            self._cb_engine.alpha_retain = float(retain_weight)
            self._cb_engine.rr_weight = float(cb_weight)

        ref_param = next(self.model.parameters(), None)
        dev = ref_param.device if ref_param is not None else torch.device("cpu")

        # 1) Reroute harmful prompts away from frozen core baseline hidden states across layers
        cb_val, cb_info = self._cb_engine(
            self.model,
            harmful_inputs,
            retain_inputs=retain_inputs,
            ref_model=self._core_ref_model,
            return_details=True,
        )

        # 2) Also penalize partition-level core vs aux cosine overlap if aux partitions fired on harmful inputs
        orth_terms: List[torch.Tensor] = []
        if aux_core_orth_weight > 0.0 and self.partitions:
            # Capture fresh partition core/aux activations from harmful_inputs if not already populated
            has_cached = any(p.last_h_core is not None and p.last_h_aux for p in self.partitions.values())
            if not has_cached and self.aux_names:
                with self.route(forward_active=("core", self.aux_names[0]), backward_active=tuple(self.active_backward_partitions)):
                    _ = _forward_model(self.model, harmful_inputs)
            for part in self.partitions.values():
                if part.last_cosine_cb_loss is not None:
                    orth_terms.append(part.last_cosine_cb_loss)
                elif part.last_h_core is not None and part.last_h_aux:
                    for h_aux in part.last_h_aux.values():
                        orth_terms.append(cosine_circuit_breaker_loss(part.last_h_core, h_aux))

        orth_loss = (
            torch.stack(orth_terms).mean().to(device=dev, dtype=torch.float32)
            if orth_terms
            else torch.zeros((), device=dev, dtype=torch.float32)
        )
        total = cb_val.to(device=dev, dtype=torch.float32) + float(aux_core_orth_weight) * orth_loss
        self.last_circuit_breaker_loss = float(total.detach().item())

        if return_details:
            cb_info["aux_core_cos_cb"] = float(orth_loss.detach().item())
            cb_info["total"] = float(total.detach().item())
            return total, cb_info
        return total


__all__ = [
    "GRAMLoRAAdapter",
    "GRAMPartitionedModule",
    "GRAMRouter",
    "residual_decorrelation_loss",
    "cosine_circuit_breaker_loss",
]
