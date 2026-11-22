"""
Adaptive Precision Thermostat & Rotating CW-Orbit Full-Rank Training Controller.

1. Adaptive Precision Thermostat (`select_sensitive_layers`, `pack_model_e8_adaptive`):
   Measures per-layer rotated-E8 quantization SNR before packing and keeps the top
   `keep_fp_fraction` (e.g., 8%) most sensitive layers in BF16/FP16 while packing
   the remaining ~92% to E8—preventing quantization-induced divergence.

2. Rotating CW-Orbit Full-Rank Controller (`RotatingCWOrbitController`):
   Replaces low-rank LoRA with O(d) `CWOFTLinear` adapters and rotates their exact
   orthogonal CW(120, 36) / Sign-FWHT orbits every `rotate_every_steps` steps,
   folding each orthogonal subspace update into the base weights with zero SVD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn as nn

from ojasx2.math.lattice import E8LatticeLinear, _get_packer_for_bitrate
from ojasx3.lattice.transforms import CWOFTLinear, apply_incoherence_transform


@dataclass(frozen=True)
class LayerSNRReport:
    name: str
    shape: Tuple[int, int]
    snr_db: float
    keep_high_precision: bool


@torch.no_grad()
def _measure_layer_rotated_e8_snr(
    weight: torch.Tensor,
    *,
    bitrate: float = 2.0,
    group_size: int = 32,
) -> float:
    w = weight.detach().to(torch.float32)
    w_rot = apply_incoherence_transform(w, inverse=False)
    packer = _get_packer_for_bitrate(bitrate)
    packed, scales, sh = packer.pack(w_rot, group_size=group_size)
    w_q = apply_incoherence_transform(
        packer.unpack(packed, scales, sh, dtype=torch.float32),
        inverse=True,
    )
    diff = w - w_q
    sig = float(w.pow(2).sum().item())
    err = float(diff.pow(2).sum().item())
    if sig <= 1e-15:
        return 100.0
    g_snr = 10.0 * math.log10(sig / max(err, 1e-15))
    if w.ndim == 2:
        col_snr = float(
            (10.0 * torch.log10(w.pow(2).sum(dim=0).clamp_min(1e-15) / diff.pow(2).sum(dim=0).clamp_min(1e-15)))
            .mean()
            .item()
        )
        row_snr = float(
            (10.0 * torch.log10(w.pow(2).sum(dim=1).clamp_min(1e-15) / diff.pow(2).sum(dim=1).clamp_min(1e-15)))
            .mean()
            .item()
        )
        return min(g_snr, col_snr, row_snr)
    return g_snr


@torch.no_grad()
def select_sensitive_layers(
    model: nn.Module,
    *,
    bitrate: float = 2.0,
    group_size: int = 32,
    keep_fp_fraction: float = 0.08,
    min_numel: int = 4096,
    always_keep_names: Sequence[str] = ("lm_head", "embed_tokens", "wte", "head"),
) -> Tuple[Set[str], List[LayerSNRReport]]:
    """Measure per-layer rotated-E8 SNR and return the set of sensitive layer names to keep in FP16/BF16."""
    candidates: List[Tuple[str, Tuple[int, int], float]] = []
    always_keep: Set[str] = set()

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(tag in name for tag in always_keep_names) or mod.weight.numel() < min_numel:
            always_keep.add(name)
            continue
        snr = _measure_layer_rotated_e8_snr(mod.weight, bitrate=bitrate, group_size=group_size)
        candidates.append((name, (mod.out_features, mod.in_features), snr))

    candidates.sort(key=lambda item: item[2])  # lowest SNR first (most sensitive)
    n_keep = max(0, int(math.ceil(len(candidates) * max(0.0, min(1.0, keep_fp_fraction)))))
    sensitive_names = set(always_keep)
    for name, _, _ in candidates[:n_keep]:
        sensitive_names.add(name)

    reports = [
        LayerSNRReport(name=n, shape=sh, snr_db=snr, keep_high_precision=(n in sensitive_names))
        for n, sh, snr in candidates
    ]
    return sensitive_names, reports


class RotatingCWOrbitController:
    """Manages periodic orthogonal CW(120,36) / Sign-FWHT orbit rotation for full-rank O(d) training."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rotate_every_steps: int = 50,
    ) -> None:
        if rotate_every_steps < 1:
            raise ValueError("rotate_every_steps must be >= 1")
        self.model = model
        self.rotate_every_steps = int(rotate_every_steps)
        self.step_count = 0
        self.rotations_completed = 0

    def attach_cwoft(
        self,
        *,
        target_modules: Optional[Sequence[str]] = None,
        min_numel: int = 256,
    ) -> int:
        """Replace eligible nn.Linear layers with CWOFTLinear adapters in-place."""
        attached = 0

        def _walk(parent: nn.Module, prefix: str = "") -> None:
            nonlocal attached
            for cname, child in list(parent.named_children()):
                full = f"{prefix}.{cname}" if prefix else cname
                if isinstance(child, nn.Linear) and not isinstance(child, CWOFTLinear):
                    if child.weight.numel() >= min_numel:
                        if target_modules is None or any(t in full for t in target_modules):
                            setattr(parent, cname, CWOFTLinear(child, orbit_id=0))
                            attached += 1
                else:
                    _walk(child, full)

        _walk(self.model)
        return attached

    def step(self, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Advance step counter; rotate all CWOFTLinear orbits when interval is reached."""
        self.step_count += 1
        if self.step_count % self.rotate_every_steps != 0:
            return False
        rotated_params: List[nn.Parameter] = []
        for mod in self.model.modules():
            if isinstance(mod, CWOFTLinear):
                mod.rotate_orbit()
                if mod.d_in is not None:
                    rotated_params.append(mod.d_in)
                if mod.d_out is not None:
                    rotated_params.append(mod.d_out)
        if optimizer is not None:
            for p in rotated_params:
                optimizer.state.pop(p, None)
        self.rotations_completed += 1
        return True


__all__ = [
    "LayerSNRReport",
    "select_sensitive_layers",
    "RotatingCWOrbitController",
]
