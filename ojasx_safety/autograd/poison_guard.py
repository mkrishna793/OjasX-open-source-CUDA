"""Device-native streaming poison & backdoor defense (zero host NumPy copies).

Replaces offline/NumPy-bound IBM ART spectral & activation detectors with
streaming Welford online statistics, randomized rank-k SVD sketching,
gradient cosine/spectral quarantine, PyTorch autograd sample-mask hooks,
and PyTorch-native TRADES consistency loss across arbitrary architectures
(Causal LMs, SSMs, Diffusion, JEPA, CNNs, MLPs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx_safety.contracts import SafetyPolicy


class StreamingWelfordStats:
    """Device-native parallel Welford online mean/variance estimator."""

    def __init__(self, shape: Tuple[int, ...] = (), device: Optional[torch.device] = None) -> None:
        self.shape = tuple(shape)
        self.count: int = 0
        self.mean: Optional[torch.Tensor] = (
            torch.zeros(self.shape, dtype=torch.float32, device=device) if device is not None else None
        )
        self.m2: Optional[torch.Tensor] = (
            torch.zeros(self.shape, dtype=torch.float32, device=device) if device is not None else None
        )

    def _ensure_init(self, x: torch.Tensor) -> None:
        if self.mean is None or self.m2 is None:
            self.shape = tuple(x.shape[1:]) if x.ndim > 1 else tuple(x.shape) if self.shape else ()
            target_shape = self.shape
            self.mean = torch.zeros(target_shape, dtype=torch.float32, device=x.device)
            self.m2 = torch.zeros(target_shape, dtype=torch.float32, device=x.device)
        elif self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.m2 = self.m2.to(x.device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Incorporate a single observation or batch ``[B, ...]`` via Chan's parallel formula."""
        vals = x.detach().float()
        if vals.ndim == len(self.shape):
            vals = vals.unsqueeze(0)
        if vals.shape[0] == 0:
            return
        self._ensure_init(vals)
        assert self.mean is not None and self.m2 is not None

        b_count = int(vals.shape[0])
        b_mean = vals.mean(dim=0)
        if b_count > 1:
            b_m2 = ((vals - b_mean.unsqueeze(0)) ** 2).sum(dim=0)
        else:
            b_m2 = torch.zeros_like(b_mean)

        if self.count == 0:
            self.count = b_count
            self.mean.copy_(b_mean)
            self.m2.copy_(b_m2)
            return

        total = self.count + b_count
        delta = b_mean - self.mean
        self.mean.add_(delta * (float(b_count) / float(total)))
        self.m2.add_(b_m2 + (delta * delta) * (float(self.count * b_count) / float(total)))
        self.count = total

    def variance(self, unbiased: bool = True) -> torch.Tensor:
        if self.mean is None or self.m2 is None:
            return torch.tensor(0.0, dtype=torch.float32)
        denom = max(1, self.count - 1) if unbiased else max(1, self.count)
        return (self.m2 / float(denom)).clamp(min=0.0)

    def std(self, min_std: float = 1e-6, rel_floor: float = 0.05) -> torch.Tensor:
        if self.mean is None or self.m2 is None or self.count < 2:
            if self.mean is not None:
                return torch.full_like(self.mean, max(min_std, 1.0))
            return torch.tensor(max(min_std, 1.0), dtype=torch.float32)
        raw_std = torch.sqrt(self.variance(unbiased=True))
        floor = torch.clamp(self.mean.abs() * float(rel_floor), min=float(min_std))
        return torch.maximum(raw_std, floor)

    def zscore(self, x: torch.Tensor, min_std: float = 1e-6, rel_floor: float = 0.05) -> torch.Tensor:
        vals = x.detach().float()
        if self.mean is None or self.count < 2:
            return torch.zeros_like(vals)
        mean = self.mean.to(vals.device)
        std = self.std(min_std=min_std, rel_floor=rel_floor).to(vals.device)
        return (vals - mean) / std


def _randomized_top_k_svd(
    matrix: torch.Tensor,
    rank: int,
    n_iter: int = 2,
    seed: int = 1337,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch device-native randomized rank-k right singular vectors ``(S, Vh)``."""
    m, d = matrix.shape
    k = max(1, min(rank, m, d))
    work = matrix.float()
    if min(m, d) <= max(k + 4, 32):
        _, s, vh = torch.linalg.svd(work, full_matrices=False)
        return s[:k], vh[:k]

    q_dim = min(d, m, k + 4)
    gen = torch.Generator(device=work.device if work.device.type != "mps" else "cpu")
    gen.manual_seed(seed)
    omega = torch.randn(d, q_dim, dtype=torch.float32, generator=gen, device=gen.device).to(work.device)
    y = work @ omega
    for _ in range(max(1, n_iter)):
        q, _ = torch.linalg.qr(y, mode="reduced")
        z, _ = torch.linalg.qr(work.t() @ q, mode="reduced")
        y = work @ z
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.t() @ work
    _, s, vh = torch.linalg.svd(b, full_matrices=False)
    return s[:k], vh[:k]


class StreamingRankKSketch:
    """Device-native Frequent-Directions + CountSketch streaming rank-k SVD tracker."""

    def __init__(
        self,
        rank: int = 4,
        sketch_dim: int = 256,
        ema_decay: float = 0.9,
        seed: int = 2026,
    ) -> None:
        self.rank = max(1, int(rank))
        self.sketch_dim = max(16, int(sketch_dim))
        self.ema_decay = float(ema_decay)
        self.seed = int(seed)

        self.feature_stats = StreamingWelfordStats()
        self.proj_stats = StreamingWelfordStats(shape=())
        self.orth_stats = StreamingWelfordStats(shape=())
        self.sketch_matrix: Optional[torch.Tensor] = None
        self.ema_direction: Optional[torch.Tensor] = None
        self.basis_vh: Optional[torch.Tensor] = None
        self.singular_values: Optional[torch.Tensor] = None
        self.sketch_count: int = 0
        self._hash_cache: Dict[Tuple[int, str, Optional[int]], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_countsketch_map(self, dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (dim, device.type, device.index)
        cached = self._hash_cache.get(key)
        if cached is not None:
            return cached
        idx = torch.arange(dim, device=device, dtype=torch.int64)
        # SplitMix-inspired deterministic integer hash on-device (zero host copy).
        h = (idx + 0x9E3779B9 + self.seed) * 2654435761
        buckets = torch.remainder(h, self.sketch_dim)
        signs = torch.where((torch.bitwise_and(h >> 16, 1) == 0), 1.0, -1.0).to(torch.float32)
        self._hash_cache[key] = (buckets, signs)
        return buckets, signs

    def project_to_sketch(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``[B, D]`` onto ``[B, d_sketch]`` without materializing dense matrices."""
        vals = x.detach().float()
        if vals.ndim == 1:
            vals = vals.unsqueeze(0)
        b, d = vals.shape
        if d <= self.sketch_dim:
            return vals
        buckets, signs = self._get_countsketch_map(d, vals.device)
        out = torch.zeros(b, self.sketch_dim, device=vals.device, dtype=torch.float32)
        scaled = vals * signs.unsqueeze(0)
        out.scatter_add_(1, buckets.unsqueeze(0).expand(b, d), scaled)
        return out

    def sketch_param_dict(self, tensors: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linearly sketch a sequence of tensors into a single ``[1, d]`` vector and global L2 norm."""
        offset = 0
        tensors_list = [t.detach().float() for t in tensors if t is not None]
        if not tensors_list:
            return torch.zeros(1, 1, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)

        dev = tensors_list[0].device
        total_elems = sum(t.numel() for t in tensors_list)
        if total_elems <= self.sketch_dim:
            flat = torch.cat([t.reshape(-1) for t in tensors_list], dim=0).unsqueeze(0)
            norm = torch.linalg.vector_norm(flat.squeeze(0))
            return flat, norm

        sketch_accum = torch.zeros(1, self.sketch_dim, device=dev, dtype=torch.float32)
        norm_sq_accum = torch.zeros((), device=dev, dtype=torch.float32)
        for t in tensors_list:
            flat = t.reshape(-1)
            n = flat.numel()
            if n == 0:
                continue
            norm_sq_accum = norm_sq_accum + torch.dot(flat, flat)
            idx = torch.arange(offset, offset + n, device=dev, dtype=torch.int64)
            h = (idx + 0x9E3779B9 + self.seed) * 2654435761
            buckets = torch.remainder(h, self.sketch_dim)
            signs = torch.where((torch.bitwise_and(h >> 16, 1) == 0), 1.0, -1.0).to(torch.float32)
            sketch_accum.squeeze(0).scatter_add_(0, buckets, flat * signs)
            offset += n

        return sketch_accum, torch.sqrt(norm_sq_accum.clamp(min=0.0))

    @torch.no_grad()
    def _refresh_basis(self, batch_centered: Optional[torch.Tensor] = None) -> None:
        if self.sketch_matrix is None and batch_centered is None:
            return
        rows: List[torch.Tensor] = []
        if self.sketch_matrix is not None:
            rows.append(self.sketch_matrix)
        if batch_centered is not None and batch_centered.numel() > 0:
            rows.append(batch_centered)
        combined = torch.cat(rows, dim=0)
        m, d = combined.shape
        cap = min(2 * self.rank + 4, d)
        s, vh = _randomized_top_k_svd(combined, rank=min(cap, m, d))
        k_keep = min(self.rank, vh.shape[0])
        self.basis_vh = vh[:k_keep].detach()
        self.singular_values = s[:k_keep].detach()
        # Frequent-Directions shrinkage on the retained sketch rows.
        if s.shape[0] >= cap:
            delta = s[cap - 1] ** 2
            shrunk = torch.sqrt(torch.clamp(s[:cap] ** 2 - delta, min=0.0))
            self.sketch_matrix = (shrunk.unsqueeze(1) * vh[:cap]).detach()
        else:
            self.sketch_matrix = (s.unsqueeze(1) * vh).detach()

    @torch.no_grad()
    def score(self, x_sketch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute device-native spectral, deviation, and cosine scores BEFORE updating stats."""
        vals = x_sketch.detach().float()
        if vals.ndim == 1:
            vals = vals.unsqueeze(0)
        b, d = vals.shape
        if self.feature_stats.mean is not None and self.feature_stats.count > 0:
            mu = self.feature_stats.mean.to(vals.device).unsqueeze(0)
        elif b > 1:
            mu = vals.mean(dim=0, keepdim=True)
        else:
            mu = torch.zeros(1, d, device=vals.device, dtype=torch.float32)

        centered = vals - mu
        dev_norm = torch.linalg.vector_norm(centered, dim=-1)

        if self.ema_direction is not None:
            ema_dir = self.ema_direction.to(vals.device).unsqueeze(0)
            cos_sim = F.cosine_similarity(vals, ema_dir, dim=-1).clamp(-1.0, 1.0)
        else:
            cos_sim = torch.ones(b, device=vals.device, dtype=torch.float32)
        cos_dist = 1.0 - cos_sim

        # Top-k streaming SVD projection & orthogonal residual
        if self.basis_vh is not None:
            vh = self.basis_vh.to(vals.device)
            proj_coords = centered @ vh.t()  # [B, k]
            proj_v1 = proj_coords[:, 0].abs()
            recon = proj_coords @ vh
            orth_norm = torch.linalg.vector_norm(centered - recon, dim=-1)
        else:
            proj_v1 = dev_norm
            orth_norm = torch.zeros_like(dev_norm)

        # Within-batch spectral signature ratio (Tran et al. 2018) when batch has B >= 4
        if b >= 4:
            batch_c = vals - vals.mean(dim=0, keepdim=True)
            _, vh_b = _randomized_top_k_svd(batch_c, rank=1)
            batch_proj = (batch_c @ vh_b[0]).abs()
            med = batch_proj.median().clamp(min=1e-6)
            batch_spectral_ratio = batch_proj / med
        else:
            batch_spectral_ratio = torch.ones(b, device=vals.device, dtype=torch.float32)

        if self.proj_stats.mean is not None and self.proj_stats.count >= 2:
            base_proj = torch.maximum(
                self.proj_stats.mean.to(vals.device).abs(),
                torch.tensor(1e-6, device=vals.device),
            )
            stream_spectral_ratio = proj_v1 / base_proj
        else:
            stream_spectral_ratio = torch.ones(b, device=vals.device, dtype=torch.float32)

        if self.orth_stats.mean is not None and self.orth_stats.count >= 2:
            base_orth = torch.maximum(
                self.orth_stats.mean.to(vals.device).abs(),
                torch.tensor(1e-6, device=vals.device),
            )
            orth_ratio = orth_norm / base_orth
        else:
            orth_ratio = torch.ones(b, device=vals.device, dtype=torch.float32)

        spectral_ratio = torch.maximum(
            batch_spectral_ratio,
            torch.maximum(stream_spectral_ratio, orth_ratio),
        )
        return {
            "centered": centered,
            "dev_norm": dev_norm,
            "cos_sim": cos_sim,
            "cos_dist": cos_dist,
            "proj_v1": proj_v1,
            "orth_norm": orth_norm,
            "spectral_ratio": spectral_ratio,
            "batch_spectral_ratio": batch_spectral_ratio,
            "stream_spectral_ratio": stream_spectral_ratio,
        }

    @torch.no_grad()
    def commit_clean(self, x_sketch: torch.Tensor) -> None:
        """Update Welford mean/var, EMA gradient direction, and Frequent-Directions SVD sketch."""
        vals = x_sketch.detach().float()
        if vals.ndim == 1:
            vals = vals.unsqueeze(0)
        if vals.shape[0] == 0:
            return
        scores = self.score(vals)
        self.feature_stats.update(vals)
        self.proj_stats.update(scores["proj_v1"])
        self.orth_stats.update(scores["orth_norm"])
        mean_vec = vals.mean(dim=0)
        mean_norm = torch.linalg.vector_norm(mean_vec).clamp(min=1e-8)
        unit_vec = mean_vec / mean_norm
        if self.ema_direction is None:
            self.ema_direction = unit_vec.clone()
        else:
            ema = self.ema_direction.to(vals.device)
            ema.mul_(self.ema_decay).add_(unit_vec, alpha=1.0 - self.ema_decay)
            ema_norm = torch.linalg.vector_norm(ema).clamp(min=1e-8)
            ema.div_(ema_norm)
        centered = vals - self.feature_stats.mean.to(vals.device).unsqueeze(0)
        self._refresh_basis(centered)
        self.sketch_count += int(vals.shape[0])


@dataclass
class PoisonGuardResult:
    quarantined: bool
    reason: str
    grad_zscore: float
    spectral_ratio: float
    cosine_similarity: float
    sample_weights: Optional[torch.Tensor] = None
    sample_scores: Optional[torch.Tensor] = None


class StreamingPoisonGuard:
    """Streaming device-native activation & gradient quarantine guard + TRADES loss."""

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        rank: int = 4,
        sketch_dim: int = 256,
        warmup_steps: int = 4,
        ema_decay: float = 0.9,
        max_quarantine_batch_frac: float = 0.5,
        latent_monitor: Optional[Any] = None,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.rank = int(rank)
        self.sketch_dim = int(sketch_dim)
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_quarantine_batch_frac = float(max_quarantine_batch_frac)
        self.latent_monitor = latent_monitor
        self.last_latent_telemetry: Optional[Any] = None
        self.tvaccine_guard: Optional[Any] = None

        self.act_sketch = StreamingRankKSketch(rank=rank, sketch_dim=sketch_dim, ema_decay=ema_decay)
        self.act_norm_stats = StreamingWelfordStats(shape=())
        self.act_dev_stats = StreamingWelfordStats(shape=())
        self.act_cos_stats = StreamingWelfordStats(shape=())

        self.grad_sketch = StreamingRankKSketch(rank=rank, sketch_dim=sketch_dim, ema_decay=ema_decay)
        self.grad_norm_stats = StreamingWelfordStats(shape=())
        self.grad_dev_stats = StreamingWelfordStats(shape=())
        self.grad_cos_stats = StreamingWelfordStats(shape=())

        self.act_steps: int = 0
        self.grad_steps: int = 0
        self.quarantined_steps: int = 0

    def extract_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Architecture-agnostic ``[B, D_feat]`` feature extraction for LM/SSM/CNN/Diffusion/JEPA."""
        x = tensor.detach().float()
        if x.ndim == 1:
            return self.act_sketch.project_to_sketch(x.unsqueeze(0))
        if x.ndim == 2:
            return self.act_sketch.project_to_sketch(x)
        b = x.shape[0]
        trailing = x.numel() // max(1, b)
        if trailing <= self.sketch_dim:
            return x.reshape(b, trailing)
        if x.ndim == 3:
            # Sequence tensor [B, S, D]: combine mean + last-token + max-abs pooling
            mean_pool = x.mean(dim=1)
            max_pool = x.abs().amax(dim=1)
            last_pool = x[:, -1, :]
            pooled = torch.cat([mean_pool, max_pool, last_pool], dim=-1)
        else:
            # Spatial/spatiotemporal tensor [B, C, ...]: pool spatial dimensions
            spatial_dims = tuple(range(2, x.ndim))
            mean_pool = x.mean(dim=spatial_dims)
            max_pool = x.abs().amax(dim=spatial_dims)
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.act_sketch.project_to_sketch(pooled)

    @torch.no_grad()
    def inspect_activations(
        self,
        activations: torch.Tensor,
        commit: bool = True,
    ) -> PoisonGuardResult:
        """Inspect intermediate hidden states/activations of any architecture."""
        if not bool(torch.isfinite(activations).all()):
            self.quarantined_steps += 1
            b = activations.shape[0] if activations.ndim >= 1 else 1
            return PoisonGuardResult(
                quarantined=True,
                reason="non_finite_activations",
                grad_zscore=float("inf"),
                spectral_ratio=float("inf"),
                cosine_similarity=0.0,
                sample_weights=torch.zeros(b, device=activations.device, dtype=torch.float32),
            )

        feats = self.extract_features(activations)
        b = feats.shape[0]
        norms = torch.linalg.vector_norm(feats, dim=-1)
        scores = self.act_sketch.score(feats)
        dev_norms = scores["dev_norm"]
        cos_sim = scores["cos_sim"]
        cos_dist = scores["cos_dist"]
        spectral_ratio = scores["spectral_ratio"]

        if self.act_steps >= self.warmup_steps:
            z_norm = self.act_norm_stats.zscore(norms, min_std=1e-4, rel_floor=0.08).abs()
            z_dev = self.act_dev_stats.zscore(dev_norms, min_std=1e-4, rel_floor=0.08).abs()
            z_cos = self.act_cos_stats.zscore(cos_dist, min_std=0.05, rel_floor=0.10)
            sample_z = torch.maximum(z_norm, torch.maximum(z_dev, z_cos))
        elif b >= 4:
            # Even before warmup completes, detect strong within-batch spectral outliers
            sample_z = torch.zeros(b, device=feats.device, dtype=torch.float32)
        else:
            sample_z = torch.zeros(b, device=feats.device, dtype=torch.float32)

        bad_z = sample_z > float(self.policy.max_grad_zscore)
        bad_spec = spectral_ratio > float(self.policy.max_spectral_outlier_ratio)
        is_outlier = bad_z | bad_spec
        sample_weights = torch.where(is_outlier, 0.0, 1.0).to(dtype=torch.float32)

        max_z_val = float(sample_z.max().item()) if b > 0 else 0.0
        max_spec_val = float(spectral_ratio.max().item()) if b > 0 else 1.0
        mean_cos = float(cos_sim.mean().item()) if b > 0 else 1.0

        bad_frac = float(is_outlier.float().mean().item()) if b > 0 else 0.0
        quarantined = False
        reasons: List[str] = []
        if b == 1 and bool(is_outlier.any()):
            quarantined = True
        elif bad_frac > self.max_quarantine_batch_frac:
            quarantined = True
        elif float(sample_z.mean().item()) > float(self.policy.max_grad_zscore):
            quarantined = True

        if bool(bad_z.any()):
            reasons.append(f"activation_zscore>{self.policy.max_grad_zscore}")
        if bool(bad_spec.any()):
            reasons.append(f"spectral_ratio>{self.policy.max_spectral_outlier_ratio}")

        if quarantined:
            self.quarantined_steps += 1
        elif commit:
            clean_mask = sample_weights > 0.5
            if bool(clean_mask.any()):
                self.act_sketch.commit_clean(feats[clean_mask])
                self.act_norm_stats.update(norms[clean_mask])
                self.act_dev_stats.update(dev_norms[clean_mask])
                self.act_cos_stats.update(cos_dist[clean_mask])
                self.act_steps += 1

        return PoisonGuardResult(
            quarantined=quarantined,
            reason=";".join(reasons) if reasons else "clean",
            grad_zscore=max_z_val,
            spectral_ratio=max_spec_val,
            cosine_similarity=mean_cos,
            sample_weights=sample_weights,
            sample_scores=torch.maximum(sample_z, spectral_ratio),
        )

    @torch.no_grad()
    def inspect_gradients(
        self,
        gradients: Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]],
        commit: bool = True,
    ) -> PoisonGuardResult:
        """Inspect flattened/sketched parameter gradients BEFORE optimizer updates weights."""
        if isinstance(gradients, dict):
            grad_list = [g for g in gradients.values() if g is not None]
        elif isinstance(gradients, torch.Tensor):
            grad_list = [gradients]
        else:
            grad_list = [g for g in gradients if g is not None]

        if not grad_list:
            return PoisonGuardResult(
                quarantined=False,
                reason="no_gradients",
                grad_zscore=0.0,
                spectral_ratio=1.0,
                cosine_similarity=1.0,
            )

        for g in grad_list:
            if not bool(torch.isfinite(g).all()):
                self.quarantined_steps += 1
                return PoisonGuardResult(
                    quarantined=True,
                    reason="non_finite_gradient",
                    grad_zscore=float("inf"),
                    spectral_ratio=float("inf"),
                    cosine_similarity=0.0,
                )

        sketch_vec, total_norm = self.grad_sketch.sketch_param_dict(grad_list)
        scores = self.grad_sketch.score(sketch_vec)
        dev_norm = scores["dev_norm"].squeeze(0)
        cos_sim = scores["cos_sim"].squeeze(0)
        cos_dist = scores["cos_dist"].squeeze(0)
        spectral_ratio = scores["spectral_ratio"].squeeze(0)

        grad_zscore = 0.0
        if self.grad_steps >= self.warmup_steps:
            z_norm = float(self.grad_norm_stats.zscore(total_norm, min_std=1e-5, rel_floor=0.08).abs().item())
            z_dev = float(self.grad_dev_stats.zscore(dev_norm, min_std=1e-5, rel_floor=0.08).abs().item())
            z_cos = float(self.grad_cos_stats.zscore(cos_dist, min_std=0.05, rel_floor=0.10).item())
            grad_zscore = max(z_norm, z_dev, z_cos)

        spec_val = float(spectral_ratio.item())
        cos_val = float(cos_sim.item())
        quarantined = False
        reasons: List[str] = []
        if self.grad_steps >= self.warmup_steps:
            if grad_zscore > float(self.policy.max_grad_zscore):
                quarantined = True
                reasons.append(f"grad_zscore={grad_zscore:.2f}>{self.policy.max_grad_zscore}")
            if spec_val > float(self.policy.max_spectral_outlier_ratio):
                quarantined = True
                reasons.append(f"spectral_ratio={spec_val:.2f}>{self.policy.max_spectral_outlier_ratio}")

        if quarantined:
            self.quarantined_steps += 1
        elif commit:
            self.grad_sketch.commit_clean(sketch_vec)
            self.grad_norm_stats.update(total_norm)
            self.grad_dev_stats.update(dev_norm)
            self.grad_cos_stats.update(cos_dist)
            self.grad_steps += 1

        return PoisonGuardResult(
            quarantined=quarantined,
            reason=";".join(reasons) if reasons else "clean",
            grad_zscore=grad_zscore,
            spectral_ratio=spec_val,
            cosine_similarity=cos_val,
        )

    def calibrate(
        self,
        clean_activations: Optional[Iterable[torch.Tensor]] = None,
        clean_gradients: Optional[Iterable[Any]] = None,
    ) -> None:
        """Warm up streaming baselines on verified clean activations and/or gradients."""
        if clean_activations is not None:
            for act in clean_activations:
                feats = self.extract_features(act)
                norms = torch.linalg.vector_norm(feats, dim=-1)
                scores = self.act_sketch.score(feats)
                self.act_sketch.commit_clean(feats)
                self.act_norm_stats.update(norms)
                self.act_dev_stats.update(scores["dev_norm"])
                self.act_cos_stats.update(scores["cos_dist"])
                self.act_steps += 1
        if clean_gradients is not None:
            for grads in clean_gradients:
                if isinstance(grads, dict):
                    g_list = [g for g in grads.values() if g is not None]
                elif isinstance(grads, torch.Tensor):
                    g_list = [grads]
                else:
                    g_list = [g for g in grads if g is not None]
                if not g_list:
                    continue
                sketch_vec, total_norm = self.grad_sketch.sketch_param_dict(g_list)
                scores = self.grad_sketch.score(sketch_vec)
                self.grad_sketch.commit_clean(sketch_vec)
                self.grad_norm_stats.update(total_norm)
                self.grad_dev_stats.update(scores["dev_norm"].squeeze(0))
                self.grad_cos_stats.update(scores["cos_dist"].squeeze(0))
                self.grad_steps += 1

    @staticmethod
    def attach_sample_mask_hook(
        tensor: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> Optional[Any]:
        """Attach an autograd backward hook enforcing exact zero gradient for ``w=0`` samples."""
        if not tensor.requires_grad:
            return None
        w = (sample_weights >= 0.5).to(device=tensor.device, dtype=tensor.dtype)
        view_shape = (w.shape[0],) + (1,) * (tensor.ndim - 1)
        mask = w.view(view_shape)
        return tensor.register_hook(lambda grad: grad * mask)

    @classmethod
    def apply_sample_mask(
        cls,
        per_sample_loss: torch.Tensor,
        sample_weights: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Apply strict ``w in {0, 1}`` forward and autograd backward sample masking."""
        w = (sample_weights >= 0.5).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        if per_sample_loss.ndim > 1:
            w_view = w.view((w.shape[0],) + (1,) * (per_sample_loss.ndim - 1))
        else:
            w_view = w
        if per_sample_loss.requires_grad:
            cls.attach_sample_mask_hook(per_sample_loss, w)
        masked = per_sample_loss * w_view
        if not normalize:
            return masked.sum()
        active = w.sum()
        if float(active.item()) <= 0.0:
            return masked.sum() * 0.0
        return masked.sum() / active

    def inspect_model_hidden_layers(
        self,
        model: nn.Module,
        inputs: Any,
        *,
        update_baseline: bool = False,
        calibrate_safe: bool = False,
    ) -> Any:
        """Inspect ALL intermediate hidden layers of ``model`` via ``AllLayerLatentMonitor``."""
        from ojasx_safety.autograd.latent_defenses import AllLayerLatentMonitor

        if self.latent_monitor is None:
            self.latent_monitor = AllLayerLatentMonitor(
                policy=self.policy,
                warmup_steps=self.warmup_steps,
                max_quarantine_batch_frac=self.max_quarantine_batch_frac,
            )
        if calibrate_safe:
            self.latent_monitor.calibrate(model, inputs)
        telem = self.latent_monitor.inspect_all_layers(model, inputs, update_baseline=update_baseline)
        self.last_latent_telemetry = telem
        if telem.quarantined:
            self.quarantined_steps += 1
        return telem

    def all_layer_lat_loss(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Optional[torch.Tensor] = None,
        *,
        loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
        rho: float = 0.5,
        top_k: Optional[int] = None,
        num_inner_steps: int = 1,
        clean_weight: float = 0.5,
        anchor_weight: float = 0.0,
        sample_weights: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute T-Vaccine all-layer latent adversarial training loss alongside TRADES."""
        from ojasx_safety.autograd.latent_defenses import TVaccineLayerGuard

        if self.tvaccine_guard is None:
            self.tvaccine_guard = TVaccineLayerGuard(
                rho=rho,
                top_k=top_k,
                num_inner_steps=num_inner_steps,
                clean_weight=clean_weight,
                anchor_weight=anchor_weight,
            )
        return all_layer_lat_loss(
            model,
            inputs,
            targets=targets,
            loss_fn=loss_fn,
            rho=rho,
            top_k=top_k,
            num_inner_steps=num_inner_steps,
            clean_weight=clean_weight,
            anchor_weight=anchor_weight,
            guard=self.tvaccine_guard,
            sample_weights=sample_weights,
            return_details=return_details,
        )


def _find_embedding_module(model: nn.Module) -> Optional[nn.Embedding]:
    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            return mod
    return None


def trades_loss(
    model_or_logits: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
    inputs_or_pert_logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    *,
    eps: float = 0.05,
    step_size: float = 0.02,
    num_steps: int = 3,
    beta: float = 6.0,
    norm: str = "linf",
    sample_weights: Optional[torch.Tensor] = None,
    embedding_module: Optional[nn.Module] = None,
    ignore_index: int = -100,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Pure PyTorch TRADES consistency loss under bounded input/embedding perturbations.

    Supports:
    1. Direct logits: ``trades_loss(clean_logits, pert_logits, targets)``
    2. Continuous input models (CNN / Diffusion / JEPA / MLP): ``trades_loss(model, x, targets)``
    3. Discrete token Causal LM / SSM models: ``trades_loss(model, input_ids, targets)``
       perturbing the embedding output via an automatic temporary forward hook.
    """
    # Mode 1: Direct precomputed logits
    if isinstance(model_or_logits, torch.Tensor):
        clean_logits = model_or_logits
        pert_logits = inputs_or_pert_logits
        return _compute_trades_from_logits(
            clean_logits=clean_logits,
            pert_logits=pert_logits,
            targets=targets,
            beta=beta,
            sample_weights=sample_weights,
            ignore_index=ignore_index,
            return_details=return_details,
        )

    model = model_or_logits
    x = inputs_or_pert_logits
    was_training = isinstance(model, nn.Module) and model.training

    def _extract_logits(out: Any) -> torch.Tensor:
        return out.logits if hasattr(out, "logits") else out

    # Clean forward pass
    clean_logits = _extract_logits(model(x))
    clean_probs_log = F.log_softmax(clean_logits.detach().float(), dim=-1)

    if torch.is_floating_point(x):
        # Continuous input perturbation (Diffusion / JEPA / CNN / MLP)
        delta = torch.empty_like(x, dtype=torch.float32).uniform_(-eps, eps).to(x.dtype)
        if norm == "l2":
            flat_n = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1, keepdim=True)
            view_shape = (delta.shape[0],) + (1,) * (delta.ndim - 1)
            delta = delta / flat_n.view(view_shape).clamp(min=1e-8) * (eps * 0.5)

        if isinstance(model, nn.Module):
            model.eval()
        for _ in range(max(1, num_steps)):
            delta_var = delta.detach().requires_grad_(True)
            with torch.enable_grad():
                adv_logits = _extract_logits(model(x + delta_var))
                kl_adv = F.kl_div(
                    F.log_softmax(adv_logits.float(), dim=-1),
                    clean_probs_log,
                    reduction="batchmean",
                    log_target=True,
                )
                grad_delta = torch.autograd.grad(kl_adv, delta_var)[0]
            with torch.no_grad():
                if norm == "l2":
                    g_flat = grad_delta.reshape(grad_delta.shape[0], -1)
                    g_norm = torch.linalg.vector_norm(g_flat, dim=-1).clamp(min=1e-8)
                    view_shape = (delta.shape[0],) + (1,) * (delta.ndim - 1)
                    delta = delta + step_size * (grad_delta / g_norm.view(view_shape))
                    d_norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1).clamp(min=1e-8)
                    scale = torch.clamp(eps / d_norm, max=1.0).view(view_shape)
                    delta = delta * scale
                else:
                    delta = torch.clamp(delta + step_size * grad_delta.sign(), min=-eps, max=eps)
        if was_training:
            model.train()
        pert_logits = _extract_logits(model(x + delta.detach()))
    else:
        # Discrete token input (Causal LM / SSM): perturb embedding output in-flight
        emb_mod = embedding_module if embedding_module is not None else (
            _find_embedding_module(model) if isinstance(model, nn.Module) else None
        )
        if emb_mod is None:
            raise ValueError("trades_loss requires an nn.Embedding module when inputs are discrete token IDs.")

        # Probe embedding shape once without grad
        captured_shape: Dict[str, Any] = {}

        def _probe_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
            captured_shape["shape"] = out.shape
            captured_shape["dtype"] = out.dtype
            captured_shape["device"] = out.device
            return out

        probe_handle = emb_mod.register_forward_hook(_probe_hook)
        try:
            with torch.no_grad():
                _ = model(x)
        finally:
            probe_handle.remove()

        delta = torch.empty(
            captured_shape["shape"],
            device=captured_shape["device"],
            dtype=captured_shape["dtype"],
        ).uniform_(-eps, eps)

        if isinstance(model, nn.Module):
            model.eval()
        for _ in range(max(1, num_steps)):
            delta_var = delta.detach().requires_grad_(True)

            def _inject_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                return out + delta_var

            h = emb_mod.register_forward_hook(_inject_hook)
            try:
                with torch.enable_grad():
                    adv_logits = _extract_logits(model(x))
                    kl_adv = F.kl_div(
                        F.log_softmax(adv_logits.float(), dim=-1),
                        clean_probs_log,
                        reduction="batchmean",
                        log_target=True,
                    )
                    grad_delta = torch.autograd.grad(kl_adv, delta_var)[0]
            finally:
                h.remove()

            with torch.no_grad():
                delta = torch.clamp(delta + step_size * grad_delta.sign(), min=-eps, max=eps)

        if was_training:
            model.train()

        delta_final = delta.detach()
        h_final = emb_mod.register_forward_hook(lambda _m, _i, out: out + delta_final)
        try:
            pert_logits = _extract_logits(model(x))
        finally:
            h_final.remove()

    return _compute_trades_from_logits(
        clean_logits=clean_logits,
        pert_logits=pert_logits,
        targets=targets,
        beta=beta,
        sample_weights=sample_weights,
        ignore_index=ignore_index,
        return_details=return_details,
    )


def _compute_trades_from_logits(
    clean_logits: torch.Tensor,
    pert_logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    beta: float,
    sample_weights: Optional[torch.Tensor],
    ignore_index: int,
    return_details: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    b = clean_logits.shape[0]
    kl_per_elem = F.kl_div(
        F.log_softmax(pert_logits.float(), dim=-1),
        F.log_softmax(clean_logits.detach().float(), dim=-1),
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    per_sample_kl = kl_per_elem.reshape(b, -1).mean(dim=-1)

    if targets is not None:
        ce = F.cross_entropy(
            clean_logits.reshape(-1, clean_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(targets.shape)
        if targets.ndim > 1:
            valid = (targets != ignore_index).float()
            per_sample_clean = (ce * valid).reshape(b, -1).sum(dim=-1) / valid.reshape(b, -1).sum(dim=-1).clamp(min=1.0)
        else:
            per_sample_clean = ce
    else:
        per_sample_clean = torch.zeros_like(per_sample_kl)

    if sample_weights is not None:
        loss_clean = StreamingPoisonGuard.apply_sample_mask(per_sample_clean, sample_weights)
        loss_kl = StreamingPoisonGuard.apply_sample_mask(per_sample_kl, sample_weights)
    else:
        loss_clean = per_sample_clean.mean()
        loss_kl = per_sample_kl.mean()

    total = loss_clean + float(beta) * loss_kl
    if return_details:
        return total, {
            "loss_clean": float(loss_clean.detach().item()),
            "loss_kl": float(loss_kl.detach().item()),
            "total": float(total.detach().item()),
        }
    return total


def all_layer_lat_loss(
    model: nn.Module,
    inputs: Any,
    targets: Optional[torch.Tensor] = None,
    *,
    loss_fn: Optional[Callable[[nn.Module, Any], torch.Tensor]] = None,
    rho: float = 0.5,
    top_k: Optional[int] = None,
    num_inner_steps: int = 1,
    clean_weight: float = 0.5,
    anchor_weight: float = 0.0,
    guard: Optional[Any] = None,
    sample_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Pure PyTorch T-Vaccine / Latent Adversarial Training loss across all hidden layers."""
    from ojasx_safety.autograd.latent_defenses import TVaccineLayerGuard, _default_task_loss

    active_guard = guard or TVaccineLayerGuard(
        rho=rho,
        top_k=top_k,
        num_inner_steps=num_inner_steps,
        clean_weight=clean_weight,
        anchor_weight=anchor_weight,
    )
    batch = (inputs, targets) if targets is not None else inputs

    if loss_fn is None and sample_weights is not None and targets is not None:
        def _masked_loss_fn(m: nn.Module, b_in: Any) -> torch.Tensor:
            x_in, y_tgt = b_in
            out = m(x_in)
            logits = out.logits if hasattr(out, "logits") else out
            if not y_tgt.is_floating_point() and logits.ndim >= 2:
                bsz = logits.shape[0]
                if logits.ndim == 3 and y_tgt.ndim == 1 and y_tgt.shape[0] == bsz:
                    per_s = F.cross_entropy(logits[:, -1, :].float(), y_tgt.long(), ignore_index=ignore_index, reduction="none")
                else:
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y_tgt.reshape(-1).long(),
                        ignore_index=ignore_index,
                        reduction="none",
                    ).reshape(bsz, -1)
                    valid = (y_tgt.reshape(bsz, -1) != ignore_index).float()
                    per_s = (ce * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
            else:
                per_s = F.mse_loss(logits.float(), y_tgt.float().to(logits.device), reduction="none").reshape(logits.shape[0], -1).mean(dim=-1)
            return StreamingPoisonGuard.apply_sample_mask(per_s, sample_weights)

        eval_fn: Callable[[nn.Module, Any], torch.Tensor] = _masked_loss_fn
    else:
        eval_fn = loss_fn or _default_task_loss

    return active_guard.adversarial_loss(
        model,
        batch,
        loss_fn=eval_fn,
        return_details=return_details,
    )


__all__ = [
    "StreamingWelfordStats",
    "StreamingRankKSketch",
    "PoisonGuardResult",
    "StreamingPoisonGuard",
    "trades_loss",
    "all_layer_lat_loss",
]
