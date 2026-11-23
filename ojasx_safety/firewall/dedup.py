"""Portable CPU/MPS deduplication and cross-sample context reconstruction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import xxhash
from sklearn.feature_extraction.text import TfidfVectorizer

_MERSENNE_31: int = (1 << 31) - 1
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str, *, lowercase: bool = True) -> str:
    """Normalize text deterministically across Unicode and whitespace forms."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.lower() if lowercase else cleaned


def resolve_portable_device(preferred: str = "cpu") -> torch.device:
    """Resolve CPU or Apple Silicon MPS device with zero CUDA assumptions."""
    mode = (preferred or "cpu").strip().lower()
    if mode == "cuda" or mode.startswith("cuda:"):
        raise ValueError("CUDA is disabled in OjasX-Aegis portable firewall; use 'cpu', 'mps', or 'auto'")
    if mode == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if mode == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if mode == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported portable device: {preferred!r}")


class _UnionFind:
    """Disjoint-set forest keeping the smallest sample index as canonical representative."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> Tuple[int, int]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, rb
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return ra, rb


@dataclass
class DedupResult:
    """Structured result from exact, fuzzy MinHash/LSH, or semantic deduplication."""

    kept_indices: List[int]
    duplicate_indices: List[int]
    duplicate_clusters: List[List[int]]
    duplicate_pairs: List[Tuple[int, int, float]]
    keep_mask: List[bool]
    method_stats: Dict[str, Any] = field(default_factory=dict)


def _select_lsh_bands(num_perm: int, jaccard_threshold: float, requested_bands: Optional[int]) -> Tuple[int, int]:
    """Select (bands, rows_per_band) dividing num_perm for high recall at threshold."""
    if requested_bands is not None:
        if requested_bands <= 0 or num_perm % requested_bands != 0:
            raise ValueError(f"bands ({requested_bands}) must be > 0 and divide num_perm ({num_perm})")
        return requested_bands, num_perm // requested_bands

    divisors = [r for r in range(1, num_perm + 1) if num_perm % r == 0]
    best_r = 1
    best_score = float("inf")
    for r in divisors:
        b = num_perm // r
        # Ensure strong capture probability >= 0.99 at jaccard_threshold when feasible.
        capture_prob = 1.0 - (1.0 - (jaccard_threshold ** r)) ** b
        inflection = (1.0 / b) ** (1.0 / r)
        penalty = 0.0 if capture_prob >= 0.99 else (0.99 - capture_prob) * 100.0
        score = penalty + abs(inflection - max(0.25, jaccard_threshold * 0.55))
        if score < best_score:
            best_score = score
            best_r = r
    return num_perm // best_r, best_r


class PortableDeduplicator:
    """CPU/MPS portable exact, fuzzy MinHash+LSH, and semantic cosine deduplicator."""

    def __init__(
        self,
        *,
        jaccard_threshold: float = 0.85,
        semantic_threshold: float = 0.90,
        num_perm: int = 128,
        bands: Optional[int] = None,
        ngram_size: int = 5,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not (0.0 < jaccard_threshold <= 1.0):
            raise ValueError(f"jaccard_threshold must be in (0, 1], got {jaccard_threshold}")
        if not (0.0 < semantic_threshold <= 1.0):
            raise ValueError(f"semantic_threshold must be in (0, 1], got {semantic_threshold}")
        if num_perm < 8:
            raise ValueError(f"num_perm must be >= 8, got {num_perm}")
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {ngram_size}")

        self.jaccard_threshold = float(jaccard_threshold)
        self.semantic_threshold = float(semantic_threshold)
        self.num_perm = int(num_perm)
        self.bands, self.rows_per_band = _select_lsh_bands(
            self.num_perm, self.jaccard_threshold, bands
        )
        self.ngram_size = int(ngram_size)
        self.seed = int(seed)
        self.device = resolve_portable_device(device)

        rng = np.random.RandomState(self.seed)
        self._perm_a = rng.randint(1, _MERSENNE_31, size=self.num_perm, dtype=np.uint64)
        self._perm_b = rng.randint(0, _MERSENNE_31, size=self.num_perm, dtype=np.uint64)

        # Streaming stateful tracking
        self._seen_exact: Dict[str, int] = {}
        self._seen_signatures: List[np.ndarray] = []
        self._lsh_buckets: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(self.bands)]
        self._stream_count: int = 0

    def reset(self) -> None:
        """Clear stateful streaming deduplication index."""
        self._seen_exact.clear()
        self._seen_signatures.clear()
        self._lsh_buckets = [defaultdict(list) for _ in range(self.bands)]
        self._stream_count = 0

    @staticmethod
    def exact_hash(text: str, *, lowercase: bool = True) -> Tuple[str, str]:
        """Compute normalized (sha256_hex, xxh64_hex) digest pair."""
        norm = normalize_for_dedup(text, lowercase=lowercase)
        raw = norm.encode("utf-8")
        return hashlib.sha256(raw).hexdigest(), xxhash.xxh64_hexdigest(raw, seed=0)

    def _shingles(self, text: str) -> Set[str]:
        norm = normalize_for_dedup(text, lowercase=True)
        if not norm:
            return {""}
        n = min(self.ngram_size, len(norm)) if len(norm) >= 2 else 1
        char_shingles = {norm[i : i + n] for i in range(max(1, len(norm) - n + 1))}
        # Blend word tokens for very short records so single-token variations are captured cleanly
        if len(norm) < 120:
            words = norm.split(" ")
            char_shingles.update(f"w:{w}" for w in words if w)
        return char_shingles

    def exact_shingle_jaccard(self, text_a: str, text_b: str) -> float:
        """Compute exact Jaccard similarity between shingle sets of two texts."""
        sa = self._shingles(text_a)
        sb = self._shingles(text_b)
        union = len(sa | sb)
        if union == 0:
            return 1.0
        return float(len(sa & sb)) / float(union)

    def compute_minhash(self, text: str) -> np.ndarray:
        """Compute portable uint32 MinHash signature via xxhash + Mersenne-31 permutations."""
        shingles = self._shingles(text)
        base_hashes = np.fromiter(
            (xxhash.xxh64_intdigest(s.encode("utf-8"), seed=self.seed) % _MERSENNE_31 for s in shingles),
            dtype=np.uint64,
            count=len(shingles),
        )
        # Vectorized universal hash matrix [num_shingles, num_perm] -> min over shingles
        # Max intermediate value < (2^31)^2 + 2^31 < 2^62 < 2^64, so uint64 never overflows.
        hashed = (base_hashes[:, None] * self._perm_a[None, :] + self._perm_b[None, :]) % _MERSENNE_31
        return hashed.min(axis=0).astype(np.uint32)

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if sig_a.shape != sig_b.shape or sig_a.size == 0:
            raise ValueError("MinHash signatures must have matching non-empty shape")
        return float(np.mean(sig_a == sig_b))

    def _band_keys(self, sig: np.ndarray) -> List[int]:
        keys: List[int] = []
        r = self.rows_per_band
        for b_idx in range(self.bands):
            band_bytes = sig[b_idx * r : (b_idx + 1) * r].tobytes()
            keys.append(xxhash.xxh64_intdigest(band_bytes, seed=self.seed + b_idx + 1))
        return keys

    def is_duplicate(self, text: str) -> Tuple[bool, Optional[int], float]:
        """Check whether text duplicates any prior streamed document and update index."""
        idx = self._stream_count
        self._stream_count += 1
        sha_hex, xxh_hex = self.exact_hash(text)
        exact_key = f"{sha_hex}:{xxh_hex}"
        if exact_key in self._seen_exact:
            return True, self._seen_exact[exact_key], 1.0

        sig = self.compute_minhash(text)
        band_keys = self._band_keys(sig)
        candidates: Set[int] = set()
        for b_idx, b_key in enumerate(band_keys):
            candidates.update(self._lsh_buckets[b_idx].get(b_key, ()))

        best_match: Optional[int] = None
        best_sim = 0.0
        for cand_idx in sorted(candidates):
            sim = self.estimate_jaccard(sig, self._seen_signatures[cand_idx])
            if sim >= self.jaccard_threshold and sim > best_sim:
                best_sim = sim
                best_match = cand_idx

        self._seen_exact[exact_key] = idx
        self._seen_signatures.append(sig)
        for b_idx, b_key in enumerate(band_keys):
            self._lsh_buckets[b_idx][b_key].append(idx)

        return (best_match is not None), best_match, best_sim

    @staticmethod
    def _build_dedup_result(
        n: int,
        uf: _UnionFind,
        pairs: List[Tuple[int, int, float]],
        stats: Dict[str, Any],
    ) -> DedupResult:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)

        kept_indices = sorted(groups.keys())
        kept_set = set(kept_indices)
        keep_mask = [i in kept_set for i in range(n)]
        duplicate_indices = [i for i in range(n) if i not in kept_set]
        duplicate_clusters = sorted(
            [sorted(members) for members in groups.values() if len(members) > 1],
            key=lambda g: g[0],
        )
        return DedupResult(
            kept_indices=kept_indices,
            duplicate_indices=duplicate_indices,
            duplicate_clusters=duplicate_clusters,
            duplicate_pairs=pairs,
            keep_mask=keep_mask,
            method_stats=stats,
        )

    def deduplicate_exact(self, texts: Sequence[str], *, lowercase: bool = True) -> DedupResult:
        """Remove exact duplicates after Unicode and whitespace normalization."""
        n = len(texts)
        uf = _UnionFind(n)
        seen: Dict[str, int] = {}
        pairs: List[Tuple[int, int, float]] = []

        for idx, text in enumerate(texts):
            sha_hex, xxh_hex = self.exact_hash(text, lowercase=lowercase)
            key = f"{sha_hex}:{xxh_hex}"
            first = seen.get(key)
            if first is not None:
                uf.union(first, idx)
                pairs.append((first, idx, 1.0))
            else:
                seen[key] = idx

        return self._build_dedup_result(
            n,
            uf,
            pairs,
            {"mode": "exact", "total": n, "unique": len(seen), "duplicates": n - len(seen)},
        )

    def deduplicate_fuzzy(
        self,
        texts: Sequence[str],
        *,
        threshold: Optional[float] = None,
        verify_exact_jaccard: bool = False,
    ) -> DedupResult:
        """Deduplicate near-duplicates via MinHash + LSH bands and connected components."""
        n = len(texts)
        if n == 0:
            return DedupResult([], [], [], [], [], {"mode": "fuzzy", "total": 0, "unique": 0, "duplicates": 0})

        thresh = self.jaccard_threshold if threshold is None else float(threshold)
        uf = _UnionFind(n)
        signatures: List[np.ndarray] = [self.compute_minhash(t) for t in texts]
        buckets: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(self.bands)]
        candidate_pairs: Set[Tuple[int, int]] = set()

        for idx, sig in enumerate(signatures):
            keys = self._band_keys(sig)
            for b_idx, b_key in enumerate(keys):
                bucket = buckets[b_idx][b_key]
                for prev in bucket:
                    candidate_pairs.add((prev, idx) if prev < idx else (idx, prev))
                bucket.append(idx)

        duplicate_pairs: List[Tuple[int, int, float]] = []
        for i, j in sorted(candidate_pairs):
            est = self.estimate_jaccard(signatures[i], signatures[j])
            if est >= thresh:
                sim = self.exact_shingle_jaccard(texts[i], texts[j]) if verify_exact_jaccard else est
                if sim >= thresh:
                    uf.union(i, j)
                    duplicate_pairs.append((i, j, float(sim)))

        res = self._build_dedup_result(
            n,
            uf,
            duplicate_pairs,
            {
                "mode": "fuzzy",
                "total": n,
                "candidate_pairs": len(candidate_pairs),
                "matched_pairs": len(duplicate_pairs),
                "num_perm": self.num_perm,
                "bands": self.bands,
                "rows_per_band": self.rows_per_band,
            },
        )
        res.method_stats["unique"] = len(res.kept_indices)
        res.method_stats["duplicates"] = len(res.duplicate_indices)
        return res

    def deduplicate_semantic(
        self,
        inputs: Union[Sequence[str], torch.Tensor, np.ndarray],
        *,
        threshold: Optional[float] = None,
        batch_size: int = 256,
    ) -> DedupResult:
        """Semantic cosine deduplication using PyTorch matrix ops on CPU or Apple MPS."""
        thresh = self.semantic_threshold if threshold is None else float(threshold)

        if isinstance(inputs, torch.Tensor):
            emb = inputs.detach().to(device=self.device, dtype=torch.float32)
        elif isinstance(inputs, np.ndarray):
            emb = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)).to(self.device)
        else:
            texts = [normalize_for_dedup(t, lowercase=True) for t in inputs]
            if not texts:
                return DedupResult([], [], [], [], [], {"mode": "semantic", "total": 0, "unique": 0, "duplicates": 0})
            if len(texts) == 1:
                return DedupResult([0], [], [], [], [True], {"mode": "semantic", "total": 1, "unique": 1, "duplicates": 0})
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, norm="l2", sublinear_tf=True)
            mat = vectorizer.fit_transform(texts).astype(np.float32).toarray()
            emb = torch.from_numpy(mat).to(self.device)

        n = int(emb.shape[0])
        if n == 0:
            return DedupResult([], [], [], [], [], {"mode": "semantic", "total": 0, "unique": 0, "duplicates": 0})

        norms = torch.linalg.norm(emb, dim=1, keepdim=True).clamp(min=1e-12)
        normed = emb / norms

        uf = _UnionFind(n)
        pairs: List[Tuple[int, int, float]] = []
        bs = max(1, int(batch_size))

        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(n, start + bs)
                sim_block = torch.matmul(normed[start:end], normed[:end].T)
                # Keep only strict lower-triangle entries (row > col) to prevent self/double matching
                rows = torch.arange(start, end, device=self.device).unsqueeze(1)
                cols = torch.arange(0, end, device=self.device).unsqueeze(0)
                mask = (rows > cols) & (sim_block >= thresh)
                hits = torch.nonzero(mask, as_tuple=False).cpu().tolist()
                if hits:
                    sims_cpu = sim_block.detach().cpu()
                    for local_r, col in hits:
                        global_r = start + int(local_r)
                        col_i = int(col)
                        score = float(sims_cpu[local_r, col_i].item())
                        uf.union(col_i, global_r)
                        pairs.append((col_i, global_r, score))

        res = self._build_dedup_result(
            n,
            uf,
            pairs,
            {
                "mode": "semantic",
                "device": str(self.device),
                "total": n,
                "matched_pairs": len(pairs),
            },
        )
        res.method_stats["unique"] = len(res.kept_indices)
        res.method_stats["duplicates"] = len(res.duplicate_indices)
        return res

    def deduplicate(
        self,
        texts: Sequence[str],
        *,
        mode: str = "all",
    ) -> DedupResult:
        """Run exact, fuzzy, semantic, or combined ('all') deduplication on texts."""
        mode_norm = mode.strip().lower()
        if mode_norm == "exact":
            return self.deduplicate_exact(texts)
        if mode_norm == "fuzzy":
            return self.deduplicate_fuzzy(texts)
        if mode_norm == "semantic":
            return self.deduplicate_semantic(texts)
        if mode_norm != "all":
            raise ValueError(f"Unknown deduplication mode: {mode!r}")

        n = len(texts)
        if n == 0:
            return DedupResult([], [], [], [], [], {"mode": "all", "total": 0, "unique": 0, "duplicates": 0})

        uf = _UnionFind(n)
        pairs: List[Tuple[int, int, float]] = []

        exact_res = self.deduplicate_exact(texts)
        for a, b, s in exact_res.duplicate_pairs:
            uf.union(a, b)
            pairs.append((a, b, s))

        # Run fuzzy + semantic only over surviving exact-unique representatives for efficiency
        reps = exact_res.kept_indices
        rep_texts = [texts[i] for i in reps]
        fuzzy_res = self.deduplicate_fuzzy(rep_texts)
        for a_loc, b_loc, s in fuzzy_res.duplicate_pairs:
            a, b = reps[a_loc], reps[b_loc]
            uf.union(a, b)
            pairs.append((a, b, s))

        fuzzy_reps = [reps[i] for i in fuzzy_res.kept_indices]
        sem_res = self.deduplicate_semantic([texts[i] for i in fuzzy_reps])
        for a_loc, b_loc, s in sem_res.duplicate_pairs:
            a, b = fuzzy_reps[a_loc], fuzzy_reps[b_loc]
            uf.union(a, b)
            pairs.append((a, b, s))

        res = self._build_dedup_result(
            n,
            uf,
            pairs,
            {
                "mode": "all",
                "total": n,
                "exact_duplicates": len(exact_res.duplicate_indices),
                "fuzzy_duplicates": len(fuzzy_res.duplicate_indices),
                "semantic_duplicates": len(sem_res.duplicate_indices),
            },
        )
        res.method_stats["unique"] = len(res.kept_indices)
        res.method_stats["duplicates"] = len(res.duplicate_indices)
        return res


@dataclass(frozen=True)
class ReconstructedWindow:
    """Sliding cross-sample context window reassembling split attack fragments."""

    current_index: int
    window_indices: Tuple[int, ...]
    space_joined: str
    boundary_joined: str
    candidates: Tuple[str, ...]


class CrossSampleReconstructor:
    """Bounded sliding-window reassembler defeating cross-sample payload splitting."""

    def __init__(self, window_size: int = 4, *, max_chunk_chars: int = 4096) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = int(window_size)
        self.max_chunk_chars = int(max_chunk_chars)
        self._buffer: Deque[Tuple[int, str]] = deque(maxlen=self.window_size)
        self._seen_count: int = 0

    def reset(self) -> None:
        """Clear sliding window state between independent streams."""
        self._buffer.clear()
        self._seen_count = 0

    @staticmethod
    def _smart_join(chunks: Sequence[str]) -> str:
        """Join fragments while healing mid-word splits (alnum -> alnum) cleanly."""
        if not chunks:
            return ""
        parts: List[str] = [chunks[0]]
        for nxt in chunks[1:]:
            if not nxt:
                continue
            prev = parts[-1]
            if prev and prev[-1].isalnum() and nxt[0].isalnum():
                parts.append(nxt)
            else:
                parts.append(" " + nxt)
        return _WHITESPACE_RE.sub(" ", "".join(parts)).strip()

    def push(self, text: str, sample_index: Optional[int] = None) -> ReconstructedWindow:
        """Append a sample chunk and emit reassembled sliding suffixes."""
        idx = self._seen_count if sample_index is None else int(sample_index)
        self._seen_count += 1
        snippet = normalize_for_dedup(text, lowercase=False)[: self.max_chunk_chars]
        self._buffer.append((idx, snippet))

        items = list(self._buffer)
        win_indices = tuple(i for i, _ in items)
        chunks = [c for _, c in items if c]

        space_joined = " ".join(chunks).strip()
        boundary_joined = self._smart_join(chunks)

        candidates: List[str] = []
        seen: Set[str] = set()
        # Generate multi-chunk suffix views (k = 2 .. len(chunks)) using both space & smart joins
        if len(chunks) >= 2:
            for k in range(2, len(chunks) + 1):
                sub = chunks[-k:]
                for joined in (self._smart_join(sub), " ".join(sub).strip()):
                    if joined and joined not in seen:
                        seen.add(joined)
                        candidates.append(joined)

        return ReconstructedWindow(
            current_index=idx,
            window_indices=win_indices,
            space_joined=space_joined,
            boundary_joined=boundary_joined,
            candidates=tuple(candidates),
        )

    def reassemble_batch(self, texts: Sequence[str]) -> List[ReconstructedWindow]:
        """Stream a sequence of text samples through the sliding window."""
        return [self.push(t, sample_index=i) for i, t in enumerate(texts)]


__all__ = [
    "CrossSampleReconstructor",
    "DedupResult",
    "PortableDeduplicator",
    "ReconstructedWindow",
    "normalize_for_dedup",
    "resolve_portable_device",
]
