"""Continuous Integration Testing Suite (TorchBench / BackendBench Workload Matrix).

Executes real-world model architectures across all 6 workload families:
1. Modern LLM Decoder (LLaMA/Qwen-style with SwiGLU/GeGLU MLPs)
2. Classic Transformer & ViT Encoder (BERT/ViT with LayerNorm)
3. Computer Vision CNN (ResNet/ConvNeXt with Conv + PackedReLU)
4. Tabular & Narrow Neural ML (Tabular MLP)
5. Parameter-Efficient LoRA Backbone
6. Real 1B+ Transformer LLM Architecture (Attention + SwiGLU MLP + Fused Linear Cross-Entropy)

Performs live forward and backward passes, verifies numerical parity with eager PyTorch
using BackendBench evaluation standards, measures speedup and memory retention,
and writes the authoritative reports/OJASX_V05_TEST_DATABASE_MATRIX.json.
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import sys
import time
import types
import math
import gc
import hashlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Compatibility shims for Python 3.9 typing in BackendBench
import typing
if not hasattr(typing._CallableType, "__or__"):
    typing._CallableType.__or__ = lambda self, other: typing.Union[self, other]
    typing._CallableType.__ror__ = lambda self, other: typing.Union[self, other]

# Stub LLM kernel generator if optional packages are not installed
if "BackendBench.backends" not in sys.modules:
    sys.modules["BackendBench.backends"] = types.ModuleType("BackendBench.backends")
if "BackendBench.backends.directory" not in sys.modules:
    mod_dir = types.ModuleType("BackendBench.backends.directory")
    mod_dir.DirectoryBackend = type("DirectoryBackend", (), {})
    sys.modules["BackendBench.backends.directory"] = mod_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKENDBENCH_DIR = REPO_ROOT / "third_party" / "BackendBench"
if str(BACKENDBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BACKENDBENCH_DIR))

from BackendBench.eval import allclose, compute_errors, CorrectnessTestResult

# OjasX imports
import ojasx2
from ojasx2.smart_router import optimize_model
from ojasx2.tile_hal import probe_hardware_profile


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUN_1B = os.environ.get("OJASX_RUN_1B", "0").strip().lower() in {"1", "true", "yes", "on"}
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORTS_DIR / "OJASX_V05_TEST_DATABASE_MATRIX.json"


# ============================================================================
# Autograd Memory & Timing Instrumentation
# ============================================================================

class SavedTapeAuditor:
    """Accurately records all activation tensors and bytes retained on the autograd tape (excluding static parameters)."""

    def __init__(self, exclude_params: List[torch.Tensor] = None):
        self.tensors: List[torch.Tensor] = []
        self.exclude_ptrs = {
            p.untyped_storage().data_ptr() for p in (exclude_params or []) if isinstance(p, torch.Tensor)
        }

    def pack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.untyped_storage().data_ptr() not in self.exclude_ptrs:
            self.tensors.append(tensor)
        return tensor

    def unpack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def total_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def tensor_evidence(self) -> List[Dict[str, Any]]:
        return [
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "bytes": tensor.numel() * tensor.element_size(),
            }
            for tensor in self.tensors
        ]


def sync_device():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def time_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    runs: int = 15,
    warmup: int = 3,
    targets: Optional[torch.Tensor] = None,
    is_opt: bool = False,
) -> float:
    """Measures steady-state latency (ms) for forward + backward pass."""
    # Warm-up
    for _ in range(warmup):
        if targets is not None:
            if is_opt:
                loss = model(x, targets=targets)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        else:
            out = model(x)
            loss = out.sum()
        loss.backward()
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    sync_device()
    start = time.perf_counter()
    for _ in range(runs):
        if targets is not None:
            if is_opt:
                loss = model(x, targets=targets)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        else:
            out = model(x)
            loss = out.sum()
        loss.backward()
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
    sync_device()
    elapsed = (time.perf_counter() - start) / runs
    return elapsed * 1000.0


def compute_grad_cosine_similarity(m_ref: nn.Module, m_opt: nn.Module) -> float:
    """Compare complete gradients by name or shared parameter identity.

    Missing, extra, shape-mismatched, and zero-norm gradients are errors, not
    perfect matches. Per-parameter products avoid one combined gradient vector.
    """
    reference = {name: parameter for name, parameter in m_ref.named_parameters() if parameter.requires_grad}
    optimized = {name: parameter for name, parameter in m_opt.named_parameters() if parameter.requires_grad}
    optimized_by_identity = {}
    for name, parameter in optimized.items():
        optimized_by_identity.setdefault(id(parameter), []).append(name)
    if any(len(names) != 1 for names in optimized_by_identity.values()):
        raise ValueError("Optimized model has ambiguous shared parameter identities")

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    matched_optimized = set()
    for name, p1 in reference.items():
        if name in optimized:
            matched_name = name
        else:
            identity_matches = optimized_by_identity.get(id(p1), [])
            if len(identity_matches) != 1:
                raise ValueError(f"Missing optimized parameter for reference parameter {name!r}")
            matched_name = identity_matches[0]
        if matched_name in matched_optimized:
            raise ValueError(f"Optimized parameter {matched_name!r} was matched more than once")
        matched_optimized.add(matched_name)
        p2 = optimized[matched_name]
        if p1.shape != p2.shape:
            raise ValueError(f"Gradient shape mismatch for {name!r}: {p1.shape} versus {p2.shape}")
        if p1.grad is None or p2.grad is None:
            raise ValueError(f"Missing gradient for parameter {name!r}")
        g1 = p1.grad.reshape(-1)
        g2 = p2.grad.reshape(-1)
        if g1.shape != g2.shape:
            raise ValueError(f"Gradient storage shape mismatch for parameter {name!r}")
        dot += (g1 * g2).sum().item()
        norm1 += (g1 * g1).sum().item()
        norm2 += (g2 * g2).sum().item()

    unmatched = set(optimized) - matched_optimized
    if unmatched:
        raise ValueError(f"Optimized model has unmatched trainable parameters: {sorted(unmatched)}")
    if not reference:
        raise ValueError("No trainable reference parameters were available for gradient comparison")
    if norm1 <= 0.0 or norm2 <= 0.0:
        raise ValueError("Cosine similarity is undefined for a zero-norm gradient")
    similarity = dot / (math.sqrt(norm1) * math.sqrt(norm2))
    if not math.isfinite(similarity):
        raise ValueError("Non-finite gradient cosine similarity")
    return float(similarity)


# ============================================================================
# Workload Model Architectures
# ============================================================================

class LLaMASwiGLUMLP(nn.Module):
    """Modern LLM non-Sequential MLP submodule."""

    def __init__(self, d: int = 128, h: int = 256):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ModernLLMDecoderBlock(nn.Module):
    """Workload Family 1: Modern LLM Decoder (LLaMA/Qwen-style with SwiGLU MLP)."""

    def __init__(self, d: int = 128, h: int = 256):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.mlp = LLaMASwiGLUMLP(d, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class ClassicTransformerEncoderBlock(nn.Module):
    """Workload Family 2: Classic Transformer & ViT Encoder (BERT/ViT with LayerNorm)."""

    def __init__(self, d: int = 128, h: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.fc2(self.act(self.fc1(h)))


class ComputerVisionCNNBlock(nn.Module):
    """Workload Family 3: Computer Vision CNN (ResNet/ConvNeXt with Conv + PackedReLU)."""

    def __init__(self, channels: int = 32, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        # Frozen feature extractor backbone
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
        self.act = nn.ReLU()
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(x))
        return self.head(h.mean(dim=[-1, -2]))


class TabularNarrowMLPBlock(nn.Module):
    """Workload Family 4: Tabular & Narrow Neural ML."""

    def __init__(self, in_features: int = 32, hidden: int = 64, out_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.out(self.act(self.fc2(h)))


class LoRABackboneBlock(nn.Module):
    """Workload Family 5: Parameter-Efficient LoRA Backbone."""

    def __init__(self, in_features: int = 128, out_features: int = 128, r: int = 8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad = False
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * 0.5


class LlamaAttentionBlock(nn.Module):
    def __init__(self, d: int = 2048, num_heads: int = 16, is_causal: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.is_causal:
            mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(b, s, d)
        return self.o_proj(out)


class LlamaMLPBlock(nn.Module):
    def __init__(self, d: int = 2048, h: int = 5504):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class WholeModel1BTransformerLLM(nn.Module):
    """Workload Family 6: Real 1B+ Parameter LLM Architecture (LLaMA/Qwen-style)."""

    def __init__(
        self,
        d: int = 2048,
        h: int = 5504,
        num_layers: int = 18,
        num_heads: int = 16,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.norm_in = nn.RMSNorm(d, eps=1e-5)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": LlamaAttentionBlock(d, num_heads, is_causal=True),
                "mlp": LlamaMLPBlock(d, h),
            })
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(d, eps=1e-5)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm_in(x)
        for layer in self.layers:
            x = x + layer["attn"](x)
            x = x + layer["mlp"](x)
        h = self.final_norm(x)
        if targets is not None:
            try:
                return self.lm_head(h, targets)
            except TypeError:
                logits = self.lm_head(h)
                return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return self.lm_head(h)


# ============================================================================
# Benchmark Runner Helper
# ============================================================================

def run_workload_benchmark(
    name: str,
    eager_factory: Any,
    input_shape: Tuple[int, ...],
    router_kwargs: Dict[str, Any] = None,
    targets_factory: Optional[Callable] = None,
    runs: int = 15,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Executes eager vs OjasX optimized runs and extracts all performance and memory metrics."""
    router_kwargs = router_kwargs or {}

    # Initialize models with exact identical weights without deepcopy overhead
    torch.manual_seed(42)
    m_eager = eager_factory().to(DEVICE)
    torch.manual_seed(42)
    m_opt = eager_factory().to(DEVICE)
    m_opt, report = optimize_model(m_opt, **router_kwargs)

    # Prepare inputs with deterministic seed for reproducible benchmark numerical parity
    torch.manual_seed(42)
    x_eager = torch.randn(input_shape, device=DEVICE, requires_grad=True)
    x_opt = x_eager.clone().detach().requires_grad_(True)

    if targets_factory is not None:
        targets_eager = targets_factory().to(DEVICE)
        targets_opt = targets_eager.clone().detach()

        auditor_eager = SavedTapeAuditor(list(m_eager.parameters()))
        with torch.autograd.graph.saved_tensors_hooks(auditor_eager.pack_hook, auditor_eager.unpack_hook):
            logits_eager = m_eager(x_eager)
            loss_eager = F.cross_entropy(logits_eager.view(-1, logits_eager.shape[-1]), targets_eager.view(-1))
        eager_tape_bytes = auditor_eager.total_bytes

        auditor_opt = SavedTapeAuditor(list(m_opt.parameters()))
        with torch.autograd.graph.saved_tensors_hooks(auditor_opt.pack_hook, auditor_opt.unpack_hook):
            loss_opt = m_opt(x_opt, targets=targets_opt)
        opt_tape_bytes = auditor_opt.total_bytes

        tape_reduction_pct = (
            (1.0 - (opt_tape_bytes / eager_tape_bytes)) * 100.0 if eager_tape_bytes > 0 else 0.0
        )

        loss_eager.backward()
        loss_opt.backward()

        grad_cos_sim = compute_grad_cosine_similarity(m_eager, m_opt)
        abs_err, rel_err = compute_errors(loss_eager, loss_opt)
        is_close = allclose(loss_eager, loss_opt, atol=1e-3, rtol=1e-3)

        eager_ms = time_forward_backward(m_eager, x_eager, runs=runs, warmup=warmup, targets=targets_eager, is_opt=False)
        opt_ms = time_forward_backward(m_opt, x_opt, runs=runs, warmup=warmup, targets=targets_opt, is_opt=True)
    else:
        # 1. Measure autograd tape memory (excluding static model parameters)
        auditor_eager = SavedTapeAuditor(list(m_eager.parameters()))
        with torch.autograd.graph.saved_tensors_hooks(auditor_eager.pack_hook, auditor_eager.unpack_hook):
            y_eager = m_eager(x_eager)
        eager_tape_bytes = auditor_eager.total_bytes

        auditor_opt = SavedTapeAuditor(list(m_opt.parameters()))
        with torch.autograd.graph.saved_tensors_hooks(auditor_opt.pack_hook, auditor_opt.unpack_hook):
            y_opt = m_opt(x_opt)
        opt_tape_bytes = auditor_opt.total_bytes

        tape_reduction_pct = (
            (1.0 - (opt_tape_bytes / eager_tape_bytes)) * 100.0 if eager_tape_bytes > 0 else 0.0
        )

        # 2. Backward passes and gradient cosine similarity
        loss_eager = y_eager.sum()
        loss_eager.backward()

        loss_opt = y_opt.sum()
        loss_opt.backward()

        grad_cos_sim = compute_grad_cosine_similarity(m_eager, m_opt)

        # 3. Numerical parity checks via BackendBench eval
        abs_err, rel_err = compute_errors(y_eager, y_opt)
        is_close = allclose(y_eager, y_opt, atol=1e-3, rtol=1e-3)

        # 4. Latency measurement
        eager_ms = time_forward_backward(m_eager, x_eager, runs=runs, warmup=warmup)
        opt_ms = time_forward_backward(m_opt, x_opt, runs=runs, warmup=warmup)

    speedup_x = round(eager_ms / opt_ms, 3) if opt_ms > 0 else 1.0

    res = {
        "workload_name": name,
        "input_shape": list(input_shape),
        "torch_version": torch.__version__,
        "device": DEVICE,
        "eager_forward_backward_ms": round(eager_ms, 3),
        "ojasx_forward_backward_ms": round(opt_ms, 3),
        "speedup_x": speedup_x,
        "eager_tape_bytes": eager_tape_bytes,
        "ojasx_tape_bytes": opt_tape_bytes,
        "eager_saved_tensors": auditor_eager.tensor_evidence,
        "ojasx_saved_tensors": auditor_opt.tensor_evidence,
        "tape_reduction_pct": round(tape_reduction_pct, 2),
        "max_abs_error": float(abs_err),
        "max_rel_error": float(rel_err),
        "gradient_cosine_similarity": round(grad_cos_sim, 6),
        "backendbench_correctness_pass": bool(is_close),
        "detected_profile": report.workload_profile,
        "optimizations_applied": list(report.action_counts.keys()),
    }

    del m_eager, m_opt, x_eager, x_opt
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return res


# Global in-memory storage for test matrix aggregation
_WORKLOAD_RESULTS: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# PyTest Benchmark Tests Across All 5 Workload Families
# ============================================================================

def test_backendbench_workload_modern_llm():
    """Workload 1: Modern LLM Decoder (LLaMA/Qwen SwiGLU MLP)."""
    res = run_workload_benchmark(
        name="Modern_LLM_Decoder_SwiGLU",
        eager_factory=lambda: ModernLLMDecoderBlock(d=128, h=256),
        input_shape=(4, 32, 128),
    )
    _WORKLOAD_RESULTS["family_1_modern_llm"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.999
    assert res["tape_reduction_pct"] > 40.0


def test_backendbench_workload_classic_transformer_vit():
    """Workload 2: Classic Transformer & ViT Encoder (BERT/ViT LayerNorm)."""
    res = run_workload_benchmark(
        name="Classic_Transformer_ViT_Encoder",
        eager_factory=lambda: ClassicTransformerEncoderBlock(d=128, h=256),
        input_shape=(4, 32, 128),
    )
    _WORKLOAD_RESULTS["family_2_classic_transformer"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.999
    assert res["tape_reduction_pct"] > 15.0


def test_backendbench_workload_computer_vision_cnn():
    """Workload 3: Computer Vision CNN (ResNet/ConvNeXt Conv + PackedReLU)."""
    res = run_workload_benchmark(
        name="Computer_Vision_CNN_PackedReLU",
        eager_factory=lambda: ComputerVisionCNNBlock(channels=32, num_classes=10),
        input_shape=(8, 32, 16, 16),
        router_kwargs={"prefer_1bit_relu2_for_relu": True},
    )
    _WORKLOAD_RESULTS["family_3_computer_vision_cnn"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.999
    assert res["tape_reduction_pct"] > 10.0


def test_backendbench_workload_tabular_narrow_ml():
    """Workload 4: Tabular & Narrow Neural ML."""
    res = run_workload_benchmark(
        name="Tabular_Narrow_MLP",
        eager_factory=lambda: TabularNarrowMLPBlock(in_features=32, hidden=64, out_features=4),
        input_shape=(32, 32),
        router_kwargs={"prefer_1bit_relu2_for_relu": True},
    )
    _WORKLOAD_RESULTS["family_4_tabular_narrow_ml"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.999


def test_backendbench_workload_lora_backbone():
    """Workload 5: Parameter-Efficient LoRA Backbone."""
    res = run_workload_benchmark(
        name="Parameter_Efficient_LoRA_Backbone",
        eager_factory=lambda: LoRABackboneBlock(in_features=128, out_features=128, r=8),
        input_shape=(4, 32, 128),
    )
    _WORKLOAD_RESULTS["family_5_lora_backbone"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.999
    assert res["eager_tape_bytes"] >= 0 and res["ojasx_tape_bytes"] >= 0
    assert res["eager_saved_tensors"] and res["ojasx_saved_tensors"]


@pytest.mark.skipif(
    not RUN_1B,
    reason="1B forward/backward qualification is memory-intensive; opt in with OJASX_RUN_1B=1 on a qualified host",
)
def test_backendbench_workload_1b_transformer_llm():
    """Opt-in workload Family 6: real 1B+ transformer forward/backward qualification."""
    res = run_workload_benchmark(
        name="Family_6_1B_Transformer_LLM",
        eager_factory=lambda: WholeModel1BTransformerLLM(d=2048, h=5504, num_layers=18, num_heads=16, vocab_size=32000),
        input_shape=(1, 512, 2048),
        router_kwargs={"prefer_unified_mlp": True},
        targets_factory=lambda: torch.randint(0, 32000, (1, 512), dtype=torch.long),
        runs=3,
        warmup=1,
    )
    _WORKLOAD_RESULTS["family_6_1b_transformer_llm"] = res
    assert res["backendbench_correctness_pass"]
    assert res["gradient_cosine_similarity"] >= 0.995
    assert res["tape_reduction_pct"] >= 70.0, f"Expected >= 70% tape reduction, got {res['tape_reduction_pct']}%"


_REQUIRED_WORKLOADS = {
    "family_1_modern_llm": {"gradient_min": 0.999, "tape_reduction_min": 40.0},
    "family_2_classic_transformer": {"gradient_min": 0.999, "tape_reduction_min": 15.0},
    "family_3_computer_vision_cnn": {"gradient_min": 0.999, "tape_reduction_min": 10.0},
    "family_4_tabular_narrow_ml": {"gradient_min": 0.999, "tape_reduction_min": None},
    # LoRA tape retention is Torch-version-dependent. Measure it, but do not
    # assert a universal reduction percentage on every supported Torch release.
    "family_5_lora_backbone": {"gradient_min": 0.999, "tape_reduction_min": None},
    "family_6_1b_transformer_llm": {"gradient_min": 0.995, "tape_reduction_min": 70.0},
}


def _derive_matrix_status(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    checks = {}
    for family, requirements in _REQUIRED_WORKLOADS.items():
        result = results.get(family)
        if result is None:
            checks[family] = {"executed": False, "passed": False}
            continue
        correctness = bool(result.get("backendbench_correctness_pass"))
        gradient = result.get("gradient_cosine_similarity", -1.0) >= requirements["gradient_min"]
        memory_measurement = (
            result.get("eager_tape_bytes", -1) >= 0
            and result.get("ojasx_tape_bytes", -1) >= 0
            and "eager_saved_tensors" in result
            and "ojasx_saved_tensors" in result
        )
        threshold = requirements["tape_reduction_min"]
        memory_threshold = threshold is None or result.get("tape_reduction_pct", -1.0) >= threshold
        checks[family] = {
            "executed": True,
            "correctness_pass": correctness,
            "gradient_pass": gradient,
            "memory_measurement_complete": memory_measurement,
            "memory_reduction_threshold": threshold,
            "memory_threshold_pass": memory_threshold,
            "observed_tape_reduction_pct": result.get("tape_reduction_pct"),
            "passed": correctness and gradient and memory_measurement and memory_threshold,
        }
    executed = [check for check in checks.values() if check["executed"]]
    passed_count = sum(check["passed"] for check in checks.values())
    if passed_count == len(checks):
        status = "passed"
    elif any(check["executed"] and not check["passed"] for check in checks.values()):
        status = "failed"
    else:
        status = "incomplete"
    return {
        "status": status,
        "families": checks,
        "pass_rate_pct": 100.0 * passed_count / len(checks),
        "backward_gradient_parity_verified": all(
            check["executed"] and check["gradient_pass"] for check in checks.values()
        ),
        # This covers only thresholds that the runner actually defines. LoRA
        # has an observation, not a cross-version memory-reduction guarantee.
        "memory_tape_reduction_verified": all(
            check["executed"] and check["memory_threshold_pass"]
            for check in checks.values()
        ),
        "executed_workload_count": len(executed),
    }


def test_backendbench_generate_v05_matrix_report():
    """Write a report derived solely from workload results gathered by this runner."""
    derived = _derive_matrix_status(_WORKLOAD_RESULTS)
    silicon = probe_hardware_profile()
    source_paths = [
        REPO_ROOT / "tests" / "test_backendbench_matrix.py",
        REPO_ROOT / "ojasx2" / "smart_router.py",
        REPO_ROOT / "ojasx2" / "holonomic_math_compiler.py",
    ]
    matrix_data = {
        "schema_version": 2,
        "status": derived["status"],
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": "0.5.0",
        "device": DEVICE,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "silicon_capabilities": {
            "chip_class": silicon.chip_class,
            "has_apple_amx": silicon.has_apple_amx,
            "has_mps_gpu": silicon.has_mps_gpu,
            "unified_memory": silicon.unified_memory,
            "physical_cpu_cores": silicon.physical_cpu_cores,
            "l1d_cache_bytes": silicon.l1d_cache_bytes,
            "l2_cache_bytes": silicon.l2_cache_bytes,
            "cache_line_bytes": silicon.cache_line_bytes,
        },
        "verification_mechanisms": {
            "mechanism_1_torch_library_opcheck": {
                "file": "tests/test_opcheck_compliance.py",
                "status": "not_executed_by_this_runner",
                "evidence_required": "Execute that module separately and attach its actual result.",
            },
            "mechanism_2_opinfo_multi_hardware_matrix": {
                "file": "tests/test_opinfo_compliance.py",
                "status": "not_executed_by_this_runner",
                "evidence_required": "Execute that module separately and attach its actual device/dtype result.",
            },
            "mechanism_3_backendbench_workload_matrix": {
                "file": "tests/test_backendbench_matrix.py",
                "framework": "PyTorch workload models evaluated with BackendBench numerical helpers",
                "workload_families_tested": derived["executed_workload_count"],
                "workload_families_required": len(_REQUIRED_WORKLOADS),
                "families": _WORKLOAD_RESULTS,
                "derived_checks": derived["families"],
                "pass_rate_pct": derived["pass_rate_pct"],
            },
        },
        "integrity_attestation": {
            "scope": "Derived from the six workload results in this file only; no opcheck/OpInfo result is inferred.",
            "backendbench_workload_matrix_verified": derived["status"] == "passed",
            "backward_gradient_parity_verified": derived["backward_gradient_parity_verified"],
            "memory_tape_reduction_verified": derived["memory_tape_reduction_verified"],
            "whole_model_1b_llm_70pct_reduction_verified": (
                derived["families"]["family_6_1b_transformer_llm"].get("passed")
                if derived["families"]["family_6_1b_transformer_llm"]["executed"]
                else None
            ),
            "lora_memory_claim": "Observed per-Torch-version measurement only; no universal percentage asserted.",
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        json.dump(matrix_data, report, indent=2)
        report.write("\n")

    assert REPORT_FILE.exists(), "Failed to generate report JSON file!"
    written = json.loads(REPORT_FILE.read_text())
    assert written["status"] == derived["status"]
    assert written["verification_mechanisms"]["mechanism_3_backendbench_workload_matrix"]["workload_families_tested"] == len(_WORKLOAD_RESULTS)
    if RUN_1B:
        assert derived["executed_workload_count"] == 6, "Opt-in qualification requires the complete six-workload matrix"
        assert derived["status"] == "passed", f"Workload matrix status was {derived['status']}"
    else:
        family_six = written["verification_mechanisms"]["mechanism_3_backendbench_workload_matrix"]["derived_checks"]["family_6_1b_transformer_llm"]
        assert family_six == {"executed": False, "passed": False}
        assert written["integrity_attestation"]["whole_model_1b_llm_70pct_reduction_verified"] is None


def _gradient_pair():
    reference = nn.Linear(2, 2, bias=False)
    optimized = nn.Linear(2, 2, bias=False)
    reference.weight.grad = torch.ones_like(reference.weight)
    optimized.weight.grad = torch.ones_like(optimized.weight)
    return reference, optimized


def test_gradient_comparator_rejects_missing_shape_extra_and_zero_evidence():
    reference, optimized = _gradient_pair()
    optimized.weight.grad = None
    with pytest.raises(ValueError, match="Missing gradient"):
        compute_grad_cosine_similarity(reference, optimized)

    reference, optimized = _gradient_pair()
    optimized.weight = nn.Parameter(torch.ones(3, 2))
    optimized.weight.grad = torch.ones_like(optimized.weight)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_grad_cosine_similarity(reference, optimized)

    reference, optimized = _gradient_pair()
    optimized.extra = nn.Parameter(torch.ones(1))
    optimized.extra.grad = torch.ones_like(optimized.extra)
    with pytest.raises(ValueError, match="unmatched trainable"):
        compute_grad_cosine_similarity(reference, optimized)

    reference, optimized = _gradient_pair()
    reference.weight.grad.zero_()
    optimized.weight.grad.zero_()
    with pytest.raises(ValueError, match="zero-norm"):
        compute_grad_cosine_similarity(reference, optimized)


def test_gradient_comparator_matches_shared_identity_and_detects_corruption():
    reference = nn.Linear(2, 2, bias=False)
    reference.weight.grad = torch.ones_like(reference.weight)

    class Renamed(nn.Module):
        def __init__(self, parameter):
            super().__init__()
            self.renamed = parameter

    optimized = Renamed(reference.weight)
    assert compute_grad_cosine_similarity(reference, optimized) == pytest.approx(1.0)

    reference, optimized = _gradient_pair()
    optimized.weight.grad = -torch.ones_like(optimized.weight)
    assert compute_grad_cosine_similarity(reference, optimized) == pytest.approx(-1.0)


def _qualified_workload_result():
    return {
        "backendbench_correctness_pass": True,
        "gradient_cosine_similarity": 1.0,
        "eager_tape_bytes": 100,
        "ojasx_tape_bytes": 0,
        "eager_saved_tensors": [{"bytes": 100}],
        "ojasx_saved_tensors": [],
        "tape_reduction_pct": 100.0,
    }


def test_report_status_is_derived_for_pass_incomplete_and_failure():
    results = {name: _qualified_workload_result() for name in _REQUIRED_WORKLOADS}
    passed = _derive_matrix_status(results)
    assert passed["status"] == "passed" and passed["pass_rate_pct"] == 100.0

    incomplete_results = dict(results)
    incomplete_results.pop("family_6_1b_transformer_llm")
    incomplete = _derive_matrix_status(incomplete_results)
    assert incomplete["status"] == "incomplete"
    assert not incomplete["backward_gradient_parity_verified"]

    failed_results = {name: value.copy() for name, value in results.items()}
    failed_results["family_2_classic_transformer"]["gradient_cosine_similarity"] = 0.0
    failed = _derive_matrix_status(failed_results)
    assert failed["status"] == "failed"
    assert failed["families"]["family_2_classic_transformer"]["passed"] is False
