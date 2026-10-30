# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# This file is part of OjasX.
#
# OjasX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OjasX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OjasX.  If not, see <https://www.gnu.org/licenses/>.

"""
tests/test_ojasx_runner.py
===========================
Comprehensive Test Suite for OjasX Standalone Inference Engine (ojasx2.runtime).

Covers:
1. Stage 1: SVD-Hadamard-E8/Leech24 Residual Decomposition & Accuracy.
2. Stage 2: Contextual Active-Row Spectral Lookahead (30%, 50%, 100% active ratios).
3. Stage 3: Precomputed 81-State Activation LUT-GEMV (multiplier-free exact equivalence).
4. Stage 4: 24D Leech Lattice (Lambda_24) Projector & 8x Compressed E8 KV-Cache.
5. End-to-End OjasXRunner execution, memory reporting, and autoregressive generation.
6. One-way architectural isolation boundary check.
"""

import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ojasx2.runtime import (
    OjasXRunner,
    OjasXRuntime,
    OjasXRunnerLinear,
    OjasXLinear,
    pack_model_runner,
    E8KVCache,
    OjasXRunnerMemoryReport,
    Leech24LatticePacker,
    conway_sloane_leech24_project,
    PrecomputedTernaryLUT,
)
from ojasx2.math.lattice import _get_ternary_e8_luts


# ==============================================================================
# 1. 24D Leech Lattice Lambda_24 Tests
# ==============================================================================

def test_conway_sloane_leech24_project_invariants():
    """Verifies that projected 24D coordinates satisfy Leech coset properties."""
    torch.manual_seed(42)
    x = torch.randn(10, 24)
    proj = conway_sloane_leech24_project(x)
    assert proj.shape == (10, 24)

    # In Leech lattice, coords are either all integers or all half-integers
    for vec in proj:
        diff_int = torch.abs(vec - torch.round(vec))
        diff_half = torch.abs((vec - 0.5) - torch.round(vec - 0.5))
        is_all_int = torch.all(diff_int < 1e-4)
        is_all_half = torch.all(diff_half < 1e-4)
        assert is_all_int or is_all_half, f"Vector {vec} is neither all-integer nor all-half-integer!"


def test_leech24_packer_roundtrip():
    """Verifies Leech24 pack and unpack roundtrip fidelity."""
    torch.manual_seed(42)
    m, n = 48, 48
    w = torch.randn(m, n)
    packed, scales, shape = Leech24LatticePacker.pack(w, group_size=24)
    assert shape == (m, n)
    assert packed.dtype == torch.uint8

    unpacked = Leech24LatticePacker.unpack(packed, scales, shape)
    assert unpacked.shape == (m, n)
    cos_sim = F.cosine_similarity(w.flatten(), unpacked.flatten(), dim=0).item()
    assert cos_sim > 0.70, f"Expected cosine similarity > 0.70, got {cos_sim:.4f}"


# ==============================================================================
# 2. Precomputed 81-State Activation LUT-GEMV Tests
# ==============================================================================

def test_precomputed_81_state_lut_exact_equivalence():
    """Verifies that multiplier-free 81-state LUT GEMV matches classical float GEMV."""
    torch.manual_seed(42)
    M = 2
    K = 32
    N = 8

    rank_to_coords, _, _ = _get_ternary_e8_luts(torch.device("cpu"))
    # Generate random 12-bit E8 codes [N, K // 8]
    w_codes = torch.randint(0, 3281, (N, K // 8), dtype=torch.int16)
    scales = torch.ones(N, K // 32, dtype=torch.float16) * 0.5

    x = torch.randn(M, K, dtype=torch.float32)

    # 1. LUT GEMV (multiplier-free)
    y_lut = PrecomputedTernaryLUT.gemv_active_rows(x, w_codes, scales, group_size=32)

    # 2. Classical Float GEMV
    coords_active = rank_to_coords[w_codes.long()].float().view(N, K)
    w_fp = coords_active * 0.5
    y_float = torch.matmul(x, w_fp.t())

    # Exact bit equivalence up to float32 arithmetic tolerance
    max_diff = torch.max(torch.abs(y_lut - y_float)).item()
    assert max_diff < 1e-4, f"LUT GEMV deviated from float GEMV: max diff = {max_diff}"


# ==============================================================================
# 3. OjasXRunnerLinear SVD + FWHT-32 + Active-Row Spectral Lookahead Tests
# ==============================================================================

def test_runner_linear_accuracy_with_svd_core():
    """Verifies that Rank-16 SVD core gives high cosine similarity on realistic weights."""
    torch.manual_seed(42)
    in_features, out_features = 128, 128
    lin = nn.Linear(in_features, out_features)

    # Synthetic realistic decaying singular values
    with torch.no_grad():
        u, _, vh = torch.linalg.svd(torch.randn(out_features, in_features), full_matrices=False)
        s = torch.exp(-torch.linspace(0, 3.0, in_features)) * 10.0
        lin.weight.copy_(u @ torch.diag(s) @ vh)

    runner_lin = OjasXRunnerLinear.from_linear(
        lin,
        bitrate=1.5,
        svd_rank=16,
        active_row_ratio=1.0,
        use_hadamard=True,
    )

    x = torch.randn(4, in_features)
    y_true = lin(x)
    y_runner = runner_lin(x)

    assert y_runner.shape == y_true.shape
    cos_sim = F.cosine_similarity(y_true.flatten(), y_runner.flatten(), dim=0).item()
    assert cos_sim > 0.90, f"Expected cos sim > 0.90 with SVD core, got {cos_sim:.4f}"


@pytest.mark.parametrize("active_ratio", [0.30, 0.50, 0.70])
def test_runner_linear_active_row_lookahead(active_ratio):
    """Verifies active-row spectral lookahead with varying pruning ratios."""
    torch.manual_seed(42)
    in_features, out_features = 64, 64
    lin = nn.Linear(in_features, out_features)

    runner_sparse = OjasXRunnerLinear.from_linear(
        lin,
        bitrate=1.5,
        svd_rank=16,
        active_row_ratio=active_ratio,
        use_hadamard=True,
    )

    x = torch.randn(2, 5, in_features)
    y_out = runner_sparse(x)
    assert y_out.shape == (2, 5, out_features)


def test_runner_linear_leech24_mode():
    """Verifies OjasXRunnerLinear with Leech24 crystal lattice packing."""
    torch.manual_seed(42)
    in_features, out_features = 48, 48
    lin = nn.Linear(in_features, out_features)

    runner_leech = OjasXRunnerLinear.from_linear(
        lin,
        svd_rank=8,
        active_row_ratio=1.0,
        lattice_type="leech24",
        group_size=24,
    )

    x = torch.randn(3, in_features)
    y = runner_leech(x)
    assert y.shape == (3, out_features)


# ==============================================================================
# 4. E8KVCache 8x Compression & Autoregressive Decoding Tests
# ==============================================================================

def test_e8_kv_cache_streaming_and_compression():
    """Verifies that E8KVCache correctly updates across autoregressive generation steps."""
    torch.manual_seed(42)
    cache = E8KVCache(bitrate=1.5, group_size=32)

    B, H, D = 1, 4, 64
    # Prefill step (prompt of 16 tokens)
    k_prompt = torch.randn(B, H, 16, D)
    v_prompt = torch.randn(B, H, 16, D)
    k_out1, v_out1 = cache.update(k_prompt, v_prompt, layer_idx=0)
    assert k_out1.shape == (B, H, 16, D)
    assert cache.get_seq_length(0) == 16

    # 3 Autoregressive decode steps (1 token at a time)
    for step in range(3):
        k_step = torch.randn(B, H, 1, D)
        v_step = torch.randn(B, H, 1, D)
        k_out, v_out = cache.update(k_step, v_step, layer_idx=0)
        expected_len = 16 + step + 1
        assert k_out.shape == (B, H, expected_len, D)
        assert v_out.shape == (B, H, expected_len, D)

    # Check compression ratio
    ratio = cache.compression_ratio()
    assert ratio >= 7.0, f"Expected KV-cache compression >= 7x, got {ratio:.2f}x"


# ==============================================================================
# 5. OjasXRunner End-to-End Model Conversion & Autoregressive Generation
# ==============================================================================

class SimpleCausalLM(nn.Module):
    def __init__(self, vocab_size=50, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        return self.lm_head(h)


def test_ojasx_runner_from_model_and_generation():
    """Verifies model conversion, memory reporting, and autoregressive generation."""
    torch.manual_seed(42)
    m = SimpleCausalLM()

    runner = OjasXRunner.from_model(
        m,
        bitrate=1.5,
        svd_rank=8,
        active_row_ratio=0.5,
        exclude_names=("lm_head",),
    )

    # Verify that layer1 and layer2 were converted to OjasXRunnerLinear
    assert isinstance(runner.model.layer1, OjasXRunnerLinear)
    assert isinstance(runner.model.layer2, OjasXRunnerLinear)
    # lm_head excluded
    assert isinstance(runner.model.lm_head, nn.Linear)

    # Memory report audit
    rep = runner.memory_report()
    assert rep.total_params > 0
    assert rep.packed_linear_params > 0
    assert rep.compression_ratio > 1.0
    summary = rep.summary()
    assert "OJASX RUNNER MEMORY REPORT" in summary

    # Autoregressive generation
    input_ids = torch.tensor([[1, 2, 3]])
    gen = runner.generate(input_ids, max_new_tokens=4, temperature=0.0)
    assert gen.shape == (1, 7)
    assert (gen[:, :3] == input_ids).all()


# ==============================================================================
# 6. Strict One-Way Architectural Isolation Boundary Test
# ==============================================================================

def test_runtime_isolation_boundary():
    """
    Verifies that ojasx2.runtime never imports from training autograd,
    loss compilers, or TorchFX modules.
    """
    import inspect
    import ojasx2.runtime.runner_linear as r_lin
    import ojasx2.runtime.kv_cache_e8 as r_kv
    import ojasx2.runtime.hf_auto_runner as r_hf
    import ojasx2.runtime.leech_and_lut_math as r_leech

    modules_to_audit = [r_lin, r_kv, r_hf, r_leech]
    forbidden_terms = [
        "torch.fx",
        "holonomic_math_compiler",
        "fx_subgraph_compiler",
        "loss_compiler",
        "fused_cross_entropy",
        "grad_free_optimizer",
    ]

    for mod in modules_to_audit:
        source = inspect.getsource(mod)
        for term in forbidden_terms:
            assert f"import {term}" not in source, f"Violation: {mod.__name__} imports {term}!"
            assert f"from {term}" not in source, f"Violation: {mod.__name__} imports from {term}!"
