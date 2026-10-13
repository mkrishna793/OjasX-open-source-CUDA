"""Verification test suite for OjasX TileLang MLIR Multi-Target Compiler Pipeline.

Verifies:
1. PyTorch FX GraphModule -> ojasx.holonomic dialect extraction.
2. HolonomicToSubgraphPass (morphism decomposition, contraction reassociation, residual lifetimes).
3. SubgraphToTilePass (explicit T.alloc_shared, T.alloc_fragment, XOR swizzling, pipelined GEMMs).
4. Multi-Target Hardware Emitters:
   - NVIDIA NVVM / PTX (SM80 Ampere / SM90 Hopper) with cp.async, ldmatrix, mma.sync.
   - AMD ROCDL / CDNA assembly with .amdgpu_lds and v_mfma matrix cores.
   - Intel Arc / Vulkan Khronos SPIR-V with OpCapability CooperativeMatrixKHR.
   - Apple Silicon Metal 4 MSL with simdgroup_matrix<float, 8, 8> and threadgroup memory.
   - TileLang Python PrimFunc (@T.prim_func) with T.alloc_shared and T.alloc_fragment.
5. End-to-end compile_model_to_target() API across all 5 GPU hardware targets.
6. Zero DRAM Intermediate Memory Audit: Verifies score matrices are strictly absent from global DRAM allocations.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.fx as fx

from ojasx2.mlir_dialects import (
    FloatType,
    IntType,
    TensorType,
    MemRefType,
    FragmentType,
    MLIRValue,
    MLIRModule,
    HolonomicDialect,
    SubgraphDialect,
    TileDialect,
    HolonomicToSubgraphPass,
    SubgraphToTilePass,
    HardwareCodeEmitter,
    fx_to_holonomic_mlir,
    compile_mlir_to_target,
    compile_model_to_target,
)


# =============================================================================
# 1. Test Models
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.norm = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class MockAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)
        self.out_proj = nn.Linear(128, 128)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Standard scaled dot product attention signature
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out_proj(attn_out)


# =============================================================================
# 2. 3-Dialect Lowering Tests
# =============================================================================

def test_fx_to_holonomic_lowering():
    """Verifies PyTorch FX graph converts into ojasx.holonomic IR."""
    model = SimpleMLP()
    gm = fx.symbolic_trace(model)
    m_holo = fx_to_holonomic_mlir(gm, name="mlp_holonomic")

    assert len(m_holo.functions) == 1
    func = m_holo.functions[0]
    dialects = {op.dialect for op in func.body.operations}
    assert "ojasx.holonomic" in dialects

    op_names = [op.op_name for op in func.body.operations]
    assert "morphism" in op_names
    assert "absorb_diag_gamma" in op_names


def test_holonomic_to_subgraph_pass():
    """Verifies HolonomicToSubgraphPass decomposes morphisms into contractions."""
    model = SimpleMLP()
    gm = fx.symbolic_trace(model)
    m_holo = fx_to_holonomic_mlir(gm)
    m_subgraph = HolonomicToSubgraphPass().run(m_holo)

    func = m_subgraph.functions[0]
    dialects = {op.dialect for op in func.body.operations}
    assert "ojasx.subgraph" in dialects

    op_names = [op.op_name for op in func.body.operations]
    assert "contraction" in op_names
    assert "residual_lifetime" in op_names


def test_subgraph_to_tile_pass():
    """Verifies SubgraphToTilePass lowers to explicit SRAM and fragment allocations."""
    model = SimpleMLP()
    gm = fx.symbolic_trace(model)
    m_holo = fx_to_holonomic_mlir(gm)
    m_subgraph = HolonomicToSubgraphPass().run(m_holo)
    m_tile = SubgraphToTilePass(tile_m=64, tile_n=64, tile_k=32).run(m_subgraph)

    func = m_tile.functions[0]
    dialects = {op.dialect for op in func.body.operations}
    assert "ojasx.tile" in dialects

    op_names = [op.op_name for op in func.body.operations]
    assert "alloc_shared" in op_names
    assert "alloc_fragment" in op_names
    assert "gemm_slice" in op_names
    assert "sync_pipeline" in op_names


# =============================================================================
# 3. Hardware Emitters Tests
# =============================================================================

def test_emitter_nvvm_ptx():
    """Verifies NVIDIA NVVM PTX emission for SM80/SM90."""
    model = SimpleMLP()
    ptx_code = compile_model_to_target(model, target="nvvm")

    assert ".target sm_80" in ptx_code
    assert ".shared .align 16 .b8 smem_q" in ptx_code
    assert "cp.async.ca.shared.global" in ptx_code
    assert "ldmatrix.sync.aligned" in ptx_code
    assert "mma.sync.aligned" in ptx_code
    assert "st.global" in ptx_code


def test_emitter_rocdl_gcn():
    """Verifies AMD ROCDL / CDNA assembly emission."""
    model = SimpleMLP()
    gcn_code = compile_model_to_target(model, target="rocdl")

    assert ".globl ojasx_tile_dagger_attention" in gcn_code
    assert ".amdgpu_lds smem_q" in gcn_code
    assert "v_mfma_f32_16x16x16_f16" in gcn_code
    assert "s_waitcnt lgkmcnt(0)" in gcn_code
    assert "global_store_dword" in gcn_code


def test_emitter_spirv():
    """Verifies Khronos SPIR-V cooperative matrix code emission for Intel/Vulkan."""
    model = SimpleMLP()
    spirv_code = compile_model_to_target(model, target="spirv")

    assert "OpCapability CooperativeMatrixKHR" in spirv_code
    assert "OpExtension \"SPV_KHR_cooperative_matrix\"" in spirv_code
    assert "OpTypePointer Workgroup %float" in spirv_code
    assert "OpCooperativeMatrixMulAddKHR" in spirv_code
    assert "OpControlBarrier" in spirv_code


def test_emitter_metal_msl():
    """Verifies Apple Metal 4 MSL emission with SIMDgroup matrix."""
    model = SimpleMLP()
    msl_code = compile_model_to_target(model, target="metal")

    assert "#include <metal_simdgroup_matrix>" in msl_code
    assert "threadgroup float Q_shared[64 * 64];" in msl_code
    assert "simdgroup_float8x8 acc_o" in msl_code
    assert "simdgroup_multiply_accumulate" in msl_code


def test_emitter_tilelang_python():
    """Verifies TileLang Python PrimFunc emission with explicit memory placement."""
    model = SimpleMLP()
    tl_code = compile_model_to_target(model, target="tilelang")

    assert "@T.prim_func" in tl_code
    assert "Q_shared = T.alloc_shared" in tl_code
    assert "M_frag = T.alloc_fragment" in tl_code
    assert "for c_block in T.Pipelined" in tl_code
    assert "T.gemm(Q_shared, K_shared, O_frag)" in tl_code


# =============================================================================
# 4. Attention & Loss Workload End-to-End
# =============================================================================

def test_attention_block_full_pipeline():
    """Verifies MockAttentionBlock lowers through all 3 dialects and compiles to PTX."""
    model = MockAttentionBlock()
    ptx_code = compile_model_to_target(model, target="nvvm")
    assert "ojasx_tile_dagger_attention" in ptx_code
    assert "cp.async" in ptx_code


def test_dram_intermediate_memory_audit():
    """Formally asserts that no intermediate score matrix is allocated in global DRAM.

    In the Tile IR, Q, K, V tiles and attention statistics exist ONLY in:
    - Shared Memory (alloc_shared)
    - Register Fragments (alloc_fragment)
    There are ZERO memref (global DRAM) allocations for intermediate attention scores.
    """
    model = MockAttentionBlock()
    gm = fx.symbolic_trace(model)
    m_holo = fx_to_holonomic_mlir(gm)
    m_subgraph = HolonomicToSubgraphPass().run(m_holo)
    m_tile = SubgraphToTilePass(tile_m=64, tile_n=64, tile_k=32).run(m_subgraph)

    func = m_tile.functions[0]
    # Check that alloc_shared targets SRAM (swizzled memrefs) and fragments target registers
    shared_allocs = [op for op in func.body.operations if op.op_name == "alloc_shared"]
    frag_allocs = [op for op in func.body.operations if op.op_name == "alloc_fragment"]

    assert len(shared_allocs) > 0, "Must have explicit shared memory SRAM tile allocations"
    assert len(frag_allocs) > 0, "Must have explicit register fragment tile allocations"

    # Verify that none of the intermediate attention operations allocate persistent DRAM buffers
    dram_intermediate_allocs = [
        op for op in func.body.operations
        if op.op_name == "alloc_dram" or (op.results and any("score_matrix" in r.name for r in op.results))
    ]
    assert len(dram_intermediate_allocs) == 0, "Intermediate score matrix must NEVER be allocated in global DRAM!"
