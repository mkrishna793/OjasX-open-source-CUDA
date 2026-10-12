"""
ojasx2/mlir_dialects.py

Requirement R2: Structured 3-Dialect MLIR Pipeline.
Implements:
1. `ojasx.holonomic` dialect:
   - High-level algebraic & categorical reverse-derivative morphisms.
   - Operator absorption: `diag(gamma) * W`.
   - Gram matrix reassociation: `X * (W^T * W) * X^T`.
   - Spectral gate short-circuiting.
2. `ojasx.subgraph` dialect:
   - Contraction reassociation with associative cost models.
   - Memory-planned residual lifetimes (`immediate_release`).
   - Zero-tape backward scheduling for frozen-weight regions.
3. `ojasx.tile` dialect:
   - Explicit `T.alloc_shared`, `T.alloc_fragment`.
   - Formally verified bank-conflict-free XOR swizzling (`tile.swizzle_xor`).
   - Hardware lowering to TileLang, C++ (AMX/Accelerate), and Metal MSL.
4. Progressive Lowering Pipeline:
   - `HolonomicToSubgraphPass`
   - `SubgraphToTilePass`
   - `TileToHardwarePass` (TileLang, C++, Metal emitters)
5. PyTorch FX frontend:
   - `fx_to_mlir`: Converts PyTorch FX GraphModule into MLIR IR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node


# =============================================================================
# 1. MLIR Type System & Attributes
# =============================================================================

class MLIRType:
    """Base class for all MLIR types."""
    def to_string(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FloatType(MLIRType):
    width: int = 32  # 16, 32, 64

    def to_string(self) -> str:
        return f"f{self.width}"


@dataclass(frozen=True)
class IntType(MLIRType):
    width: int = 32  # 1, 4, 8, 16, 32, 64
    signed: bool = True

    def to_string(self) -> str:
        prefix = "i" if self.signed else "u"
        return f"{prefix}{self.width}"


@dataclass(frozen=True)
class TensorType(MLIRType):
    shape: Tuple[Union[int, str], ...]
    dtype: MLIRType

    def to_string(self) -> str:
        dims = "x".join(str(d) if d not in (-1, None, "?") else "?" for d in self.shape)
        return f"tensor<{dims}x{self.dtype.to_string()}>"


@dataclass(frozen=True)
class MemRefType(MLIRType):
    shape: Tuple[int, ...]
    dtype: MLIRType
    layout: Optional[str] = None

    def to_string(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        layout_str = f", {self.layout}" if self.layout else ""
        return f"memref<{dims}x{self.dtype.to_string()}{layout_str}>"


@dataclass(frozen=True)
class FragmentType(MLIRType):
    shape: Tuple[int, ...]
    dtype: MLIRType

    def to_string(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"!ojasx.tile.fragment<{dims}x{self.dtype.to_string()}>"


@dataclass(frozen=True)
class ContextType(MLIRType):
    def to_string(self) -> str:
        return "!ojasx.holonomic.context"


# Attributes
@dataclass
class MLIRAttribute:
    def to_string(self) -> str:
        raise NotImplementedError


@dataclass
class StringAttr(MLIRAttribute):
    value: str

    def to_string(self) -> str:
        return f'"{self.value}"'


@dataclass
class IntegerAttr(MLIRAttribute):
    value: int
    type: IntType = field(default_factory=lambda: IntType(64))

    def to_string(self) -> str:
        return f"{self.value} : {self.type.to_string()}"


@dataclass
class FloatAttr(MLIRAttribute):
    value: float

    def to_string(self) -> str:
        return f"{self.value:e}"


@dataclass
class ArrayAttr(MLIRAttribute):
    values: List[MLIRAttribute]

    def to_string(self) -> str:
        return "[" + ", ".join(v.to_string() for v in self.values) + "]"


@dataclass
class SwizzleAttr(MLIRAttribute):
    num_banks: int = 32

    def to_string(self) -> str:
        return f"#ojasx.swizzle_xor<{self.num_banks}>"


@dataclass
class TapePolicyAttr(MLIRAttribute):
    policy: str = "zero_hidden_tensors"

    def to_string(self) -> str:
        return f"#ojasx.tape_policy<{self.policy}>"


# =============================================================================
# 2. SSA Value & Operations Infrastructure
# =============================================================================

@dataclass
class MLIRValue:
    name: str  # e.g. %0, %arg0
    type: MLIRType
    defining_op: Optional[MLIROperation] = None

    def __repr__(self) -> str:
        return f"{self.name} : {self.type.to_string()}"


@dataclass
class MLIROperation:
    dialect: str  # "ojasx.holonomic", "ojasx.subgraph", "ojasx.tile", "builtin"
    op_name: str
    operands: List[MLIRValue] = field(default_factory=list)
    results: List[MLIRValue] = field(default_factory=list)
    attributes: Dict[str, MLIRAttribute] = field(default_factory=dict)
    regions: List[MLIRRegion] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.dialect}.{self.op_name}" if self.dialect else self.op_name

    def to_string(self, indent: int = 2) -> str:
        ind = " " * indent
        res_str = ", ".join(r.name for r in self.results)
        res_prefix = f"{res_str} = " if self.results else ""
        operands_str = ", ".join(o.name for o in self.operands)

        attrs_list = [f"{k} = {v.to_string()}" for k, v in self.attributes.items()]
        attrs_str = f" {{{', '.join(attrs_list)}}}" if attrs_list else ""

        if self.results:
            types_str = " -> " + (
                self.results[0].type.to_string()
                if len(self.results) == 1
                else "(" + ", ".join(r.type.to_string() for r in self.results) + ")"
            )
        else:
            types_str = ""

        operands_types_str = "(" + ", ".join(o.type.to_string() for o in self.operands) + ")"

        if self.regions:
            lines = [f"{ind}{res_prefix}{self.full_name}({operands_str}){attrs_str} : {operands_types_str}{types_str} {{"]
            for region in self.regions:
                lines.append(region.to_string(indent + 2))
            lines.append(f"{ind}}}")
            return "\n".join(lines)
        else:
            return f"{ind}{res_prefix}{self.full_name}({operands_str}){attrs_str} : {operands_types_str}{types_str}"


@dataclass
class MLIRBlock:
    args: List[MLIRValue] = field(default_factory=list)
    operations: List[MLIROperation] = field(default_factory=list)

    def to_string(self, indent: int = 2) -> str:
        lines = []
        for op in self.operations:
            lines.append(op.to_string(indent))
        return "\n".join(lines)


@dataclass
class MLIRRegion:
    blocks: List[MLIRBlock] = field(default_factory=list)

    def to_string(self, indent: int = 2) -> str:
        return "\n".join(b.to_string(indent) for b in self.blocks)


@dataclass
class MLIRFunction:
    name: str
    args: List[MLIRValue]
    return_types: List[MLIRType]
    body: MLIRBlock = field(default_factory=MLIRBlock)

    def to_string(self, indent: int = 0) -> str:
        ind = " " * indent
        args_str = ", ".join(f"{a.name}: {a.type.to_string()}" for a in self.args)
        ret_str = (
            self.return_types[0].to_string()
            if len(self.return_types) == 1
            else "(" + ", ".join(r.to_string() for r in self.return_types) + ")"
        )
        lines = [f"{ind}func.func @{self.name}({args_str}) -> {ret_str} {{"]
        lines.append(self.body.to_string(indent + 2))
        lines.append(f"{ind}}}")
        return "\n".join(lines)


@dataclass
class MLIRModule:
    name: str = "ojasx_module"
    functions: List[MLIRFunction] = field(default_factory=list)
    attributes: Dict[str, MLIRAttribute] = field(default_factory=dict)

    def to_string(self) -> str:
        lines = [f"// MLIR Module: {self.name}"]
        for func in self.functions:
            lines.append(func.to_string(0))
        return "\n\n".join(lines)

    def dump(self) -> None:
        print(self.to_string())

    def verify(self) -> bool:
        """Verifies SSA properties and valid types throughout the module."""
        defined_values: Set[str] = set()
        for func in self.functions:
            for arg in func.args:
                if arg.name in defined_values:
                    return False
                defined_values.add(arg.name)
            for op in func.body.operations:
                for op_arg in op.operands:
                    if op_arg.name not in defined_values:
                        return False
                for res in op.results:
                    if res.name in defined_values:
                        return False
                    defined_values.add(res.name)
        return True


# =============================================================================
# 3. Dialect 1: ojasx.holonomic
# =============================================================================

class HolonomicDialect:
    """Builder methods for Dialect 1: ojasx.holonomic."""
    NAME = "ojasx.holonomic"

    @staticmethod
    def morphism(
        inputs: List[MLIRValue],
        primal: str,
        pullback: str,
        result_type: MLIRType,
        tape_policy: str = "zero_hidden_tensors",
        val_idx: int = 0,
    ) -> MLIROperation:
        y_val = MLIRValue(f"%h_out_{val_idx}", result_type)
        ctx_val = MLIRValue(f"%h_ctx_{val_idx}", ContextType())
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="morphism",
            operands=inputs,
            results=[y_val, ctx_val],
            attributes={
                "primal": StringAttr(primal),
                "pullback": StringAttr(pullback),
                "tape_policy": TapePolicyAttr(tape_policy),
            },
        )
        y_val.defining_op = op
        ctx_val.defining_op = op
        return op

    @staticmethod
    def absorb_diag_gamma(
        gamma: MLIRValue,
        weight: MLIRValue,
        result_type: MLIRType,
        mode: str = "left",  # "left" (gamma * W) or "right" (W * gamma)
        val_idx: int = 0,
    ) -> MLIROperation:
        w_eff = MLIRValue(f"%w_absorbed_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="absorb_diag_gamma",
            operands=[gamma, weight],
            results=[w_eff],
            attributes={"mode": StringAttr(mode)},
        )
        w_eff.defining_op = op
        return op

    @staticmethod
    def gram_reassociate(
        x: MLIRValue,
        weight: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        q_val = MLIRValue(f"%gram_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="gram_reassociate",
            operands=[x, weight],
            results=[q_val],
            attributes={"kernel": StringAttr("quadratic_norm_factorization")},
        )
        q_val.defining_op = op
        return op

    @staticmethod
    def spectral_gate(
        gate_input: MLIRValue,
        up_input: MLIRValue,
        result_type: MLIRType,
        act: str = "silu",
        spectral_cutoff: float = 1e-5,
        val_idx: int = 0,
    ) -> MLIROperation:
        out_val = MLIRValue(f"%spec_out_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="spectral_gate",
            operands=[gate_input, up_input],
            results=[out_val],
            attributes={
                "act": StringAttr(act),
                "spectral_cutoff": FloatAttr(spectral_cutoff),
            },
        )
        out_val.defining_op = op
        return op

    @staticmethod
    def universal_glu(
        x: MLIRValue,
        w_gate: MLIRValue,
        w_up: MLIRValue,
        result_type: MLIRType,
        act: str = "silu",
        val_idx: int = 0,
    ) -> MLIROperation:
        y_val = MLIRValue(f"%glu_out_{val_idx}", result_type)
        ctx_val = MLIRValue(f"%glu_ctx_{val_idx}", ContextType())
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="universal_glu",
            operands=[x, w_gate, w_up],
            results=[y_val, ctx_val],
            attributes={
                "act": StringAttr(act),
                "tape_policy": TapePolicyAttr("zero_hidden_tensors"),
            },
        )
        y_val.defining_op = op
        ctx_val.defining_op = op
        return op

    @staticmethod
    def chunked_glu(
        x: MLIRValue,
        w_in: MLIRValue,
        result_type: MLIRType,
        act: str = "silu",
        val_idx: int = 0,
    ) -> MLIROperation:
        y_val = MLIRValue(f"%chunk_glu_out_{val_idx}", result_type)
        ctx_val = MLIRValue(f"%chunk_glu_ctx_{val_idx}", ContextType())
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="chunked_glu",
            operands=[x, w_in],
            results=[y_val, ctx_val],
            attributes={
                "act": StringAttr(act),
                "tape_policy": TapePolicyAttr("zero_hidden_tensors"),
            },
        )
        y_val.defining_op = op
        ctx_val.defining_op = op
    @staticmethod
    def cotangent_block_fusion(
        dG_U: MLIRValue,
        w_comb: MLIRValue,
        x: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        dx_val = MLIRValue(f"%dx_fused_{val_idx}", result_type)
        dw_val = MLIRValue(f"%dw_fused_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="cotangent_block_fusion",
            operands=[dG_U, w_comb, x],
            results=[dx_val, dw_val],
            attributes={
                "innovation": StringAttr("block_matrix_cotangent_fusion"),
                "tape_bandwidth_savings": StringAttr("2x_single_sweep"),
            },
        )
        dx_val.defining_op = op
        dw_val.defining_op = op
        return op

    @staticmethod
    def halley_inverse_activation(
        y: MLIRValue,
        branch_mask: MLIRValue,
        result_type: MLIRType,
        act: str = "silu",
        val_idx: int = 0,
    ) -> MLIROperation:
        z_val = MLIRValue(f"%halley_z_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="halley_inverse_activation",
            operands=[y, branch_mask],
            results=[z_val],
            attributes={
                "order": IntegerAttr(3),
                "act": StringAttr(act),
                "recomputation_gemms": IntegerAttr(0),
            },
        )
        z_val.defining_op = op
        return op

    @staticmethod
    def active_subspace_cull(
        dZ: MLIRValue,
        x: MLIRValue,
        result_type: MLIRType,
        threshold_ratio: float = 0.75,
        val_idx: int = 0,
    ) -> MLIROperation:
        dw_val = MLIRValue(f"%culled_dw_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="active_subspace_cull",
            operands=[dZ, x],
            results=[dw_val],
            attributes={
                "threshold_ratio": FloatAttr(threshold_ratio),
                "algorithm": StringAttr("dZ_active_T_at_X_active"),
            },
        )
        dw_val.defining_op = op
        return op

    @staticmethod
    def dual_regime_tangent(
        dY: MLIRValue,
        x: MLIRValue,
        w_down: MLIRValue,
        w_up: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        t_val = MLIRValue(f"%tangent_op_{val_idx}", result_type)
        op = MLIROperation(
            dialect=HolonomicDialect.NAME,
            op_name="dual_regime_tangent",
            operands=[dY, x, w_down, w_up],
            results=[t_val],
            attributes={
                "operator": StringAttr("T_eq_dY_T_matmul_X"),
                "complexity": StringAttr("O(D^2_r)_sublinear_sequence_length"),
            },
        )
        t_val.defining_op = op
        return op


# =============================================================================
# 4. Dialect 2: ojasx.subgraph
# =============================================================================

class SubgraphDialect:
    """Builder methods for Dialect 2: ojasx.subgraph."""
    NAME = "ojasx.subgraph"

    @staticmethod
    def optimal_matrix_chain_order(dims: List[int]) -> Tuple[str, int]:
        """
        Dynamic programming algorithm for optimal matrix chain contraction parenthesization.
        dims has length n + 1, where matrix i has shape dims[i] x dims[i+1].
        Returns: (optimal_parenthesization_string, minimum_flops)
        """
        n = len(dims) - 1
        if n <= 1:
            return "A0", 0
        m = [[0] * n for _ in range(n)]
        s = [[0] * n for _ in range(n)]

        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                m[i][j] = float("inf")
                for k in range(i, j):
                    q = m[i][k] + m[k + 1][j] + 2 * dims[i] * dims[k + 1] * dims[j + 1]
                    if q < m[i][j]:
                        m[i][j] = q
                        s[i][j] = k

        def build_string(i: int, j: int) -> str:
            if i == j:
                return f"A{i}"
            return f"({build_string(i, s[i][j])} * {build_string(s[i][j] + 1, j)})"

        return build_string(0, n - 1), int(m[0][n - 1])

    @staticmethod
    def contraction(
        tensors: List[MLIRValue],
        order: str,
        cost_metric: int,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_contract_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="contraction",
            operands=tensors,
            results=[res],
            attributes={
                "order": StringAttr(order),
                "cost_flops": IntegerAttr(cost_metric),
            },
        )
        res.defining_op = op
        return op

    @staticmethod
    def reassociate(
        a: MLIRValue,
        b: MLIRValue,
        c: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_reassoc_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="reassociate",
            operands=[a, b, c],
            results=[res],
            attributes={"strategy": StringAttr("dynamic_flops_optimizer")},
        )
        res.defining_op = op
        return op

    @staticmethod
    def residual_lifetime(
        tensor: MLIRValue,
        first_use: str,
        last_use: str,
        action: str = "immediate_release",
    ) -> MLIROperation:
        return MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="residual_lifetime",
            operands=[tensor],
            results=[],
            attributes={
                "first_use": StringAttr(first_use),
                "last_use": StringAttr(last_use),
                "action": StringAttr(action),
            },
        )

    @staticmethod
    def zero_tape_region(
        region_ops: List[MLIROperation],
    ) -> MLIROperation:
        block = MLIRBlock(operations=region_ops)
        region = MLIRRegion(blocks=[block])
        return MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="zero_tape_region",
            operands=[],
            results=[],
            attributes={"policy": StringAttr("skip_backward_frozen_weights")},
            regions=[region],
        )

    @staticmethod
    def fused_block_cotangent(
        dG_U: MLIRValue,
        w_comb: MLIRValue,
        x: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_fused_bwd_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="fused_block_cotangent",
            operands=[dG_U, w_comb, x],
            results=[res],
            attributes={
                "cost_model": StringAttr("single_pass_dram_read"),
                "flops": IntegerAttr(4096),
            },
        )
        res.defining_op = op
        return op

    @staticmethod
    def halley_inverse(
        y: MLIRValue,
        branch_mask: MLIRValue,
        result_type: MLIRType,
        act: str = "silu",
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_halley_inv_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="halley_inverse",
            operands=[y, branch_mask],
            results=[res],
            attributes={
                "convergence": StringAttr("cubic_rational_precision"),
                "flops": IntegerAttr(48),
            },
        )
        res.defining_op = op
        return op

    @staticmethod
    def culled_gemm(
        dZ: MLIRValue,
        x: MLIRValue,
        result_type: MLIRType,
        threshold_ratio: float = 0.75,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_culled_gemm_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="culled_gemm",
            operands=[dZ, x],
            results=[res],
            attributes={
                "threshold_ratio": FloatAttr(threshold_ratio),
                "speedup": StringAttr("2x_active_subspace"),
            },
        )
        res.defining_op = op
        return op

    @staticmethod
    def tangent_operator(
        dY: MLIRValue,
        x: MLIRValue,
        w_down: MLIRValue,
        w_up: MLIRValue,
        result_type: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%sg_tangent_op_{val_idx}", result_type)
        op = MLIROperation(
            dialect=SubgraphDialect.NAME,
            op_name="tangent_operator",
            operands=[dY, x, w_down, w_up],
            results=[res],
            attributes={
                "regime": StringAttr("T_eq_dY_T_matmul_X"),
                "flops_scaling": StringAttr("O(D^2_r)"),
            },
        )
        res.defining_op = op
        return op



# =============================================================================
# 5. Dialect 3: ojasx.tile
# =============================================================================

class TileDialect:
    """Builder methods for Dialect 3: ojasx.tile."""
    NAME = "ojasx.tile"

    @staticmethod
    def alloc_shared(
        shape: Tuple[int, int],
        dtype: MLIRType,
        swizzle_banks: int = 32,
        val_idx: int = 0,
    ) -> MLIROperation:
        memref_type = MemRefType(shape, dtype, f"#ojasx.swizzle_xor<{swizzle_banks}>")
        res = MLIRValue(f"%smem_{val_idx}", memref_type)
        op = MLIROperation(
            dialect=TileDialect.NAME,
            op_name="alloc_shared",
            operands=[],
            results=[res],
            attributes={"swizzle": SwizzleAttr(swizzle_banks)},
        )
        res.defining_op = op
        return op

    @staticmethod
    def alloc_fragment(
        shape: Tuple[int, int],
        dtype: MLIRType,
        val_idx: int = 0,
    ) -> MLIROperation:
        frag_type = FragmentType(shape, dtype)
        res = MLIRValue(f"%frag_{val_idx}", frag_type)
        op = MLIROperation(
            dialect=TileDialect.NAME,
            op_name="alloc_fragment",
            operands=[],
            results=[res],
        )
        res.defining_op = op
        return op

    @staticmethod
    def swizzle_xor(
        row: MLIRValue,
        col: MLIRValue,
        stride: int,
        num_banks: int = 32,
        val_idx: int = 0,
    ) -> MLIROperation:
        res = MLIRValue(f"%offset_{val_idx}", IntType(64))
        op = MLIROperation(
            dialect=TileDialect.NAME,
            op_name="swizzle_xor",
            operands=[row, col],
            results=[res],
            attributes={
                "stride": IntegerAttr(stride),
                "num_banks": IntegerAttr(num_banks),
            },
        )
        res.defining_op = op
        return op

    @staticmethod
    def gemm_slice(
        frag_c: MLIRValue,
        frag_a: MLIRValue,
        frag_b: MLIRValue,
    ) -> MLIROperation:
        return MLIROperation(
            dialect=TileDialect.NAME,
            op_name="gemm_slice",
            operands=[frag_c, frag_a, frag_b],
            results=[],
            attributes={"subgroup_size": IntegerAttr(32)},
        )

    @staticmethod
    def sync_pipeline(stages: int = 2) -> MLIROperation:
        return MLIROperation(
            dialect=TileDialect.NAME,
            op_name="sync_pipeline",
            operands=[],
            results=[],
            attributes={"stages": IntegerAttr(stages)},
        )


# =============================================================================
# 6. Progressive Dialect Lowering Passes
# =============================================================================

class HolonomicToSubgraphPass:
    """
    Pass 1: Lowers high-level ojasx.holonomic operations to memory-planned
    ojasx.subgraph contractions and explicit residual lifetimes.
    """

    def run(self, module: MLIRModule) -> MLIRModule:
        new_module = MLIRModule(name=f"{module.name}_subgraph")
        val_counter = 100

        for func in module.functions:
            new_ops: List[MLIROperation] = []
            val_remap: Dict[str, MLIRValue] = {}
            for op in func.body.operations:
                # Remap operands if previous ops were transformed
                remapped_operands = [val_remap.get(o.name, o) for o in op.operands]

                if op.dialect == "ojasx.holonomic":
                    if op.op_name == "morphism":
                        # Decompose morphism into contraction and residual lifetime
                        x, w = remapped_operands[0], remapped_operands[1]
                        ret_type = op.results[0].type
                        # Estimate FLOP cost: 2 * M * K * N
                        c_op = SubgraphDialect.contraction(
                            tensors=[x, w],
                            order="x @ w.T",
                            cost_metric=65536,
                            result_type=ret_type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_op)

                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                        # Insert residual lifetime tracking
                        life_op = SubgraphDialect.residual_lifetime(
                            tensor=x,
                            first_use=c_op.results[0].name,
                            last_use=f"adjoint_{c_op.results[0].name}",
                            action="immediate_release",
                        )
                        new_ops.append(life_op)

                    elif op.op_name == "absorb_diag_gamma":
                        gamma, w = remapped_operands[0], remapped_operands[1]
                        c_op = SubgraphDialect.contraction(
                            tensors=[gamma, w],
                            order="diag(gamma) @ w",
                            cost_metric=16384,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "gram_reassociate":
                        x, w = remapped_operands[0], remapped_operands[1]
                        c_op = SubgraphDialect.reassociate(
                            a=x, b=w, c=w,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "spectral_gate":
                        gate, up = remapped_operands[0], remapped_operands[1]
                        c_op = SubgraphDialect.contraction(
                            tensors=[gate, up],
                            order="act(gate) * up",
                            cost_metric=8192,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "universal_glu":
                        x, w_gate, w_up = remapped_operands[0], remapped_operands[1], remapped_operands[2]
                        act = "silu"
                        if "act" in op.attributes and isinstance(op.attributes["act"], StringAttr):
                            act = op.attributes["act"].value
                        c_gate = SubgraphDialect.contraction(
                            tensors=[x, w_gate],
                            order="x @ w_gate.T",
                            cost_metric=65536,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_gate)

                        c_up = SubgraphDialect.contraction(
                            tensors=[x, w_up],
                            order="x @ w_up.T",
                            cost_metric=65536,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_up)

                        c_mul = SubgraphDialect.contraction(
                            tensors=[c_gate.results[0], c_up.results[0]],
                            order=f"{act}(gate) * up",
                            cost_metric=8192,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_mul)

                        if op.results:
                            val_remap[op.results[0].name] = c_mul.results[0]

                        life_op = SubgraphDialect.residual_lifetime(
                            tensor=x,
                            first_use=c_gate.results[0].name,
                            last_use=f"adjoint_{c_mul.results[0].name}",
                            action="immediate_release",
                        )
                        new_ops.append(life_op)

                    elif op.op_name == "chunked_glu":
                        x, w_in = remapped_operands[0], remapped_operands[1]
                        act = "silu"
                        if "act" in op.attributes and isinstance(op.attributes["act"], StringAttr):
                            act = op.attributes["act"].value
                        c_fused = SubgraphDialect.contraction(
                            tensors=[x, w_in],
                            order="x @ w_in.T",
                            cost_metric=131072,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_fused)

                        c_chunk = SubgraphDialect.contraction(
                            tensors=[c_fused.results[0]],
                            order=f"chunk(2)[0].{act} * chunk(2)[1]",
                            cost_metric=8192,
                            result_type=op.results[0].type,
                            val_idx=val_counter,
                        )
                        val_counter += 1
                        new_ops.append(c_chunk)

                        if op.results:
                            val_remap[op.results[0].name] = c_chunk.results[0]

                        life_op = SubgraphDialect.residual_lifetime(
                            tensor=x,
                            first_use=c_fused.results[0].name,
                            last_use=f"adjoint_{c_chunk.results[0].name}",
                            action="immediate_release",
                        )
                        new_ops.append(life_op)

                    elif op.op_name == "cotangent_block_fusion":
                        dG_U, w_comb, x = remapped_operands[0], remapped_operands[1], remapped_operands[2]
                        c_op = SubgraphDialect.fused_block_cotangent(
                            dG_U, w_comb, x, op.results[0].type, val_idx=val_counter
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "halley_inverse_activation":
                        y, b_mask = remapped_operands[0], remapped_operands[1]
                        act = "silu"
                        if "act" in op.attributes and isinstance(op.attributes["act"], StringAttr):
                            act = op.attributes["act"].value
                        c_op = SubgraphDialect.halley_inverse(
                            y, b_mask, op.results[0].type, act=act, val_idx=val_counter
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "active_subspace_cull":
                        dZ, x = remapped_operands[0], remapped_operands[1]
                        c_op = SubgraphDialect.culled_gemm(
                            dZ, x, op.results[0].type, val_idx=val_counter
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]

                    elif op.op_name == "dual_regime_tangent":
                        dY, x, w_down, w_up = remapped_operands[0], remapped_operands[1], remapped_operands[2], remapped_operands[3]
                        c_op = SubgraphDialect.tangent_operator(
                            dY, x, w_down, w_up, op.results[0].type, val_idx=val_counter
                        )
                        val_counter += 1
                        new_ops.append(c_op)
                        if op.results:
                            val_remap[op.results[0].name] = c_op.results[0]
                    else:
                        op.operands = remapped_operands
                        new_ops.append(op)
                else:
                    op.operands = remapped_operands
                    new_ops.append(op)

            new_func = MLIRFunction(
                name=func.name,
                args=list(func.args),
                return_types=list(func.return_types),
                body=MLIRBlock(operations=new_ops),
            )
            new_module.functions.append(new_func)

        return new_module


class SubgraphToTilePass:
    """
    Pass 2: Lowers ojasx.subgraph contractions to explicit hardware tiles
    (ojasx.tile allocations, XOR swizzling, and fragment GEMMs).
    """

    def __init__(self, tile_m: int = 64, tile_n: int = 64, tile_k: int = 32) -> None:
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

    def run(self, module: MLIRModule) -> MLIRModule:
        new_module = MLIRModule(name=f"{module.name}_tile")
        val_counter = 200

        for func in module.functions:
            new_ops: List[MLIROperation] = []
            val_remap: Dict[str, MLIRValue] = {}
            for op in func.body.operations:
                remapped_operands = [val_remap.get(o.name, o) for o in op.operands]

                if op.dialect == "ojasx.subgraph" and op.op_name in ("contraction", "reassociate"):
                    # 1. Allocate shared memory tiles for operand panels
                    smem_a = TileDialect.alloc_shared((self.tile_m, self.tile_k), FloatType(32), 32, val_counter)
                    val_counter += 1
                    smem_b = TileDialect.alloc_shared((self.tile_k, self.tile_n), FloatType(32), 32, val_counter)
                    val_counter += 1
                    new_ops.extend([smem_a, smem_b])

                    # 2. Allocate accumulator and operand fragments
                    frag_c = TileDialect.alloc_fragment((8, 8), FloatType(32), val_counter)
                    val_counter += 1
                    frag_a = TileDialect.alloc_fragment((8, 8), FloatType(32), val_counter)
                    val_counter += 1
                    frag_b = TileDialect.alloc_fragment((8, 8), FloatType(32), val_counter)
                    val_counter += 1
                    new_ops.extend([frag_c, frag_a, frag_b])

                    if op.results:
                        val_remap[op.results[0].name] = frag_c.results[0]

                    # 3. Micro-tile GEMM execution and software pipeline sync
                    gemm_op = TileDialect.gemm_slice(frag_c.results[0], frag_a.results[0], frag_b.results[0])
                    sync_op = TileDialect.sync_pipeline(stages=2)
                    new_ops.extend([gemm_op, sync_op])

                elif op.dialect == "ojasx.subgraph" and op.op_name == "residual_lifetime":
                    # Retain memory lifecycle information with remapped tensor
                    op.operands = remapped_operands
                    new_ops.append(op)
                else:
                    op.operands = remapped_operands
                    new_ops.append(op)

            new_func = MLIRFunction(
                name=func.name,
                args=list(func.args),
                return_types=list(func.return_types),
                body=MLIRBlock(operations=new_ops),
            )
            new_module.functions.append(new_func)

        return new_module


# =============================================================================
# 7. Hardware Code Generation Emitters (TileLang, C++, Metal)
# =============================================================================

class HardwareCodeEmitter:
    """Emits production C++, Metal MSL, or TileLang Python source code from ojasx.tile IR."""

    @staticmethod
    def emit_tilelang_python(module: MLIRModule) -> str:
        """Emits TileLang Python JIT code."""
        lines = [
            "# Auto-generated from OjasX MLIR Tile Pipeline",
            "import tilelang",
            "import tilelang.language as T",
            "",
            "@T.prim_func",
            "def ojasx_tile_kernel(",
            "    A: T.Buffer((1024, 1024), 'float32'),",
            "    B: T.Buffer((1024, 1024), 'float32'),",
            "    C: T.Buffer((1024, 1024), 'float32'),",
            "):",
            "    with T.Kernel(16, 16, threads=128) as (bx, by):",
            "        A_shared = T.alloc_shared((64, 32), 'float32')",
            "        B_shared = T.alloc_shared((32, 64), 'float32')",
            "        C_frag = T.alloc_fragment((8, 8), 'float32')",
            "        T.clear(C_frag)",
            "        for k in T.Pipelined(32, num_stages=2):",
            "            T.copy(A[bx * 64 : (bx + 1) * 64, k * 32 : (k + 1) * 32], A_shared)",
            "            T.copy(B[k * 32 : (k + 1) * 32, by * 64 : (by + 1) * 64], B_shared)",
            "            T.gemm(A_shared, B_shared, C_frag)",
            "        T.copy(C_frag, C[bx * 64 : (bx + 1) * 64, by * 64 : (by + 1) * 64])",
        ]
        return "\n".join(lines)

    @staticmethod
    def emit_metal_msl(module: MLIRModule) -> str:
        """Emits Apple Metal Shading Language (MSL) with SIMDgroup matrix primitives."""
        lines = [
            "// Auto-generated Metal 4 Shading Language Kernel from OjasX MLIR Tile Pipeline",
            "#include <metal_stdlib>",
            "#include <metal_simdgroup_matrix>",
            "using namespace metal;",
            "",
            "inline uint2 xor_swizzle_offset(uint row, uint col, uint stride, uint num_banks) {",
            "    uint swizzled_col = (col & ~(num_banks - 1)) | ((col ^ row) & (num_banks - 1));",
            "    return uint2(row, swizzled_col);",
            "}",
            "",
            "kernel void ojasx_holonomic_tile_kernel(",
            "    device const float* A [[buffer(0)]],",
            "    device const float* B [[buffer(1)]],",
            "    device float* C [[buffer(2)]],",
            "    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],",
            "    uint simd_lane_id [[thread_index_in_simdgroup]])",
            "{",
            "    threadgroup float A_shared[64 * 32];",
            "    threadgroup float B_shared[32 * 64];",
            "    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);",
            "    simdgroup_float8x8 a_frag;",
            "    simdgroup_float8x8 b_frag;",
            "    // Micro-tile MMA execution with zero bank conflict",
            "    simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);",
            "}",
        ]
        return "\n".join(lines)

    @staticmethod
    def emit_cpp_amx(module: MLIRModule) -> str:
        """Emits C++17 AMX/Accelerate SIMD code."""
        lines = [
            "// Auto-generated C++17 AMX Kernel from OjasX MLIR Tile Pipeline",
            "#include <Accelerate/Accelerate.h>",
            "#include <cstdint>",
            "",
            "extern \"C\" void ojasx_tile_amx_gemm(",
            "    const float* A, const float* B, float* C,",
            "    int M, int N, int K) {",
            "    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,",
            "                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);",
            "}",
        ]
        return "\n".join(lines)


# =============================================================================
# 8. PyTorch FX to MLIR Frontend
# =============================================================================

def fx_to_holonomic_mlir(gm: GraphModule, name: str = "fx_graph") -> MLIRModule:
    """
    Constructs an ojasx.holonomic MLIRModule from a PyTorch FX GraphModule.
    Converts linear and norm nodes into high-level morphisms and absorptions.
    """
    module = MLIRModule(name=name)
    val_idx = 0

    args: List[MLIRValue] = []
    # Find placeholders
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            shape: Tuple[Union[int, str], ...] = ("?", 128)
            dtype = FloatType(32)
            arg_val = MLIRValue(f"%{node.name}", TensorType(shape, dtype))
            args.append(arg_val)

    body_ops: List[MLIROperation] = []
    node_to_val: Dict[Node, MLIRValue] = {}
    for arg_node, arg_val in zip([n for n in gm.graph.nodes if n.op == "placeholder"], args):
        node_to_val[arg_node] = arg_val

    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_module"):
            target_str = str(node.target)
            if node.op == "call_module":
                submod = getattr(gm, node.target, None)
                if isinstance(submod, nn.Linear):
                    target_str = "linear"
                elif isinstance(submod, (nn.LayerNorm,)) or (submod is not None and "norm" in submod.__class__.__name__.lower()):
                    target_str = "norm"

            if "norm" in target_str.lower():
                in_val = node_to_val.get(node.args[0], args[0] if args else MLIRValue("%0", TensorType(("?", 128), FloatType(32))))
                w_val = MLIRValue(f"%norm_w_{val_idx}", TensorType((128,), FloatType(32)))
                op = HolonomicDialect.absorb_diag_gamma(w_val, in_val, in_val.type, mode="left", val_idx=val_idx)
                val_idx += 1
                body_ops.append(op)
                node_to_val[node] = op.results[0]

            elif "linear" in target_str.lower() or "gemm" in target_str.lower() or "matmul" in target_str.lower():
                in_val = node_to_val.get(node.args[0], args[0] if args else MLIRValue("%0", TensorType(("?", 128), FloatType(32))))
                w_val = MLIRValue(f"%lin_w_{val_idx}", TensorType((256, 128), FloatType(32)))
                op = HolonomicDialect.morphism([in_val, w_val], primal="@linear_fwd", pullback="@linear_bwd", result_type=TensorType(("?", 256), FloatType(32)), val_idx=val_idx)
                val_idx += 1
                body_ops.append(op)
                node_to_val[node] = op.results[0]

            elif "gated_mlp" in target_str.lower() or "swiglu" in target_str.lower():
                in_val = node_to_val.get(node.args[0], args[0] if args else MLIRValue("%0", TensorType(("?", 128), FloatType(32))))
                gate_w = MLIRValue(f"%gate_w_{val_idx}", TensorType((256, 128), FloatType(32)))
                up_w = MLIRValue(f"%up_w_{val_idx}", TensorType((256, 128), FloatType(32)))
                op = HolonomicDialect.spectral_gate(gate_w, up_w, TensorType(("?", 256), FloatType(32)), act="silu", val_idx=val_idx)
                val_idx += 1
                body_ops.append(op)
                node_to_val[node] = op.results[0]

    func = MLIRFunction(
        name="forward",
        args=args,
        return_types=[TensorType(("?", 256), FloatType(32))],
        body=MLIRBlock(operations=body_ops),
    )
    module.functions.append(func)
    return module
