"""
Phase 3: Typed `OjasX-LogicIR` & Hardware ALU Crossover Cost Router.

Routes every linear/matmul/activation op to the physically fastest ALU on the target:
  - `dual_rail_popcnt`: 1–2 bit ternary / sparse MX-NAF bitplanes -> AND + OR + POPCNT
  - `crt3_int8`:        Exact dense BF16 on CPU/INT8-rich targets -> 3/4-channel INT8 GEMM
  - `native_bf16_tc`:   Dense BF16 LoRA / sensitive layers on GPU Tensor Cores / AMX
  - `bit_circuit`:      Z3-verified bitwise ReLU, abs, fast_exp, SiLU-PWL, GELU-PWL, rsqrt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LogicOp:
    op_id: str
    kind: str  # "dual_rail_gemm", "crt_bf16_gemm", "dense_bf16_gemm", "bit_relu", "bit_silu", "bit_gelu", "bit_fast_exp", "bit_rsqrt", "cw_rotate", "add", "passthrough"
    inputs: Tuple[str, ...]
    output: str
    shape: Tuple[int, ...] = ()
    alu_route: str = "native_bf16_tc"
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicIRModule:
    name: str
    ops: List[LogicOp] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    target: str = "metal-spirv"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for op in self.ops:
            counts[op.kind] = counts.get(op.kind, 0) + 1
        return counts


def choose_alu_route(
    *,
    weight_bits: float = 16.0,
    target: str = "metal-spirv",
    M: int = 1,
    K: int = 1024,
    N: int = 1024,
    zero_sparsity: float = 0.0,
) -> str:
    """Honest hardware ALU crossover cost model (never forces slow bit-slicing onto dense Tensor Core ops)."""
    # 1. Low-bit (1.0 - 2.0 bit ternary / E8) or ultra-sparse MX-NAF -> Dual-Rail POPCNT wins!
    if weight_bits <= 2.0 or zero_sparsity >= 0.70:
        return "dual_rail_popcnt"
    # 2. Dense BF16 on CPU without AMX/GPU Tensor Cores -> CRT-3 / CRT-4 INT8 VNNI/I8MM!
    if target in ("llvm-cpu", "cpu") and K <= 4096:
        return "crt3_int8" if K <= 478 else "crt4_int8"
    # 3. Dense BF16 on Apple Metal / NVIDIA CUDA / AMD ROCm -> Native BF16 Tensor Cores!
    return "native_bf16_tc"


__all__ = ["LogicOp", "LogicIRModule", "choose_alu_route"]
