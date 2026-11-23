"""
Phase 4 & 5: Target-Filtered Standard MLIR Emitter (`func`, `linalg`, `arith`, `math`, `tensor`).

Lowers `OjasX-LogicIR` modules directly into standard MLIR dialects with ZERO
custom hardware ISA hacks, cleanly respecting backend ISA capabilities:
  - `metal-spirv` (Apple Silicon M1–M5, SPIR-V 1.5):
      Emits `dual_rail_popcnt_tile` (`math.ctpop`), `bit_relu`, `bit_abs`, and
      `dense_bf16_gemm` (`linalg.matmul` on bf16/f32). Omits `i8` CRT matmul
      so IREE 3.14.0 never requests SPIR-V 1.6 `DotProduct`.
  - `llvm-cpu` (ARM64 NEON/I8MM, x86_64 AVX-512 VNNI):
      Emits ALL 5 kernels including `crt_int8_channel_gemm` (`i8 x i8 -> i32`).
  - `cuda` / `rocm`:
      Emits `dual_rail_popcnt_tile`, `bit_relu`, `bit_abs`, `dense_bf16_gemm`,
      and `crt_int8_channel_gemm`.
"""

from __future__ import annotations

from typing import List, Set

from ojasx_logic.ir.logic_ir import LogicIRModule


def _emit_bit_relu_func(n_elems: int = 1024) -> str:
    return f"""
  // Z3-verified branchless IEEE-754 ReLU: x & ~(x >> 31)
  func.func @ojasx_bit_relu(%x: tensor<{n_elems}xf32>) -> tensor<{n_elems}xf32> {{
    %init = tensor.empty() : tensor<{n_elems}xf32>
    %c31 = arith.constant 31 : i32
    %c_neg1 = arith.constant -1 : i32
    %res = linalg.generic {{
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }} ins(%x : tensor<{n_elems}xf32>) outs(%init : tensor<{n_elems}xf32>) {{
    ^bb0(%in: f32, %out: f32):
      %bits = arith.bitcast %in : f32 to i32
      %sign_mask = arith.shrsi %bits, %c31 : i32
      %keep_mask = arith.xori %sign_mask, %c_neg1 : i32
      %masked = arith.andi %bits, %keep_mask : i32
      %val = arith.bitcast %masked : i32 to f32
      linalg.yield %val : f32
    }} -> tensor<{n_elems}xf32>
    return %res : tensor<{n_elems}xf32>
  }}
"""


def _emit_bit_abs_func(n_elems: int = 1024) -> str:
    return f"""
  // Z3-verified branchless IEEE-754 abs: x & 0x7FFFFFFF
  func.func @ojasx_bit_abs(%x: tensor<{n_elems}xf32>) -> tensor<{n_elems}xf32> {{
    %init = tensor.empty() : tensor<{n_elems}xf32>
    %c_mask = arith.constant 2147483647 : i32
    %res = linalg.generic {{
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }} ins(%x : tensor<{n_elems}xf32>) outs(%init : tensor<{n_elems}xf32>) {{
    ^bb0(%in: f32, %out: f32):
      %bits = arith.bitcast %in : f32 to i32
      %masked = arith.andi %bits, %c_mask : i32
      %val = arith.bitcast %masked : i32 to f32
      linalg.yield %val : f32
    }} -> tensor<{n_elems}xf32>
    return %res : tensor<{n_elems}xf32>
  }}
"""


def _emit_dual_rail_popcnt_func(m: int = 64, n: int = 64, k_words: int = 16) -> str:
    return f"""
  // Multiplier-free Dual-Rail (P, N) Bitplane GEMM (32-bit words for native Metal/SPIR-V 1.5):
  //   dot(A, B) = popcnt((Ap & Bp) | (An & Bn)) - popcnt((Ap & Bn) | (An & Bp))
  func.func @ojasx_dual_rail_popcnt_tile(
      %a_pos: tensor<{m}x{k_words}xi32>,
      %a_neg: tensor<{m}x{k_words}xi32>,
      %b_pos: tensor<{n}x{k_words}xi32>,
      %b_neg: tensor<{n}x{k_words}xi32>) -> tensor<{m}x{n}xi32> {{
    %c0 = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<{m}x{n}xi32>
    %init = linalg.fill ins(%c0 : i32) outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %res = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%a_pos, %a_neg, %b_pos, %b_neg :
           tensor<{m}x{k_words}xi32>, tensor<{m}x{k_words}xi32>,
           tensor<{n}x{k_words}xi32>, tensor<{n}x{k_words}xi32>)
       outs(%init : tensor<{m}x{n}xi32>) {{
    ^bb0(%ap: i32, %an: i32, %bp: i32, %bn: i32, %acc: i32):
      %pp = arith.andi %ap, %bp : i32
      %nn = arith.andi %an, %bn : i32
      %pos_bits = arith.ori %pp, %nn : i32
      %pn = arith.andi %ap, %bn : i32
      %np = arith.andi %an, %bp : i32
      %neg_bits = arith.ori %pn, %np : i32
      %c_pos = math.ctpop %pos_bits : i32
      %c_neg = math.ctpop %neg_bits : i32
      %diff = arith.subi %c_pos, %c_neg : i32
      %new_acc = arith.addi %acc, %diff : i32
      linalg.yield %new_acc : i32
    }} -> tensor<{m}x{n}xi32>
    return %res : tensor<{m}x{n}xi32>
  }}
"""


def _emit_crt_int8_channel_gemm(m: int = 64, k: int = 128, n: int = 64) -> str:
    return f"""
  // Exact CRT-3 / CRT-4 INT8 channel matmul (i8 x i8 -> i32; llvm-cpu / cuda / rocm)
  func.func @ojasx_crt_int8_channel_gemm(
      %a: tensor<{m}x{k}xi8>,
      %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
    %c0 = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<{m}x{n}xi32>
    %init = linalg.fill ins(%c0 : i32) outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %res = linalg.matmul ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
                         outs(%init : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    return %res : tensor<{m}x{n}xi32>
  }}
"""


def _emit_dense_f32_gemm(m: int = 64, k: int = 128, n: int = 64) -> str:
    return f"""
  // Portable GPU/CPU dense matmul (f32 x f32 -> f32; SPIR-V 1.5 / Metal 4 clean)
  func.func @ojasx_dense_gemm(
      %a: tensor<{m}x{k}xf32>,
      %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %res = linalg.matmul ins(%a, %b : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
                         outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %res : tensor<{m}x{n}xf32>
  }}
"""


def emit_standalone_mlir_suite(
    *,
    target: str = "metal-spirv",
    m: int = 64,
    n: int = 64,
    k_words: int = 16,
    k_int8: int = 128,
    act_elems: int = 1024,
) -> str:
    """Emit target-filtered standard MLIR ready for `iree-compile`."""
    parts = [
        f"// OjasX-LogicIR Standard MLIR Module (target={target})",
        "module @ojasx_logic_suite {",
        _emit_dual_rail_popcnt_func(m=m, n=n, k_words=k_words),
        _emit_bit_relu_func(n_elems=act_elems),
        _emit_bit_abs_func(n_elems=act_elems),
        _emit_dense_f32_gemm(m=m, k=k_int8, n=n),
    ]
    # SPIR-V 1.5 (metal-spirv) lacks SPIR-V 1.6 DotProduct for i8 matmul;
    # route CRT INT8 strictly to llvm-cpu / cuda / rocm per choose_alu_route().
    if target != "metal-spirv":
        parts.append(_emit_crt_int8_channel_gemm(m=m, k=k_int8, n=n))
    parts.append("}")
    return "\n".join(parts) + "\n"


def emit_logic_ir_mlir(ir_mod: LogicIRModule) -> str:
    """Lower a dynamic `LogicIRModule` (e.g., from PyTorch FX) to target-filtered standard MLIR."""
    header: List[str] = [
        "// OjasX-LogicIR -> Standard MLIR Lowering",
        f"// Model: {ir_mod.name} | Target: {ir_mod.target} | Ops: {len(ir_mod.ops)}",
    ]
    emitted: Set[str] = set()
    for op in ir_mod.ops:
        header.append(
            f"//   {op.op_id}: kind={op.kind} route={op.alu_route} shape={op.shape} -> {op.output}"
        )
        emitted.add(op.kind)

    body: List[str] = [f"module @{ir_mod.name.replace('-', '_')} {{"]
    if "dual_rail_gemm" in emitted:
        body.append(_emit_dual_rail_popcnt_func())
    if "crt_bf16_gemm" in emitted:
        if ir_mod.target == "metal-spirv":
            body.append(_emit_dense_f32_gemm())
        else:
            body.append(_emit_crt_int8_channel_gemm())
    if "dense_bf16_gemm" in emitted and _emit_dense_f32_gemm() not in body:
        body.append(_emit_dense_f32_gemm())
    if "bit_relu" in emitted or "bit_silu" in emitted or "bit_gelu" in emitted:
        body.append(_emit_bit_relu_func())
    if "bit_abs" in emitted:
        body.append(_emit_bit_abs_func())
    if len(body) == 1:
        body.append(_emit_bit_relu_func())
    body.append("}")
    return "\n".join(header + body) + "\n"


__all__ = ["emit_logic_ir_mlir", "emit_standalone_mlir_suite"]
