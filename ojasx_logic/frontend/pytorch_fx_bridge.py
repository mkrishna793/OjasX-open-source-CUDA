"""
Phase 3: Dynamic PyTorch FX Frontend -> `AlgMat-IR` -> `OjasX-LogicIR`.

Adapts dynamically to ANY PyTorch `nn.Module` (Transformers, ConvNets, ViTs, SSMs,
custom residual/skip architectures) with varying layer counts and activations—never
hardcoding HuggingFace layer names.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ojasx_logic.ir.logic_ir import LogicIRModule, LogicOp, choose_alu_route


_ACT_MAP = {
    "relu": "bit_relu",
    "silu": "bit_silu",
    "gelu": "bit_gelu",
    "exp": "bit_fast_exp",
    "rsqrt": "bit_rsqrt",
    "abs": "bit_abs",
}


def lower_pytorch_module_to_logic_ir(
    model: Any,
    *,
    name: str = "ojasx_model",
    target: str = "metal-spirv",
    weight_bits: float = 1.5,
) -> LogicIRModule:
    """Dynamically trace any PyTorch nn.Module via torch.fx and lower to OjasX-LogicIR."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.fx as fx

    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    ir_mod = LogicIRModule(name=name, target=target)

    for idx, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
            ir_mod.inputs.append(node.name)
            continue
        if node.op == "output":
            out_args = node.args[0]
            if isinstance(out_args, (tuple, list)):
                ir_mod.outputs.extend(str(a) for a in out_args)
            else:
                ir_mod.outputs.append(str(out_args))
            continue

        in_names = tuple(str(a) for a in node.args if isinstance(a, fx.Node))
        op_id = f"op_{idx}"

        if node.op == "call_module":
            sub = modules.get(str(node.target))
            if isinstance(sub, nn.Linear):
                route = choose_alu_route(
                    weight_bits=weight_bits,
                    target=target,
                    K=sub.in_features,
                    N=sub.out_features,
                )
                kind = (
                    "dual_rail_gemm"
                    if route == "dual_rail_popcnt"
                    else ("crt_bf16_gemm" if route.startswith("crt") else "dense_bf16_gemm")
                )
                ir_mod.ops.append(
                    LogicOp(
                        op_id=op_id,
                        kind=kind,
                        inputs=in_names,
                        output=node.name,
                        shape=(sub.out_features, sub.in_features),
                        alu_route=route,
                        attrs={"module_name": str(node.target), "bias": sub.bias is not None},
                    )
                )
            elif isinstance(sub, nn.ReLU):
                ir_mod.ops.append(LogicOp(op_id, "bit_relu", in_names, node.name, alu_route="bit_circuit"))
            elif isinstance(sub, nn.SiLU):
                ir_mod.ops.append(LogicOp(op_id, "bit_silu", in_names, node.name, alu_route="bit_circuit"))
            elif isinstance(sub, nn.GELU):
                ir_mod.ops.append(LogicOp(op_id, "bit_gelu", in_names, node.name, alu_route="bit_circuit"))
            else:
                ir_mod.ops.append(
                    LogicOp(
                        op_id=op_id,
                        kind="passthrough",
                        inputs=in_names,
                        output=node.name,
                        alu_route="native_bf16_tc",
                        attrs={"module_type": type(sub).__name__ if sub is not None else "unknown"},
                    )
                )
        elif node.op == "call_function":
            fn_name = getattr(node.target, "__name__", str(node.target)).lower()
            if fn_name in _ACT_MAP:
                ir_mod.ops.append(LogicOp(op_id, _ACT_MAP[fn_name], in_names, node.name, alu_route="bit_circuit"))
            elif node.target in (F.linear, torch.matmul, torch.mm):
                route = choose_alu_route(weight_bits=weight_bits, target=target)
                kind = "dual_rail_gemm" if route == "dual_rail_popcnt" else "dense_bf16_gemm"
                ir_mod.ops.append(LogicOp(op_id, kind, in_names, node.name, alu_route=route))
            elif fn_name in ("add", "iadd"):
                ir_mod.ops.append(LogicOp(op_id, "add", in_names, node.name, alu_route="native_bf16_tc"))
            else:
                ir_mod.ops.append(
                    LogicOp(op_id, "passthrough", in_names, node.name, attrs={"fn": fn_name})
                )
        else:
            ir_mod.ops.append(LogicOp(op_id, "passthrough", in_names, node.name, attrs={"op": node.op}))

    return ir_mod


def make_logic_ir_dynamo_backend(
    *,
    target: str = "metal-spirv",
    weight_bits: float = 1.5,
) -> Callable[..., Any]:
    """Drop-in torch.compile(..., backend=make_logic_ir_dynamo_backend()) bridge."""

    def _backend(gm: Any, example_inputs: List[Any]) -> Callable[..., Any]:
        ir_mod = lower_pytorch_module_to_logic_ir(gm, name="dynamo_graph", target=target, weight_bits=weight_bits)
        gm._ojasx_logic_ir = ir_mod
        return gm.forward

    return _backend


__all__ = ["lower_pytorch_module_to_logic_ir", "make_logic_ir_dynamo_backend"]
