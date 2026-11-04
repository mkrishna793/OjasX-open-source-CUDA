"""Strict parser for BackendBench tensor-specification expressions.

The dataset is data, not Python code. Only the small expression grammar used by
BackendBench is accepted; no builtins, arbitrary calls, or general attributes are
available to the parser.
"""

from __future__ import annotations

import ast
from typing import Any, Callable

import torch


_DTYPE_ALIASES = {
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "b8": torch.bool,
    "c32": torch.complex32,
    "c64": torch.complex64,
    "c128": torch.complex128,
}
_TORCH_ATTRIBUTES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "strided": torch.strided,
    "sparse_coo": torch.sparse_coo,
}


def parse_tensor_spec_args(
    source: str,
    tensor_spec: Callable[..., Any],
    *,
    max_source_bytes: int = 1_000_000,
    max_nodes: int = 100_000,
    max_depth: int = 64,
) -> Any:
    """Parse one BackendBench ``args`` value without executing Python code."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > max_source_bytes:
        raise ValueError("Tensor-specification source is missing or too large")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > max_nodes:
        raise ValueError("Tensor-specification expression has too many nodes")

    def evaluate(node: ast.AST, depth: int = 0) -> Any:
        if depth > max_depth:
            raise ValueError("Tensor-specification expression is nested too deeply")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if node.value is None or type(node.value) in (bool, int, float, str):
                return node.value
            raise ValueError(f"Unsupported constant in tensor specification: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id in _DTYPE_ALIASES:
                return _DTYPE_ALIASES[node.id]
            if node.id == "inf":
                return float("inf")
            raise ValueError(f"Unknown name in tensor specification: {node.id}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr in _TORCH_ATTRIBUTES:
                return _TORCH_ATTRIBUTES[node.attr]
            raise ValueError("Only explicitly allowed torch dtype/layout attributes are supported")
        if isinstance(node, ast.List):
            return [evaluate(item, depth + 1) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item, depth + 1) for item in node.elts)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Dictionary unpacking is not allowed in tensor specifications")
            return {
                evaluate(key, depth + 1): evaluate(value, depth + 1)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = evaluate(node.operand, depth + 1)
            if type(value) not in (int, float):
                raise ValueError("Unary minus is restricted to numeric tensor-specification values")
            return -value
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "T":
                raise ValueError("Only the T(...) tensor-specification constructor is allowed")
            if len(node.args) > 3 or any(keyword.arg != "stride" for keyword in node.keywords):
                raise ValueError("T(...) accepts only shape, dtype, and stride arguments")
            args = [evaluate(arg, depth + 1) for arg in node.args]
            kwargs = {keyword.arg: evaluate(keyword.value, depth + 1) for keyword in node.keywords}
            return tensor_spec(*args, **kwargs)
        raise ValueError(f"Unsupported tensor-specification syntax: {type(node).__name__}")

    return evaluate(tree)
