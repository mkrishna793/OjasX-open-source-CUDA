"""OjasX3 Unified Architecture-Agnostic Compiler."""
from ojasx3.compiler.universal import (
    UniversalCompilerConfig,
    UniversalCompiledModule,
    compile_universal,
    rewrite_linear_cross_entropy,
    chain_cost,
    factored_gemm_pullback,
    linear_chain,
    demand_plan,
    StructuredDerivative,
    softmax_derivative,
    rmsnorm_input_derivative,
    layernorm_input_derivative,
    checked_value_and_grad,
    stable_scaled_layer_norm,
)

__all__ = [
    "UniversalCompilerConfig",
    "UniversalCompiledModule",
    "compile_universal",
    "rewrite_linear_cross_entropy",
    "chain_cost",
    "factored_gemm_pullback",
    "linear_chain",
    "demand_plan",
    "StructuredDerivative",
    "softmax_derivative",
    "rmsnorm_input_derivative",
    "layernorm_input_derivative",
    "checked_value_and_grad",
    "stable_scaled_layer_norm",
]
