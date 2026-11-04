"""Conservative drop-in Dynamo/AOT backend for proof-gated core schedules."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
import operator
import os
from pathlib import Path
import platform
import sysconfig
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from .activation_math import ACTIVATION_NAMES
from .egraph import Expr
from .m4 import (
    M4Config,
    M4Input,
    M4Plan,
    M4Problem,
    activation,
    default_rewrite_rules,
    plan_m4,
    tensor_input,
)
from .planning import ExecutionObjective


@dataclass(frozen=True)
class HardwareTargetProfile:
    """Hardware execution profile across GPU, accelerator, and CPU targets."""

    device_type: str
    architecture: str
    preferred_executor: str
    supports_inductor: bool
    supports_tilelang: bool
    prefer_aot_boxed: bool
    vector_width_bits: int
    sram_budget_bytes: int
    simd_isa: str = "scalar"

    def to_dict(self) -> Dict[str, object]:
        return {
            "device_type": self.device_type,
            "architecture": self.architecture,
            "preferred_executor": self.preferred_executor,
            "supports_inductor": self.supports_inductor,
            "supports_tilelang": self.supports_tilelang,
            "prefer_aot_boxed": self.prefer_aot_boxed,
            "vector_width_bits": self.vector_width_bits,
            "sram_budget_bytes": self.sram_budget_bytes,
            "simd_isa": self.simd_isa,
        }


def _has_cpython_headers() -> bool:
    include_dir = sysconfig.get_paths().get("include", "")
    return (bool(include_dir) and Path(include_dir, "Python.h").exists()) or bool(
        os.environ.get("CPATH")
    )


def _detect_cpu_isa(hint: Optional[str] = None) -> Tuple[str, str, int, int]:
    if hint == "avx2":
        return "x86_64_avx2", "avx2", 256, 32768
    if hint == "neon":
        return "arm64_neon", "neon", 128, 65536
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64_avx2", "avx2", 256, 32768
    if machine in {"arm64", "aarch64"}:
        return "arm64_neon", "neon", 128, 65536
    return "cpu_generic", "scalar", 128, 32768


def detect_hardware_target(
    device: Optional[Union[str, torch.device, torch.Tensor, Sequence[object]]] = None,
) -> HardwareTargetProfile:
    """Inspect PyTorch/hardware environment and determine optimal execution settings."""
    isa_hint: Optional[str] = None
    if isinstance(device, torch.Tensor):
        raw_type = device.device.type
    elif isinstance(device, (list, tuple)):
        tensor_arg = next((item for item in device if isinstance(item, torch.Tensor)), None)
        raw_type = tensor_arg.device.type if tensor_arg is not None else ""
    elif isinstance(device, torch.device):
        raw_type = device.type
    elif isinstance(device, str):
        cleaned = device.strip().lower()
        parts = cleaned.split(":")
        raw_type = parts[0]
        if len(parts) > 1 and parts[1] in {"avx2", "neon"}:
            isa_hint = parts[1]
    else:
        raw_type = ""

    if not raw_type:
        if torch.cuda.is_available():
            raw_type = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            raw_type = "mps"
        elif hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
            raw_type = "xpu"
        else:
            raw_type = "cpu"

    if raw_type == "hip":
        raw_type = "rocm"

    if raw_type == "cuda":
        return HardwareTargetProfile(
            device_type="cuda",
            architecture="nvidia_cuda",
            preferred_executor="tilelang",
            supports_inductor=True,
            supports_tilelang=True,
            prefer_aot_boxed=False,
            vector_width_bits=1024,
            sram_budget_bytes=163840,
            simd_isa="simt",
        )
    if raw_type == "rocm":
        return HardwareTargetProfile(
            device_type="rocm",
            architecture="amd_rocm",
            preferred_executor="tilelang",
            supports_inductor=True,
            supports_tilelang=True,
            prefer_aot_boxed=False,
            vector_width_bits=2048,
            sram_budget_bytes=65536,
            simd_isa="simt",
        )
    if raw_type == "mps":
        return HardwareTargetProfile(
            device_type="mps",
            architecture="apple_silicon_mps",
            preferred_executor="tilelang",
            supports_inductor=True,
            supports_tilelang=True,
            prefer_aot_boxed=True,
            vector_width_bits=1024,
            sram_budget_bytes=32768,
            simd_isa="metal_simd",
        )
    if raw_type == "xpu":
        return HardwareTargetProfile(
            device_type="xpu",
            architecture="intel_xpu",
            preferred_executor="inductor",
            supports_inductor=True,
            supports_tilelang=False,
            prefer_aot_boxed=False,
            vector_width_bits=512,
            sram_budget_bytes=131072,
            simd_isa="simt",
        )
    if raw_type == "openreg":
        return HardwareTargetProfile(
            device_type="openreg",
            architecture="openreg",
            preferred_executor="aot_boxed",
            supports_inductor=False,
            supports_tilelang=False,
            prefer_aot_boxed=True,
            vector_width_bits=128,
            sram_budget_bytes=32768,
            simd_isa="scalar",
        )
    if raw_type == "meta":
        return HardwareTargetProfile(
            device_type="meta",
            architecture="meta",
            preferred_executor="aot_boxed",
            supports_inductor=False,
            supports_tilelang=False,
            prefer_aot_boxed=True,
            vector_width_bits=64,
            sram_budget_bytes=0,
            simd_isa="none",
        )

    arch, simd_isa, vec_bits, sram_bytes = _detect_cpu_isa(isa_hint)
    has_headers = _has_cpython_headers()
    return HardwareTargetProfile(
        device_type="cpu",
        architecture=arch,
        preferred_executor="aot_boxed",
        supports_inductor=has_headers,
        supports_tilelang=True,
        prefer_aot_boxed=True,
        vector_width_bits=vec_bits,
        sram_budget_bytes=sram_bytes,
        simd_isa=simd_isa,
    )


_NATIVE_TARGETS: Dict[object, str] = {
    torch.relu: "relu",
    F.relu: "relu",
    torch.sigmoid: "sigmoid",
    torch.tanh: "tanh",
    torch.exp: "exp",
    torch.log: "log",
    torch.sqrt: "sqrt",
    torch.sin: "sin",
    torch.cos: "cos",
    F.silu: "silu",
    F.gelu: "gelu",
    torch.square: "square",
    torch.neg: "neg",
    operator.neg: "neg",
}
if hasattr(F, "sigmoid"):
    _NATIVE_TARGETS[F.sigmoid] = "sigmoid"
if hasattr(F, "tanh"):
    _NATIVE_TARGETS[F.tanh] = "tanh"
for _name in ACTIVATION_NAMES:
    _NATIVE_TARGETS[getattr(torch.ops.aten, _name).default] = _name


@dataclass(frozen=True)
class BackendRegionReport:
    operations: Tuple[str, ...]
    replaced: bool
    reason: str
    baseline_expression: str
    selected_expression: str
    baseline_cost: Mapping[str, int]
    selected_cost: Mapping[str, int]
    proof_obligations: int


@dataclass(frozen=True)
class BackendCompilationReport:
    changed: bool
    regions: Tuple[BackendRegionReport, ...]
    original_nodes: Tuple[str, ...]
    hardware_profile: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class BackendConfig:
    planner: M4Config = field(
        default_factory=lambda: M4Config(max_iterations=8, exact_node_limit=32)
    )
    min_chain_length: int = 2
    use_inductor: bool = False
    target_device: Optional[str] = None
    auto_hardware: bool = True
    enable_tilelang: bool = True

    def __post_init__(self) -> None:
        if self.min_chain_length < 2:
            raise ValueError("Drop-in replacement requires at least two native activations")


def _activation_node(node: torch.fx.Node) -> Optional[Tuple[str, str]]:
    if node.op == "call_function":
        kind = _NATIVE_TARGETS.get(node.target)
    elif node.op == "call_method" and node.target in ACTIVATION_NAMES:
        kind = str(node.target)
    else:
        kind = None
    if kind is None or not node.args:
        return None
    approximate = str(node.kwargs.get("approximate", "none"))
    if kind == "gelu" and len(node.args) > 1:
        approximate = str(node.args[1])
    if kind != "gelu":
        approximate = "none"
    if approximate not in {"none", "tanh"}:
        return None
    return kind, approximate


def _metadata(node: torch.fx.Node) -> Optional[Tuple[Tuple[int, ...], torch.dtype, bool]]:
    value = node.meta.get("val", node.meta.get("example_value"))
    if value is None:
        value = node.meta.get("tensor_meta")
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
    except (TypeError, RuntimeError):
        return None
    return shape, value.dtype, bool(getattr(value, "requires_grad", True))


def _chains(graph: torch.fx.Graph, minimum: int) -> Iterable[Tuple[torch.fx.Node, ...]]:
    visited = set()
    for node in graph.nodes:
        if node in visited or _activation_node(node) is None:
            continue
        chain = [node]
        current = node
        while len(current.users) == 1:
            successor = next(iter(current.users))
            if _activation_node(successor) is None or successor.args[0] is not current:
                break
            chain.append(successor)
            current = successor
        visited.update(chain)
        if len(chain) >= minimum:
            yield tuple(chain)


def _expression(chain: Sequence[torch.fx.Node], spec: M4Input) -> Expr:
    result = tensor_input(spec)
    for node in chain:
        kind, approximate = _activation_node(node) or (_raise("Invalid activation chain"))
        result = activation(result, kind, approximate=approximate)
    return result


def _raise(message: str):
    raise RuntimeError(message)


def _emit_native(graph: torch.fx.Graph, source: torch.fx.Node, expr: Expr) -> torch.fx.Node:
    if expr.op == "activation_chain":
        return graph.call_function(
            torch.ops.ojasx_verified.activation_chain.default,
            (source, list(dict(expr.attrs)["operations"]), expr.policy),
        )
    if expr.op in ACTIVATION_NAMES:
        kwargs = {"approximate": dict(expr.attrs).get("approximate", "none")} if expr.op == "gelu" else {}
        return graph.call_function(getattr(torch.ops.aten, expr.op).default, (source,), kwargs)
    raise RuntimeError(f"Cannot lower selected backend expression {expr.op!r}")


def _replace_chain(
    gm: torch.fx.GraphModule,
    chain: Sequence[torch.fx.Node],
    plan: M4Plan,
) -> None:
    graph = gm.graph
    with graph.inserting_before(chain[-1]):
        memo: Dict[Expr, torch.fx.Node] = {}

        def emit(expr: Expr) -> torch.fx.Node:
            if expr in memo:
                return memo[expr]
            if expr.op == "input":
                result = chain[0].args[0]
            else:
                args = tuple(emit(arg) for arg in expr.args)
                if len(args) != 1:
                    raise RuntimeError("Activation backend only lowers unary regions")
                result = _emit_native(graph, args[0], expr)
            memo[expr] = result
            return result

        replacement = emit(plan.selected_expression)
    chain[-1].replace_all_uses_with(replacement)
    for node in reversed(chain):
        if not node.users:
            graph.erase_node(node)
    gm.recompile()


def _region_report(chain: Sequence[torch.fx.Node], plan: M4Plan, replaced: bool) -> BackendRegionReport:
    operations = tuple(
        kind if approximate == "none" else f"{kind}:{approximate}"
        for kind, approximate in (_activation_node(node) for node in chain)
    )
    return BackendRegionReport(
        operations=operations,
        replaced=replaced,
        reason="strict_objective_improvement" if replaced else "original_not_strictly_worse",
        baseline_expression=plan.baseline_expression.structural(),
        selected_expression=plan.selected_expression.structural(),
        baseline_cost=plan.baseline_cost.to_dict(),
        selected_cost=plan.selected_cost.to_dict(),
        proof_obligations=len(plan.proof_obligations),
    )


class OjasXBackend:
    """Dynamo backend with original-graph fallback and AOT/Inductor execution."""

    def __init__(self, config: BackendConfig = BackendConfig()) -> None:
        self.config = config
        self.reports: List[BackendCompilationReport] = []
        self._rules = tuple(
            rule for rule in default_rewrite_rules() if rule.family == "streaming"
        )

    def _resolve_hardware_profile(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> HardwareTargetProfile:
        device_hint: Optional[Union[str, torch.device, Sequence[object]]] = None
        if self.config.target_device is not None:
            device_hint = self.config.target_device
        elif example_inputs:
            device_hint = example_inputs
        else:
            for node in gm.graph.nodes:
                val = node.meta.get("val", node.meta.get("example_value"))
                if hasattr(val, "device"):
                    device_hint = val.device
                    break
            if device_hint is None and not self.config.auto_hardware:
                device_hint = "cpu"
        profile = detect_hardware_target(device_hint)
        if not self.config.enable_tilelang and profile.supports_tilelang:
            fallback_exec = "inductor" if profile.supports_inductor and not profile.prefer_aot_boxed else "aot_boxed"
            profile = dataclass_replace(
                profile,
                supports_tilelang=False,
                preferred_executor=(
                    fallback_exec
                    if profile.preferred_executor == "tilelang"
                    else profile.preferred_executor
                ),
            )
        if self.config.use_inductor and profile.supports_inductor:
            profile = dataclass_replace(
                profile,
                preferred_executor="inductor",
                prefer_aot_boxed=False,
            )
        return profile

    def optimize(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.fx.GraphModule:
        original_nodes = tuple(node.name for node in gm.graph.nodes)
        hw_profile = self._resolve_hardware_profile(gm, example_inputs)
        reports: List[BackendRegionReport] = []
        chains = list(_chains(gm.graph, self.config.min_chain_length))
        for chain in chains:
            # Earlier disjoint rewrites can invalidate only a previously shared
            # boundary. Check that every node is still attached before planning.
            if any(node.graph is not gm.graph for node in chain):
                continue
            metadata = _metadata(chain[0])
            if metadata is None:
                continue
            shape, dtype, requires_grad = metadata
            spec = M4Input("backend_input", shape, dtype, requires_grad)
            problem = M4Problem((spec,), _expression(chain, spec))
            plan = plan_m4(problem, config=self.config.planner, rules=self._rules)
            replace = plan.discovered
            if replace:
                if any(not obligation.proofs for obligation in plan.proof_obligations):
                    raise RuntimeError("The backend cannot emit a schedule without proofs")
                _replace_chain(gm, chain, plan)
            reports.append(_region_report(chain, plan, replace))
        report = BackendCompilationReport(
            bool(reports) and any(region.replaced for region in reports),
            tuple(reports),
            original_nodes,
            hardware_profile=hw_profile.to_dict(),
        )
        self.reports.append(report)
        gm._ojasx_backend_report = report
        return gm

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]):
        optimized = self.optimize(gm, example_inputs)
        if self.config.use_inductor:
            from torch._inductor.compile_fx import compile_fx

            def inductor_compiler(graph_module, inputs):
                compiled = compile_fx(graph_module, inputs)

                def unboxed(*args):
                    return compiled(*args)

                # AOT and Inductor use different wrapper layers. Delegate
                # positionally, then expose the standard boxed AOT contract.
                return make_boxed_func(unboxed)

            compiler = aot_autograd(
                fw_compiler=inductor_compiler,
                bw_compiler=inductor_compiler,
            )
        else:
            def boxed(graph_module, inputs):
                return make_boxed_func(graph_module.forward)

            compiler = aot_autograd(fw_compiler=boxed, bw_compiler=boxed)
        return compiler(optimized, example_inputs)


backend = OjasXBackend()


def register_backend() -> None:
    if "ojasx_verified" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(name="ojasx_verified", compiler_fn=backend)
    if "ojasx3" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(name="ojasx3", compiler_fn=backend)


register_backend()


__all__ = [
    "BackendCompilationReport",
    "BackendConfig",
    "BackendRegionReport",
    "ExecutionObjective",
    "HardwareTargetProfile",
    "OjasXBackend",
    "backend",
    "detect_hardware_target",
    "register_backend",
]
