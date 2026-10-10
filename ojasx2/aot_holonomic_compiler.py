"""
ojasx2/aot_holonomic_compiler.py

AOTAutograd Joint Forward-Backward Graph Compiler & Custom Partitioner.
Hooks into `torch._functorch.aot_autograd` (`aot_module_simplified` & Dynamo `aot_autograd`)
to intercept, partition, and optimize both the Forward FX Graph and Backward FX Graph ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import default_partition

from .ops import rmsnorm
from .generated_holonomic_ops import holonomic_attention_softmax, holonomic_rope


@dataclass
class AOTCompilationLedger:
    """Audits AOTAutograd joint graph partitioning, saved boundary tensors, and backward contractions."""
    joint_graph_nodes: int = 0
    fw_graph_nodes: int = 0
    bw_graph_nodes: int = 0
    saved_tensors_across_cut: int = 0
    bw_gemm_contractions: int = 0
    fw_custom_ops: List[str] = field(default_factory=list)
    bw_ops_sample: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_graph_nodes": self.joint_graph_nodes,
            "fw_graph_nodes": self.fw_graph_nodes,
            "bw_graph_nodes": self.bw_graph_nodes,
            "saved_tensors_across_cut": self.saved_tensors_across_cut,
            "bw_gemm_contractions": self.bw_gemm_contractions,
            "fw_custom_ops": list(self.fw_custom_ops),
            "bw_ops_sample": list(self.bw_ops_sample),
        }


LAST_AOT_LEDGER = AOTCompilationLedger()


def _rewrite_fx_for_aot(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Lower trace-compatible RMSNorm and Softmax nodes into custom_op primitives before AOT tracing."""
    graph = gm.graph
    rmsnorm_targets = tuple(t for t in (getattr(F, "rms_norm", None), getattr(torch, "rms_norm", None)) if t is not None)

    for node in list(graph.nodes):
        if node.op == "call_function":
            if node.target in rmsnorm_targets:
                args = list(node.args)
                kwargs = dict(node.kwargs)
                x_arg = args[0]
                gamma_arg = args[2] if len(args) > 2 else kwargs.get("weight")
                eps_arg = args[3] if len(args) > 3 else kwargs.get("eps", 1e-6)
                if gamma_arg is not None:
                    with graph.inserting_after(node):
                        new_node = graph.call_function(rmsnorm, args=(x_arg, gamma_arg, eps_arg))
                        node.replace_all_uses_with(new_node)
                        graph.erase_node(node)
            elif node.target in (F.softmax, torch.softmax):
                args = list(node.args)
                kwargs = dict(node.kwargs)
                dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
                if dim in (-1,):
                    with graph.inserting_after(node):
                        new_node = graph.call_function(holonomic_attention_softmax, args=(args[0],))
                        node.replace_all_uses_with(new_node)
                        graph.erase_node(node)

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return gm


def make_holonomic_partition_fn(ledger: Optional[AOTCompilationLedger] = None) -> Callable[..., Tuple[torch.fx.GraphModule, torch.fx.GraphModule]]:
    target_ledger = ledger if ledger is not None else LAST_AOT_LEDGER

    def holonomic_partition_fn(
        joint_gm: torch.fx.GraphModule,
        joint_inputs: List[Any],
        **kwargs: Any,
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        joint_gm.graph.eliminate_dead_code()
        joint_gm.recompile()
        target_ledger.joint_graph_nodes = len(joint_gm.graph.nodes)

        fw_module, bw_module = default_partition(joint_gm, joint_inputs, **kwargs)
        fw_module.graph.eliminate_dead_code()
        bw_module.graph.eliminate_dead_code()
        fw_module.recompile()
        bw_module.recompile()

        # Count saved tensors crossing the forward-backward partition boundary
        bw_placeholders = [n for n in bw_module.graph.nodes if n.op == "placeholder"]
        # Typically the trailing placeholder(s) are tangents (grad_outputs); preceding are saved tensors
        target_ledger.saved_tensors_across_cut = max(0, len(bw_placeholders) - 1)
        return fw_module, bw_module

    return holonomic_partition_fn


def make_holonomic_aot_compilers(
    ledger: Optional[AOTCompilationLedger] = None,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    target_ledger = ledger if ledger is not None else LAST_AOT_LEDGER

    def holonomic_fw_compiler(gm: torch.fx.GraphModule, sample_inputs: List[Any]) -> Any:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        target_ledger.fw_graph_nodes = len(gm.graph.nodes)
        target_ledger.fw_custom_ops = [
            str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function" and ("ojasx" in str(n.target))
        ]
        return make_boxed_func(gm.forward)

    def holonomic_bw_compiler(gm: torch.fx.GraphModule, sample_inputs: List[Any]) -> Any:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        target_ledger.bw_graph_nodes = len(gm.graph.nodes)
        call_targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        target_ledger.bw_gemm_contractions = sum(
            1 for t in call_targets if any(k in t for k in ("aten.mm", "aten.bmm", "aten.addmm", "matmul"))
        )
        target_ledger.bw_ops_sample = call_targets[:12]
        return make_boxed_func(gm.forward)

    return holonomic_fw_compiler, holonomic_bw_compiler


def holonomic_aot_backend(gm: torch.fx.GraphModule, sample_inputs: List[torch.Tensor]) -> Any:
    """
    Full TorchDynamo + AOTAutograd backend (`torch.compile(model, backend="holonomic_aot")`).
    1. Rewrites FX graph to inject `ojasx_holonomic` and `ojasx_verified` custom ops.
    2. Traces the Joint Forward-Backward FX graph via AOTAutograd.
    3. Partitions and optimizes both `fw_module` and `bw_module`.
    """
    gm = _rewrite_fx_for_aot(gm)
    fw_comp, bw_comp = make_holonomic_aot_compilers(LAST_AOT_LEDGER)
    part_fn = make_holonomic_partition_fn(LAST_AOT_LEDGER)
    aot_backend = aot_autograd(fw_compiler=fw_comp, bw_compiler=bw_comp, partition_fn=part_fn)
    return aot_backend(gm, sample_inputs)


class _TupleOutputAdapter(nn.Module):
    """Wraps an nn.Module so its output is always a tuple for `aot_module_simplified`."""
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        out = self.inner(*args, **kwargs)
        return out if isinstance(out, tuple) else (out,)


def compile_module_aot_simplified(
    model: nn.Module,
    sample_args: Tuple[Any, ...],
    ledger: Optional[AOTCompilationLedger] = None,
) -> Callable[..., Any]:
    """
    Direct `aot_module_simplified` wrapper (matching PyTorch's official AOTAutograd recipe).
    Automatically handles single-tensor outputs and records the forward/backward FX graphs.
    """
    target_ledger = ledger if ledger is not None else LAST_AOT_LEDGER
    fw_comp, bw_comp = make_holonomic_aot_compilers(target_ledger)
    part_fn = make_holonomic_partition_fn(target_ledger)

    wrapped = _TupleOutputAdapter(model)
    compiled_tuple_fn = aot_module_simplified(
        wrapped,
        sample_args,
        fw_compiler=fw_comp,
        bw_compiler=bw_comp,
        partition_fn=part_fn,
    )

    def runner(*args: Any, **kwargs: Any) -> Any:
        res = compiled_tuple_fn(*args, **kwargs)
        return res[0] if isinstance(res, (tuple, list)) and len(res) == 1 else res

    return runner


try:
    import torch._dynamo
    torch._dynamo.register_backend(name="holonomic_aot", compiler_fn=holonomic_aot_backend)
except Exception:
    pass
