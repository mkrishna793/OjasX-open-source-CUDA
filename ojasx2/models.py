"""Semantics-preserving module adaptation; experimental shared projections."""
import copy
from typing import Sequence, Tuple
import torch
from torch import nn
from .ops import packed_relu


class PackedReLU(nn.Module):
    def forward(self, x):
        return packed_relu(x)


def optimize_relu_residuals(model: nn.Module, policy: str = 'conservative') -> nn.Module:
    """Copy the model and replace only non-inplace ReLUs, including the root.

    Conservative policy retains ReLU immediately before a Sequential Linear:
    that Linear already saves its input, so adding a packed mask can add storage.
    'pack' forces eligible replacement for experiments. Other graphs still need
    measured liveness analysis. Functional and inplace ReLUs remain unchanged.
    This is module adaptation, not a compiler backend.
    """
    if policy not in ('conservative', 'pack'):
        raise ValueError("policy must be 'conservative' or 'pack'")
    cloned = copy.deepcopy(model)

    def visit(module):
        if isinstance(module, nn.ReLU) and not module.inplace:
            return PackedReLU()
        children = list(module.named_children())
        for index, (name, child) in enumerate(children):
            if (policy == 'conservative' and isinstance(module, nn.Sequential)
                    and isinstance(child, nn.ReLU) and index+1 < len(children)
                    and isinstance(children[index+1][1], nn.Linear)):
                continue
            setattr(module, name, visit(child))
        return module

    return visit(cloned)


def shared_projections(x: torch.Tensor, weights: Sequence[torch.Tensor], accuracy: str = "strict") -> Tuple[torch.Tensor, ...]:
    """Group projections sharing X. Algebraically exact; floating reduction order may change.

    'strict' preserves independent PyTorch operations. 'algebraic' opts into a
    packed GEMM and requires tolerance validation on each device and workload.
    No reduction in leading FLOPs is claimed, only GEMM-call count.
    """
    if not weights:
        raise ValueError("At least one projection is required")
    if accuracy not in ("strict", "algebraic"):
        raise ValueError("accuracy must be 'strict' or 'algebraic'")
    for w in weights:
        if w.ndim != 2 or w.shape[1] != x.shape[-1] or w.device != x.device or w.dtype != x.dtype:
            raise ValueError("Projection weights must match X's last dimension, dtype and device")
    if accuracy == "strict":
        return tuple(torch.nn.functional.linear(x, w) for w in weights)
    from .projections import packed_projections
    return packed_projections(x,tuple(weights))
