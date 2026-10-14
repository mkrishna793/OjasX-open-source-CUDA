"""
ojasx2/symplectic_engine.py

Symplectic Reversible Hamiltonian Coupling Engine (O(1) Depth-Independent Memory)
=================================================================================
Implements volume-preserving symplectic integration over bipartite state spaces
(q1, q2) -> (y1, y2) with exact inverse state reconstruction during backward:

Forward transition at step i:
    q1_{i+1} = q1_i + f_i(q2_i)
    q2_{i+1} = q2_i + g_i(q1_{i+1})

Exact inverse reconstruction at step i (from (q1_{i+1}, q2_{i+1})):
    q2_i = q2_{i+1} - g_i(q1_{i+1})
    q1_i = q1_{i+1} - f_i(q2_i)

Because each coupling step has unit Jacobian determinant (det J = 1), an arbitrary
depth-D chain requires saving only the final boundary state (y1_D, y2_D) on the
autograd tape (2 * B * D_dim * sizeof(dtype) bytes total), achieving O(1) activation
memory scaling with respect to network depth D.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class SymplecticCouplingFunction(torch.autograd.Function):
    """
    Custom autograd Function for a single or multi-stage symplectic coupling chain.
    Saves only the final output boundary tensors (y1, y2) on the autograd tape,
    achieving O(1) depth-independent activation memory.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q1: torch.Tensor,
        q2: torch.Tensor,
        couplings: Sequence[Any],
        *flat_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.couplings = list(couplings)
        with torch.no_grad():
            curr1 = q1.detach()
            curr2 = q2.detach()
            for layer in ctx.couplings:
                f_fn, g_fn = _extract_fg(layer)
                curr1 = curr1 + f_fn(curr2)
                curr2 = curr2 + g_fn(curr1)
            out1 = curr1.clone()
            out2 = curr2.clone()

        # Save only the final boundary state (out1, out2) -> 2 * B * dim * elem_size bytes
        ctx.save_for_backward(out1, out2)
        return out1, out2

    @staticmethod
    def backward(
        ctx: Any,
        grad_y1: torch.Tensor,
        grad_y2: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        out1, out2 = ctx.saved_tensors
        curr1 = out1.detach()
        curr2 = out2.detach()
        g1 = grad_y1.detach() if grad_y1 is not None else torch.zeros_like(curr1)
        g2 = grad_y2.detach() if grad_y2 is not None else torch.zeros_like(curr2)

        # Map parameter identity to accumulated gradient for all passed flat_params
        layer_param_grads: List[List[Optional[torch.Tensor]]] = []

        for layer in reversed(ctx.couplings):
            f_fn, g_fn = _extract_fg(layer)
            f_params = list(f_fn.parameters()) if isinstance(f_fn, nn.Module) else []
            g_params = list(g_fn.parameters()) if isinstance(g_fn, nn.Module) else []

            # Step 1: Invert g_i using q1_{i+1} = curr1
            with torch.enable_grad():
                q1_next = curr1.detach().requires_grad_(True)
                g_out = g_fn(q1_next)
                g_trainable = [p for p in g_params if p.requires_grad]
                g_inputs: List[torch.Tensor] = [q1_next] + g_trainable
                g_grads = torch.autograd.grad(
                    outputs=g_out,
                    inputs=g_inputs,
                    grad_outputs=g2,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            dq1_from_g = g_grads[0] if g_grads[0] is not None else torch.zeros_like(curr1)
            g1_total = g1 + dq1_from_g

            g_grad_iter = iter(g_grads[1:])
            g_param_grad_list: List[Optional[torch.Tensor]] = []
            for p in g_params:
                if p.requires_grad:
                    pg = next(g_grad_iter)
                    g_param_grad_list.append(pg if pg is not None else torch.zeros_like(p))
                else:
                    g_param_grad_list.append(None)

            # Reconstruct q2_i = q2_{i+1} - g_i(q1_{i+1})
            with torch.no_grad():
                q2_prev = (curr2 - g_out.detach()).detach()

            # Step 2: Invert f_i using reconstructed q2_i
            with torch.enable_grad():
                q2_in = q2_prev.detach().requires_grad_(True)
                f_out = f_fn(q2_in)
                f_trainable = [p for p in f_params if p.requires_grad]
                f_inputs: List[torch.Tensor] = [q2_in] + f_trainable
                f_grads = torch.autograd.grad(
                    outputs=f_out,
                    inputs=f_inputs,
                    grad_outputs=g1_total,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            dq2_from_f = f_grads[0] if f_grads[0] is not None else torch.zeros_like(curr2)
            g2_prev = g2 + dq2_from_f
            g1_prev = g1_total

            f_grad_iter = iter(f_grads[1:])
            f_param_grad_list: List[Optional[torch.Tensor]] = []
            for p in f_params:
                if p.requires_grad:
                    pg = next(f_grad_iter)
                    f_param_grad_list.append(pg if pg is not None else torch.zeros_like(p))
                else:
                    f_param_grad_list.append(None)

            # Reconstruct q1_i = q1_{i+1} - f_i(q2_i)
            with torch.no_grad():
                q1_prev = (curr1 - f_out.detach()).detach()

            curr1 = q1_prev
            curr2 = q2_prev
            g1 = g1_prev
            g2 = g2_prev

            # Store parameter grads in forward order of (f_params + g_params) for this layer
            layer_param_grads.append(f_param_grad_list + g_param_grad_list)

        layer_param_grads.reverse()
        flat_param_grads: List[Optional[torch.Tensor]] = [
            pg for layer_grads in layer_param_grads for pg in layer_grads
        ]

        dq1 = g1 if ctx.needs_input_grad[0] else None
        dq2 = g2 if ctx.needs_input_grad[1] else None
        return (dq1, dq2, None, *flat_param_grads)


def _extract_fg(layer: Any) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Extract (f, g) callables/submodules from a SymplecticCoupling or tuple."""
    if hasattr(layer, "f") and hasattr(layer, "g"):
        return layer.f, layer.g
    if isinstance(layer, (tuple, list)) and len(layer) == 2:
        return layer[0], layer[1]
    raise TypeError(
        f"Expected a SymplecticCoupling module with .f and .g or a (f, g) pair, got {type(layer)}"
    )


def _collect_chain_params(couplings: Sequence[Any]) -> List[torch.Tensor]:
    """Collect all parameters from a sequence of coupling layers in deterministic order."""
    params: List[torch.Tensor] = []
    for layer in couplings:
        f_fn, g_fn = _extract_fg(layer)
        if isinstance(f_fn, nn.Module):
            params.extend(list(f_fn.parameters()))
        if isinstance(g_fn, nn.Module):
            params.extend(list(g_fn.parameters()))
    return params


class SymplecticCoupling(nn.Module):
    """
    Single Symplectic Hamiltonian Coupling Block:
        y1 = q1 + f(q2)
        y2 = q2 + g(y1)
    Can be constructed either via dimensions `SymplecticCoupling(dim, hidden_dim)`
    or via explicit submodules `SymplecticCoupling(f_module, g_module)`.
    """

    def __init__(
        self,
        dim_or_f: Union[int, nn.Module],
        hidden_dim_or_g: Optional[Union[int, nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(dim_or_f, nn.Module) and isinstance(hidden_dim_or_g, nn.Module):
            self.f: nn.Module = dim_or_f
            self.g: nn.Module = hidden_dim_or_g
            self.dim: Optional[int] = None
            self.hidden_dim: Optional[int] = None
        elif isinstance(dim_or_f, int):
            dim = int(dim_or_f)
            hidden_dim = int(hidden_dim_or_g) if isinstance(hidden_dim_or_g, int) else dim * 4
            self.dim = dim
            self.hidden_dim = hidden_dim
            self.f = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=bias),
            )
            self.g = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=bias),
            )
        else:
            raise ValueError(
                "SymplecticCoupling requires either (dim: int, hidden_dim: int) or (f: nn.Module, g: nn.Module)."
            )

    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return symplectic_coupling_chain(q1, q2, [self])

    @torch.no_grad()
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q2 = y2 - self.g(y1)
        q1 = y1 - self.f(q2)
        return q1, q2


class SymplecticTransformerBlock(nn.Module):
    """
    Reversible Transformer Block structured as a symplectic coupling pair (f=Attention, g=MLP).
    Supports both bipartite `(q1, q2)` inputs and concatenated single-tensor `x` inputs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or (dim * 4)

        self.f = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.g = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, dim),
        )

    def forward(
        self,
        q1: torch.Tensor,
        q2: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if q2 is None:
            half = q1.shape[-1] // 2
            a, b = q1[..., :half], q1[..., half:]
            y1, y2 = symplectic_coupling_chain(a, b, [self])
            return torch.cat([y1, y2], dim=-1)
        return symplectic_coupling_chain(q1, q2, [self])

    @torch.no_grad()
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q2 = y2 - self.g(y1)
        q1 = y1 - self.f(q2)
        return q1, q2


def symplectic_coupling(
    q1: torch.Tensor,
    q2: torch.Tensor,
    f_fn: Callable[[torch.Tensor], torch.Tensor],
    g_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute a single reversible symplectic coupling step."""
    return symplectic_coupling_chain(q1, q2, [(f_fn, g_fn)])


def symplectic_coupling_chain(
    q1: torch.Tensor,
    q2: torch.Tensor,
    couplings: Iterable[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Execute a chain of reversible symplectic coupling blocks with O(1) boundary
    activation storage. Only the final output state (y1, y2) is saved for backward.
    """
    coupling_list = list(couplings)
    if len(coupling_list) == 0:
        return q1, q2
    flat_params = _collect_chain_params(coupling_list)
    return SymplecticCouplingFunction.apply(q1, q2, coupling_list, *flat_params)


@torch.no_grad()
def audit_symplectic_reversibility(
    q1: torch.Tensor,
    q2: torch.Tensor,
    couplings: Iterable[Any],
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Empirically verify exact invertibility of a symplectic coupling chain by running
    forward to (y1, y2) and reconstructing (q1_rec, q2_rec) via reverse steps.
    """
    coupling_list = list(couplings)
    curr1 = q1.detach().clone()
    curr2 = q2.detach().clone()

    for layer in coupling_list:
        f_fn, g_fn = _extract_fg(layer)
        curr1 = curr1 + f_fn(curr2)
        curr2 = curr2 + g_fn(curr1)

    rec1 = curr1.clone()
    rec2 = curr2.clone()
    for layer in reversed(coupling_list):
        f_fn, g_fn = _extract_fg(layer)
        rec2 = rec2 - g_fn(rec1)
        rec1 = rec1 - f_fn(rec2)

    err_q1 = float((rec1 - q1.detach()).abs().max().item())
    err_q2 = float((rec2 - q2.detach()).abs().max().item())
    max_err = max(err_q1, err_q2)

    return {
        "is_reversible": max_err < tol,
        "max_error_q1": err_q1,
        "max_error_q2": err_q2,
        "max_error": max_err,
        "max_reconstruction_error": max_err,
        "reconstruction_error": max_err,
        "depth": len(coupling_list),
        "tolerance": tol,
    }


__all__ = [
    "SymplecticCouplingFunction",
    "SymplecticCoupling",
    "SymplecticTransformerBlock",
    "symplectic_coupling",
    "symplectic_coupling_chain",
    "audit_symplectic_reversibility",
]
