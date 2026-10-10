"""Experimental exact-real RMSNorm contraction; sampled CPU/MPS qualification."""
import math
import torch
import torch.nn.functional as F
from .nonlinear import _all_finite, gram_layernorm_chain_cost
from .devices import algebraic_device_supported


def _finite_inputs(values):
    if values[0].device.type == 'mps':
        # Enqueue all reductions before a single host check. Unlike testing
        # each tensor's scalar separately, this avoids repeated GPU waits.
        endpoints = [endpoint for value in values for endpoint in torch.aminmax(value)]
        return bool(torch.isfinite(torch.stack(endpoints)).all())
    return all(_all_finite(value) for value in values)


def _safe_moment(moment, eps):
    if moment.device.type == 'mps':
        low, high = torch.aminmax(moment)
        return bool(torch.isfinite(low) & torch.isfinite(high) & (low > 10*eps))
    return _all_finite(moment) and bool((moment > 10*eps).all())


def rms_core_cost(m, k, n, p, needs=(True, True, True, True)):
    # Identical four main products to the Gram LayerNorm graph, without centering.
    result = gram_layernorm_chain_cost(m, k, n, p, needs)
    result['scope'] = ('Main-GEMM forward/native-AD model; excludes gamma/vector/pointwise '
                       'operations, copies, finite/moment guards and dispatch')
    return result


def rms_core_linear_chain(x, w1, w2, gamma=None, bias2=None, eps=1e-5,
                          accuracy='strict', evidence=None):
    """Linear(no bias)-RMSNorm-Linear, with all original parameters trainable.

    torch Linear layout: w1[n,k], w2[p,n]; X may have leading batch/token axes.
    Ordinary autograd differentiates
    the smaller graph, including higher derivatives. Reassociation changes
    rounding. Guards are heuristic; no universal floating-point error bound.
    """
    if accuracy not in ('strict', 'algebraic'):
        raise ValueError('Invalid accuracy contract')
    def native():
        return F.linear(F.rms_norm(F.linear(x, w1), (w1.shape[0],), gamma, eps), w2, bias2)
    eligible = (accuracy == 'algebraic' and isinstance(eps, (int, float))
                and math.isfinite(eps) and eps > 0
                and x.ndim >= 2 and w1.ndim == w2.ndim == 2
                and min(*x.shape, *w1.shape, *w2.shape) > 0
                and x.shape[-1] == w1.shape[1] and w2.shape[1] == w1.shape[0]
                and (x.ndim == 2 or x.is_contiguous())
                and algebraic_device_supported(x) and x.device == w1.device == w2.device
                and x.dtype in (torch.float32, torch.float64) and x.dtype == w1.dtype == w2.dtype
                and all(a is None or (a.ndim == 1 and a.shape[0] == size
                        and a.dtype == x.dtype and a.device == x.device)
                        for a, size in ((gamma, w1.shape[0]), (bias2, w2.shape[0]))))
    reason = 'strict' if accuracy == 'strict' else 'unsupported'
    if eligible:
        flat = x.reshape(-1, x.shape[-1])
        cost = rms_core_cost(flat.shape[0], flat.shape[1], w1.shape[0], w2.shape[0],
                            (x.requires_grad, w1.requires_grad, w2.requires_grad,
                             gamma is not None and gamma.requires_grad))
        if cost['scalar_saving'] > 0:
            with torch.no_grad():
                finite = _finite_inputs(tuple(a for a in (x, w1, w2, gamma, bias2) if a is not None))
            if finite:
                gram = w1.T @ w1
                moment = ((flat @ gram)*flat).sum(-1, keepdim=True)/w1.shape[0]
                with torch.no_grad():
                    safe = _safe_moment(moment, eps)
                if safe:
                    projected = (w2 if gamma is None else w2*gamma) @ w1
                    y = (flat @ projected.T)*torch.rsqrt(moment+eps)
                    if bias2 is not None:
                        y = y+bias2
                    with torch.no_grad():
                        finite = _all_finite(y)
                    if finite:
                        if evidence is not None:
                            evidence.append({'path': 'rms_core', 'cost': cost,
                                             'conditioning_guard': 'heuristic, not certified'})
                        return y.reshape(*x.shape[:-1], w2.shape[0])
                reason = 'conditioning' if not safe else 'nonfinite'
            else:
                reason = 'nonfinite'
        else:
            reason = 'unprofitable'
    if evidence is not None:
        evidence.append({'path': reason})
    return native()
