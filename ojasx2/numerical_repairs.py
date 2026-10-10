"""MPS-only stable normalization and conservative qualification routing.

Stable normalization is an exact-real rewrite, not bitwise native equivalence.
Its higher-order pullback retains the original u/scale dependencies.
"""
import math
import torch
import torch.nn.functional as F
from .gradient_factor_compiler import native_projection
from .spectral_compiler import spectral_projection, spectral_plan
from .invariant_radial_compiler import native_radial, radial_gate


class _StableScaleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, scale, eps):
        ctx.save_for_backward(u, scale)
        ctx.eps = eps
        v = u-u.mean(-1, keepdim=True)
        denominator = scale.square()*v.square().mean(-1, keepdim=True)+eps
        return scale*v*torch.rsqrt(denominator)

    @staticmethod
    def backward(ctx, p):
        # Recompute small statistics from original differentiable operands.
        # No detached first-order tape is substituted into higher derivatives.
        u, scale = ctx.saved_tensors
        v = u-u.mean(-1, keepdim=True)
        variance = v.square().mean(-1, keepdim=True)
        denominator = scale.square()*variance+ctx.eps
        inv = torch.rsqrt(denominator)
        pv = (p*v).mean(-1, keepdim=True)
        du = scale*inv*(p-p.mean(-1, keepdim=True)
                       -v*(scale.square()/denominator)*pv)
        # eps*inv**3 is algebraically valid but can overflow before cancellation.
        # (eps/denominator)*inv avoids that avoidable intermediate overflow.
        ds = (ctx.eps/denominator)*inv*(p*v).sum(-1, keepdim=True)
        return du, ds, None


def stable_scaled_layer_norm(u, scale, eps=1e-5):
    if u.device.type != "mps" or scale.device.type != "mps":
        raise ValueError("MPS only; CPU numerical fallback forbidden")
    if u.dtype != torch.float32 or scale.dtype != torch.float32:
        raise ValueError("Qualified research dtype is FP32")
    if u.ndim < 1 or u.shape[-1] == 0 or scale.shape != u.shape[:-1]+(1,):
        raise ValueError("Expected nonempty coordinate axis and one scale per row")
    if not math.isfinite(eps) or not 1e-30 <= eps <= 1:
        raise ValueError("Representable bounded positive epsilon required")
    return _StableScaleNorm.apply(u, scale, float(eps))


def radial_layer_norm(x, w, bias, q, kind="sigmoid", gate=None,
                      eps=1e-5, accuracy="strict", evidence=None):
    """Existing radial projection followed by unaffined LayerNorm.

    Native is default. Stable algebraic mode needs whole-program qualification.
    Parameter objects are never replaced by independently trained cores.
    """
    if accuracy not in ("strict", "algebraic"):
        raise ValueError("Unknown accuracy")
    if any(t.device.type != "mps" for t in (x, w, bias, q)):
        raise ValueError("MPS only")
    def native():
        return F.layer_norm(native_radial(x, w, bias, q, kind, gate),
                            (q.shape[1],), eps=eps)
    values = (x, w, bias, q)+((gate,) if gate is not None else ())
    if accuracy == "strict":
        if evidence is not None: evidence.append({"path": "native_normalized_program"})
        return native()
    if (any(t.dtype != torch.float32 or t.numel() == 0 for t in values)
            or torch.is_autocast_enabled("mps")
            or not math.isfinite(eps) or not 1e-30 <= eps <= 1):
        return native()
    with torch.no_grad():
        safe = all(bool(torch.isfinite(t).all() & (t.abs().amax() <= 8))
                   for t in values)
    if not safe:
        return native()
    a = torch.cat((x, torch.ones_like(x[:, :1])), 1)
    v = torch.cat((w, bias[None, :]), 0)
    c = v@v.T
    energy = ((a@c)*a).sum(-1, keepdim=True)/w.shape[1]
    with torch.no_grad():
        bound = a.square().sum(-1, keepdim=True)*v.square().sum()/w.shape[1]
        good = bool(torch.isfinite(energy).all() & (energy >= 0).all()
                    & ((energy > 1e-5*bound) | (bound == 0)).all())
    if not good:
        return native()
    scale, _, _ = radial_gate(energy, kind, gate)
    if evidence is not None: evidence.append({"path": "stable_joint_radial_layernorm"})
    return stable_scaled_layer_norm(a@(v@q), scale, eps)


def reliable_projection(x, w, bias, q, kind="gelu", degree=8,
                        accuracy="strict", derivative_order=1,
                        whole_program_qualified=False, evidence=None):
    """Conservative reliability policy; not automatic universal certification.

    Low-degree requests select the tested degree-8 family when approximation
    was explicitly accepted. Wide, unqualified, higher-order and unadmitted
    two-input SiLU workloads retain the original native function.
    """
    if accuracy not in ("strict", "approximate"):
        raise ValueError("Unknown accuracy")
    if derivative_order < 1:
        raise ValueError("Positive derivative order required")
    if any(t.device.type != "mps" for t in (x, w, bias, q)):
        raise ValueError("MPS only")
    selected_degree = 8 if degree < 8 else degree
    reason = None
    if accuracy == "strict" or not whole_program_qualified:
        reason = "native_unqualified_or_strict"
    elif derivative_order > 1:
        reason = "native_original_higher_derivatives"
    elif kind == "silu" and x.shape[1] >= 2:
        reason = "native_silu_without_repeated_speed_admission"
    elif kind not in ("gelu", "silu") or selected_degree not in (8, 12):
        reason = "native_unsupported_approximation"
    elif not spectral_plan(x.shape[0], x.shape[1], w.shape[1],
                           q.shape[1], selected_degree).eligible:
        reason = "native_feature_growth_or_cost"
    if reason:
        if evidence is not None: evidence.append({"path": reason})
        return native_projection(x, w, bias, q, kind)
    if evidence is not None:
        evidence.append({"path": "degree_repaired", "requested": degree,
                         "selected": selected_degree})
    return spectral_projection(x, w, bias, q, kind=kind,
                               degree=selected_degree, accuracy="approximate",
                               evidence=evidence)


def checked_value_and_grad(native, candidate, operands, create_graph=False,
                           evidence=None):
    """Check a scalar loss and all original gradients on each invocation.

    Rejected candidates return native loss/gradients. This is an explicit,
    costly reliability API, not a production graph optimizer or optimizer
    trajectory certificate. Callbacks must be pure/deterministic, with fixed
    random masks and no mutation. Native NaN semantics survive rejection.
    """
    if any(t.device.type != "mps" for t in operands):
        raise ValueError("MPS only")
    active = tuple(t for t in operands if t.requires_grad)
    def run(fn, higher):
        value = fn(*operands)
        if value.numel() != 1 or value.device.type != "mps":
            raise ValueError("Expected scalar MPS loss")
        grads = (torch.autograd.grad(value, active, create_graph=higher,
                                    allow_unused=True) if active else ())
        it = iter(grads)
        return value, tuple(next(it) if t.requires_grad else None for t in operands)
    reference = run(native, create_graph)
    if create_graph:
        if evidence is not None: evidence.append({"path": "native_higher_order"})
        return reference
    try:
        trial = run(candidate, False)
    except (RuntimeError, ValueError) as error:
        if evidence is not None:
            evidence.append({"path": "checked_native_candidate_error",
                             "raw_candidate_passed": False,
                             "reason": str(error)})
        return reference
    def match(a, b):
        if a is None or b is None:
            return a is b
        if a.shape != b.shape:
            return False
        with torch.no_grad():
            rel = torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-20)
            return bool(torch.isfinite(a).all() & torch.isfinite(b).all()) and bool(
                torch.allclose(a, b, rtol=2e-4, atol=2e-4)) and rel.item() <= 2e-4
    accepted = match(trial[0], reference[0]) and all(
        match(a, b) for a, b in zip(trial[1], reference[1]))
    if evidence is not None:
        evidence.append({"path": "checked_candidate" if accepted else "checked_native",
                         "raw_candidate_passed": accepted})
    return trial if accepted else reference
