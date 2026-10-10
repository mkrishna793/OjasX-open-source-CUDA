"""Objective-aware linear CE schedules. MPS numerical execution only.

`strict` preserves the native expression. `algebraic` admits finite FP32,
first-order index-target CE after guards. This is an opt-in research primitive,
not an all-operator backend. Real identities are proved in LossCompiler.lean;
floating point is qualified separately. Higher derivatives use strict mode.
"""
import math
import torch
import torch.nn.functional as F


def head_plan(rows, width, classes, active, chunk_rows=128, need_h=True, need_w=True,
              need_bias=False):
    if min(rows, width, classes, active) < 0 or active > rows or chunk_rows < 1:
        raise ValueError('Invalid head dimensions')
    direct = 2 * rows * width * classes * (1 + int(need_h) + int(need_w))
    selected = 2 * active * width * classes * (1 + int(need_h) + int(need_w))
    replay = selected + (2 * active * width * classes if need_h or need_w or need_bias else 0)
    return dict(native_leading_flops=direct, selected_leading_flops=selected,
                replay_leading_flops=replay, native_logit_elements=rows*classes,
                selected_logit_elements=active*classes,
                replay_logit_elements=min(active, chunk_rows)*classes,
                scope='Leading dense matrix work, excluding guards, indexing, pointwise loss, bias and accumulation; not a timing prediction')


def resident_state_ledger(parameters, parameter_bits=16, gradient_bits=16,
                          moment_bits=32, moments=2, master_bits=32, budget_bytes=32_000_000_000):
    values = (parameters, parameter_bits, gradient_bits, moment_bits, moments, master_bits, budget_bytes)
    if any(not isinstance(v, int) or v < 0 for v in values):
        raise ValueError('Nonnegative integer storage contract required')
    sizes = dict(parameters=(parameters*parameter_bits+7)//8,
                 gradients=(parameters*gradient_bits+7)//8,
                 moments=(parameters*moment_bits*moments+7)//8,
                 master_weights=(parameters*master_bits+7)//8)
    return dict(bytes=sizes, total_bytes=sum(sizes.values()), budget_bytes=budget_bytes,
                resident_fits=sum(sizes.values()) <= budget_bytes,
                scope='Dense independent parameter state; excludes activations, workspaces and allocator overhead. No allocation performed.')


def _project(h, w, b):
    z = h @ w
    return z if b is None else z + b


class _ReplayCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, b, target, weight, indices, denominator, chunk, smoothing):
        ctx.save_for_backward(h, w, b, target, weight, indices, denominator)
        ctx.chunk, ctx.smoothing = chunk, smoothing
        value = h.new_zeros(())
        for ids in indices.split(chunk):
            value.add_(F.cross_entropy(_project(h.index_select(0, ids), w, b),
                       target.index_select(0, ids), weight=weight,
                       reduction='sum', label_smoothing=smoothing))
        return value / denominator

    @staticmethod
    def backward(ctx, upstream):
        if torch.is_grad_enabled():
            raise RuntimeError('Replay schedule is first-order; request derivative_order=2 before forward')
        h, w, b, target, weight, indices, denominator = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        dh = torch.zeros_like(h) if needs[0] else None
        dw = torch.zeros_like(w) if needs[1] else None
        db = torch.zeros_like(b) if b is not None and needs[2] else None
        # Native local VJPs preserve PyTorch loss/activation formulas, with the
        # global normalizer applied before local backward. Only the head replays.
        for ids in indices.split(ctx.chunk):
            with torch.enable_grad():
                hc = h.index_select(0, ids).detach().requires_grad_(needs[0])
                wc = w.detach().requires_grad_(needs[1])
                bc = b.detach().requires_grad_(needs[2]) if b is not None else None
                vals = (hc, wc, bc)
                demand = tuple(v for v in vals if v is not None and v.requires_grad)
                loss = F.cross_entropy(_project(hc, wc, bc), target.index_select(0, ids),
                    weight=weight, reduction='sum', label_smoothing=ctx.smoothing)
                gs = iter(torch.autograd.grad(loss, demand, grad_outputs=upstream/denominator))
            for i, v in enumerate(vals):
                if v is not None and needs[i]:
                    g = next(gs)
                    if i == 0: dh.index_copy_(0, ids, g)
                    elif i == 1: dw.add_(g)
                    else: db.add_(g)
        return dh, dw, db, None, None, None, None, None, None


def linear_cross_entropy(h, w, target, bias=None, *, weight=None, ignore_index=-100,
                         reduction='mean', label_smoothing=0., accuracy='strict',
                         derivative_order=1, chunk_rows=128, schedule='auto',
                         logit_budget_bytes=32*1024*1024, evidence=None):
    """h[...,K], w[K,V], targets h.shape[:-1]; sum/mean index-target CE.

    No token is removed from upstream attention. Shared/tied parameters receive
    autograd's normal sum of contributions. Unsupported numerical contracts use
    the native MPS expression. The budget bounds estimated head intermediates
    for replay (six chunk logits), not total training memory.
    """
    tensors = tuple(t for t in (h, w, target, bias, weight) if t is not None)
    if any(t.device.type != 'mps' for t in tensors):
        raise ValueError('All numerical tensors must be on MPS; no CPU fallback')
    if h.ndim < 2 or w.ndim != 2 or h.shape[-1] != w.shape[0] or tuple(target.shape) != tuple(h.shape[:-1]):
        raise ValueError('Expected h[...,K], w[K,V], index target[...]')
    if bias is not None and tuple(bias.shape) != (w.shape[1],):
        raise ValueError('Bias must have V entries')
    if target.dtype != torch.int64 or w.shape[1] < 1:
        raise ValueError('Nonempty class axis and int64 index targets required')
    if accuracy not in ('strict', 'algebraic') or schedule not in ('auto', 'selected', 'replay'):
        raise ValueError('Unknown accuracy or schedule')
    if reduction not in ('sum', 'mean') or not 0 <= label_smoothing <= 1 or derivative_order < 1:
        raise ValueError('Invalid loss or derivative contract')
    if chunk_rows < 1 or logit_budget_bytes < 1:
        raise ValueError('Positive chunk and budget required')
    if weight is not None and (tuple(weight.shape) != (w.shape[1],) or weight.requires_grad):
        raise ValueError('Fixed class weights with V entries required')
    hh, tt = h.reshape(-1, h.shape[-1]), target.reshape(-1)
    classes = w.shape[1]
    valid = tt != ignore_index
    if bool(((tt < 0) & valid | (tt >= classes) & valid).any()):
        raise ValueError('Target out of class range')
    def record(route, reason, **extra):
        if evidence is not None: evidence.append(dict(route=route, reason=reason, **extra))
    def native(reason):
        record('native', reason)
        return F.cross_entropy(_project(hh, w, bias), tt, weight=weight,
            ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    if accuracy == 'strict' or derivative_order > 1:
        return native('Strict expression or higher derivative')
    if torch.is_autocast_enabled('mps'):
        return native('Autocast requires the native dtype policy')
    if any(t.dtype != torch.float32 for t in tensors if t is not target):
        return native('Only FP32 algebraic qualification')
    if not hh.numel():
        return native('Empty features')
    # A conservative finite-logit envelope. It also checks ignored rows, whose
    # overflow could otherwise poison native parameter gradients (0 * inf).
    envelope = hh.detach().abs().amax() * w.detach().abs().amax() * h.shape[-1]
    if bias is not None: envelope = envelope + bias.detach().abs().amax()
    if not bool(envelope < torch.finfo(h.dtype).max / 16):
        return native('Head overflow risk')
    indices = valid.nonzero().flatten()
    if indices.numel() == 0:
        return native('All ignored: preserve native mean NaN and zero gradients')
    if weight is not None and not bool(torch.isfinite(weight).all() & (weight >= 0).all()):
        return native('Nonfinite or negative class weights')
    denom = (weight.index_select(0, tt.index_select(0, indices)).sum() if weight is not None
             else h.new_full((), indices.numel())) if reduction == 'mean' else h.new_ones(())
    if weight is not None and not bool(torch.isfinite(denom) & (denom > 0)):
        return native('Nonpositive or nonfinite global normalizer')
    per_row = 6 * classes * h.element_size()
    safe_chunk = min(chunk_rows, logit_budget_bytes // per_row)
    if safe_chunk < 1:
        raise MemoryError('Budget cannot hold one CE row; use vocabulary tiling or increase head budget')
    route = schedule
    if route == 'auto': route = 'selected' if indices.numel()*per_row <= logit_budget_bytes else 'replay'
    if route == 'selected' and indices.numel()*per_row > logit_budget_bytes:
        raise MemoryError('Selected schedule exceeds head intermediate budget; choose replay')
    record(route, 'Finite FP32 index-target contract', active=indices.numel(), rows=hh.shape[0],
           chunk_rows=safe_chunk, plan=head_plan(hh.shape[0], hh.shape[1], classes, indices.numel(),
              safe_chunk, h.requires_grad, w.requires_grad,
              bias is not None and bias.requires_grad))
    if route == 'selected':
        return F.cross_entropy(_project(hh.index_select(0, indices), w, bias),
            tt.index_select(0, indices), weight=weight, reduction='sum',
            label_smoothing=label_smoothing) / denom
    return _ReplayCE.apply(hh, w, bias, tt, weight, indices, denom, safe_chunk, label_smoothing)
