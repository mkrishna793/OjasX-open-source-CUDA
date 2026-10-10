"""Finite matrix-chain planning for forward plus demanded first-order VJPs.

Optimal only among binary parenthesizations in the declared scalar model.
Native autograd differentiates the selected tree; no rank approximation.
"""
from functools import lru_cache
import torch
from .devices import algebraic_device_supported


def _node_cost(a, b, c, left_grad, right_grad):
    value = 2*a*b*c-a*c
    if left_grad:
        value += 2*a*b*c-a*b
    if right_grad:
        value += 2*a*b*c-b*c
    return value


@lru_cache(maxsize=256)
def plan_matrix_chain(dimensions, needs):
    """Return cached immutable cost/tree tuples for a fixed shape/demand."""
    n = len(dimensions)-1
    if n < 1 or len(needs) != n or min(dimensions) <= 0:
        raise ValueError('Positive compatible chain dimensions and leaf demand required')
    costs, trees = {}, {}
    for i in range(n):
        costs[i, i], trees[i, i] = 0, i
    for length in range(2, n+1):
        for i in range(n-length+1):
            j = i+length-1
            choices = []
            for split in range(i, j):
                local = _node_cost(dimensions[i], dimensions[split+1], dimensions[j+1],
                                   any(needs[i:split+1]), any(needs[split+1:j+1]))
                choices.append((costs[i, split]+costs[split+1, j]+local, split))
            cost, split = min(choices)
            costs[i, j] = cost
            trees[i, j] = (trees[i, split], trees[split+1, j])
    baseline, left_tree = 0, 0
    for i in range(1, n):
        baseline += _node_cost(dimensions[0], dimensions[i], dimensions[i+1],
                               any(needs[:i]), needs[i])
        left_tree = (left_tree, i)
    # Tuples prevent caller mutation from contaminating the cache.
    return costs[0, n-1], baseline, trees[0, n-1], left_tree


def matrix_chain_plan(dimensions, needs=None):
    dimensions = tuple(dimensions)
    needs = tuple(bool(x) for x in (needs if needs is not None else [True]*(len(dimensions)-1)))
    cost, baseline, tree, native_tree = plan_matrix_chain(dimensions, needs)
    return dict(dimensions=dimensions, needs=needs, scalar_ops=cost,
                native_scalar_ops=baseline, scalar_saving=baseline-cost,
                tree=tree, native_tree=native_tree,
                scope='First-order GEMM scalar model; excludes copies, tape, bias, dispatch and hardware throughput')


def _left_fold(values):
    out = values[0]
    for value in values[1:]:
        out = out @ value
    return out


def matrix_chain(values, accuracy='strict'):
    """Original matrix leaves remain trainable; strict keeps left-fold order."""
    values = tuple(values)
    if not values:
        raise ValueError('At least one matrix required')
    if accuracy not in ('strict', 'algebraic'):
        raise ValueError('Invalid accuracy contract')
    if accuracy == 'strict' or len(values) == 1:
        return _left_fold(values)
    eligible = (all(v.ndim == 2 and min(v.shape) > 0 and v.device == values[0].device
                    and v.dtype == values[0].dtype for v in values)
                and algebraic_device_supported(values[0])
                and all(a.shape[1] == b.shape[0] for a, b in zip(values, values[1:])))
    if not eligible:
        return _left_fold(values)
    plan = matrix_chain_plan((values[0].shape[0], *(v.shape[1] for v in values)),
                             (v.requires_grad for v in values))
    if plan['scalar_saving'] <= 0:
        return _left_fold(values)
    def evaluate(tree):
        return values[tree] if isinstance(tree, int) else evaluate(tree[0]) @ evaluate(tree[1])
    return evaluate(plan['tree'])


def linear_sequence(x, weights, accuracy='strict'):
    """Bias-free consecutive Linear layers; caller must preserve nonlinear boundaries."""
    return matrix_chain((x, *(w.T for w in weights)), accuracy)
