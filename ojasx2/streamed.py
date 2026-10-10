"""Operand-only rank-7 tape and a streaming joint pullback.

This is an algebraic/tolerance operator. Strict mode delegates to native GEMM.
The reference uses ATen buffers, not claimed GPU-register residency. No GEMM
is recomputed in backward; only small pointwise encoder expressions are rebuilt.
"""
import torch
from .bilinear import _blocks, _encode_a, _encode_b


def _validate(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Expected compatible 2D matrices')
    if min(*a.shape, *b.shape) <= 0 or any(d % 2 for d in (*a.shape, *b.shape)):
        raise ValueError('Rank-7 requires positive even dimensions')
    if a.device != b.device or a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64):
        raise ValueError('Rank-7 qualification currently requires matching FP32/FP64 tensors')


def _forward_stream(a, b):
    # Encoding is shared within forward, but NONE is retained as a backward tape.
    ea, eb = _encode_a(_blocks(a)), _encode_b(_blocks(b))
    y = a.new_empty((a.shape[0], b.shape[1]))
    y11,y12,y21,y22 = _blocks(y)
    p = ea[0] @ eb[0]
    y11.copy_(p); y12.copy_(p)
    del p
    y11.add_(ea[1] @ eb[1])
    y12.add_(ea[5] @ eb[5])  # u1
    y21.copy_(y12)
    y21.add_(ea[6] @ eb[6])  # u2
    y22.copy_(y21)
    p = ea[4] @ eb[4]
    y12.add_(p); y22.add_(p)
    del p
    y12.add_(ea[2] @ eb[2])
    y21.sub_(ea[3] @ eb[3])
    return y


def _a_channel(blocks, i):
    a,b,c,d = blocks
    return {0:lambda:a, 1:lambda:b, 2:lambda:b-((c+d)-a),
            3:lambda:d, 4:lambda:c+d, 5:lambda:(c+d)-a, 6:lambda:a-c}[i]()


def _b_channel(blocks, i):
    a,b,c,d = blocks
    return {0:lambda:a, 1:lambda:c, 2:lambda:d, 3:lambda:(d-(b-a))-c,
            4:lambda:b-a, 5:lambda:d-(b-a), 6:lambda:d-b}[i]()


def _joint_pullback(a, b, g, need_a=True, need_b=True):
    ab = _blocks(a) if need_b else None
    bb = _blocks(b) if need_a else None
    gb = _blocks(g)
    g11,g12,g21,g22 = gb
    v1 = g21+g22
    v3 = g12+v1
    da = g.new_empty((g.shape[0], b.shape[0])) if need_a else None
    db = g.new_empty((a.shape[1], g.shape[1])) if need_b else None
    aa = _blocks(da) if da is not None else None
    ba = _blocks(db) if db is not None else None

    # Joint order: 6,3,4,5,1,7,2. Unused gradient-output slots temporarily hold
    # shared encoder cotangents. Each channel product is generated once.
    for i in (5,2,3,4,0,6,1):
        if i == 5: d = v3
        elif i == 2: d = g12
        elif i == 3: d = -g21
        elif i == 4: d = g12+g22
        elif i == 0: d = g11+v3
        elif i == 6: d = v1
        else: d = g11
        if aa is not None:
            p = d @ _b_channel(bb, i).T
            a11,a12,a21,a22 = aa
            if i == 5: a21.copy_(p)
            elif i == 2:
                a21.sub_(p); a12.copy_(p); a11.copy_(-a21)
            elif i == 3: a22.copy_(p)
            elif i == 4: a21.add_(p); a22.add_(a21)
            elif i == 0: a11.add_(p)
            elif i == 6: a11.add_(p); a21.sub_(p)
            else: a12.add_(p)
            del p
        if ba is not None:
            p = _a_channel(ab, i).T @ d
            b11,b12,b21,b22 = ba
            if i == 5: b22.copy_(p)
            elif i == 2: b11.copy_(p)  # pending c3, consumed before overwrite
            elif i == 3: b22.add_(p); b21.copy_(-p)
            elif i == 4:
                b12.copy_(p); b12.sub_(b22); b22.add_(b11); b11.copy_(-b12)
            elif i == 0: b11.add_(p)
            elif i == 6: b12.sub_(p); b22.add_(p)
            else: b21.add_(p)
            del p
    return da, db


@torch.library.custom_op('ojasx_verified::operand_tape_gemm', mutates_args=())
def operand_tape_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _validate(a, b)
    return _forward_stream(a, b)


@operand_tape_gemm.register_fake
def _fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1]))


def _setup(ctx, inputs, output):
    a,b = inputs
    # During setup PyTorch appends its registration metadata argument; its
    # backward wrapper presents only the two operator inputs.
    need_a,need_b = ctx.needs_input_grad[:2]
    ctx.save_for_backward(a if need_b else None, b if need_a else None)


def _backward(ctx, g):
    a,b = ctx.saved_tensors
    need_a,need_b = ctx.needs_input_grad
    if torch.is_grad_enabled():
        # A differentiable higher-order route; do not apply output-slot reuse
        # to a backward pass whose own graph must remain available.
        return (g @ b.T if need_a else None, a.T @ g if need_b else None)
    return _joint_pullback(a, b, g, need_a, need_b)


operand_tape_gemm.register_autograd(_backward, setup_context=_setup)


def streamed_gemm(a, b, accuracy='strict'):
    """Drop-in matmul shape; explicit algebraic opt-in for operand-only rank-7.

    Strict preserves native PyTorch GEMM. 'algebraic' changes rounded arithmetic
    and must be qualified against each workload's numerical tolerances.
    """
    if accuracy == 'strict':
        return a @ b
    if accuracy != 'algebraic':
        raise ValueError("accuracy must be 'strict' or 'algebraic'")
    return operand_tape_gemm(a, b)


def streamed_cost(m, k, n):
    if min(m,k,n) <= 0 or any(d % 2 for d in (m,k,n)):
        raise ValueError('Expected positive even dimensions')
    # Reconstruct one encoder channel at a time in backward: seven binary
    # additions per operand, instead of four with a full encoded cache.
    forward = 7*m*k*n//4+m*k+k*n
    backward = 7*m*k*n//2+m*n+7*(m*k+k*n)//4
    dense = 6*m*k*n-m*n-m*k-k*n
    return {'forward_scalar_ops':forward, 'backward_scalar_ops':backward,
            'joint_scalar_ops':forward+backward, 'dense_joint_scalar_ops':dense,
            'scalar_saving':dense-forward-backward,
            'saved_operand_elements':m*k+k*n,
            'saved_encoded_elements':0, 'forward_gemms':7, 'backward_gemms':14,
            'scope':'first-order joint operation model; unary signs/copies/traffic/launches excluded'}
