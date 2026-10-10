"""Packed projections with explicit gradient demand and operand-only tape."""
import torch
import torch.nn.functional as F


class _PackedProjections(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,*weights):
        ctx.set_materialize_grads(False)
        ctx.widths=tuple(w.shape[0] for w in weights)
        ctx.input_shape=tuple(x.shape)
        need_x=ctx.needs_input_grad[0]
        need_weights=any(ctx.needs_input_grad[1:])
        ctx.save_for_backward(x if need_weights else None,
                              *(w if need_x else None for w in weights))
        # Concatenation is transient, and is reconstructed only if dX is needed.
        y=F.linear(x,torch.cat(weights,dim=0))
        return tuple(t.contiguous() for t in y.split(ctx.widths,dim=-1))

    @staticmethod
    def backward(ctx,*dys):
        x,*weights=ctx.saved_tensors
        active=[i for i,g in enumerate(dys) if g is not None]
        dx=None
        packed_g=None
        if ctx.needs_input_grad[0] and active:
            packed_g=torch.cat(tuple(dys[i] for i in active),dim=-1)
            w=torch.cat(tuple(weights[i] for i in active),dim=0)
            dx=packed_g@w.conj()
            del w
        requested=[i for i in active if ctx.needs_input_grad[1+i]]
        gradients=[None]*len(dys)
        if requested:
            rows=1
            for dim in ctx.input_shape[:-1]:rows*=dim
            widths=tuple(ctx.widths[i] for i in requested)
            g=packed_g if requested==active and packed_g is not None else torch.cat(tuple(dys[i] for i in requested),dim=-1)
            dw=g.reshape(rows,sum(widths)).T@x.reshape(rows,ctx.input_shape[-1]).conj()
            for index,value in zip(requested,dw.split(widths,dim=0)):
                gradients[index]=value
        return dx,*gradients


def packed_projections(x,weights):
    return _PackedProjections.apply(x,*weights)
