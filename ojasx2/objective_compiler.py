"""Joint objective/gradient schedules for deterministic, separable ML programs.

These are reference mathematical schedules, not a device code generator.
Algebraic mode changes reduction order; only bounded MPS FP32 is qualified.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from .residuals import SavedTensorAudit


def _gpu(tensors):
    if any(t.device.type != "mps" for t in tensors):
        raise ValueError("Numerical operands must be on MPS; CPU fallback is prohibited")
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
        raise ValueError("This qualification covers FP32 only")
    if torch.is_autocast_enabled("mps"):
        raise ValueError("Autocast is not qualified")


def quadratic_plan(m, k, n, need_h=True, need_w=True, has_bias=False, need_bias=False):
    """Leading multiply/add counts, including all demanded matrix gradients.

    Pointwise reductions, guards, allocation and the feature network are extra.
    Target differentiation requires the native route.
    """
    native = 2*m*k*n*(1+int(need_h)+int(need_w))
    encoded_k=k+int(has_bias)
    gram = (2*m*encoded_k*encoded_k + 2*m*encoded_k*n + 2*encoded_k*encoded_k*n
            + int(need_h)*(2*m*encoded_k*encoded_k+2*m*n*encoded_k)
            + int(need_w or need_bias)*2*encoded_k*encoded_k*n)
    return dict(native_matrix_flops=native, gram_matrix_flops=gram,
                eligible=min(m,k,n)>0 and gram<native,
                saved_wide_residual_elements=m*n,
                saved_core_elements=2*encoded_k*encoded_k+encoded_k*n,
                original_feature_width=k,encoded_feature_width=encoded_k,
                scope="Leading matrix arithmetic only; dense target and output gradients remain")


class _QuadraticGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_h, original_w, y, bias, evidence):
        h,w=original_h,original_w
        if bias is not None:
            h=torch.cat((h,torch.ones_like(h[:,:1])),dim=1)
            w=torch.cat((w,bias[None]),dim=0)
        gram = h.T @ h
        cross = h.T @ y
        metric = w @ w.T
        energy = (gram*metric).sum()
        coupling = (cross*w).sum()
        target_energy = torch.linalg.vector_norm(y).square()
        numerator = energy - 2*coupling + target_energy
        scale = energy.abs()+2*coupling.abs()+target_energy.abs()
        # A rejection heuristic, not an IEEE error certificate. Near a fit the
        # Gram loss subtracts large energies and must retain the native formula.
        native = not bool(torch.isfinite(numerator) & torch.isfinite(scale)
                          & (numerator > 1e-3*scale))
        ctx.native = native
        ctx.has_bias=bias is not None
        ctx.evidence=evidence
        ctx.save_for_backward(original_h,original_w,y,
                              bias if bias is not None else h.new_empty(0),h,w,gram,cross,metric)
        if evidence is not None:
            evidence.append(dict(selected="native" if native else "quadratic_gram",
                                 reason="cancellation_or_nonfinite" if native else "shape_and_energy_guards",
                                 ieee_certificate=False))
        return F.mse_loss(original_h@original_w+(0 if bias is None else bias),y) if native else numerator/y.numel()

    @staticmethod
    def backward(ctx, upstream):
        if torch.is_grad_enabled():
            raise RuntimeError("For higher derivatives select derivative_order=2 before the forward call")
        original_h,original_w,y,bias,h,w,gram,cross,metric = ctx.saved_tensors
        factor = 2*upstream/y.numel()
        nh,nw=ctx.needs_input_grad[:2];nb=ctx.needs_input_grad[3]
        fallback=ctx.native
        dh=dw=None
        if not fallback:
            conditions=[]
            if nh:
                left=h@metric;right=y@w.T;dh=left-right
                if ctx.has_bias:dh=dh[:,:-1];left=left[:,:-1];right=right[:,:-1]
                conditions.append(torch.isfinite(dh).all() &
                    (torch.linalg.vector_norm(dh)>1e-3*(torch.linalg.vector_norm(left)+torch.linalg.vector_norm(right))))
            if nw or nb:
                left=gram@w;dw=left-cross
                blocks=[]
                if nw:blocks.append((dw[:-1],left[:-1],cross[:-1]) if ctx.has_bias else (dw,left,cross))
                if nb:blocks.append((dw[-1],left[-1],cross[-1]))
                for difference,positive,negative in blocks:
                    conditions.append(torch.isfinite(difference).all() &
                        (torch.linalg.vector_norm(difference)>1e-3*(torch.linalg.vector_norm(positive)+torch.linalg.vector_norm(negative))))
            fallback=bool(conditions) and not bool(torch.stack(conditions).all())
        if fallback:
            # Match the original bias placement and scale the residual BEFORE
            # its matrix contractions. This path pays for recomputation.
            residual=(original_h@original_w+(bias if ctx.has_bias else 0)-y)*factor
            dh=residual@original_w.T if nh else None
            dw_original=original_h.T@residual if nw else None
            db=residual.sum(0) if nb else None
            if ctx.evidence is not None:ctx.evidence.append(dict(backward_selected="native",reason="loss_or_gradient_cancellation",recomputed=True))
        else:
            dh=None if dh is None else factor*dh
            db=factor*dw[-1] if nb else None
            dw_original=(factor*(dw[:-1] if ctx.has_bias else dw)) if nw else None
        return dh,dw_original,None,db,None


def quadratic_loss(h,w,y,bias=None,*,accuracy="strict",derivative_order=1,evidence=None):
    """Mean squared error of h@w+b, without requesting the wide predictions.

    h may be produced by any differentiable network. Original h/w/b gradients
    flow through the graph. Target gradients and higher order retain native.
    """
    tensors=(h,w,y)+(() if bias is None else (bias,))
    _gpu(tensors)
    if accuracy not in ("strict","algebraic") or derivative_order not in (1,2):
        raise ValueError("Invalid accuracy/derivative order")
    if h.ndim!=2 or w.ndim!=2 or y.ndim!=2 or h.shape[1]!=w.shape[0] or y.shape!=(h.shape[0],w.shape[1]):
        raise ValueError("Expected h[m,k], w[k,n], target[m,n]")
    if min(*h.shape,*w.shape)==0 or (bias is not None and bias.shape!=(w.shape[1],)):
        raise ValueError("Nonempty dimensions and a vector bias are required")
    plan=quadratic_plan(h.shape[0],h.shape[1],w.shape[1],h.requires_grad,w.requires_grad,
                        has_bias=bias is not None,need_bias=bias is not None and bias.requires_grad)
    if accuracy=="strict" or derivative_order>1 or y.requires_grad or not plan["eligible"]:
        if evidence is not None:evidence.append(dict(selected="native",reason="mode_demand_or_cost",plan=plan))
        return F.mse_loss(h@w+(0 if bias is None else bias),y)
    if not bool(torch.stack(tuple(torch.isfinite(t).all() for t in tensors)).all()):
        if evidence is not None:evidence.append(dict(selected="native",reason="nonfinite_operand"))
        return F.mse_loss(h@w+(0 if bias is None else bias),y)
    return _QuadraticGram.apply(h,w,y,bias,evidence)


@dataclass(frozen=True)
class SeparableContract:
    """Explicit caller obligations; declaring these is not a formal proof.

    The callback returns one scalar loss per independent example. Parameters,
    buffers and random state do not change; all chunks use the same parameters.
    A batch-coupled loss, training BatchNorm or fresh dropout violates this API.
    """
    independent_examples: bool
    deterministic: bool
    no_mutation: bool
    justification: str
    uniform_parameter_usage: bool = False
    finite_per_example_loss_and_gradients: bool = False

    def validate(self):
        if not (self.independent_examples and self.deterministic and self.no_mutation and self.justification.strip()):
            raise ValueError("Unmet separability, deterministic replay or mutation contract")


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    parameter_grads: tuple
    input_grads: tuple
    evidence: dict


class SeparableProgram:
    def __init__(self,per_example_loss,contract,chunk_size):
        contract.validate()
        if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError("Positive chunk size required")
        self.loss_fn,self.contract,self.chunk_size=per_example_loss,contract,chunk_size

    def value_and_grad(self,parameters,batch,*,weights=None,normalization="batch",measure_saved=False):
        """First-order full-objective gradients; update/clip only after return.

        Zero-weight examples can be skipped only under the declared contract.
        No activation approximation or replay of a chunk's forward is used.
        All-zero weights evaluate natively to preserve unused-gradient semantics.
        """
        parameters,batch=tuple(parameters),tuple(batch)
        if not batch or batch[0].ndim==0:raise ValueError("A nonempty batched input is required")
        n=batch[0].shape[0]
        if n<1 or any(t.ndim==0 or t.shape[0]!=n for t in batch):raise ValueError("Mismatched batch axes")
        if len({id(p) for p in parameters})!=len(parameters):raise ValueError("Pass each shared parameter once")
        if any(not p.is_leaf for p in parameters):raise ValueError("Optimizer parameters must be leaf tensors")
        if normalization not in ("batch","weight_sum","sum"):raise ValueError("Invalid normalization")
        if weights is None:weights=torch.ones(n,device=batch[0].device,dtype=torch.float32)
        _gpu(parameters+batch+(weights,))
        if weights.shape!=(n,) or weights.requires_grad:raise ValueError("Weights must be fixed scalars per example")
        if not bool(torch.isfinite(weights).all() & (weights>=0).all()):raise ValueError("Finite nonnegative weights required")
        finite_checks=tuple(torch.isfinite(t).all() for t in parameters+batch if t.is_floating_point())
        if finite_checks and not bool(torch.stack(finite_checks).all()):
            raise ValueError("Finite operands required for zero-weight elimination")
        denom=(weights.sum() if normalization=="weight_sum" else weights.new_tensor(n if normalization=="batch" else 1))
        if not bool(denom>0):raise ValueError("Objective normalization must be positive")
        elide=self.contract.uniform_parameter_usage and self.contract.finite_per_example_loss_and_gradients
        active=(weights.nonzero().flatten() if elide
                else torch.arange(n,device=weights.device))
        # This branch deliberately does not invent zero gradients for unused
        # parameters, which would change optimizer weight-decay behavior.
        all_zero=active.numel()==0
        if all_zero:active=torch.arange(n,device=weights.device)
        pactive=tuple(p for p in parameters if p.requires_grad)
        pacc=[None]*len(parameters)
        bacc=[torch.zeros_like(t) if t.requires_grad else None for t in batch]
        bseen=[False]*len(batch)
        loss_total=weights.new_zeros(())
        memory=[]
        chunks=0
        versions=tuple(t._version for t in parameters+batch)
        step=n if all_zero else self.chunk_size
        for start in range(0,active.numel(),step):
            ix=active[start:start+step]
            pieces=tuple(t.index_select(0,ix).detach().requires_grad_(t.requires_grad) for t in batch)
            bactive=tuple(t for t in pieces if t.requires_grad)
            audit=SavedTensorAudit(list(parameters+batch+(weights,))) if measure_saved else None
            with audit if audit is not None else nullcontext():
                losses=self.loss_fn(parameters,pieces)
                if losses.shape!=(ix.numel(),) or losses.device.type!="mps" or losses.dtype!=torch.float32:
                    raise ValueError("Callback must return one FP32 MPS loss per example")
                scalar=(losses*weights.index_select(0,ix)).sum()/denom
            if audit is not None:memory.append(audit.summary())
            demands=pactive+bactive
            grads=torch.autograd.grad(scalar,demands,allow_unused=True) if demands and scalar.requires_grad else (None,)*len(demands)
            loss_total.add_(scalar.detach())
            cursor=0
            for j,p in enumerate(parameters):
                if not p.requires_grad:continue
                g=grads[cursor];cursor+=1
                if g is not None:
                    if pacc[j] is None:pacc[j]=g.detach().clone()
                    else:pacc[j].add_(g)
            for j,t in enumerate(pieces):
                if not t.requires_grad:continue
                g=grads[cursor];cursor+=1
                if g is not None:
                    bacc[j].index_copy_(0,ix,g)
                    bseen[j]=True
            chunks+=1
            del scalar,losses,grads,pieces
        if tuple(t._version for t in parameters+batch)!=versions:
            raise RuntimeError("Callback mutated an operand; discard this result")
        bacc=[g if seen else None for g,seen in zip(bacc,bseen)]
        return ObjectiveResult(loss_total,tuple(pacc),tuple(bacc),dict(
            rule="separable_objective",batch_size=n,active_examples=0 if all_zero else active.numel(),
            chunks=chunks,chunk_size=step,normalization=normalization,
            declaration=self.contract.justification,caller_contract_is_not_a_proof=True,
            zero_weight_elision=elide,
            optimizer_steps_inside=0,forward_recomputation=False,derivative_order=1,
            maximum_chunk_fresh_saved_bytes=max((m["fresh_saved_storage_bytes"] for m in memory),default=None),
            gradient_accumulator_bytes=sum(g.numel()*g.element_size() for g in pacc+bacc if g is not None),
            memory_scope="Maximum single-chunk autograd saved storage plus separately reported gradient buffers; not total allocator peak",
            all_zero_native_semantics=all_zero))


def compile_separable_objective(per_example_loss,*,contract,chunk_size=32):
    return SeparableProgram(per_example_loss,contract,chunk_size)
