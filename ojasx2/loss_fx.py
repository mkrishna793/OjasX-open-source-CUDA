"""Conservative FX frontend for loss-directed schedules; ordinary model code.

Only private, single-use Linear -> cross_entropy outputs are rewritten. The
runtime admits 2D FP32 index-target scalar losses, or calls the original MPS
expression. This is a prototype graph pass, not a torch.compile backend.
"""
import torch
import torch.nn.functional as F
from .loss_compiler import linear_cross_entropy


def _linear_ce(h, weight_matrix, bias, target, *, loss_kwargs, accuracy, derivative_order, head_budget):
    tensors=(h,weight_matrix,target)+(tuple() if bias is None else (bias,))
    if any(t.device.type!='mps' for t in tensors):raise ValueError('MPS tensors required; no CPU fallback')
    supported=(accuracy=='algebraic' and derivative_order==1 and not torch.is_autocast_enabled('mps')
               and h.ndim==2 and target.ndim==1 and target.dtype==torch.int64
               and loss_kwargs.get('size_average') is None and loss_kwargs.get('reduce') is None
               and loss_kwargs.get('reduction','mean') in ('mean','sum'))
    if not supported:
        return F.cross_entropy(F.linear(h,weight_matrix,bias),target,**loss_kwargs)
    return linear_cross_entropy(h,weight_matrix.T,target,bias,
        weight=loss_kwargs.get('weight'),ignore_index=loss_kwargs.get('ignore_index',-100),
        reduction=loss_kwargs.get('reduction','mean'),label_smoothing=loss_kwargs.get('label_smoothing',0.),
        accuracy=accuracy,derivative_order=derivative_order,logit_budget_bytes=head_budget)


def rewrite_linear_cross_entropy(graph_module, *, accuracy='strict', derivative_order=1,
                                 head_budget=32*1024*1024):
    """Mutate an FX GraphModule and return an explicit selection ledger.

    Custom module hooks/subclasses and externally observed logits are retained.
    Tracing itself requires a traceable graph; arbitrary Python effects are not
    certified by this pass. Native loss kwargs are retained by runtime fallback.
    """
    ledger=[];graph=graph_module.graph
    for node in list(graph.nodes):
        if node.op!='call_function' or node.target is not F.cross_entropy:continue
        logits=node.args[0] if node.args else node.kwargs.get('input')
        target=node.args[1] if len(node.args)>1 else node.kwargs.get('target')
        if not isinstance(logits,torch.fx.Node) or target is None:
            ledger.append(dict(node=node.name,route='native',reason='Unrecognized loss arguments'));continue
        if len(logits.users)!=1 or len(node.args)>2:
            ledger.append(dict(node=node.name,route='native',reason='Observed logits or extra positional loss arguments'));continue
        h=w=b=None;module=None
        if logits.op=='call_module':
            module=graph_module.get_submodule(logits.target)
            hooks=torch.nn.modules.module
            global_hooks=hooks._global_forward_hooks or hooks._global_forward_pre_hooks or hooks._global_backward_hooks
            if type(module) is not torch.nn.Linear or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks or global_hooks:
                ledger.append(dict(node=node.name,route='native',reason='Custom module or hooks'));continue
            if not logits.args:
                ledger.append(dict(node=node.name,route='native',reason='Keyword-only linear input'));continue
            h=logits.args[0]
        elif logits.op=='call_function' and logits.target is F.linear:
            h=logits.args[0] if logits.args else logits.kwargs.get('input')
            w=logits.args[1] if len(logits.args)>1 else logits.kwargs.get('weight')
            b=logits.args[2] if len(logits.args)>2 else logits.kwargs.get('bias')
            if h is None or w is None:
                ledger.append(dict(node=node.name,route='native',reason='Unrecognized linear arguments'));continue
        else:
            ledger.append(dict(node=node.name,route='native',reason='Loss is not directly fed by linear'));continue
        kw={k:v for k,v in node.kwargs.items() if k not in ('input','target')}
        with graph.inserting_before(node):
            if module is not None:
                w=graph.get_attr(logits.target+'.weight')
                b=graph.get_attr(logits.target+'.bias') if module.bias is not None else None
            replacement=graph.call_function(_linear_ce,args=(h,w,b,target),kwargs=dict(loss_kwargs=kw,
                accuracy=accuracy,derivative_order=derivative_order,head_budget=head_budget))
        node.replace_all_uses_with(replacement);graph.erase_node(node);graph.erase_node(logits)
        ledger.append(dict(node=replacement.name,route='loss_compiler',reason='Private linear output; runtime domain guard'))
    graph.lint();graph_module.recompile()
    return ledger
