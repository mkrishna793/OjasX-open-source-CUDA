"""A real FX graph rewrite backend, preserving the normal training interface.

Returns a rewritten ATen/custom-op graph. It is not a native GPU code generator.
Algebraic mode is explicit; guards retain native GEMM for unqualified shapes.
"""
import operator
import torch
import torch.nn.functional as F
from .streamed import operand_tape_gemm, streamed_cost
from .ops import swiglu
from .contractions import linear_chain
from .planner import linear_sequence
from .radial_core import rms_core_linear_chain


def _dispatch_rms_core(x, w1, w2, normalized_shape, gamma, eps, bias2):
    if tuple(normalized_shape) == (w1.shape[0],):
        return rms_core_linear_chain(x, w1, w2, gamma, bias2, eps, 'algebraic')
    return F.linear(F.rms_norm(F.linear(x, w1), normalized_shape, gamma, eps), w2, bias2)


def _eligible(a, b, minimum_inner):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return False
    if a.shape[1] < minimum_inner or min(*a.shape,*b.shape) <= 0:
        return False
    if any(d % 2 for d in (*a.shape,*b.shape)):
        return False
    if a.dtype not in (torch.float32,torch.float64) or a.dtype != b.dtype or a.device != b.device:
        return False
    # A runtime guard, not an assertion that scalar savings imply speed.
    return streamed_cost(a.shape[0],a.shape[1],b.shape[1])['scalar_saving'] > 0


def _dispatch_mm(a, b, minimum_inner):
    return operand_tape_gemm(a,b) if _eligible(a,b,minimum_inner) else a@b


def _dispatch_linear(x, w, bias, minimum_inner):
    b = w.T
    if _eligible(x,b,minimum_inner):
        y = operand_tape_gemm(x,b)
        return y if bias is None else y+bias
    return F.linear(x,w,bias)


def _dispatch_swiglu(gate, up):
    if (isinstance(up,torch.Tensor) and gate.shape==up.shape and gate.dtype==up.dtype
            and gate.device==up.device and gate.device.type=='cpu'
            and gate.dtype in (torch.float32,torch.float64)):
        return swiglu(gate,up)
    return F.silu(gate)*up


def make_backend(accuracy='strict', minimum_inner=128, memory_policy='native',
                 contraction_policy='native'):
    """Use torch.compile(model, backend=make_backend(...)).

    Strict returns the captured reference graph. Algebraic rewrites supported
    matmul/linear nodes; odd, thin, small and unsupported cases retain native ops.
    This backend does not promise speed or all-device numeric qualification.
    """
    if (accuracy not in ('strict','algebraic') or minimum_inner < 1
            or memory_policy not in ('native','pointwise')
            or contraction_policy not in ('native','linear_chain','linear_chain_backward','matrix_chain','rms_core')
            or (contraction_policy!='native' and accuracy!='algebraic')):
        raise ValueError('Invalid accuracy/minimum_inner contract')
    evidence = []

    def backend(gm, example_inputs):
        replacements = 0
        fusions = 0
        chains = 0
        planned_chains = 0
        radial_chains = 0
        if contraction_policy == 'rms_core':
            linear_targets = (F.linear, torch.ops.aten.linear.default)
            def linear_node(node):
                return (isinstance(node, torch.fx.Node) and node.op == 'call_function'
                        and node.target in linear_targets and 2 <= len(node.args) <= 3
                        and all(k == 'bias' for k in node.kwargs))
            for node in list(gm.graph.nodes):
                if not linear_node(node):continue
                norm = node.args[0]
                if (not isinstance(norm, torch.fx.Node) or norm.op != 'call_function'
                        or norm.target not in (F.rms_norm, torch.rms_norm) or len(norm.users) != 1
                        or not 2 <= len(norm.args) <= 4
                        or any(k not in ('weight', 'eps') for k in norm.kwargs)):continue
                first = norm.args[0]
                if (not linear_node(first) or len(first.users) != 1
                        or (first.args[2] if len(first.args) > 2 else first.kwargs.get('bias')) is not None):continue
                gamma = norm.args[2] if len(norm.args) > 2 else norm.kwargs.get('weight')
                eps = norm.args[3] if len(norm.args) > 3 else norm.kwargs.get('eps')
                bias = node.args[2] if len(node.args) > 2 else node.kwargs.get('bias')
                with gm.graph.inserting_before(node):
                    new = gm.graph.call_function(_dispatch_rms_core,
                        (first.args[0], first.args[1], node.args[1], norm.args[1], gamma, eps, bias))
                node.replace_all_uses_with(new)
                for old in (node, norm, first):gm.graph.erase_node(old)
                radial_chains += 1
        if contraction_policy == 'matrix_chain':
            linear_targets = (F.linear,torch.ops.aten.linear.default)
            def unbiased(node):
                return (node.op=='call_function' and node.target in linear_targets
                        and 2<=len(node.args)<=3 and all(k=='bias' for k in node.kwargs)
                        and (node.args[2] if len(node.args)>2 else node.kwargs.get('bias')) is None)
            for node in reversed(list(gm.graph.nodes)):
                if not unbiased(node):continue
                inputs=node.args[0]; weights=[node.args[1]]; members=[node]
                while isinstance(inputs,torch.fx.Node) and unbiased(inputs) and len(inputs.users)==1:
                    members.append(inputs); weights.append(inputs.args[1]);inputs=inputs.args[0]
                if len(weights)<2:continue
                with gm.graph.inserting_before(node):
                    new=gm.graph.call_function(linear_sequence,(inputs,tuple(reversed(weights)),'algebraic'))
                node.replace_all_uses_with(new)
                for old in members:gm.graph.erase_node(old)
                planned_chains+=1
        if contraction_policy in ('linear_chain','linear_chain_backward'):
            linear_targets = (F.linear,torch.ops.aten.linear.default)
            def linear_node(node):
                return (node.op=='call_function' and node.target in linear_targets
                        and 2<=len(node.args)<=3 and all(k=='bias' for k in node.kwargs))
            for node in list(gm.graph.nodes):
                if not linear_node(node):continue
                first=node.args[0]
                if not isinstance(first,torch.fx.Node) or not linear_node(first) or len(first.users)!=1:continue
                mode='backward_only' if contraction_policy=='linear_chain_backward' else 'joint'
                with gm.graph.inserting_before(node):
                    bias1=first.args[2] if len(first.args)>2 else first.kwargs.get('bias')
                    bias2=node.args[2] if len(node.args)>2 else node.kwargs.get('bias')
                    new=gm.graph.call_function(linear_chain,(first.args[0],first.args[1],node.args[1],'algebraic',mode,bias1,bias2))
                node.replace_all_uses_with(new)
                gm.graph.erase_node(node);gm.graph.erase_node(first)
                chains+=1
        if memory_policy == 'pointwise':
            for node in list(gm.graph.nodes):
                if node.op != 'call_function' or node.target not in (operator.mul,torch.mul,torch.ops.aten.mul.Tensor) or len(node.args)!=2 or node.kwargs:
                    continue
                act,up = node.args
                if (not isinstance(act,torch.fx.Node) or act.op!='call_function'
                        or act.target not in (F.silu,torch.ops.aten.silu.default)
                        or len(act.args)!=1 or act.kwargs.get('inplace',False)
                        or any(k!='inplace' for k in act.kwargs) or len(act.users)!=1):
                    continue
                with gm.graph.inserting_before(node):
                    new=gm.graph.call_function(_dispatch_swiglu,(act.args[0],up))
                node.replace_all_uses_with(new)
                gm.graph.erase_node(node);gm.graph.erase_node(act)
                fusions += 1
        if accuracy == 'algebraic' and contraction_policy == 'native':
            mm_targets = (operator.matmul,torch.matmul,torch.mm,torch.ops.aten.mm.default)
            linear_targets = (F.linear,torch.ops.aten.linear.default)
            for node in list(gm.graph.nodes):
                if len(node.args) < 2 or any(k != 'bias' for k in node.kwargs):
                    continue
                if node.op == 'call_function' and node.target in mm_targets:
                    a,b = node.args[:2]
                    with gm.graph.inserting_before(node):
                        new = gm.graph.call_function(_dispatch_mm,(a,b,minimum_inner))
                elif node.op == 'call_method' and node.target in ('matmul','mm','__matmul__'):
                    a,b = node.args[:2]
                    with gm.graph.inserting_before(node):
                        new = gm.graph.call_function(_dispatch_mm,(a,b,minimum_inner))
                elif node.op == 'call_function' and node.target in linear_targets:
                    x,w = node.args[:2]
                    bias = node.args[2] if len(node.args)>2 else node.kwargs.get('bias')
                    with gm.graph.inserting_before(node):
                        new = gm.graph.call_function(_dispatch_linear,(x,w,bias,minimum_inner))
                else:
                    continue
                node.replace_all_uses_with(new)
                gm.graph.erase_node(node)
                replacements += 1
        gm.graph.lint()
        gm.recompile()
        evidence.append({'accuracy':accuracy,'memory_policy':memory_policy,
                         'rewritten_nodes':replacements,'pointwise_fusions':fusions,
                         'contraction_policy':contraction_policy,'linear_chain_rewrites':chains,
                         'matrix_chain_rewrites':planned_chains,
                         'rms_core_rewrites':radial_chains,
                         'lowering':'guarded ATen/custom-op graph; not native machine-code emission'})
        return gm.forward

    backend.evidence = evidence
    return backend
