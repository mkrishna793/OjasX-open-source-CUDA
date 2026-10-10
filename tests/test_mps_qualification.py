"""Real MPS execution; sampled FP32 conformance, no implicit CPU fallback."""
import copy
import itertools
import os
import pytest
import torch
import torch.nn.functional as F
from ojasx2 import (linear_chain, matrix_chain, rms_core_linear_chain,
                    reversible_coupling_chain, streamed_gemm, gelu, silu,
                    swiglu, rmsnorm, make_backend, SavedTensorAudit)
from test_workload_coverage import model_case, NAMES

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')


@pytest.fixture(autouse=True)
def explicit_gpu_contract():
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1'
    torch.manual_seed(901)


def compare(a, b, tol=2e-4):
    assert a.device.type == b.device.type == 'mps'
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)
    denominator = torch.linalg.vector_norm(a).clamp_min(1e-20)
    assert float((torch.linalg.vector_norm(a-b)/denominator).detach()) < tol


@pytest.mark.parametrize('mode', ['joint', 'backward_only'])
@pytest.mark.parametrize('needs', [n for n in itertools.product((False, True), repeat=3) if any(n)])
def test_linear_all_gradient_demands(mode, needs):
    xs = tuple((torch.randn(*shape, device='mps')*.1).requires_grad_(need)
               for shape, need in zip(((128, 8), (256, 8), (4, 256)), needs))
    a = linear_chain(*xs)
    b = linear_chain(*xs, accuracy='algebraic', mode=mode)
    g = torch.randn_like(a)
    selected = tuple(x for x in xs if x.requires_grad)
    ga = torch.autograd.grad(a, selected, g)
    gb = torch.autograd.grad(b, selected, g)
    for u, v in zip((a, *ga), (b, *gb)):
        compare(u, v)
    with SavedTensorAudit(list(xs)) as audit:
        linear_chain(*xs, accuracy='algebraic', mode=mode)
    assert not any(r['shape'] == [128, 256] for r in audit.records)


@pytest.mark.parametrize('op', ['gelu', 'gelu_tanh', 'silu', 'swiglu', 'rmsnorm', 'rank7', 'matrix_chain'])
def test_ops_forward_and_vjp(op):
    x = torch.randn(32, 16, device='mps', requires_grad=True)
    other = torch.randn_like(x, requires_grad=True)
    if op == 'gelu': fn, ref, xs = gelu, F.gelu, (x,)
    elif op == 'gelu_tanh': fn, ref, xs = lambda a:gelu(a, 'tanh'), lambda a:F.gelu(a, approximate='tanh'), (x,)
    elif op == 'silu': fn, ref, xs = silu, F.silu, (x,)
    elif op == 'swiglu': fn, ref, xs = swiglu, lambda a,b:F.silu(a)*b, (x,other)
    elif op == 'rmsnorm':
        gamma = torch.randn(16, device='mps', requires_grad=True)
        fn, ref, xs = rmsnorm, lambda a,g:F.rms_norm(a,(16,),g,1e-6), (x,gamma)
    elif op == 'rank7':
        b = torch.randn(16, 8, device='mps', requires_grad=True)
        fn, ref, xs = lambda a,b:streamed_gemm(a,b,'algebraic'), lambda a,b:a@b, (x,b)
    else:
        xs = (x, torch.randn(16, 256, device='mps', requires_grad=True),
              torch.randn(256, 8, device='mps', requires_grad=True))
        fn, ref = lambda *args:matrix_chain(args,'algebraic'), lambda *args:matrix_chain(args)
    a, b = ref(*xs), fn(*xs)
    g = torch.randn_like(a)
    ga, gb = torch.autograd.grad(a,xs,g), torch.autograd.grad(b,xs,g)
    for u,v in zip((a,*ga),(b,*gb)): compare(u,v)


@pytest.mark.parametrize('case', ['optimized', 'zero_gamma', 'conditioning', 'unprofitable', 'noncontiguous', 'autocast'])
def test_rms_path_and_fallbacks(case):
    shape = (2,32,8,16) if case == 'unprofitable' else (128,8,256,4)
    m,k,n,p = shape
    x,a,b,gamma,bias = tuple((torch.randn(*s,device='mps')*.2).requires_grad_()
                             for s in ((m,k),(n,k),(p,n),(n,),(p,)))
    if case == 'zero_gamma':
        with torch.no_grad(): gamma[::2] = 0
    if case == 'conditioning':
        with torch.no_grad(): x.zero_()
    if case == 'noncontiguous': x=x.reshape(4,32,k).transpose(0,1)
    evidence=[]
    with torch.autocast('mps', enabled=case == 'autocast'):
        y=rms_core_linear_chain(x,a,b,gamma,bias)
        z=rms_core_linear_chain(x,a,b,gamma,bias,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path'] == {
        'optimized':'rms_core','zero_gamma':'rms_core','conditioning':'conditioning',
        'unprofitable':'unprofitable','noncontiguous':'unsupported','autocast':'unsupported'}[case]
    g=torch.randn_like(y)
    xs=(x,a,b,gamma,bias)
    gy,gz=torch.autograd.grad(y,xs,g),torch.autograd.grad(z,xs,g)
    for u,v in zip((y,*gy),(z,*gz)): compare(u,v,3e-4 if case == 'autocast' else 2e-4)


@pytest.mark.parametrize('activation', ['gelu', 'silu'])
def test_reversible_local_replay_and_tape(activation):
    xs=tuple(torch.randn(64,16,device='mps',requires_grad=True) for _ in range(2))
    blocks=tuple(tuple((torch.randn(*s,device='mps')*.1).requires_grad_()
                       for s in ((16,16),(16,),(16,16),(16,))) for _ in range(16))
    inputs=(*xs,*(p for b in blocks for p in b))
    a=reversible_coupling_chain(*xs,blocks,activation=activation)
    with SavedTensorAudit(list(inputs)) as audit:
        b=reversible_coupling_chain(*xs,blocks,activation=activation,accuracy='algebraic')
    g=tuple(torch.randn_like(x) for x in a)
    ga,gb=torch.autograd.grad(a,inputs,g),torch.autograd.grad(b,inputs,g)
    for u,v in zip((*a,*ga),(*b,*gb)):compare(u,v)
    assert audit.summary()['fresh_saved_storage_bytes'] == 2*64*16*4


@pytest.mark.parametrize('op', ['linear', 'rms'])
def test_second_derivative_directional_comparison(op):
    xs=tuple((torch.randn(*s,device='mps')*.2).requires_grad_()
             for s in ((64,4),(64,4),(2,64)))
    fn=linear_chain if op=='linear' else rms_core_linear_chain
    a,b=fn(*xs),fn(*xs,accuracy='algebraic')
    ga=torch.autograd.grad(a.square().mean(),xs,create_graph=True)
    gb=torch.autograd.grad(b.square().mean(),xs,create_graph=True)
    vs=tuple(torch.randn_like(x) for x in xs)
    ha=torch.autograd.grad(sum((g*v).sum() for g,v in zip(ga,vs)),xs)
    hb=torch.autograd.grad(sum((g*v).sum() for g,v in zip(gb,vs)),xs)
    for u,v in zip((*ga,*ha),(*gb,*hb)):compare(u,v)


@pytest.mark.parametrize('value', [float('nan'),float('inf'),float('-inf')])
def test_nonfinite_rms_inputs_keep_native_graph(value):
    x,a,b=tuple(torch.randn(*s,device='mps',requires_grad=True)
                for s in ((64,4),(64,4),(2,64)))
    with torch.no_grad():a[0,0]=value
    evidence=[]
    y=rms_core_linear_chain(x,a,b)
    z=rms_core_linear_chain(x,a,b,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path']=='nonfinite'
    torch.testing.assert_close(y,z,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('needs', list(itertools.product((False,True),repeat=4)))
def test_rms_original_parameter_gradient_demands(needs):
    xs=tuple((torch.randn(*s,device='mps')*.2).requires_grad_(need)
             for s,need in zip(((128,8),(256,8),(4,256),(256,),(4,)),(*needs,True)))
    with torch.no_grad():xs[3][::2]=0
    a=rms_core_linear_chain(*xs)
    evidence=[]
    b=rms_core_linear_chain(*xs,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path']=='rms_core'
    selected=tuple(x for x in xs if x.requires_grad)
    g=torch.randn_like(a)
    ga,gb=torch.autograd.grad(a,selected,g),torch.autograd.grad(b,selected,g)
    for u,v in zip((a,*ga),(b,*gb)):compare(u,v)


@pytest.mark.parametrize('name', NAMES)
def test_compiled_models_accumulation_and_adamw(name):
    torch._dynamo.reset()
    a,x,rewrites=model_case(name)
    a=a.to('mps');x=x.to('mps');b=copy.deepcopy(a)
    parameters=tuple(b.parameters())
    backend=make_backend(accuracy='algebraic',contraction_policy='rms_core')
    compiled=torch.compile(b,backend=backend,fullgraph=True)
    oa=torch.optim.AdamW(a.parameters(),lr=.001)
    ob=torch.optim.AdamW(b.parameters(),lr=.001)
    for step in range(3):
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        for micro in range(2):
            torch.manual_seed(920+2*step+micro);y=a(x)
            torch.manual_seed(920+2*step+micro);z=compiled(x)
            compare(y,z)
            (y.square().mean()/2).backward();(z.square().mean()/2).backward()
        for p,q in zip(a.parameters(),b.parameters()):compare(p.grad,q.grad)
        oa.step();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            compare(p,q)
            for key in ('exp_avg','exp_avg_sq'):compare(oa.state[p][key],ob.state[q][key])
        for p,q in zip(a.buffers(),b.buffers()):
            if p.is_floating_point():compare(p,q)
            else:assert torch.equal(p,q)
    assert backend.evidence[-1]['rms_core_rewrites']==rewrites
    assert all(p is q for p,q in zip(parameters,b.parameters()))
    torch._dynamo.reset()
