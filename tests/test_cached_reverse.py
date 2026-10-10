import pytest
import torch
from ojasx2 import reversible_coupling_chain, SavedTensorAudit
from ojasx2.cached_reverse import cached_reverse_cost
from test_reversible_optic import sample


@pytest.mark.parametrize('activation',['gelu','silu'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('interval',[1,2,4])
def test_cached_reconstruction_gradients_and_storage(activation,dtype,interval):
    xs,blocks=sample(8,dtype,m=32,d=8)
    inputs=(*xs,*(v for b in blocks for v in b))
    native=reversible_coupling_chain(*xs,blocks,activation=activation)
    with SavedTensorAudit(list(inputs)) as audit:
        cached=reversible_coupling_chain(*xs,blocks,activation=activation,accuracy='algebraic',
            residual_policy='preactivation',cache_interval=interval)
    g=tuple(torch.randn_like(v) for v in native)
    ga,gb=torch.autograd.grad(native,inputs,g),torch.autograd.grad(cached,inputs,g)
    for a,b in zip(native,cached):assert torch.equal(a,b)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=2e-5 if dtype==torch.float32 else 1e-11,
                                                   atol=2e-5 if dtype==torch.float32 else 1e-11)
    cached_blocks=(8+interval-1)//interval
    assert audit.summary()['fresh_saved_storage_bytes']==(2+2*cached_blocks)*32*8*xs[0].element_size()


@pytest.mark.parametrize('interval',[1,2,4])
def test_actual_gemm_counts_match_refined_theorem(interval):
    xs,blocks=sample(8,torch.float32,m=16,d=8)
    inputs=(*xs,*(v for b in blocks for v in b))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        ys=reversible_coupling_chain(*xs,blocks,activation='silu',accuracy='algebraic',
                                    residual_policy='preactivation',cache_interval=interval)
        torch.autograd.grad(ys,inputs,tuple(torch.ones_like(y) for y in ys))
    count=sum(e.count for e in profiler.key_averages() if e.key in ('aten::mm','aten::addmm'))
    assert count==cached_reverse_cost(8,(8+interval-1)//interval,16,8)['cached_joint_gemms']


def test_shared_frozen_parameters_multiaxis_noncontiguous_cotangents():
    xs,blocks=sample(1,m=12,d=4)
    xs=tuple(x.reshape(2,6,4) for x in xs);blocks=blocks*4
    blocks[0][1].requires_grad_(False)
    selected=(*xs,*(v for v in blocks[0] if v.requires_grad))
    a=reversible_coupling_chain(*xs,blocks,activation='silu')
    b=reversible_coupling_chain(*xs,blocks,activation='silu',accuracy='algebraic',residual_policy='preactivation')
    g=tuple(torch.randn(2,4,6,dtype=x.dtype).transpose(1,2) for x in xs)
    ga,gb=torch.autograd.grad(a,selected,g),torch.autograd.grad(b,selected,g)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)


def test_cache_version_and_higher_order_checks():
    xs,blocks=sample(2)
    fn=lambda:reversible_coupling_chain(*xs,blocks,accuracy='algebraic',residual_policy='preactivation')
    a=fn()
    with pytest.raises(RuntimeError,match='first-order'):
        torch.autograd.grad(sum(v.sum() for v in a),xs,create_graph=True)
    a=fn()
    with torch.no_grad():blocks[0][0].add_(.2)
    with pytest.raises(RuntimeError,match='modified by an inplace'):
        torch.autograd.grad(sum(v.sum() for v in a),xs)


def test_cached_numerical_first_derivative_check():
    xs,blocks=sample(2,m=3,d=2)
    parameters=tuple(v for b in blocks for v in b)
    fn=lambda *a:reversible_coupling_chain(*a[:2],(a[2:6],a[6:]),activation='silu',
                          accuracy='algebraic',residual_policy='preactivation')
    assert torch.autograd.gradcheck(fn,(*xs,*parameters),fast_mode=True)


def test_solver_refinements_and_counterexamples():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from verify_cached_reverse_z3 import verify
    r=verify()
    assert r['all_proofs_passed'] and r['all_refutations_verified']


def test_FP32_cancellation_remains_a_negative_qualification():
    # Native float backprop differentiates the actual saved operands. A real
    # inverse cannot restore an operand erased by a rounded state addition.
    xs=tuple(torch.ones(1,1,requires_grad=True) for _ in range(2))
    block=tuple(v.requires_grad_() for v in (torch.zeros(1,1),torch.tensor([16777216.]),
                                           torch.zeros(1,1),torch.tensor([16777216.])))
    a=reversible_coupling_chain(*xs,(block,),activation='silu',scale=1.)
    b=reversible_coupling_chain(*xs,(block,),activation='silu',scale=1.,accuracy='algebraic',
                                 residual_policy='preactivation')
    ga=torch.autograd.grad(sum(y.sum() for y in a),block)[0]
    gb=torch.autograd.grad(sum(y.sum() for y in b),block)[0]
    assert float(ga)==1. and float(gb)==0.


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
@pytest.mark.parametrize('interval',[1,2])
@pytest.mark.parametrize('backend',['native','metal'])
def test_mps_cached_and_metal_joint_reverse(interval,backend):
    import os
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='1'
    xs,blocks=sample(16,torch.float32,m=128,d=32)
    xs=tuple(x.detach().to('mps').requires_grad_() for x in xs)
    blocks=tuple(tuple(v.detach().to('mps').requires_grad_() for v in b) for b in blocks)
    inputs=(*xs,*(v for b in blocks for v in b))
    a=reversible_coupling_chain(*xs,blocks,activation='silu')
    b=reversible_coupling_chain(*xs,blocks,activation='silu',accuracy='algebraic',
        residual_policy='preactivation',cache_interval=interval,pointwise_backend=backend)
    g=tuple(torch.randn_like(x) for x in xs)
    ga,gb=torch.autograd.grad(a,inputs,g),torch.autograd.grad(b,inputs,g)
    for x,y in zip((*a,*ga),(*b,*gb)):
        assert x.device.type==y.device.type=='mps'
        torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_metal_shared_frozen_parameters_and_strided_cotangents():
    xs,blocks=sample(1,torch.float32,m=24,d=8)
    xs=tuple(x.detach().to('mps').reshape(2,12,8).requires_grad_() for x in xs)
    block=tuple(v.detach().to('mps').requires_grad_() for v in blocks[0])
    block[1].requires_grad_(False);blocks=(block,)*8
    selected=(*xs,*(v for v in block if v.requires_grad))
    a=reversible_coupling_chain(*xs,blocks,activation='silu')
    b=reversible_coupling_chain(*xs,blocks,activation='silu',accuracy='algebraic',
                               residual_policy='preactivation',pointwise_backend='metal')
    g=tuple(torch.randn(2,8,12,device='mps').transpose(1,2) for _ in xs)
    ga,gb=torch.autograd.grad(a,selected,g),torch.autograd.grad(b,selected,g)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-4)
