import pytest
import torch
from ojasx2 import reversible_coupling_chain, SavedTensorAudit


def sample(depth,dtype=torch.double,m=12,d=4):
    torch.manual_seed(812)
    xs=tuple(torch.randn(m,d,dtype=dtype,requires_grad=True) for _ in range(2))
    blocks=tuple(tuple((torch.randn(*shape,dtype=dtype)*.1).requires_grad_()
                       for shape in ((d,d),(d,),(d,d),(d,))) for _ in range(depth))
    return xs,blocks


@pytest.mark.parametrize('depth',[1,8,32])
@pytest.mark.parametrize('activation',['gelu','silu'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_reconstruction_replay_all_inputs_and_parameters(depth,activation,dtype):
    xs,blocks=sample(depth,dtype)
    inputs=(*xs,*(v for block in blocks for v in block))
    a=reversible_coupling_chain(*xs,blocks,activation=activation)
    b=reversible_coupling_chain(*xs,blocks,activation=activation,accuracy='algebraic')
    cotangents=tuple(torch.randn_like(v) for v in a)
    ga=torch.autograd.grad(a,inputs,cotangents)
    gb=torch.autograd.grad(b,inputs,cotangents)
    for u,v in zip(a,b):assert torch.equal(u,v)
    tol=1e-5 if dtype==torch.float32 else 1e-11
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=tol,atol=tol)


def test_reversible_frozen_and_shared_parameters_accumulate():
    xs,blocks=sample(1)
    blocks=(blocks[0],blocks[0])
    blocks[0][1].requires_grad_(False)
    parameters=tuple(v for v in blocks[0] if v.requires_grad)
    a=reversible_coupling_chain(*xs,blocks)
    b=reversible_coupling_chain(*xs,blocks,accuracy='algebraic')
    ga=torch.autograd.grad(sum(v.square().sum() for v in a),(*xs,*parameters))
    gb=torch.autograd.grad(sum(v.square().sum() for v in b),(*xs,*parameters))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=1e-11,atol=1e-11)


def test_final_state_tape_is_depth_independent_in_storage_audit():
    fresh=[]
    for depth in (1,8,32):
        xs,blocks=sample(depth)
        with SavedTensorAudit([*xs,*(v for b in blocks for v in b)]) as audit:
            reversible_coupling_chain(*xs,blocks,accuracy='algebraic')
        fresh.append(audit.summary()['fresh_saved_storage_bytes'])
    assert fresh==[2*12*4*8]*3


def test_parameter_update_before_replay_is_rejected():
    xs,blocks=sample(2)
    y=reversible_coupling_chain(*xs,blocks,accuracy='algebraic')
    with torch.no_grad():blocks[0][0].add_(.1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        sum(v.sum() for v in y).backward()


def test_higher_order_uses_native_and_replay_rejects_it_explicitly():
    xs,blocks=sample(1,m=3,d=2)
    params=tuple(v for b in blocks for v in b)
    fn=lambda *args:reversible_coupling_chain(*args[:2],(args[2:],))
    assert torch.autograd.gradgradcheck(fn,(*xs,*params),fast_mode=True)
    y=reversible_coupling_chain(*xs,blocks,accuracy='algebraic')
    with pytest.raises(RuntimeError,match='first-order'):
        torch.autograd.grad(sum(v.sum() for v in y),xs,create_graph=True)


def test_replay_numerical_gradcheck_and_multiaxis():
    xs,blocks=sample(2,m=6,d=2)
    xs=tuple(x.reshape(2,3,2) for x in xs)
    params=tuple(v for b in blocks for v in b)
    fn=lambda *args:reversible_coupling_chain(*args[:2],(args[2:6],args[6:]),accuracy='algebraic')
    assert torch.autograd.gradcheck(fn,(*xs,*params),fast_mode=True)
