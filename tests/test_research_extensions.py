import copy
import itertools
import pytest
import torch
import torch.nn.functional as F
from ojasx2 import (matrix_chain, matrix_chain_plan, make_backend, relu_linear_chain,
                    softmax_pullback, layernorm_input_pullback, attention_softmax_row_scalar, SavedTensorAudit,
                    layernorm_linear_chain, gram_layernorm_linear_chain)


@pytest.fixture(autouse=True)
def isolate_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def trees(i, j):
    if i == j:
        return [i]
    return [(a, b) for s in range(i, j) for a in trees(i, s) for b in trees(s+1, j)]


def independent_tree_cost(t, dims, needs):
    if isinstance(t, int):
        return 0, t, t
    ca, i, s = independent_tree_cost(t[0], dims, needs)
    cb, _, j = independent_tree_cost(t[1], dims, needs)
    m, k, n = dims[i], dims[s+1], dims[j+1]
    # Explicit multiply/add counts for each forward and demanded VJP product.
    cost = m*n*(2*k-1)
    if any(needs[i:s+1]):cost += m*k*(2*n-1)
    if any(needs[s+1:j+1]):cost += k*n*(2*m-1)
    return ca+cb+cost, i, j


@pytest.mark.parametrize('dims',[(13, 7, 19, 3), (11, 5, 17, 23, 2), (9, 7, 11, 13, 5, 3)])
def test_joint_planner_is_optimal_over_all_binary_trees_and_demands(dims):
    for need in itertools.product([False, True], repeat=len(dims)-1):
        p = matrix_chain_plan(dims, need)
        costs = [independent_tree_cost(t, dims, need)[0] for t in trees(0, len(dims)-2)]
        assert p['scalar_ops'] == min(costs)
        assert independent_tree_cost(p['tree'], dims, need)[0] == min(costs)
        assert p['native_scalar_ops'] == independent_tree_cost(p['native_tree'], dims, need)[0]


@pytest.mark.parametrize('dtype',[torch.float32, torch.float64])
def test_four_matrix_chain_all_leaf_gradients(dtype):
    torch.manual_seed(24)
    dims = (17, 7, 19, 23, 3)
    xs = tuple((torch.randn(a, b, dtype=dtype)*.1).requires_grad_() for a, b in zip(dims, dims[1:]))
    native = matrix_chain(xs)
    got = matrix_chain(xs, 'algebraic');g = torch.randn_like(got)
    want = torch.autograd.grad(native, xs, g)
    gradients = torch.autograd.grad(got, xs, g)
    tol = 1e-11 if dtype == torch.double else 3e-5
    torch.testing.assert_close(got, native, rtol=tol, atol=tol)
    for a,b in zip(want,gradients):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_planned_higher_derivatives_and_shared_leaf_accumulation():
    torch.manual_seed(21)
    xs = tuple(torch.randn(a,b,dtype=torch.double,requires_grad=True)*.1 for a,b in [(5,3),(3,7),(7,9),(9,1)])
    fn = lambda *args: matrix_chain(args, 'algebraic')
    assert torch.autograd.gradcheck(fn,xs)
    assert torch.autograd.gradgradcheck(fn,xs)
    w = torch.randn(4,4,dtype=torch.double,requires_grad=True)
    x = torch.randn(11,4,dtype=torch.double,requires_grad=True)
    a = matrix_chain((x,w,w,w));b=matrix_chain((x,w,w,w),'algebraic')
    ga=torch.autograd.grad(a.square().sum(),(x,w));gb=torch.autograd.grad(b.square().sum(),(x,w))
    for h,k in zip(ga,gb):torch.testing.assert_close(h,k,rtol=1e-11,atol=1e-11)


def test_planner_cache_cannot_be_corrupted_and_native_autocast():
    p = matrix_chain_plan((11,7,19,23,3));expected=p['tree'];p['tree']='broken'
    assert matrix_chain_plan((11,7,19,23,3))['tree']==expected
    xs = tuple(torch.randn(a,b,requires_grad=True) for a,b in [(11,7),(7,19),(19,23),(23,3)])
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert torch.equal(matrix_chain(xs), matrix_chain(xs,'algebraic'))


def test_actual_compiler_long_chain_and_nonlinear_boundary():
    torch.manual_seed(25)
    model=torch.nn.Sequential(torch.nn.Linear(7,19,bias=False),torch.nn.Linear(19,23,bias=False),torch.nn.Linear(23,3,bias=False)).double()
    other=copy.deepcopy(model);parameters=tuple(other.parameters())
    backend=make_backend(accuracy='algebraic',contraction_policy='matrix_chain')
    compiled=torch.compile(other,backend=backend,fullgraph=True)
    x=torch.randn(11,7,dtype=torch.double,requires_grad=True)
    a=model(x);b=compiled(x);g=torch.randn_like(a)
    ga=torch.autograd.grad(a,(x,*model.parameters()),g);gb=torch.autograd.grad(b,(x,*other.parameters()),g)
    for h,k in zip(ga,gb):torch.testing.assert_close(h,k,rtol=1e-11,atol=1e-11)
    assert backend.evidence[-1]['matrix_chain_rewrites']==1
    assert all(p is q for p,q in zip(parameters,other.parameters()))
    torch._dynamo.reset()
    blocked=torch.nn.Sequential(model[0],torch.nn.ReLU(),model[1],torch.nn.SiLU(),model[2])
    back=make_backend(accuracy='algebraic',contraction_policy='matrix_chain')
    torch.compile(blocked,backend=back,fullgraph=True)(x)
    assert back.evidence[-1]['matrix_chain_rewrites']==0


def common_args(dtype=torch.double, biases=True):
    torch.manual_seed(73)
    x=(torch.randn(11,7,dtype=dtype)*.1).requires_grad_()
    w1=(torch.randn(19,7,dtype=dtype)*.1).requires_grad_()
    w2=(torch.randn(3,19,dtype=dtype)*.1).requires_grad_()
    b1=torch.tensor([3. if i%2 else -3. for i in range(19)],dtype=dtype,requires_grad=True)
    b2=torch.randn(3,dtype=dtype,requires_grad=True)*.1 if biases else None
    return x,w1,w2,b1,b2


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('bias2',[False,True])
def test_common_mask_relu_all_gradients_and_native_forward_bits(dtype,bias2):
    args=common_args(dtype,bias2);x,w1,w2,b1,b2=args
    y=F.linear(F.relu(F.linear(x,w1,b1)),w2,b2)
    evidence=[];z=relu_linear_chain(*args,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path']=='common_mask'
    assert torch.equal(y.contiguous().view(torch.uint8),z.contiguous().view(torch.uint8))
    wanted=tuple(a for a in args if a is not None);g=torch.randn_like(y)
    ga=torch.autograd.grad(y,wanted,g);gb=torch.autograd.grad(z,wanted,g)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
    assert torch.count_nonzero(gb[1][::2])==0
    assert torch.count_nonzero(gb[2][:,::2])==0


def test_common_relu_gradcheck_and_gradgradcheck():
    args=common_args()
    fn=lambda *xs:relu_linear_chain(*xs,accuracy='algebraic')
    assert torch.autograd.gradcheck(fn,args)
    assert torch.autograd.gradgradcheck(fn,args)


def test_common_relu_tape_and_version_checks():
    args=common_args();wanted=[a for a in args if a is not None]
    with SavedTensorAudit(wanted) as audit:y=relu_linear_chain(*args,accuracy='algebraic')
    assert audit.summary()['fresh_saved_storage_bytes']==args[1].shape[0]
    torch.autograd.grad(y,wanted,torch.randn_like(y))
    y=relu_linear_chain(*args,accuracy='algebraic')
    with torch.no_grad():args[1].add_(.001)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):y.sum().backward()


@pytest.mark.parametrize('case',['distinct','nonfinite','strict','autocast'])
def test_common_relu_native_fallbacks(case):
    args=list(common_args(torch.float32))
    if case=='distinct':
        args[3]=None;args[0]=torch.cat((torch.ones(5,7),-torch.ones(6,7))).requires_grad_()
    if case=='nonfinite':
        with torch.no_grad():args[0][0,0]=float('nan')
    evidence=[]
    context=torch.autocast('cpu',enabled=case=='autocast',dtype=torch.bfloat16)
    with context:
        y=F.linear(F.relu(F.linear(args[0],args[1],args[3])),args[2],args[4])
        z=relu_linear_chain(*args,accuracy='strict' if case=='strict' else 'algebraic',evidence=evidence)
    torch.testing.assert_close(y,z,rtol=0,atol=0,equal_nan=True)
    assert evidence[-1]['path']!='common_mask'


def test_zero_boundary_relu_selection_matches_native():
    x=torch.zeros(11,7,requires_grad=True);w1=torch.zeros(19,7,requires_grad=True);w2=torch.randn(3,19,requires_grad=True)
    a=F.linear(F.relu(F.linear(x,w1)),w2);b=relu_linear_chain(x,w1,w2,accuracy='algebraic')
    ga=torch.autograd.grad(a.sum(),(x,w1,w2));gb=torch.autograd.grad(b.sum(),(x,w1,w2))
    for h,k in zip(ga,gb):assert torch.equal(h,k)


@pytest.mark.parametrize('exception',[float('nan'),float('inf'),-float('inf')])
def test_scalar_endpoint_finite_guard_rejects_all_exception_positions(exception):
    from ojasx2.nonlinear import _all_finite
    for position in [0,7,12]:
        v=torch.linspace(-2,2,13)
        assert _all_finite(v)
        v[position]=exception
        assert not _all_finite(v)


@pytest.mark.parametrize('eps',[0.,1e-5,.1])
def test_layernorm_vjp_with_epsilon_and_gamma(eps):
    torch.manual_seed(27)
    x=torch.randn(5,13,dtype=torch.double,requires_grad=True)
    gamma=torch.randn(13,dtype=torch.double,requires_grad=True);g=torch.randn_like(x,requires_grad=True)
    y=F.layer_norm(x,(13,),gamma,eps=eps)
    ref=torch.autograd.grad(y,x,g,create_graph=True)[0]
    got=layernorm_input_pullback(x,g,gamma,eps)
    torch.testing.assert_close(got,ref,rtol=1e-11,atol=1e-11)
    t=torch.randn_like(x)
    ga=torch.autograd.grad(ref,(x,g,gamma),t);gb=torch.autograd.grad(got,(x,g,gamma),t)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    assert torch.allclose(got.sum(-1),torch.zeros(5,dtype=torch.double),atol=1e-12)


def test_softmax_and_attention_scalar_identity():
    torch.manual_seed(15)
    z=torch.randn(11,23,dtype=torch.double,requires_grad=True);g=torch.randn_like(z)
    p=z.softmax(-1);ref=torch.autograd.grad(p,z,g)[0]
    torch.testing.assert_close(softmax_pullback(p,g),ref,rtol=1e-12,atol=1e-12)
    v=torch.randn(23,7,dtype=torch.double);go=torch.randn(11,7,dtype=torch.double)
    explicit=(p*(go@v.T)).sum(-1,keepdim=True)
    torch.testing.assert_close(attention_softmax_row_scalar(p@v,go),explicit,rtol=1e-12,atol=1e-12)


def norm_args(dtype=torch.double,affine=True):
    torch.manual_seed(103)
    x=(torch.randn(17,3,dtype=dtype)*.2).requires_grad_()
    w1=(torch.randn(23,3,dtype=dtype)*.2).requires_grad_()
    w2=(torch.randn(2,23,dtype=dtype)*.2).requires_grad_()
    gamma=torch.linspace(-1,1,23,dtype=dtype,requires_grad=True) if affine else None
    beta=(torch.randn(23,dtype=dtype)*.1).requires_grad_() if affine else None
    bias=torch.randn(2,dtype=dtype,requires_grad=True)
    return x,w1,w2,gamma,beta,bias


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('affine',[False,True])
def test_structured_layernorm_native_forward_and_all_parameter_gradients(dtype,affine):
    args=norm_args(dtype,affine);x,w1,w2,gamma,beta,bias=args
    y=F.linear(F.layer_norm(F.linear(x,w1),(23,),gamma,beta,1e-5),w2,bias)
    evidence=[];z=layernorm_linear_chain(*args,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path']=='structured_layernorm'
    assert torch.equal(y.view(torch.uint8),z.view(torch.uint8))
    wanted=tuple(a for a in args if a is not None);g=torch.randn_like(y)
    ga=torch.autograd.grad(y,wanted,g);gb=torch.autograd.grad(z,wanted,g)
    tol=3e-5 if dtype==torch.float32 else 1e-11
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


@pytest.mark.parametrize('needs',[(False,True,True),(True,False,True),(True,True,False),(False,False,True)])
def test_structured_layernorm_gradient_demand(needs):
    args=list(norm_args())
    for a,need in zip(args,needs):a.requires_grad_(need)
    y=F.linear(F.layer_norm(F.linear(args[0],args[1]),(23,),args[3],args[4]),args[2],args[5])
    z=layernorm_linear_chain(*args,accuracy='algebraic');g=torch.randn_like(y)
    wanted=tuple(a for a in args if a.requires_grad)
    ga=torch.autograd.grad(y,wanted,g);gb=torch.autograd.grad(z,wanted,g)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_structured_layernorm_higher_derivatives_use_native_replay():
    args=norm_args()
    fn=lambda *xs:layernorm_linear_chain(*xs,accuracy='algebraic')
    assert torch.autograd.gradcheck(fn,args)
    assert torch.autograd.gradgradcheck(fn,args)


def test_layernorm_tape_version_and_exceptional_upstream():
    args=norm_args()
    with SavedTensorAudit(args) as audit:y=layernorm_linear_chain(*args,accuracy='algebraic')
    # Only narrow output and two native row-statistics are fresh saved tensors.
    assert audit.summary()['fresh_saved_storage_bytes']==17*(2+2)*8
    g=torch.randn_like(y);g[0,0]=float('nan')
    native=F.linear(F.layer_norm(F.linear(args[0],args[1]),(23,),args[3],args[4]),args[2],args[5])
    ga=torch.autograd.grad(native,args,g);gb=torch.autograd.grad(y,args,g)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    y=layernorm_linear_chain(*args,accuracy='algebraic')
    with torch.no_grad():args[3].add_(.01)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):y.sum().backward()


@pytest.mark.parametrize('case',['strict','unprofitable','autocast','nonfinite'])
def test_layernorm_native_fallback(case):
    args=list(norm_args(torch.float32))
    if case=='unprofitable':args[:3]=[torch.randn(3,17,requires_grad=True),torch.randn(23,17,requires_grad=True),torch.randn(2,23,requires_grad=True)]
    if case=='nonfinite':
        with torch.no_grad():args[0][0,0]=float('nan')
    evidence=[]
    with torch.autocast('cpu',enabled=case=='autocast',dtype=torch.bfloat16):
        y=F.linear(F.layer_norm(F.linear(args[0],args[1]),(23,),args[3],args[4]),args[2],args[5])
        z=layernorm_linear_chain(*args,accuracy='strict' if case=='strict' else 'algebraic',evidence=evidence)
    torch.testing.assert_close(y,z,rtol=0,atol=0,equal_nan=True)
    assert evidence[-1]['path']!='structured_layernorm'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('affine',[False,True])
def test_gram_layernorm_all_original_gradients(dtype,affine):
    args=norm_args(dtype,affine);x,w1,w2,gamma,beta,bias=args
    native=F.linear(F.layer_norm(F.linear(x,w1),(23,),gamma,beta),w2,bias)
    evidence=[];got=gram_layernorm_linear_chain(*args,accuracy='algebraic',evidence=evidence)
    assert evidence[-1]['path']=='gram_layernorm'
    wanted=tuple(a for a in args if a is not None);g=torch.randn_like(native)
    ga=torch.autograd.grad(native,wanted,g);gb=torch.autograd.grad(got,wanted,g)
    tol=3e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(got,native,rtol=tol,atol=tol)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_gram_layernorm_higher_derivatives_and_conservative_fallbacks():
    args=norm_args();fn=lambda *xs:gram_layernorm_linear_chain(*xs,accuracy='algebraic')
    assert torch.autograd.gradcheck(fn,args)
    assert torch.autograd.gradgradcheck(fn,args)
    for changed in ['small_variance','large_offset','strict']:
        xs=list(norm_args(torch.float32))
        with torch.no_grad():
            if changed=='small_variance':xs[0].mul_(1e-4)
            if changed=='large_offset':xs[1].add_(1e4)
        evidence=[];got=gram_layernorm_linear_chain(*xs,accuracy='strict' if changed=='strict' else 'algebraic',evidence=evidence)
        native=F.linear(F.layer_norm(F.linear(xs[0],xs[1]),(23,),xs[3],xs[4]),xs[2],xs[5])
        assert torch.equal(got,native)
        assert evidence[-1]['path']!='gram_layernorm'
