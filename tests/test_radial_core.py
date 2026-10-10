import itertools
import pytest
import torch
import torch.nn.functional as F
from ojasx2 import rms_core_linear_chain
from ojasx2 import make_backend
import copy


def sample(dtype=torch.double, dimensions=(48, 3, 32, 2)):
    torch.manual_seed(608)
    m, k, n, p = dimensions
    return tuple(a.requires_grad_() for a in
                 (torch.randn(m, k, dtype=dtype), torch.randn(n, k, dtype=dtype),
                  torch.randn(p, n, dtype=dtype), torch.randn(n, dtype=dtype),
                  torch.randn(p, dtype=dtype)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rms_core_output_all_original_gradients_and_hessian_vector(dtype):
    inputs = sample(dtype)
    evidence = []
    a = rms_core_linear_chain(*inputs)
    b = rms_core_linear_chain(*inputs, accuracy='algebraic', evidence=evidence)
    assert evidence[-1]['path'] == 'rms_core'
    ga = torch.autograd.grad(a.square().mean(), inputs, create_graph=True)
    gb = torch.autograd.grad(b.square().mean(), inputs, create_graph=True)
    vectors = tuple(torch.randn_like(x) for x in inputs)
    ha = torch.autograd.grad(sum((g*v).sum() for g, v in zip(ga, vectors)), inputs)
    hb = torch.autograd.grad(sum((g*v).sum() for g, v in zip(gb, vectors)), inputs)
    tol = 5e-5 if dtype == torch.float32 else 2e-11
    for u, v in zip((a, *ga, *ha), (b, *gb, *hb)):
        torch.testing.assert_close(u, v, rtol=tol, atol=tol)


def test_rms_core_every_gradient_demand_and_zero_gamma():
    for needs in itertools.product((False, True), repeat=4):
        xs = sample()
        for x, need in zip(xs, needs):
            x.requires_grad_(need)
        with torch.no_grad():
            xs[3][::2] = 0
        a = rms_core_linear_chain(*xs)
        b = rms_core_linear_chain(*xs, accuracy='algebraic')
        selected = tuple(x for x in xs if x.requires_grad)
        ga = torch.autograd.grad(a.square().sum(), selected)
        gb = torch.autograd.grad(b.square().sum(), selected)
        for u, v in zip((a, *ga), (b, *gb)):
            torch.testing.assert_close(u, v, rtol=2e-11, atol=2e-11)


def test_rms_core_no_cached_weight_version_and_numerical_gradcheck():
    xs = sample(dimensions=(18, 2, 16, 1))
    fn = lambda *args: rms_core_linear_chain(*args, accuracy='algebraic')
    assert torch.autograd.gradcheck(fn, xs, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, xs, fast_mode=True)
    with torch.no_grad():
        xs[1].add_(.2)
    torch.testing.assert_close(fn(*xs), rms_core_linear_chain(*xs), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('case', ['cost', 'zero', 'amp', 'strict'])
def test_rms_core_native_fallbacks(case):
    xs = sample(dimensions=(2, 12, 8, 4) if case == 'cost' else (48, 3, 32, 2))
    if case == 'zero':
        with torch.no_grad():
            xs[0].zero_()
    evidence = []
    with torch.autocast('cpu', enabled=case == 'amp'):
        a = rms_core_linear_chain(*xs)
        b = rms_core_linear_chain(*xs, accuracy='strict' if case == 'strict' else 'algebraic', evidence=evidence)
    assert evidence[-1]['path'] != 'rms_core'
    assert torch.equal(a, b)


def test_rms_core_optional_scale_bias_and_noncontiguous_inputs():
    x, a, b, _, _ = sample()
    x = x.T.contiguous().T
    a = a.T.contiguous().T
    reference = F.linear(F.rms_norm(F.linear(x, a), (a.shape[0],), eps=1e-5), b)
    result = rms_core_linear_chain(x, a, b, accuracy='algebraic')
    torch.testing.assert_close(result, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('bias1, extra_use, eps', [(False, False, 1e-5), (True, False, 1e-5),
                                                (False, True, 1e-5), (False, False, None)])
def test_compile_preserves_model_parameters_and_respects_graph_boundaries(bias1, extra_use, eps):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(3, 32, bias=bias1)
            self.norm = torch.nn.RMSNorm(32, eps=eps)
            self.b = torch.nn.Linear(32, 2)
        def forward(self, x):
            z = self.a(x)
            y = self.b(self.norm(z))
            return y+z.sum(-1, keepdim=True) if extra_use else y
    torch._dynamo.reset()
    torch.manual_seed(607)
    a = Model().double()
    b = copy.deepcopy(a)
    parameters = tuple(b.parameters())
    backend = make_backend(accuracy='algebraic', contraction_policy='rms_core')
    compiled = torch.compile(b, backend=backend, fullgraph=True)
    x = torch.randn(48, 3, dtype=torch.double, requires_grad=True)
    y = a(x)
    z = compiled(x)
    ga = torch.autograd.grad(y.square().mean(), (x, *a.parameters()))
    gb = torch.autograd.grad(z.square().mean(), (x, *b.parameters()))
    assert all(u is v for u, v in zip(parameters, b.parameters()))
    assert backend.evidence[-1]['rms_core_rewrites'] == (0 if bias1 or extra_use else 1)
    for u, v in zip((y, *ga), (z, *gb)):
        torch.testing.assert_close(u, v, rtol=2e-11, atol=2e-11)
    torch._dynamo.reset()
