"""Representative synthetic graph families, not every ML architecture/dataset."""
import copy
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from ojasx2 import make_backend, rms_core_linear_chain


class Core(nn.Module):
    def __init__(self, d=16, n=128):
        super().__init__()
        self.a = nn.Linear(d, n, bias=False)
        self.norm = nn.RMSNorm(n, eps=1e-5)
        self.b = nn.Linear(n, d)
    def forward(self, x):
        return self.b(self.norm(self.a(x)))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Core() for _ in range(3)])
    def forward(self, x):
        for block in self.blocks:
            x = x+block(x)*.1
        return x


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core()
    def forward(self, x):
        return self.core(x)+self.core(x*.8)


class Conv(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(16)
        self.head = Core()
    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        return self.head(h.mean((-1, -2)))


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 16)
        self.hidden = nn.Linear(16, 16)
        self.core = Core()
    def forward(self, x):
        h = torch.zeros_like(x[:, 0])
        for t in range(x.shape[1]):
            h = torch.tanh(self.input(x[:, t])+self.hidden(h))
        return self.core(h)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(16, eps=1e-5)
        self.qkv = nn.Linear(16, 48, bias=False)
        self.proj = nn.Linear(16, 16, bias=False)
        self.ffnorm = nn.RMSNorm(16, eps=1e-5)
        self.gate = nn.Linear(16, 64, bias=False)
        self.up = nn.Linear(16, 64, bias=False)
        self.down = nn.Linear(64, 16, bias=False)
    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        h = x+self.proj(F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], is_causal=True)[:, 0])
        a = self.ffnorm(h)
        return h+self.down(F.silu(self.gate(a))*self.up(a))


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core()
    def forward(self, x):
        z = self.core.a(x)
        return self.core.b(self.core.norm(z))+z.mean(-1, keepdim=True)


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.norm = nn.RMSNorm(16, eps=1e-5)
    def forward(self, x):
        return F.linear(self.norm(self.embedding(x)), self.embedding.weight)


def model_case(name):
    if name == 'residual_tokens':return Residual(), torch.randn(8, 32, 16), 3
    if name == 'shared_weights':return Shared(), torch.randn(128, 16), 2
    if name == 'convolution_bn':return Conv(), torch.randn(128, 3, 8, 8), 1
    if name == 'recurrent_tanh':return Recurrent(), torch.randn(128, 4, 16), 1
    if name == 'standard_transformer_swiglu':return Transformer(), torch.randn(4, 16, 16), 0
    if name == 'branch_hidden_consumer':return Branch(), torch.randn(128, 16), 0
    if name == 'embedding_tied_decoder':return Embedding(), torch.randint(0, 64, (4, 16)), 0
    if name == 'gelu_dropout':
        return nn.Sequential(Core(), nn.GELU(), nn.Dropout(.2), nn.Linear(16, 4)), torch.randn(128, 16), 1
    raise ValueError(name)


NAMES = ['residual_tokens', 'shared_weights', 'convolution_bn', 'recurrent_tanh',
         'standard_transformer_swiglu', 'branch_hidden_consumer', 'embedding_tied_decoder', 'gelu_dropout']


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_representative_models_accumulation_optimizer_and_aliases(name, dtype):
    torch.manual_seed(702)
    torch._dynamo.reset()
    a, x, rewrites = model_case(name)
    a = a.to(dtype=dtype)
    if x.is_floating_point():x = x.to(dtype=dtype)
    b = copy.deepcopy(a)
    original_parameters = tuple(b.parameters())
    backend = make_backend(accuracy='algebraic', contraction_policy='rms_core')
    compiled = torch.compile(b, backend=backend, fullgraph=True)
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-3)
    tol = 2e-4 if dtype == torch.float32 else 1e-10
    for step in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        # Two microbatches at one weight version before the optimizer update.
        for microbatch in range(2):
            torch.manual_seed(720+step*2+microbatch)
            y = a(x)
            torch.manual_seed(720+step*2+microbatch)
            z = compiled(x)
            torch.testing.assert_close(y, z, rtol=tol, atol=tol)
            (y.square().mean()/2).backward()
            (z.square().mean()/2).backward()
        assert backend.evidence[-1]['rms_core_rewrites'] == rewrites
        for p, q in zip(a.parameters(), b.parameters()):
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=tol, atol=tol)
        oa.step(); ob.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=tol, atol=tol)
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(oa.state[p][key], ob.state[q][key], rtol=tol, atol=tol)
        for p, q in zip(a.buffers(), b.buffers()):
            torch.testing.assert_close(p, q, rtol=tol, atol=tol)
    assert all(p is q for p, q in zip(original_parameters, b.parameters()))
    torch._dynamo.reset()


@pytest.mark.parametrize('shape,k,n,p', [((256,),8,128,4), ((4,64),8,128,4),
    ((2,4,32),8,128,4), ((64,),32,64,16), ((1,),16,256,4),
    ((1024,),32,4096,16), ((128,),64,512,32), ((256,),16,128,64)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scaled_shapes_and_all_gradients(shape, k, n, p, dtype):
    torch.manual_seed(711)
    xs = tuple(v.requires_grad_() for v in (torch.randn(*shape,k,dtype=dtype),
         torch.randn(n,k,dtype=dtype)*.1, torch.randn(p,n,dtype=dtype)*.1,
         torch.randn(n,dtype=dtype)))
    a = rms_core_linear_chain(*xs)
    evidence = []
    b = rms_core_linear_chain(*xs, accuracy='algebraic', evidence=evidence)
    g = torch.randn_like(a)
    ga = torch.autograd.grad(a,xs,g)
    gb = torch.autograd.grad(b,xs,g)
    # Scale-aware norm contract; per-entry differences are covered separately.
    for u, v in zip((a,*ga),(b,*gb)):
        error = torch.linalg.vector_norm(u-v)/torch.linalg.vector_norm(u).clamp_min(1e-30)
        assert float(error.detach()) < (1e-5 if dtype == torch.float32 else 1e-11)
    if len(shape)>1:
        assert evidence[-1]['path']=='rms_core'


def test_noncontiguous_tokens_preserve_native_fallback():
    x = torch.randn(32,8,4).transpose(0,1).requires_grad_()
    a = torch.randn(64,4,requires_grad=True)
    b = torch.randn(2,64,requires_grad=True)
    evidence=[]
    y = rms_core_linear_chain(x,a,b,accuracy='algebraic',evidence=evidence)
    z = rms_core_linear_chain(x,a,b)
    assert evidence[-1]['path']=='unsupported'
    assert torch.equal(y,z)


def test_batched_token_second_derivatives():
    torch.manual_seed(718)
    xs=tuple(v.requires_grad_() for v in (torch.randn(2,16,3,dtype=torch.double),
             torch.randn(32,3,dtype=torch.double),torch.randn(2,32,dtype=torch.double),
             torch.randn(32,dtype=torch.double)))
    fn=lambda *args:rms_core_linear_chain(*args,accuracy='algebraic')
    assert torch.autograd.gradcheck(fn,xs,fast_mode=True)
    assert torch.autograd.gradgradcheck(fn,xs,fast_mode=True)


def batchnorm_bias_negative_qualification():
    torch.manual_seed(702)
    a = Conv(bias=True)
    x = torch.randn(128,3,8,8)
    b = copy.deepcopy(a)
    compiled = torch.compile(b, backend=make_backend(accuracy='algebraic', contraction_policy='rms_core'), fullgraph=True)
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-3)
    max_difference = 0.
    for step in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        for _ in range(2):
            y = a(x); z = compiled(x)
            (y.square().mean()/2).backward(); (z.square().mean()/2).backward()
        if step == 0:
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-4)
            first_bias_grad = float(a.conv.bias.grad.abs().max())
            first_bias_grad_delta = float((a.conv.bias.grad-b.conv.bias.grad).abs().max())
        oa.step(); ob.step()
        max_difference = max(max_difference, float((a.conv.bias-b.conv.bias).detach().abs().max()))
    torch._dynamo.reset()
    return {'model':'Conv2d with bias -> BatchNorm -> ReLU -> pooling -> RMS core',
            'dtype':'float32', 'optimizer':'AdamW lr=.001, default eps=1e-8', 'steps':3,
            'first_step_bias_gradient_max_abs':first_bias_grad,
            'first_step_bias_gradient_max_abs_delta':first_bias_grad_delta,
            'maximum_bias_parameter_abs_delta':max_difference,
            'parameter_contract_atol':2e-4,
            'qualified':max_difference<=2e-4,
            'local_gradient_comparison_passed':True,
            'implication':'Local gradient tolerance does not guarantee optimizer trajectory tolerance. This case needs strict execution or separate numerical qualification.'}


def test_fp32_batchnorm_bias_adamw_can_amplify_small_gradient_differences():
    """Known negative qualification: acceptable local VJP error is insufficient."""
    report = batchnorm_bias_negative_qualification()
    assert not report['qualified']
