"""Offline mathematical reference, not a replacement PyTorch backend.

Exact commutative scalar circuits and tensor-valued elementwise experiments.
The counted operations exclude all linear encoders, decoder, copies and memory.
"""
import argparse
import json
from pathlib import Path


def prefix_suffix_pullback(factors, upstream, multiply=None):
    """Division-free shared prefix/suffix circuit for all product cotangents.

    Uses 4*r-6 multiplications for r>=2, zero for r<=1.
    Stores O(r) prefix/suffix entries; no constant-memory claim is made.
    """
    factors = tuple(factors)
    r = len(factors)
    if r == 0:
        return ()
    if r == 1:
        return (upstream,)
    mul = multiply if multiply is not None else lambda x, y: x*y
    prefix = [factors[0]]
    for i in range(1, r-1):
        prefix.append(mul(prefix[-1], factors[i]))
    suffix = [None]*r
    suffix[-1] = factors[-1]
    for i in range(r-2, 0, -1):
        suffix[i] = mul(factors[i], suffix[i+1])
    gradients = [mul(upstream, suffix[1])]
    for i in range(1, r-1):
        gradients.append(mul(upstream, mul(prefix[i-1], suffix[i+1])))
    gradients.append(mul(upstream, prefix[r-2]))
    return tuple(gradients)



def joint_product_pullback(factors, upstream, multiply=None):
    """Rolling cotangent with prefixes rebuilt during backward.

    Counts (r-2)+2*(r-1)=3*r-4 products for r>=2. A conventional reverse
    pass with retained forward prefixes needs just 2*r-2 backward products.
    No division; O(r) temporary prefixes and required gradient outputs.
    """
    factors=tuple(factors)
    r=len(factors)
    if not r:return ()
    if r==1:return (upstream,)
    mul=multiply if multiply is not None else lambda x,y:x*y
    prefixes=[factors[0]]
    for i in range(1,r-1):prefixes.append(mul(prefixes[-1],factors[i]))
    gradients=[None]*r
    rolling=upstream
    for i in range(r-1,0,-1):
        gradients[i]=mul(rolling,prefixes[i-1])
        rolling=mul(rolling,factors[i])
    gradients[0]=rolling
    return tuple(gradients)


def multilinear_circuit(encoders, decoder, operands, upstream):
    """Reference for arbitrary finite, heterogeneous real operand dimensions."""
    if not encoders or len(encoders) != len(operands):
        raise ValueError('Need a nonempty encoder for every operand')
    encoded = tuple(u @ a for u, a in zip(encoders, operands))
    product = encoded[0]
    for e in encoded[1:]:
        product = product*e
    output = decoder @ product
    decoded = decoder.T @ upstream
    channels = joint_product_pullback(encoded, decoded)
    gradients = tuple(u.T @ h for u, h in zip(encoders, channels))
    return output, gradients


def sphere_project(x, g):
    """Euclidean sphere tangent projection; assumes x.T@x=1."""
    return g-x*(x@g)


def stiefel_project(x, g):
    """Euclidean induced metric; assumes x.T@x=I."""
    xtg = x.T@g
    return g-x@((xtg+xtg.T)*0.5)


def explore():
    import sympy as sp
    import torch
    symbolic = []
    for r in range(1, 9):
        factors = sp.symbols('e0:'+str(r))
        upstream = sp.Symbol('g')
        calls = []
        def counted(a, b):
            calls.append(1)
            return a*b
        got = joint_product_pullback(factors, upstream, counted)
        expected = tuple(sp.diff(sp.prod(factors), e)*upstream for e in factors)
        assert all(sp.expand(a-b) == 0 for a, b in zip(got, expected))
        cost = 0 if r == 1 else 3*r-4
        assert len(calls) == cost
        naive = 0 if r == 1 else r*(r-1)
        symbolic.append({'operands':r, 'shared_multiplications':cost,
                         'independent_multiplications':naive,
                         'prefix_suffix_multiplications':0 if r==1 else 4*r-6,
                         'retained_prefix_reverse_multiplications':0 if r==1 else 2*r-2,
                         'reconstruction_overhead_vs_retained_reverse':0 if r==1 else r-2,
                         'symbolic_equal':True,
                         'saving_fraction':None if not naive else (naive-cost)/naive})
    torch.manual_seed(907)
    numerical = []
    for dims in [(2,3), (2,3,4), (3,2,5,4), (2,3,2,3,2,3,2,3)]:
        q,c = 5,3
        encoders = tuple(torch.randn(q,d,dtype=torch.float64) for d in dims)
        operands = tuple(torch.randn(d,dtype=torch.float64,requires_grad=True) for d in dims)
        decoder = torch.randn(c,q,dtype=torch.float64)
        upstream = torch.randn(c,dtype=torch.float64)
        y,gradients = multilinear_circuit(encoders, decoder, operands, upstream)
        reference = torch.autograd.grad(y,operands,upstream)
        for got,expected in zip(gradients,reference):
            torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-10)
        numerical.append({'operand_dimensions':dims,'channels':q,'outputs':c,
                          'all_gradients_match':True,
                          'max_absolute_error':max(float((a-b).abs().max())
                            for a,b in zip(gradients,reference))})
    x=torch.randn(7,dtype=torch.float64);x=x/torch.linalg.vector_norm(x)
    g=torch.randn_like(x);p=sphere_project(x,g)
    torch.testing.assert_close(x@p,torch.zeros((),dtype=x.dtype),atol=1e-12,rtol=0)
    q=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q
    g=torch.randn_like(q);p=stiefel_project(q,g)
    tangent=q.T@p+p.T@q
    torch.testing.assert_close(tangent,torch.zeros_like(tangent),atol=1e-12,rtol=0)
    return {'symbolic_product_circuits':symbolic,'multilinear_gradient_samples':numerical,
            'sphere_and_stiefel_tangency_samples_passed':True,
            'scope':'Exact ring proofs and sampled FP64 math reference; no GPU, native backend, historical novelty or universal 50% claim.',
            'cost_scope':'Scalar product pullback only, excluding encoder/decoder linear contractions, memory and execution time.',
            'storage_scope':'Rolling and prefix/suffix circuits use O(r) temporary scalar or channel-vector entries; operand and output storage remains required. A retained-prefix baseline needs fewer backward products.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/math-structure.json')
    args=parser.parse_args()
    data=explore()
    Path(args.output).write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))
