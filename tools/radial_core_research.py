"""Reproducible CPU forward+all-gradient qualification, with shape controls."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ojasx2 import rms_core_linear_chain, rms_core_cost
from research_extensions import qualify


def train():
    torch.manual_seed(619)
    a = [torch.nn.Parameter(v) for v in (torch.randn(64,4)*.1, torch.randn(2,64)*.1,
                                         torch.randn(64), torch.randn(2)*.1)]
    b = [torch.nn.Parameter(v.detach().clone()) for v in a]
    oa = torch.optim.AdamW(a, lr=.001)
    ob = torch.optim.AdamW(b, lr=.001)
    max_grad = 0.
    evidence = []
    for _ in range(40):
        # Keep this favorable control away from the small-moment fallback.
        x = torch.nn.functional.normalize(torch.randn(64,4), dim=-1)*.5
        target = torch.randn(64,2)*.1
        ya = rms_core_linear_chain(x, *a)
        yb = rms_core_linear_chain(x, *b, accuracy='algebraic', evidence=evidence)
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        (ya-target).square().mean().backward()
        (yb-target).square().mean().backward()
        for u,v in zip(a,b):
            max_grad = max(max_grad, float(torch.linalg.vector_norm(u.grad-v.grad)/
                                         torch.linalg.vector_norm(u.grad).clamp_min(1e-30)))
        oa.step(); ob.step()
    param_abs = max(float((u-v).abs().max()) for u,v in zip(a,b))
    selected = sum(v['path']=='rms_core' for v in evidence)
    passed = max_grad < 1e-4 and param_abs < 1e-5 and selected == 40
    if not passed:raise RuntimeError('RMS synthetic AdamW trajectory failed')
    return {'steps':40, 'optimizer':'AdamW lr=.001, default remaining parameters',
            'selected_steps':selected,
            'max_gradient_relative_l2':max_grad, 'max_parameter_abs_difference':param_abs,
            'passed':passed, 'scope':'Synthetic favorable-shape regression, not real-data convergence'}


def main():
    torch.set_num_threads(2)
    cases = {}
    for m, k, n, p in ((512, 32, 2048, 16), (128, 256, 512, 128)):
        torch.manual_seed(608)
        xs = tuple(a.requires_grad_() for a in
                   (torch.randn(m,k)*.1, torch.randn(n,k)*.1,
                    torch.randn(p,n)*.1, torch.randn(n), torch.randn(p)*.1))
        g = torch.randn(m,p)
        evidence = []
        key = f'{m}-{k}-{n}-{p}'
        cases[key] = {'dimensions_m_k_n_p': [m,k,n,p], 'cost': rms_core_cost(m,k,n,p),
                      'measurements': qualify('rms-core-'+key,
                       {'native': lambda: rms_core_linear_chain(*xs),
                        'core': lambda: rms_core_linear_chain(*xs, accuracy='algebraic')}, xs,g)}
        rms_core_linear_chain(*xs, accuracy='algebraic', evidence=evidence)
        cases[key]['selection'] = evidence[-1]
        # Retain the existing per-element tolerance results, including failures.
        # This separate experiment qualifies relative L2, not every near-zero entry.
        passed = all(err['relative_l2'] < 1e-5 for mode in cases[key]['measurements'].values()
                     for err in mode['errors_output_then_gradients'])
        cases[key]['relative_l2_qualification'] = {'threshold': 1e-5, 'passed': passed,
            'elementwise_contract': 'Recorded separately; failures are not suppressed'}
        if not passed:
            (ROOT/'reports/radial-core.json').write_text(json.dumps(cases, indent=2))
            raise RuntimeError('RMS relative L2 qualification failed')
    result = {'device': 'cpu', 'torch': torch.__version__, 'threads': 2, 'cases': cases,
              'training': train(),
              'scope': 'Synthetic isolated blocks, randomized timing order. Main-product cost excludes guards/pointwise work. Profiler tensor peak is not whole-process or GPU memory. No whole-model speed claim.'}
    (ROOT/'reports/radial-core.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: {'path': v['selection']['path'],
                       'speedup': v['measurements']['core']['speedup'],
                       'max_gradient_relative_l2': max(e['relative_l2'] for e in v['measurements']['core']['errors_output_then_gradients'][1:])}
                       for key,v in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
