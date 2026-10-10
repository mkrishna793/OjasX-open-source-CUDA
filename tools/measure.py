"""CPU reference measurements; saved storage and operator events, not GPU speed."""
import json
import platform
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ojasx2 import packed_relu, swiglu, rmsnorm, shared_projections, optimize_relu_residuals
from ojasx2.residuals import SavedTensorAudit
from ojasx2.bilinear import winograd_forward, winograd_pullback, forward_cost


def saved(fn, external):
    with SavedTensorAudit(external) as audit:
        y = fn()
    y.sum().backward()
    return audit.summary()


def gemms(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
    counts = {e.key: e.count for e in prof.key_averages() if e.key in ('aten::mm', 'aten::bmm', 'aten::addmm')}
    return {"events": counts, "total": sum(counts.values()), "scope": "ATen CPU events, includes forward and explicit backward"}


def measure():
    torch.manual_seed(20261006)
    torch.set_num_threads(2)
    result = {"runtime": {"python": platform.python_version(), "torch": torch.__version__,
              "machine": platform.machine(), "cuda": torch.cuda.is_available(),
              "mps": torch.backends.mps.is_available()}, "device": "cpu", "dtype": "float32"}
    x = torch.randn(256, 1024, requires_grad=True)
    result['relu_saved_storage'] = {"shape": list(x.shape), "eager": saved(lambda: F.relu(x), [x]),
                                   "packed": saved(lambda: packed_relu(x), [x])}
    gate, up = (torch.randn(64, 1024, requires_grad=True) for _ in range(2))
    result['swiglu_saved_storage'] = {"eager": saved(lambda: F.silu(gate)*up, [gate, up]),
                                    "rebuilt": saved(lambda: swiglu(gate, up), [gate, up])}
    gamma = torch.ones(1024, requires_grad=True)
    result['rmsnorm_saved_storage'] = {
        "eager_expression": saved(lambda: (x * torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6))*gamma, [x, gamma]),
        "rebuilt": saved(lambda: rmsnorm(x, gamma), [x, gamma]),
        "contract": "expression forward bits; tolerance-gradient validation; extra row statistic retained"}
    model = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 17))
    rebuilt = optimize_relu_residuals(model, policy='pack')
    conservative = optimize_relu_residuals(model)
    model_input = torch.randn(64, 128, requires_grad=True)
    result['mlp_saved_storage'] = {"eager": saved(lambda: model(model_input), [model_input, *model.parameters()]),
                                 "forced_packed": saved(lambda: rebuilt(model_input), [model_input, *rebuilt.parameters()]),
                                 "conservative": saved(lambda: conservative(model_input), [model_input, *conservative.parameters()]),
                                 "finding": "Next Linear saves ReLU output. A separate mask adds storage; conservative Sequential policy avoids it."}

    def projections(mode):
        p = torch.randn(64, 128, requires_grad=True)
        ws = tuple(torch.randn(128, 128, requires_grad=True) for _ in range(3))
        ys = shared_projections(p, ws, mode)
        torch.autograd.grad(ys, (p, *ws), tuple(torch.ones_like(y) for y in ys))
    result['qkv_gemm_events'] = {mode: gemms(lambda: projections(mode)) for mode in ('strict', 'algebraic')}
    result['qkv_gemm_events']['contract'] = 'Packing changes floating reductions. Leading arithmetic unchanged; fewer dispatches only.'

    a, b = (torch.randn(32, 32) for _ in range(2))
    g = torch.randn(32, 32)
    result['winograd_gemm_events'] = gemms(lambda: winograd_pullback(winograd_forward(a, b)[0], g))
    result['winograd_scalar_models'] = {str(n): forward_cost(n, n, n) for n in (12, 32, 128, 512)}
    result['memory_scope'] = 'Autograd saved-tensor storage, excludes total peak, ephemeral pack/unpack buffers, allocator caches, optimizer state.'
    return result


if __name__ == '__main__':
    data = measure()
    Path('reports/measurements.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))
