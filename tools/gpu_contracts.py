"""Shared bounded MPS-only qualification utilities. CPU scalar reporting is OK."""
import os
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
    raise RuntimeError('Disable numerical CPU fallback before importing torch')
import hashlib, statistics, time
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
ROOT = Path(__file__).resolve().parents[1]


class GPUOnly(TorchDispatchMode):
    def __init__(self): self.calls = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        flat, _ = tree_flatten((args, kw))
        if any(isinstance(t, torch.Tensor) and t.device.type != 'mps' for t in flat):
            raise RuntimeError('CPU numerical input blocked: '+str(func))
        device = kw.get('device')
        if device is not None and torch.device(device).type != 'mps':
            raise RuntimeError('CPU allocation blocked: '+str(func))
        result = func(*args, **kw)
        flat, _ = tree_flatten(result)
        if any(isinstance(t, torch.Tensor) and t.device.type != 'mps' for t in flat):
            raise RuntimeError('Non-MPS result blocked: '+str(func))
        self.calls += 1
        return result


def require_gpu():
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; refusing CPU execution')
    torch.set_default_device('mps')


def compare(a, b, tol=2e-4):
    if a is None or b is None: return dict(passed=a is b, none=True)
    if a.shape != b.shape: return dict(passed=False, reason='shape')
    if not a.numel(): return dict(passed=True, empty=True)
    if not (a.is_floating_point() or a.is_complex()): return dict(passed=bool(torch.equal(a,b)))
    finite = torch.isfinite(a) & torch.isfinite(b)
    special_ok = bool(((torch.isnan(a) & torch.isnan(b)) | (a == b) | finite).all())
    aa, bb = torch.where(finite, a, 0), torch.where(finite, b, 0)
    delta = (aa-bb).float()
    norm = (torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(bb.float()).clamp_min(1e-20)).item()
    return dict(passed=special_ok and bool(torch.allclose(aa, bb, atol=tol, rtol=tol)) and norm <= tol,
                max_abs=delta.abs().amax().item(), relative_l2=norm, special_values_match=special_ok)


def hashes(paths):
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def timed(fn, warmup=2, repeats=5):
    for _ in range(warmup): fn()
    torch.mps.synchronize()
    times=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();torch.mps.synchronize();times.append(time.perf_counter()-start)
    return dict(seconds_median=statistics.median(times), seconds_samples=times)
