"""Exact Boolean residual packing; native tensor byte accounting."""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import torch


_PACK_WEIGHTS_CACHE: Dict[Any, torch.Tensor] = {}


def _get_pack_weights(device: torch.device, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        if isinstance(mode, FakeTensorMode) or (ref is not None and is_fake(ref)):
            return torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
    except Exception:
        pass

    if device not in _PACK_WEIGHTS_CACHE:
        w = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
        _PACK_WEIGHTS_CACHE[device] = w
        _PACK_WEIGHTS_CACHE[str(device)] = w
        if hasattr(device, "type") and getattr(device, "index", None) == 0:
            _PACK_WEIGHTS_CACHE[torch.device(device.type)] = w
    return _PACK_WEIGHTS_CACHE[device]


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    flat = torch.nn.functional.pad(flat, (0, (-flat.numel()) % 8))
    weights = _get_pack_weights(flat.device, flat)
    return (flat.reshape(-1, 8) * weights).sum(dim=1).to(torch.uint8)


def unpack_mask(packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    weights = _get_pack_weights(packed.device, packed)
    n = 1
    for dim in shape:
        n *= dim
    return ((packed[:, None] & weights) != 0).reshape(-1)[:n].reshape(shape)


@dataclass
class SavedTensorAudit:
    """Measures tensors retained for autograd via `saved_tensors_hooks` plus MPS allocator deltas.

    Logical bytes count each saved tensor. Unique storage bytes deduplicate aliases.
    Fresh storage excludes explicitly supplied inputs/parameters; no threshold is zero.
    All custom autograd functions in `ojasx2` pass every retained tensor through
    `ctx.save_for_backward(...)` with zero hidden `ctx` tensor attributes.
    """
    external: List[torch.Tensor] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    mps_bytes_before: int = 0
    mps_bytes_after: int = 0

    @staticmethod
    def key(t):
        storage = t.untyped_storage()
        return (str(t.device), storage.data_ptr(), storage.nbytes())

    def __enter__(self):
        self.external_keys = {self.key(t) for t in self.external if isinstance(t, torch.Tensor)}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            self.mps_bytes_before = int(torch.mps.current_allocated_memory())
        self._hook = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hook.__enter__()
        return self

    def _pack(self, tensor):
        self.records.append({
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_bytes": tensor.numel() * tensor.element_size(),
            "storage_key": self.key(tensor),
        })
        return tensor

    def __exit__(self, *args):
        res = self._hook.__exit__(*args)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            self.mps_bytes_after = int(torch.mps.current_allocated_memory())
        return res

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def bytes_saved(self) -> int:
        return sum(r["logical_bytes"] for r in self.records)

    def summary(self):
        keys = {r["storage_key"] for r in self.records}
        return {
            "saved_tensors": len(self.records),
            "logical_saved_bytes": sum(r["logical_bytes"] for r in self.records),
            "unique_saved_storage_bytes": sum(k[2] for k in keys),
            "fresh_saved_storage_bytes": sum(k[2] for k in keys - self.external_keys),
            "mps_allocated_delta_bytes": max(0, self.mps_bytes_after - self.mps_bytes_before),
            "measurement_scope": "autograd-retained tensor storage (100% via save_for_backward) + MPS allocator delta",
        }
