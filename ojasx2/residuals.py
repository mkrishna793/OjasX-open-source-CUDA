"""Exact Boolean residual packing; native tensor byte accounting."""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import torch


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    flat = torch.nn.functional.pad(flat, (0, (-flat.numel()) % 8))
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=flat.device, dtype=torch.uint8)
    return (flat.reshape(-1, 8) * weights).sum(dim=1).to(torch.uint8)


def unpack_mask(packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    n = 1
    for dim in shape:
        n *= dim
    return ((packed[:, None] & weights) != 0).reshape(-1)[:n].reshape(shape)


@dataclass
class SavedTensorAudit:
    """Measures tensors retained for autograd, not total process/GPU peak memory.

    Logical bytes count each saved tensor. Unique storage bytes deduplicate aliases.
    Fresh storage excludes explicitly supplied inputs/parameters; no threshold is zero.
    Hooks disable PyTorch version checks, so use only for measurement on immutable inputs.
    """
    external: List[torch.Tensor] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def key(t):
        storage = t.untyped_storage()
        return (str(t.device), storage.data_ptr(), storage.nbytes())

    def __enter__(self):
        self.external_keys = {self.key(t) for t in self.external}
        self._hook = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hook.__enter__()
        return self

    def _pack(self, tensor):
        self.records.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "logical_bytes": tensor.numel() * tensor.element_size(),
                             "storage_key": self.key(tensor)})
        return tensor

    def __exit__(self, *args):
        return self._hook.__exit__(*args)

    def summary(self):
        keys = {r["storage_key"] for r in self.records}
        return {"saved_tensors": len(self.records),
                "logical_saved_bytes": sum(r["logical_bytes"] for r in self.records),
                "unique_saved_storage_bytes": sum(k[2] for k in keys),
                "fresh_saved_storage_bytes": sum(k[2] for k in keys - self.external_keys),
                "measurement_scope": "autograd-retained tensor storage; not total peak memory"}
