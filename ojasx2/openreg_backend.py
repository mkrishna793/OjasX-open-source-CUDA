"""
ojasx2/openreg_backend.py

OpenReg-Style Accelerator Extender & Dispatcher Subsystem.
1. Binds OjasX to PyTorch's `PrivateUse1` dispatch key (`torch.ojasx`) via `rename_privateuse1_backend`
   and `_register_device_module`, providing stream synchronization, RNG seeding, and unified memory guards.
2. Provides `HolonomicDispatchMode` (`TorchDispatchMode`) to intercept and audit ATen dispatcher calls
   even in dynamic eager execution with Python control flow.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Dict
import gc
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _OjasXDeviceModule(ModuleType):
    """OpenReg-compatible device module registered at `torch.ojasx`."""

    def __init__(self) -> None:
        super().__init__("torch.ojasx")
        self._current_device = 0
        self._seed = 2026

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    def current_device(self) -> int:
        return self._current_device

    def set_device(self, device: int) -> None:
        self._current_device = int(device)

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    def manual_seed(self, seed: int) -> None:
        self._seed = int(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(int(seed))

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed(seed)

    @staticmethod
    def synchronize(device: Any = None) -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    @staticmethod
    def empty_cache() -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()

    @staticmethod
    def memory_allocated() -> int:
        if torch.backends.mps.is_available():
            return int(torch.mps.current_allocated_memory())
        return 0

    @staticmethod
    def set_per_process_memory_fraction(fraction: float = 0.45) -> None:
        if torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(float(fraction))

    def memory_stats(self) -> Dict[str, Any]:
        return {
            "backend": "ojasx_privateuse1",
            "physical_backing": "mps" if torch.backends.mps.is_available() else "cpu",
            "allocated_bytes": self.memory_allocated(),
            "device_count": self.device_count(),
        }


import ctypes
import os

_OPENREG_REGISTERED = False
_PRIVATEUSE1_BOUND = False
_CPP_GUARD_LIB: Any = None
ojasx_device_module = _OjasXDeviceModule()


def register_ojasx_openreg(bind_privateuse1: bool = False) -> _OjasXDeviceModule:
    """
    Load C++ `DeviceGuardImplInterface` (`libojasx_openreg.dylib`) and mount `torch.ojasx`.
    Note: In PyTorch 2.8, calling `rename_privateuse1_backend` overrides `at::getAccelerator()`,
    which conflicts with `MPS` stream checks in `autograd/engine.cpp:1084`. Therefore, `bind_privateuse1`
    is opt-in for standalone `ojasx` device simulations, while `MPS` workloads keep `MPS` as primary accelerator.
    """
    global _OPENREG_REGISTERED, _PRIVATEUSE1_BOUND, _CPP_GUARD_LIB
    if not _OPENREG_REGISTERED:
        dylib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "libojasx_openreg.dylib")
        if os.path.exists(dylib_path):
            _CPP_GUARD_LIB = ctypes.CDLL(dylib_path)
        setattr(torch, "ojasx", ojasx_device_module)
        _OPENREG_REGISTERED = True

    if bind_privateuse1 and not _PRIVATEUSE1_BOUND:
        try:
            torch.utils.rename_privateuse1_backend("ojasx")
        except RuntimeError:
            pass
        try:
            torch._register_device_module("ojasx", ojasx_device_module)
        except RuntimeError:
            pass
        _PRIVATEUSE1_BOUND = True

    return ojasx_device_module


class HolonomicDispatchMode(TorchDispatchMode):
    """
    C++ Dispatcher Boundary Interceptor (`__torch_dispatch__`).
    Audits ATen operator traffic (`aten.mm`, `aten.addmm`, `aten.silu`, `aten.gelu`, `aten._softmax`)
    and records a live operator dispatch ledger across eager and compiled runs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.op_counts: Dict[str, int] = {}
        self.total_dispatched: int = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if kwargs is None:
            kwargs = {}
        op_name = str(func)
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        self.total_dispatched += 1
        return func(*args, **kwargs)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_aten_calls": self.total_dispatched,
            "unique_aten_ops": len(self.op_counts),
            "op_counts": dict(sorted(self.op_counts.items(), key=lambda kv: -kv[1])),
        }


# Automatically initialize OpenReg registration on import
register_ojasx_openreg()
