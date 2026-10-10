"""Device/dtype boundary for experimentally qualified algebraic schedules."""
import torch


def algebraic_device_supported(x):
    # MPS has no FP64 tensor support. Other accelerators retain native fallback
    # until they have their own numerical and execution qualification.
    supported = ((x.device.type == 'cpu' and x.dtype in (torch.float32, torch.float64))
                 or (x.device.type == 'mps' and x.dtype == torch.float32))
    return supported and not torch.is_autocast_enabled(x.device.type)
