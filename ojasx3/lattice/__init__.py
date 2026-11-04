"""OjasX3 E8 Gosset Root Lattice & Sphere Packing Subsystem."""

from .transforms import fwht_block32, fast_walsh_hadamard_transform, randomized_hadamard_transform
from .e8 import conway_sloane_e8_project, E8LatticePacker, TernaryE8LatticePacker, RootE8LatticePacker, _FusedE8GEMMFunction
from .layers import E8LatticeLinear, E8QuantizationAwareLinear, pack_model_e8
from .streaming import StreamingE8LoadReport, load_hf_checkpoint_e8, pack_weight_rowwise

__all__ = [
    "fwht_block32",
    "fast_walsh_hadamard_transform",
    "randomized_hadamard_transform",
    "conway_sloane_e8_project",
    "E8LatticePacker",
    "TernaryE8LatticePacker",
    "RootE8LatticePacker",
    "_FusedE8GEMMFunction",
    "E8LatticeLinear",
    "E8QuantizationAwareLinear",
    "pack_model_e8",
    "StreamingE8LoadReport",
    "load_hf_checkpoint_e8",
    "pack_weight_rowwise",
]
