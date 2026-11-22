"""OjasX3 E8 Gosset Root Lattice & Sphere Packing Subsystem."""

from ojasx2.math.lattice import fwht_block32, fast_walsh_hadamard_transform, randomized_hadamard_transform
from .transforms import (
    cw_weight4_support,
    cw_weight16_support,
    fwht_block,
    apply_circulant_weighing,
    select_incoherence_spec,
    apply_incoherence_transform,
    CWOFTLinear,
)
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
