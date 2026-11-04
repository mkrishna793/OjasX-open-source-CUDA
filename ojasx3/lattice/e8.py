"""Conway-Sloane E8 Gosset Lattice projection and vector quantization."""

from __future__ import annotations
import math
from typing import Tuple, Optional
import torch

from ojasx2.math.lattice import (
    conway_sloane_e8_project,
    E8LatticePacker,
    TernaryE8LatticePacker,
    RootE8LatticePacker,
    _FusedE8GEMMFunction,
)

__all__ = [
    "conway_sloane_e8_project",
    "E8LatticePacker",
    "TernaryE8LatticePacker",
    "RootE8LatticePacker",
    "_FusedE8GEMMFunction",
]
