"""E8 Lattice Linear layers for low-RAM training, fine-tuning, and QAT."""

from __future__ import annotations
import math
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn

from ojasx2.math.lattice import (
    E8LatticeLinear,
    E8QuantizationAwareLinear,
    _E8STEFunction,
    pack_model_e8,
)

__all__ = [
    "E8LatticeLinear",
    "E8QuantizationAwareLinear",
    "_E8STEFunction",
    "pack_model_e8",
]
