"""Holonomic Grad-Free Optimizer with E8 moment quantization and exact rational accumulation."""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import torch

from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer

__all__ = ["HolonomicGradFreeOptimizer"]
