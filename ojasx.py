# Copyright (C) 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# This file is part of OjasX.
#
# OjasX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OjasX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OjasX.  If not, see <https://www.gnu.org/licenses/>.

"""
ojasx
=====
Top-level compatibility alias for `ojasx2`.
Enables seamless `import ojasx` across all user scripts and workflows.
"""

from ojasx2 import *
import ojasx2 as _ojasx2
from ojasx2 import prepare, optimize, loss
from ojasx2 import (
    OjasXRunner,
    OjasXRuntime,
    OjasXRunnerLinear,
    OjasXLinear,
    pack_model_runner,
    pack_model_ojasx_runner,
    E8KVCache,
    OjasXRunnerMemoryReport,
    OjasXMemoryReport,
    OptimalRootE8Packer,
    PureLeech24RootPacker,
    MatryoshkaRootLinear,
    fold_consecutive_layers_symplectic,
    pack_matryoshka_runner,
    pack_hybrid_moe_runner,
    HybridMoERunner,
    MatryoshkaRunner,
)

__version__ = _ojasx2.__version__
__all__ = _ojasx2.__all__ if hasattr(_ojasx2, "__all__") else [
    "prepare",
    "optimize",
    "loss",
    "OjasXRunner",
    "OjasXRuntime",
    "OjasXRunnerLinear",
    "OjasXLinear",
    "pack_model_runner",
    "pack_model_ojasx_runner",
    "pack_hybrid_moe_runner",
    "pack_matryoshka_runner",
    "HybridMoERunner",
    "MatryoshkaRunner",
    "E8KVCache",
    "OjasXRunnerMemoryReport",
    "OjasXMemoryReport",
    "OptimalRootE8Packer",
    "PureLeech24RootPacker",
    "MatryoshkaRootLinear",
    "fold_consecutive_layers_symplectic",
    "__version__",
]
