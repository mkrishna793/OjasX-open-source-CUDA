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
ojasx2.runtime (OjasX Runner)
=============================
Standalone Ultra-Low-RAM Inference & Execution Engine for OjasX.
Strictly isolated from training autograd and TorchFX so it can be deployed
or extracted as a standalone runtime package at any time.
"""

from .leech_and_lut_math import (
    Leech24LatticePacker,
    conway_sloane_leech24_project,
    PrecomputedTernaryLUT,
)
from .runner_linear import (
    OjasXRunnerLinear,
    OjasXLinear,
    pack_model_runner,
    pack_model_ojasx_runner,
)
from .kv_cache_e8 import (
    E8KVCache,
)
from .hf_auto_runner import (
    OjasXRunner,
    OjasXRuntime,
    OjasXRunnerMemoryReport,
    OjasXMemoryReport,
)

__all__ = [
    "Leech24LatticePacker",
    "conway_sloane_leech24_project",
    "PrecomputedTernaryLUT",
    "OjasXRunnerLinear",
    "OjasXLinear",
    "pack_model_runner",
    "pack_model_ojasx_runner",
    "E8KVCache",
    "OjasXRunner",
    "OjasXRuntime",
    "OjasXRunnerMemoryReport",
    "OjasXMemoryReport",
]
