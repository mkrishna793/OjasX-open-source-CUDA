# Copyright 2026 Ojas-Arya Open Foundation & Antigravity Prime Architecture
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ojasx
=====
Top-level compatibility alias for `ojasx2`.
Enables seamless `import ojasx` across all user scripts and workflows.
"""

from ojasx2 import *
import ojasx2 as _ojasx2
from ojasx2 import prepare, optimize, loss

__version__ = _ojasx2.__version__
__all__ = _ojasx2.__all__ if hasattr(_ojasx2, "__all__") else ["prepare", "optimize", "loss", "__version__"]
