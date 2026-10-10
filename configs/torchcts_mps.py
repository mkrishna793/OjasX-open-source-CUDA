# OjasX bounded adapter configuration. Stock CTS CPU-reference runner is not authorized by GPU-only testing.
# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Description: device registration + ~20 core ops (float32/int64/bool only)

import torch

manifest = {
    "manifest_version": 1,
    "device_name": "mps",
    "backend_import": None,
    "supported_dtypes": {
        torch.float32: True,
        torch.int64: True,
        torch.bool: True,
    },
    "device_count": 1,
    "ieee754_seed": 67,
    "max_samples": 5,            # Max passing samples per test node (clean tier). 0 = no cap.
    "max_samples_ieee754": 3,     # Max passing samples per test node (NaN/Inf tiers). 0 = no cap.
    "semantic_level": 2,          # Minimal correctness without specialized backend features.
    "hardware": {
        "memory_model": "unified",
        "device_memory_gb": "auto",
        "system_memory_gb": "auto",
        "oom_recoverable": True,
    },
    "resource_limits": {
        "max_device_memory_mb": 512,
        "max_system_memory_mb": 1024,
        "max_tensor_size_mb": 8,
        "cleanup_threshold_pct": 80,
    },
    "capabilities": {
        "inference": True,
        "training": True,
        "serialization": False,
        "rng": False,
        "device_generator": False,
        "rng_distributions": False,
        "double_backward": False,
        "gradcheck": False,
        "gradient_checkpointing": False,
        "autocast": False,
        "fused_optimizer": False,
        "dataloader": False,
        "module_hooks": False,
        "channels_last": False,
        "sparse": False,
        "nested": False,
        "named_tensor": False,
        "foreach": False,
        "fp8": False,
        "quantized_container_plumbing": False,
        "native_quantization": False,
        "custom_quantized_decode": False,
        "compile": False,
        "pinned_memory": False,
        "streams": False,
        "events": False,
        "deterministic": False,
        "guard_alloc": False,
        "device_api": True,
        "multi_device": False,
        "ieee754": False,
    },
    "skip_ops": [],
    "tolerance_overrides": {},
    "supported_container_formats": {},
    # Optional semantic decode hooks for custom_quantized_decode.
    # Values are "module:function" callables:
    # decode(packed, scale, zero_point, shape, dtype, device) -> Tensor.
    "custom_container_decoders": {},
    "custom_test_dirs": [],
}
