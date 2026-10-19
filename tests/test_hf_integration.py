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
tests/test_hf_integration.py
============================

Unit test suite verifying end-to-end HuggingFace model loading, E8 lattice packing,
OjasX graph optimization, and live fine-tuning with HolonomicGradFreeOptimizer.
"""

import pytest
import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import ojasx


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_llama_e8_finetuning():
    """Verifies that a HuggingFace LLaMA model packs into E8 and trains with Adafactor."""
    config = AutoConfig.for_model(
        "llama",
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config)

    # 1. Pack with E8 lattice + rank-8 trainable adapters
    model = ojasx.pack_model_e8(model, rank=8, group_size=32)

    # 2. Optimize graph with OjasX
    model, report = ojasx.optimize_model(model)

    # 3. Initialize in-backward optimizer
    opt = ojasx.HolonomicGradFreeOptimizer(model.parameters(), lr=1e-3, algorithm="adafactor")

    # 4. Forward + backward
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()

    out1 = model(input_ids=input_ids, labels=labels)
    loss1 = out1.loss
    assert torch.isfinite(loss1)

    loss1.backward()

    # 5. Second step: verify loss progression
    out2 = model(input_ids=input_ids, labels=labels)
    loss2 = out2.loss
    assert torch.isfinite(loss2)
    assert loss2.item() < loss1.item()


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_qwen2_e8_finetuning():
    """Verifies that a HuggingFace Qwen2 model packs into E8 and trains with Adafactor."""
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=1500,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config)

    model = ojasx.pack_model_e8(model, rank=8, group_size=32)
    model, report = ojasx.optimize_model(model)

    opt = ojasx.HolonomicGradFreeOptimizer(model.parameters(), lr=1e-3, algorithm="adafactor")

    input_ids = torch.randint(0, 1500, (2, 16))
    labels = input_ids.clone()

    out1 = model(input_ids=input_ids, labels=labels)
    loss1 = out1.loss
    loss1.backward()

    out2 = model(input_ids=input_ids, labels=labels)
    loss2 = out2.loss
    assert loss2.item() < loss1.item()
