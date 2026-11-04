"""Regression tests for bounded checkpoint-direct E8 ingestion."""

import json
from pathlib import Path
import runpy

import pytest
import torch
from torch import nn

from ojasx2.math.lattice import E8LatticeLinear, E8LatticePacker, RootE8LatticePacker, TernaryE8LatticePacker, fwht_block32
from ojasx2.streaming_e8 import load_hf_checkpoint_e8, pack_weight_rowwise


@pytest.mark.parametrize(
    "bitrate,packer",
    [
        (2.0, E8LatticePacker),
        (1.5, TernaryE8LatticePacker),
        (1.0, RootE8LatticePacker),
    ],
)
@pytest.mark.parametrize("use_hadamard", [False, True])
def test_rowwise_packing_is_byte_exact_to_whole_matrix(bitrate, packer, use_hadamard):
    torch.manual_seed(410)
    weight = torch.randn(12, 64)
    transformed = fwht_block32(weight) if use_hadamard else weight
    expected_codes, expected_scales, _ = packer.pack(transformed, group_size=32)

    actual_codes, actual_scales = pack_weight_rowwise(
        weight,
        bitrate=bitrate,
        group_size=32,
        row_block=5,  # Deliberately does not divide the row count.
        use_hadamard=use_hadamard,
    )

    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(actual_scales, expected_scales)


def test_rowwise_packing_rejects_layout_that_would_cross_row_groups():
    with pytest.raises(ValueError, match="not divisible"):
        pack_weight_rowwise(torch.randn(2, 30), group_size=32)
    with pytest.raises(ValueError, match="multiple of 32"):
        pack_weight_rowwise(torch.randn(2, 48), group_size=24)


def _tiny_llama_config(tie_word_embeddings=False):
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_sharded_hf_checkpoint_streams_into_e8_and_backpropagates(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(411)
    config = _tiny_llama_config(tie_word_embeddings=False)
    source = transformers.LlamaForCausalLM(config)
    source.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="2KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()

    model, report = load_hf_checkpoint_e8(
        str(tmp_path),
        bitrate=1.5,
        group_size=32,
        lora_rank=2,
        row_block=3,
        use_hadamard=True,
    )

    assert len(report.checkpoint_files) > 1
    assert report.quantized_linear_layers == 8
    assert report.quantized_source_parameters == 14336
    assert report.max_source_row_block_bytes <= 3 * 64 * 4
    assert report.packed_weight_bytes > 0
    assert report.packed_scale_bytes > 0
    assert report.estimated_persistent_tensor_bytes == (
        report.parameter_bytes
        + report.packed_weight_bytes
        + report.packed_scale_bytes
        + report.other_buffer_bytes
    )
    assert not [name for name, value in model.named_parameters() if value.is_meta]
    assert not [name for name, value in model.named_buffers() if value.is_meta]

    source_modules = dict(source.named_modules())
    converted = {name: module for name, module in model.named_modules() if isinstance(module, E8LatticeLinear)}
    assert len(converted) == 8
    for name, layer in converted.items():
        # Streaming must preserve the exact legacy packed representation, not
        # merely produce a similar reconstruction.
        expected = E8LatticeLinear.from_linear(
            source_modules[name],
            rank=2,
            bitrate=1.5,
            group_size=32,
            use_hadamard=True,
        )
        assert torch.equal(layer.packed_weight, expected.packed_weight), name
        assert torch.equal(layer.block_scales, expected.block_scales), name

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    assert report.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    input_ids = torch.tensor([[1, 2, 3, 4]])
    output = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (1, 4, 128)
    output.loss.backward()
    for name in trainable:
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_tied_lm_head_requires_explicit_dense_exclusion(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers")

    source = transformers.LlamaForCausalLM(_tiny_llama_config(tie_word_embeddings=True))
    source.save_pretrained(tmp_path, safe_serialization=True)

    with pytest.raises(ValueError, match="break weight tying"):
        load_hf_checkpoint_e8(str(tmp_path), lora_rank=1)

    model, _ = load_hf_checkpoint_e8(
        str(tmp_path),
        lora_rank=1,
        exclude_modules=("lm_head",),
    )
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, E8LatticeLinear)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not model.lm_head.weight.is_meta
    assert not model.lm_head.weight.requires_grad
    output = model(torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, 128)


def test_missing_dense_checkpoint_tensor_fails_closed(tmp_path):
    pytest.importorskip("accelerate")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")

    source = transformers.LlamaForCausalLM(_tiny_llama_config(tie_word_embeddings=False))
    source.save_pretrained(tmp_path, safe_serialization=True)
    checkpoint_path = tmp_path / "model.safetensors"
    state = safetensors_torch.load_file(str(checkpoint_path))
    state.pop("model.embed_tokens.weight")
    safetensors_torch.save_file(state, str(checkpoint_path), metadata={"format": "pt"})

    with pytest.raises(ValueError, match="missing required dense parameters or persistent buffers"):
        load_hf_checkpoint_e8(str(tmp_path), lora_rank=1)


def test_cli_records_complete_model_mechanics_and_rss(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    pytest.importorskip("psutil")
    transformers = pytest.importorskip("transformers")

    source = transformers.LlamaForCausalLM(_tiny_llama_config(tie_word_embeddings=False))
    source.save_pretrained(tmp_path, safe_serialization=True)
    output_path = tmp_path / "run.json"
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools" / "run_streaming_e8_finetune.py")
    )
    result = namespace["main"]([
        "--checkpoint", str(tmp_path),
        "--output", str(output_path),
        "--max-steps", "1",
        "--allow-synthetic",
        "--sequence-length", "4",
        "--lora-rank", "2",
        "--row-block", "3",
        "--sample-interval", "0.01",
        "--cpu-threads", "2",
    ])
    assert result == 0
    report = json.loads(output_path.read_text())
    assert report["status"] == "completed"
    assert report["complete_model_loaded"] is True
    assert report["training_completed"] is True
    assert report["trainable_parameter_delta_l2"] > 0
    assert len(report["steps"]) == 1
    assert report["steps"][0]["observed_adapter_gradients"] == 16
    assert report["verdict"]["representative_layer_extrapolation"] is False
    assert report["verdict"]["quality_verified"] is False
    assert report["rss"]["observed_peak_bytes"] > 0
    assert report["execution_integration"]["loss"].startswith("stock HuggingFace")
