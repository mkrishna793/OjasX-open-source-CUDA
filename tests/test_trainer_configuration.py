"""Behavioral tests for trainer precision and fused-loss configuration."""

from unittest.mock import patch

import torch
import torch.nn as nn

import ojasx2.trainer as trainer_module
from ojasx2.trainer import LowRAMTrainer, train


class RecordingLinear(nn.Linear):
    def forward(self, inputs):
        output = super().forward(inputs)
        self.last_output_dtype = output.dtype
        return output


def _batch():
    return [(torch.randn(3, 4), torch.tensor([0, 1, 2]))]


def test_bfloat16_selects_autocast_with_fp32_master_weights():
    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.bfloat16,
        quantize_moments=False, fused_loss=False,
    )
    trainer.train_epoch(_batch(), log_interval=100)
    assert trainer.use_autocast is True
    assert model.last_output_dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_float64_converts_model_and_output_precision():
    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.float64,
        quantize_moments=False, fused_loss=False,
    )
    trainer.train_epoch(_batch(), log_interval=100)
    assert trainer.use_autocast is False
    assert model.last_output_dtype == torch.float64
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


def test_high_level_train_reaches_fused_lm_head_loss():
    torch.manual_seed(522)
    model = nn.Linear(4, 4, bias=False)
    head = torch.randn(6, 4)
    with patch(
        "ojasx2.trainer.fused_linear_cross_entropy",
        wraps=trainer_module.fused_linear_cross_entropy,
    ) as fused:
        result = train(
            model,
            [(torch.randn(3, 4), torch.tensor([0, 2, 5]))],
            epochs=1,
            device="cpu",
            dtype=torch.float32,
            quantize_moments=False,
            lm_head_weight=head,
            log_interval=100,
        )
    assert fused.call_count == 1
    assert result["history"][0]["optimizer_steps"] == 1


def test_non_fused_head_explicitly_materializes_standard_logits():
    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.float32,
        quantize_moments=False, fused_loss=False,
    )
    head = torch.randn(6, 4)
    with patch("ojasx2.trainer.fused_linear_cross_entropy") as fused:
        trainer.train_epoch(_batch(), lm_head_weight=head, log_interval=100)
    fused.assert_not_called()


def test_trainer_wires_ojasx_verified_dynamo_backend_by_default():
    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.float32,
        quantize_moments=False, fused_loss=False,
    )
    assert trainer.compile_backend == "ojasx_verified"
    assert trainer._compiled_model is not None
    assert trainer.model is model

    metrics = trainer.train_epoch(_batch(), log_interval=100)
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 1
    assert metrics["avg_loss"] > 0.0
    assert model.last_output_dtype == torch.float32


def test_trainer_honors_explicit_compile_backend_disabling():
    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.float32,
        quantize_moments=False, fused_loss=False,
        compile_backend=None,
    )
    assert trainer.compile_backend is None
    assert trainer._compiled_model is None
    metrics = trainer.train_epoch(_batch(), log_interval=100)
    assert metrics["steps"] == 1
    assert model.last_output_dtype == torch.float32


def test_trainer_safe_fallback_on_dynamo_unsupported_construct():
    from unittest.mock import Mock

    model = RecordingLinear(4, 4)
    trainer = LowRAMTrainer(
        model, device="cpu", dtype=torch.float32,
        quantize_moments=False, fused_loss=False,
    )
    trainer._compiled_model = Mock(side_effect=RuntimeError("unsupported dynamic construct"))
    metrics = trainer.train_epoch(_batch(), log_interval=100)
    assert metrics["steps"] == 1
    assert model.last_output_dtype == torch.float32
    assert trainer._compiled_model.call_count >= 1


def test_high_level_train_defaults_to_ojasx_verified_backend():
    model = RecordingLinear(4, 4)
    result = train(
        model,
        _batch(),
        epochs=1,
        device="cpu",
        dtype=torch.float32,
        quantize_moments=False,
        fused_loss=False,
        log_interval=100,
    )
    assert result["trainer"].compile_backend == "ojasx_verified"
    assert result["trainer"]._compiled_model is not None
    assert len(result["history"]) == 1
    assert result["history"][0]["optimizer_steps"] == 1
