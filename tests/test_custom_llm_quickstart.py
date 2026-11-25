"""Tests for OjasX3 v0.17.0 Turnkey Custom LLM Trainer & CLI Quickstart."""

from __future__ import annotations

import json
from pathlib import Path
import torch

import ojasx3
from ojasx3.cli import main as cli_main
from ojasx3.trainer.custom_llm import (
    CUSTOM_LLM_PRESETS,
    CustomLLMConfig,
    build_custom_llm,
    load_text_or_jsonl_batches,
    train_custom_llm,
)


def test_custom_llm_train_dual_rail_e8_and_bf16_modes(tmp_path: Path):
    corpus_path = tmp_path / "train.jsonl"
    rows = [
        {"text": "OjasX trains custom LLMs with 2.125-bit Dual-Rail and FusedLinearCrossEntropy."},
        {"prompt": "How does OjasX save memory?", "completion": "By compressing linear weights 7.53x and freeing gradients in-backward."},
    ]
    corpus_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = CustomLLMConfig(vocab_size=512, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, n_layers=2)

    for mode in ("dual_rail", "e8", "bf16"):
        save_dir = tmp_path / f"ckpt_{mode}"
        res = ojasx3.train_custom_llm(
            corpus_path,
            preset=cfg,
            mode=mode,
            steps=6,
            seq_len=32,
            batch_size=2,
            lr=5e-3,
            lora_r=4,
            group_size=32,
            device="cpu",
            save_dir=save_dir,
        )
        assert res.loss_end < res.loss_start, f"Expected loss to decrease in mode={mode}, got {res.loss_start} -> {res.loss_end}"
        assert res.tokens_per_sec > 0.0
        assert (save_dir / "ojasx_train_summary.json").exists()
        assert (save_dir / "config.json").exists()

        # Verify greedy generation works on the trained custom LLM
        prompt = torch.randint(4, 100, (1, 8), dtype=torch.long)
        gen = res.model.generate_greedy(prompt, max_new_tokens=4)
        assert gen.shape == (1, 12)


def test_cli_train_llm_subcommand(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Custom LLM CLI smoke test corpus for OjasX3 v0.17.0.\n" * 20, encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    rc = cli_main(
        [
            "train-llm",
            "--preset",
            "nano-15m",
            "--data",
            str(txt_file),
            "--mode",
            "dual_rail",
            "--steps",
            "2",
            "--seq-len",
            "32",
            "--device",
            "cpu",
            "--save-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "ojasx_train_summary.json").exists()
