"""Universal Training Recipe Verification Script for OjasX v0.5.1.

Validates the exact user-specified recipe across CPU and MPS:
1. optimize_model with compile_mlir=True and mlir_target="auto".
2. HolonomicGradFreeOptimizer with AdamW and post-accumulate gradient hooks.
3. Multi-iteration training loop verifying loss descent, zero NaN, and parameter updates.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import ojasx2
from ojasx2.grad_free_optimizer import HolonomicGradFreeOptimizer


class RecipeTransformer(nn.Module):
    """Clean reference transformer block with LM head to validate recipe."""
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits


def test_recipe_on_device(device_str: str):
    print(f"\n=======================================================")
    print(f" Testing Universal Training Recipe on [{device_str.upper()}]")
    print(f"=======================================================")
    device = torch.device(device_str)
    torch.manual_seed(42)

    # 1. Pick available accelerator
    model = RecipeTransformer().to(device)
    initial_param = model.lm_head.weight.clone()

    # 2. Optimize model layers AND lower through the 3-dialect MLIR pipeline
    model, report = ojasx2.optimize_model(
        model,
        batch_tokens=128,
        prefer_unified_mlp=True,
        compile_mlir=True,
        mlir_target="auto",
    )

    table = report.format_table()
    assert table is not None and len(table) > 0, "Optimization report table is empty!"
    print(table)
    print("MLIR Pipeline:", report.mlir_pipeline)
    assert report.mlir_pipeline is not None, "MLIR pipeline report missing!"
    assert report.mlir_pipeline.get("ssa_verified", False), "SSA verification failed!"
    assert report.mlir_pipeline.get("holonomic_op_count", 0) > 0, "No holonomic ops lowered!"
    assert report.mlir_pipeline.get("tile_op_count", 0) > 0, "No tile ops lowered!"

    # 3. Attach zero-persistent-grad optimizer
    optimizer = HolonomicGradFreeOptimizer(
        model.parameters(), lr=1e-3, algorithm="adamw", weight_decay=0.01
    )

    # 4. Train with reduced activation + gradient memory
    # Generate deterministic batch
    batch_size = 4
    seq_len = 32
    inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    losses = []
    print("\nStarting 10 training iterations:")
    for step in range(10):
        optimizer.zero_grad()
        loss = model(inputs, targets) if hasattr(model, "lm_head") else F.cross_entropy(model(inputs), targets)
        loss_val = float(loss.item())
        assert not torch.isnan(loss), f"Encountered NaN loss at step {step}!"
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step:2d} | Loss: {loss_val:.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert not torch.equal(model.lm_head.weight, initial_param), "Parameters did not update!"
    print(f"\n[PASS] Device {device_str}: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}. Gradients & optimizer verified.")


if __name__ == "__main__":
    test_recipe_on_device("cpu")
    if torch.backends.mps.is_available():
        test_recipe_on_device("mps")
    print("\n>>> ALL UNIVERSAL RECIPE TESTS PASSED SUCCESSFULLY! <<<\n")
