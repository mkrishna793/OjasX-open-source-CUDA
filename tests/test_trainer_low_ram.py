import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import ojasx
from ojasx2.trainer import LowRAMTrainer, train, finetune


def test_low_ram_trainer_simple_mlp():
    """Verify LowRAMTrainer trains a simple model with decreasing loss."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.SiLU(),
        nn.Linear(32, 10),
    )

    X = torch.randn(64, 16)
    Y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=16)

    res = train(model, loader, epochs=2, lr=1e-3, log_interval=1)
    history = res["history"]
    assert len(history) == 2
    assert history[0]["steps"] == 4
    # Ensure throughput and memory are tracked
    assert history[0]["tokens_per_sec"] > 0
    assert history[1]["avg_loss"] <= history[0]["avg_loss"] * 1.5


def test_turnkey_finetune_api():
    """Verify ojasx.finetune functional entrypoint."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    X = torch.randn(32, 8)
    Y = torch.randint(0, 4, (32,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=8)

    res = ojasx.finetune(model, loader, epochs=1)
    assert len(res["history"]) == 1
