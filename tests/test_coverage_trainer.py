"""Coverage tests for the Trainer."""

from __future__ import annotations

import torch

from xaker.config import Config
from xaker.model.model import Model
from xaker.training.trainer import Fit, Trainer


def test_trainer_basic() -> None:
    """Trainer constructs with a model, Fit, and device."""
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=20, max_seq_len=8)
    trainer = Trainer(model, Fit(epochs=1, lr=1e-3), torch.device("cpu"))
    assert trainer.iter == 0


def test_trainer_step() -> None:
    """Trainer.step runs forward / loss / backward / step."""
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=20, max_seq_len=8)
    trainer = Trainer(model, Fit(epochs=1, lr=1e-3), torch.device("cpu"))
    batch = (torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8)))
    metrics = trainer.step(batch)
    assert "loss" in metrics
    assert trainer.iter == 1


def test_trainer_epoch() -> None:
    """Trainer.epoch runs a full pass over a dataloader."""
    from torch.utils.data import DataLoader, TensorDataset
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=20, max_seq_len=8)
    trainer = Trainer(model, Fit(epochs=1, lr=1e-3), torch.device("cpu"))
    data = TensorDataset(
        torch.randint(0, 20, (4, 8)),
        torch.randint(0, 20, (4, 8)),
    )
    loader = DataLoader(data, batch_size=2)
    metrics = trainer.epoch(loader)
    assert "epoch_loss" in metrics
    assert "elapsed" in metrics


def test_trainer_loss() -> None:
    """Trainer.loss calls ce with configured smoothing."""
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=20, max_seq_len=8)
    trainer = Trainer(model, Fit(smooth=0.2), torch.device("cpu"))
    logits = torch.randn(2, 8, 20)
    labels = torch.randint(0, 20, (2, 8))
    loss = trainer.loss(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_trainer_with_scheduler() -> None:
    """Trainer accepts an external optimizer and scheduler."""
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=20, max_seq_len=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    trainer = Trainer(model, Fit(), torch.device("cpu"), optimizer=opt, scheduler=sched)
    batch = (torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8)))
    trainer.step(batch)
