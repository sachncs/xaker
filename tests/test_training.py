"""Tests for the training utilities."""

from __future__ import annotations

import torch

from xaker.config import Config
from xaker.model.model import Model
from xaker.training.loss import ce as ce_loss
from xaker.training.trainer import Fit, Trainer


def test_fit_defaults() -> None:
    """Fit has documented defaults."""
    f = Fit()
    assert f.epochs == 10
    assert f.lr == 1e-3
    assert f.smooth == 0.1


def test_ce_returns_scalar() -> None:
    """ce(logits, labels) returns a 0-d tensor."""
    logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = ce_loss(logits, labels, smoothing=0.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ce_with_smoothing() -> None:
    """ce(..., smooth=s) returns finite loss."""
    logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = ce_loss(logits, labels, smoothing=0.1)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_ce_3d_input() -> None:
    """ce accepts 3D logits (batch, length, vocab)."""
    logits = torch.randn(2, 8, 50)
    labels = torch.randint(0, 50, (2, 8))
    loss = ce_loss(logits, labels, smoothing=0.0)
    assert loss.ndim == 0


def test_ce_ignore_index() -> None:
    """ce(..., ignore_index=-1) skips positions labelled -1."""
    logits = torch.randn(4, 10)
    labels = torch.tensor([1, 2, -1, 3])
    loss = ce_loss(logits, labels, smoothing=0.0, ignore_index=-1)
    assert torch.isfinite(loss)


def test_trainer_step_count() -> None:
    """Trainer.iter increments on each step."""
    cfg = Config(dim=32, heads=2, drop=0.0)
    model = Model(cfg, num_layers=1, vocab_size=50, max_seq_len=16)
    trainer = Trainer(model, Fit(), torch.device("cpu"))
    initial = trainer.iter
    batch = (torch.randint(0, 50, (2, 16)), torch.randint(0, 50, (2, 16)))
    trainer.step(batch)
    assert trainer.iter == initial + 1
