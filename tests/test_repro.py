"""Reproducibility tests across the public API."""

from __future__ import annotations

import torch

from xaker.utils.rng import seed, snapshot, restore
from xaker.attention import BLOCK
from xaker.config import Config


def test_seed_reproducible() -> None:
    """Setting the same seed twice produces the same RNG state."""
    seed(42)
    a = torch.randn(8)
    seed(42)
    b = torch.randn(8)
    assert torch.allclose(a, b)


def test_round_trip() -> None:
    """snapshot/restore preserves the RNG state."""
    seed(42)
    torch.randn(5)
    state = snapshot()
    a = torch.randn(5)
    restore(state)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_attention_reproducible() -> None:
    """Standard attention with same seed produces same output."""
    seed(0)
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["standard"](cfg).eval()
    x = torch.randn(2, 16, cfg.dim)
    with torch.no_grad():
        a = attn(x)
    seed(0)
    attn2 = BLOCK["standard"](cfg).eval()
    with torch.no_grad():
        b = attn2(x)
    assert torch.allclose(a, b)
