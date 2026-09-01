"""Numerical stability tests for the xaker attention stack.

Verifies finiteness under extreme input magnitudes and long sequences.
"""

from __future__ import annotations

import torch

from xaker.attention import BLOCK
from xaker.config import Config
from xaker.utils.finite import finite


def test_std_large() -> None:
    """Standard attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["standard"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "std large", raise_error=False)


def test_std_small() -> None:
    """Standard attention produces finite output for x * 1e-6."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["standard"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 1e-6
    out = attn(x)
    assert finite(out, "std small", raise_error=False)


def test_fused_large() -> None:
    """Fused attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["fused"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "fused large", raise_error=False)


def test_xsa_large() -> None:
    """Xsa attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0, mode="subtract")
    attn = BLOCK["xsa"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "xsa large", raise_error=False)


def test_fused_long() -> None:
    """Fused is stable on a long sequence (length=64)."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["fused"](cfg).eval()
    x = torch.randn(1, 64, cfg.dim)
    out = attn(x)
    assert finite(out, "long sequence", raise_error=False)