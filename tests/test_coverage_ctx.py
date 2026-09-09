"""Coverage tests for Ctx."""

from __future__ import annotations

import torch

from xaker.utils.ctx import Ctx, toctx


def test_default() -> None:
    """Ctx defaults: cpu, float32."""
    c = Ctx()
    assert c.device.type == "cpu"
    assert c.dtype == torch.float32


def test_explicit() -> None:
    """Ctx accepts explicit device and dtype."""
    c = Ctx(device=torch.device("cpu"), dtype=torch.float64)
    assert c.device.type == "cpu"
    assert c.dtype == torch.float64


def test_resolve_none() -> None:
    """Ctx.resolve(None) returns cuda if available else cpu."""
    c = Ctx()
    d = c.resolve(None)
    assert isinstance(d, torch.device)


def test_resolve_string() -> None:
    """Ctx.resolve('cpu') returns torch.device('cpu')."""
    c = Ctx()
    d = c.resolve("cpu")
    assert d.type == "cpu"


def test_toctx() -> None:
    """toctx moves a tensor to the Ctx's device and dtype."""
    c = Ctx(dtype=torch.float32)
    x = torch.randn(2, 3, dtype=torch.float64)
    y = toctx(x, c)
    assert y.dtype == torch.float32
