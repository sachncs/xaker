"""Tests for the polymorphism dispatch tables.

Verifies:
- :func:`Make` returns the right strategy for each ``config.precond`` literal.
- :func:`XsaStrategy` returns the right strategy for each ``config.mode`` literal.
- ``BLOCK[kind]`` returns the right attention class for each kind literal.
- :class:`xsa_scale` is always an ``nn.Parameter`` regardless of mode.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xaker.attention import BLOCK, Kernel, Laker, Standard, Xsa
from xaker.config import Config
from xaker.solver.precond import (
    Cccp,
    Diagonal,
    Fast,
    Identity,
    Make,
)
from xaker.attention.xsa import Mask, Projection, XsaStrategy, Zero


class TestPrecond:
    """Make(config) returns the correct strategy."""

    @pytest.mark.parametrize("mode", ["identity", "diagonal", "fast", "cccp"])
    def test_kind(self, mode: str) -> None:
        cfg = Config(dim=64, heads=4, precond=mode, rank=4)
        pre = Make(cfg)
        cls = {"identity": Identity, "diagonal": Diagonal, "fast": Fast, "cccp": Cccp}[mode]
        assert isinstance(pre, cls)


class TestBlock:
    """BLOCK registry returns the correct attention class."""

    @pytest.mark.parametrize(
        "kind,cls",
        [("standard", Standard), ("xsa", Xsa), ("fused", Laker)],
    )
    def test_kind(self, kind: str, cls) -> None:
        cfg = Config(dim=64, heads=4)
        attn = BLOCK[kind](cfg)
        assert isinstance(attn, cls)


class TestXsaStrategy:
    """XsaStrategy(config, scale) returns the correct strategy."""

    @pytest.mark.parametrize(
        "mode,cls",
        [("subtract", Projection), ("zero", Zero), ("mask", Mask)],
    )
    def test_kind(self, mode: str, cls) -> None:
        cfg = Config(dim=64, heads=4, mode=mode)
        scale = torch.ones(1)
        strat = XsaStrategy(cfg, scale)
        assert isinstance(strat, cls)


class TestXsaScaleAlwaysParam:
    """xsa_scale is always a trainable parameter regardless of mode."""

    @pytest.mark.parametrize("mode", ["subtract", "zero", "mask"])
    def test_scale(self, mode: str) -> None:
        cfg = Config(dim=64, heads=4, mode=mode)
        attn = Xsa(cfg)
        assert isinstance(attn.xsa_scale, nn.Parameter)


class TestKernel:
    """Kernel is constructable and callable."""

    def test_kernel(self) -> None:
        k = Kernel(headdim=16)
        q = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = k(q, v)
        assert out.shape == (2, 4, 8, 8)


class TestGuard:
    """Guards for config-side polymorphism invariants."""

    def test_use_fused_gone(self) -> None:
        cfg = Config(dim=64, heads=4)
        assert not hasattr(cfg, "use_fused")

    def test_pcg_gone(self) -> None:
        cfg = Config(dim=64, heads=4)
        assert not hasattr(cfg, "effective_pcg_iters")

    def test_clip_gone(self) -> None:
        cfg = Config(dim=64, heads=4)
        assert not hasattr(cfg, "clip_abs")