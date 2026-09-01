"""Tests for the attention module surface.

Covers :class:`Standard`, :class:`Xsa`, and the polymorphic dispatch.
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.attention.standard import Standard
from xaker.attention.xsa import Xsa


@pytest.fixture
def config() -> Config:
    """Standard (dim=64, heads=4, headdim=16) config."""
    return Config(
        dim=64,
        heads=4,
        headdim=16,
        drop=0.0,
        eps=1e-6,
        rank=4,
    )


@pytest.fixture
def sample_input(config: Config) -> torch.Tensor:
    """Random (2, 32, dim) Gaussian input."""
    return torch.randn(2, 32, config.dim)


class TestStd:
    """Tests for Standard."""

    def test_shape(self, config: Config, sample_input: torch.Tensor) -> None:
        attn = Standard(config)
        out = attn(sample_input)
        assert out.shape == sample_input.shape

    def test_finite(self, config: Config, sample_input: torch.Tensor) -> None:
        attn = Standard(config)
        out = attn(sample_input)
        assert torch.isfinite(out).all()

    def test_grad(self, config: Config, sample_input: torch.Tensor) -> None:
        attn = Standard(config)
        x = sample_input.clone().requires_grad_(True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_mask(self, config: Config) -> None:
        attn = Standard(config)
        batch, length = 2, 32
        x = torch.randn(batch, length, config.dim)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        mask = ~mask
        out = attn(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape


class TestXsa:
    """Tests for Xsa."""

    def test_shape(self, config: Config, sample_input: torch.Tensor) -> None:
        attn = Xsa(config)
        out = attn(sample_input)
        assert out.shape == sample_input.shape

    def test_finite(self, config: Config, sample_input: torch.Tensor) -> None:
        attn = Xsa(config)
        out = attn(sample_input)
        assert torch.isfinite(out).all()

    def test_xsa_exclusion(self) -> None:
        """Xsa subtract mode reduces correlation with each token's value."""
        cfg = Config(dim=64, heads=4, headdim=16, drop=0.0, mode="subtract")
        attn = Xsa(cfg)
        attn.eval()
        batch, length = 2, 16
        x = torch.randn(batch, length, cfg.dim)
        with torch.no_grad():
            v = attn.qkv_proj.w_v(x)
            v = v.view(batch, length, cfg.heads, cfg.headdim).transpose(1, 2)
            out = attn(x)
            out = out.view(batch, length, cfg.heads, cfg.headdim).transpose(1, 2)
            for i in range(length):
                oi = out[:, :, i, :]
                vi = v[:, :, i, :]
                cos = torch.nn.functional.cosine_similarity(oi, vi, dim=-1)
                assert cos.abs().mean().item() < 0.5

    def test_zero(self, config: Config, sample_input: torch.Tensor) -> None:
        c = Config(dim=64, heads=4, headdim=16, drop=0.0, mode="zero")
        attn = Xsa(c)
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert torch.isfinite(out).all()

    def test_maskmode(self) -> None:
        c = Config(dim=64, heads=4, headdim=16, drop=0.0, mode="mask")
        attn = Xsa(c)
        x = torch.randn(2, 32, c.dim)
        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()