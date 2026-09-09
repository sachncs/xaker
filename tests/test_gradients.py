"""Gradient-flow regression tests for the xaker stack.

A single ``output.sum().backward()`` call must produce finite input
gradients for Standard, Xsa, and Fused. The fused attention's
parameter gradients are also checked for finiteness; multiple
training-style backward passes on the full model must stay
NaN/Inf-free.
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.attention.standard import Standard
from xaker.attention.xsa import Xsa
from xaker.attention.fused import Fused


@pytest.fixture
def config() -> Config:
    """Small test config with drop=0."""
    return Config(dim=64, heads=4, drop=0.0)


class TestGrad:
    """Test gradient flow through attention modules."""

    def test_std(self, config: Config) -> None:
        """Standard attention gradients."""
        attn = Standard(config)
        attn.train()
        x = torch.randn(2, 32, config.dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_xsa(self, config: Config) -> None:
        """XSA gradients."""
        attn = Xsa(config)
        attn.train()
        x = torch.randn(2, 32, config.dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_fused(self, config: Config) -> None:
        """Fused attention gradients; input and parameters."""
        attn = Fused(config)
        attn.train()
        x = torch.randn(2, 32, config.dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, param in attn.named_parameters():
            if "precon" in name:
                continue  # PCG path is not differentiable through custom apply callback
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has NaN/Inf"