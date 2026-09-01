"""Property-based tests for the xaker attention stack.

Verifies shape preservation, finiteness, and determinism under random inputs
across Standard, Xsa, and Fused.
"""

from __future__ import annotations

import torch

from xaker.attention import BLOCK
from xaker.config import Config


def test_shape_std() -> None:
    """Standard preserves (batch, length, dim)."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    mod = BLOCK["standard"](cfg).eval()
    for _ in range(5):
        b, l = torch.randint(1, 4, (1,)).item(), torch.randint(4, 32, (1,)).item()
        x = torch.randn(b, l, cfg.dim)
        assert mod(x).shape == x.shape


def test_shape_xsa() -> None:
    """Xsa preserves (batch, length, dim)."""
    cfg = Config(dim=64, heads=4, drop=0.0, mode="subtract")
    mod = BLOCK["xsa"](cfg).eval()
    for _ in range(5):
        b, l = torch.randint(1, 4, (1,)).item(), torch.randint(4, 32, (1,)).item()
        x = torch.randn(b, l, cfg.dim)
        assert mod(x).shape == x.shape


def test_shape_fused() -> None:
    """Fused preserves (batch, length, dim)."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    mod = BLOCK["fused"](cfg).eval()
    for _ in range(5):
        b, l = torch.randint(1, 4, (1,)).item(), torch.randint(4, 32, (1,)).item()
        x = torch.randn(b, l, cfg.dim)
        assert mod(x).shape == x.shape


def test_finite_inputs() -> None:
    """Standard produces finite outputs across random Gaussian inputs."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    mod = BLOCK["standard"](cfg).eval()
    for _ in range(10):
        x = torch.randn(2, 16, cfg.dim)
        assert torch.isfinite(mod(x)).all()


def test_deterministic() -> None:
    """Fused is deterministic in eval() mode."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    mod = BLOCK["fused"](cfg).eval()
    x = torch.randn(2, 16, cfg.dim)
    with torch.no_grad():
        out1 = mod(x)
        out2 = mod(x)
    assert torch.allclose(out1, out2)


def test_kernel_module() -> None:
    """Kernel module returns finite values for random Q/K."""
    from xaker.attention import Kernel
    cfg = Config(dim=64, heads=4, drop=0.0)
    k = Kernel(headdim=cfg.dim // cfg.heads)
    q = torch.randn(2, 4, 16, cfg.dim // cfg.heads)
    v = torch.randn(2, 4, 16, cfg.dim // cfg.heads)
    out = k(q, v)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 4, 16, 16)