"""Tests for :class:`Fused` and its kernel/preconditioner stack.

Covers the fused-XSA modules in detail:
- :class:`Kernel` — learnable-temperature kernel used by :class:`Fused`.
- :class:`Fused` — fused XSA + kernel attention with PCG.
- :class:`Precond` factory — :func:`Make` returns the configured strategy
  (``Identity``, ``Diagonal``, ``Fast``, ``Cccp``).
- :func:`compute_kernel_matrix` from :mod:`xaker.attention.func`.

Verified invariants:
- :class:`Fused` is deterministic in ``eval()`` mode.
- ``lam`` is strictly positive (softplus-parameterised).
- :func:`zerodiag` returns a kernel whose main diagonal is zero.
- :class:`Make(config)` returns a concrete ``PrecondProto`` instance for each
  supported ``config.precond`` literal.
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.attention.fused import Fused
from xaker.attention.kernel import Kernel
from xaker.attention.func import kernel as compute_kernel_matrix
from xaker.attention.ops import zerodiag, rms
from xaker.attention.xsa import (
    XsaStrategy,
    Projection,
    Zero,
    Mask,
    XSA_MODE,
)
from xaker.solver.precond import (
    Make,
    Identity,
    Diagonal,
    Fast,
    Cccp,
    Cache,
    BOUND,
)


@pytest.fixture
def config() -> Config:
    """Small default config."""
    return Config(dim=64, heads=4, drop=0.0, eps=1e-6)


class TestKernel:
    """Tests for :class:`Kernel`."""

    def test_shape(self) -> None:
        k = Kernel(headdim=16)
        q = torch.randn(2, 4, 32, 16)
        v = torch.randn(2, 4, 32, 16)
        out = k(q, v)
        assert out.shape == (2, 4, 32, 32)

    def test_finite(self) -> None:
        k = Kernel(headdim=16)
        q = torch.randn(2, 4, 16, 16)
        v = torch.randn(2, 4, 16, 16)
        out = k(q, v)
        assert torch.isfinite(out).all()

    def test_temp_property(self) -> None:
        k = Kernel(headdim=16, temp=2.0)
        assert 0.05 <= float(k.temp.item()) <= 100.0

    def test_learnable(self) -> None:
        k = Kernel(headdim=16, learnable=True)
        assert isinstance(k.logtemp, torch.nn.Parameter)

    def test_fixed(self) -> None:
        k = Kernel(headdim=16, learnable=False)
        assert not isinstance(k.logtemp, torch.nn.Parameter)


class TestFused:
    """Tests for :class:`Fused`."""

    def test_shape(self, config: Config) -> None:
        attn = Fused(config)
        attn.eval()
        x = torch.randn(2, 32, config.dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_finite(self, config: Config) -> None:
        attn = Fused(config)
        attn.eval()
        x = torch.randn(2, 32, config.dim)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_grad(self, config: Config) -> None:
        attn = Fused(config)
        attn.train()
        x = torch.randn(2, 32, config.dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_lam_positive(self, config: Config) -> None:
        attn = Fused(config)
        assert attn.lam.item() > 0

    def test_xsa_scale_param(self, config: Config) -> None:
        """xsa_scale is always nn.Parameter regardless of mode."""
        for mode in ["subtract", "zero", "mask"]:
            c = Config(dim=64, heads=4, mode=mode)
            attn = Fused(c)
            assert isinstance(attn.xsa_scale, torch.nn.Parameter), (
                f"xsa_scale not nn.Parameter for mode={mode}"
            )

    def test_zerodiag(self) -> None:
        k = torch.randn(2, 4, 16, 16)
        result = zerodiag(k)
        diag = torch.diagonal(result, dim1=-2, dim2=-1)
        assert (diag.abs() < 1e-6).all()

    def test_rms(self) -> None:
        x = torch.randn(2, 4, 32, 16)
        normed = rms(x, 1e-6)
        assert normed.shape == x.shape
        assert torch.isfinite(normed).all()

    def test_deterministic(self, config: Config) -> None:
        attn = Fused(config)
        attn.eval()
        x = torch.randn(2, 32, config.dim)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_mask(self, config: Config) -> None:
        attn = Fused(config)
        attn.eval()
        x = torch.randn(2, 16, config.dim)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        out = attn(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestPrecond:
    """Tests for preconditioner factory and strategies."""

    def test_identity(self, config: Config) -> None:
        pre = Make(Config(dim=64, heads=4, precond="identity"))
        assert isinstance(pre, Identity)
        data = pre.build(torch.randn(2, 4, 16, 16), torch.tensor(0.1), 16)
        assert isinstance(data, Cache)

    def test_diagonal(self, config: Config) -> None:
        pre = Make(Config(dim=64, heads=4, precond="diagonal"))
        assert isinstance(pre, Diagonal)

    def test_fast(self, config: Config) -> None:
        pre = Make(Config(dim=64, heads=4, precond="fast", rank=4))
        assert isinstance(pre, Fast)

    def test_cccp(self, config: Config) -> None:
        pre = Make(Config(dim=64, heads=4, precond="cccp"))
        assert isinstance(pre, Cccp)

    def test_apply(self) -> None:
        pre = Make(Config(dim=64, heads=4, precond="identity"))
        residual = torch.randn(2, 4, 16, 8)
        out = pre.apply_pre(residual, None)
        assert torch.equal(out, residual)


class TestXsaStrategy:
    """Tests for :func:`XsaStrategy` polymorphism."""

    def test_subtract(self, config: Config) -> None:
        s = XsaStrategy(config, torch.ones(1))
        assert isinstance(s, Projection)

    def test_zero(self, config: Config) -> None:
        c = Config(dim=64, heads=4, mode="zero")
        s = XsaStrategy(c, torch.ones(1))
        assert isinstance(s, Zero)

    def test_mask(self, config: Config) -> None:
        c = Config(dim=64, heads=4, mode="mask")
        s = XsaStrategy(c, torch.ones(1))
        assert isinstance(s, Mask)

    def test_dispatch_table(self) -> None:
        assert set(XSA_MODE.keys()) == {"subtract", "zero", "mask"}


class TestFunctionalKernel:
    """Tests for stateless :func:`compute_kernel_matrix`."""

    def test_shape(self) -> None:
        q = torch.randn(2, 32, 16)
        v = torch.randn(2, 32, 16)
        k = compute_kernel_matrix(q, v, normalize=True)
        assert k.shape == (2, 32, 32)

    def test_finite(self) -> None:
        q = torch.randn(4, 32, 16)
        v = torch.randn(4, 32, 16)
        k = compute_kernel_matrix(q, v)
        assert torch.isfinite(k).all()

    def test_symmetric(self) -> None:
        q = torch.randn(2, 32, 16)
        v = torch.randn(2, 32, 16)
        k = compute_kernel_matrix(q, v, symmetric=True)
        assert torch.allclose(k, k.transpose(-2, -1))


class TestBound:
    """Tests for BOUND constant."""

    def test_bound(self) -> None:
        assert BOUND == 1e6