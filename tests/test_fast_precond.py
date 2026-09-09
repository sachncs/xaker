"""Tests for the Fast preconditioner cache and per-block step counter.

Verifies the two guarantees that issue #37 and #45 demanded:

* The cache payload is invalidated whenever a learnable factor is
  mutated between two ``build`` calls (``#37``).
* The ``Fast.iter`` step counter is *per* ``Fast`` instance:
  two ``Fused`` blocks construct two ``Fast`` strategies, so
  each block's counter advances independently (``#45``).
"""

from __future__ import annotations

import torch

from xaker.attention import Fused, Xsa
from xaker.config import Config
from xaker.model.block import Block
from xaker.solver.precond import Fast, Identity, Make


def _kernel(b: int = 1, h: int = 2, n: int = 16) -> torch.Tensor:
    """A small positive semi-definite kernel."""
    a = torch.randn(b, h, n, n)
    return torch.matmul(a, a.transpose(-2, -1))


class TestFastCacheInvalidation:
    """``Fast.build`` invalidates the cache when learnable factors change."""

    def test_cache_hit_when_unchanged(self) -> None:
        """A second ``build`` between two ``step % freq`` hits returns the cached object."""
        cfg = Config(dim=64, heads=2, rank=4, freq=2, precond="fast")
        fast = Make(cfg)
        assert isinstance(fast, Fast)
        k = _kernel()
        lam = torch.tensor(0.1)
        # First build to populate the cache.
        fast.build(k, lam, k.shape[-1])
        # Bump step once: with freq=2, step=1 -> step % freq != 0, so the
        # step-counter branch returns the cached payload on the next call.
        fast.iter.add_(1)
        a = fast.build(k, lam, k.shape[-1])
        b = fast.build(k, lam, k.shape[-1])
        assert a is b

    def test_cache_invalidates_on_param_change(self) -> None:
        """Mutating ``lr_base`` between two ``build`` calls rebuilds the payload."""
        cfg = Config(dim=64, heads=2, rank=4, freq=2, precond="fast")
        fast = Make(cfg)
        assert isinstance(fast, Fast)
        k = _kernel()
        lam = torch.tensor(0.1)
        a = fast.build(k, lam, k.shape[-1])

        # Mutation that re-allocates storage (typical of an optimizer step).
        with torch.no_grad():
            fast.lr_base.data = torch.zeros_like(fast.lr_base.data) * 0.01

        b = fast.build(k, lam, k.shape[-1])
        # Either the identity differs (rebuilt) or, in the rare case of a
        # storage-stable mutation, the underlying data has changed.
        assert a is not b or not torch.equal(a.data[1], b.data[1])

    def test_reset_cache_clears(self) -> None:
        """``Fast.reset_cache`` clears the payload and step counter."""
        cfg = Config(dim=64, heads=2, rank=4, freq=2, precond="fast")
        fast = Make(cfg)
        assert isinstance(fast, Fast)
        k = _kernel()
        fast.build(k, torch.tensor(0.1), k.shape[-1])
        fast.iter.add_(1)
        assert fast.cache is not None
        fast.reset_cache()
        assert fast.cache is None
        assert int(fast.iter[0]) == 0


class TestFastCounterPerBlock:
    """Each ``Fused`` block owns its own ``Fast`` step counter."""

    def test_two_fused_blocks_have_independent_counters(self) -> None:
        cfg = Config(dim=64, heads=2, rank=4, freq=1, precond="fast")
        # Manual model with two stacked Fused blocks.
        layer1 = Fused(cfg)
        layer2 = Fused(cfg)
        assert isinstance(layer1.precon, Fast)
        assert isinstance(layer2.precon, Fast)
        assert layer1.precon is not layer2.precon

        x = torch.randn(2, 16, cfg.dim)
        layer1(x)
        layer2(x)

        # Each block's counter has advanced exactly once.
        assert int(layer1.precon.iter[0]) == 1
        assert int(layer2.precon.iter[0]) == 1


class TestIdentityBudget:
    """Sanity: a precond='identity' block does not allocate a Fast instance."""

    def test_identity_returns_identity(self) -> None:
        cfg = Config(dim=64, heads=2, rank=4, precond="identity")
        pre = Make(cfg)
        assert isinstance(pre, Identity)
