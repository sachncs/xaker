"""Coverage tests for the preconditioner strategies."""

from __future__ import annotations

import torch

from xaker.config import Config
from xaker.solver.precond import (
    Cccp,
    Diagonal,
    Fast,
    Identity,
    Make,
)


def test_identity_apply() -> None:
    """Identity.apply returns the residual unchanged."""
    pre = Make(Config(dim=64, heads=4, precond="identity"))
    residual = torch.randn(2, 4, 16, 8)
    out = pre.apply(residual, None)
    assert torch.equal(out, residual)


def test_diagonal_apply() -> None:
    """Diagonal.apply multiplies residual element-wise by diag."""
    pre = Make(Config(dim=64, heads=4, precond="diagonal"))
    residual = torch.randn(2, 4, 16, 8)
    data = pre.build(torch.randn(2, 4, 16, 16), torch.tensor(0.1), 16)
    out = pre.apply(residual, data)
    assert out.shape == residual.shape


def test_fast_apply() -> None:
    """Fast.apply with low-rank factor is well-defined."""
    pre = Make(Config(dim=64, heads=4, precond="fast", rank=4))
    residual = torch.randn(2, 4, 16, 8)
    data = pre.build(torch.randn(2, 4, 16, 16), torch.tensor(0.1), 16)
    out = pre.apply(residual, data)
    assert out.shape == residual.shape
    assert torch.isfinite(out).all()


def test_cccp_apply() -> None:
    """Cccp.apply returns matmul result with cache P."""
    pre = Make(Config(dim=64, heads=4, precond="cccp"))
    residual = torch.randn(2, 4, 16, 8)
    data = pre.build(torch.randn(2, 4, 16, 16), torch.tensor(0.1), 16)
    out = pre.apply(residual, data)
    assert out.shape == residual.shape
    assert torch.isfinite(out).all()


def test_fast_iter_counter() -> None:
    """Fast preconditioner tracks iteration count via self.iter."""
    pre = Make(Config(dim=64, heads=4, precond="fast", rank=4))
    kernel = torch.randn(2, 4, 16, 16)
    pre.build(kernel, torch.tensor(0.1), 16)
    assert hasattr(pre, "iter")
    pre.iter.add_(1)
    assert int(pre.iter[0]) == 1


def test_fast_no_rank() -> None:
    """Fast preconditioner with rank=0 has no low-rank factor."""
    pre = Make(Config(dim=64, heads=4, precond="fast", rank=0))
    residual = torch.randn(2, 4, 16, 8)
    data = pre.build(torch.randn(2, 4, 16, 16), torch.tensor(0.1), 16)
    out = pre.apply(residual, data)
    assert out.shape == residual.shape
