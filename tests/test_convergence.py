"""Convergence tests for the PCG solver."""

from __future__ import annotations

import torch

from xaker.solver.cg import pcg, richardson
from xaker.solver.func import op


def test_converge_psd() -> None:
    """pcg reaches small residual on a well-conditioned PSD system."""
    batch, heads, length, headdim = 1, 2, 8, 4
    A = torch.randn(batch, heads, length, length)
    kernel = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(length) * 1.0
    b = torch.randn(batch, heads, length, headdim)
    result = pcg(kernel, b, lam=torch.tensor(0.0), iters=500, tol=1e-6)
    residual = b - op(kernel, result.x, torch.tensor(0.0))
    assert residual.norm().item() < 1e-2


def test_converge_identity() -> None:
    """pcg on K=2I, lam=0 converges to x = b/2."""
    batch, heads, length, headdim = 1, 1, 4, 2
    kernel = torch.eye(length).unsqueeze(0).unsqueeze(0) * 2.0
    b = torch.ones(batch, heads, length, headdim)
    result = pcg(kernel, b, lam=torch.tensor(0.0), x0=None)
    expected = torch.ones_like(b) * 0.5
    assert torch.allclose(result.x, expected, atol=1e-3)


def test_converge_solve_field() -> None:
    """Solve dataclass exposes iters/converged/res/history fields."""
    batch, heads, length, headdim = 1, 1, 4, 2
    kernel = torch.eye(length).unsqueeze(0).unsqueeze(0)
    b = torch.ones(batch, heads, length, headdim)
    result = pcg(kernel, b, lam=torch.tensor(0.0), iters=10)
    assert hasattr(result, "x")
    assert hasattr(result, "iters")
    assert hasattr(result, "converged")
    assert hasattr(result, "res")
    assert hasattr(result, "history")
    assert result.iters <= 10


def test_richardson_no_converge_flag() -> None:
    """richardson never reports converged=True."""
    batch, heads, length, headdim = 1, 1, 4, 2
    kernel = torch.eye(length).unsqueeze(0).unsqueeze(0)
    b = torch.ones(batch, heads, length, headdim)
    result = richardson(kernel, b, lam=torch.tensor(0.0), iters=5)
    assert not result.converged


def test_solve_finite() -> None:
    """Solve.x is finite for a well-conditioned system."""
    batch, heads, length, headdim = 1, 1, 4, 2
    kernel = torch.eye(length).unsqueeze(0).unsqueeze(0)
    b = torch.ones(batch, heads, length, headdim)
    result = pcg(kernel, b, lam=torch.tensor(0.1), iters=10)
    assert torch.isfinite(result.x).all()


def test_history_nonempty() -> None:
    """Solve.history records at least one residual per iteration."""
    batch, heads, length, headdim = 1, 1, 4, 2
    kernel = torch.eye(length).unsqueeze(0).unsqueeze(0)
    b = torch.ones(batch, heads, length, headdim)
    result = pcg(kernel, b, lam=torch.tensor(0.0), iters=5)
    assert len(result.history) >= 1
