"""Tests for the solver stack.

Covers the PCG-style solver and the regularized kernel operator.
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.solver.cg import pcg, op


@pytest.fixture
def config() -> Config:
    """(dim=64, heads=4, headdim=16, rank=4) config."""
    return Config(dim=64, heads=4, headdim=16, rank=4)


class TestCg:
    """Tests for the PCG-style solver."""

    def test_converge(self) -> None:
        """pcg converges on a PSD system."""
        batch, heads, length, headdim = 1, 2, 16, 8
        A = torch.randn(batch, heads, length, length)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        b = torch.randn(batch, heads, length, headdim)
        result = pcg(kernel, b, lam=torch.tensor(0.1), iters=100, tol=1e-6)
        assert result.converged
        residual = b - op(kernel, result.x, torch.tensor(0.1))
        assert residual.norm().item() < 1.0

    def test_apply(self) -> None:
        """pcg with a preconditioner callback stays finite."""
        batch, heads, length, headdim = 1, 2, 16, 8
        A = torch.randn(batch, heads, length, length)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        b = torch.randn(batch, heads, length, headdim)

        def apply_precond(r, data):
            return r * data

        result = pcg(
            kernel, b, lam=torch.tensor(0.1),
            iters=100, tol=1e-6,
            precond_data=torch.tensor(0.1),
            apply_pre=apply_precond,
        )
        assert torch.isfinite(result.x).all()

    def test_zero(self) -> None:
        """pcg with x0=None on K=2I converges to x=0.5."""
        batch, heads, length, headdim = 1, 1, 8, 4
        kernel = torch.eye(length).unsqueeze(0).unsqueeze(0) * 2.0
        b = torch.ones(batch, heads, length, headdim)
        result = pcg(kernel, b, lam=torch.tensor(0.0), x0=None)
        expected = torch.ones_like(b) * 0.5
        assert torch.allclose(result.x, expected, atol=1e-4)