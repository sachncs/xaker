"""Tests for the functional (stateless) API of XAKER.

Covers 2-D, 3-D, and 4-D kernel inputs, default finite/positive results,
normalization, symmetry, temp, unequal query/key lengths, and the
regularized kernel operator's shape and simple identity cases.
"""

from __future__ import annotations

import pytest
import torch

from xaker.attention.func import kernel
from xaker.solver.func import op

class TestComputeKernelMatrix:
    """Tests for kernel functional API."""

    def test_shape_2d(self) -> None:
        q = torch.randn(16, 8)
        k = torch.randn(16, 8)
        K = kernel(q, k)
        assert K.shape == (16, 16)

    def test_shape_3d(self) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = kernel(q, k)
        assert K.shape == (2, 16, 16)

    def test_shape_4d(self) -> None:
        q = torch.randn(2, 4, 16, 8)
        k = torch.randn(2, 4, 16, 8)
        K = kernel(q, k)
        assert K.shape == (2, 4, 16, 16)

    def test_finite_values(self) -> None:
        q = torch.randn(4, 32, 16)
        k = torch.randn(4, 32, 16)
        K = kernel(q, k)
        assert torch.isfinite(K).all()

    def test_positive_values(self) -> None:
        q = torch.randn(4, 32, 16)
        k = torch.randn(4, 32, 16)
        K = kernel(q, k)
        # exp + eps is always positive
        assert (K > 0).all()

    @pytest.mark.parametrize("normalize_qk", [True, False])
    def test_normalize_modes(self, normalize_qk: bool) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = kernel(q, k, normalize_qk=normalize_qk)
        assert K.shape == (2, 16, 16)
        assert torch.isfinite(K).all()

    @pytest.mark.parametrize("symmetric", [True, False])
    def test_symmetric_modes(self, symmetric: bool) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = kernel(q, k, symmetric=symmetric)
        if symmetric:
            assert torch.allclose(K, K.transpose(-2, -1))

    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 5.0, 20.0])
    def test_temperature_values(self, temp: float) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = kernel(q, k, temp=temp)
        assert torch.isfinite(K).all()

    def test_shape_qk(self) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 32, 8)
        K = kernel(q, k)
        assert K.shape == (2, 16, 32)

class TestApplyKernelOperator:
    """Tests for op functional API."""

    def test_shape_op(self) -> None:
        kernel = torch.randn(2, 4, 32, 32)
        x = torch.randn(2, 4, 32, 16)
        lam = torch.tensor(0.1)
        result = op(kernel, x, lam)
        assert result.shape == x.shape

    def test_finite_output(self) -> None:
        kernel = torch.randn(2, 4, 32, 32)
        x = torch.randn(2, 4, 32, 16)
        lam = torch.tensor(0.1)
        result = op(kernel, x, lam)
        assert torch.isfinite(result).all()

    def test_zero_lam(self) -> None:
        n = 8
        kernel = torch.eye(n).unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
        x = torch.randn(1, 1, n, 4)
        lam = torch.tensor(0.0)
        result = op(kernel, x, lam)
        # With identity kernel and lambda=0, result should equal x
        assert torch.allclose(result, x, atol=1e-5)

    def test_lam_effect(self) -> None:
        kernel = torch.eye(8).unsqueeze(0).unsqueeze(0)
        x = torch.randn(1, 1, 8, 4)
        r0 = op(kernel, x, torch.tensor(0.0))
        r2 = op(kernel, x, torch.tensor(2.0))
        # Adding lambda*I should produce different results
        assert (r0 != r2).any()

    def test_broadcast(self) -> None:
        kernel = torch.randn(2, 4, 16, 16)
        x = torch.randn(2, 4, 16, 8)
        lam = torch.tensor(1.0).view(1, 1, 1, 1)
        result = op(kernel, x, lam)
        assert result.shape == x.shape

    def test_zero(self) -> None:
        kernel = torch.zeros(2, 4, 16, 16)
        x = torch.randn(2, 4, 16, 8)
        lam = torch.tensor(1.0)
        result = op(kernel, x, lam)
        assert torch.allclose(result, x, atol=1e-5)  # K=0, lam=1 → result = x
