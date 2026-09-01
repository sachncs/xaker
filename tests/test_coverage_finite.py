"""Coverage tests for the finite utility."""

from __future__ import annotations

import math

import torch

from xaker.utils.finite import finite


def test_finite_all_finite() -> None:
    """finite returns True for all-finite input."""
    x = torch.randn(4, 4)
    assert finite(x, "ok") is True


def test_finite_nan() -> None:
    """finite returns False for NaN input."""
    x = torch.randn(4, 4)
    x[0, 0] = float("nan")
    assert finite(x, "nan input", raise_error=False) is False


def test_finite_inf() -> None:
    """finite returns False for +Inf input."""
    x = torch.randn(4, 4)
    x[0, 0] = float("inf")
    assert finite(x, "inf input", raise_error=False) is False


def test_finite_neg_inf() -> None:
    """finite returns False for -Inf input."""
    x = torch.randn(4, 4)
    x[0, 0] = float("-inf")
    assert finite(x, "neginf input", raise_error=False) is False


def test_finite_raises() -> None:
    """finite raises ValueError when raise_error=True."""
    x = torch.randn(4)
    x[0] = float("nan")
    try:
        finite(x, "nan", raise_error=True)
    except ValueError as e:
        assert "nan" in str(e).lower() or "non-finite" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
