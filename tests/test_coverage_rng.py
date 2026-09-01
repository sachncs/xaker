"""Coverage tests for the rng module."""

from __future__ import annotations

import torch

from xaker.utils.rng import restore, seed, snapshot


def test_seed_basic() -> None:
    """seed(N) does not raise."""
    seed(0)


def test_seed_twice() -> None:
    """seed(N) called twice produces the same random sequence."""
    seed(0)
    a = torch.randn(5)
    seed(0)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_snapshot_keys() -> None:
    """snapshot() returns a dict with required keys."""
    state = snapshot()
    assert "python" in state
    assert "numpy" in state
    assert "torch" in state


def test_restore_missing_python() -> None:
    """restore() raises KeyError if python key missing."""
    try:
        restore({"numpy": None, "torch": None})
    except KeyError as e:
        assert "python" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_restore_missing_numpy() -> None:
    """restore() raises KeyError if numpy key missing."""
    try:
        restore({"python": None, "torch": None})
    except KeyError as e:
        assert "numpy" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_restore_missing_torch() -> None:
    """restore() raises KeyError if torch key missing."""
    try:
        restore({"python": None, "numpy": None})
    except KeyError as e:
        assert "torch" in str(e)
    else:
        raise AssertionError("expected KeyError")
