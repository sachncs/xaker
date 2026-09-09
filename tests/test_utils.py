"""Tests for utility helpers (masks, shape check, seeding).

Covers :func:`causal` (lower-triangular ``(1, seq_len, seq_len)``),
:func:`padding` (``(batch, 1, 1, seq_len)`` with valid tokens; ``2D``
enforcement raises :class:`ValueError`), :func:`shape` (``(None, ...)``
wildcards, :class:`ValueError` on mismatch), and :func:`seed` /
:func:`snapshot` / :func:`restore` round-trip.
"""

from __future__ import annotations

import pytest
import torch

from xaker.utils.ops import (
    causal,
    padding,
    shape,
)
from xaker.utils.rng import seed, snapshot, restore


class TestCausal:
    """Tests for :func:`causal`."""

    @pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
    def test_shape(self, seq_len: int) -> None:
        mask = causal(seq_len)
        assert mask.shape == (1, seq_len, seq_len)

    def test_lower(self) -> None:
        mask = causal(4)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        assert torch.equal(mask[0], expected)

    def test_device(self) -> None:
        if torch.cuda.is_available():
            mask = causal(4, device=torch.device("cuda"))
            assert mask.device.type == "cuda"


class TestPadding:
    """Tests for :func:`padding`."""

    def test_shape(self) -> None:
        pm = torch.tensor([[True, False, False, True]])
        mask = padding(pm)
        assert mask.shape == (1, 1, 1, 4)

    def test_padding_false(self) -> None:
        pm = torch.tensor([[True, False, False, True]])
        mask = padding(pm)
        assert not mask[0, 0, 0, 0].item()
        assert mask[0, 0, 0, 1].item()

    def test_no_padding(self) -> None:
        pm = torch.tensor([[False, False, False, False]])
        mask = padding(pm)
        assert mask.all()

    def test_all_padding(self) -> None:
        pm = torch.tensor([[True, True, True, True]])
        mask = padding(pm)
        assert not mask.any()

    def test_batch(self) -> None:
        pm = torch.tensor([[True, False, False], [False, True, False]])
        mask = padding(pm)
        assert mask.shape == (2, 1, 1, 3)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            padding(torch.randn(1, 1, 4).bool())


class TestShape:
    """Tests for :func:`shape`."""

    def test_match(self) -> None:
        x = torch.randn(2, 128, 512)
        assert shape(x, (2, 128, 512)) is True

    def test_wildcards(self) -> None:
        x = torch.randn(2, 128, 512)
        assert shape(x, (None, 128, 512)) is True
        assert shape(x, (None, None, None)) is True
        assert shape(x, (2, None, 512)) is True

    def test_mismatch_count(self) -> None:
        x = torch.randn(2, 128, 512)
        with pytest.raises(ValueError, match="dimensions"):
            shape(x, (2, 128))

    def test_mismatch_value(self) -> None:
        x = torch.randn(2, 128, 512)
        with pytest.raises(ValueError, match="dimension"):
            shape(x, (2, 64, 512))

    def test_none_ok(self) -> None:
        x = torch.randn(5, 100, 200)
        assert shape(x, (None, None, 200)) is True


class TestRng:
    """Tests for :func:`seed`, :func:`snapshot`, :func:`restore`."""

    def test_seed(self) -> None:
        """seed does not raise."""
        seed(42)

    def test_round_trip(self) -> None:
        """Round-trip of rng state snapshot/restore."""
        seed(42)
        torch.randn(5)
        state = snapshot()
        a2 = torch.randn(5)
        restore(state)
        a2_again = torch.randn(5)
        assert torch.allclose(a2, a2_again)

    def test_snapshot_keys(self) -> None:
        state = snapshot()
        assert "python" in state
        assert "numpy" in state
        assert "torch" in state

    def test_restore_missing(self) -> None:
        with pytest.raises(KeyError):
            restore({"numpy": None, "torch": None})

    def test_reproducible(self) -> None:
        seed(123)
        x1 = torch.randn(10)
        seed(123)
        x2 = torch.randn(10)
        assert torch.allclose(x1, x2)