"""Tests for the paper-worthiness rubric."""

from __future__ import annotations

import pytest

from xaker.rubric import Dimension, Score, grade


def test_grade_runs() -> None:
    """grade('.') returns a Rubric with all six dimensions."""
    r = grade(".")
    assert set(r.dims.keys()) == {
        "novelty", "repro", "correctness", "efficiency", "stability", "usability",
    }


def test_score_in_range() -> None:
    """Every score is in [0, 3]."""
    r = grade(".")
    for d in r.dims.values():
        assert 0 <= d.score.value <= 3


def test_score_validates_range() -> None:
    """Score dataclass rejects values outside [0, 3]."""
    with pytest.raises(ValueError):
        Score(value=4, evidence="bad")
    with pytest.raises(ValueError):
        Score(value=-1, evidence="bad")


def test_total_is_sum() -> None:
    """total = sum of all dimension scores."""
    r = grade(".")
    expected = sum(d.score.value for d in r.dims.values())
    assert r.total == expected


def test_dimension_dataclass() -> None:
    """Dimension dataclass is constructable."""
    s = Score(value=2, evidence="ok")
    d = Dimension(name="test", score=s)
    assert d.name == "test"
    assert d.score.value == 2


@pytest.mark.rubric(name="novelty")
def test_rubric_novelty_marker(rubric: dict) -> None:
    """The @pytest.mark.rubric marker exposes the rubric fixture."""
    assert rubric["novelty"] >= 0