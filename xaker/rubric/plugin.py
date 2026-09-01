"""pytest plugin providing the rubric marker.

Loaded automatically via ``tests/conftest.py``:

    pytest_plugins = ["xaker.rubric.plugin"]

Usage:

    @pytest.mark.rubric(name="novelty")
    def test_novelty(self, rubric):
        assert rubric["novelty"] >= 2
"""

from __future__ import annotations

import pytest

from xaker.rubric.grader import grade


@pytest.fixture(scope="session")
def rubric() -> dict:
    """The current rubric result as a dict mapping dimension name -> score int."""
    r = grade(".")
    return {name: d.score.value for name, d in r.dims.items()}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "rubric(name): paper-worthiness rubric gating test",
    )