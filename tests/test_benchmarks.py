"""Smoke tests for the v2 bench infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path

from xaker.bench import Metrics, Spec, Result, run, write


def test_run_smoke() -> None:
    """Bench run produces a Result with at least one entry."""
    spec = Spec(lengths=[16], dim=32, heads=2, kinds=["standard"], warmup=1, runs=2, seeds=[0])
    result = run(spec)
    assert isinstance(result, Result)
    assert len(result.results) >= 1


def test_metrics_dataclass() -> None:
    """Metrics has expected field names."""
    m = Metrics()
    assert hasattr(m, "forward_ms_mean")
    assert hasattr(m, "memory_mib")
    assert hasattr(m, "converged")


def test_write_persists() -> None:
    """write(result, path) writes a JSON file."""
    spec = Spec(lengths=[8], dim=16, heads=2, kinds=["xsa"], warmup=1, runs=1, seeds=[0])
    result = run(spec)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.json"
        write(result, p)
        assert p.exists()
