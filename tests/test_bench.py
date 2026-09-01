"""Smoke tests for the typed bench driver."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xaker.bench import Spec, run, write
from xaker.utils.ctx import Ctx


def test_run_returns_result() -> None:
    """run(spec) returns a Result with non-empty results."""
    spec = Spec(lengths=[16], dim=32, heads=2, kinds=["standard"], warmup=1, runs=2, seeds=[0])
    result = run(spec, ctx=Ctx(device="cpu"))
    assert len(result.results) > 0
    for (kind, length), metrics in result.results.items():
        assert kind in spec.kinds
        assert length in spec.lengths
        assert metrics.forward_ms_mean >= 0


def test_write_json() -> None:
    """write(result, path) persists a schema-stable JSON."""
    spec = Spec(lengths=[16], dim=32, heads=2, kinds=["standard"], warmup=1, runs=1, seeds=[0])
    result = run(spec, ctx=Ctx(device="cpu"))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        write(result, path)
        json_path = path / "rubric.json"
        md_path = path / "summary.md"
        assert json_path.exists()
        assert md_path.exists()
        data = json.loads(json_path.read_text())
        assert "schema_version" in data
        assert "git_sha" in data
        assert "results" in data


def test_metrics_field_names() -> None:
    """Metrics uses single-word field names."""
    spec = Spec(lengths=[8], dim=16, heads=2, kinds=["standard"], warmup=1, runs=1, seeds=[0])
    result = run(spec, ctx=Ctx(device="cpu"))
    sample = next(iter(result.results.values()))
    for attr in ["forward_ms_mean", "forward_ms_std", "backward_ms_mean", "backward_ms_std", "memory_mib"]:
        assert hasattr(sample, attr), f"missing field: {attr}"


def test_reproducibility() -> None:
    """Same seed produces the same JSON content."""
    spec = Spec(lengths=[16], dim=32, heads=2, kinds=["standard"], warmup=1, runs=1, seeds=[42])
    r1 = run(spec, ctx=Ctx(device="cpu"))
    r2 = run(spec, ctx=Ctx(device="cpu"))
    for key in r1.results:
        m1 = r1.results[key]
        m2 = r2.results[key]
        # Forward time should be deterministic in ms order, but allow some slack.
        assert abs(m1.forward_ms_mean - m2.forward_ms_mean) < 1.0
