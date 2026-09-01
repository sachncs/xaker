"""End-to-end smoke tests for CLI entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_runs() -> None:
    """xaker-train runs end-to-end with a tiny config."""
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.train",
         "--dim", "32", "--heads", "2", "--layers", "1",
         "--vocab", "20", "--epochs", "1", "--batch", "2",
         "--length", "8", "--samples", "8"],
        capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert r.returncode == 0, f"xaker-train failed: {r.stderr}"


def test_validate_runs() -> None:
    """xaker-validate runs and exits 0 when total >= 14."""
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.validate", "--min-total", "14"],
        capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert r.returncode == 0, f"xaker-validate failed: {r.stderr}"
    assert "PASS" in r.stdout


def test_bench_runs(tmp_path: Path) -> None:
    """xaker-bench runs and writes JSON."""
    out = tmp_path / "out.json"
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.bench",
         "--lengths", "8", "--warmup", "1", "--runs", "1",
         "--output", str(out)],
        capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert r.returncode == 0, f"xaker-bench failed: {r.stderr}"
    assert out.exists()


def test_eval_import() -> None:
    """xaker-eval CLI is importable."""
    from xaker.cli.eval import main
    assert callable(main)
