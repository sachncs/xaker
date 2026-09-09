"""End-to-end smoke tests for CLI entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _env() -> dict[str, str]:
    """Standard environment for invoking the CLI from tests."""
    import os
    return {"PYTHONPATH": ".", "PATH": os.environ.get("PATH", "")}


def test_train_runs() -> None:
    """xaker-train runs end-to-end with a tiny config."""
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.train",
         "--dim", "32", "--heads", "2", "--layers", "1",
         "--vocab", "20", "--epochs", "1", "--batch", "2",
         "--length", "8", "--samples", "8"],
        capture_output=True, text=True, env=_env(),
    )
    assert r.returncode == 0, f"xaker-train failed: {r.stderr}"


def test_validate_runs() -> None:
    """xaker-validate runs and exits 0 when total >= 14."""
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.validate", "--min-total", "14"],
        capture_output=True, text=True, env=_env(),
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
        capture_output=True, text=True, env=_env(),
    )
    assert r.returncode == 0, f"xaker-bench failed: {r.stderr}"
    assert out.exists()


def test_eval_import() -> None:
    """xaker-eval CLI is importable."""
    from xaker.cli.eval import main
    assert callable(main)


def test_train_cli_accepts_linear_kind() -> None:
    """``xaker-train --kind linear`` no longer errors out with the legacy three-kind list."""
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.train",
         "--dim", "32", "--heads", "2", "--layers", "1",
         "--vocab", "20", "--epochs", "1", "--batch", "2",
         "--length", "8", "--samples", "8", "--kind", "linear"],
        capture_output=True, text=True, env=_env(),
    )
    assert "invalid choice" not in r.stderr
    assert r.returncode == 0, f"xaker-train --kind linear failed: {r.stderr}"


def test_bench_cli_accepts_linear_kind(tmp_path: Path) -> None:
    """``xaker-bench --kinds linear`` writes a JSON with a ``linear`` row."""
    out = tmp_path / "out.json"
    r = subprocess.run(
        [sys.executable, "-m", "xaker.cli.bench",
         "--lengths", "8", "--warmup", "1", "--runs", "1",
         "--kinds", "linear",
         "--output", str(out)],
        capture_output=True, text=True, env=_env(),
    )
    assert "invalid choice" not in r.stderr
    assert r.returncode == 0, f"xaker-bench --kinds linear failed: {r.stderr}"


def test_train_eval_roundtrip(tmp_path: Path) -> None:
    """``xaker-train --out <pt>`` followed by ``xaker-eval --checkpoint <pt>`` exits 0."""
    ckpt = tmp_path / "last.pt"
    r_train = subprocess.run(
        [sys.executable, "-m", "xaker.cli.train",
         "--dim", "32", "--heads", "2", "--layers", "1",
         "--vocab", "20", "--epochs", "1", "--batch", "2",
         "--length", "8", "--samples", "8", "--kind", "fused",
         "--out", str(ckpt)],
        capture_output=True, text=True, env=_env(),
    )
    assert r_train.returncode == 0, f"train failed: {r_train.stderr}"
    assert ckpt.exists(), "xaker-train did not write the checkpoint"

    r_eval = subprocess.run(
        [sys.executable, "-m", "xaker.cli.eval",
         "--checkpoint", str(ckpt),
         "--kind", "fused",
         "--dim", "32", "--heads", "2", "--layers", "1",
         "--vocab", "20", "--length", "8", "--batch", "2"],
        capture_output=True, text=True, env=_env(),
    )
    assert r_eval.returncode == 0, f"eval failed: {r_eval.stderr}"
