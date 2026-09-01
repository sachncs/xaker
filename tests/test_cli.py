"""Smoke tests for CLI entry points.

Each CLI is importable and exposes a callable ``main``.
"""

from xaker.cli.train import main as train_main
from xaker.cli.bench import main as bench_main
from xaker.cli.eval import main as eval_main


def test_train() -> None:
    """Train CLI is callable."""
    assert callable(train_main)


def test_bench() -> None:
    """Bench CLI is callable."""
    assert callable(bench_main)


def test_eval() -> None:
    """Eval CLI is callable."""
    assert callable(eval_main)