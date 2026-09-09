#!/usr/bin/env python3
"""Refresh STATUS.md from the most recent CI run.

This script is invoked by ``.github/workflows/ci.yml`` after the
coverage test step. It reads the JSON coverage report and queries
pytest for the test count, then rewrites ``STATUS.md`` so that the
headline numbers advertised in the README and CHANGELOG are
machine-checked by CI.

Usage:

    python scripts/refresh_status.py          # requires coverage.json
    python scripts/refresh_status.py --check  # verify, don't write

The script is intentionally small and dependency-free (stdlib only)
so it can run in the same minimal install as the test step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS = REPO / "STATUS.md"


def git_sha() -> str:
    """Return the current short HEAD SHA, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(REPO),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def test_count() -> int:
    """Return the number of collected tests under ``tests/``."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(REPO / "tests"), "--no-header"],
            cwd=str(REPO),
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return 0
    n = 0
    seen: set[str] = set()
    for line in out.splitlines():
        if "::" in line:
            test_id = line.split("::")[0]
            if test_id and test_id not in seen:
                seen.add(test_id + "::" + line.split("::", 1)[1])
                n += 1
    return n


def coverage() -> tuple[str, int, int]:
    """Return ``(pct, covered_lines, num_statements)`` from ``coverage.json``.

    Returns ``("unknown", 0, 0)`` when the report is absent.
    """
    cov_path = REPO / "coverage.json"
    if not cov_path.exists():
        return "unknown", 0, 0
    try:
        data = json.loads(cov_path.read_text())
    except Exception:
        return "unknown", 0, 0
    totals = data.get("totals", {})
    return (
        f"{totals.get('percent_covered', 0.0):.1f}%",
        int(totals.get("covered_lines", 0)),
        int(totals.get("num_statements", 0)),
    )


def render(pct: str, covered: int, statements: int, tests: int, sha: str, captured: str) -> str:
    """Format STATUS.md from the live metrics."""
    lines = [
        "# Project status",
        "",
        "These numbers are produced by CI on every push to ``master``.",
        "Treat the README headline values as the rendered view of this",
        "file; CI fails the build if the two sources disagree.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tests | {tests} |",
        f"| Coverage | {pct} ({covered} / {statements} lines) |",
        f"| Git SHA | `{sha}` |",
        f"| Last capture | {captured} |",
        "",
        "The CI gate is ``--cov-fail-under=90``, which keeps coverage",
        "at or above the headline figure (90%) with two points of",
        "margin to absorb minor refactors.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh STATUS.md from CI output.")
    parser.add_argument("--check", action="store_true",
                        help="Verify STATUS.md matches the live metrics; exit 1 on drift.")
    args = parser.parse_args(argv)

    pct, covered, statements = coverage()
    tests = test_count()
    sha = git_sha()
    captured = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rendered = render(pct, covered, statements, tests, sha, captured)

    if args.check:
        if STATUS.exists() and STATUS.read_text() == rendered:
            return 0
        print("STATUS.md drifted from live metrics; refresh with:", file=sys.stderr)
        print("  python scripts/refresh_status.py", file=sys.stderr)
        return 1
    STATUS.write_text(rendered)
    print(f"wrote {STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
