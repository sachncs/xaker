"""Markdown / JSON rendering for :class:`Rubric` results."""

from __future__ import annotations

import json
from pathlib import Path

from xaker.rubric.rubric import Rubric


def markdown(r: Rubric) -> str:
    """Render a :class:`Rubric` as a Markdown table."""
    rows = []
    for name, d in r.dims.items():
        rows.append(f"| {name} | {d.score.value}/3 | {d.score.evidence} |")
    body = "\n".join(rows)
    status = "PASS" if r.passed else "FAIL"
    return (
        f"# Paper-Worthiness Rubric\n\n"
        f"**Status:** {status}  **Total:** {r.total}/18\n\n"
        f"| Dimension | Score | Evidence |\n"
        f"|---|---|---|\n"
        f"{body}\n"
    )


def write(r: Rubric, path: Path) -> None:
    """Write the rubric result as both Markdown and JSON to ``path``.

    Args:
        r: Rubric result to serialize.
        path: Output directory. Created if missing.

    Side Effects:
        Writes ``path/summary.md`` and ``path/rubric.json``.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.md").write_text(markdown(r), encoding="utf-8")
    payload = {
        "total": r.total,
        "passed": r.passed,
        "dims": {
            name: {"value": d.score.value, "evidence": d.score.evidence}
            for name, d in r.dims.items()
        },
    }
    (path / "rubric.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
