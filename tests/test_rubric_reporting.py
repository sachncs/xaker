"""Tests for the rubric markdown / JSON rendering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xaker.rubric import grade, markdown, write


def test_markdown() -> None:
    """markdown(r) returns a string with the table."""
    r = grade(".")
    md = markdown(r)
    assert "Paper-Worthiness Rubric" in md
    assert "| Dimension" in md


def test_write_dir() -> None:
    """write(r, dir) writes rubric.json and summary.md into the directory."""
    r = grade(".")
    with tempfile.TemporaryDirectory() as d:
        write(r, Path(d))
        assert (Path(d) / "rubric.json").exists()
        assert (Path(d) / "summary.md").exists()


def test_write_file() -> None:
    """write(r, file_path) writes a JSON file (or directory if path is dir)."""
    r = grade(".")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "subdir.json"
        # Use the file-path API by ensuring the parent dir exists.
        out.parent.mkdir(exist_ok=True)
        write(r, out)
        assert out.exists() or (out / "rubric.json").exists()
