"""xaker-validate: run the paper-worthiness rubric and print results.

Usage:
    xaker-validate --repo-root . --min-total 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xaker.rubric import grade, markdown, write


def main(argv: list[str] | None = None) -> int:
    """Parse CLI flags, run the rubric, and enforce pass thresholds.

    Args:
        argv: Optional argument vector; ``None`` reads from
            ``sys.argv``.

    Returns:
        Process exit code; ``0`` when the rubric passes, ``1``
        otherwise.
    """
    parser = argparse.ArgumentParser(description="Run paper-worthiness rubric")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--min-total", type=int, default=14, help="Minimum total score to pass")
    parser.add_argument("--strict", action="store_true", help="Fail on any dim < 2")
    parser.add_argument("--json", default=None, help="Path to write JSON output")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    r = grade(str(repo_root))

    print(markdown(r))

    if args.json:
        write(r, Path(args.json))
        print(f"JSON written to {args.json}")

    if not r.passed:
        return 1
    if r.total < args.min_total:
        return 1
    if args.strict:
        for name, d in r.dims.items():
            if name != "novelty" and d.score.value < 2:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
