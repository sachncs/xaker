"""Individual rubric dimension graders.

Each grader inspects the repository and returns a :class:`Score` plus
evidence text describing what was found.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from xaker.rubric.rubric import Dimension, Rubric, Score


def _grep_count(repo_root: Path, pattern: str, paths: list[str]) -> int:
    cmd = ["git", "-C", str(repo_root), "grep", "-r", "-l", pattern, "--"] + paths
    try:
        return len(subprocess.check_output(cmd, stderr=subprocess.DEVNULL).split())
    except subprocess.CalledProcessError:
        return 0


def novelty(repo_root: Path) -> Dimension:
    """Score novelty by counting the *novel* attention variants in ``xaker/attention/``.

    The baseline classes (``standard``, ``xsa``) and the kernel
    helper are intentionally excluded from "novelty" because they
    reproduce the published baselines. Only the project's own
    contributions count: the flagship ``Fused`` block and the
    ``Linear`` baseline (added in 0.5.0 to support the XSA + LAKER
    paper benchmarking).

    The 0.4.0 release renamed ``laker.py`` to ``fused.py``; the
    grader tracks the current name so the evidence string matches
    the file you can actually see on disk.
    """
    novel = ("fused.py", "linear.py")
    found = tuple(
        v for v in novel
        if (repo_root / "xaker" / "attention" / v).exists()
    )
    value = min(len(found), 3)
    return Dimension(
        name="novelty",
        score=Score(value=value, evidence=f"novel={len(found)}/2 ({','.join(found) or 'none'})"),
    )


def repro(repo_root: Path) -> Dimension:
    """Score reproducibility by checking seeds, deterministic flags, and JSON outputs.

    A passing score requires ``paper_runs/`` to contain at least
    one schema-stable JSON (not the ``.gitkeep`` placeholder that
    ships in an empty clone).
    """
    seeds_ok = (repo_root / "xaker" / "utils" / "rng.py").exists()
    bench_ok = (repo_root / "xaker" / "bench" / "bench.py").exists()
    paper_dir = repo_root / "paper_runs"
    json_files = list(paper_dir.glob("*.json")) if paper_dir.exists() else []
    json_ok = bool(json_files)
    cudnn = _grep_count(repo_root, "cudnn.deterministic", ["xaker/"]) > 0
    score_value = sum([seeds_ok, bench_ok, json_ok, cudnn])
    score_value = min(score_value, 3)
    return Dimension(
        name="repro",
        score=Score(
            value=score_value,
            evidence=f"seeds={seeds_ok} bench={bench_ok} paper_runs={len(json_files)} cudnn={cudnn}",
        ),
    )


def correctness(repo_root: Path) -> Dimension:
    """Score correctness by checking the test surface for invariants.

    The grader now looks for the *current* flagship test
    (``tests/test_fused.py``) instead of the obsolete
    ``tests/test_laker*.py`` glob, mirroring the 0.4.0 rename of
    ``Laker`` to ``Fused``.
    """
    test_files = list((repo_root / "tests").glob("test_*.py")) if (repo_root / "tests").exists() else []
    has_fused = (repo_root / "tests" / "test_fused.py").exists()
    has_cg = any("solver" in str(f) or "cg" in str(f) for f in test_files)
    has_dispatch = (repo_root / "tests" / "test_dispatch.py").exists()
    score_value = sum([has_fused, has_cg, has_dispatch])
    score_value = min(score_value, 3)
    return Dimension(
        name="correctness",
        score=Score(
            value=score_value,
            evidence=f"fused={has_fused} cg={has_cg} dispatch={has_dispatch}",
        ),
    )


def efficiency(repo_root: Path) -> Dimension:
    """Score efficiency by checking for benchmark scripts and benchmark JSON outputs.

    The grader now requires at least two committed benchmark JSON
    files under ``paper_runs/`` before awarding the third point,
    ensuring the rubric is not passed by directory presence alone.
    """
    has_bench = (repo_root / "xaker" / "bench" / "bench.py").exists()
    has_yaml = (repo_root / "examples" / "specs").exists()
    paper_dir = repo_root / "paper_runs"
    benchmark_jsons = [p for p in paper_dir.glob("*.json") if p.is_file()] if paper_dir.exists() else []
    runs_count = len(benchmark_jsons)
    has_runs = runs_count >= 2
    score_value = sum([has_bench, has_yaml, has_runs])
    score_value = min(score_value, 3)
    return Dimension(
        name="efficiency",
        score=Score(
            value=score_value,
            evidence=f"bench={has_bench} specs={has_yaml} runs={runs_count}",
        ),
    )


def stability(repo_root: Path) -> Dimension:
    """Score stability by checking for multi-seed and dtype sweeps."""
    seeds = _grep_count(repo_root, "seeds=\\[", ["xaker/", "examples/"]) > 0
    paper_dir = repo_root / "paper_runs"
    multi_seed_runs = (
        bool(list(paper_dir.glob("*.json"))) if paper_dir.exists() else False
    )
    dtype_present = _grep_count(repo_root, "dtype", ["xaker/bench/"]) > 0
    score_value = sum([seeds, multi_seed_runs, dtype_present])
    score_value = min(score_value, 3)
    return Dimension(
        name="stability",
        score=Score(value=score_value, evidence=f"seeds={seeds} runs={multi_seed_runs} dtype={dtype_present}"),
    )


def usability(repo_root: Path) -> Dimension:
    """Score usability by checking CLI, README, docs."""
    has_cli = (repo_root / "xaker" / "cli").exists()
    has_readme = (repo_root / "README.md").exists()
    has_rubric = (repo_root / "xaker" / "rubric").exists()
    score_value = sum([has_cli, has_readme, has_rubric])
    score_value = min(score_value, 3)
    return Dimension(
        name="usability",
        score=Score(value=score_value, evidence=f"cli={has_cli} readme={has_readme} rubric={has_rubric}"),
    )


GRADERS = {
    "novelty": novelty,
    "repro": repro,
    "correctness": correctness,
    "efficiency": efficiency,
    "stability": stability,
    "usability": usability,
}


def grade(repo_root: str = ".") -> "Rubric":
    """Run all six graders and assemble a :class:`Rubric`."""
    root = Path(repo_root).resolve()
    dims = {name: g(root) for name, g in GRADERS.items()}
    return Rubric(dims=dims)
