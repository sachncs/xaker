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
    """Score novelty by counting attention variants in xaker/attention/."""
    variants = {"standard.py", "xsa.py", "laker.py", "kernel.py"}
    found = 0
    for v in variants:
        if (repo_root / "xaker" / "attention" / v).exists():
            found += 1
    value = min(found, 3)
    return Dimension(
        name="novelty",
        score=Score(value=value, evidence=f"{found}/4 attention variants"),
    )


def repro(repo_root: Path) -> Dimension:
    """Score reproducibility by checking seeds, deterministic flags, and JSON outputs."""
    seeds_ok = (repo_root / "xaker" / "utils" / "rng.py").exists()
    bench_ok = (repo_root / "xaker" / "bench" / "bench.py").exists()
    json_ok = (repo_root / "paper_runs").exists() or (repo_root / "paper_runs").is_dir()
    cudnn = _grep_count(repo_root, "cudnn.deterministic", ["xaker/"]) > 0
    score_value = sum([seeds_ok, bench_ok, json_ok, cudnn])
    score_value = min(score_value, 3)
    return Dimension(
        name="repro",
        score=Score(
            value=score_value,
            evidence=f"seeds={seeds_ok} bench={bench_ok} paper_runs={json_ok} cudnn={cudnn}",
        ),
    )


def correctness(repo_root: Path) -> Dimension:
    """Score correctness by checking test files for invariants."""
    test_files = list((repo_root / "tests").glob("test_*.py")) if (repo_root / "tests").exists() else []
    has_laker = any("laker" in str(f) for f in test_files)
    has_cg = any("solver" in str(f) or "cg" in str(f) for f in test_files)
    has_dispatch = (repo_root / "tests" / "test_dispatch.py").exists()
    score_value = sum([has_laker, has_cg, has_dispatch])
    score_value = min(score_value, 3)
    return Dimension(
        name="correctness",
        score=Score(
            value=score_value,
            evidence=f"laker={has_laker} cg={has_cg} dispatch={has_dispatch}",
        ),
    )


def efficiency(repo_root: Path) -> Dimension:
    """Score efficiency by checking for benchmark scripts and JSON outputs."""
    has_bench = (repo_root / "xaker" / "bench" / "bench.py").exists()
    has_yaml = (repo_root / "examples" / "specs").exists()
    has_runs = (repo_root / "paper_runs").exists()
    score_value = sum([has_bench, has_yaml, has_runs])
    score_value = min(score_value, 3)
    return Dimension(
        name="efficiency",
        score=Score(value=score_value, evidence=f"bench={has_bench} specs={has_yaml} runs={has_runs}"),
    )


def stability(repo_root: Path) -> Dimension:
    """Score stability by checking for multi-seed and dtype sweeps."""
    seeds = _grep_count(repo_root, "seeds=\\[", ["xaker/", "examples/"]) > 0
    multi_seed_runs = (repo_root / "paper_runs").exists()
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