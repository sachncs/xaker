"""xaker-run-paper-experiment: typed experiment driver.

Reads a YAML spec from examples/specs/ and runs it through the bench
driver, writing JSON to paper_runs/.

Usage:
    python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml --check
    python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from xaker.bench import Spec, run, write
from xaker.utils.ctx import Ctx


def load_spec(path: Path) -> Spec:
    """Load a Spec from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Spec(
        lengths=data["lengths"],
        dim=data.get("dim", 64),
        heads=data.get("heads", 4),
        batch=data.get("batch", 2),
        kinds=data.get("kinds", ["standard", "xsa", "fused"]),
        warmup=data.get("warmup", 2),
        runs=data.get("runs", 5),
        seeds=data.get("seeds", [0]),
        precond=data.get("precond", "fast"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run XAKER paper experiment")
    parser.add_argument("--spec", required=True, help="Path to YAML spec")
    parser.add_argument("--output", default=None, help="Output JSON path (default: paper_runs/<spec>.json)")
    parser.add_argument("--check", action="store_true", help="Smoke test (tiny config)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args(argv)

    spec = load_spec(Path(args.spec))
    if args.check:
        spec = Spec(
            lengths=[min(spec.lengths)],
            dim=min(spec.dim, 32),
            heads=min(spec.heads, 2),
            kinds=spec.kinds[:1],
            warmup=1,
            runs=1,
            seeds=[0],
        )

    ctx = Ctx(device="cuda" if args.cuda and __import__("torch").cuda.is_available() else "cpu")
    result = run(spec, ctx=ctx)

    if args.output:
        out = Path(args.output)
    else:
        name = Path(args.spec).stem
        out = Path("paper_runs") / f"{name}.json"
    write(result, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())