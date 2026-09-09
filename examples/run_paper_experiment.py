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

from xaker.attention import BLOCK
from xaker.bench import Spec, run, write
from xaker.utils.ctx import Ctx


def load_spec(path: Path) -> Spec:
    """Load a Spec from a YAML file.

    Args:
        path: Path to a YAML spec file. The file must define
            ``lengths`` and may define any of the optional
            ``Spec`` fields.

    Returns:
        The parsed :class:`xaker.bench.Spec`.

    Raises:
        ValueError: If the spec names an attention kind that is
            not registered in :data:`xaker.attention.BLOCK`, or if a
            required field is missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    if "lengths" not in data:
        raise ValueError(f"{path}: missing required field 'lengths'")

    valid_precond = {"identity", "diagonal", "fast", "cccp"}
    precond = data.get("precond", "fast")
    if precond not in valid_precond:
        raise ValueError(
            f"{path}: precond {precond!r} not in {sorted(valid_precond)}"
        )

    kinds = data.get("kinds", ["standard", "xsa", "fused"])
    bad = [k for k in kinds if k not in BLOCK]
    if bad:
        raise ValueError(
            f"{path}: unknown kind(s) {bad}; known: {sorted(BLOCK)}"
        )
    return Spec(
        lengths=data["lengths"],
        dim=data.get("dim", 64),
        heads=data.get("heads", 4),
        batch=data.get("batch", 2),
        kinds=kinds,
        warmup=data.get("warmup", 2),
        runs=data.get("runs", 5),
        seeds=data.get("seeds", [0]),
        precond=precond,
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