"""xaker-bench: run the typed benchmark driver and write JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from xaker.bench import Spec, run, write
from xaker.utils.ctx import Ctx


def main(argv: list[str] | None = None) -> int:
    """Parse CLI flags and dispatch to :func:`xaker.bench.run`.

    Args:
        argv: Optional argument vector; ``None`` reads from
            ``sys.argv``.

    Returns:
        Process exit code; ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Run XAKER benchmarks")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--kinds", nargs="+", choices=["standard", "xsa", "fused"], default=["standard", "xsa", "fused"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", default="paper_runs/bench.json")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args(argv)

    spec = Spec(
        lengths=args.lengths,
        dim=args.dim,
        heads=args.heads,
        kinds=args.kinds,
        warmup=args.warmup,
        runs=args.runs,
        seeds=args.seeds,
    )
    ctx = Ctx(device="cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    result = run(spec, ctx=ctx)
    out = Path(args.output)
    write(result, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
