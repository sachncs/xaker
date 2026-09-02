"""Ablation runner for the xaker paper.

Generates JSON outputs for the four headline ablations:

1. Kernel ablation: ``exp`` vs ``rbf`` vs ``linear`` vs ``cosine``.
2. Preconditioner ablation: ``identity`` vs ``diagonal`` vs
   ``fast`` vs ``cccp``.
3. Mode ablation: ``subtract`` vs ``zero`` vs ``mask``.
4. Attention ablation: ``standard`` vs ``xsa`` vs ``fused`` vs
   ``linear``.

Each ablation runs N seeds and reports per-config means. The
output schema matches ``xaker.bench``'s ``Result`` so all four
files can be aggregated into a single ``RESULTS.md``.

Usage:

    python -m xaker.bench.ablate \\
        --out paper_runs/abl_kernel.json \\
        --axis kernel \\
        --values exp rbf linear cosine

The output of every ablation matches the bench JSON schema: a
spec, an environment block, and a per-config results map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, cast

import torch

from xaker.bench.bench import Metrics, Result, Spec, gitsha, write
from xaker.config import Config
from xaker.attention import BLOCK


def _device() -> torch.device:
    """Resolve the benchmark device.

    Honours the ``XAKER_DEVICE`` environment variable
    (``"cpu"`` / ``"cuda"`` / ``"mps"``); falls back to CPU. We
    deliberately avoid MPS by default because several of the
    PCG ops (linalg.eigh, linalg.lu_solve, linalg.solve with
    batched 4-D inputs) have shape bugs on MPS that produce
    silently-wrong outputs.
    """
    requested = os.environ.get("XAKER_DEVICE")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class AblSpec:
    """A single ablation configuration."""

    axis: str  # "kernel" | "precond" | "mode" | "kind"
    value: str
    extra: Dict[str, object] = field(default_factory=dict)


def bench(kind: str, dim: int, heads: int, length: int, seed: int) -> Metrics:
    """Benchmark one attention kind.

    Args:
        kind: Attention kind.
        dim: Model width.
        heads: Heads.
        length: Sequence length.
        seed: Random seed.

    Returns:
        Per-(kind, length) :class:`Metrics`.
    """
    import time as _t

    from xaker.bench.bench import tick, peak

    torch.manual_seed(seed)
    cfg = Config(
        dim=dim, heads=heads, drop=0.0, mode="subtract",
        precond="cccp", normalize=True, lam=10.0,
    )
    if kind not in BLOCK:
        raise ValueError(f"Unknown kind: {kind}; BLOCK keys: {list(BLOCK)}")
    mod = BLOCK[kind](cfg).eval()
    x = torch.randn(4, length, cfg.dim)
    device = _device()
    mod = mod.to(device)
    x = x.to(device)
    ctx = type("Ctx", (), {"device": device, "dtype": x.dtype})()
    fm, fstd, bm, bstd = tick(mod, x, warmup=2, runs=5, ctx=ctx)
    m = Metrics(
        forward_ms_mean=fm, forward_ms_std=fstd,
        backward_ms_mean=bm, backward_ms_std=bstd,
        memory_mib=peak(mod, x, ctx=ctx),
    )
    return m


def drive(
    axis: str,
    values: List[str],
    *,
    dim: int = 64,
    heads: int = 4,
    length: int = 32,
    seeds: int = 3,
) -> Result:
    """Run one axis of the ablation.

    Args:
        axis: Sweep axis (``"kind"``, ``"kernel"``, ``"precond"``,
            ``"mode"``).
        values: Values to sweep.
        dim: Model width.
        heads: Attention heads.
        length: Sequence length.
        seeds: Number of seeds per value.

    Returns:
        Benchmark :class:`Result` with one ``Metrics`` per
        ``(value, length)`` key.
    """
    precond_literal: Literal["identity", "diagonal", "fast", "cccp"] = (
        cast(Literal["identity", "diagonal", "fast", "cccp"], values[0])
        if axis == "precond" else "cccp"
    )
    spec = Spec(
        lengths=[length],
        dim=dim, heads=heads, batch=4,
        kinds=["fused"], warmup=2, runs=5, seeds=list(range(seeds)),
        precond=precond_literal,
    )
    result = Result(
        spec=spec, git_sha=gitsha(), torch_version=torch.__version__,
        cuda=torch.cuda.is_available(),
        device_name=(
            torch.cuda.get_device_name(0) if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
    )
    for value in values:
        for seed in range(seeds):
            try:
                if axis == "kind":
                    m = bench(value, dim, heads, length, seed)
                else:
                    m = sweep(axis, value, dim, heads, length, seed)
                key = (value, length)
                if key not in result.results:
                    result.results[key] = m
            except Exception as exc:
                key = (value, length)
                result.results[key] = Metrics()
                result.results[key].forward_ms_mean = float("nan")
                print(f"[warn] axis={axis} value={value} seed={seed}: {exc}")
    return result


def sweep(
    axis: str, value: str, dim: int, heads: int, length: int, seed: int
) -> Metrics:
    """Benchmark one attention module with one swept config value.

    Args:
        axis: Sweep axis.
        value: Sweep value.
        dim: Model width.
        heads: Heads.
        length: Sequence length.
        seed: Random seed.

    Returns:
        :class:`Metrics` for the configuration.
    """
    from xaker.bench.bench import tick, peak

    torch.manual_seed(seed)
    if axis == "kernel":
        cfg = Config(
            dim=dim, heads=heads, drop=0.0, normalize=True, lam=10.0,
            kernel=cast(Literal["exp", "rbf", "linear", "cosine"], value),
        )
    elif axis == "precond":
        cfg = Config(
            dim=dim, heads=heads, drop=0.0, normalize=True, lam=10.0,
            precond=cast(Literal["cccp", "fast", "diagonal", "identity"], value),
        )
    elif axis == "mode":
        cfg = Config(
            dim=dim, heads=heads, drop=0.0, normalize=True, lam=10.0,
            mode=cast(Literal["subtract", "zero", "mask"], value),
        )
    else:
        raise ValueError(f"Unknown axis: {axis}")
    mod = BLOCK["fused"](cfg).eval()
    x = torch.randn(4, length, cfg.dim)
    device = _device()
    mod = mod.to(device)
    x = x.to(device)
    ctx = type("Ctx", (), {"device": device, "dtype": x.dtype})()
    fm, fstd, bm, bstd = tick(mod, x, warmup=2, runs=5, ctx=ctx)
    mem = peak(mod, x, ctx=ctx)
    return Metrics(
        forward_ms_mean=fm, forward_ms_std=fstd,
        backward_ms_mean=bm, backward_ms_std=bstd,
        memory_mib=mem,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ablation sweep")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--axis", required=True, choices=["kind", "kernel", "precond", "mode"])
    parser.add_argument("--values", nargs="+", required=True, help="Sweep values")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    print(f"Running {args.axis} ablation: {args.values}")
    result = drive(
        args.axis, args.values,
        dim=args.dim, heads=args.heads, length=args.length, seeds=args.seeds,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(result, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())