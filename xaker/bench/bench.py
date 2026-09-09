from __future__ import annotations

from pathlib import Path

import gc
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from torch import nn

from xaker.attention import BLOCK
from xaker.attention.fused import Fused
from xaker.config import Config
from xaker.solver.cg import pcg
from xaker.utils.ctx import Ctx
from xaker.utils.rng import seed


@dataclass
class Spec:
    """Typed benchmark specification.

    Attributes:
        lengths: Sequence lengths to measure.
        dim: Embedding dimension.
        heads: Number of attention heads.
        batch: Batch size.
        kinds: Attention kinds to measure.
        warmup: Warmup iterations (untimed).
        runs: Timed iterations.
        seeds: Random seeds for multi-seed runs.
        precond: Preconditioner kind for the fused block.
    """

    lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    dim: int = 256
    heads: int = 8
    batch: int = 4
    kinds: List[Literal["standard", "xsa", "fused"]] = field(
        default_factory=lambda: ["standard", "xsa", "fused"]
    )
    warmup: int = 10
    runs: int = 50
    seeds: List[int] = field(default_factory=lambda: [0])
    precond: Literal["identity", "diagonal", "fast", "cccp"] = "fast"


@dataclass
class Metrics:
    """Per-(kind, length) measurements."""

    forward_ms_mean: float = 0.0
    forward_ms_std: float = 0.0
    backward_ms_mean: float = 0.0
    backward_ms_std: float = 0.0
    memory_mib: float = 0.0
    iters_mean: Optional[float] = None
    iters_std: Optional[float] = None
    converged: bool = False


@dataclass
class Result:
    """Complete benchmark result with environment block."""

    spec: Spec
    git_sha: str = ""
    torch_version: str = ""
    cuda: bool = False
    device_name: str = ""
    cudnn_deterministic: bool = False
    results: Dict[tuple, Metrics] = field(default_factory=dict)


def gitsha() -> str:
    """Return the current git HEAD SHA, or empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.getcwd(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _sync() -> None:
    """Synchronize the active accelerator.

    Synchronizes CUDA when available; otherwise MPS. Used to
    bracket timed regions in :func:`tick` and :func:`peak`.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def tick(
    module: nn.Module,
    x: torch.Tensor,
    *,
    warmup: int,
    runs: int,
    ctx: Ctx,
) -> tuple[float, float, float, float]:
    """Measure forward and forward+backward latency.

    Args:
        module: Module to benchmark. Moved to ``ctx`` device and dtype.
        x: Input tensor. Moved to ``ctx`` device and dtype.
        warmup: Number of untimed forward passes before measurement.
        runs: Number of timed forward (and forward+backward) passes.
        ctx: Execution context carrying device and dtype.

    Returns:
        Tuple ``(forward_ms_mean, forward_ms_std, backward_ms_mean,
        backward_ms_std)`` in milliseconds.
    """
    module = module.to(ctx.device).to(ctx.dtype)
    x = x.to(ctx.device).to(ctx.dtype)

    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)

    _sync()

    ft: List[float] = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = module(x)
        _sync()
        ft.append((time.perf_counter() - t0) * 1000)
    fm = sum(ft) / len(ft)
    fstd = (sum((t - fm) ** 2 for t in ft) / max(len(ft) - 1, 1)) ** 0.5

    module.train()
    x_g = x.clone().requires_grad_(True)
    bt: List[float] = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        out = module(x_g)
        out.sum().backward()
        _sync()
        bt.append((time.perf_counter() - t0) * 1000)
        x_g.grad = None
    bm = sum(bt) / len(bt)
    bstd = (sum((t - bm) ** 2 for t in bt) / max(len(bt) - 1, 1)) ** 0.5
    return fm, fstd, bm, bstd


def peak(module: nn.Module, x: torch.Tensor, *, ctx: Ctx) -> float:
    """Measure peak GPU memory in MiB.

    Args:
        module: Module to benchmark.
        x: Input tensor.
        ctx: Execution context.

    Returns:
        Peak GPU memory in MiB. ``0.0`` when neither CUDA nor MPS
        is available.
    """
    module = module.to(ctx.device).to(ctx.dtype)
    x = x.to(ctx.device).to(ctx.dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x_g = x.clone().requires_grad_(True)
        out = module(x_g)
        out.sum().backward()
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return 0.0  # MPS does not expose peak memory stats
    return 0.0


def converge(
    attn: nn.Module,
    x: torch.Tensor,
    *,
    ctx: Ctx,
    iters: int = 50,
    runs: int = 5,
    tol: float = 1e-2,
) -> tuple[bool, float, float]:
    """Measure PCG convergence on the fused attention kernel.

    Runs ``runs`` independent PCG solves against a representative
    right-hand side derived from ``x`` and returns the convergence
    flag plus the mean and population std of the iteration count.
    A run is reported as ``converged`` only if every individual
    call cleared ``tol``.

    Args:
        attn: Attention module to probe. Only :class:`Fused`
            exposes PCG; non-fused callers receive a synthetic
            ``(False, 0.0, 0.0)`` triple.
        x: Input tensor.
        ctx: Execution context.
        iters: Maximum PCG iterations per solve.
        runs: Number of independent solves for the std estimate.
        tol: Relative residual convergence threshold.

    Returns:
        Tuple ``(converged, iters_mean, iters_std)``.
    """
    # Only ``Fused`` solves via PCG. Other variants return a
    # well-defined ``(False, 0.0, 0.0)`` triple so downstream
    # consumers can distinguish "no iterative solver used" from
    # "PCG converged".
    if not isinstance(attn, Fused) or not attn.kernel_fn or not attn.precon:
        return False, 0.0, 0.0

    attn = attn.to(ctx.device).to(ctx.dtype)
    x = x.to(ctx.device).to(ctx.dtype)

    headdim = attn.headdim
    if headdim is None:
        return False, 0.0, 0.0

    with torch.no_grad():
        # Warm pass to ensure parameters and the preconditioner
        # cache have stabilised before timing.
        _ = attn(x)

        history: list = []
        for _ in range(max(runs, 1)):
            q, k, v = attn.qkv_proj(x)
            qh = q.view(q.shape[0], q.shape[1], -1, headdim).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], -1, headdim).transpose(1, 2)
            vh = v.view(v.shape[0], v.shape[1], -1, headdim).transpose(1, 2)
            kernel = attn.kernel_fn(qh, kh)
            length = qh.size(-2)
            lam = attn.lam.view(1, 1, 1, 1)
            data = attn.precon.build(kernel, lam, length)
            solve = pcg(
                kernel=kernel,
                b=vh,
                lam=lam,
                precond_data=data,
                apply_pre=attn.precon.apply_pre,
                iters=iters,
                tol=tol,
                miniters=3,
            )
            history.append(solve)

    converged = all(s.converged for s in history)
    iters_arr = [s.iters for s in history]
    mean = sum(iters_arr) / len(iters_arr)
    var = sum((i - mean) ** 2 for i in iters_arr) / max(len(iters_arr), 1)
    return converged, mean, var ** 0.5


def run(spec: Spec, *, ctx: Optional[Ctx] = None) -> Result:
    """Run the benchmark suite.

    Args:
        spec: Benchmark specification.
        ctx: Execution context. ``None`` falls back to a CPU
            ``float32`` :class:`Ctx`.

    Returns:
        Populated :class:`Result` with environment block and per-
        (kind, length) :class:`Metrics`.
    """
    ctx = ctx or Ctx()

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )

    result = Result(
        spec=spec,
        git_sha=gitsha(),
        torch_version=torch.__version__,
        cuda=torch.cuda.is_available(),
        device_name=device_name,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
    )

    for s in spec.seeds:
        seed(s)
        for kind in spec.kinds:
            cls = BLOCK[kind]
            for length in spec.lengths:
                cfg = Config(
                    dim=spec.dim,
                    heads=spec.heads,
                    precond=spec.precond,
                )
                mod = cls(cfg).to(ctx.device).to(ctx.dtype)
                x = torch.randn(
                    spec.batch, length, spec.dim, device=ctx.device, dtype=ctx.dtype
                )

                fm, fstd, bm, bstd = tick(
                    mod, x, warmup=spec.warmup, runs=spec.runs, ctx=ctx
                )
                mem = peak(mod, x, ctx=ctx)
                # Default to (False, 0.0, 0.0) for non-fused kinds so
                # ``Metrics.converged`` honestly reflects whether an
                # iterative solver was actually exercised.
                conv, itm, its = (False, 0.0, 0.0)
                if kind == "fused":
                    conv, itm, its = converge(mod, x, ctx=ctx)

                key = (kind, length)
                if key not in result.results:
                    result.results[key] = Metrics()
                m = result.results[key]
                m.forward_ms_mean = fm
                m.forward_ms_std = fstd
                m.backward_ms_mean = bm
                m.backward_ms_std = bstd
                m.memory_mib = mem
                m.iters_mean = itm
                m.iters_std = its
                m.converged = conv
                gc.collect()

    return result


def write(result: Result, path: str | Path) -> None:
    """Persist ``result`` as JSON with schema validation.

    Args:
        result: Bench result to serialize.
        path: Output path. A directory writes ``rubric.json`` and
            ``summary.md``; a file path writes a single JSON file.

    Side Effects:
        Creates parent directories as needed.
    """
    d = {
        "schema_version": 1,
        "git_sha": result.git_sha,
        "torch_version": result.torch_version,
        "cuda": result.cuda,
        "device_name": result.device_name,
        "cudnn_deterministic": result.cudnn_deterministic,
        "spec": asdict(result.spec),
        "results": {
            f"{kind}:{length}": asdict(metrics)
            for (kind, length), metrics in result.results.items()
        },
    }
    p = Path(path)
    if p.is_dir() or (not p.exists() and str(path).endswith("/")):
        p.mkdir(parents=True, exist_ok=True)
        (p / "rubric.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        (p / "summary.md").write_text(_summary_md(result), encoding="utf-8")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _summary_md(result: Result) -> str:
    """Brief Markdown summary of the run."""
    lines = [f"# Bench run {result.git_sha[:8] if result.git_sha else '(none)'}",
             "", f"torch {result.torch_version} cuda={result.cuda}", "",
             "| kind | length | fwd_ms | bwd_ms | mem_mib |", "|---|---|---|---|---|"]
    for (kind, length), m in result.results.items():
        lines.append(f"| {kind} | {length} | {m.forward_ms_mean:.2f} | {m.backward_ms_mean:.2f} | {m.memory_mib:.1f} |")
    return "\n".join(lines) + "\n"
