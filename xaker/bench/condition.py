"""Kernel condition-number benchmark for the xaker paper.

Compares the conditioning of the matrices that each attention kind
actually operates on:

- **Standard**: the score matrix ``Q K^T / sqrt(d)`` before softmax.
  Standard applies softmax to this; softmax gives a row-stochastic
  matrix with eigenvalues in ``[0, 1]`` which is trivially
  well-conditioned but represents a fundamentally different
  operation (attention as weighted average) from Fused.
- **Xsa**: the score matrix with diagonal zeroing.
- **Fused**: the kernel matrix ``K + lam I`` with diagonal zeroing,
  which is the matrix that PCG actually solves.

A meaningful comparison must therefore keep both at the same
mathematical level: comparing the raw ``Q K^T / sqrt(d)`` matrix
(Standard) with the post-XSA kernel matrix (Fused) measures the
conditioning of the corresponding *linear* system each method is
solving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from xaker import BLOCK, Config
from xaker.attention.ops import zerodiag
from xaker.bench.bench import gitsha


def cond(K: torch.Tensor) -> float:
    """Average condition number over batch and heads.

    Args:
        K: Square matrix of shape ``(batch, heads, n, n)``.

    Returns:
        ``kappa(K)`` averaged over ``(batch, heads)``.
    """
    sv = torch.linalg.svdvals(K)
    sv_max: torch.Tensor = sv.max(dim=-1).values
    sv_min: torch.Tensor = sv.min(dim=-1).values
    ratio: torch.Tensor = sv_max / (sv_min + 1e-10)
    return float(ratio.mean().item())


def measure(dim: int, heads: int, length: int, lam: float, n_seeds: int = 3, normalize: bool = True) -> dict:
    """Measure condition numbers per kind.

    Args:
        dim: Model width.
        heads: Number of attention heads.
        length: Sequence length.
        lam: Ridge regulariser.
        n_seeds: Number of random seeds.
        normalize: Whether to L2-normalise Q/K (matches Fused's
            default behaviour).

    Returns:
        ``{kind: {score_cond, kernel_cond, ratio_kernel_to_score}}``.
    """
    results: dict = {}
    for kind in ["standard", "xsa", "fused"]:
        score_conds = []
        kernel_conds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            cfg = Config(dim=dim, heads=heads, drop=0.0, mode="subtract",
                         precond="cccp", normalize=normalize, lam=lam)
            attn = BLOCK[kind](cfg).eval()
            x = torch.randn(4, length, cfg.dim)
            with torch.no_grad():
                q, k, v = attn.qkv_proj(x)
                batch, seq, dim = q.shape
                headdim = attn.headdim
                q = q.view(batch, seq, heads, headdim).transpose(1, 2)
                k = k.view(batch, seq, heads, headdim).transpose(1, 2)
                if normalize:
                    q = torch.nn.functional.normalize(q, dim=-1)
                    k = torch.nn.functional.normalize(k, dim=-1)
                scale = 1.0 / math.sqrt(headdim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                # Score cond = cond(scores) + small lam for stability
                eye = torch.eye(length, device=q.device).unsqueeze(0).unsqueeze(0)
                score_conds.append(cond(scores + 0.01 * eye))
                # Kernel matrix
                kernel = torch.exp(torch.matmul(q, k.transpose(-2, -1)) / cfg.temp)
                if kind != "standard":
                    kernel = zerodiag(kernel)
                kernel_conds.append(cond(kernel + lam * eye))
        score_mean = sum(score_conds) / len(score_conds)
        kernel_mean = sum(kernel_conds) / len(kernel_conds)
        ratio = kernel_mean / score_mean if score_mean > 0 else float("nan")
        results[kind] = {
            "score_cond": score_mean,
            "kernel_cond": kernel_mean,
            "ratio_kernel_to_score": ratio,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Condition-number benchmark")
    parser.add_argument("--out", default="paper_runs/condition.json")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    print(f"Condition-number benchmark, dim={args.dim}, heads={args.heads}, lam={args.lam}")
    results: dict = {
        "config": {"dim": args.dim, "heads": args.heads, "lam": args.lam, "seeds": args.seeds},
        "git_sha": gitsha(), "torch_version": torch.__version__,
        "by_length": {},
    }
    for L in args.lengths:
        print(f"  length={L} ...")
        per_kind = measure(args.dim, args.heads, L, args.lam, args.seeds)
        results["by_length"][f"L={L}"] = per_kind
        for kind, m in per_kind.items():
            print(f"    {kind}: score={m['score_cond']:.2f}, kernel={m['kernel_cond']:.2f}, ratio={m['ratio_kernel_to_score']:.3f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
