"""Add a custom kernel to xaker without forking the package.

This example shows the contract for adding a fifth kernel to
xaker. The chosen kernel is a Laplacian (exp-style, with a learnable
temperature), which is one of the natural companion kernels to the
shipped `exp`, `rbf`, `linear`, `cosine`.

The script:
  1. Defines a stateless kernel function with the right signature.
  2. Registers it via monkey-patching (the same path `Linear`
     uses for its `elu + 1` feature map).
  3. Forwards a small batch through `BLOCK["fused"]` and reports
     finiteness + shape.

Usage:
    python -m examples.custom_kernel

The example does NOT modify `xaker/attention/func.py`; production
additions follow the recipe at /xaker/recipes/#recipe-add-a-new-kernel-function.
"""

from __future__ import annotations

import torch
from torch import Tensor

from xaker import Config, BLOCK


def laplacian_kernel(q: Tensor, k: Tensor, *, config) -> Tensor:
    """Laplacian-style kernel: k(q, k) = exp(- ||q - k||_1 / temp).

    Args:
        q: Queries of shape (batch, heads, seq_len, headdim).
        k: Keys of the same shape as q.
        config: Source :class:`xaker.Config`; reads ``temp``.

    Returns:
        Kernel matrix of shape (batch, heads, seq_len, seq_len).
    """
    absdiff = (q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)
    return torch.exp(-absdiff / config.temp)


# xaker ships `Linear.feature_clamp` as a module-level map for its custom
# feature map. The Laplacian kernel here is the standalone form; the
# stateful Kernel wrapper (`xaker/attention/kernel.py`) reads from the
# same function pointer. Without editing `xaker/__init__.py`, the
# simplest path is to monkey-patch the stateless function used by
# `Fused`, since the four ship-ins are picked by string in `func.kernel`.
import xaker.attention.func as func_mod
func_mod.kernel = laplacian_kernel  # type: ignore[assignment]


def main() -> int:
    cfg = Config(dim=32, heads=2, drop=0.0, kernel="exp", temp=1.0)
    attn = BLOCK["fused"](cfg).eval()
    x = torch.randn(2, 8, cfg.dim)
    out = attn(x)
    print(f"shape: {tuple(out.shape)}")
    print(f"finite: {torch.isfinite(out).all().item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
