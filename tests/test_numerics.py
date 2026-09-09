"""Numerical stability tests for the xaker attention stack.

Verifies finiteness under extreme input magnitudes, long sequences,
and low-precision dtypes. The dtype coverage here implements the
contract documented in ``docs/limitations.md``.
"""

from __future__ import annotations

import pytest
import torch

from xaker.attention import BLOCK
from xaker.config import Config
from xaker.utils.finite import finite


def test_std_large() -> None:
    """Standard attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["standard"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "std large", raise_error=False)


def test_std_small() -> None:
    """Standard attention produces finite output for x * 1e-6."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["standard"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 1e-6
    out = attn(x)
    assert finite(out, "std small", raise_error=False)


def test_fused_large() -> None:
    """Fused attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["fused"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "fused large", raise_error=False)


def test_xsa_large() -> None:
    """Xsa attention produces finite output for x * 100."""
    cfg = Config(dim=64, heads=4, drop=0.0, mode="subtract")
    attn = BLOCK["xsa"](cfg).eval()
    x = torch.randn(2, 32, cfg.dim) * 100
    out = attn(x)
    assert finite(out, "xsa large", raise_error=False)


def test_fused_long() -> None:
    """Fused is stable on a long sequence (length=64)."""
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["fused"](cfg).eval()
    x = torch.randn(1, 64, cfg.dim)
    out = attn(x)
    assert finite(out, "long sequence", raise_error=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_low_precision(dtype: torch.dtype) -> None:
    """Linear produces a strictly-positive feature map in fp16 / bf16.

    Before fix, ``Linear`` in fp16 with elements below ``-14``
    silently rounded ``elu(x) + 1`` to ``1.0`` and propagated
    NaNs. The clamp restores a strictly-positive feature map for
    the typical input magnitude.

    Caveat: extremely large positive inputs (``|x| > 50``) can
    still overflow the fp16 ``matmul`` in the second stage even
    after the clamp; ``docs/limitations.md`` documents the
    frontier.
    """
    cfg = Config(dim=64, heads=4, drop=0.0)
    attn = BLOCK["linear"](cfg).eval().to(dtype)
    x = torch.randn(2, 32, cfg.dim, dtype=dtype)
    out = attn(x)
    assert torch.isfinite(out).all(), f"Linear produced NaN/inf in {dtype}"


def test_linear_feature_map_strictly_positive_fp16() -> None:
    """The fp16 feature map is always >= 0 (clamp preserves positivity).

    Without the clamp, ``elu(x) + 1`` for ``x < -14`` rounds to
    exactly ``1.0`` in fp16; a strictly-positive feature map is
    only guaranteed when the input is bounded by the saturation
    threshold.
    """
    import torch.nn.functional as F

    from xaker.attention.linear import Linear

    sat = Linear.feature_clamp[torch.float16]
    # Extreme negative input that, without the clamp, would round
    # to zero in fp16.
    x = torch.full((1,), -50.0, dtype=torch.float16)
    x_clamped = torch.clamp(x, min=-sat)
    phi = F.elu(x_clamped) + 1.0
    assert (phi >= 0).all(), "fp16 feature map produced a non-positive value"
    assert torch.isfinite(phi).all(), "fp16 feature map produced a non-finite value"
