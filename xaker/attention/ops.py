"""Free-function helpers for attention modules.

These are stateless single-purpose functions used by :class:`Fused` and
:class:`Xsa`. They live outside any class to keep the attention modules
focused on module-level concerns.
"""

from __future__ import annotations

import torch


def zerodiag(kernel: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``kernel`` with the main diagonal zeroed.

    Args:
        kernel: Kernel matrix of shape ``(batch, heads, n, n)``.

    Returns:
        Tensor with ``kernel[..., i, i] == 0`` for every ``i``;
        off-diagonal entries unchanged. Does not modify ``kernel`` in place.
    """
    n = kernel.shape[-1]
    eye = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
    eye = eye.view(1, 1, n, n)
    return kernel * (1.0 - eye)


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize ``x`` by its per-(batch, head) root-mean-square.

    Args:
        x: Tensor of shape ``(batch, heads, n, headdim)``.
        eps: Numerical-stability floor inside the square root.

    Returns:
        Tensor with the same shape and dtype as ``x``.
    """
    rms_val = torch.sqrt((x * x).mean(dim=(-2, -1), keepdim=True) + eps)
    return x / rms_val