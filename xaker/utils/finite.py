"""Standalone finite-value detection.

The current attention and solver implementations use their own
local clamps; this module exposes the finite-check helper for
external callers that want explicit numerical verification.
"""

from __future__ import annotations

import torch


def finite(
    x: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = True,
) -> bool:
    """Check that a tensor contains only finite values.

    The check is a single :func:`torch.isfinite` reduction across
    the entire tensor. On failure, the function can either raise
    a :class:`ValueError` with a precise breakdown (number of
    ``NaN``s, ``+Inf``s, and ``-Inf``s) or simply return
    ``False`` for non-raising callers. The breakdown is computed
    only on the error path, so the hot path is a single reduction
    regardless of the answer.

    Args:
        x: Tensor to check. Any dtype is accepted, but the check
            is most meaningful for floating-point tensors; integer
            tensors are always finite.
        name: Human-readable name used in error messages to
            identify which tensor failed. Defaults to
            ``"tensor"``.
        raise_error: When ``True`` (the default), raise
            :class:`ValueError` on the first non-finite sample.
            When ``False``, return ``False`` and stay silent -
            useful for callers that prefer to log and continue.

    Returns:
        ``True`` if every entry of ``x`` is finite. ``False`` if
        any entry is non-finite and ``raise_error`` is ``False``.

    Raises:
        ValueError: If ``raise_error`` is ``True`` and ``x``
            contains at least one ``NaN``, ``+Inf``, or
            ``-Inf``. The message reports the per-kind counts
            and the tensor's shape.

    Example:
        >>> x = torch.randn(10, 10)
        >>> finite(x)  # Returns True
        >>> x[0, 0] = float('inf')
        >>> finite(x, raise_error=False)  # Returns False
    """
    is_finite = torch.isfinite(x).all()

    if not is_finite:
        if raise_error:
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isposinf(x).sum().item()
            neg_inf_count = torch.isneginf(x).sum().item()

            raise ValueError(
                f"{name} contains non-finite values: "
                f"{nan_count} NaNs, {inf_count} +Infs, {neg_inf_count} -Infs. "
                f"Shape: {x.shape}"
            )
        return False

    return True
