"""Typed execution context (device, dtype).

Every tensor-creating helper in :mod:`xaker.cli`, :mod:`xaker.bench`, and
:mod:`xaker.rubric` takes a :class:`Ctx` rather than re-resolving the device
inside module-level ``if torch.cuda.is_available()`` literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Ctx:
    """Device + dtype for tensor operations.

    Attributes:
        device: ``torch.device`` for all tensor allocations.
        dtype: Floating-point dtype used by the bench/rubric drivers.
    """

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def resolve(self, requested: str | torch.device | None = None) -> torch.device:
        """Resolve a requested device string against CUDA availability.

        Args:
            requested: ``"cuda"``, ``"cpu"``, or a fully-specified
                ``torch.device``. ``None`` falls back to CUDA if available,
                else CPU.
        """
        if requested is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(requested, torch.device):
            return requested
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)


def toctx(t: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    """Move ``t`` to the device and dtype of ``ctx``."""
    return t.to(device=ctx.device, dtype=ctx.dtype)
