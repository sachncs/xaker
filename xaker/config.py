"""Configuration values shared by XAKER attention and solver modules.

The :class:`Config` dataclass groups shape, kernel, exclusion,
preconditioner, and iterative-solver settings. Every field is single-word.

Training uses the separate :class:`xaker.training.trainer.Fit`. Seeding

:meth:`Config.__post_init__` derives ``headdim`` from ``dim`` and ``heads``,
and validates categorical fields and a few numeric ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class Config:
    """Single-word hyperparameter dataclass for XAKER.

    Construction runs :meth:`__post_init__`, which fills in the default
    ``headdim`` and validates the categorical fields plus a small number
    of numeric ranges. It does not validate every field.

    Three concerns:

    1. *Attention shape* (``dim``, ``heads``, ``headdim``, ``drop``).
    2. *Self-exclusion and kernel formulation* (``mode``, ``kernel``,
       ``temp``, ``symmetric``, ``normalize``).
    3. *Linear-system solve and preconditioning* (``lam``, ``precond``
       and its parameters, ``pcg``, ``tol``, ``freq``).
    """

    dim: int
    heads: int
    headdim: Optional[int] = None
    drop: float = 0.0
    eps: float = 1e-6
    lam: float = 3.0

    kernel: Literal["exp", "rbf", "linear", "cosine"] = "exp"
    mode: Literal["subtract", "zero", "mask"] = "subtract"

    precond: Literal["cccp", "fast", "diagonal", "identity"] = "fast"
    rank: Optional[int] = 32
    directions: int = 64
    iters: int = 20
    gamma: float = 0.1
    rho: float = 0.01
    eps_shrink: float = 1e-8

    pcg: int = 20
    tol: float = 1e-2

    freq: int = 1

    temp: float = 1.0
    symmetric: bool = False
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate and finalize the configuration."""
        if self.headdim is None:
            self.headdim = self.dim // self.heads

        if self.dim % self.heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by heads ({self.heads})"
            )

        valid_kernels = ("exp", "rbf", "linear", "cosine")
        if self.kernel not in valid_kernels:
            raise ValueError(
                f"kernel must be one of {valid_kernels}, got '{self.kernel}'"
            )

        valid_mode = ("subtract", "zero", "mask")
        if self.mode not in valid_mode:
            raise ValueError(
                f"mode must be one of {valid_mode}, got '{self.mode}'"
            )

        valid_precond = ("cccp", "fast", "diagonal", "identity")
        if self.precond not in valid_precond:
            raise ValueError(
                f"precond must be one of {valid_precond}, got '{self.precond}'"
            )

        if self.pcg < 1:
            raise ValueError("pcg must be >= 1")
        if self.drop < 0.0 or self.drop > 1.0:
            raise ValueError("drop must be in [0,1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.lam < 0:
            raise ValueError("lam must be non-negative")