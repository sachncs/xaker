"""Linear-system operators, iterative routines, and preconditioners.

The attention path uses these modules with the regularized operator
``A(alpha) = K alpha + lambda alpha``.

Public surface:
- :func:`pcg`, :func:`richardson` — iterative solvers returning :class:`Solve`
- :class:`Solve` — outcome dataclass (x, iters, converged, res, history)
- :func:`Make` — preconditioner factory
- :class:`Identity`, :class:`Diagonal`, :class:`Fast`, :class:`Cccp` — strategies
- :class:`Cache` — payload dataclass for preconditioner state
"""

from __future__ import annotations

from xaker.solver.cg import Solve, pcg, richardson
from xaker.solver.precond import (
    BOUND,
    Cache,
    Cccp,
    Diagonal,
    Fast,
    Identity,
    Make,
    PrecondProto,
)
from xaker.solver.func import op

__all__ = [
    "BOUND",
    "Cache",
    "Cccp",
    "Diagonal",
    "Fast",
    "Identity",
    "Make",
    "PrecondProto",
    "Solve",
    "op",
    "pcg",
    "richardson",
]