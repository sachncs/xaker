"""Preconditioned Conjugate Gradient (PCG) and Richardson iterations.

Both functions apply the regularized operator ``A(x) = Kx + lambda x``
in batches. :func:`pcg` implements the PCG-style recurrence associated in
this repository with LAKER, and :func:`richardson` implements a fixed
number of preconditioned Richardson updates.

Both functions return a :class:`Solve` dataclass so callers can observe
convergence status (``converged``, ``iters``, ``res``, ``history``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from xaker.solver.func import op


@dataclass
class Solve:
    """Outcome of a PCG or Richardson solve.

    Attributes:
        x: Final iterate.
        iters: Iterations actually executed.
        converged: True iff the relative residual fell below ``tol``
            (PCG only; Richardson always reports ``converged=False``).
        res: Final relative residual norm.
        history: Per-iteration relative residual (including initial).
    """

    x: torch.Tensor
    iters: int
    converged: bool
    res: float
    history: List[float] = field(default_factory=list)


def pcg(
    kernel: torch.Tensor,
    b: torch.Tensor,
    lam: torch.Tensor,
    precond_data=None,
    apply_pre: Optional[Callable] = None,
    iters: int = 50,
    tol: float = 1e-3,
    miniters: int = 3,
    x0: Optional[torch.Tensor] = None,
) -> Solve:
    """Solve ``(K + lambda I) x = b`` by PCG.

    Args:
        kernel: Kernel matrix of shape ``(batch, heads, n, n)``.
        b: Right-hand side of shape ``(batch, heads, n, ...)``.
        lam: Ridge regulariser broadcastable to kernel's diagonal.
        precond_data: Payload from ``Make(config).build(...)``; may
            be ``None`` for the unpreconditioned fallback.
        apply_pre: Callable ``apply_pre(r, precond_data) -> Tensor``
            implementing ``P(r)``. ``None`` falls back to identity.
        iters: Maximum iteration count.
        tol: Relative residual convergence threshold.
        miniters: Minimum iterations before the residual is checked.
        x0: Optional warm start; ``None`` starts from zero.

    Returns:
        :class:`Solve` with the final iterate, iteration count,
        convergence flag, residual norm, and per-iteration history.
    """
    if x0 is not None:
        x = x0
    else:
        x = torch.zeros_like(b)

    ax = op(kernel, x, lam)
    r = b - ax

    bnorm = torch.sqrt((b * b).sum(dim=(-2, -1), keepdim=True))

    if apply_pre is not None and precond_data is not None:
        z = apply_pre(r, precond_data)
    else:
        z = r

    p = z
    rz_old = (r * z).sum(dim=(-2, -1), keepdim=True)
    history: List[float] = []

    converged = False
    iters_done = 0
    for iteration in range(iters):
        ap = op(kernel, p, lam)
        pap = (p * ap).sum(dim=(-2, -1), keepdim=True)
        delta = rz_old / (pap + 1e-12)
        x = x + delta * p
        r = r - delta * ap
        iters_done = iteration + 1

        if iteration >= miniters - 1:
            rnorm = torch.sqrt((r * r).sum(dim=(-2, -1), keepdim=True))
            relres = rnorm / (bnorm + 1e-12)
            history.append(float(relres.mean().item()))
            if (relres < tol).all() and iteration >= miniters:
                converged = True
                break
        else:
            history.append(float(torch.sqrt((r * r).sum(dim=(-2, -1))).mean().item()
                                 / (torch.sqrt((b * b).sum(dim=(-2, -1))).mean().item() + 1e-12)))

        if apply_pre is not None and precond_data is not None:
            z = apply_pre(r, precond_data)
        else:
            z = r

        rz_new = (r * z).sum(dim=(-2, -1), keepdim=True)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new
        x = torch.clamp(x, -1e6, 1e6)

    final_res = history[-1] if history else float("nan")
    return Solve(x=x, iters=iters_done, converged=converged, res=final_res, history=history)


def richardson(
    kernel: torch.Tensor,
    b: torch.Tensor,
    lam: torch.Tensor,
    precond_data=None,
    apply_pre: Optional[Callable] = None,
    iters: int = 10,
    omega: float = 1.0,
) -> Solve:
    """Solve ``(K + lambda I) x = b`` by preconditioned Richardson iteration.

    Args:
        kernel: Kernel matrix of shape ``(batch, heads, n, n)``.
        b: Right-hand side of shape ``(batch, heads, n, ...)``.
        lam: Ridge regulariser broadcastable to kernel's diagonal.
        precond_data: Payload from ``Make(config).build(...)``; may
            be ``None``.
        apply_pre: Callable ``apply_pre(r, precond_data) -> Tensor``; ``None``
            falls back to identity.
        iters: Number of fixed updates to perform.
        omega: Scalar step size.

    Returns:
        :class:`Solve` with the iterate and per-iteration history.
        ``converged`` is always ``False`` (no residual test).
    """
    x = torch.zeros_like(b)
    history: List[float] = []
    for i in range(iters):
        ax = op(kernel, x, lam)
        residual = b - ax
        if apply_pre is not None and precond_data is not None:
            update = apply_pre(residual, precond_data)
        else:
            update = residual
        x = x + omega * update
        x = torch.clamp(x, -1e6, 1e6)
        history.append(float(torch.sqrt((residual * residual).sum(dim=(-2, -1))).mean().item()
                             / max(float(torch.sqrt((b * b).sum(dim=(-2, -1))).mean().item()), 1e-12)))
    final_res = history[-1] if history else float("nan")
    return Solve(x=x, iters=iters, converged=False, res=final_res, history=history)
