"""Fused XSA + LAKER attention.

Pipeline (per head):

1. Project to Q, K, V in the base class.
2. Compute ``K = exp(sim(q_i, k_j) / temperature)``.
3. Apply external mask (multiplicative if numeric).
4. If no mask: zero the kernel diagonal (XSA diagonal removal).
5. ``lambda = softplus(raw_lambda) + eps`` for positivity.
6. Build or reuse the configured preconditioner.
7. Apply :func:`pcg` to ``(K + lambda I) alpha = V``. Check convergence.
8. Clamp alpha to ``[-BOUND, BOUND]`` and RMS-normalize.
9. If mode is ``subtract``, subtract a scaled projection of alpha onto V.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, cast

import torch
from torch import nn
from torch.nn.functional import softplus

from xaker.config import Config
from xaker.attention.core import Base, heads, merge, keep
from xaker.attention.kernel import Kernel
from xaker.attention.ops import rms, zerodiag
from xaker.attention.xsa import XsaStrategy
from xaker.solver.cg import pcg
from xaker.solver.precond import BOUND, Make

logger = logging.getLogger(__name__)


class Laker(Base):
    """Fused XSA + LAKER attention (v2).

    Attributes:
        kernel_fn: :class:`Kernel` producing the exponential kernel.
        precon: preconditioner strategy from :func:`Make`.
        raw_lambda: ``nn.Parameter`` backing :attr:`lam`.
        xsa_scale: ``nn.Parameter`` for projection-removal strength.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.kernel_fn = Kernel(
            headdim=self.headdim,
            temp=config.temp,
            symmetric=config.symmetric,
            learnable=True,
            normalize_qk=config.normalize,
            eps=config.eps,
        )
        self.precon = Make(config)
        self.raw_lambda = nn.Parameter(torch.tensor(config.lam))
        # Always trainable (per Phase J decision).
        self.xsa_scale = nn.Parameter(torch.ones(1))
        self.xsa = XsaStrategy(config, self.xsa_scale)

    def init(self) -> None:
        """Initialize Q/K/V/output projections with a Gaussian."""
        std = 0.02 / math.sqrt(2.0)
        for proj in [
            self.qkv_proj.w_q,
            self.qkv_proj.w_k,
            self.qkv_proj.w_v,
            self.w_o,
        ]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    @property
    def lam(self) -> torch.Tensor:
        """``softplus(raw_lambda) + eps``. Positive scalar."""
        return softplus(self.raw_lambda) + self.config.eps

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        m: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute fused XSA + LAKER attention per head.

        Steps:
        1. Compute the exponential kernel matrix.
        2. Apply external m; if no m, zero the kernel diagonal.
        3. Build (or reuse) the preconditioner payload.
        4. Call :func:`pcg`; check convergence.
        5. Clamp and RMS-normalize.
        6. Apply XSA strategy's output transform.
        """
        _, _, length, _ = q.shape
        kernel = self.kernel_fn(q, k)
        kernel = keep(kernel, m, fill=-1e9) if m is not None else zerodiag(kernel)
        lam = self.lam.view(1, 1, 1, 1)
        # Update preconditioner cache step counter (Fast uses it)
        data = self.precon.build(kernel, lam, length)
        solve = pcg(
            kernel=kernel,
            b=v,
            lam=lam,
            precond_data=data,
            apply=self.precon.apply,
            iters=self.config.pcg,
            tol=self.config.tol,
            miniters=3,
        )
        # If PCG did not converge and is non-finite, fall back to dense solve.
        if not solve.converged or not torch.isfinite(solve.x).all():
            logger.warning(
                "PCG did not converge; falling back to direct solve. iters=%d res=%.4f",
                solve.iters, solve.res,
            )
            eye = torch.eye(length, device=kernel.device, dtype=kernel.dtype)
            eye = eye.view(1, 1, length, length)
            solve_x = torch.linalg.solve(kernel + lam * eye, v)
        else:
            solve_x = solve.x
        out = torch.clamp(solve_x, -BOUND, BOUND)
        out = rms(out, self.config.eps)
        # Bump Fast preconditioner step counter
        if hasattr(self.precon, "iter"):
            self.precon.iter.add_(1)
        return self.xsa.apply(out, v)