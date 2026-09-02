"""Preconditioner strategies for the LAKER kernel solve.

Public surface:
- :func:`Make` -- factory function (single entry point)
- :class:`Identity`, :class:`Diagonal`, :class:`Fast`, :class:`Cccp` -- concrete
  ``nn.Module`` strategies
- :class:`Cache` -- payload dataclass returned by ``.build()``
- :class:`PrecondProto` -- typing Protocol for static checks

The factory dispatches on ``config.precond`` and constructs the configured
strategy. Each strategy is a real ``nn.Module`` that implements
``build(kernel, lam, length) -> Cache`` and ``apply_pre(residual, data) -> Tensor``.

There is exactly one public symbol named ``Make``. Strategies have their
own single-word names. No aliasing, no shims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union, cast, runtime_checkable

import torch
from torch import nn


BOUND = 1e6


CacheData = Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], None]


@dataclass
class Cache:
    """Preconditioner payload returned by ``.build()`` and consumed by ``.apply_pre()``."""

    data: CacheData = None


@runtime_checkable
class PrecondProto(Protocol):
    """Typing Protocol: a strategy exposes ``build`` and ``apply_pre``."""

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache: ...
    def apply_pre(self, residual: torch.Tensor, data: Cache) -> torch.Tensor: ...


class Identity(nn.Module):
    """No preconditioning."""

    def __init__(self, config) -> None:
        """Build an ``Identity`` strategy.

        Args:
            config: Source :class:`Config` (unused).
        """
        super().__init__()

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        """Return an empty preconditioner payload.

        Args:
            kernel: Regularised kernel matrix (unused).
            lam: Ridge regulariser (unused).
            length: Sequence length (unused).

        Returns:
            :class:`Cache` with ``data=None``.
        """
        return Cache(data=None)

    def apply_pre(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        """Return ``residual`` unchanged.

        Args:
            residual: PCG residual.
            data: Preconditioner payload (unused).

        Returns:
            ``residual`` unchanged.
        """
        return residual


class Diagonal(nn.Module):
    """Jacobi-style diagonal preconditioner."""

    def __init__(self, config) -> None:
        """Build a learnable diagonal scale.

        Args:
            config: Source :class:`Config`; reads ``heads`` and ``eps``.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, config.heads, 1))
        self.eps = config.eps

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        """Extract a per-row diagonal scaling.

        Args:
            kernel: Kernel matrix of shape
                ``(batch, heads, n, n)``.
            lam: Ridge regulariser broadcastable to kernel's diagonal.
            length: Sequence length (unused).

        Returns:
            :class:`Cache` whose ``data`` is the per-row inverse scale
            of shape ``(batch, heads, n)``.
        """
        diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        out = torch.nn.functional.softplus(diag + lam.squeeze(-1).squeeze(-1))
        out = out * self.scale.abs() + self.eps
        return Cache(data=out)

    def apply_pre(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        """Apply the diagonal preconditioner.

        Args:
            residual: PCG residual of shape
                ``(batch, heads, n, ...)``.
            data: :class:`Cache` from :meth:`build`.

        Returns:
            Element-wise scaled residual.
        """
        diag = data.data
        assert isinstance(diag, torch.Tensor)
        return residual * diag.unsqueeze(-1)


class Fast(nn.Module):
    """Low-rank-plus-diagonal preconditioner with step-counter cache."""

    def __init__(self, config) -> None:
        """Build the learnable low-rank factors and scale.

        Args:
            config: Source :class:`Config`; reads ``heads``, ``rank``,
                ``eps``, ``freq``, and the maximum supported
                ``seq_len`` of ``2048``.

        Side Effects:
            Registers ``self.iter`` as a zero-initialised buffer for
            step counting and ``self.cache`` for the latest payload.
        """
        super().__init__()
        if config.rank is None or config.rank <= 0:
            self.register_buffer("lr_base", torch.empty(config.heads, 0, 0))
            self.register_buffer("lr_imp", torch.empty(config.heads, 0))
            self.rank = 0
        else:
            self.lr_base = nn.Parameter(
                torch.randn(config.heads, 2048, config.rank) * 0.01
            )
            self.lr_imp = nn.Parameter(torch.zeros(config.heads, config.rank))
            self.rank = config.rank
        self.scale = nn.Parameter(torch.ones(1, config.heads, 1))
        self.eps = config.eps
        self.freq = config.freq
        self.register_buffer("iter", torch.zeros(1, dtype=torch.long))
        self.cache: Cache | None = None
        self.iter: torch.Tensor

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        if (
            self.cache is not None
            and self.freq > 0
            and int(self.iter[0]) % self.freq != 0
        ):
            return self.cache
        diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag + lam.squeeze(-1).squeeze(-1))
        diag = diag * self.scale.abs() + self.eps
        if self.rank > 0:
            lr = self.lr_base[:, :length, :] * torch.nn.functional.softplus(self.lr_imp).unsqueeze(1)
            lr = lr.unsqueeze(0).expand(kernel.shape[0], -1, -1, -1)
        else:
            lr = None
        payload: CacheData = (diag, lr)
        self.cache = Cache(data=payload)
        return self.cache

    def apply_pre(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        payload = data.data
        assert isinstance(payload, tuple)
        diag, lr = payload
        out = residual * diag.unsqueeze(-1)
        if lr is not None:
            ut_r = torch.matmul(lr.transpose(-2, -1), residual)
            out = out + torch.matmul(lr, ut_r)
        return out


class Cccp(nn.Module):
    """CCCP-based angular-sampling preconditioner."""

    def __init__(self, config) -> None:
        """Build a CCCP preconditioner.

        Args:
            config: Source :class:`Config`; reads ``eps``,
                ``eps_shrink``, ``gamma``, ``rho``, ``directions``,
                and ``iters``.
        """
        super().__init__()
        self.eps = config.eps
        self.eps_shrink = config.eps_shrink
        self.gamma = config.gamma
        self.rho = config.rho
        self.N_r = config.directions
        self.iters = config.iters

    def samples(self, kernel: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Generate ``N_r`` angular samples from the kernel matrix.

        Args:
            kernel: Kernel matrix of shape
                ``(batch, heads, n, n)``.
            lam: Ridge regulariser broadcastable to kernel's diagonal.

        Returns:
            Angular direction samples used to fit Tyler's M-estimator.
        """
        batch, heads, n, _ = kernel.shape
        device, dtype = kernel.device, kernel.dtype
        out = []
        for _ in range(self.N_r):
            z = torch.randn(batch, heads, n, 1, device=device, dtype=dtype)
            kz = torch.matmul(kernel, z)
            u = kz + lam * z
            norm = torch.sqrt(torch.sum(u * u, dim=-2, keepdim=True))
            ubar = u / (norm + self.eps_shrink)
            out.append(ubar.squeeze(-1))
        return torch.stack(out, dim=0)

    def step(self, ubar: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """One CCCP fixed-point update for Tyler's M-estimator.

        Args:
            ubar: Direction samples of shape ``(N_r, batch, heads, n)``.
            sigma: Current covariance estimate of shape
                ``(batch, heads, n, n)``.

        Returns:
            Updated ``Sigma`` estimate, shape
            ``(batch, heads, n, n)``.
        """
        batch, heads, n, _ = sigma.shape
        device, dtype = sigma.device, sigma.dtype
        sigma_inv = torch.linalg.inv(sigma)
        denom_sum = torch.zeros(batch, heads, n, n, device=device, dtype=dtype)
        for k in range(ubar.shape[0]):
            u = ubar[k]
            su = torch.matmul(sigma_inv, u.unsqueeze(-1)).squeeze(-1)
            denom = (u * su).sum(dim=-1)
            outer = torch.einsum("...i,...j->...ij", u, u)
            denom_sum = denom_sum + outer / (denom.unsqueeze(-1).unsqueeze(-1) + self.eps_shrink)
        scale = n / ubar.shape[0]
        eye = torch.eye(n, device=device, dtype=dtype)
        f = scale * denom_sum + self.gamma * eye.unsqueeze(0).unsqueeze(0)
        f = f / (1.0 + self.gamma / n)
        st = (1.0 - self.rho) * f + self.rho * eye.unsqueeze(0).unsqueeze(0)
        trace = torch.diagonal(st, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        out: torch.Tensor = st * n / (trace.unsqueeze(-1) + self.eps_shrink)
        return out

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        batch, heads, n, _ = kernel.shape
        device, dtype = kernel.device, kernel.dtype
        ubar = self.samples(kernel, lam)
        eye = torch.eye(n, device=device, dtype=dtype)
        sigma = eye.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1).clone()
        for _ in range(self.iters):
            sigma = self.step(ubar, sigma)
        eigvals_raw, eigvecs = torch.linalg.eigh(sigma)
        eigvals = torch.clamp(eigvals_raw, min=self.eps)
        inv_sqrt = eigvals.pow(-0.5)
        p = eigvecs @ (inv_sqrt.unsqueeze(-1) * eigvecs.transpose(-2, -1))
        return Cache(data=p)

    def apply_pre(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        p = data.data
        assert isinstance(p, torch.Tensor)
        return torch.matmul(p, residual)


MODE = {
    "identity": Identity,
    "diagonal": Diagonal,
    "fast": Fast,
    "cccp": Cccp,
}


def Make(config) -> Union[Identity, Diagonal, Fast, Cccp]:
    """Build the preconditioner strategy selected by ``config.precond``."""
    cls = MODE[config.precond]
    assert cls is not None
    return cast(Union[Identity, Diagonal, Fast, Cccp], cls(config))
