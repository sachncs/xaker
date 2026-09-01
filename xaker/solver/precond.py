"""Preconditioner strategies for the LAKER kernel solve.

Public surface:
- :func:`Make` — factory function (single entry point)
- :class:`Identity`, :class:`Diagonal`, :class:`Fast`, :class:`Cccp` — concrete
  ``nn.Module`` strategies
- :class:`Cache` — payload dataclass returned by ``.build()``
- :class:`PrecondProto` — typing Protocol for static checks

The factory dispatches on ``config.precond`` and constructs the configured
strategy. Each strategy is a real ``nn.Module`` that implements
``build(kernel, lam, length) -> Cache`` and ``apply(residual, data) -> Tensor``.

There is exactly one public symbol named ``Make``. Strategies have their
own single-word names. No aliasing, no shims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


BOUND = 1e6


@dataclass
class Cache:
    """Preconditioner payload returned by ``.build()`` and consumed by `` ``.apply()``."""

    data: object = None


@runtime_checkable
class PrecondProto(Protocol):
    """Typing Protocol: a strategy exposes ``build`` and ``apply``."""

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache: ...
    def apply(self, residual: torch.Tensor, data: Cache) -> torch.Tensor: ...


class Identity(nn.Module):
    """No preconditioning."""

    def __init__(self, config) -> None:
        super().__init__()

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        return Cache(data=None)

    def apply(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        return residual


class Diagonal(nn.Module):
    """Jacobi-style diagonal preconditioner."""

    def __init__(self, config) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, config.heads, 1))
        self.eps = config.eps

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        out = torch.nn.functional.softplus(diag + lam.squeeze(-1).squeeze(-1))
        out = out * self.scale.abs() + self.eps
        return Cache(data=out)

    def apply(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        return residual * data.data.unsqueeze(-1)


class Fast(nn.Module):
    """Low-rank-plus-diagonal preconditioner with step-counter cache."""

    def __init__(self, config) -> None:
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
        self.cache = Cache(data=(diag, lr))
        return self.cache

    def apply(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        diag, lr = data.data
        out = residual * diag.unsqueeze(-1)
        if lr is not None:
            ut_r = torch.matmul(lr.transpose(-2, -1), residual)
            out = out + torch.matmul(lr, ut_r)
        return out


class Cccp(nn.Module):
    """CCCP-based angular-sampling preconditioner."""

    def __init__(self, config) -> None:
        super().__init__()
        self.eps = config.eps
        self.eps_shrink = config.eps_shrink
        self.gamma = config.gamma
        self.rho = config.rho
        self.N_r = config.directions
        self.iters = config.iters

    def samples(self, kernel: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Generate ``N_r`` angular samples.

        For k = 1..N_r:
            z_k ~ N(0, I); u_k = (lambda I + K) z_k; ubar_k = u_k / ||u_k||_2
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
        """One CCCP fixed-point update for Tyler's M-estimator."""
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
        return st * n / (trace.unsqueeze(-1) + self.eps_shrink)

    def build(self, kernel: torch.Tensor, lam: torch.Tensor, length: int) -> Cache:
        batch, heads, n, _ = kernel.shape
        device, dtype = kernel.device, kernel.dtype
        ubar = self.samples(kernel, lam)
        eye = torch.eye(n, device=device, dtype=dtype)
        sigma = eye.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1).clone()
        for _ in range(self.iters):
            sigma = self.step(ubar, sigma)
        eigvals, eigvecs = torch.linalg.eigh(sigma)
        eigvals = torch.clamp(eigvals, min=self.eps)
        inv_sqrt = eigvals.pow(-0.5)
        P = eigvecs @ (inv_sqrt.unsqueeze(-1) * eigvecs.transpose(-2, -1))
        return Cache(data=P)

    def apply(self, residual: torch.Tensor, data: Cache) -> torch.Tensor:
        return torch.matmul(data.data, residual)


MODE = {
    "identity": Identity,
    "diagonal": Diagonal,
    "fast": Fast,
    "cccp": Cccp,
}


def Make(config) -> nn.Module:
    """Build the preconditioner strategy selected by ``config.precond``."""
    cls = MODE[config.precond]
    return cls(config)