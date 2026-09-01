"""Exponential attention-kernel implementation used by LAKER v2.

The module computes ``exp(similarity / temp) + eps`` with optional Q/K
normalization and optional symmetric averaging. Scores are clamped before
exponentiation, but ``exp(100)`` can still overflow in lower-precision dtypes.
The constructor does not validate temp or ``eps``; non-positive
initial temperatures fail in ``math.log`` or can produce non-finite state, and
negative ``eps`` can invalidate entrywise positivity.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

class Kernel(nn.Module):
    """Exponential attention kernel used by ``Laker``.

    Computes ``exp(sim(q_i, k_j) / temp)``, optionally averages the
    result with its transpose, then adds ``eps``. Symmetry and entrywise
    positivity do not imply positive semidefiniteness. Entrywise positivity
    itself requires a non-negative-enough ``eps`` and finite exponentiation.

    Two similarity modes are supported and mirror the classical attention
    variants:

    * ``normalize_qk=True`` (default) — Q and K are L2-normalised along the
      feature dimension, so scores are cosine similarities in ``[-1, 1]``.
    * ``normalize_qk=False`` — raw scaled-dot scores, divided by
      ``sqrt(headdim)`` to match Vaswani-style attention scaling.

    Attributes:
        headdim: Feature dimension per attention head.
        symmetric: When ``True``, returns ``(K + K^T) / 2``. This requires a
            square score matrix and enforces symmetry, not positive
            semidefiniteness.
        normalize_qk: When ``True``, L2-normalise Q/K before scoring.
        eps: Scalar added to every kernel entry. The constructor does not
            validate its sign.
        logtemp: Log-domain learnable (or buffer) temp used to
            enforce ``temp ∈ [0.05, 100]`` via the :attr:`temp`
            property.
    """

    def __init__(
        self,
        headdim: int,
        temp: float = 1.0,
        symmetric: bool = False,
        learnable: bool = True,
        normalize_qk: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """Initialise the kernel.

        Args:
            headdim: Per-head feature dimension; required for the
                scaled-dot branch and stored on the module.
            temp: Initial value passed to ``math.log``. It must be
                positive for normal operation; the constructor relies on
                ``math.log`` rather than validating it explicitly.
            symmetric: Whether to average ``K`` with its transpose.
            learnable: When ``True``, store temp as an
                ``nn.Parameter``; otherwise it is a non-trainable buffer.
            normalize_qk: When ``True``, use cosine similarity; otherwise
                use scaled dot-product.
            eps: Scalar added to the output; its sign is not validated.

        Raises:
            ValueError: If ``temp`` is zero or negative and
                ``math.log`` rejects it.

        Side Effects:
            Allocates :attr:`logtemp` as an ``nn.Parameter`` when
            learnable, otherwise as a registered buffer. The stored value is
            ``math.log(temp)``.
        """
        super().__init__()
        self.headdim = headdim
        self.symmetric = symmetric
        self.normalize_qk = normalize_qk
        self.eps = eps

        if learnable:
            self.logtemp = nn.Parameter(torch.tensor(math.log(temp)))
        else:
            self.register_buffer("logtemp", torch.tensor(math.log(temp)))

    @property
    def temp(self) -> torch.Tensor:
        """Effective temp derived from :attr:`logtemp`.

        The clamp ``[0.05, 100]`` bounds the effective temp. It does not
        modify the underlying ``logtemp`` parameter.

        Returns:
            Scalar 0-d :class:`torch.Tensor` of dtype matching
            ``logtemp``, equal to
            ``exp(logtemp).clamp(min=0.05, max=100.0)``.
        """
        return torch.exp(self.logtemp).clamp(min=0.05, max=100.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute ``K_{ij} = exp(sim(q_i, k_j) / temp)``.

        Args:
            q: Queries of shape ``(batch, heads, seq_len, headdim)``.
            k: Keys of shape ``(batch, heads, seq_len, headdim)``.

        Returns:
            Kernel matrix of shape
            ``(batch, heads, query_len, key_len)``. Symmetric mode requires
            equal query and key lengths.

        Numerical Notes:
            With normalized finite Q/K and positive ``eps``, mathematical
            pre-rounding values lie in
            ``[exp(-1/T) + eps, exp(1/T) + eps]``. In the raw-dot branch,
            clamping scores to ``100`` does not prevent ``exp`` overflow in
            dtypes whose finite logarithmic range is smaller (for example,
            float32). Non-finite inputs are not sanitized.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul`,
                :func:`torch.exp`, or :func:`torch.nn.functional.normalize`
                for incompatible shapes, dtypes, or devices.

        Side Effects:
            None; the module is stateless apart from the parameter/buffer.
        """
        temp = self.temp

        if self.normalize_qk:
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1))
        else:
            scale = 1.0 / math.sqrt(self.headdim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        scores = scores / temp
        scores = torch.clamp(scores, -100.0, 100.0)
        kernel = torch.exp(scores)

        if self.symmetric:
            kernel = 0.5 * (kernel + kernel.transpose(-2, -1))

        return kernel + self.eps

__all__ = ["Kernel"]

def kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    normalize_qk: bool = True,
    symmetric: bool = False,
    temp: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stateless helper retained for legacy imports of ``kernel``.

    New code should instantiate :class:`Kernel` directly (it is stateful
    and supports a learnable temp). This shim constructs a non-learnable
    :class:`Kernel` with the requested ``temp``/``eps`` and
    invokes it once.

    Args:
        q: Queries with shape ``(..., query_len, headdim)``.
        k: Keys with shape ``(..., key_len, headdim)``. Symmetric mode
            requires ``key_len == query_len``.
        normalize_qk: Cosine (default) or scaled-dot similarity.
        symmetric: Average the matrix with its transpose when ``True``;
            enforces symmetry but not positive semidefiniteness.
        temp: Scalar passed to ``math.log`` at construction.
        eps: Scalar added to every result entry.

    Returns:
        Kernel matrix with shape ``(..., query_len, key_len)``.

    Raises:
        ValueError: If ``temp`` is invalid for ``math.log``.
        RuntimeError: Propagated from the underlying
            :class:`Kernel` for incompatible shapes, dtypes,
            or devices.
    """
    headdim = int(q.shape[-1])
    kernel = Kernel(
        headdim=headdim,
        temp=temp,
        symmetric=symmetric,
        learnable=False,
        normalize_qk=normalize_qk,
        eps=eps,
    )
    return cast(torch.Tensor, kernel(q, k))
