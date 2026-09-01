"""Exclusive Self Attention (XSA).

Three exclusion strategies (concrete classes):
- :class:`Projection` — subtract the projection of the output onto each token's value vector.
- :class:`Zero` — zero the score diagonal before softmax.
- :class:`Mask` — combine zero-diagonal with projection subtraction.

The factory :func:`XsaStrategy` selects one based on ``Config.mode``.
``scale`` is always ``nn.Parameter(torch.ones(1))`` allocated by the
caller (``Xsa``); strategies receive it via constructor and use it
(or ignore it, for :class:`Zero`).

Mode values: ``"subtract"`` (default), ``"zero"``, ``"mask"``.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from xaker.config import Config
from xaker.attention.core import Base, keep


class XsaMode(Protocol):
    """XSA strategy: modifies scores pre-softmax and/or output post-softmax."""

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor: ...
    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


class Projection:
    """Subtract the projection of ``output`` onto each token's value vector.

    Output: ``output - scale * (output · v) / (v · v + eps) * v``
    """

    def __init__(self, config: Config, scale: torch.Tensor) -> None:
        """Store ``scale`` and ``config.eps``.

        Args:
            config: Source :class:`Config`; reads ``eps``.
            scale: Learnable scalar ``nn.Parameter`` allocated by
                the owning :class:`Xsa` module.
        """
        self.scale = scale
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Identity; ``Projection`` only touches the post-softmax output.

        Args:
            scores: Pre-softmax attention scores.
            mask: Optional attention mask (unused).

        Returns:
            ``scores`` unchanged.
        """
        return scores

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dot = (output * v).sum(dim=-1, keepdim=True)
        vnorm = (v * v).sum(dim=-1, keepdim=True) + self.eps
        return output - self.scale * (dot / vnorm) * v


class Zero:
    """Zero the score diagonal before softmax; do not modify output."""

    def __init__(self, config: Config, scale: torch.Tensor) -> None:
        """Store ``config.eps``; ignore ``scale``.

        Args:
            config: Source :class:`Config`; reads ``eps``.
            scale: Unused; kept for signature symmetry.
        """
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Replace the score diagonal with ``-inf``.

        Args:
            scores: Pre-softmax attention scores.
            mask: Optional attention mask (unused).

        Returns:
            ``scores`` with the main diagonal masked to ``-inf``.
        """
        n = scores.shape[-1]
        diag = torch.eye(n, device=scores.device, dtype=torch.bool)
        return scores.masked_fill(diag, float("-inf"))

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Identity; ``Zero`` does not touch the post-softmax output.

        Args:
            output: Post-softmax attention output.
            v: Value tensor (unused).

        Returns:
            ``output`` unchanged.
        """
        return output


class Mask:
    """Zero the diagonal AND subtract the projection."""

    def __init__(self, config: Config, scale: torch.Tensor) -> None:
        """Store ``scale`` and ``config.eps``.

        Args:
            config: Source :class:`Config`; reads ``eps``.
            scale: Learnable scalar ``nn.Parameter`` allocated by
                the owning :class:`Xsa` module.
        """
        self.scale = scale
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Replace the score diagonal with ``-inf``.

        Args:
            scores: Pre-softmax attention scores.
            mask: Optional attention mask (unused).

        Returns:
            ``scores`` with the main diagonal masked to ``-inf``.
        """
        n = scores.shape[-1]
        diag = torch.eye(n, device=scores.device, dtype=torch.bool)
        return scores.masked_fill(diag, float("-inf"))

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Subtract the projection of ``output`` onto ``v``.

        Args:
            output: Post-softmax attention output.
            v: Value tensor.

        Returns:
            ``output - scale * (output · v) / (v · v + eps) * v``.
        """
        dot = (output * v).sum(dim=-1, keepdim=True)
        vnorm = (v * v).sum(dim=-1, keepdim=True) + self.eps
        return output - self.scale * (dot / vnorm) * v


XSA_MODE = {
    "subtract": Projection,
    "zero": Zero,
    "mask": Mask,
}


def XsaStrategy(config: Config, scale: torch.Tensor) -> XsaMode:
    """Build the XSA strategy selected by ``Config.mode``.

    Args:
        config: Source :class:`Config`; reads ``mode``.
        scale: Learnable scalar ``nn.Parameter`` allocated by the
            owning :class:`Xsa`. Consumed by :class:`Projection` and
            :class:`Mask`; unused by :class:`Zero`.

    Returns:
        An :class:`XsaMode` strategy instance.
    """
    return XSA_MODE[config.mode](config, scale)


class Xsa(Base):
    """Exclusive Self Attention (XSA) module.

    The XSA strategy (``self.xsa``) handles pre-softmax score modifications
    and post-softmax output cleaning. This class does not branch on
    ``Config.mode``.
    """

    def __init__(self, config: Config) -> None:
        """Build Q/K/V projections and the XSA strategy.

        Args:
            config: Source :class:`Config`; reads ``dim``, ``heads``,
                ``headdim``, ``drop``, ``eps``, and ``mode``.

        Side Effects:
            Allocates ``self.scale`` as a trainable
            ``nn.Parameter(torch.ones(1))`` regardless of mode. This
            keeps ``state_dict`` keys stable across modes.
        """
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.headdim)
        # Always allocate as a trainable parameter — no mode branching.
        self.xsa_scale = nn.Parameter(torch.ones(1))
        self.xsa = XsaStrategy(config, self.xsa_scale)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attention with XSA self-exclusion.

        Args:
            q: Queries, shape ``(batch, heads, seq_len, headdim)``.
            k: Keys, same shape as ``q``.
            v: Values, same shape as ``q``.
            m: Optional attention mask broadcastable to
                ``(batch, heads, seq_len, seq_len)``.

        Returns:
            Per-head attention output, shape
            ``(batch, heads, seq_len, headdim)``.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = self.xsa.prepare(scores, m)
        scores = keep(scores, m) if m is not None else scores
        weights = torch.nn.functional.softmax(scores, dim=-1)
        if self.drop is not None:
            weights = self.drop(weights)
        output = torch.matmul(weights, v)
        return self.xsa.apply(output, v)