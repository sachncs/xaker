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
from typing import Optional, Protocol

import torch
from torch import nn
import torch.nn.functional

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
        self.scale = scale
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return scores

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dot = (output * v).sum(dim=-1, keepdim=True)
        vnorm = (v * v).sum(dim=-1, keepdim=True) + self.eps
        return output - self.scale * (dot / vnorm) * v


class Zero:
    """Zero the score diagonal before softmax; do not modify output."""

    def __init__(self, config: Config, scale: torch.Tensor) -> None:
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        n = scores.shape[-1]
        diag = torch.eye(n, device=scores.device, dtype=torch.bool)
        return scores.masked_fill(diag, float("-inf"))

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return output


class Mask:
    """Zero the diagonal AND subtract the projection."""

    def __init__(self, config: Config, scale: torch.Tensor) -> None:
        self.scale = scale
        self.eps = config.eps

    def prepare(self, scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        n = scores.shape[-1]
        diag = torch.eye(n, device=scores.device, dtype=torch.bool)
        return scores.masked_fill(diag, float("-inf"))

    def apply(self, output: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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

    The ``scale`` is consumed by :class:`Projection` and :class:`Mask`,
    and unused by :class:`Zero`.
    """
    return XSA_MODE[config.mode](config, scale)


class Xsa(Base):
    """Exclusive Self Attention (XSA) module.

    The XSA strategy (``self.xsa``) handles pre-softmax score modifications
    and post-softmax output cleaning. This class does not branch on
    ``Config.mode``.
    """

    def __init__(self, config: Config) -> None:
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
        """Compute attention with XSA self-exclusion."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = self.xsa.prepare(scores, m)
        scores = keep(scores, m) if m is not None else scores
        weights = torch.nn.functional.softmax(scores, dim=-1)
        if self.drop is not None:
            weights = self.drop(weights)
        output = torch.matmul(weights, v)
        return self.xsa.apply(output, v)