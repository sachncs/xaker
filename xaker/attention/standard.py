"""Standard Multi-Head Self-Attention.

Reference implementation of scaled dot-product attention as described in
"Attention Is All You Need" (Vaswani et al., 2017). Serves as the baseline
for comparing vanilla attention against xaker's XSA-only and fused-XSA
variants.

The math is the textbook form,
``Attention(Q, K, V) = softmax(Q K^T / sqrt(headdim)) V``,
broadcast independently over heads and batches. Masked positions are
filled with ``-inf``; their softmax probability is zero when the row contains
at least one finite score, while a fully masked row produces NaNs. If
``drop`` is configured, it is applied to the post-softmax weights.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional

from xaker.config import Config
from xaker.attention.core import Base, keep

class Standard(Base):
    """Standard scaled dot-product multi-head attention.

    Implements
    ``Attention(Q, K, V) = softmax(Q K^T / sqrt(headdim)) V``
    per head, with the QKV/Output projections and head reshape handled by the
    :class:`~xaker.attention.core.Base` base class.

    Example:
        >>> config = Config(dim=512, heads=8)
        >>> attn = Standard(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: Config) -> None:
        """Initialise the standard scaled dot-product attention.

        Args:
            config: :class:`xaker.config.Config` consumed
                by the base class. Stores ``1 / sqrt(headdim)`` as
                ``self.scale`` for the per-head score scaling.

        Side Effects:
            Allocates the Q/K/V/output projection layers and
            optional drop via the base class.
        """
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.headdim)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        m: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute scaled dot-product attention per head.

        Args:
            q: Queries ``(batch, heads, seq_len, headdim)``.
            k: Keys ``(batch, heads, seq_len, headdim)``.
            v: Values ``(batch, heads, seq_len, headdim)``.
            m: Optional m broadcastable to
                ``(batch, heads, seq_len, seq_len)``; masked entries are
                filled with ``-inf`` before softmax.

        Returns:
            Weighted value tensor of shape
            ``(batch, heads, seq_len, headdim)`` to be merged by the
            base class.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul`,
                :func:`torch.nn.functional.softmax`, or
                :func:`m` for incompatible shapes, dtypes,
                or devices.

        Numerical notes:
            A masked score is ``-inf`` and therefore has zero softmax weight
            when its row also contains a finite score. If every entry in a row
            is masked, softmax receives only ``-inf`` values and returns NaNs.
            Dropout, when configured, is applied after softmax, so retained row
            sums need not remain one.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = keep(scores, m) if m is not None else scores
        weights = torch.nn.functional.softmax(scores, dim=-1)
        if self.drop is not None:
            weights = self.drop(weights)
        return torch.matmul(weights, v)
