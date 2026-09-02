"""Linear (linearized-complexity) self-attention.

Reference implementation of the linear attention kernel from
Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive
Transformers with Linear Attention" (2020).

The trick: replace softmax with a feature map phi (we use the
positive-elephant kernel `elu(x) + 1` by default), then use the
associativity of matrix multiplication to compute attention in O(n)
instead of O(n^2):

    output = (Q' @ K'^T) @ V     -- O(n^2 d) memory and time
    output = Q' @ (K'^T @ V)     -- O(n d^2) memory and time

where `Q' = phi(Q)` and `K' = phi(K)`. The second form scales
linearly in sequence length, which is the headline advantage of
linear attention over softmax.

This module is a baseline for the XSA + kernel ridge regression
fusion in :class:`Fused`. It uses the same :class:`Base` interface
so it can be plugged into :data:`xaker.attention.BLOCK`.
"""

from __future__ import annotations

from typing import Optional

import torch

from xaker.config import Config
from xaker.attention.core import Base


class Linear(Base):
    """Linear-complexity attention with `elu + 1` feature map.

    The feature map is `phi(x) = elu(x) + 1`, which is strictly
    positive and gives a well-defined kernel. The matrix
    associativity trick lets us compute attention as
    ``phi(Q) @ (phi(K)^T @ V)`` in O(n d^2) time and memory instead
    of the O(n^2 d) softmax form.

    Trade-offs:

    - Linear memory in sequence length.
    - No exact softmax; approximation can hurt accuracy on tasks
      where softmax matters.
    - No XSA exclusion built in; this is a pure baseline.

    Example:
        >>> config = Config(dim=512, heads=8)
        >>> attn = Linear(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: Config) -> None:
        """Initialise the linear attention module.

        Args:
            config: :class:`xaker.config.Config`; only `dim`,
                `heads`, `headdim`, and `drop` are read.
        """
        super().__init__(config)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        m: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute linear-complexity attention per head.

        Reorders the matmul to run in O(n d^2) instead of O(n^2 d):

        ``out = phi(q) @ (phi(k).transpose(-2, -1) @ v)``

        Args:
            q: Queries ``(batch, heads, seq_len, headdim)``.
            k: Keys, same shape as ``q``.
            v: Values, same shape as ``q``.
            m: Optional mask broadcastable to scores; ignored
                (linear attention has no softmax row that the mask
                could fill).

        Returns:
            Weighted value tensor of shape
            ``(batch, heads, seq_len, headdim)``.
        """
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0
        # Reorder to O(n d^2): kv first, then q.
        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv)
        if self.drop is not None:
            out = self.drop(out)
        return out
