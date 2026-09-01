"""Core abstractions and shared utilities for multi-head attention.

Provides:

* :func:`heads` / :func:`merge` — convert between the
  ``(batch, seq_len, dim)`` layout produced by Linear projections and the
  ``(batch, heads, seq_len, headdim)`` layout expected by attention math.
* :class:`Qkv` — three independent bias-free linear projections used
  by subclasses of :class:`Base`.
* :func:`m` / :func:`broadcast` — masked filling and limited
  3-D-to-4-D m expansion.
* :class:`Base` — abstract base implementing the projection
  boilerplate via the template-method pattern; subclasses only override
  :meth:`~Base.attend`.
* Input validation in :class:`Base` checks rank, width, and
  non-finite values before projection.

The clamp bound (``BOUND = 1e6``) lives in :mod:`xaker.solver.precond`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

import torch
from torch import nn

from xaker.config import Config
from xaker.solver.precond import BOUND

logger = logging.getLogger(__name__)

def heads(x: torch.Tensor, heads: int, headdim: int) -> torch.Tensor:
    """Reshape a projected tensor into per-head layout.

    Args:
        x: Tensor of shape ``(batch, seq_len, dim)``. The last dimension
            must equal ``heads * headdim``.
        heads: Number of attention heads.
        headdim: Per-head feature dimension.

    Returns:
        Tensor of shape ``(batch, heads, seq_len, headdim)`` suitable for
        batched attention matmuls. The view + transpose is non-contiguous; do
        not assume the returned tensor is contiguous in memory.

    Raises:
        ValueError: If ``x`` is not three-dimensional and its shape cannot be
            unpacked as ``(batch, seq_len, width)``.
        RuntimeError: If ``view`` cannot reshape the tensor to the requested
            number and width of heads.
    """
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, heads, headdim).transpose(1, 2)

def merge(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`heads`; merges heads back into ``dim``.

    Args:
        x: Tensor of shape ``(batch, heads, seq_len, headdim)``.

    Returns:
        Contiguous tensor of shape ``(batch, seq_len, dim)`` where
        ``dim = heads * headdim``. The transpose + ``contiguous()``
        call triggers a memory copy which is safe under autograd but not free.

    Raises:
        ValueError: If ``x`` is not four-dimensional and its shape cannot be
            unpacked as per-head layout.
        RuntimeError: If the final ``view`` cannot merge the head dimensions.
    """
    batch, _, seq_len, _ = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

def broadcast(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ensure a 3-D mask broadcasts against an attention-score tensor.

    Args:
        mask: ``(batch, seq_len, seq_len)`` mask (1 means keep, 0 means fill)
            or any tensor already compatible with ``target``.
        target: Score tensor of shape ``(batch, heads, seq_len, seq_len)``
            (or another rank) used to decide whether to insert a head axis.

    Returns:
        Mask with an inserted singleton at position 1 when ``mask.dim() == 3``
        and ``target.dim() == 4``; otherwise ``mask`` is returned unchanged.
    """
    if mask.dim() == 3 and target.dim() == 4:
        return mask.unsqueeze(1)
    return mask

def keep(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    fill: float = float("-inf"),
) -> torch.Tensor:
    """Replace positions where ``mask == 0`` in a score tensor.

    The default fill value of ``-inf`` is intended for scores passed to
    softmax.

    Args:
        scores: ``(batch, heads, seq_len, seq_len)`` raw or scaled scores.
        mask: ``(batch, seq_len, seq_len)`` or already-broadcastable mask,
            or ``None`` to return scores unchanged.
        fill: Value inserted at masked positions.

    Returns:
        Masked scores with the same shape and dtype as ``scores``.
    """
    if mask is None:
        return scores
    m_expanded = broadcast(mask, scores)
    return scores.masked_fill(m_expanded == 0, fill)


class Qkv(nn.Module):
    """Shared Q, K, V linear projections for multi-head attention.

    Wraps three independent ``nn.Linear`` layers with ``bias=False``. Weights
    are managed by PyTorch's parameter system and can be initialised externally
    (``Laker.init`` does so).
    """

    def __init__(self, config: Config) -> None:
        """Initialise the Q, K, V linear projections.

        Args:
            config: :class:`xaker.config.Config` whose
                ``dim`` field drives every projection width.

        Side Effects:
            Allocates three independent
            :class:`torch.nn.Linear` layers (``w_q``, ``w_k``,
            ``w_v``), each ``dim -> dim`` with ``bias=False``.
            Initial weights follow PyTorch's default initialization;
            external modules (e.g. :meth:`Laker.init`)
            may reinitialise them.
        """
        super().__init__()
        dim = config.dim
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, dim)``.

        Returns:
            ``(q, k, v)`` each of shape ``(batch, seq_len, dim)``; reshape
            into per-head layout with :func:`heads` before the
            attention op.

        Raises:
            RuntimeError: Propagated from :class:`torch.nn.Linear` for
                incompatible shapes, dtypes, or devices.
        """
        return self.w_q(x), self.w_k(x), self.w_v(x)

class Base(nn.Module, ABC):
    """Abstract base for multi-head attention with template-method pattern.

    Centralises Q/K/V projection, drop configuration, the output Linear and
    input validation so subclasses can focus solely on the per-head attention
    computation. Subclasses must implement :meth:`attend`, which
    receives the pre-projected, per-head Q/K/V tensors and returns the per-head
    output.

    The forward signature is fixed at ``forward(x, m=None)`` and is shared
    by every attention module in this subpackage, which allows them to be used
    interchangeably inside Transformer blocks.

    Attributes:
        config: ``Config`` driving projection widths and drop.
        heads: Number of attention heads.
        headdim: Per-head feature dimension.
        dim: Configured input/output width. A conflicting explicit
            ``headdim`` is not validated here and can cause reshape failure.
        qkv_proj: Module producing ``(q, k, v)`` of shape
            ``(batch, seq_len, dim)``.
        w_o: Output projection applied after head merging.
        drop: Optional ``nn.Dropout`` applied inside
            :meth:`attend` for variants that use softmax weights.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the base multi-head attention.

        Args:
            config: :class:`xaker.config.Config` consumed
                by the base class. ``headdim`` is read via
                :func:`cast` (assumed already validated by
                :meth:`Config.__post_init__`).

        Side Effects:
            Allocates :class:`Qkv`, the output linear
            :attr:`w_o`, and (when ``config.drop > 0.0``) an
            :class:`torch.nn.Dropout` stored on :attr:`drop`.
        """
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.headdim = cast(int, config.headdim)
        self.dim = config.dim

        self.qkv_proj = Qkv(config)
        self.w_o = nn.Linear(config.dim, config.dim, bias=False)

        self.drop: Optional[nn.Dropout] = None
        if config.drop > 0.0:
            self.drop = nn.Dropout(config.drop)

    def check(self, x: torch.Tensor) -> None:
        """Check input rank and width and clamp infinities in place.

        Args:
            x: Candidate input tensor.

        Raises:
            ValueError: If ``x`` is not 3-D or its last dimension does not
                match ``self.dim``.
            RuntimeError: If the in-place non-finite clamp is rejected, for
                example because ``x`` is a leaf tensor requiring gradients.

        Side Effects:
            When non-finite entries are detected, logs a warning and attempts
            an in-place clamp to ``[-BOUND, BOUND]``.
            Infinities become finite bounds; NaNs remain NaN.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, dim), got shape {x.shape}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dim {x.shape[-1]} != dim {self.dim}")
        if not torch.isfinite(x).all():
            logger.warning("Non-finite values detected in attention input; clamping.")
            x.clamp_(-BOUND, BOUND)

    @abstractmethod
    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        m: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-head attention from already-projected Q, K, V.

        Args:
            q: Queries of shape ``(batch, heads, seq_len, headdim)``.
            k: Keys of shape ``(batch, heads, seq_len, headdim)``.
            v: Values of shape ``(batch, heads, seq_len, headdim)``.
            m: Optional m broadcastable to
                ``(batch, heads, seq_len, seq_len)``; ``None`` if no
                masking should be applied.

        Returns:
            Per-head attention output of shape
            ``(batch, heads, seq_len, headdim)``; the base class merges
            heads and applies ``w_o`` afterward.
        """

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project, compute attention, merge heads and apply output projection.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, dim)``.
            mask: Optional attention mask; semantics depend on the subclass.

        Returns:
            Tensor of shape ``(batch, seq_len, dim)`` ready to feed into
            the next Transformer sublayer.

        Raises:
            ValueError: Propagated from :meth:`check` when
                ``x`` is not 3-D or its width does not match
                ``self.dim``.
            RuntimeError: Propagated from :meth:`attend`
                or the downstream matmul/resize operations for
                incompatible shapes, dtypes, or devices.
        """
        self.check(x)

        q_raw, k_raw, v_raw = self.qkv_proj(x)

        q = heads(q_raw, self.heads, self.headdim)
        k = heads(k_raw, self.heads, self.headdim)
        v = heads(v_raw, self.heads, self.headdim)

        out_heads = self.attend(q, k, v, mask)

        out = merge(out_heads)

        return cast(torch.Tensor, self.w_o(out))
