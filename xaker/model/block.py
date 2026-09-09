"""Transformer block for xaker.

The block uses the pre-norm residual pattern:

.. code-block:: text

    x = x + Dropout(Attention(LayerNorm1(x)))
    x = x + Dropout(Mlp(LayerNorm2(x)))

Normalizing before each sublayer controls the scale presented to the selected
attention and feed-forward implementations. Attention type is fixed when the
block is constructed.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from xaker.config import Config
from xaker.attention import BLOCK

class Mlp(nn.Module):
    """Position-wise feed-forward Mlp.

    A two-layer, bias-free feed-forward network with a GELU or ReLU
    activation. Optional drop is applied between the activation and second
    projection.

    .. math::

        \\text{Mlp}(x) = W_2 \\cdot \\text{activation}(W_1 \\cdot x)

    Attributes:
        linear1: First linear layer mapping ``dim`` to ``d_ff``.
        linear2: Second linear layer mapping ``d_ff`` back to
            ``dim``. Both layers are bias-free.
        drop: Optional drop applied between the activation and
            the second projection. ``None`` when ``drop == 0`` so
            the forward pass can skip the call entirely.
        activation: Callable applied to the hidden activations. Either
            :func:`torch.nn.functional.gelu` or
            :func:`torch.nn.functional.relu`, selected once at
            construction.

    Tensor Shapes:
        * Input: ``(..., dim)``.
        * Hidden: ``(..., d_ff)``.
        * Output: ``(..., dim)``.
    """

    def __init__(
        self,
        dim: int,
        d_ff: int,
        drop: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        """Initialize the Mlp.

        Args:
            dim: Input and output feature dimension.
            d_ff: Hidden feature dimension. Larger values increase
                capacity and compute roughly linearly.
            drop: Dropout probability applied after the activation.
                ``0.0`` disables drop and the corresponding module
                is not allocated.
            activation: ``"gelu"`` selects GELU; any other runtime string,
                including the annotated ``"relu"``, selects ReLU.
        """
        super().__init__()
        self.linear1 = nn.Linear(dim, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, dim, bias=False)
        self.drop: Optional[nn.Dropout] = None
        if drop > 0.0:
            self.drop = nn.Dropout(drop)
        self.activation = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Mlp to the input.

        Args:
            x: Input tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Tensor of shape ``(batch, seq_len, dim)`` after the
            two linear projections, activation, and (optional)
            drop.
        """
        x = self.linear1(x)
        x = self.activation(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.linear2(x)
        return x

class Block(nn.Module):
    """Single Transformer block with configurable attention.

    The block follows the pre-norm residual pattern:

    .. code-block:: text

        x = x + Dropout(Attention(LayerNorm1(x)))
        x = x + Dropout(Mlp(LayerNorm2(x)))

    The pre-norm order is part of the architecture; the implementation does not
    claim or check that normalization improves the conditioning of a particular
    attention kernel.

    The attention module is chosen once at construction time and is
    *not* re-evaluated per forward pass. The supported modes are:

    * ``"standard"`` - :class:`Standard` (the
      softmax-attention baseline used for ablations).
    * ``"xsa"`` - :class:`Xsa` (XSA only, no
      kernel solve).
    * ``"kernel"`` - :class:`Kernel` (the legacy
      LAKER path; kept for backward compatibility).
    * ``"fused"`` - :class:`Laker` (the v1 fusion).
    * ``"fused_v2"`` (default) - :class:`Laker`, using the
      configured v2 kernel, preconditioner mode, and PCG-style solve.

    ``attention_type`` alone selects the module.

    Attributes:
        config: The shared :class:`Config` instance. Stored
            on the block for introspection and for sub-modules that
            re-read it.
        norm1: LayerNorm applied to the block input before the
            attention sub-layer. Uses the config's ``eps``.
        norm2: LayerNorm applied to the block input before the Mlp
            sub-layer.
        attention: The selected attention module. Concrete type
            depends on ``attention_type``; one of
            :class:`Standard`,
            :class:`Xsa`,
            :class:`Kernel`,
            :class:`Laker`, or
            :class:`Laker`.
        mlp: The position-wise feed-forward network.
        drop: Optional drop applied to both the attention
            output and the Mlp output before the residual addition.
            ``None`` when ``drop == 0``.

    Tensor Shapes:
        * Input ``x``:  ``(batch, seq_len, dim)``.
        * Input ``mask``: ``(batch, seq_len, seq_len)`` or
          ``(batch, 1, seq_len, seq_len)``; passed through to the
          attention module which interprets it.
        * Output: ``(batch, seq_len, dim)``.
    """

    def __init__(
        self,
        config: Config,
        d_ff: Optional[int] = None,
        drop: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
        attention_type: Literal[
            "standard", "xsa", "kernel", "fused", "fused_v2", "linear"
        ] = "fused_v2",
    ) -> None:
        """Initialize the Transformer block.

        Args:
            config: Shared configuration. ``dim`` and ``eps`` are
                used to build the LayerNorms; the rest is forwarded
                to the attention module.
            d_ff: Hidden dimension of the Mlp. ``None`` (the default)
                selects ``4 * dim`` - the standard Transformer
                ratio. Block attention itself does not depend on
                ``d_ff``; this only sizes the Mlp.
            drop: Dropout probability applied to both the
                attention output and the Mlp output before each
                residual addition. ``0.0`` (the default) disables
                drop; the corresponding module is not allocated
                and the residual path is taken verbatim.
            activation: Mlp activation. ``"gelu"`` (default) or
                ``"relu"``.
            attention_type: Which attention module to instantiate.
                See the class docstring for the full enumeration. The
                default ``"fused_v2"`` selects
                :class:`xaker.attention.Laker`, the v2 path.

        Raises:
            ValueError: If ``attention_type`` is not one of the
                recognized values.
        """
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.dim, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.dim, eps=config.eps)

        # Aliases: "kernel" and "fused_v2" both select the fused (XSA + kernel) block.
        kind_map = {"fused_v2": "fused", "kernel": "fused"}
        kind = kind_map.get(attention_type, attention_type)
        if kind not in BLOCK:
            raise ValueError(f"Unknown attention type: {attention_type}")
        self.attention = BLOCK[kind](config)

        d_ff = d_ff if d_ff is not None else config.dim * 4
        self.mlp = Mlp(config.dim, d_ff, drop, activation)

        self.drop: Optional[nn.Dropout] = None
        if drop > 0.0:
            self.drop = nn.Dropout(drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the block to the input sequence.

        Runs the pre-norm attention sub-layer, adds the residual,
        runs the pre-norm Mlp sub-layer, and adds the second
        residual. Dropout (when configured) is applied to each
        sub-layer's output before the residual addition.

        Args:
            x: Input tensor of shape ``(batch, seq_len, dim)``.
            mask: Optional attention mask forwarded to the attention
                module. The block does not interpret the mask itself;
                it is passed through verbatim. May be ``None`` for
                fully-visible sequences.

        Returns:
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        if self.drop is not None:
            attn_out = self.drop(attn_out)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        if self.drop is not None:
            mlp_out = self.drop(mlp_out)
        x = x + mlp_out

        return x

    attention: nn.Module
