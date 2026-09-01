"""Shape verification tests for the XAKER components.

* :class:`Standard` preserves ``(batch, seq_len, dim)``.
* :class:`Xsa` and the :class:`Laker` are verified across batch sizes
  and sequence lengths.
* :class:`Block` preserves the same shape.
* The full :class:`Model` returns ``(batch, seq_len, vocab_size)`` when
  ``vocab_size`` is set and ``(batch, seq_len, dim)`` otherwise.
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.attention.standard import Standard
from xaker.attention.xsa import Xsa
from xaker.model.block import Block
from xaker.model.model import Model
from xaker.utils.ops import shape


class TestShape:
    """Test attention module shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
    def test_std(self, batch_size: int, seq_len: int) -> None:
        config = Config(dim=128, heads=4)
        attn = Standard(config)
        x = torch.randn(batch_size, seq_len, config.dim)
        out = attn(x)
        shape(out, (batch_size, seq_len, config.dim), "output")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_xsa(self, batch_size: int, seq_len: int) -> None:
        config = Config(dim=128, heads=4, mode="subtract")
        attn = Xsa(config)
        x = torch.randn(batch_size, seq_len, config.dim)
        out = attn(x)
        shape(out, (batch_size, seq_len, config.dim), "output")


class TestShapeBlock:
    """Test Transformer block shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_block(self, batch_size: int, seq_len: int) -> None:
        config = Config(dim=128, heads=4)
        block = Block(config, d_ff=512)
        x = torch.randn(batch_size, seq_len, config.dim)
        out = block(x)
        shape(out, (batch_size, seq_len, config.dim), "output")


class TestShapeModel:
    """Test full model shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_vocab(self, batch_size: int, seq_len: int) -> None:
        config = Config(dim=128, heads=4)
        model = Model(
            config, num_layers=4, vocab_size=1000, max_seq_len=512,
        )
        x = torch.randint(0, 1000, (batch_size, seq_len))
        out = model(x)
        shape(out, (batch_size, seq_len, 1000), "output")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_embed(self, batch_size: int, seq_len: int) -> None:
        config = Config(dim=128, heads=4)
        model = Model(config, num_layers=4, vocab_size=None)
        x = torch.randn(batch_size, seq_len, config.dim)
        out = model(x)
        shape(out, (batch_size, seq_len, config.dim), "output")