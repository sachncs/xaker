"""Tests for the model wrappers: :class:`Mlp`, block, and full model.

Covers :class:`xaker.model.block.Mlp` (shape preservation,
finite outputs, ``drop`` toggle, GELU / ReLU gradients),
:class:`Block` (every documented ``attention_type``
— ``standard`` / ``xsa`` / ``kernel`` / ``fused`` / ``fused_v2`` — and
``ValueError`` on unknown values), and :class:`Model`
(token-id path ``(batch, seq_len) → (batch, seq_len, vocab_size)``;
embedding path ``(batch, seq_len, dim) → (batch, seq_len, dim)``;
``ValueError`` on mismatched shapes; causal mask passthrough; finite
parameter gradients on the token-id path).
"""

from __future__ import annotations

import pytest
import torch

from xaker.config import Config
from xaker.model.block import Mlp, Block
from xaker.model.model import Model

class TestMLP:
    """Tests for Mlp feed-forward block."""

    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_shape(self, activation: str) -> None:
        mlp = Mlp(dim=64, d_ff=256, activation=activation)
        x = torch.randn(2, 32, 64)
        out = mlp(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_finite(self, activation: str) -> None:
        mlp = Mlp(dim=64, d_ff=256, activation=activation)
        x = torch.randn(2, 32, 64)
        out = mlp(x)
        assert torch.isfinite(out).all()

    def test_drop_off(self) -> None:
        mlp = Mlp(dim=64, d_ff=256, drop=0.0)
        assert mlp.drop is None

    def test_drop_on(self) -> None:
        mlp = Mlp(dim=64, d_ff=256, drop=0.1)
        assert mlp.drop is not None

    def test_grad(self) -> None:
        mlp = Mlp(dim=64, d_ff=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_drop_train(self) -> None:
        mlp = Mlp(dim=64, d_ff=256, drop=0.5)
        x = torch.randn(2, 16, 64)
        mlp.train()
        out_train = mlp(x)
        assert out_train.shape == x.shape

class TestTransformerBlock:
    """Tests for Block."""

    def test_fused_v2_block(self) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=256, attention_type="fused_v2")
        block.eval()
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_std_block(self) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=128, attention_type="standard")
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_xsa_block(self) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=128, attention_type="xsa")
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        "attn_type", ["standard", "xsa", "kernel", "fused", "fused_v2"]
    )
    def test_kinds(self, attn_type: str) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=128, attention_type=attn_type)
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_invalid(self) -> None:
        cfg = Config(dim=64, heads=4)
        with pytest.raises(ValueError, match="Unknown attention type"):
            Block(cfg, attention_type="imaginary")

    def test_drop(self) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=128, drop=0.1)
        assert block.drop is not None

    def test_grad_block(self) -> None:
        cfg = Config(dim=64, heads=4, drop=0.0)
        block = Block(cfg, d_ff=128, attention_type="fused_v2")
        block.train()
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

class TestFullModel:
    """Tests for Model."""

    @pytest.fixture
    def config(self) -> Config:
        return Config(dim=64, heads=4, drop=0.0)

    def test_token(self, config: Config) -> None:
        model = Model(config, num_layers=2, vocab_size=500)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 500)

    def test_embed(self, config: Config) -> None:
        model = Model(config, num_layers=2, vocab_size=None)
        model.eval()
        x = torch.randn(2, 32, config.dim)
        out = model(x)
        assert out.shape == x.shape

    def test_validate(self, config: Config) -> None:
        model = Model(config, num_layers=2, vocab_size=500)
        model.eval()
        with pytest.raises(ValueError, match="2D"):
            model(torch.randn(2, 32, config.dim))

    def test_finite(self, config: Config) -> None:
        model = Model(config, num_layers=2, vocab_size=500)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        out = model(x)
        assert torch.isfinite(out).all()

    def test_causal(self, config: Config) -> None:
        model = Model(config, num_layers=2, vocab_size=None)
        model.eval()
        x = torch.randn(2, 16, config.dim)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        out = model(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape

    def test_grad_token(self, config: Config) -> None:
        model = Model(config, num_layers=1, vocab_size=100)
        model.train()
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN in {name}"
