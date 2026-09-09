"""Tests for :class:`Config` construction and validation.

Covers canonical defaults, single-word field values, and validation
rejections for illegal inputs (non-divisible dim, unknown enum strings,
pcg <= 0, eps, negative lam).
"""

from __future__ import annotations

import pytest

from xaker.config import Config


class TestConfigConstruction:
    """Tests for valid config construction."""

    def test_valid_default(self) -> None:
        cfg = Config(dim=128, heads=4)
        assert cfg.dim == 128
        assert cfg.heads == 4
        assert cfg.headdim == 32

    def test_explicit_headdim(self) -> None:
        cfg = Config(dim=128, heads=8, headdim=32)
        assert cfg.headdim == 32

    def test_auto_headdim(self) -> None:
        cfg = Config(dim=256, heads=8)
        assert cfg.headdim == 32

    def test_all_precond(self) -> None:
        for ptype in ["cccp", "fast", "diagonal", "identity"]:
            cfg = Config(dim=64, heads=4, precond=ptype)
            assert cfg.precond == ptype

    def test_all_mode(self) -> None:
        for mode in ["subtract", "zero", "mask"]:
            cfg = Config(dim=64, heads=4, mode=mode)
            assert cfg.mode == mode

    def test_all_kernel(self) -> None:
        for kt in ["exp", "rbf", "linear", "cosine"]:
            cfg = Config(dim=64, heads=4, kernel=kt)
            assert cfg.kernel == kt

    def test_pcg_field(self) -> None:
        cfg = Config(dim=64, heads=4, pcg=30)
        assert cfg.pcg == 30


class TestConfigInvalid:
    """Tests for config validation."""

    def test_div(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            Config(dim=65, heads=4)

    def test_kernel_invalid(self) -> None:
        with pytest.raises(ValueError, match="kernel"):
            Config(dim=64, heads=4, kernel="invalid")

    def test_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            Config(dim=64, heads=4, mode="invalid")

    def test_precond_invalid(self) -> None:
        with pytest.raises(ValueError, match="precond"):
            Config(dim=64, heads=4, precond="invalid")

    def test_pcg_zero(self) -> None:
        with pytest.raises(ValueError, match="pcg"):
            Config(dim=64, heads=4, pcg=0)

    def test_drop_low(self) -> None:
        with pytest.raises(ValueError, match="drop"):
            Config(dim=64, heads=4, drop=-0.1)

    def test_drop_high(self) -> None:
        with pytest.raises(ValueError, match="drop"):
            Config(dim=64, heads=4, drop=1.1)

    def test_eps_zero(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            Config(dim=64, heads=4, eps=0.0)

    def test_eps_neg(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            Config(dim=64, heads=4, eps=-0.1)

    def test_lam_neg(self) -> None:
        with pytest.raises(ValueError, match="lam"):
            Config(dim=64, heads=4, lam=-1.0)


class TestConfigDefault:
    """Tests for config default values."""

    def test_default(self) -> None:
        cfg = Config(dim=64, heads=4)
        assert cfg.drop == 0.0
        assert cfg.eps == 1e-6
        assert cfg.lam == 3.0
        assert cfg.kernel == "exp"
        assert cfg.mode == "subtract"
        assert cfg.precond == "fast"
        assert cfg.temp == 1.0
        assert cfg.symmetric is False
        assert cfg.normalize is True
        assert cfg.freq == 1
        assert cfg.tol == 1e-2