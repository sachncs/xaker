"""Tests for the bench ablation / datasets / linear attention modules."""

from __future__ import annotations

import math

import pytest
import torch

from xaker import BLOCK, Config
from xaker.attention.linear import Linear
from xaker.bench.ablate import bench as ablate_bench
from xaker.bench.ablate import drive, sweep
from xaker.bench.bench import Metrics
from xaker.bench.condition import cond
from xaker.bench.copy_task import dataset as copy_dataset
from xaker.bench.lra import (
    TASKS,
    addition,
    copy as copy_task_fn,
    retrieval,
    reversal,
)
from xaker.datasets import CopyTask, ReversalTask, build, vocab


class TestLinearAttention:
    """Tests for the Linear attention baseline."""

    def test_shape(self) -> None:
        """Linear attention preserves (batch, seq, dim)."""
        cfg = Config(dim=32, heads=4)
        attn = Linear(cfg).eval()
        x = torch.randn(2, 16, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_finite(self) -> None:
        """Linear attention output is finite."""
        cfg = Config(dim=32, heads=4)
        attn = Linear(cfg).eval()
        x = torch.randn(2, 16, 32)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_block_dispatch(self) -> None:
        """Linear is registered in BLOCK."""
        assert "linear" in BLOCK
        cfg = Config(dim=32, heads=4)
        attn = BLOCK["linear"](cfg)
        assert isinstance(attn, Linear)

    def test_backward(self) -> None:
        """Linear attention is differentiable."""
        cfg = Config(dim=32, heads=4)
        attn = Linear(cfg).eval()
        x = torch.randn(2, 16, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestDatasets:
    """Tests for the dataset loaders."""

    def test_copy_task(self) -> None:
        """Copy task: input == target."""
        ds = CopyTask(vocab=16, length=8, size=32)
        assert len(ds) == 32
        x, y = ds[0]
        assert x.shape == (8,)
        assert torch.equal(x, y)

    def test_reversal_task(self) -> None:
        """Reversal task: target is reversed input."""
        ds = ReversalTask(vocab=16, length=8, size=32)
        x, y = ds[0]
        assert torch.equal(x, y.flip(dims=[0]))

    def test_build_factory(self) -> None:
        """build() factory dispatches by name."""
        copy = build("copy", vocab=8, length=4, size=4)
        assert isinstance(copy, CopyTask)
        rev = build("reversal", vocab=8, length=4, size=4)
        assert isinstance(rev, ReversalTask)
        with pytest.raises(ValueError):
            build("nonexistent")

    def test_vocab_lookup(self) -> None:
        """vocab() returns sensible defaults."""
        assert vocab("wikitext") == 256
        assert vocab("copy") == 64
        with pytest.raises(ValueError):
            vocab("nonexistent")


class TestLRATasks:
    """Tests for the Long Range Arena-style task generators."""

    def test_copy(self) -> None:
        """LRA copy: input == target."""
        ds = copy_task_fn(vocab=8, length=8, size=16)
        x, y = ds[0]
        assert torch.equal(x, y)

    def test_reversal(self) -> None:
        """LRA reversal: target is reversed."""
        ds = reversal(vocab=8, length=8, size=16)
        x, y = ds[0]
        assert torch.equal(x, y.flip(dims=[0]))

    def test_addition(self) -> None:
        """LRA addition: sum of two halves."""
        ds = addition(vocab=10, length=10, size=16)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_retrieval(self) -> None:
        """LRA retrieval: layout is correct."""
        ds = retrieval(vocab=10, length=32, size=16)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_task_registry(self) -> None:
        """All four tasks are registered."""
        assert set(TASKS.keys()) == {"copy", "reversal", "addition", "retrieval"}


class TestCopyTaskBench:
    """Tests for the copy_task bench dataset."""

    def test_make_copy(self) -> None:
        """dataset generates input == target."""
        x, y = copy_dataset(vocab=8, length=8, size=16)
        assert x.shape == (16, 8)
        assert y.shape == (16, 8)
        assert torch.equal(x, y)

    def test_make_copy_seed(self) -> None:
        """Same seed gives same data."""
        x1, y1 = copy_dataset(vocab=8, length=8, size=16, seed=42)
        x2, y2 = copy_dataset(vocab=8, length=8, size=16, seed=42)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


class TestConditionBench:
    """Tests for the condition number bench helper."""

    def test_cond_identity(self) -> None:
        """Identity matrix has condition number 1."""
        I = torch.eye(8).unsqueeze(0).unsqueeze(0)
        assert abs(cond(I) - 1.0) < 1e-3

    def test_cond_well_conditioned(self) -> None:
        """Diagonal matrix has condition = ratio of extremes."""
        D = torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0])).unsqueeze(0).unsqueeze(0)
        assert abs(cond(D) - 8.0) < 1e-3


class TestAblateBench:
    """Smoke tests for the ablation bench runner."""

    def test_bench_standard(self) -> None:
        """bench returns Metrics with positive values for Standard."""
        m = ablate_bench("standard", dim=32, heads=4, length=8, seed=0)
        assert isinstance(m, Metrics)
        assert m.forward_ms_mean >= 0.0

    def test_sweep_kernel(self) -> None:
        """sweep runs with kernel sweep."""
        m = sweep("kernel", "exp", dim=32, heads=4, length=8, seed=0)
        assert isinstance(m, Metrics)

    def test_drive_kind(self) -> None:
        """drive returns a populated Result."""
        result = drive("kind", ["standard"], dim=32, heads=4, length=8, seeds=1)
        assert len(result.results) >= 1
