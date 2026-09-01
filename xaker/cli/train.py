"""xaker-train: train a XAKER model on synthetic reversal data."""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from xaker.config import Config
from xaker.model.model import Model
from xaker.training.trainer import Fit, Trainer
import xaker.utils.rng


def dummy(samples: int, length: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic (input, label) pair.

    Args:
        samples: Number of sequences.
        length: Sequence length.
        vocab: Vocabulary size.

    Returns:
        Tuple ``(x, y)`` where ``y`` is a copy of ``x``. The reversal
        task trains the model to copy the input.
    """
    x = torch.randint(0, vocab, (samples, length))
    y = x.clone()
    return x, y


def main(argv: list[str] | None = None) -> int:
    """Parse CLI flags and run :class:`Trainer` on synthetic data.

    Args:
        argv: Optional argument vector; ``None`` reads from
            ``sys.argv``.

    Returns:
        Process exit code; ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Train XAKER model")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--kind", choices=["standard", "xsa", "fused"], default="fused")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args(argv)

    xaker.utils.rng.seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    cfg = Config(dim=args.dim, heads=args.heads, drop=0.1)
    model = Model(
        cfg,
        num_layers=args.layers,
        vocab_size=args.vocab,
        max_seq_len=args.length,
        drop=0.1,
        attention_type=args.kind,
    ).to(device)

    x, y = dummy(args.samples, args.length, args.vocab)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True)
    trainer = Trainer(model, Fit(epochs=args.epochs, lr=args.lr), device)

    print(f"Training {args.kind} attention on {device}")
    for epoch in range(args.epochs):
        trainer.epoch(loader)
        print(f"epoch {epoch + 1}/{args.epochs} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
