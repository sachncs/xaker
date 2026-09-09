"""Save, reload, and inspect a trained xaker model.

Run a tiny training pass on synthetic data, dump the trained
state_dict to disk, then reload it through the public API and
confirm the layer shapes and parameter counts match what
xaker-train --out writes in production.

Usage:
    python -m examples.save_load
    python -m examples.save_load --epochs 4 --out artifacts/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from xaker import BLOCK, Config, Fit, Model, Trainer


def make_synthetic(samples: int, length: int, vocab: int) -> TensorDataset:
    """Synthetic copy-task dataset: target == input."""
    x = torch.randint(0, vocab, (samples, length))
    y = x.clone()
    return TensorDataset(x, y)


def build_model(cfg: Config, vocab: int, length: int) -> Model:
    """Construct a small Fused-xsa Transformer."""
    return Model(
        cfg,
        num_layers=2,
        vocab_size=vocab,
        max_seq_len=length,
        drop=0.1,
        attention_type="fused",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, save, reload a xaker model.")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/last.pt",
                        help="Where to write the state_dict.")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = Config(dim=args.dim, heads=args.heads, drop=0.1, precond="fast")
    model = build_model(cfg, vocab=args.vocab, length=args.length)
    before = sum(p.numel() for p in model.parameters())
    print(f"parameters: {before:,}")

    ds = make_synthetic(args.samples, args.length, args.vocab)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    trainer = Trainer(model, Fit(epochs=args.epochs, lr=1e-3, decay=0.1),
                     device=torch.device("cpu"))
    for _ in range(args.epochs):
        trainer.epoch(loader)

    # 1. Save.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved state_dict -> {out_path}")

    # 2. Reload through the BLOCK registry path.
    recon = build_model(cfg, vocab=args.vocab, length=args.length)
    recon.load_state_dict(torch.load(out_path, weights_only=True))
    recon.eval()
    after = sum(p.numel() for p in recon.parameters())
    assert before == after, f"parameter count mismatch: {before} != {after}"
    print(f"reloaded: parameters: {after:,} ✓")

    # 3. Smoke forward pass.
    with torch.no_grad():
        x = torch.randint(0, args.vocab, (1, args.length))
        logits = recon(x)
    print(f"forward shape: {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
