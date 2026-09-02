"""WikiText-2 training benchmark for the xaker paper.

Trains each attention variant on WikiText-2 character-level language
modelling for a fixed number of steps, then reports the validation
loss and perplexity. CPU-friendly (uses small dim).

Outputs JSON to ``paper_runs/wikitext.json`` with one entry per
``(kind, layer, dim)`` configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xaker import BLOCK, Config, Fit, Trainer
from xaker.bench.bench import gitsha, tick, write
from xaker.config import Config
from xaker.model import Model
from xaker.training.loss import ce

sys.path.insert(0, ".")
from xaker.datasets import WikiText, build, vocab


def evaluate(model: torch.nn.Module, data: DataLoader, device: torch.device) -> float:
    """Compute mean cross-entropy and convert to perplexity.

    Args:
        model: Trained model.
        data: Validation DataLoader yielding ``(x, y)``.
        device: Device for evaluation.

    Returns:
        Perplexity (``exp(mean_loss)``).
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y, smoothing=0.0)
            losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


def trainstep(kind: str, dim: int, vocab_size: int, length: int, *, epochs: int, device: torch.device) -> dict:
    """Train one model end-to-end on WikiText-2.

    Args:
        kind: Attention kind.
        dim: Model width.
        vocab_size: Vocabulary size.
        length: Sequence length.
        epochs: Number of training epochs.
        device: Training device.

    Returns:
        Dict with ``train_loss``, ``val_loss``, ``val_ppl``, ``wall_seconds``.
    """
    torch.manual_seed(0)
    train_ds = WikiText("train", length=length)
    val_ds = WikiText("validation", length=length)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    cfg = Config(dim=dim, heads=4, drop=0.1)
    model = Model(cfg, num_layers=2, vocab_size=vocab_size, max_seq_len=length, attention_type=kind)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_loss_total = 0.0
    train_steps = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y, smoothing=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_total += loss.item()
            train_steps += 1
    wall = time.perf_counter() - t0
    val_loss_total = 0.0
    val_steps = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y, smoothing=0.0)
            val_loss_total += loss.item()
            val_steps += 1
    train_loss = train_loss_total / max(train_steps, 1)
    val_loss = val_loss_total / max(val_steps, 1)
    val_ppl = math.exp(val_loss)
    return {
        "kind": kind, "dim": dim, "length": length, "epochs": epochs,
        "train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl,
        "wall_seconds": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train on WikiText-2 and report perplexity")
    parser.add_argument("--out", default="paper_runs/wikitext.json")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(os.environ.get("XAKER_DEVICE", "cpu"))
    print(f"Training on {device}, dim={args.dim}, length={args.length}, epochs={args.epochs}")
    vocab_size = 256  # byte-level
    results = {
        "config": {
            "device": str(device),
            "dim": args.dim,
            "length": args.length,
            "epochs": args.epochs,
        },
        "git_sha": gitsha(),
        "torch_version": torch.__version__,
        "results": [],
    }
    for kind in ["standard", "xsa", "fused", "linear"]:
        print(f"\n=== Training {kind} ===")
        try:
            r = trainstep(kind, args.dim, vocab_size, args.length, epochs=args.epochs, device=device)
            print(f"  train_loss={r['train_loss']:.4f} val_loss={r['val_loss']:.4f} val_ppl={r['val_ppl']:.2f} wall={r['wall_seconds']:.1f}s")
            results["results"].append(r)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results["results"].append({"kind": kind, "error": str(exc)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())