"""Copy-task training benchmark for the xaker paper.

Two-layer Transformer, length=16, 64-dim. Trains each attention
variant on a synthetic copy task and reports the loss curve plus
final accuracy. Designed to be CPU-friendly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from xaker import BLOCK, Config, Model
from xaker.bench.bench import gitsha
from xaker.training.loss import ce


def dataset(vocab: int, length: int, size: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``size`` copy-task samples.

    Args:
        vocab: Vocabulary size.
        length: Sequence length.
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Tuple ``(x, y)`` of long tensors, both of shape
        ``(size, length)``.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (size, length), generator=gen)
    y = x.clone()
    return x, y


def train(kind: str, *, dim: int, length: int, vocab: int, epochs: int, size: int, device: torch.device) -> dict:
    """Train one model on the copy task.

    Args:
        kind: Attention kind.
        dim: Model width.
        length: Sequence length.
        vocab: Vocabulary size.
        epochs: Number of epochs.
        size: Number of training samples.
        device: Training device.

    Returns:
        Dict with ``kind``, ``train_loss``, ``val_loss``, ``accuracy``,
        ``loss_curve``.
    """
    torch.manual_seed(0)
    x_train, y_train = dataset(vocab, length, size)
    x_val, y_val = dataset(vocab, length, 32, seed=1)
    cfg = Config(dim=dim, heads=4, drop=0.1)
    model = Model(cfg, num_layers=2, vocab_size=vocab, max_seq_len=length, attention_type=kind)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), 8):
            bx = x_train[perm[i:i+8]].to(device)
            by = y_train[perm[i:i+8]].to(device)
            logits = model(bx)
            loss = ce(logits, by, smoothing=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    wall = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        logits = model(x_val[:32].to(device))
        preds = logits.argmax(dim=-1)
        acc = (preds == y_val[:32].to(device)).float().mean().item()
        val_loss = ce(logits, y_val[:32].to(device), smoothing=0.0).item()
    return {
        "kind": kind, "dim": dim, "length": length, "vocab": vocab,
        "epochs": epochs, "train_loss": losses[-1], "val_loss": val_loss,
        "accuracy": acc, "wall_seconds": wall, "loss_curve": losses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy-task training benchmark")
    parser.add_argument("--out", default="paper_runs/copy_task.json")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(os.environ.get("XAKER_DEVICE", "cpu"))
    print(f"Copy task on {device}, dim={args.dim}, length={args.length}, epochs={args.epochs}")
    results = {
        "config": {"device": str(device), "dim": args.dim, "length": args.length,
                   "vocab": args.vocab, "epochs": args.epochs, "size": args.size},
        "git_sha": gitsha(), "torch_version": torch.__version__,
        "results": [],
    }
    for kind in ["standard", "xsa", "fused", "linear"]:
        print(f"\n=== Training {kind} ===")
        try:
            r = train(kind, dim=args.dim, length=args.length, vocab=args.vocab,
                          epochs=args.epochs, size=args.size, device=device)
            print(f"  train_loss={r['train_loss']:.4f} val_loss={r['val_loss']:.4f} acc={r['accuracy']:.2f} wall={r['wall_seconds']:.1f}s")
            r_clean = {k: v for k, v in r.items() if k != "loss_curve"}
            r_clean["loss_curve_length"] = len(r["loss_curve"])
            results["results"].append(r_clean)
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
