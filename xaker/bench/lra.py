"""Long Range Arena-style benchmark for xaker.

Five synthetic tasks that stress long-context memory and reasoning:

1. ListOps -- hierarchical operator-precedence parsing.
2. Copy -- reproduce a long sequence verbatim.
3. Reversal -- reproduce a sequence in reverse.
4. Addition -- sum two long numbers digit-by-digit.
5. Retrieval -- find a key-value pair from a list.

These are mini-LRA tasks designed to run on CPU. Each task is scored
by validation accuracy; per-task numbers feed into ``paper_runs/lra.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from xaker import Config, Model
from xaker.bench.bench import gitsha
from xaker.training.loss import ce


def listops(vocab: int, length: int, size: int, seed: int = 0) -> Dataset:
    """Generate ListOps-style nested operator sequences.

    Each sequence encodes a small arithmetic expression with + and
    max; the target is the integer result. Simplified for the
    byte-level setting.

    Args:
        vocab: Vocabulary size (>= 14: digits 0-9 + [, ], +, max, pad).
        length: Sequence length.
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Dataset yielding ``(seq, target)`` where ``seq`` is the
        expression as a length-``length`` token tensor and
        ``target`` is a single integer.
    """
    raise NotImplementedError("ListOps requires a richer tokenizer than the byte level; skipped in v0")


def copy(vocab: int, length: int, size: int, seed: int = 0) -> Dataset:
    """Copy task: input = output.

    Args:
        vocab: Vocabulary size.
        length: Sequence length.
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Dataset of ``(x, y)`` token tensors.
    """
    class _DS(Dataset):
        def __init__(self):
            gen = torch.Generator().manual_seed(seed)
            self.x = torch.randint(0, vocab, (size, length), generator=gen)
            self.y = self.x.clone()
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.y[i]
    return _DS()


def reversal(vocab: int, length: int, size: int, seed: int = 0) -> Dataset:
    """Reversal task: output = input reversed.

    Args:
        vocab: Vocabulary size.
        length: Sequence length.
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Dataset of ``(x, y)`` token tensors.
    """
    class _DS(Dataset):
        def __init__(self):
            gen = torch.Generator().manual_seed(seed)
            self.x = torch.randint(0, vocab, (size, length), generator=gen)
            self.y = self.x.flip(dims=[1])
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.y[i]
    return _DS()


def addition(vocab: int, length: int, size: int, seed: int = 0) -> Dataset:
    """Long-addition task: two numbers of half_length digits each.

    Args:
        vocab: Vocabulary size (>= 10).
        length: Total sequence length (must be even).
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Dataset of ``(x, y)`` token tensors.
    """
    half = length // 2
    class _DS(Dataset):
        def __init__(self):
            gen = torch.Generator().manual_seed(seed)
            self.x = torch.zeros(size, length, dtype=torch.long)
            self.y = torch.zeros(size, length, dtype=torch.long)
            for i in range(size):
                a = torch.randint(0, vocab, (half,), generator=gen)
                b = torch.randint(0, vocab, (half,), generator=gen)
                # Build inputs: a || b
                self.x[i] = torch.cat([a, b])
                # Compute sum digit-by-digit (right-aligned)
                carry = 0
                out = torch.zeros(half, dtype=torch.long)
                for j in range(half - 1, -1, -1):
                    s = int(a[j]) + int(b[j]) + carry
                    out[j] = s % vocab
                    carry = s // vocab
                # Pad sum to length
                self.y[i] = torch.cat([torch.zeros(half, dtype=torch.long), out])
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.y[i]
    return _DS()


def retrieval(vocab: int, length: int, size: int, seed: int = 0) -> Dataset:
    """Key-value retrieval: at end of sequence, output value matching earlier key.

    Args:
        vocab: Vocabulary size.
        length: Sequence length (must be at least 6 for the format).
        size: Number of samples.
        seed: RNG seed.

    Returns:
        Dataset of ``(x, y)`` token tensors.
    """
    class _DS(Dataset):
        def __init__(self):
            gen = torch.Generator().manual_seed(seed)
            self.x = torch.zeros(size, length, dtype=torch.long)
            self.y = torch.zeros(size, length, dtype=torch.long)
            for i in range(size):
                # Format: [noise...][key][noise...][value][query_key][...padding]
                # Token layout: half noise, key, value, query, answer
                key = torch.randint(0, vocab, (1,), generator=gen).item()
                value = torch.randint(0, vocab, (1,), generator=gen).item()
                idx_key = length // 4
                idx_val = length // 2
                idx_query = 3 * length // 4
                idx_answer = length - 1
                self.x[i, idx_key] = key
                self.x[i, idx_val] = value
                self.x[i, idx_query] = key
                # If query matches key, output value; else 0
                self.y[i, idx_answer] = value
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.y[i]
    return _DS()


TASKS = {
    "copy": copy,
    "reversal": reversal,
    "addition": addition,
    "retrieval": retrieval,
}


def train(
    task_name: str,
    kind: Literal["standard", "xsa", "fused", "linear"],
    *,
    dim: int = 32,
    length: int = 32,
    vocab: int = 16,
    epochs: int = 5,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one attention kind on one LRA-style task.

    Args:
        task_name: One of the registered task functions.
        kind: Attention kind.
        dim: Model width.
        length: Sequence length.
        vocab: Vocabulary size.
        epochs: Number of training epochs.
        device: Training device.

    Returns:
        Dict with ``task``, ``kind``, ``accuracy``, ``val_loss``,
        ``wall_seconds``.
    """
    torch.manual_seed(0)
    task_fn = TASKS[task_name]
    train_ds = task_fn(vocab, length, 64)
    val_ds = task_fn(vocab, length, 32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    cfg = Config(dim=dim, heads=4, drop=0.0)
    model = Model(cfg, num_layers=2, vocab_size=vocab, max_seq_len=length, attention_type=kind)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_losses = []
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
            train_losses.append(loss.item())
    wall = time.perf_counter() - t0
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y, smoothing=0.0)
            val_losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return {
        "task": task_name,
        "kind": kind,
        "dim": dim,
        "length": length,
        "epochs": epochs,
        "train_loss": sum(train_losses) / max(len(train_losses), 1),
        "val_loss": sum(val_losses) / max(len(val_losses), 1),
        "accuracy": correct / max(total, 1),
        "wall_seconds": wall,
    }


def main() -> int:
    """Command-line entrypoint.
    
    Parses CLI args, runs the benchmark, and writes JSON output.
    """
    parser = argparse.ArgumentParser(description="LRA-style benchmark sweep")
    parser.add_argument("--out", default="paper_runs/lra.json")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(os.environ.get("XAKER_DEVICE", "cpu"))
    print(f"LRA benchmark on {device}, dim={args.dim}, length={args.length}, epochs={args.epochs}")
    results: dict = {
        "config": {"device": str(device), "dim": args.dim, "length": args.length, "epochs": args.epochs},
        "git_sha": gitsha(), "torch_version": torch.__version__,
        "tasks": {},
    }
    kinds: list[Literal["standard", "xsa", "fused", "linear"]] = ["standard", "xsa", "fused", "linear"]
    for task in ["copy", "reversal", "retrieval", "addition"]:
        task_results: list[dict] = []
        results["tasks"][task] = task_results
        vocab = 16 if task != "addition" else 10
        for kind in kinds:
            try:
                print(f"\n=== {task} / {kind} ===")
                r = train(task, kind, dim=args.dim, length=args.length, vocab=vocab, epochs=args.epochs, device=device)
                print(f"  acc={r['accuracy']:.4f} val_loss={r['val_loss']:.4f} wall={r['wall_seconds']:.1f}s")
                task_results.append(r)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                task_results.append({"task": task, "kind": kind, "error": str(exc)})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
