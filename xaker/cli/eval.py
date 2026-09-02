"""xaker-eval: load a checkpoint and run a smoke forward pass."""

from __future__ import annotations

import argparse
import sys

import torch

from xaker.config import Config
from xaker.model.model import Model
import xaker.utils.rng


def main(argv: list[str] | None = None) -> int:
    """Load a checkpoint and run a smoke forward pass.

    Args:
        argv: Optional argument vector; ``None`` reads from
            ``sys.argv``.

    Returns:
        Process exit code; ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Evaluate XAKER checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    xaker.utils.rng.seed(42)

    cfg = Config(dim=64, heads=4)
    model = Model(cfg, num_layers=2, vocab_size=args.vocab, max_seq_len=args.length, attention_type="fused_v2")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    x = torch.randint(0, args.vocab, (args.batch, args.length), device=device)
    with torch.no_grad():
        out = model(x)
    print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
