"""Dataset loaders for the xaker paper experiments.

Three datasets are supported, all CPU-friendly:

- :class:`CopyTask` -- Synthetic copy task. Each sequence is
  random tokens; the target is the same sequence. Used in the
  paper's headline experiments.
- :class:`WikiText` -- WikiText-2 character-level language
  modelling. Loaded via the ``datasets`` library and cached
  locally.
- :class:`ReversalTask` -- Reversal task: predict the input in
  reverse order. Tests whether attention can learn position-aware
  structure.

The loaders return :class:`torch.utils.data.Dataset` instances that
the existing :class:`Trainer` can consume. Tokenisation is
character-level for reproducibility on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset


CACHE = Path(os.environ.get("XAKER_CACHE", Path.home() / ".cache" / "xaker"))


class CopyTask(Dataset):
    """Synthetic copy task.

    Each sample is a sequence of ``length`` random token IDs in
    ``[0, vocab)``. The target is the input itself.
    """

    def __init__(self, vocab: int, length: int, size: int, seed: int = 0) -> None:
        """Generate ``size`` random samples with the given seed.

        Args:
            vocab: Vocabulary size.
            length: Sequence length.
            size: Number of samples.
            seed: RNG seed for reproducibility.
        """
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, vocab, (size, length), generator=gen)
        self.y = self.x.clone()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ReversalTask(Dataset):
    """Reversal task: predict the input reversed.

    Tests whether attention learns positional structure: the i-th
    output must reconstruct the (length-1-i)-th input token.
    """

    def __init__(self, vocab: int, length: int, size: int, seed: int = 0) -> None:
        """Generate ``size`` random samples with the given seed.

        Args:
            vocab: Vocabulary size.
            length: Sequence length.
            size: Number of samples.
            seed: RNG seed.
        """
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, vocab, (size, length), generator=gen)
        self.y = self.x.flip(dims=[1])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class WikiText(Dataset):
    """WikiText-2 character-level language modelling dataset.

    Wraps the Hugging Face ``datasets`` library. Tokens are
    characters (vocab = 256 byte values). Falls back to a synthetic
    corpus if the download fails so the suite stays runnable on
    offline machines.
    """

    def __init__(self, split: str = "train", length: int = 128, cache: bool = True) -> None:
        """Load WikiText-2 and tokenise at the character level.

        Args:
            split: One of ``"train"``, ``"validation"``, ``"test"``.
            length: Sequence length for each sample.
            cache: If ``True``, write the cached token tensor to
                ``~/.cache/xaker/wikitext_<split>.pt``.
        """
        self.length = length
        tokens = self._load(split, cache)
        # Chunk into fixed-length samples
        n_samples = len(tokens) // length
        tokens = tokens[: n_samples * length]
        self.x = tokens.view(n_samples, length)
        self.y = torch.roll(self.x, shifts=-1, dims=1)

    def _load(self, split: str, cache: bool) -> torch.Tensor:
        """Download or load cached tokens.

        Args:
            split: ``"train"`` / ``"validation"`` / ``"test"``.
            cache: Whether to write the cache file.

        Returns:
            1-D byte-valued tensor of length ``(n_samples * length)``.
        """
        cache_path = CACHE / f"wikitext_{split}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            assert isinstance(cached, torch.Tensor)
            return cached

        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n".join(x["text"] for x in ds if x["text"].strip())
        except Exception:
            # Offline fallback: a deterministic synthetic corpus
            text = _synthetic_corpus(int(1e5))
        tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, cache_path)
        return tokens

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _synthetic_corpus(size: int) -> str:
    """Deterministic synthetic text for offline fallback.

    Args:
        size: Approximate number of characters.

    Returns:
        Pseudo-text string of length ``~size``.
    """
    base = (
        "the quick brown fox jumps over the lazy dog. "
        "xaker is a python library for self-attention. "
        "exclusive self attention removes the projection of the "
        "output onto each token's value vector. "
    )
    out: List[str] = []
    total = 0
    while total < size:
        out.append(base)
        total += len(base)
    return "".join(out)[:size]


def build(name: str, **kwargs: Any) -> Dataset[Any]:
    """Factory for dataset construction.

    Args:
        name: One of ``"copy"``, ``"reversal"``, ``"wikitext"``.
        ``**kwargs``: Forwarded to the corresponding class.

    Returns:
        A :class:`torch.utils.data.Dataset`.
    """
    table: dict[str, type[Dataset[Any]]] = {
        "copy": CopyTask,
        "reversal": ReversalTask,
        "wikitext": WikiText,
    }
    if name not in table:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(table)}")
    return table[name](**kwargs)


def vocab(name: str) -> int:
    """Return the vocabulary size for a given dataset.

    Args:
        name: Dataset name.

    Returns:
        Vocabulary size (256 for WikiText byte-level, configurable
        otherwise).
    """
    if name == "wikitext":
        return 256
    if name in ("copy", "reversal"):
        return 64
    raise ValueError(f"Unknown dataset: {name}")