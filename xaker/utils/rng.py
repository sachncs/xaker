"""Global random-number-generator seeding and state snapshots.

The helpers cover Python, NumPy, PyTorch CPU, and visible CUDA generators.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy
import torch


def seed(s: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    When CUDA is available, additionally sets
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` for better reproducibility.
    """
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def snapshot() -> Dict[str, Any]:
    """Snapshot the current RNG state.

    Captures Python, NumPy, PyTorch CPU, and CUDA RNG states.
    """
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore(states: Dict[str, Any]) -> None:
    """Restore RNG states from a snapshot.

    Requires ``python``, ``numpy``, ``torch`` keys. Optional ``torch_cuda``.
    """
    if "python" not in states:
        raise KeyError("Missing 'python' key in states")
    if "numpy" not in states:
        raise KeyError("Missing 'numpy' key in states")
    if "torch" not in states:
        raise KeyError("Missing 'torch' key in states")
    random.setstate(states["python"])
    numpy.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])