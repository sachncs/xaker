"""Training configuration, step orchestration, and label-smoothed loss.

``Trainer`` constructs AdamW when no optimizer is supplied and can step an
injected scheduler. Callers provide dataloaders and orchestrate epochs, logging,
evaluation cadence, and checkpointing.
"""

from __future__ import annotations

from xaker.training.trainer import Trainer, Fit
from xaker.training.loss import ce

__all__ = [
    "Trainer",
    "Fit",
    "ce",
]
