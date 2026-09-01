"""Attention module for XAKER.

Public surface for the attention subpackage. Provides:

* :class:`Standard` — Vaswani-style scaled dot-product baseline.
* :class:`Xsa` — XSA, which removes self-aligned components from outputs.
* :class:`Laker` — v2 module combining XSA-related transformations with
  preconditioned kernel-regression inverse mixing.
* :class:`Kernel` — exponential attention kernel used by :class:`Laker`.

Shared utilities (:class:`Base`, :class:`Qkv`, mask helpers,
``reshape_*``) live in :mod:`xaker.attention.core`; kernel implementations
live in :mod:`xaker.attention.kernel` and :mod:`xaker.attention.func`.

The polymorphism registry lives here: ``BLOCK = {"standard": Standard,
"xsa": Xsa, "fused": Laker}`` maps a kind string to the concrete class.
Adding a new attention variant is one class + one entry in ``BLOCK``.
"""

from __future__ import annotations

from xaker.attention.core import (
    Base,
    Qkv,
    keep,
    broadcast,
    heads,
    merge,
)
from xaker.attention.func import kernel
from xaker.attention.kernel import Kernel
from xaker.attention.standard import Standard
from xaker.attention.xsa import (
    Mask,
    Projection,
    Xsa,
    XsaMode,
    XsaStrategy,
    Zero,
)
from xaker.attention.laker import Laker

BLOCK = {
    "standard": Standard,
    "xsa": Xsa,
    "fused": Laker,
}

__all__ = [
    "Base",
    "BLOCK",
    "Kernel",
    "Laker",
    "Mask",
    "Projection",
    "Qkv",
    "Standard",
    "Xsa",
    "XsaMode",
    "XsaStrategy",
    "Zero",
    "keep",
    "broadcast",
    "compute_kernel_matrix",
    "heads",
    "merge",
]