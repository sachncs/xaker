"""Attention module for xaker.

Public surface for the attention subpackage. Provides:

* :class:`Standard` -- Vaswani-style scaled dot-product baseline.
* :class:`Xsa` -- XSA, which removes self-aligned components from outputs.
* :class:`Fused` -- flagship block combining XSA with kernel
  attention solved by Preconditioned Conjugate Gradient.
* :class:`Linear` -- linear-complexity attention baseline
  (Katharopoulos et al., 2020). Reference comparison.
* :class:`Kernel` -- exponential attention kernel used by :class:`Fused`.

Shared utilities (:class:`Base`, :class:`Qkv`, mask helpers) live
in :mod:`xaker.attention.core`; kernel implementations live in
:mod:`xaker.attention.kernel` and :mod:`xaker.attention.func`.

The polymorphism registry lives here: ``BLOCK = {"standard": Standard,
"xsa": Xsa, "fused": Fused, "linear": Linear}`` maps a kind string
to the concrete class. Adding a new attention variant is one class
+ one entry in ``BLOCK``.
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
from xaker.attention.linear import Linear
from xaker.attention.standard import Standard
from xaker.attention.xsa import (
    Mask,
    Projection,
    Xsa,
    XsaMode,
    XsaStrategy,
    Zero,
)
from xaker.attention.fused import Fused

BLOCK: dict[str, type[Base]] = {
    "standard": Standard,
    "xsa": Xsa,
    "fused": Fused,
    "linear": Linear,
}

__all__ = [
    "Base",
    "BLOCK",
    "Fused",
    "Kernel",
    "Linear",
    "Mask",
    "Projection",
    "Qkv",
    "Standard",
    "Xsa",
    "XsaMode",
    "XsaStrategy",
    "Zero",
    "broadcast",
    "heads",
    "keep",
    "kernel",
    "merge",
]