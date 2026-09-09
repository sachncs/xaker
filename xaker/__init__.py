"""Public API for xaker attention, solver, model, training, and rubric modules.

xaker is a Python library that implements Exclusive Self Attention
(XSA) for Transformer models, with a polymorphic strategy-based
dispatch and a single-word public API.

Public surface:
- :class:`Config` — single-word dataclass for attention/solver params.
- :class:`Standard`, :class:`Xsa`, :class:`Fused`, :class:`Kernel` —
  attention modules; :class:`Projection`, :class:`Zero`,
  :class:`Mask` — XSA strategies.
- :class:`Block`, :class:`Mlp`, :class:`Model` — Transformer building blocks.
- :class:`Trainer`, :class:`Fit` — training utilities.
- :func:`pcg`, :func:`richardson`, :class:`Solve` — iterative solvers.
- :func:`kernel`, :func:`op` — stateless functional forms.
- :func:`causal`, :func:`padding`, :func:`shape`, :func:`clamp` — utilities.
- :func:`finite` — finite-value check.
- :func:`seed`, :func:`snapshot`, :func:`restore` — RNG control.
- :class:`Ctx` — typed execution context.
- :class:`Rubric`, :class:`Score`, :class:`Dimension`, :func:`grade` — paper rubric.

Example:
    >>> from xaker import Fused, Config
    >>> import torch
    >>> cfg = Config(dim=64, heads=4)
    >>> attn = Fused(cfg)
    >>> out = attn(torch.randn(2, 8, 64))
"""

from xaker.config import Config
from xaker.attention import (
    BLOCK,
    Base,
    Fused,
    Kernel,
    Linear,
    Mask,
    Projection,
    Qkv,
    Standard,
    Xsa,
    Zero,
)
from xaker.attention.func import kernel
from xaker.attention.ops import zerodiag, rms
from xaker.model.block import Block, Mlp
from xaker.model.model import Model
from xaker.training.loss import ce
from xaker.training.trainer import Fit, Trainer
from xaker.solver.cg import Solve, pcg, richardson
from xaker.solver.func import op
from xaker.solver.precond import (
    BOUND,
    Cache,
    Cccp,
    Diagonal,
    Fast,
    Identity,
    Make,
    PrecondProto,
)
from xaker.utils.finite import finite
from xaker.utils.ops import causal, clamp, padding, shape
from xaker.utils.rng import restore, seed, snapshot
from xaker.utils.ctx import Ctx, toctx

__all__ = [
    "BOUND",
    "Base",
    "Block",
    "Cache",
    "Cccp",
    "Config",
    "Ctx",
    "Diagonal",
    "Fast",
    "Fit",
    "Fused",
    "Identity",
    "Kernel",
    "Linear",
    "Make",
    "Mask",
    "Mlp",
    "Model",
    "PrecondProto",
    "Projection",
    "Qkv",
    "Solve",
    "Standard",
    "Trainer",
    "Xsa",
    "Zero",
    "causal",
    "ce",
    "clamp",
    "finite",
    "kernel",
    "op",
    "padding",
    "pcg",
    "restore",
    "richardson",
    "rms",
    "seed",
    "shape",
    "snapshot",
    "toctx",
    "zerodiag",
]

__version__ = "0.5.1"
