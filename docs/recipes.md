---
layout: page
title: Recipes
description: Common patterns and answer-to-stack-overflow recipes for tinkerers and contributors.
permalink: /recipes/
---

Each recipe is a self-contained answer to "how do I do X with
xaker?" The audience is a tinkerer who has read
[Getting started](/xaker/getting-started/) and the
[Tutorial](/xaker/tutorial/) and now wants to push the package
somewhere it has not been pushed before.

> **Heads-up.** Every recipe below assumes the source install
> (`pip install -e '.[dev]'`) is in place. The PyPI distribution is
> not yet available; see [Install](/xaker/installation/#troubleshooting).

## Recipe: add a new attention variant

The polymorphic `BLOCK` registry makes a new variant cheap:

1. Subclass `xaker.attention.core.Base`.
2. Implement `attend(self, q, k, v, m) -> Tensor`.
3. Add `from xaker.attention.my_variant import MyVariant` to
   `xaker/attention/__init__.py`.
4. Register the entry in `BLOCK: dict[str, type[Base]]`.

```python
# xaker/attention/my_variant.py
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from xaker.attention.core import Base


class MyVariant(Base):
    """One-sentence description of the variant."""

    def attend(self, q: Tensor, k: Tensor, v: Tensor,
               m: Optional[Tensor]) -> Tensor:
        # Write the math here.
        return ...
```

```python
# xaker/attention/__init__.py
from xaker.attention.my_variant import MyVariant

BLOCK["my_variant"] = MyVariant     # add this line
```

```python
# anywhere
from xaker import BLOCK
attn = BLOCK["my_variant"](cfg)
```

Tests should land in a sibling file `tests/test_my_variant.py` and
cover the four standard guards: shape, finiteness, gradient flow,
and dispatch. Use `tests/test_dispatch.py` as the template.

## Recipe: add a new preconditioner

Preconditioners plug into `xaker/solver/precond.py`. The contract is
a single `nn.Module` with two methods:

```python
# xaker/solver/precond.py
from torch import nn

class MyPrecond(nn.Module):
    """One-sentence description of the preconditioner."""

    def __init__(self, config):
        super().__init__()
        # Read whatever you need from ``config``.
        self.foo = nn.Parameter(torch.zeros(...))
        self.bar = config.bar

    def build(self, kernel, lam, length):
        # Return a ``Cache`` with whatever payload
        # ``apply_pre`` needs.
        return Cache(data=...)

    def apply_pre(self, residual, data):
        # Use ``data`` and your parameters to transform ``residual``.
        return ...
```

Add the new entry to `MODE`:

```python
MODE = {
    "identity": Identity,
    "diagonal": Diagonal,
    "fast": Fast,
    "cccp": Cccp,
    "my_precond": MyPrecond,    # add this line
}
```

The factory `Make(config)` reads `config.precond` and dispatches
automatically. The 0.4.0 single-word naming rule applies: do not
prefix the class with `_` and do not import it `as`.

## Recipe: add a new kernel function

Kernels live in `xaker/attention/func.py`. The contract is one
stateless function with the signature `(q, k, config) -> Tensor`:

```python
# xaker/attention/func.py
def my_kernel(q, k, *, config):
    """Compute K[i,j] = some_function(q[i], k[j])."""
    ...
```

`Kernel` (the stateful wrapper used by `Fused`) reads the same
function name in `__init__` and stores any learned parameter.
Adding a new `kernel` value also requires touching the
`Literal[...]` in `Config.kernel` and the validator in
`Config.__post_init__`.

## Recipe: log the PCG convergence trajectory

The `Solve` dataclass returns `history: List[float]` with one entry
per iteration:

```python
from xaker.solver.cg import pcg

# Inside the Fused block, after building the kernel:
solve = pcg(
    kernel=kernel, b=v, lam=lam,
    precond_data=data, apply_pre=precon.apply_pre,
    iters=cfg.pcg, tol=cfg.tol, miniters=3,
)
print(solve.history)   # e.g. [0.498, 0.341, 0.213, 0.097, 0.012]
```

The convergence trajectory is the same data the paper uses to
plot "iterations to `tol`"; if you want to log it from a training
loop, monkey-patch `pcg` like the tutorial does, or wrap
`Fused.attend` in a thin logging shell.

## Recipe: benchmark a single kernel variant

The fastest way to compare one variant against the others on the
same hardware is `xaker/bench/ablate.py`:

```bash
python -m xaker.bench.ablate \
    --axis kind \
    --values standard xsa fused linear \
    --length 32 \
    --out paper_runs/abl_kind.json
```

Use `--axis kernel` to sweep kernels (exp, rbf, linear, cosine),
`--axis precond` to sweep preconditioners, `--axis mode` to sweep
XSA modes. The JSON output schema is documented in
`xaker.bench.bench.Result`.

## Recipe: serialise a checkpoint

Use the `--out` flag added in 0.5.2:

```bash
xaker-train --dim 32 --heads 4 --layers 2 --epochs 1 \
            --out artifacts/last.pt
xaker-eval  --checkpoint artifacts/last.pt --kind fused
```

The checkpoint is a vanilla `state_dict`; load it from Python with
the standard `torch.load(...)` + `model.load_state_dict(...)` flow.

## Recipe: keep the Fast preconditioner cache fresh

`Fast.build` returns a cached payload when `freq > 1` and the
learnable factors are untouched. Optimizer steps usually reallocate
storage, so the cache is invalidated automatically. To force a
rebuild for any other reason:

```python
from xaker.attention import Fused
fused = Fused(cfg)
fused.precon.reset_cache()    # Fast.reset_cache
```

`reset_cache` clears the payload and resets the step counter;
the next `Fused` forward pass rebuilds the preconditioner from
scratch.

## Recipe: profile one block end-to-end

```python
import torch, time
from xaker import Config, BLOCK

cfg = Config(dim=256, heads=8, precond="fast", rank=16)
attn = BLOCK["fused"](cfg).eval()
x = torch.randn(4, 64, cfg.dim)

# Warm
for _ in range(5):
    with torch.no_grad():
        _ = attn(x)

torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        _ = attn(x)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"{(time.perf_counter() - t0) / 20 * 1000:.2f} ms / forward")
```

The package ships a strict-mode-friendly version of this same
script as `xaker/bench/bench.py:tick`; the JSON output of
`xaker-bench` is the file you want to compare across commits.

## Recipe: contribute a bug fix

1. **Reproduce** in a one-off script under `examples/` or as a
   failing test under `tests/`.
2. **Branch** off `master` with the prefix from
   [CONTRIBUTING.md on GitHub][contrib] (`fix/`, `feat/`, etc.).
3. **Fix** the smallest possible change. Add a regression test
   that fails *before* the fix and passes *after*.
4. **Verify** with `pytest tests/ -v --cov=xaker
   --cov-fail-under=90 --strict-markers -m "not slow"`. The
   `--cov-fail-under=90` number is enforced in CI.
5. **Open a PR**. Reference issues with `Closes #N` in the
   body so the linked issues close automatically when the PR
   merges.

[contrib]: https://github.com/sachncs/xaker/blob/master/CONTRIBUTING.md

## Where to go next

- [Architecture](/xaker/architecture/) — module tree, polymorphism
  registries, the `Fused` pipeline in depth.
- [Mathematical foundations](/xaker/math/) — derivations matching every
  public function.
- [Design decisions](/xaker/design_decisions/) — what the code chose,
  and the why.
- [FAQ](/xaker/faq/) — short answers to common questions.
