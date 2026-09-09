---
layout: page
title: Tutorial
description: Build and inspect a four-block Transformer from scratch; learn the Fused pipeline by changing it.
permalink: /tutorial/
---

**Audience: new contributors and graduate students who want to follow one full pass through the `Fused` pipeline.**

**Time: 15 minutes.**

This is the guided walk-through. If you just want to run a forward
pass, see [Getting started](/xaker/getting-started/). If you want to
poke at the internals before opening a PR, see
[Architecture](/xaker/architecture/); this page sits between them.

The model in this tutorial is a four-block, length-16 toy
Transformer running on CPU; a modern laptop finishes the entire
walk-through in roughly a minute.

## 1. Construct a Config

The `Config` dataclass is the single entry point for every
hyperparameter. Validation runs in `__post_init__`, so an
inconsistent value raises `ValueError` immediately:

```python
from xaker import Config

cfg = Config(
    dim=32,
    heads=4,            # dim % heads == 0, so headdim = 8
    precond="fast",
    rank=4,             # low-rank preconditioner size
)
```

| Field | Default | Why we change it here |
|---|---|---|
| `dim` | required | 32 keeps the matrix multiplication cheap for the tutorial. |
| `heads` | required | 4 produces eight-dimensional per-head views. |
| `precond` | `"fast"` | The default, but we spell it out so the ship-the-numbers argument is visible. |
| `rank` | `32` | 4 is enough for the toy kernel. |
| `kernel` | `"exp"` | Default. The exponential kernel paired with cosine-similarity q/k is the paper’s canonical case. |
| `mode` | `"subtract"` | Default. Other options are `"zero"` and `"mask"`. |

## 2. Build an attention block

Every attention variant ships behind a single factory:

```python
from xaker import BLOCK

attn = BLOCK["fused"](cfg)
print(attn)
# Fused(
#   (qkv_proj): Qkv(...)
#   (w_o): Linear(...)
#   (kernel_fn): Kernel()
#   (precon): Fast()
# )
```

`BLOCK` is a polymorphic registry that maps a kind string to the
concrete class. Adding a new variant is one class plus one entry
in the dispatch table; see [Architecture § Polymorphism
registries](/xaker/design_decisions/#polymorphism-over-mode-strings) for the
machinery.

The `precon: Fast()` allocation you see in the dump is a
**per-block** instance. Each `Fused` block builds its own
`Make(config)` strategy, so the preconditioner's step counter
(`Fast.iter`) is local to the block, even in a deep Transformer.
This is intentional: a shared counter would let one block’s
schedule churn poison another block’s cache.

## 3. Run a forward pass

```python
import torch

x = torch.randn(2, 16, cfg.dim)   # (batch, seq_len, dim)
out = attn(x)
print(out.shape)                  # (2, 16, 32)
```

The forward pass projects `x` to `q`, `k`, `v`, builds the kernel,
computes the diagonal-removed form (XSA), runs PCG against
`(K + lam I) alpha = v`, clamps, RMS-normalises, and applies the
output-projection strategy. `docs/math.md` walks through every
step; the next three sections give the high level.

### 3a. Inside `Fused.attend`

```python
def attend(self, q, k, v, m):
    _, _, length, _ = q.shape
    kernel = self.kernel_fn(q, k)               # (b, h, n, n)
    kernel = keep(kernel, m, fill=-1e9) if m is not None else zerodiag(kernel)
    lam = self.lam.view(1, 1, 1, 1)
    data = self.precon.build(kernel, lam, length)
    solve = pcg(kernel=kernel, b=v, lam=lam,
                precond_data=data,
                apply_pre=self.precon.apply_pre,
                iters=cfg.pcg, tol=cfg.tol, miniters=3)
    if not solve.converged or not torch.isfinite(solve.x).all():
        # Fall back to a single dense solve.
        eye = torch.eye(length, device=kernel.device, dtype=kernel.dtype)
        solve_x = torch.linalg.solve(kernel + lam * eye, v)
    else:
        solve_x = solve.x
    out = torch.clamp(solve_x, -BOUND, BOUND)
    out = rms(out, cfg.eps)
    self.xsa.apply(out, v)
    # Bump Fast step counter; per-block, never shared.
    return self.xsa.apply(out, v)
```

Note that the diagonal-removal step (`zerodiag(kernel)`) is the
defining move of XSA — it stops the kernel from contributing a
self-aligned component to each output row. The diagonal entries
are zeroed before the matrix is solved, not after softmax.

## 4. Stack four blocks into a Transformer

The four-block Transformer is one line away:

```python
from xaker import Model

model = Model(
    cfg,
    num_layers=4,
    vocab_size=100,
    max_seq_len=16,
    attention_type="fused",
)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
# parameters: 49,540  (approximate)
```

The `Model` class wires the four `Fused` blocks together with
pre-norm residual streams, an MLP between blocks, and a tied
embedding. The architecture is documented in
[Architecture](/xaker/architecture/#fused-pipeline).

## 5. Train for one pass

```python
from xaker import Fit, Trainer

fit = Fit(epochs=1, lr=1e-3, decay=0.1)
trainer = Trainer(model, fit, torch.device("cpu"))

batch = torch.randint(0, 100, (8, 16))
labels = batch.clone()
metrics = trainer.step((batch, labels))
print(metrics)
# {'loss': 4.605...}   # uniform from a 100-token vocab
```

`Trainer.step` runs forward + backward + AdamW, clips the gradient
norm to `Fit.grad`, and advances any scheduler you supplied. Run
`Trainer.epoch(loader)` to aggregate a full pass.

## 6. Inspect the PCG solve

The `Fused` block returns a `Solve` dataclass every forward pass.
You can grab it by monkey-patching `pcg` for one call:

```python
from xaker.solver.cg import pcg as real_pcg
from xaker.solver.cg import Solve

solves = []

def hook(kernel, b, lam, **kwargs):
    solve = real_pcg(kernel=kernel, b=b, lam=lam, **kwargs)
    solves.append(solve)
    return solve

import xaker.attention.fused as fused_mod
fused_mod.pcg = hook
_ = model(torch.randint(0, 100, (1, 16)))
print(solves[0])
# Solve(x=tensor(...), iters=4, converged=True, res=0.003, history=[...])
```

The `history` list has one entry per iteration with the relative
residual — useful for plotting the convergence rate.

> **Tip:** if you want a guaranteed-converged trace, run the kernel
> with `precond="diagonal"` and you will typically see 3–6
> iterations to `tol=1e-2`. `precond="identity"` is the same but
> without the speedup; useful for sanity-checking your data shapes.

## 7. Swap the kernel

The four-kernel switch is one config field away:

```python
exp_cfg = cfg                                               # already exp
rbf_cfg  = type(cfg)(dim=32, heads=4, kernel="rbf",  rank=4)
lin_cfg  = type(cfg)(dim=32, heads=4, kernel="linear", rank=4)
cos_cfg  = type(cfg)(dim=32, heads=4, kernel="cosine", rank=4)

for label, variant in [("exp", exp_cfg), ("rbf", rbf_cfg),
                       ("linear", lin_cfg), ("cosine", cos_cfg)]:
    block = BLOCK["fused"](variant)
    out = block(torch.randn(1, 16, 32))
    print(f"{label:6s} -> finite={torch.isfinite(out).all().item()}")
```

If you want to add your own, see
[Architecture § Kernel choice](/xaker/design_decisions/#kernel-choice) — it
is one function with the right signature.

## 8. Swap the preconditioner

Same drill:

```python
for precond in ("identity", "diagonal", "fast", "cccp"):
    sub_cfg = type(cfg)(dim=32, heads=4, precond=precond, rank=4)
    block = BLOCK["fused"](sub_cfg)
    out = block(torch.randn(1, 16, 32))
    print(f"{precond:9s} -> finite={torch.isfinite(out).all().item()}")
```

`cccp` builds an O(n³) covariance and is the slowest preconditioner
to set up; it converges fastest on ill-conditioned kernels.
`diagonal` is the cheapest; `fast` is the default for production.

## 9. Reproduce a benchmark

The benchmarks are typed, schema-stable JSON outputs. Reproducing
the headline condition-number comparison is one command:

```bash
python -m xaker.bench.condition \
    --lam 10.0 --lengths 16 32 64 128 --out paper_runs/condition.json
```

Five YAML specs under `examples/specs/` drive the typed experiment
driver:

```bash
python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml
```

Both commands exit 0 and emit a JSON file under `paper_runs/`. The
schema is documented in `xaker/bench/__init__.py`; the headline
tables are in `RESULTS.md`.

## Where to go next

- [Architecture](/xaker/architecture/) — module tree, the `Fused`
  pipeline, the preconditioner factories, what ties everything
  together.
- [Mathematical foundations](/xaker/math/) — the derivation that backs
  the code, with byte-equivalent formulas.
- [API reference](/xaker/api/) — every public symbol, every `Config`
  field, every ablation runner.
- [Design decisions](/xaker/design_decisions/) — the why behind the
  polymorphism registers, the dtype contract, and the single-word
  naming rule.
