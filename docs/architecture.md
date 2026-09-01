# Architecture

XAKER (XSA + LAKER) fuses two complementary attention mechanisms
into a single module solved with a preconditioned iterative method.

## Module tree

```
xaker/
├── __init__.py
├── config.py                Config dataclass
├── attention/
│   ├── core.py              Base, Qkv, keep, merge, broadcast, heads
│   ├── func.py              kernel
│   ├── kernel.py            Kernel
│   ├── laker.py             Laker
│   ├── ops.py               rms, zerodiag
│   ├── standard.py          Standard
│   └── xsa.py               Xsa, XsaStrategy, Projection, Zero, Mask
├── bench/
│   ├── bench.py             Spec, Result, Metrics, tick, peak, converge, run, write, gitsha
│   └── __init__.py
├── cli/
│   ├── bench.py             xaker-bench CLI
│   ├── eval.py              xaker-eval CLI
│   ├── train.py             xaker-train CLI
│   └── validate.py          xaker-validate CLI
├── model/
│   ├── block.py             Block, Mlp
│   └── model.py             Model
├── rubric/
│   ├── grader.py            grade, GRADERS dict
│   ├── plugin.py            pytest plugin providing rubric fixture
│   ├── reporting.py         markdown, write
│   └── rubric.py            Rubric, Score, Dimension
├── solver/
│   ├── cg.py                pcg, richardson, Solve
│   ├── func.py              op
│   └── precond.py           Make, Identity, Diagonal, Fast, Cccp, Cache, BOUND
├── training/
│   ├── loss.py              ce
│   └── trainer.py           Trainer, Fit
└── utils/
    ├── ctx.py               Ctx, toctx
    ├── finite.py            finite
    ├── ops.py                causal, padding, shape, clamp
    └── rng.py                seed, snapshot, restore
```

## Polymorphism registries

Three places where behavior varies through a name lookup, not
a string of `if/elif` branches.

| Site | Factory | Strategies |
|---|---|---|
| `xaker/solver/precond.py` | `Make(config)` | `Identity`, `Diagonal`, `Fast`, `Cccp` |
| `xaker/attention/__init__.py` | `BLOCK[name](config)` | `Standard`, `Xsa`, `Laker` |
| `xaker/attention/xsa.py` | `XsaStrategy(config, scale)` | `Projection`, `Zero`, `Mask` |

Adding a new variant is one class + one entry in the dispatch table.

## Fused Laker pipeline

For each forward pass on per head:

1. Project input to Q, K, V via `Base.qkv_proj`.
2. Compute kernel `K = Kernel(q, k)`.
3. Apply external mask if supplied; otherwise zero the kernel diagonal
   (XSA diagonal removal).
4. Build or reuse the preconditioner payload via `Make(config).build(K, lam, length)`.
5. Call `pcg(K, v, lam, apply=Make(config).apply, ...)` and receive `Solve`.
6. If `Solve.converged` is False, fall back to `torch.linalg.solve`.
7. Clamp to `[-BOUND, BOUND]` and RMS-normalize.
8. Apply XSA strategy `Projection`/`Zero`/`Mask` for output cleaning.