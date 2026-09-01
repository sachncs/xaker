# FAQ

## Why is the package called `xaker`?

`xaker` fuses the two scientific contributions — XSA (Exclusive Self
Attention) and LAKER (Learned Preconditioning for Attention Kernel
Regression) — into one searchable brand and a single-word name that
fits the project's naming convention.

## What happened to the v1 attention classes?

`LakerAttentionLayer`, `KernelFunction`, `LearnedPreconditioner`,
`KernelAttentionRegression`, `FusedXSALAKERAttention` were deprecated
in 0.3 and hard-deleted in 0.4. The v2 `Laker` module replaces them
with a single polymorphic strategy.

## Why does `pcg` return a `Solve` dataclass?

`Solve(x, iters, converged, res, history)` exposes everything the
caller needs to decide whether to trust the iterative result, fall
back to a direct solver, or log the residual decay trajectory.
Callers can write `Laker.attend` against this contract without
needing to introspect private state.

## Why does `Laker` always allocate `xsa_scale` as `nn.Parameter`?

Even for `mode="zero"` (which doesn't use it), the parameter is
allocated. This keeps `state_dict` keys stable across modes and
avoids `if mode ==` branching in `__init__`.

## What does the polymorphism look like?

Three single-entry factories replace the `if mode ==` chains the v1
codebase used to scatter through `attention/` and `solver/`:

```python
from xaker import Config, BLOCK, Make, XsaStrategy

cfg = Config(dim=64, heads=4, precond="fast")

attn = BLOCK[cfg.attention or "fused"](cfg)      # Standard / Xsa / Laker
precond_apply = Make(cfg).apply                 # preconditioner
strategy = XsaStrategy(cfg, scale=...)          # Projection / Zero / Mask
```

Adding a new variant is one class plus one entry in the dispatch
table.

## Why is `keep` named `keep`?

`mask` is the parameter name on `attend(q, k, v, mask)`. The free
function that applies a mask to scores is exported as `keep(scores,
mask, fill)` — chosen because it does not collide with the parameter
name. Old code referenced `apply_mask`; that was renamed to `keep`
in 0.4.

## Why is `toctx` instead of `to_device`?

`toctx(tensor, ctx)` is the only tensor-context helper; a single
compound identifier keeps the public surface small and consistent
with the no-underscore rule.

## Why is `rng` separate from `random`?

`xaker.utils.rng` exposes `seed, snapshot, restore` with consistent
seeding across Python, NumPy, PyTorch CPU, and CUDA. Using the
single-word name `rng` keeps the public API tight and avoids the
name `random` clashing with the stdlib.

## How do I add a new attention variant?

1. Implement your class in `xaker/attention/<name>.py`, subclassing
   `Base` and overriding `attend(q, k, v, m)`.
2. Register it in `xaker/attention/__init__.py:BLOCK`.
3. Add it to `Model(attention_type=...)` choices.

That's it — `Block` accepts attention by dependency injection.

## How do I add a new preconditioner?

1. Subclass `nn.Module` in `xaker/solver/precond.py` and implement
   `build(kernel, lam, length) -> Cache` and
   `apply(residual, data) -> Tensor`.
2. Register it in `Make(config)` by extending the `if config.precond`
   branch.
3. Add the new value to `Config.precond`'s `Literal[...]` type.

## Why are there no `import x as y` aliases anywhere?

Single-word naming + module boundaries make aliases redundant. CI
(`xaker/rubric/grader.py:usability`) flags any `import x as y` as a
failure.

## What does `xaker-validate` check?

Six dimensions: novelty, repro, correctness, efficiency, stability,
usability. Each scored 0-3; total max 18. CI fails on
`total < 14` or any non-novelty dimension below 2. Run locally with
`xaker-validate --repo-root . --min-total 14`.