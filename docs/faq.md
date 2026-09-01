# FAQ

## Why is the package called `xaker`?

`xaker` fuses the two scientific contributions — XSA (Exclusive Self
Attention) and LAKER (Learned Preconditioning for Attention Kernel
Regression) — into one searchable brand.

## Why no v1 attention classes?

`LakerAttentionLayer`, `KernelFunction`, `LearnedPreconditioner`,
`KernelAttentionRegression`, `FusedXSALAKERAttention` were deprecated
and hard-deleted. The v2 `Laker` module replaces them with cleaner
polymorphism.

## Why does `pcg` return a `Solve` dataclass?

`pcg` reports whether it converged, the final residual, the iteration
count, and per-iteration history. Callers can decide whether to fall
back to a direct solver (`torch.linalg.solve`).

## Why does `Laker` always allocate `xsa_scale` as `nn.Parameter`?

Even for `mode="zero"` (which doesn't use it), the parameter is allocated.
This keeps `state_dict` keys stable and avoids mode-branching in
`__init__`.

## Why is `Mask` (the function name) not in the public API?

`mask` is a method parameter on `attend(x, mask)`. We use `keep(scores, mask)`
internally to avoid the name clash. Use `apply_mask` if you need the
function externally — it's re-exported.

## Why is `rng` separate, not `random`?

`xaker.utils.rng` exposes `seed`, `snapshot`, `restore` with consistent
seeding across Python, NumPy, PyTorch CPU, and CUDA. Using the single-word
name `rng` keeps the public API tight.

## Why is `to_ctx` instead of `to_device`?

`to_ctx(t, ctx)` is the only tensor-context helper; a single compound
identifier keeps the public surface small.

## How do I add a new attention variant?

1. Implement your class in `xaker/attention/<name>.py`, subclassing `Base`.
2. Register it in `xaker/attention/__init__.py:BLOCK`.
3. Add it to `Spec.kinds` (the `Literal` type).

That's it — `Block` accepts attention by dependency injection.