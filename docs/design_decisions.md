# Design Decisions

This document captures the design choices behind XAKER's current shape,
including what the code preserves exactly from the two source papers and
where it extends them. It is the living companion to the math document
(`docs/math.md`) and the public API (`docs/api.md`).

## Source papers

- XSA — [arXiv:2603.09078](https://arxiv.org/abs/2603.09078)
- LAKER — [arXiv:2604.25138](https://arxiv.org/html/2604.25138v1)

The implementation is the spec; if a derivation in either paper
contradicts the code, the code wins until a paper revision or
explicit note moves the goalposts.

## Polymorphism over mode strings

Three single-entry factories replace the `if mode ==` chains the v1
codebase used to scatter through `attention/` and `solver/`:

| Site | Factory | Strategies |
|---|---|---|
| `xaker/solver/precond.py` | `Make(config)` | `Identity`, `Diagonal`, `Fast`, `Cccp` |
| `xaker/attention/__init__.py` | `BLOCK[name](config)` | `Standard`, `Xsa`, `Laker` |
| `xaker/attention/xsa.py` | `XsaStrategy(config, scale)` | `Projection`, `Zero`, `Mask` |

Adding a new variant is one class plus one entry in the dispatch
table. The branches that used to live in algorithm code are gone.

## Attention variants

The package ships three attention modules. They share `Base` and
`Qkv` from `xaker/attention/core.py` and only override
`attend(q, k, v, m)`.

### `Standard` (`attention/standard.py`)

Vaswani-style scaled dot-product attention with L2-normalised
queries and keys. The baseline against which `Xsa` and `Laker` are
benchmarked.

### `Xsa` (`attention/xsa.py`)

The paper's projection-removal step
(`y_i <- y_i - proj_{v_i}(y_i)`) is one of three concrete
strategies:

- `Projection` — the paper's formula. Default; matches `mode="subtract"`.
- `Zero` — zero the kernel diagonal before softmax. Matches `mode="zero"`.
- `Mask` — combine diagonal zeroing with projection subtraction. Matches `mode="mask"`.

The factory `XsaStrategy(config, scale)` picks one based on `config.mode`.
`scale` is always `nn.Parameter(torch.ones(1))` allocated by `Xsa.__init__`,
even when the strategy will ignore it. Keeping the allocation unconditional
preserves `state_dict` keys across modes.

### `Laker` (`attention/laker.py`)

The flagship. Combines an exponential kernel (`exp(cosine(q, k) / temp)`)
with one of the four preconditioners and solves the regularised system
`(K + lam I) alpha = V` by Preconditioned Conjugate Gradient. The XSA
strategy is the same as for the standalone `Xsa` module.

Solver outcomes land in a `Solve` dataclass
(`x, iters, converged, res, history`); `Laker.attend` only falls back to
`torch.linalg.solve` when `not converged and not finite`.

## Kernel choice

`Config.kernel` selects one of four stateless functions in
`xaker/attention/func.py`:

| `kernel` | `k(q, k)` | Notes |
|---|---|---|
| `exp` | `exp(cosine(q, k) / temp)` with L2-normalised q/k | default; matches the LAKER paper |
| `rbf` | `exp(-||q - k||^2 / (2 * sigma^2))` | classical Gaussian |
| `linear` | `q . k` | no positivity by itself; the lambda regulariser compensates |
| `cosine` | `(q . k) / (||q|| * ||k||)` | scale-invariant, range [-1, 1] |

`Kernel(dim, temp, symmetric, normalize, eps)` (`attention/kernel.py`)
is the stateful counterpart for `Laker` — same math, learnable `temp`.

## Preconditioners

`Config.precond` selects one of four concrete `nn.Module` strategies.
Each implements `build(kernel, lam, length) -> Cache` and
`apply(residual, data) -> Tensor`.

| `precond` | Where it shines | Cost per step |
|---|---|---|
| `identity` | debugging, sanity | O(n^2) for the matvec |
| `diagonal` | per-token scale dominant | O(n^2) once to build, O(n) per apply |
| `fast` | default; low-rank + diagonal `P = diag(d) + UU^T` | O(n * r * d) build, O(n * r) apply |
| `cccp` | best convergence, slowest build | O(n^3) for the direction samples |

`BOUND = 1e6` (`xaker/solver/precond.py`) clamps intermediate values
before they can blow up. The `Cache` dataclass carries the build
payload so the same preconditioner can be reused across layers in a
Transformer block.

## Numerical-stability choices

- **Lambda positivity**: `lam = softplus(raw_lambda) + eps`. Softplus
  is the one place we use a soft-clip; everywhere else we clamp
  after the fact.
- **PCG residual clamping**: `BOUND = 1e6` on every entry of the
  solution vector. Tuned so the regularised `(K + lam I)` cannot
  overflow before `lam` dominates.
- **Ridge regulariser**: `lam > 0` keeps the regularised operator
  invertible on every kernel. The default `lam = 3.0` is the smallest
  value that kept the `Fast` preconditioner stable on a 128-token
  batch with `dim = 768`.
- **Fallback direct solve**: when `pcg` reports `not converged`,
  `Laker.attend` switches to `torch.linalg.solve` only when the
  residual is finite; an infinite residual is propagated.

## Reproducibility

Reproducibility is a single command: `xaker.utils.rng.seed(N)`. It
seeds Python, NumPy, PyTorch CPU, and CUDA, sets
`torch.backends.cudnn.deterministic = True` and
`cudnn.benchmark = False`, then returns. CLIs call it at the top of
`main()`. The `Trainer` does **not** call it on construction — the
caller seeds before constructing both the model and the trainer.

`Ctx` (`xaker/utils/ctx.py`) carries the device/dtype pair through
the bench driver so that every measurement is comparable.

## Single-word naming

Every public symbol — modules, classes, functions, methods, dataclass
fields — is one word. No `_private` style, no `apply_kernel_operator`
style, no `from x import y as z` aliasing. Module boundaries and
`__all__` declarations replace them. The rule is enforced by CI
(`.github/workflows/ci.yml`).

Renames that drove this rule:

- `apply_mask` → `keep` (the function parameter is also `mask`)
- `to_ctx` → `toctx` (one compound word, no separator)
- `time` → `tick` in the bench driver (avoids stdlib clash)
- `_git_sha` → `gitsha` (no leading underscore)

## Test strategy

- **Property** (`test_property.py`) — shape and finiteness invariants.
- **Convergence** (`test_convergence.py`) — PCG `Solve` properties.
- **Dispatch** (`test_dispatch.py`) — `BLOCK`, `Make`, `XsaStrategy`
  return the right concrete classes for every allowed `kind`.
- **Rubric** (`test_rubric.py`) — graders spot-check their inputs.
- **Bench** (`test_bench.py`) — `Spec`/`Result`/`Metrics` schemas.
- **Examples** (`test_examples.py`) — five YAML specs are valid and
  the driver imports cleanly.

The paper rubric (`xaker-validate`) sits on top: it inspects the
repository itself and reports whether tests, schemas, docs, and CLI
gates are all present. CI fails the build on `total < 14` or any
non-novelty dimension below 2.

## What we did not build

- Sparse or Nyström kernel approximations.
- Custom CUDA kernels; everything is `torch.*`.
- AMP / mixed-precision training paths.
- Hugging Face `transformers` integration.

These are listed as future work in the LAKER paper itself; until
someone asks for them, they stay out of the package.