---
layout: page
title: "Mathematical foundations"
description: "The math behind every public function in xaker — kernel construction, regularised operator, PCG, preconditioners."
permalink: "/math/"
---

**Audience: readers who want the derivations that back every public function.**

**Time: 20 minutes for a full read; each section is self-contained.**

# Mathematical Foundations

This document states the math behind every public function in
XAKER, in the same order the package dispatches them: kernel
construction, regularised operator, iterative solve, output
projection.

The implementation is the spec; if a derivation here contradicts the
code, the code wins. Cross-reference: `docs/design_decisions.md` for
the choices the math does not force, `docs/api.md` for the function
signatures.

## Notation

- `q, k, v` — query, key, value tensors of shape
  `(batch, heads, seq_len, headdim)`.
- `K` — kernel matrix of shape
  `(batch, heads, seq_len, seq_len)`.
- `lam` — ridge regulariser (always positive after
  `softplus(raw) + eps`).
- `BOUND = 1e6` — clamp constant.
- `n = seq_len`, `d = headdim`.

## 1. Kernel construction

The four supported kernels live in
`xaker/attention/func.py:kernel` and
`xaker/attention/kernel.py:Kernel` (stateful counterpart).

### Exponential kernel (`kernel = "exp"`)

```math
K_{ij} = \exp\left( \frac{\cos(q_i, k_j)}{temp} \right)
```

with `q` and `k` L2-normalised by default (`normalize = True`). The
exponential argument is clamped to `[-100, 100]` before `exp` to
prevent overflow in lower-precision dtypes. With `temp = 1` this
reduces to the cosine-similarity exponential kernel used as the
default in `xaker`.

### RBF kernel (`kernel = "rbf"`)

```math
K_{ij} = \exp\left( -\frac{\|q_i - k_j\|^2}{2 \sigma^2} \right)
```

with `sigma = 1`. Positive definite by construction; smooth and
translation-invariant.

### Linear kernel (`kernel = "linear"`)

```math
K_{ij} = q_i \cdot k_j
```

Not positive definite by itself; the ridge regulariser compensates.

### Cosine kernel (`kernel = "cosine"`)

```math
K_{ij} = \frac{q_i \cdot k_j}{\|q_i\| \cdot \|k_j\|}
```

Scale-invariant; range `[-1, 1]`. Same caveat as linear regarding
positive-definiteness.

## 2. XSA projection removal

For each token `i` and value vector `v_i`, XSA subtracts the
projection of the attention output `y_i` onto `v_i`:

```math
y_i^{\text{XSA}} = y_i - \frac{y_i \cdot v_i}{v_i \cdot v_i} \cdot v_i
```

`y_i^{XSA}` is therefore orthogonal to `v_i` by construction.
The XSA paper describes this as the canonical strategy; XAKER
implements it as the `Projection` class, with two further
strategies wired into the `Xsa` strategy enum:

- `Zero` (`mode = "zero"`) — set `K_{ii} = 0` before softmax, so
  `y_i` no longer contains any self-aligned component.
- `Mask` (`mode = "mask"`) — combine diagonal zeroing with
  projection subtraction.

`XsaStrategy(config, scale)` is the single entry point that picks
one based on `config.mode`. `scale` is always
`nn.Parameter(torch.ones(1))` regardless of mode, allocated by
`Xsa.__init__`.

## 3. Regularised operator

The :class:`Fused` attention rewrites attention as kernel ridge regression:

```math
(K + \lambda I) \alpha = v
```

with output `K alpha`. `xaker.solver.func:op` evaluates the
left-hand side:

```math
op(K, x, \lambda) = K x + \lambda x
```

returned in a single matvec + scaled identity product.

## 4. Preconditioned Conjugate Gradient

`xaker.solver.cg:pcg` solves `(K + lam I) alpha = v` by PCG with a
configurable preconditioner `P`. Given the preconditioner factory
`apply_pre = Make(config).apply_pre`:

```math
\alpha_0 = 0,\ r_0 = v,\ z_0 = P(r_0),\ p_0 = z_0
```

```math
\alpha_{t+1} = \alpha_t + \frac{r_t \cdot z_t}{p_t \cdot A p_t} p_t
```

```math
r_{t+1} = r_t - \frac{r_t \cdot z_t}{p_t \cdot A p_t} A p_t
```

```math
z_{t+1} = P(r_{t+1})
```

```math
\beta_{t+1} = \frac{r_{t+1} \cdot z_{t+1}}{r_t \cdot z_t}
```

```math
p_{t+1} = z_{t+1} + \beta_{t+1} p_t
```

Convergence stops when `\|r_t\| < tol * \|v\|` or when
`iters == config.pcg`. The result is returned as a `Solve` dataclass:

```python
@dataclass
class Solve:
    x: torch.Tensor
    iters: int
    converged: bool
    res: float
    history: List[float]
```

`Fused.attend` falls back to `torch.linalg.solve` when
`not solve.converged and finite(solve.x)`; an infinite solution is
propagated so the surrounding `finite()` check fails loudly.

## 5. Preconditioner parameterisations

Each strategy in `xaker/solver/precond.py` exposes
`build(K, lam, length) -> Cache` and `apply_pre(r, data) -> Tensor`.
The four strategies below are listed in registration order; the
math here matches `xaker/solver/precond.py` byte-for-byte.

### `Identity`

```math
P(r) = r
```

No build cost, no convergence improvement. Used for sanity checks.

### `Diagonal`

Learned softplus-positive diagonal preconditioner. One scalar
`scale` per head and the diagonal of `K + lam I` enter a
`softplus` that is then floored by `eps`. Byte-equivalent of
`xaker/solver/precond.py:99-115`:

```python
diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
out = torch.nn.functional.softplus(diag + lam.squeeze(-1).squeeze(-1))
out = out * self.scale.abs() + self.eps
```

```math
d = \mathrm{softplus}(\mathrm{diag}(K) + \lambda) \odot |\text{scale}| + \epsilon
```

```math
P(r) = d \odot r
```

Cheap and effective when the kernel diagonal dominates.

### `Fast`

Learned low-rank-plus-diagonal preconditioner. The basis
`lr_base` is allocated once in `__init__` of shape
`(heads, 2048, rank)` and **never** resampled per build; the
importance vector `lr_imp` is a learnable `nn.Parameter` that
gets multiplied with `lr_base` at every build via softplus. A
scalar `scale` lives on a per-head basis, and `eps` is a stability
floor.

```python
diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
diag = torch.nn.functional.softplus(diag + lam.squeeze(-1).squeeze(-1))
diag = diag * self.scale.abs() + self.eps
lr = self.lr_base[:, :length, :] * torch.nn.functional.softplus(self.lr_imp).unsqueeze(1)
lr = lr.unsqueeze(0).expand(kernel.shape[0], -1, -1, -1)
```

```math
d = \mathrm{softplus}(\mathrm{diag}(K) + \lambda) \odot |\text{scale}| + \epsilon
```

```math
U_{:,:,r} = \mathrm{lr\_base}[:,\,:\text{length},\,:] \odot \mathrm{softplus}(\mathrm{lr\_imp})
```

```math
P(r) = d \odot r + U (U^T r)
```

`Fast.build` caches `(d, U)` on the module instance. The cache is
invalidated by either (a) `freq <= 0` or `step % freq == 0`, **or**
(b) any of `lr_base`, `lr_imp`, or `scale` having their storage
re-allocated (detected by `Tensor.data_ptr()`). Call
`Fast.reset_cache()` to force an immediate rebuild; the
`Fused.attend` call site bumps `Fast.iter` once per forward pass
so `step % freq` is the local-to-block step count.

### `Cccp`

Concave-Convex Procedure. Maintains Tyler's M-estimator direction
estimates and refines them through fixed-point iteration, then
inverts the resulting covariance via `torch.linalg.eigh`:

```python
ubar = self.samples(kernel, lam)        # angular direction samples
sigma = eye.expand(B, H, -1, -1).clone()
for _ in range(self.iters):
    sigma = self.step(ubar, sigma)
eigvals_raw, eigvecs = torch.linalg.eigh(sigma)
eigvals = torch.clamp(eigvals_raw, min=self.eps)
inv_sqrt = eigvals.pow(-0.5)
p = eigvecs @ (inv_sqrt.unsqueeze(-1) * eigvecs.transpose(-2, -1))
return Cache(data=p)
```

The Tyler M-estimator fixed point per iteration is

```math
\Sigma_{t+1} = \frac{n}{r} \sum_i \frac{k_i k_i^T}{k_i^T \Sigma_t k_i}
```

`apply` evaluates `P(r) = \Sigma^{-1/2} r` using the symmetric
eigendecomposition:

```math
P(r) = V \, \mathrm{diag}(\max(\lambda, \epsilon)^{-1/2}) \, V^T \, r
```

Most expensive preconditioner (O(n^3) for the eigh); best
convergence on ill-conditioned kernels.

## 6. Convergence analysis

For the regularised system `(K + lam I) alpha = v`:

- Without preconditioning, the convergence rate depends on the
  condition number `kappa(K + lam I)`. Adding `lam I` shifts every
  eigenvalue by `lam`, so `kappa(K + lam I) <= kappa(K)` and the
  matrix is guaranteed invertible when `lam > 0`.
- With preconditioning, the effective condition number is
  `kappa(P (K + lam I))`. The learned preconditioners approximate
  `(K + lam I)^{-1}` so `kappa` approaches `1` as `P` improves.

The `Solve.history` list lets callers verify the residual decay
trajectory in tests.

## 7. Gradient flow

PCG is unrolled; gradients flow through every intermediate
`apply` and `op` call. There is no custom `torch.autograd.Function`;
the existing `torch.*` ops give full differentiability.

Mitigations:

- `BOUND = 1e6` clamping prevents gradient overflow.
- The direct-solve fallback only runs on `not converged and finite`;
  an infinite residual aborts the forward pass with `finite()`
  raising `ValueError`.
- The `Fast` preconditioner uses `softplus` to keep the diagonal
  strictly positive, so `1 / diag` is finite everywhere.

## 8. Computational complexity

| Operation | `Standard` | `Xsa` | `Fused` (PCG) | `Linear` |
|---|---|---|---|---|
| Q/K/V projection | `O(n * d^2)` | `O(n * d^2)` | `O(n * d^2)` | `O(n * d^2)` |
| Kernel / scores | `O(n^2 * d)` | `O(n^2 * d)` | `O(n^2 * d)` | `O(n * d^2)` |
| Solve | `O(n^2 * d)` direct | `O(n^2 * d)` direct | `O(T * n^2 * d)` | `O(n * d^2)` |
| Preconditioner | — | — | `O(T * n * r * d)` (`Fast`) | — |
| Output projection | `O(n * d^2)` | `O(n * d^2)` | `O(n * d^2)` | `O(n * d^2)` |
| **Total** | `O(n * d^2 + n^2 * d)` | `O(n * d^2 + n^2 * d)` | `O(n * d^2 + T * n^2 * d)` | `O(n * d^2)` |

`Linear` is `O(n)` in sequence length at the cost of feature-map
approximation. The trade-off is positional information loss: the
`elu + 1` feature map does not encode token position, so `Linear`
fails on tasks that need it (LRA copy).

`T = config.pcg`, default `20`. `r = config.rank`, default `32`.

## 9. Worked example

A `Fused` call with `mode = "subtract"` on a `4`-head,
`dim = 64` block, `seq_len = 16`, single sample:

1. Project `x` to `q, k, v` via `Qkv`.
2. Compute `K = exp(cosine(q, k) / temp)` with `temp = 1` and
   L2-normalised inputs.
3. Zero the diagonal: `K = zerodiag(K)`. This is the XSA
   diagonal-removal step.
4. `lam = softplus(raw_lambda) + eps = 3.0` after init.
5. `cache = Make(config).build(K, lam, 16)`.
6. `solve = pcg(K, v, lam, apply_pre=Make(config).apply_pre, ...,
   iters=20, tol=1e-2)`. With `precond = "fast"` this converges in
   typically 4-6 iterations for this size.
7. `alpha = solve.x`; if `not solve.converged` and `finite(alpha)`,
   fall back to `torch.linalg.solve(K + lam * I, v)`.
8. `alpha = clamp(alpha, -BOUND, BOUND)`, then `alpha = rms(alpha, eps)`.
9. XSA step: `alpha = alpha - scale * (alpha * v).sum(-1, keepdim=True) * v`.
10. Output: `K alpha`, merged across heads, projected through `w_o`.



## Next steps
- [Architecture](/xaker/architecture/) — how the math maps to modules.
- [Design decisions](/xaker/design_decisions/) — why the implementation made the calls it did.
