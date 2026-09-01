# XAKER API Reference

Single-word public surface.

## Configuration

| Symbol | Description |
|---|---|
| `Config` | Attention/solver config dataclass |

`Config` fields (all single-word):

| Field | Type | Default |
|---|---|---|
| `dim` | int | required |
| `heads` | int | required |
| `headdim` | int \| None | `None` (auto) |
| `drop` | float | 0.0 |
| `eps` | float | 1e-6 |
| `lam` | float | 3.0 |
| `kernel` | Literal["exp","rbf","linear","cosine"] | "exp" |
| `mode` | Literal["subtract","zero","mask"] | "subtract" |
| `precond` | Literal["cccp","fast","diagonal","identity"] | "fast" |
| `rank` | int \| None | 32 |
| `directions` | int | 64 |
| `iters` | int | 20 |
| `gamma` | float | 0.1 |
| `rho` | float | 0.01 |
| `eps_shrink` | float | 1e-8 |
| `pcg` | int | 20 |
| `tol` | float | 1e-2 |
| `freq` | int | 1 |
| `temp` | float | 1.0 |
| `symmetric` | bool | False |
| `normalize` | bool | True |

## Attention

| Symbol | Description |
|---|---|
| `Standard` | Vanilla MHA |
| `Xsa` | Exclusive Self Attention |
| `Laker` | Fused XSA + LAKER (flagship) |
| `Kernel` | Exponential attention kernel |
| `Projection`, `Zero`, `Mask` | XSA strategies |
| `BLOCK` | Polymorphic registry `{standard, xsa, fused} -> class` |

## Solver

| Symbol | Description |
|---|---|
| `pcg`, `richardson` | Iterative solvers |
| `Solve` | Result dataclass (x, iters, converged, res, history) |
| `op` | Regularized kernel operator `A(x) = Kx + lam x` |
| `kernel` | Stateless kernel matrix |
| `Make` | Preconditioner factory |
| `Identity`, `Diagonal`, `Fast`, `Cccp` | Preconditioner strategies |
| `Cache` | Payload dataclass |
| `PrecondProto` | Typing Protocol |
| `BOUND` | Clamp bound `1e6` |

## Model

| Symbol | Description |
|---|---|
| `Block` | Pre-norm Transformer block |
| `Mlp` | Position-wise FFN |
| `Model` | Full Transformer encoder |

## Training

| Symbol | Description |
|---|---|
| `Trainer` | Training loop |
| `Fit` | Training config |
| `ce` | Cross-entropy with label smoothing |

## Utilities

| Symbol | Description |
|---|---|
| `causal`, `padding`, `shape`, `clamp` | Tensor ops |
| `finite` | Finite-value check |
| `seed`, `snapshot`, `restore` | RNG control |
| `Ctx`, `toctx` | Typed execution context |

## Bench

| Symbol | Description |
|---|---|
| `Spec`, `Result`, `Metrics` | Dataclasses |
| `tick`, `peak`, `converge` | Measurement helpers |
| `run`, `write`, `gitsha` | Driver + persistence |

## Rubric

| Symbol | Description |
|---|---|
| `Rubric`, `Score`, `Dimension` | Dataclasses |
| `grade` | Run all six graders |
| `markdown`, `write` | Render results |