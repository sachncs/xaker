---
layout: page
title: API Reference
description: Every public symbol in xaker, every Config field, and every ablation runner. Single-word naming throughout.
permalink: /api/
---

**Audience: anyone using the public surface from a notebook or a script.**

**Time: 5 minutes for a symbol lookup, longer for the full read.**

This page is the single source of truth for the public surface.
Every name listed here is reachable as `from xaker import <name>`
or via `xaker.<submodule>`. The full `__all__` in
`xaker/__init__.py` is authoritative; this document is the
narrated form.

The package groups are:

- [Configuration](#configuration) — `Config`, `Fit`.
- [Attention](#attention) — `Standard`, `Xsa`, `Fused`, `Linear`,
  `Kernel`, the XSA strategies, `BLOCK`, `Base`, `Qkv`.
- [Solver](#solver) — `pcg`, `richardson`, `Solve`, `op`,
  `kernel`, `Make`, `Identity`, `Diagonal`, `Fast`, `Cccp`,
  `Cache`, `PrecondProto`, `BOUND`.
- [Model](#model) — `Block`, `Mlp`, `Model`.
- [Training](#training) — `Trainer`, `Fit`, `ce`.
- [Utilities](#utilities) — `causal`, `padding`, `shape`,
  `clamp`, `finite`, `seed`, `snapshot`, `restore`, `Ctx`,
  `toctx`.
- [Bench](#bench) — `Spec`, `Result`, `Metrics`, `tick`,
  `peak`, `converge`, `run`, `write`, `gitsha`, plus the
  ablation runners.
- [Datasets](#datasets) — `CopyTask`, `ReversalTask`, `WikiText`,
  `build`, `vocab`.
- [Rubric](#rubric) — `Rubric`, `Score`, `Dimension`, `grade`,
  `markdown`, `write`.
- [Quick reference](#quick-reference) — one-screen cheatsheet.

## Configuration

### Config

Attention and solver hyperparameters. Every field is single-word.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dim` | `int` | required | Model width. |
| `heads` | `int` | required | Number of attention heads. |
| `headdim` | `Optional[int]` | `None` | Per-head width; auto = `dim // heads`. |
| `drop` | `float` | `0.0` | Dropout rate for the residual stream. |
| `eps` | `float` | `1e-6` | Numerical-stability floor. |
| `lam` | `float` | `3.0` | Ridge regulariser (after `softplus + eps`). |
| `kernel` | `Literal["exp","rbf","linear","cosine"]` | `"exp"` | Kernel function. |
| `mode` | `Literal["subtract","zero","mask"]` | `"subtract"` | XSA exclusion mode. |
| `precond` | `Literal["cccp","fast","diagonal","identity"]` | `"fast"` | Preconditioner kind. |
| `rank` | `Optional[int]` | `32` | Rank for the `fast` preconditioner. `None` or `0` disables the low-rank component. |
| `directions` | `int` | `64` | Direction samples for `cccp`. |
| `iters` | `int` | `20` | Iteration budget for `cccp`. |
| `gamma` | `float` | `0.1` | CCCP step size. |
| `rho` | `float` | `0.01` | CCCP shrinkage floor. |
| `eps_shrink` | `float` | `1e-8` | CCCP shrinkage epsilon. |
| `pcg` | `int` | `20` | Maximum PCG iterations. |
| `tol` | `float` | `1e-2` | PCG convergence tolerance (relative residual). |
| `freq` | `int` | `1` | How often `fast` rebuilds its preconditioner payload. |
| `temp` | `float` | `1.0` | Kernel temperature. |
| `symmetric` | `bool` | `False` | Average `K` with its transpose. |
| `normalize` | `bool` | `True` | L2-normalise q/k before the kernel. |

`__post_init__` fills `headdim`, validates `dim % heads == 0`, and
checks every categorical field against its `Literal`. Pass an
invalid value to any field and `Config(...)` raises `ValueError`.

### Fit

Training hyperparameters.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `epochs` | `int` | `10` | Outer epoch count. The trainer does not iterate epochs; the caller does. |
| `lr` | `float` | `1e-3` | Learning rate for the default AdamW optimiser. |
| `decay` | `float` | `0.01` | Weight decay for AdamW. |
| `warmup` | `int` | `1000` | Metadata for caller-driven LR schedules. |
| `grad` | `float` | `1.0` | Gradient L2 norm ceiling for `clip_grad_norm_`. |
| `smooth` | `float` | `0.1` | Label smoothing for cross-entropy. |
| `log` | `int` | `100` | Steps between caller-driven log lines. |
| `eval` | `int` | `1000` | Steps between caller-driven evaluation passes. |
| (constructor) | `device` | required | Compute device for batch tensors. |

## Attention

### Block

`xaker.attention.block.Block` — a single pre-norm Transformer
block. The attention is dependency-injected via the constructor
so the same `Block` class wraps every variant in `BLOCK`.

Public attributes:

- `cfg` — the source `Config`.
- `attn` — the attention module injected at construction.
- `norm`, `mlp`, `drop` — surrounding submodules.

### Mlp

Position-wise feed-forward network between blocks. Two-line
configurable activation (GELU or ReLU) and a configurable
hidden ratio.

### Standard

`xaker.attention.standard.Standard` — Vaswani-style scaled
dot-product attention. L2-normalises `q` and `k` before the dot
product. The baseline against which `Xsa`, `Fused`, and
`Linear` are benchmarked.

### Xsa

`xaker.attention.xsa.Xsa` — Exclusive Self Attention with
strategy dispatch. The XSA strategy (`self.xsa`) handles
pre-softmax score modifications and post-softmax output
cleaning. The class does not branch on `Config.mode`.

### Fused

`xaker.attention.fused.Fused` — flagship XSA + kernel + PCG
block. Composes an exponential kernel (`exp(cosine(q, k) / temp)`)
with one of the four preconditioners and solves the regularised
system `(K + λI)α = v` by Preconditioned Conjugate Gradient. The
XSA strategy is the same as for the standalone `Xsa` module.

Solver outcomes land in a `Solve` dataclass
(`x, iters, converged, res, history`); `Fused.attend` only falls
back to `torch.linalg.solve` when `not converged and not finite`.

### Linear

`xaker.attention.linear.Linear` — linear-complexity attention
reference (Katharopoulos et al., 2020). Uses `elu(x) + 1` as
the feature map; saturates in fp16 / bf16 for inputs below the
per-dtype `feature_clamp`. The header is the public surface
table; the dtype frontier lives in
[Limitations](/xaker/limitations/#dtype-contract).

### Kernel

`xaker.attention.kernel.Kernel` — stateful exponential
attention kernel with a learnable `temp`. Used by `Fused`.
Same math as the stateless `kernel` function in
`xaker/attention/func.py`.

### XsaStrategy and the three XSA modes

`XsaStrategy(config, scale)` constructs one of:

| Mode (config.mode) | Class | Behaviour |
|---|---|---|
| `"subtract"` | `Projection` | Subtract `output`'s projection on `value`'s vector. |
| `"zero"` | `Zero` | Zero the score diagonal before softmax. |
| `"mask"` | `Mask` | Combine zero-diagonal with projection subtraction. |

The `Projection` strategy is the canonical paper formulation.
The strategy APIs:

- `prepare(scores, mask)` — pre-softmax hook. Identity for
  `Projection`; zeros the diagonal for `Zero` and `Mask`.
- `apply(output, v)` — post-softmax hook. Returns
  `output - scale * (output · v) / (v · v + eps) * v` for
  `Projection` and `Mask`. Identity for `Zero`.

### BLOCK

`xaker.attention.BLOCK` — polymorphic dict
`{"standard": Standard, "xsa": Xsa, "fused": Fused, "linear": Linear}`.
`BLOCK[kind](config)` returns an instance of the chosen class.
Adding a new variant is one class plus one entry.

### Base, Qkv, broadcast, heads, merge

`Base` is the abstract attention module. Subclasses override
`attend(q, k, v, m)` and inherit everything else.
`Qkv` is the bias-free Q/K/V projection.

## Solver

### pcg

```python
def pcg(kernel, b, lam, *, precond_data=None, apply_pre=None,
        iters=50, tol=1e-3, miniters=3, x0=None) -> Solve
```

Solves `(K + λI) α = b` by Preconditioned Conjugate Gradient.
Convergence stops at `iters` or when the relative residual falls
below `tol`. The returned `Solve` carries `x, iters, converged,
res, history`.

### richardson

```python
def richardson(kernel, b, lam, *, precond_data=None,
               apply_pre=None, iters=10, omega=1.0) -> Solve
```

Fixed-iteration preconditioned Richardson iteration. Always
reports `converged = False`.

### Solve

```python
@dataclass
class Solve:
    x: Tensor
    iters: int
    converged: bool
    res: float
    history: List[float]
```

### op

```python
def op(kernel, x, lam) -> Tensor
```

Evaluates `K x + λ x`. Stateless. Single fused matvec.

### kernel (and the kernel functions)

`xaker.attention.func.kernel(q, k, *, config)` — stateless,
returns the kernel matrix. The shipped kernels:

- `"exp"`: `K_ij = exp(cos(q_i, k_j) / temp)`. Default.
- `"rbf"`: `K_ij = exp(-‖q_i - k_j‖² / (2σ²))`. σ = 1.
- `"linear"`: `K_ij = q_i · k_j`. Not PSD; ridge compensates.
- `"cosine"`: `K_ij = (q_i · k_j) / (‖q_i‖ ‖k_j‖)`. Range [-1, 1].

### Preconditioners

| Name | Build | Apply | Cost |
|---|---|---|---|
| `Identity` | `Cache(data=None)` | `r` | O(n²) for the matvec. |
| `Diagonal` | `softplus(diag(K) + λ) · ‖scale‖ + ε` | `r · diag` | O(n²) once. |
| `Fast` | learned low-rank + diagonal | 2 matvecs + scale | O(n · r) apply. |
| `Cccp` | Tyler M-estimator; eigh inverse half-power | full apply | O(n³) per build. |

### BOUND

`xaker.solver.precond.BOUND = 1e6`. Clamp ceiling on PCG
iterates. Lower to `1e4` for fp16-grade accuracy.

## Model

`Model(cfg, num_layers, vocab_size, max_seq_len, *,
       attention_type, drop, drop_attn)` — full Transformer
encoder. `attention_type` must be one of the four strings in
`BLOCK`. The constructor wires `num_layers` `Block` instances
together with a tied embedding/output projection.

## Training

### Trainer

The bundled training loop. Three operations:

- `step((input_ids, labels))` — one forward + backward + AdamW
  step + gradient clip + optional scheduler.
- `epoch(loader)` — convenience wrapper that aggregates a full
  pass over `loader` and returns epoch loss + elapsed + steps.
- `eval(loader)` — single eval pass returning `{"eval_loss": ...}`.

The trainer does **not** implement logging or checkpointing.
Callers handle that.

### ce

`xaker.training.loss.ce(logits, labels, *, smoothing, ignore_index=-1)`
— label-smoothed cross-entropy.

## Utilities

| Name | Meaning |
|---|---|
| `causal` | Causal mask builder. |
| `padding` | Padding-mask builder. |
| `shape` | Shape coercion. |
| `clamp` | Clamp with default bound. |
| `finite` | Finite-value check; raises or returns `False`. |
| `seed` | Seed PyTorch CPU + CUDA + NumPy + Python. |
| `snapshot` | Snapshot the RNG state across all backends. |
| `restore` | Restore the RNG state from a snapshot. |
| `Ctx` | Typed execution context (`device`, `dtype`). |
| `toctx` | Convert a `(device, dtype)` pair or one device to a `Ctx`. |

## Bench

### Spec

Typed benchmark specification. Required fields: `lengths`,
`dim`, `heads`. Optional fields and their defaults:

| Field | Default |
|---|---|
| `batch` | `4` |
| `kinds` | `["standard", "xsa", "fused"]` |
| `warmup` | `10` |
| `runs` | `50` |
| `seeds` | `[0]` |
| `precond` | `"fast"` |

### Result

Container for a full bench run. Top-level fields are `spec`,
`git_sha`, `torch_version`, `cuda`, `device_name`,
`cudnn_deterministic`. The per-`(kind, length)` data lives
under `results: Dict[Tuple[str, int], Metrics]`.

### Metrics

Per-`(kind, length)` measurements. Keys:
`forward_ms_mean/std`, `backward_ms_mean/std`, `memory_mib`,
`iters_mean/std`, `converged`.

### tick, peak, converge

- `tick(module, x, *, warmup, runs, ctx)` — wall-clock per forward
  + forward+backward; returns
  `(forward_ms_mean/std, backward_ms_mean/std)`.
- `peak(module, x, *, ctx)` — peak GPU memory in MiB (0 on
  CPU/MPS).
- `converge(attn, x, *, ctx, iters=50, runs=5, tol=1e-2)` —
  actually runs PCG against `attn` and returns the real
  `(converged, iters_mean, iters_std)`. Non-Fused callers receive
  `(False, 0.0, 0.0)`.

### run, write

- `run(spec, *, ctx=None)` — run a full bench suite; returns
  `Result`.
- `write(result, path)` — persist JSON. A directory writes
  `rubric.json` + `summary.md`; a file writes a single JSON file.

### Ablation runners

| Module | Sweep |
|---|---|
| `xaker.bench.ablate` | `--axis kind\|kernel\|precond\|mode` |
| `xaker.bench.condition` | `--lengths 16 32 64 128` |
| `xaker.bench.copy_task` | `--epochs 30` |
| `xaker.bench.lra` | `--epochs 5` |
| `xaker.bench.wikitext` | WikiText-2 with synthetic fallback |

Every runner honours `XAKER_DEVICE` (`cpu` / `cuda` / `mps`);
the default is `cpu` because several PyTorch ops have shape
bugs on MPS for batched 4-D inputs.

## Datasets

| Name | Use |
|---|---|
| `CopyTask` | Synthetic copy: target == input. |
| `ReversalTask` | Synthetic reversal: target == reversed input. |
| `WikiText` | WikiText-2 character-level language modelling. Caches to `~/.cache/xaker/wikitext_<split>.pt`. |
| `build(name)` | Factory by name. |
| `vocab(name)` | Vocabulary size lookup by dataset name. |

## Rubric

`xaker.rubric.grade(repo_root=".")` runs all six graders and
returns a `Rubric`. The six dimensions and their evidence shape:

| Dimension | What it inspects |
|---|---|
| `novelty` | `xaker/attention/{fused,linear}.py` exist. |
| `repro` | `rng.py`, `bench.py`, ≥1 JSON in `paper_runs/`, `cudnn.deterministic` flag. |
| `correctness` | `tests/test_fused.py`, `tests/test_solver*.py`, `tests/test_dispatch.py`. |
| `efficiency` | `xaker/bench/bench.py`, `examples/specs/`, ≥2 JSON in `paper_runs/`. |
| `stability` | `seeds=[...]` in code, JSON artifacts, dtype sweep. |
| `usability` | `xaker/cli/`, `README.md`, `xaker/rubric/`. |

`Score.value` is in `[0, 3]` per dimension; the gate is
`total ≥ 17` (relaxable to `14` for short-lived branches) and
`no dim < 2` outside `novelty`.

## Quick reference

```python
import torch
from xaker import (
    Config, Fit, BLOCK, Model, Trainer,
    ce, pcg, op, kernel,
    seed, snapshot, restore, finite, Ctx, toctx,
)

# Train a small fused-xsa Transformer.
seed(0)
cfg = Config(dim=64, heads=4, kernel="exp", mode="subtract", precond="fast")
model = Model(cfg, num_layers=2, vocab_size=100, max_seq_len=32, attention_type="fused")

x = torch.randint(0, 100, (4, 32))
logits = model(x)
loss = ce(logits, x, smoothing=0.1)

# Polymorphic dispatch by kind.
attn = BLOCK["xsa"](cfg)
attn = BLOCK["fused"](cfg)
attn = BLOCK["standard"](cfg)
attn = BLOCK["linear"](cfg)
```



## Next steps

- [Architecture](/xaker/architecture/) — the module tree,
  registries, Fused pipeline.
- [Math](/xaker/math/) — derivations matching the implementation
  byte-for-byte.
- [Recipes](/xaker/recipes/) — how to add a variant, a kernel,
  a preconditioner.
