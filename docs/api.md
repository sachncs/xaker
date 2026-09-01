# XAKER API Reference

Single-word public surface. Every entry in this document is reachable
as `from xaker import <name>` (or via `xaker.<submodule>` for module
members). The full `__all__` is the authoritative list.

## Configuration

### `Config`

Attention/solver hyperparameters. All fields single-word.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dim` | `int` | required | Model width |
| `heads` | `int` | required | Number of attention heads |
| `headdim` | `Optional[int]` | `None` | Per-head width; auto = `dim // heads` |
| `drop` | `float` | `0.0` | Dropout rate for the residual stream |
| `eps` | `float` | `1e-6` | Numerical-stability floor |
| `lam` | `float` | `3.0` | Ridge regulariser (after softplus + eps) |
| `kernel` | `Literal["exp","rbf","linear","cosine"]` | `"exp"` | Kernel function |
| `mode` | `Literal["subtract","zero","mask"]` | `"subtract"` | XSA exclusion mode |
| `precond` | `Literal["cccp","fast","diagonal","identity"]` | `"fast"` | Preconditioner |
| `rank` | `Optional[int]` | `32` | Rank for `fast` preconditioner |
| `directions` | `int` | `64` | Direction samples for `cccp` |
| `iters` | `int` | `20` | Iteration budget for `cccp` |
| `gamma` | `float` | `0.1` | CCCP step size |
| `rho` | `float` | `0.01` | CCCP shrinkage floor |
| `eps_shrink` | `float` | `1e-8` | CCCP shrinkage epsilon |
| `pcg` | `int` | `20` | Maximum PCG iterations |
| `tol` | `float` | `1e-2` | PCG convergence tolerance |
| `freq` | `int` | `1` | How often `cccp` rebuilds |
| `temp` | `float` | `1.0` | Kernel temperature |
| `symmetric` | `bool` | `False` | Average `K` with its transpose |
| `normalize` | `bool` | `True` | L2-normalise q/k before the kernel |

### `Fit`

Training hyperparameters.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `epochs` | `int` | required | Number of training epochs |
| `lr` | `float` | `3e-4` | Learning rate |
| `decay` | `float` | `0.1` | AdamW weight decay |
| `warmup` | `int` | `1000` | Linear warmup steps |
| `grad` | `float` | `1.0` | Maximum gradient norm |
| `smooth` | `float` | `0.1` | Label smoothing for `ce` |
| `log` | `int` | `100` | Steps between log lines |
| `eval` | `int` | `1000` | Steps between `eval()` calls |

## Attention

| Symbol | Description |
|---|---|
| `Standard` | Vaswani-style scaled dot-product attention |
| `Xsa` | Exclusive Self Attention with strategy dispatch |
| `Fused` | Fused XSA + kernel attention (flagship), PCG-solved |
| `Linear` | Linear-complexity attention baseline (Katharopoulos et al., 2020) |
| `Kernel` | Stateful exponential attention kernel |
| `Projection`, `Zero`, `Mask` | XSA strategies |
| `BLOCK` | Polymorphic registry `{standard, xsa, fused, linear} -> class` |
| `Base` | Abstract base; subclasses override `attend()` |
| `Qkv` | Bias-free Q/K/V projections |

## Solver

| Symbol | Description |
|---|---|
| `pcg`, `richardson` | Iterative solvers returning `Solve` |
| `Solve` | Outcome dataclass (x, iters, converged, res, history) |
| `op` | Regularised operator `A(x) = Kx + lam x` |
| `kernel` | Stateless kernel matrix |
| `Make` | Preconditioner factory |
| `Identity`, `Diagonal`, `Fast`, `Cccp` | Preconditioner strategies |
| `Cache` | Build-payload dataclass |
| `PrecondProto` | Typing Protocol for static checks |
| `BOUND` | Clamp constant `1e6` |

## Model

| Symbol | Description |
|---|---|
| `Block` | Pre-norm Transformer block with dependency-injected attention |
| `Mlp` | Position-wise FFN (GELU or ReLU) |
| `Model` | Full Transformer encoder; takes `kind = standard\|xsa\|fused` |

## Training

| Symbol | Description |
|---|---|
| `Trainer` | Training loop with optional optimizer / scheduler injection |
| `Fit` | Training config (see above) |
| `ce` | Cross-entropy with label smoothing + `ignore_index` |

## Utilities

| Symbol | Description |
|---|---|
| `causal`, `padding`, `shape`, `clamp` | Tensor helpers |
| `finite` | Finite-value check (raises or returns `False`) |
| `seed`, `snapshot`, `restore` | RNG control across Python, NumPy, PyTorch CPU, CUDA |
| `Ctx`, `toctx` | Typed execution context (`device`, `dtype`) |

## Bench

| Symbol | Description |
|---|---|
| `Spec` | Benchmark specification (lengths, dim, heads, kinds, seeds, …) |
| `Result` | Complete result with environment block (`git_sha`, `torch_version`, …) |
| `Metrics` | Per-(kind, length) measurements (`forward_ms_mean`, `iters_mean`, …) |
| `tick`, `peak`, `converge` | Measurement helpers |
| `run`, `write` | Driver and JSON serializer |
| `gitsha` | Current git HEAD SHA |

### Ablation runners

| Module | Description |
|---|---|
| `xaker.bench.ablate` | Sweep over `kind`, `kernel`, `precond`, or `mode` and emit per-config JSON |
| `xaker.bench.condition` | Compare kernel matrix condition numbers across lengths |
| `xaker.bench.copy_task` | End-to-end training comparison on a copy task |
| `xaker.bench.lra` | Four synthetic LRA-style tasks (copy / reversal / retrieval / addition) |
| `xaker.bench.wikitext` | WikiText-2 training benchmark (GPU recommended) |

All runners honour the `XAKER_DEVICE` env var (`cpu` / `cuda` /
`mps`); default is `cpu` because several PCG ops have bugs on MPS.

## Datasets

| Symbol | Description |
|---|---|
| `CopyTask` | Synthetic copy task; target = input |
| `ReversalTask` | Synthetic reversal task; target = reversed input |
| `WikiText` | WikiText-2 character-level language modelling |
| `build` | Factory by name |
| `vocab` | Vocabulary size lookup by dataset name |

## Rubric

| Symbol | Description |
|---|---|
| `Rubric`, `Score`, `Dimension` | Dataclasses describing the rubric result |
| `grade` | Run all six graders against a path |
| `markdown`, `write` | Render the result as markdown or JSON |

## Quick reference

```python
import torch
from xaker import (
    Config, Fit, BLOCK, Model, Trainer,
    ce, pcg, op, kernel,
    seed, snapshot, restore, finite, Ctx, toctx,
)

# Train a small fused-xsa Transformer
seed(0)
cfg = Config(dim=64, heads=4, kernel="exp", mode="subtract", precond="fast")
model = Model(cfg, num_layers=2, vocab_size=100, max_seq_len=32, attention_type="fused")

x = torch.randint(0, 100, (4, 32))
logits = model(x)
loss = ce(logits, x, smoothing=0.1)

# Polymorphic dispatch by kind
attn = BLOCK["xsa"](cfg)        # Xsa
attn = BLOCK["fused"](cfg)      # Fused
attn = BLOCK["standard"](cfg)   # Standard
attn = BLOCK["linear"](cfg)     # Linear
```