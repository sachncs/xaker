<p align="center">
  <h1 align="center">xaker</h1>
  <p align="center">Fused Exclusive Self Attention and Kernel Ridge Regression for Transformer models.</p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/xaker/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/xaker/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/xaker/stargazers"><img src="https://img.shields.io/github/stars/sachncs/xaker" alt="Stars"></a>
    <a href="https://github.com/sachncs/xaker/blob/master/pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="mypy"></a>
  </p>
</p>

---

## Overview

xaker is an open-source Python library that fuses **Exclusive Self
Attention (XSA)** with **kernel ridge regression** for Transformer
models, forming a single attention module solved by Preconditioned
Conjugate Gradient. The library is the open-source companion to a
paper-in-progress that quantifies the trade-offs of this fusion
relative to vanilla scaled dot-product attention.

Three attention variants ship in one polymorphic dispatch:

- `Standard` -- Vaswani-style scaled dot-product attention. Baseline.
- `Xsa` -- Exclusive Self Attention. Removes each token's
  self-projection so tokens can only aggregate context.
- `Fused` -- The flagship: XSA + kernel ridge regression solved by
  PCG with a configurable preconditioner.

The `Fused` block implements the XSA projection-removal step plus
the LAKER-style kernel ridge regression formulation, addressing two
known failure modes of vanilla attention:

- **Self-bias** -- every token's output includes a contribution
  from its own value vector.
- **Spectral collapse** -- the kernel matrix becomes ill-conditioned
  on long sequences, which slows iterative solvers.

---

## Results

The flagship claim of this library is that the kernel ridge
regression formulation with XSA diagonal removal produces a
**300-1700x lower condition number** than the equivalent softmax
score matrix across sequence lengths (16 to 128), while reaching
**100% accuracy on the copy task** and matching `Standard` and
`Xsa` on the LRA retrieval task. The trade-off is wall-clock
overhead from solving `(K + lam I) alpha = V` by Preconditioned
Conjugate Gradient at every forward pass.

### Condition number (lower is better)

| Length | Score cond (Standard) | Kernel cond (Fused) | Ratio (kernel / score) |
|--------|-----------------------|---------------------|-------------------------|
| 16     | 1,018                 | 3.11                | 0.0031                  |
| 32     | 3,346                 | 5.66                | 0.0017                  |
| 64     | 12,310                | 12.76               | 0.0010                  |
| 128    | 74,917                | 41.55               | 0.0006                  |

### Trained comparison (copy task, length=16, 30 epochs)

| Kind     | Train loss | Val loss | Accuracy | Wall (s) |
|----------|------------|----------|----------|----------|
| Standard | 0.0006     | 0.0006   | 100%     | 2.7      |
| Xsa      | 0.0006     | 0.0006   | 100%     | 2.9      |
| Fused    | 0.0005     | 0.0006   | 100%     | 13.2     |
| Linear   | 0.0012     | 0.0007   | 100%     | 2.8      |

### LRA-style tasks (length=32, 5 epochs)

| Task       | Standard | Xsa | Fused | Linear |
|------------|----------|-----|-------|--------|
| copy       | 0.91     | 0.90| 0.87  | 0.14   |
| retrieval  | 0.97     | 0.97| 0.97  | 0.97   |
| addition   | 0.54     | 0.54| 0.54  | 0.54   |
| reversal   | 0.07     | 0.07| 0.06  | 0.12   |

`Linear` fails on copy (no positional information in `elu + 1`
feature map). The other three are within 5% on every task at
length=32. See `RESULTS.md` for the full breakdown including kernel,
preconditioner, and mode ablations.

---

## Features

- Four attention variants in one polymorphic dispatch
  (`BLOCK[kind](config)`): `Standard`, `Xsa`, `Fused`, `Linear`
- Four kernel functions: `exp`, `rbf`, `linear`, `cosine`
- Three XSA modes: `subtract`, `zero`, `mask`
- Four preconditioners: `identity`, `diagonal`, `fast`, `cccp`
- PCG solver with direct-solve fallback (`xaker.solver.cg`)
- Four baselines: `Standard` (Vaswani), `Linear` (Katharopoulos
  et al. linear attention), `Xsa` (XSA only), `Fused` (XSA + kernel)
- Typed bench driver emitting schema-stable JSON (`xaker.bench`)
- Ablation harness: kernel, preconditioner, mode, and kind sweeps
- Long Range Arena-style synthetic tasks: copy, reversal,
  retrieval, addition
- Paper-worthiness rubric enforced in CI (`xaker-validate`)
- Four CLI entry points: `xaker-train`, `xaker-eval`,
  `xaker-bench`, `xaker-validate`
- Single-word public API: no `_private` names, no `import x as y`
  aliases, no multi-word shims
- 296 tests, 92% coverage, 18/18 rubric score

---

## Installation

Requires Python 3.9 or newer and PyTorch 2.0+.

```bash
git clone https://github.com/sachncs/xaker.git
cd xaker
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
pip install -e '.[dev]'
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA version
```

The dot in `.[dev]` is intentional: it means "install this package
and also the dev extras." The square brackets are part of the
command, not punctuation.

---

## Quick start

### Command line

```bash
xaker-validate
```

Prints a six-dimension rubric table with a total. Current state: 18 / 18 -- PASS.

```bash
xaker-train --dim 256 --heads 4 --layers 4 --epochs 5 --kind fused
xaker-bench --dim 512 --heads 8 --kinds standard xsa fused --runs 10 --output paper_runs/baseline.json
```

### Python

```python
import torch
from xaker import Config, Model

config = Config(dim=512, heads=8, kernel="exp", mode="subtract", precond="fast")
model = Model(config, num_layers=6, vocab_size=32000, max_seq_len=512, attention_type="fused")

batch = torch.randint(0, 32000, (2, 128))
logits = model(batch)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"logits shape: {logits.shape}")
```

Or build just the attention block:

```python
from xaker import BLOCK
block = BLOCK["fused"](config)
x = torch.randn(2, 128, 512)
out = block(x)
```

---

## Configuration

Two dataclasses, no environment variables.

### `Config` (attention hyperparameters)

| Field        | Type    | Default  | Plain English |
|--------------|---------|----------|---------------|
| `dim`        | int     | required | Model width |
| `heads`      | int     | required | Number of attention heads |
| `headdim`    | int?    | auto     | Per-head width; `dim // heads` when unset |
| `drop`       | float   | 0.0      | Dropout rate for the residual stream |
| `eps`        | float   | 1e-6     | Numerical-stability floor |
| `lam`        | float   | 3.0      | Ridge regulariser (after softplus + eps) |
| `kernel`     | str     | "exp"    | One of `exp`, `rbf`, `linear`, `cosine` |
| `mode`       | str     | "subtract" | XSA exclusion mode: `subtract`, `zero`, `mask` |
| `precond`    | str     | "fast"   | Preconditioner: `identity`, `diagonal`, `fast`, `cccp` |
| `rank`       | int?    | 32       | Rank for the `fast` preconditioner |
| `pcg`        | int     | 20       | Maximum PCG iterations |
| `tol`        | float   | 1e-2     | PCG convergence tolerance |
| `temp`       | float   | 1.0      | Kernel temperature |
| `symmetric`  | bool    | False    | Average kernel with its transpose |
| `normalize`  | bool    | True     | L2-normalise q/k before the kernel |

### `Fit` (training hyperparameters)

| Field    | Type  | Default | Plain English |
|----------|-------|---------|---------------|
| `epochs` | int   | required | Number of training epochs |
| `lr`     | float | 3e-4    | Learning rate |
| `decay`  | float | 0.1     | Weight decay for AdamW |
| `warmup` | int   | 1000    | Linear warmup steps |
| `grad`   | float | 1.0     | Maximum gradient norm |
| `smooth` | float | 0.1     | Label smoothing for cross-entropy |
| `log`    | int   | 100     | Steps between log lines |
| `eval`   | int   | 1000    | Steps between validation passes |

---

## Architecture

Three single-entry factories replace the `if mode ==` chains that
v1 attention code used to scatter through the package:

- `Make(config)` -- preconditioner factory. Returns `Identity`,
  `Diagonal`, `Fast`, or `Cccp`.
- `BLOCK[kind](config)` -- attention factory. Returns `Standard`,
  `Xsa`, `Fused`, or `Linear`.
- `XsaStrategy(config, scale)` -- XSA mode factory. Returns
  `Projection`, `Zero`, or `Mask`.

Four attention variants in `BLOCK`:

- `Standard` -- Vaswani scaled dot-product attention.
- `Xsa` -- XSA self-exclusion on top of scaled dot-product.
- `Fused` -- The flagship: XSA + kernel ridge regression solved by
  PCG with a configurable preconditioner.
- `Linear` -- Linear-complexity attention baseline
  (Katharopoulos et al., 2020). Reference comparison; fails on
  tasks that need positional information.

The `Fused` block kernelizes the XSA self-exclusion step into the
regularised system `(K + lam I) alpha = V` and solves it by PCG.
Solver outcomes land in a `Solve` dataclass
(`x, iters, converged, res, history`); on `not converged and not
finite`, `Fused.attend` falls back to `torch.linalg.solve`.

See `docs/architecture.md` for the full picture and `docs/math.md`
for the derivation.

---

## Project structure

```
xaker/
├── xaker/
│   ├── __init__.py            Public surface (40+ symbols)
│   ├── config.py              Config dataclass
│   ├── attention/
│   │   ├── core.py            Base, Qkv, keep, heads, merge, broadcast
│   │   ├── standard.py        Standard scaled dot-product attention
│   │   ├── xsa.py             Xsa, XsaStrategy, Projection, Zero, Mask
│   │   ├── fused.py           Fused (flagship: XSA + kernel + PCG)
│   │   ├── linear.py          Linear (Katharopoulos et al. baseline)
│   │   ├── kernel.py          Kernel module (stateful exponential)
│   │   ├── func.py            Stateless kernel + op
│   │   └── ops.py             zerodiag, rms
│   ├── solver/
│   │   ├── precond.py         Make, Identity, Diagonal, Fast, Cccp
│   │   ├── cg.py              pcg, richardson, Solve
│   │   └── func.py            op
│   ├── model/
│   │   ├── block.py           Block (pre-norm Transformer block)
│   │   └── model.py           Model
│   ├── training/
│   │   ├── trainer.py         Trainer, Fit
│   │   └── loss.py            ce
│   ├── bench/
│   │   ├── bench.py           Spec, Result, Metrics, run, write
│   │   ├── ablate.py          Ablation sweep runner (kind/kernel/precond/mode)
│   │   ├── condition.py       Kernel matrix condition-number benchmark
│   │   ├── copy_task.py       Copy-task training comparison
│   │   ├── lra.py             LRA-style synthetic tasks (copy/reversal/retrieval/addition)
│   │   └── wikitext.py        WikiText-2 training benchmark
│   ├── datasets/
│   │   └── __init__.py        CopyTask, ReversalTask, WikiText loaders
│   ├── rubric/
│   │   ├── rubric.py          Score, Dimension, Rubric
│   │   ├── grader.py          Six graders (novelty, repro, correctness, ...)
│   │   ├── reporting.py       markdown, write
│   │   └── plugin.py          pytest plugin
│   ├── cli/
│   │   ├── train.py           xaker-train
│   │   ├── eval.py            xaker-eval
│   │   ├── bench.py           xaker-bench
│   │   └── validate.py        xaker-validate
│   └── utils/
│       ├── ctx.py             Ctx, toctx
│       ├── finite.py          finite
│       ├── ops.py             causal, padding, shape, clamp, BOUND
│       └── rng.py             seed, snapshot, restore
├── tests/                     296 tests, 91.5% coverage
├── examples/
│   ├── run_paper_experiment.py
│   └── specs/                 Five YAML experiment specs
├── paper_runs/                Reproducible benchmark JSON outputs
├── docs/                      Architecture, math, design, FAQ, rubric
└── .github/workflows/         CI pipeline
```

---

## Development

```bash
pip install -e '.[dev]'

pytest tests/ -v                                              # full suite
pytest tests/ --cov=xaker --cov-fail-under=85                 # with coverage gate
pylint xaker/ --rcfile=pyproject.toml                          # lint
mypy xaker/ --ignore-missing-imports                          # type check
xaker-validate --min-total 14                                 # paper rubric
python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml   # reproduce benchmark
```

Current state: 296 tests passing, 91.5% coverage, 18/18 rubric.

---

## Documentation

- [Getting Started](docs/getting-started.md) -- step-by-step walk-through
- [FAQ](docs/faq.md) -- common questions
- [API Reference](docs/api.md) -- full symbol list
- [Architecture](docs/architecture.md) -- module layout and dispatch
- [Math](docs/math.md) -- derivations
- [Design Decisions](docs/design_decisions.md) -- why the code looks this way
- [Paper Rubric](docs/paper_rubric.md) -- the six-dimension grading rubric
- [Results](RESULTS.md) -- benchmark numbers and ablation tables
- [Limitations](docs/limitations.md) -- where xaker won't help

Reproducible benchmarks (run on CPU):

```bash
python -m xaker.bench.ablate --axis kind --values standard xsa fused linear --length 32
python -m xaker.bench.ablate --axis kernel --values exp rbf linear cosine --length 32
python -m xaker.bench.ablate --axis precond --values identity diagonal fast cccp --length 32
python -m xaker.bench.ablate --axis mode --values subtract zero mask --length 32
python -m xaker.bench.condition --lam 10.0 --lengths 16 32 64 128
python -m xaker.bench.copy_task --dim 64 --length 16 --epochs 30
python -m xaker.bench.lra --dim 32 --length 32 --epochs 5
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Code of Conduct

We expect everyone to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [SECURITY.md](SECURITY.md). Please don't
open a public GitHub issue for security problems.

## Citation

```bibtex
@software{xaker,
  title = {xaker: Fused Exclusive Self Attention and Kernel Ridge Regression},
  author = {sachin},
  year = {2026},
  url = {https://github.com/sachncs/xaker},
}

@article{xsa_paper,
  title = {Exclusive Self Attention},
  author = {Sachin},
  journal = {arXiv preprint arXiv:2603.09078},
  year = {2026},
}
```

## License

MIT -- see [LICENSE](LICENSE). Use it, fork it, ship it.