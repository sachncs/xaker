<p align="center">
  <h1 align="center">XAKER</h1>
  <p align="center">A Python library that fuses two attention mechanisms to fix self-bias and spectral collapse in Transformers.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/xaker/releases/latest"><img src="https://img.shields.io/github/v/release/sachncs/xaker" alt="Latest release"></a>
    <a href="https://github.com/sachncs/xaker/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/xaker/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/xaker/pkgs/container/xaker"><img src="https://img.shields.io/badge/ghcr.io-xaker-blue" alt="Docker image"></a>
    <a href="https://github.com/sachncs/xaker/stargazers"><img src="https://img.shields.io/github/stars/sachncs/xaker" alt="Stars"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="mypy"></a>
    <a href="https://github.com/sachncs/xaker/blob/master/pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  </p>
</p>

---

## What is this?

XAKER is a small Python library that answers one question:

> *"How do I replace standard Transformer attention with something
> that doesn't suffer from tokens copying themselves or eigenvalues
> collapsing?"*

xaker implements Exclusive Self Attention (XSA) for Transformer
models, with a flagship fused variant that combines the XSA
self-exclusion step with a learnable exponential kernel and a
Preconditioned Conjugate Gradient solve. The reference paper is
[XSA: arXiv:2603.09078](https://arxiv.org/abs/2603.09078); the
implementation is the spec, and the tests are the contract.

The flagship class — `Fused` — combines XSA with a kernel ridge
regression formulation solved by PCG. Swap it in for `Standard`
attention and the rest of your Transformer keeps working.

---

## Who is this for?

You, even if:

- You're new to PyTorch and only know `nn.Linear`.
- You've never heard of "kernel ridge regression" or "conjugate
  gradient".
- You've written Transformers before but the math papers felt
  impenetrable.

If you can install a Python package and read a function signature,
you can use XAKER. When the docs use a word you don't recognise,
look it up in the [Glossary](docs/glossary.md).

If you've used PyTorch before, you'll be productive in five minutes.

If you're a researcher, the
[paper rubric](docs/paper_rubric.md) tells you exactly what the code
claims and how it's checked.

---

## What can it do?

- **`Standard` attention** — Drop-in baseline that matches vanilla
  scaled dot-product attention. ([Glossary: attention](docs/glossary.md))
- **`Xsa` attention** — Exclusive Self Attention: removes each
  token's self-projection so tokens can only aggregate context.
  ([Glossary: self-bias](docs/glossary.md))
- **`Fused` attention** — Kernel ridge regression with XSA
  self-exclusion, solved by PCG. The flagship module.
  ([Glossary: kernel ridge regression](docs/glossary.md))
- **Four kernel types** — `exp`, `rbf`, `linear`, `cosine`. All
  share one config field.
- **Three XSA modes** — `subtract`, `zero`, `mask`. Pick how
  self-bias is removed.
- **Four preconditioners** — `identity`, `diagonal`, `fast`,
  `cccp`. Pick by speed-vs-quality trade-off.
- **Single-word API** — Every public identifier is one word. No
  `apply_kernel_operator`-style shims.
- **Polymorphic dispatch** — `BLOCK[kind](config)` switches
  between `Standard`, `Xsa`, and `Fused` without `if/elif` mode
  chains.
- **Typed bench driver** — `xaker.bench.Spec` runs reproducibility
  sweeps and writes schema-stable JSON.
- **Paper rubric** — Six-dimension score (novelty/repro/correctness/
  efficiency/stability/usability) enforced in CI.
- **CLI tools** — `xaker-train`, `xaker-eval`, `xaker-bench`,
  `xaker-validate`. No Python required for the common workflows.

---

## Before you start

You'll need **Python 3.9 or newer** installed on your computer.

If you don't know what Python is or whether you have it:

1. Open a terminal (on macOS: `Cmd + Space`, type "Terminal"; on
   Windows: open "PowerShell"; on Linux: open your usual terminal).
2. Type `python3 --version` and press Enter.
3. If you see a version number starting with `3.9` or newer, you're
   set.
4. If you see "command not found" or an older version, follow the
   [official Python installer guide](https://realpython.com/installing-python/).

You'll also need **PyTorch 2.0+**. Install it from
[pytorch.org](https://pytorch.org/get-started/locally/) before
installing XAKER, because the wheel varies by CUDA version.

---

## Installation

From source (recommended for development):

A "virtual environment" is an isolated Python sandbox that keeps this
package's stuff from interfering with your other Python projects.
([Glossary: virtual environment](docs/glossary.md))

```bash
# 1. Download the code
git clone https://github.com/sachncs/xaker.git
cd xaker

# 2. Make a sandbox for it
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows (PowerShell)

# 3. Install XAKER and its dev tools
pip install -e '.[dev]'

# 4. Install PyTorch (pick the CUDA version that matches your box)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 💡 **The dot in `.[dev]` is intentional.** It means "install this
> package and also the dev extras." The square brackets are part of
> the command, not punctuation.

After this, your terminal prompt will probably have `(.venv)` at the
front. That tells you the sandbox is active. To leave the sandbox
later, type `deactivate`.

---

## Your first run — the command line

The fastest way to see XAKER work. No Python required:

```bash
xaker-validate
```

You'll see a table print to your terminal with six rubric scores
plus a total. Current state: **18 / 18 — PASS**. ([Glossary:
rubric](docs/glossary.md))

You can also run a training smoke test on synthetic data:

```bash
xaker-train --dim 256 --heads 4 --layers 4 --epochs 5 --attention fused
```

Or run a benchmark sweep across attention variants:

```bash
xaker-bench --dim 512 --heads 8 --kinds standard,xsa,fused --runs 10 --out paper_runs/baseline.json
cat paper_runs/baseline.json | python3 -m json.tool | head -20
```

---

## Your first run — Python

Open a Python interpreter (`python3` in your terminal) and try this:

```python
import torch                           # pytorch is the deep-learning library
from xaker import Config, Model        # import the config + the Transformer

# A small Transformer with the Fused (XSA + kernel) attention variant.
config = Config(dim=512, heads=8, layers=4, vocab=32000, attention="fused")

# Build the model
model = Model(config)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# Run a forward pass on a random batch
batch = torch.randint(0, config.vocab, (2, 128))   # (batch, seq)
logits = model(batch)                              # (batch, seq, vocab)
print(f"logits shape: {logits.shape}")
```

You'll see something like:

```
parameters: 12,345,678
logits shape: torch.Size([2, 128, 32000])
```

That means: a ~12M parameter Transformer produced a `(2, 128, 32000)`
tensor of token logits. The `fused` attention is doing the work
inside `Model`.

To compare with vanilla attention, swap the keyword:

```python
standard_config = Config(dim=512, heads=8, layers=4, vocab=32000, attention="standard")
standard_model = Model(standard_config)
```

Or build the attention block by itself:

```python
from xaker import BLOCK
block = BLOCK["fused"](config)            # polymorphic dispatch
x = torch.randn(2, 128, 512)
out = block(x)                            # (2, 128, 512)
```

The full walk-through with explanations of every line lives in
[Getting Started](docs/getting-started.md).

---

## Configuration

XAKER uses one `Config` dataclass for attention hyperparameters and
one `Fit` dataclass for training hyperparameters. No environment
variables required.

### Attention hyperparameters (`Config`)

| Field | Plain English |
|---|---|
| `dim` | Model width — the size of every token vector. |
| `heads` | Number of attention heads per block. `dim` must be divisible by `heads`. |
| `headdim` | Per-head width. Defaults to `dim // heads`. |
| `drop` | Dropout rate for the residual stream. `0.0` means no dropout. |
| `lam` | λ — kernel ridge-regression regulariser. Larger = more stable, less expressive. |
| `kernel` | Which kernel function to use: `exp`, `rbf`, `linear`, or `cosine`. |
| `mode` | How XSA removes self-bias: `subtract`, `zero`, or `mask`. Ignored for `standard`. |
| `precond` | Which preconditioner to use: `identity`, `diagonal`, `fast`, or `cccp`. |
| `rank` | Rank for the `fast` preconditioner. Ignored otherwise. |
| `freq` | How often the `cccp` preconditioner updates. |
| `tol` | PCG convergence tolerance. Smaller = more iterations, more accuracy. |
| `attention` | Which block to instantiate: `standard`, `xsa`, or `fused`. |

### Training hyperparameters (`Fit`)

| Field | Plain English |
|---|---|
| `epochs` | How many passes over the training set. |
| `lr` | Learning rate. |
| `decay` | Weight decay for AdamW. |
| `warmup` | Linear warmup steps. |
| `grad` | Maximum gradient norm for clipping. |
| `smooth` | Label smoothing for the cross-entropy loss. |
| `log` | How often to print the loss. |
| `eval` | How often to run validation. |

To change something, construct the dataclass directly:

```python
from xaker import Config, Fit

config = Config(dim=768, heads=12, kernel="exp", mode="subtract", precond="cccp")
fit = Fit(epochs=20, lr=3e-4, decay=0.1, warmup=1000)
```

Then pass both into the trainer:

```python
from xaker import Trainer, Model
model = Model(config)
trainer = Trainer(model, fit)
trainer.run(data)
```

---

## Where to go next

For users:

- **[Getting Started](docs/getting-started.md)** — A complete
  beginner's walk-through, building your first model step by step.
- **[FAQ](docs/faq.md)** — Common questions answered in plain
  English.
- **[Glossary](docs/glossary.md)** — Every technical term, defined.
- **[API Reference](docs/api.md)** — The full list of classes and
  functions, with examples. Bookmark this once you start writing
  real code.
- **[Architecture](docs/architecture.md)** — How the package is put
  together, for the curious.
- **[Limitations](docs/limitations.md)** — Where XAKER *won't* help
  (and what to do instead).

For researchers:

- **[Math](docs/math.md)** — The full mathematical derivation.
- **[Paper rubric](docs/paper_rubric.md)** — Six dimensions the
  implementation is graded against. Enforced in CI.
- **[Design decisions](docs/design_decisions.md)** — Why the code
  looks the way it does.

For operators / maintainers:

- **[Release process](docs/release.md)** — How new versions get
  published.
- **[Deployment](docs/deployment.md)** — Run XAKER on a server.

---

## Project structure

```
xaker/
├── xaker/                    # Main package
│   ├── config.py             # Config dataclass
│   ├── attention/            # Attention implementations
│   │   ├── core.py           # Base, Qkv, free functions
│   │   ├── standard.py       # Standard scaled dot-product
│   │   ├── xsa.py            # Exclusive Self Attention
│   │   ├── fused.py          # Fused XSA + kernel attention (flagship)
│   │   ├── kernel.py         # Kernel module
│   │   ├── func.py           # Stateless kernel + op
│   │   └── ops.py            # zerodiag, rms
│   ├── solver/               # Iterative solvers
│   │   ├── precond.py        # Identity/Diagonal/Fast/Cccp
│   │   ├── cg.py             # pcg + richardson
│   │   └── func.py           # op
│   ├── model/                # Transformer models
│   │   ├── block.py          # One Transformer block
│   │   └── model.py          # Full Transformer
│   ├── training/             # Training utilities
│   │   ├── trainer.py        # Trainer + Fit
│   │   └── loss.py           # ce
│   ├── bench/                # Typed bench driver
│   │   └── bench.py          # Spec, Result, Metrics
│   ├── rubric/               # Paper-worthiness rubric
│   │   ├── rubric.py         # Score, Dimension, Rubric
│   │   ├── grader.py         # 6 graders
│   │   ├── reporting.py      # markdown writer
│   │   └── plugin.py         # pytest plugin
│   ├── cli/                  # CLI entry points
│   │   ├── train.py          # xaker-train
│   │   ├── eval.py           # xaker-eval
│   │   ├── bench.py          # xaker-bench
│   │   └── validate.py       # xaker-validate
│   └── utils/                # Ctx, finite, ops, rng
├── tests/                    # 276 tests, 85% coverage
├── examples/                 # Single typed experiment driver
│   ├── run_paper_experiment.py
│   └── specs/                # YAML experiment specs
├── docs/                     # Architecture, math, design docs
└── .github/workflows/        # CI pipeline
```

---

## Development

```bash
# Install with dev dependencies
pip install -e '.[dev]'

# Run tests
pytest tests/ -v

# Run tests with coverage (gate: >= 85%)
pytest tests/ --cov=xaker --cov-fail-under=85

# Lint
pylint xaker/ --rcfile=pyproject.toml

# Type check
mypy xaker/ --ignore-missing-imports

# Run the paper rubric
xaker-validate --min-total 14

# Build distribution
python -m build
```

### Code style

- **Naming**: every public identifier is a single word. No
  `apply_kernel_operator`-style names. No `_private` style. No
  `import x as y` aliasing.
- **Polymorphism**: dispatch through `BLOCK[kind](config)` and
  `Make(config)`. No `if mode ==` chains in algorithm code.
- **Docstrings**: Google-style throughout.
- **Type hints**: required on all public signatures.
- **Formatter**: [black](https://github.com/psf/black) at 88 columns.
- **Linter**: [pylint](https://pylint.pycqa.org/) at 9.5+ with
  project config.
- **Type checker**: [mypy](https://mypy-lang.org/) at strict-ish.

### Testing

```bash
pytest tests/ -v                         # full suite
pytest tests/ --cov=xaker --cov-fail-under=85    # with coverage gate
pytest -m "not slow"                     # skip the long benchmarks
```

Current state: **276 tests passing, 85.41% coverage**.

---

## Tech stack

| Category | Technology |
|---|---|
| Language | Python 3.9+ |
| Deep learning | [PyTorch](https://pytorch.org/) 2.0+ |
| Numerical | [NumPy](https://numpy.org/) 1.20+ |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Lint | [pylint](https://pylint.pycqa.org/) 2.17+ |
| Format | [black](https://github.com/psf/black) 23+ |
| Type check | [mypy](https://mypy-lang.org/) 1.0+ |
| Benchmarks | [matplotlib](https://matplotlib.org/), pandas, pyyaml |

---

## Contributing

Want to improve XAKER? See [CONTRIBUTING.md](CONTRIBUTING.md) for
how to set up a development environment and submit changes.

## Code of Conduct

We expect everyone to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) — please don't
open a public GitHub issue for security problems.

## Citation

```bibtex
@software{xaker,
  title = {xaker: Exclusive Self Attention for Transformer Models},
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

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it.