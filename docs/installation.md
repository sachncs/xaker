---
layout: page
title: Install
description: Set up xaker in a fresh Python environment, including PyTorch CUDA, troubleshooting, and project metadata.
permalink: /installation/
---

**Audience: anyone who needs PyTorch CUDA wheels or environment troubleshooting.**

**Time: 10 minutes.**

This page is the long-form install guide. For a quick walk-through,
see [Getting started](/xaker/getting-started/). Both pages follow the
same content in different shapes: this one is exhaustive, the other
is the two-minute version.

## System requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.9, 3.10, 3.11, or 3.12 | Tested in CI on every row of the matrix. |
| PyTorch | 2.0 or newer | Match your CUDA build: see `pytorch.org/get-started`. |
| Disk | ~200 MB | Source clone plus `pip install -e .[dev]`. |
| RAM | 1 GB minimum | The benchmark suite defaults to CPU and uses `dim=64`. |

The package has no compiled extensions; it ships pure-Python on top
of `torch`. No C compiler, no CMake, no CUDA toolkit needed at
install time.

## Source install

```bash
git clone https://github.com/sachncs/xaker.git
cd xaker
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

pip install -e '.[dev]'
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
```

The `.` in `.[dev]` is intentional. It means "install this package
*and* also the dev extras." Square brackets are part of the
command, not punctuation.

If you do not have a CUDA-capable host, the PyTorch index step is
optional: `pip install -e '.[dev]'` already pulls PyTorch 2.0+ from
PyPI as a CPU build.

## Optional extras

The `pyproject.toml` declares two extras:

| Extra | Pulls in | When you need it |
|---|---|---|
| `dev` | `pytest`, `pylint`, `mypy`, `black`, `pytest-randomly` | Running the test suite or contributing. |
| `paper` | `matplotlib`, `pandas`, `pyyaml` | Building the paper figures from `paper_runs/*.json`. |

Install both with `pip install -e '.[dev,paper]'`.

## PyTorch CUDA wheels

If you intend to run the benchmarks on GPU, install PyTorch with a
matching CUDA build before running the rest:

```bash
# CUDA 12.1 host
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 host
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch
```

The benchmark suite defaults to CPU because three PyTorch ops
(`linalg.solve`, `linalg.lu_solve`, `linalg.eigh`) have
shape-bugs on MPS for batched 4-D inputs. Set `XAKER_DEVICE=cuda`
to opt in once you have a CUDA host running.

## Verify the install

```bash
xaker-validate
python -c "from xaker import Config, Fused; cfg = Config(dim=64, heads=4); print(Fused(cfg))"
```

The first command runs the paper-worthiness rubric. The second one
exits 0 with the module summary; if you see `ImportError`, the
package did not register the entry-point scripts (re-run the
`pip install -e '.[dev]'` step above).

## Troubleshooting

### `pip install xaker` fails

The package is currently published from source only. The working
install is `git clone` + `pip install -e '.[dev]'`, not `pip install
xaker`.

### `import torch` fails with `undefined symbol`

Your pip resolver pulled in a CPU PyTorch build but your code path
expects CUDA. Reinstall with the matching `--index-url` (see
above) **before** running the package tests.

### `xaker-validate: command not found`

The install ran successfully but the entry-point scripts are not
on your `PATH`. Activate your virtualenv (see the `source
.venv/bin/activate` step above) or use `python -m
xaker.cli.validate` directly.

### Tests fail with `ModuleNotFoundError: pytorch`

You are inside a conda environment with PyTorch pinned to an
unsupported build. Remove the pin and reinstall with the pip
`--index-url` URL above.

## Project metadata

The repository surfaces the following metadata; the GitHub UI
mirrors the values listed here after you push the code:

| Field | Value |
|---|---|
| Name | `xaker` |
| Version | `0.5.1` |
| Description | Fused Exclusive Self Attention (XSA) + Kernel Ridge Regression for PyTorch Transformers. |
| Homepage | `https://github.com/sachncs/xaker` |
| Documentation | `https://sachncs.github.io/xaker/` |
| License | MIT |
| Topics | `attention`, `transformer`, `xsa`, `kernel-methods`, `kernel-ridge-regression`, `preconditioned-conjugate-gradient`, `pytorch`, `deep-learning` |

If you are publishing a fork, set the `name`, `description`, and
`urls.Homepage` fields in `pyproject.toml` to match your project.



## Next steps

- [Getting started](/xaker/getting-started/) — once the install is verified, run the four-line quickstart.
- [Troubleshooting](/xaker/troubleshooting/) — symptom-driven fixes for the rare mistakes a new user hits.
- [API reference](/xaker/api/) — once the package is installed, the public surface is documented there.
