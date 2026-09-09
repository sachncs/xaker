---
layout: page
title: Troubleshooting
description: Common failures, what they mean, and how to recover. Read this when xaker does something unexpected.
permalink: /troubleshooting/
---

**Audience: anyone hitting an unexpected error.**

**Time: minutes, by symptom.**

The goal of this page is fast diagnosis. Every entry lists the symptom,
the most likely root cause in one sentence, the fastest fix, and a
pointer to the longer write-up.

## Installation

### `pip install xaker` fails with "No matching distribution found"

Symptom: `ERROR: Could not find a version that satisfies the requirement xaker`.

Cause: `xaker` is not yet on PyPI.

Fix: install from source instead. See [Installation](/xaker/installation/),
then `git clone https://github.com/sachncs/xaker.git` and
`pip install -e '.[dev]'`.

### `import torch` fails with `undefined symbol: ...`

Symptom: import of `torch` fails after a fresh install.

Cause: PyTorch was installed for CPU, then `import xaker` is using
an op that requires the CUDA build (or vice versa).

Fix: pick one PyTorch build (CPU or CUDA), install it with the matching
index URL, then install `xaker` from source over the top. See
[PyTorch CUDA wheels](/xaker/installation/#pytorch-cuda-wheels).

### `xaker-validate: command not found`

Symptom: the console script is not on `$PATH`.

Cause: the install ran successfully but the active shell does not
point at the virtualenv where the entry points were registered.

Fix: activate the virtualenv
(`source .venv/bin/activate`) or invoke the script directly with
`python -m xaker.cli.validate`.

## Runtime

### NaNs in `Fused.attend` on fp16 / bf16 inputs

Symptom: `out = attn(x)` produces `nan` or `inf` rows.

Cause: the dtype frontier described in
[Limitations](/xaker/limitations/#dtype-contract). Most common
sources are:

- input magnitude above the kernel's exp saturation point;
- `Linear.feature_clamp` was bypassed by an explicit conversion;
- `BOUND = 1e6` is exceeded because the kernel is not well-conditioned.

Fix: cast inputs to `fp32` for the experiment, then narrow down the
guard that fails. The `elephants` test in `tests/test_numerics.py`
exercises fp16 + bf16 with the shipped clamp and asserts finiteness.

### PCG does not converge, dense solve fires every forward pass

Symptom: `logger.warning("PCG did not converge; ...")` prints once per
forward pass, and wall-clock per forward skyrockets.

Cause: the regularised system `(K + λI)α = v` is poorly conditioned
because the chosen kernel + length combination is outside the solver's
convergence envelope at the current PCG iteration budget.

Fix: raise `Config.pcg` (default 20) and tighten `Config.tol`
(default `1e-2`). If convergence is required, switch the preconditioner
to `diagonal` or `fast`; `cccp` is the heaviest and best-converging,
`identity` is the cheapest and often insufficient.

### `torch.linalg.solve` raises on a 4-D kernel on Apple Silicon

Symptom: `Fused.attend` raises inside the fallback path on macOS.

Cause: PyTorch's MPS backend has shape bugs on 4-D batched
`linalg.solve` / `linalg.eigh` (see [Limitations](/xaker/limitations/)).

Fix: run on CPU (`xaker.bench` and the bundled runners default to CPU
for that reason), or on a CUDA host. MPS is supported for training
inference but the dense solve fallback is not.

### Kernel-cache warnings during training

Symptom: `Fast` preconditioner logs a warning that its cache is stale.

Cause: the cached payload was invalidated by an optimiser step
that reallocated the `lr_base` / `lr_imp` / `scale` storage. This
is expected behaviour; the cache rebuilds on the next forward.

Fix: do nothing — the rebuild path is correct. If the warning is
noisy, call `fast.reset_cache()` once per epoch (manually) so
the rebuild is predictable.

## Numerics

### Output drift across PyTorch versions

Symptom: a paper table that you re-run in `xaker.bench` no longer
matches `paper_runs/*.json` to the last digit.

Cause: PyTorch's `matmul`, `linalg.solve`, and `softplus` have
historically differed across versions in subtle ways (precision
of fused-multiply-add, ordering of reductions, etc.).

Fix: pin PyTorch + Python + a single `torch.cuda` deterministic
configuration. `xaker.utils.rng.seed(N)` plus the commit
SHA captured in the `git_sha` field of the JSON is the right
combination. If exact reproduction is impossible, expect
single-digit-percent drift and call it out in the write-up.

### "Softmax-shaped" outputs from `Linear` look wrong on copy

Symptom: the Linear baseline reaches near-zero accuracy on copy tasks.

Cause: `Linear` uses `elu(x) + 1` as the feature map. That map
loses positional information, so copy tasks at length ≥ 16 collapse.
This is by design — `Linear` is a baseline, not a position-aware
attend block.

Fix: switch `attention_type` to `standard`, `xsa`, or `fused` for
position-sensitive tasks. See [FAQ](/xaker/faq/#why-does-linear-fail-on-the-lra-copy-task).

## Benchmarks

### `xaker-bench` writes JSON to the wrong place

Symptom: the `--output` flag is ignored.

Cause: an older release used positional args; the current CLI is
flag-only.

Fix: always pass `--output <path>` explicitly. Use
`paper_runs/<spec>.json` for committed JSONs.

### `xaker-validate` exits non-zero on a clean repo

Symptom: the rubric gate fails despite tests + lint + mypy passing.

Cause: at least one of the six rubric dimensions is below the
required threshold. The output lists each dimension with its
evidence string. Read the string; it tells you which grader
under-indexed.

Fix: address the grader, not the score. Common regressions:

- `novelty` — `xaker.py` no longer exists. Re-check the variant list.
- `repro` — `paper_runs/` lost a JSON. Re-run the driver.
- `correctness` — `tests/test_fused.py` removed. Restore or extend.
- `efficiency` — fewer than two committed JSONs in `paper_runs/`.
- `stability` — no dtype sweep under `xaker/bench/`.
- `usability` — README, CLI, or rubric removed.

See [Paper rubric](/xaker/paper_rubric/) for the full grading contract.

## Filing an issue

Use [GitHub Issues](https://github.com/sachncs/xaker/issues) for
bug reports and feature requests. The minimum reproduction script
should:

1. Pin `python --version`, `pip show torch | grep Version`, and
   the `xaker.__version__`.
2. Print `xaker.bench.gitsha()` so the commit is on the record.
3. Run the failing call inside a `with torch.no_grad():` where
   relevant.
4. Capture the rubric score (`xaker-validate --min-total 14`) and
   the first ~20 lines of traceback.

Without these, the maintainer cannot reproduce the failure.



## Next steps

- [Installation](/xaker/installation/) — install path, CUDA wheels,
  environment isolation.
- [Limitations](/xaker/limitations/) — the dtype frontier and the
  solver fallback contract.
- [FAQ](/xaker/faq/) — short answers to questions that recur.
- [Design decisions](/xaker/design_decisions/) — the why behind
  every choice that has bitten a contributor in the past.
