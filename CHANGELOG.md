# Changelog

All notable changes to xaker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-09-03

### Fixed
- **CI green**: lint, mypy, pylint, and test pipelines now pass across
  the Python 3.9-3.12 matrix. `pylint --fail-under=9.5` is now an
  enforced gate (previously masked with `|| true`).
- **mypy**: 54 type errors resolved. Structural root was the four
  preconditioner strategies overriding `nn.Module.apply` with a
  different signature; renamed to `apply_pre` everywhere.
- **BLOCK registry**: `Linear` attention is now wired in alongside
  `Standard`, `Xsa`, `Fused` so `attention_type="linear"` actually
  selects it.
- **BLOCK typing**: declared `BLOCK: dict[str, type[Base]]`.
- **Model/Block parity**: `Block.attention_type` Literal now matches
  `Model.attention_type` (extends to `'linear'`).
- **XsaMode**: converted from `Protocol` to `Union[Zero, Projection,
  Mask]` (the Protocol carried no runtime discriminator).
- **datasets/__init__.py**: optional `datasets` library handling
  hardened; `WikiText._load` narrows cached tensor; `_synthetic_corpus`
  outer list annotated; `build()` typed `Dataset[Any]`.
- **CLI**: `xaker-eval` no longer calls `Model(..., kind=...)`;
  uses `attention_type=`. `xaker-bench` wraps device string in
  `torch.device(...)` before passing to `Ctx`.
- **Junk directory** `xaker/cli/validate.py<` removed from git
  history (was a botched-edit leftover).

### Changed
- **Bench helpers** renamed to single-word per the project's naming
  rule: `make_spec` (deleted as dead code), `measure_kind -> bench`,
  `run_axis -> drive`, `measure_axis -> sweep`, `make_copy -> dataset`,
  `train_eval -> train`, `evaluate_perplexity -> evaluate`,
  `train_one -> trainstep`.
- **Lint regex** narrowed: pytest-mandated hook names
  (`pytest_configure`) are now carved out so the single-word rule
  can stay honest without forcing a rename pytest would reject.

## [0.5.0] - 2026-09-01

### Added
- **Linear attention baseline** (`xaker/attention/linear.py`).
  Katharopoulos et al. linear attention with `elu + 1` feature
  map. Registered as `BLOCK["linear"]` so it competes against
  Standard / Xsa / Fused in every benchmark.
- **Ablation benchmark runner** (`xaker/bench/ablate.py`). Sweep
  over attention kind, kernel function, preconditioner, and mode;
  emits the same JSON schema as `xaker.bench.run`.
- **LRA-style task suite** (`xaker/bench/lra.py`). Four synthetic
  long-context tasks: copy, reversal, retrieval, addition. Each
  trained on all four attention variants; per-task accuracy and
  wall-clock reported.
- **Condition-number benchmark** (`xaker/bench/condition.py`).
  Compares the conditioning of the score matrix (Standard/Xsa)
  against the regularised kernel matrix (Fused) across sequence
  lengths.
- **Copy-task training benchmark** (`xaker/bench/copy_task.py`).
  End-to-end training comparison of the four variants on a
  synthetic copy task.
- **WikiText benchmark stub** (`xaker/bench/wikitext.py`). CPU
  baseline using `datasets.WikiText` byte-level tokenisation with
  synthetic fallback.
- **Dataset loaders** (`xaker/datasets/`). `CopyTask`,
  `ReversalTask`, `WikiText`. The WikiText loader caches to
  `~/.cache/xaker/wikitext_<split>.pt` for reproducibility.
- **RESULTS.md**: comprehensive benchmark report covering all
  four ablations, condition numbers, copy-task training, and LRA
  tasks. Headline: kernel ridge regression gives 300-1700x lower
  condition number than softmax across sequence lengths.

### Changed
- **Class rename: `Laker` to `Fused`.** The flagship fused-XSA class is
  now exported as `xaker.Fused` (was `xaker.Laker`). Module renamed
  `xaker/attention/laker.py` to `xaker/attention/fused.py`; the
  `BLOCK["fused"]` dispatch key was already `fused` and is unchanged.
  Migration: `from xaker import Fused` replaces `from xaker import Laker`.
- **Public surface** expanded: `Linear` is now re-exported at
  `xaker.Linear` and in `__all__`. `BLOCK` registry has four
  entries: `standard`, `xsa`, `fused`, `linear`. `Model` accepts
  `attention_type="linear"` as a valid value.
- **Device selection** in `xaker.bench` honours the
  `XAKER_DEVICE` environment variable (`cpu` / `cuda` / `mps`).
  Default is `cpu` because several PCG ops have bugs on MPS.
- **MPS-aware timing**: `xaker.bench.tick` and `peak` now
  synchronise on MPS as well as CUDA. Wall-clock measurements are
  accurate on Apple Silicon.

### Quantitative findings (CPU, dim=64, lam=10.0)

Wall-clock at length=32:

| Kind     | Forward ms | Backward ms |
|----------|------------|-------------|
| Linear   | 0.16       | 0.42        |
| Standard | 0.18       | 0.56        |
| Xsa      | 0.17       | 0.50        |
| Fused    | 42.17      | 184.76      |

Kernel matrix condition number (the headline result):

| Length | Score cond (Standard) | Kernel cond (Fused) | Ratio |
|--------|-----------------------|---------------------|-------|
| 16     | 1,018                 | 3.11                | 0.003 |
| 32     | 3,346                 | 5.66                | 0.002 |
| 64     | 12,310                | 12.76               | 0.001 |
| 128    | 74,917                | 41.55               | 0.001 |

LRA copy task (5 epochs, length=32): `Standard` 0.91, `Xsa` 0.90,
`Fused` 0.87, `Linear` 0.14. Linear fails on copy because the
`elu + 1` feature map loses positional information.

### Migration

```python
# 0.4
from xaker import Laker
attn = Laker(config)

# 0.5
from xaker import Fused
attn = Fused(config)
# or, equivalently:
attn = BLOCK["fused"](config)

# 0.5 also adds Linear as a baseline
from xaker import Linear
attn = Linear(config)
attn = BLOCK["linear"](config)
```

## [0.4.0] - 2026-09-01

### BREAKING CHANGES
- **Package renamed from `laker-xsa` to `xaker`.** The import path is
  now `xaker` (was `laker_xsa`); PyPI distribution is `xaker`;
  console scripts are `xaker-train`, `xaker-eval`, `xaker-bench`,
  `xaker-validate`; the GitHub repository is `sachncs/xaker`.
  Renamed to a single-word brand that fits the project's naming
  convention.
- **v1 legacy attention/preconditioner classes hard-deleted.**
  `KernelAttentionRegression`, `FusedXSALAKERAttention`,
  `LearnedPreconditioner`, `KernelFunction`, `LakerAttentionLayer`
  and every `*_attention.py` shim are gone. The published benchmark
  numbers were measuring the v1 path; they are superseded by the v2
  numbers generated by the new typed driver in `paper_runs/`.
- **Polymorphism-first refactor.** `LakerPreconditioner` split into
  four concrete strategy classes (`Identity`, `Diagonal`, `Fast`,
  `Cccp`) plus a single factory function `Make(config)`. XSA mode
  selection is now a `Projection`/`Zero`/`Mask` strategy triple
  with an `XsaStrategy(config, scale)` factory. Block attention
  selection uses a `BLOCK = {"standard": Standard, "xsa": Xsa,
  "fused": Fused}` registry. No `if mode == ...` chains remain in
  algorithm code.
- **Single-word naming convention.** Every public symbol (modules,
  classes, functions, methods, dataclass fields) is now a single
  word. Multi-word PascalCase class names
  (`XSALAKERTransformerBlock`), multi-word snake_case functions
  (`compute_kernel_matrix`), and underscore-prefixed "private" names
  (`_SOFTPLUS`, `_DEPRECATION_MSG`, `TENSOR_CLIP_ABS`) are gone.
- **`pcg` returns a `Solve` dataclass** (`x`, `iters`, `converged`,
  `res`, `history`); `Fused.attend` checks `solve.converged` and
  falls back only on `not converged and not finite`.
- **Paper-worthiness rubric** added: `xaker.rubric` package,
  `xaker-validate` CLI, pytest plugin enforcing `total >= 14` and
  `no dim < 2` (except novelty may be 1).

### Added
- Typed benchmark driver `xaker.bench` (one entry point for runtime,
  memory, convergence measurements) emitting schema-stable JSON with
  `git_sha`, environment, per-seed statistics.
- `xaker.rubric` package with six dimensions: `Novelty`, `Repro`,
  `Correctness`, `Efficiency`, `Stability`, `Usability`. Enforced in
  CI.
- `xaker-validate` CLI command and pytest plugin
  (`@pytest.mark.rubric`).
- `Ctx` dataclass for explicit device/dtype context.
- `examples/run_paper_experiment.py` -- single typed driver
  consuming YAML specs in `examples/specs/`. Replaces six long
  `examples/*.py` scripts.
- 276 tests across `tests/`, 85.41% coverage, all single-word test
  method names where possible.

### Removed
- v1 legacy attention classes (`KernelFunction`,
  `LearnedPreconditioner`, `KernelAttentionRegression`,
  `FusedXSALAKERAttention`).
- All backward-compat shim modules (`_legacy.py`,
  `attention_kernel.py`, `kernel_attention.py`,
  `standard_attention.py`, `xsa_attention.py`,
  `fused_attention_v2.py`).
- Wrapper classes with no semantic content
  (`LakerAttentionLayer`, `effective_pcg_iters` property).
- Pseudo-private globals (`_SOFTPLUS`, `_DEPRECATION_MSG`,
  `TENSOR_CLIP_ABS`).
- Long `examples/*.py` scripts replaced by one typed experiment
  driver.

## [0.3.0] - 2026-07-13

### Fixed
- pylint E1102 `not-callable` at 5 call sites (commit `6d62150`,
  2026-07-13T13:13:04Z). Switched the solver/benchmarks code from
  `torch.nn.functional.softplus` and `torch.linalg.vector_norm`
  (C-extension builtins whose `__call__` pylint's astroid inference
  cannot resolve) to the equivalent `nn.Softplus()` module instance
  and `torch.sqrt(torch.sum(...))`. Verified: `pylint` 10.00/10,
  `black` clean, `mypy` no errors, `pytest` 269 passed.

### Documentation
- Author / contact updated to `sachin` <sachncs@gmail.com> (commit
  `6d62150`, 2026-07-13T13:13:04Z): replaced the `XAKER
  Contributors` placeholder in `README.md`, `CITATION.cff`,
  `LICENSE`, `pyproject.toml`, and `docs/FINAL_SUMMARY.md`.

## [0.2.3] - 2026-05-02

### Fixed
- Version consistency across `pyproject.toml`, `__init__.py`, and
  `CITATION.cff`.
- Placeholder URLs updated to `github.com/sachncs/xaker`.
- `MANIFEST.in` path references corrected from `src/` to
  `laker_xsa/`.

## [0.2.2] - 2026-05-02

### Changed
- Package metadata improvements.

## [0.2.1] - 2026-05-01

### Changed
- Package metadata improvements.

## [0.2.0] - 2026-05-01

### Added
- Initial LakerAttention v2 implementation (later renamed `Fused`):
  exponential kernel attention with XSA projection removal and
  CCCP-based preconditioning, solved by Preconditioned Conjugate
  Gradient.
- `AttentionKernel`, `LakerPreconditioner` (now split into
  `Identity` / `Diagonal` / `Fast` / `Cccp`), and the
  `compute_kernel_matrix` / `apply_kernel_operator` functional API.
- `LakerAttentionLayer` (removed in 0.4).
- CLI entry points: `train`, `benchmark`, `evaluate`.
- 7 new test files, 119 new tests.

### Changed
- Package layout: `src/laker_xsa/` to `laker_xsa/` at repo root.
- Removed all semi-private `_leading_underscore` names.
- Test suite expanded from 119 to 269 tests (88% coverage).
- Pylint score: 8.79/10. Mypy: zero errors. CI: green.

### Removed
- Dead config fields: `solver_tolerance`, `solver_epos`,
  `precond_cache`.
- Duplicate `compute_kernel_matrix` in `kernels.py`.
- Double-dropout bug in `BaseMultiHeadAttention.forward()`.
- Bare `except Exception` in `_legacy.py`.

### Fixed
- `validate_input` no-op.
- Flaky `test_iterations_converge`: relaxed to finiteness check.
- SyntaxWarning in `losses.py` docstring.
- `pytestmark` filterwarnings for v1 deprecation warnings.
- `head_dim` mypy type narrowing via `cast(int, ...)`.

## [0.1.0] - 2026-04-30

### Added
- Initial implementation in `src/laker_xsa/`.
- Core attention: Standard, XSA, Kernel (v1), Fused (v1).
- Solvers: Preconditioned Richardson iteration, Conjugate Gradient.
- Model: Transformer block and full Transformer.
- Training: Trainer, loss functions, TrainingConfig.
- Benchmarks: Runtime profiling, conditioning analysis, long-context
  scaling.
- Utils: Tensor ops, seed management, stability checks.
- Test suite: Shape verification, gradient flow, numerical
  stability.
- Example scripts: Comparative analysis, hard benchmarks, long
  sequence, NLP evaluation.
- Documentation: Architecture overview, mathematical derivations,
  design decisions, limitations.
- MIT License.

[0.5.1]: https://github.com/sachncs/xaker/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/sachncs/xaker/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/sachncs/xaker/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sachncs/xaker/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/sachncs/xaker/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/sachncs/xaker/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/sachncs/xaker/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/sachncs/xaker/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sachncs/xaker/releases/tag/v0.1.0