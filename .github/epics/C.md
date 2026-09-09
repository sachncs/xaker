# Epic C: Single-word file renames

## User Story
As a reader navigating the source tree, I want every module file to have a single-word name, so that I can locate a concept (kernel, block, model, rng, ops) in one mental hop and not have to remember which snake_case synonym the codebase happens to use.

## Why this matters
- The repo currently mixes single-word (`xsa.py`) with multi-word (`laker_preconditioner.py`, `transformer_block.py`, `tensor_ops.py`) filenames.
- File naming is part of the public surface: `from xaker.solver.precond import Make` reads better than `from xaker.solver.laker_preconditioner import LakerPreconditioner`.
- Consistent single-word names pair cleanly with single-word class names (Phase D), single-word function names (Phase E), and single-word config fields (Phase G).

## Acceptance Criteria
- [ ] Every module file in `xaker/`, `tests/`, and `examples/` has a single-word filename (no underscores except where the word is genuinely compound like `paper_runs/`).
- [ ] No internal `from xaker.*.foo_bar` paths survive the rename.
- [ ] All imports updated; `pytest -q` green after every rename.
- [ ] New files created: `xaker/attention/ops.py`, `xaker/bench/bench.py`, `xaker/rubric/` package skeleton.

## Out of Scope
- Function/class renames inside the files (Phase D, E, F).
- Replacing v1 files (already deleted in Phase B).

## Technical Checklist (atoms)
- [ ] C01 `attention/functional.py` → `attention/func.py`
- [ ] C02 `attention/kernels.py` → `attention/kernel.py`
- [ ] C03 `attention/laker.py` (already single-word, unchanged)
- [ ] C04 `attention/standard.py` (already single-word, unchanged)
- [ ] C05 `attention/xsa.py` (already single-word, unchanged)
- [ ] C06 `attention/core.py` (already single-word, unchanged)
- [ ] C07 `solver/conjugate_gradient.py` → `solver/cg.py`
- [ ] C08 `solver/functional.py` → `solver/func.py`
- [ ] C09 `solver/laker_preconditioner.py` → `solver/precond.py`
- [ ] C10 `model/full_model.py` → `model/model.py`
- [ ] C11 `model/transformer_block.py` → `model/block.py`
- [ ] C12 `training/losses.py` → `training/loss.py`
- [ ] C13 `training/trainer.py` (already single-word, unchanged)
- [ ] C14 `utils/seed.py` → `utils/rng.py`
- [ ] C15 `utils/stability.py` → `utils/finite.py`
- [ ] C16 `utils/tensor_ops.py` → `utils/ops.py`
- [ ] C17 `cli/benchmark.py` → `cli/bench.py`
- [ ] C18 `cli/evaluate.py` → `cli/eval.py`
- [ ] C19 `cli/train.py` (already single-word, unchanged)
- [ ] C20 `benchmarks/` → `bench/`
- [ ] C21 `benchmarks/conditioning.py` → `bench/cond.py`
- [ ] C22 Delete `benchmarks/long_context.py`
- [ ] C23 Delete `benchmarks/runtime.py`
- [ ] C24 Create `bench/bench.py` (typed driver)
- [ ] C25 Create `rubric/` package skeleton
- [ ] C26 Create `attention/ops.py` (free functions `zerodiag`, `rms`)

## Definition of Done
- `git ls-files | grep -E 'functional\.py|conjugate_gradient|laker_preconditioner|full_model|transformer_block|losses\.py|seed\.py|stability\.py|tensor_ops|benchmark\.py|evaluate\.py|long_context|runtime\.py'` returns 0
- `pytest tests/ -q` green after each rename
