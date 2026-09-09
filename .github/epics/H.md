# Epic H: Single-word Fit (training) dataclass fields

## User Story
As a user constructing a `Fit` training config, I want every field to have a single-word name (`lr`, `decay`, `epochs`, `warmup`, `grad`, `smooth`), so that YAML training configs are greppable and `Fit(seed=42)` is unambiguous.

## Why this matters
- `learning_rate`, `weight_decay`, `label_smoothing` — every field is two words; none need to be.
- `seed` is deleted from `Fit` because `xaker.utils.rng.seed` is the single RNG entry point (already named `seed`).
- `log_interval`, `eval_interval` → `log`, `eval` — the unit is implied by the trainer.

## Acceptance Criteria
- [ ] Every field of `Fit` has a single-word name.
- [ ] `seed` is removed from `Fit`; the CLI sets `seed` via `xaker.utils.rng.seed`.
- [ ] Trainer, CLI, examples, docs all use the new field names.

## Out of Scope
- Removing log/eval intervals (still used for caller-driven cadence).

## Technical Checklist (atoms)
- [ ] H01 `num_epochs` → `epochs`
- [ ] H02 `learning_rate` → `lr`
- [ ] H03 `weight_decay` → `decay`
- [ ] H04 `warmup_steps` → `warmup`
- [ ] H05 `max_grad_norm` → `grad`
- [ ] H06 `label_smoothing` → `smooth`
- [ ] H07 `log_interval` → `log`
- [ ] H08 `eval_interval` → `eval`
- [ ] H09 DELETE field `seed`

## Definition of Done
- `git grep -nE 'num_epochs|learning_rate|weight_decay|warmup_steps|max_grad_norm|label_smoothing|log_interval|eval_interval' xaker/training/trainer.py tests/ examples/` returns 0
- `pytest tests/test_training.py -q` green
