# Epic F: Single-word method renames

## User Story
As a reader of class definitions, I want every method to have a single-word name (`attend`, `init`, `lam`, `temp`, `build`, `apply`, `step`, `eval`, `loss`), so that reading the class surface takes one mental beat per method.

## Why this matters
- `compute_attention` is verbose; the verb is `attend`, the noun is `attention`; the class is `Xsa`/`Laker`. One word is enough.
- `init_weights` → `init`, `lambda_reg` → `lam`, `compute_preconditioner` → `build`, `apply_preconditioner` → `apply`, `train_step` → `step`, `compute_loss` → `loss` — every one of these is a one-word swap.
- Internal-only helpers (`zero_diagonal`, `clean_self_projection`, `rms_normalize` on `Laker`) are deleted and replaced by free functions (`zerodiag`, `rms`) plus the strategy pattern (Phase J).

## Acceptance Criteria
- [ ] Every method on every class in `xaker/` has a single-word name.
- [ ] `Laker.zero_diagonal` and `Laker.rms_normalize` become free functions `zerodiag(K)` and `rms(x, eps)` in `xaker/attention/ops.py`.
- [ ] `Laker.clean_self_projection` is removed; the LAKER path uses the same `Projection.apply` strategy as XSA.
- [ ] `Trainer.step_count` is renamed `Trainer.iter`.
- [ ] `Precond.generate_angular_samples`, `.cccp_iteration`, `.cccp_preconditioner`, `.fast_preconditioner`, `.diagonal_preconditioner` are deleted (inlined into the strategy classes).

## Out of Scope
- Strategy pattern (Phase J).
- Function-level renames (Phase E).

## Technical Checklist (atoms)
- [ ] F01 `Base.validate_input` → `Base.check`
- [ ] F02 `Base.compute_attention` → `Base.attend`
- [ ] F03 `Laker.init_weights` → `Laker.init`
- [ ] F04 `Laker.lambda_reg` property → `Laker.lam`
- [ ] F05 `Laker.compute_attention` → `Laker.attend`
- [ ] F06 `Laker.zero_diagonal` → DELETE (move to free function `zerodiag`)
- [ ] F07 `Laker.clean_self_projection` → DELETE (use `Projection.apply`)
- [ ] F08 `Laker.rms_normalize` → DELETE (move to free function `rms`)
- [ ] F09 `Xsa.compute_attention` → `Xsa.attend`
- [ ] F10 `Kernel.temperature` property → `Kernel.temp`
- [ ] F11 `Kernel.log_temperature` attr → `Kernel.logtemp`
- [ ] F12 `Trainer.train_step` → `Trainer.step`
- [ ] F13 `Trainer.train_epoch` → `Trainer.epoch`
- [ ] F14 `Trainer.evaluate` → `Trainer.eval`
- [ ] F15 `Trainer.compute_loss` → `Trainer.loss`
- [ ] F16 `Trainer.step_count` attr → `Trainer.iter`
- [ ] F17 `Precond.compute_preconditioner` → `build`
- [ ] F18 `Precond.apply_preconditioner` → `apply`
- [ ] F19 DELETE `Precond.generate_angular_samples`
- [ ] F20 DELETE `Precond.cccp_iteration`
- [ ] F21 DELETE `Precond.cccp_preconditioner`
- [ ] F22 DELETE `Precond.fast_preconditioner`
- [ ] F23 DELETE `Precond.diagonal_preconditioner`

## Definition of Done
- `git grep -nE '\bdef (compute_attention|init_weights|lambda_reg|compute_preconditioner|apply_preconditioner|train_step|train_epoch|compute_loss|generate_angular_samples|cccp_iteration|cccp_preconditioner|fast_preconditioner|diagonal_preconditioner|validate_input)\b' xaker/` returns 0
- `pytest tests/ -q` green
