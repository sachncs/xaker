# Epic I: Constants and module-level helpers — single-word

## User Story
As a reader scanning module tops, I want every constant and helper to have a single-word name, so that no leading-underscore 'private' trick is used as a substitute for proper module boundaries.

## Why this matters
- `TENSOR_CLIP_ABS` is a module-global constant in `attention/core.py`. Move it to `xaker/utils/ops.py` as `BOUND`.
- `_DEPRECATION_MSG`, `_SOFTPLUS` are pseudo-private names that vanish with the v1 deletion and inline `nn.Softplus()`.

## Acceptance Criteria
- [ ] No leading-underscore module-level identifier remains in `xaker/`.
- [ ] All clamp/clamp-related helpers are accessible via `xaker.utils.ops.clamp`.
- [ ] `_SOFTPLUS` instances replaced with `nn.Softplus()` in strategy classes.

## Out of Scope
- Class renames.

## Technical Checklist (atoms)
- [ ] I01 DELETE `TENSOR_CLIP_ABS` in `core.py`; replace with `BOUND = 1e6` in `xaker/utils/ops.py`
- [ ] I02 DELETE `_DEPRECATION_MSG` (file gone in Phase B)
- [ ] I03 DELETE `_SOFTPLUS` in `solver/precond.py` (use `nn.Softplus()`)
- [ ] I04 DELETE `_SOFTPLUS` in `laker_preconditioner.py` (file renamed; covered)

## Definition of Done
- `git grep -nE 'TENSOR_CLIP_ABS|_DEPRECATION_MSG|_SOFTPLUS' xaker/` returns 0
- `git grep -nE '\b_[a-zA-Z]' xaker/` returns only `__init__`, `__all__`, `__post_init__`, `__version__`
- `pytest tests/ -q` green
