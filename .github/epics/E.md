# Epic E: Single-word function renames

## User Story
As a reader of the public API, I want every module-level function in `xaker/` to have a single-word name (`pcg`, `kernel`, `mask`, `clamp`, `seed`, `snapshot`, `restore`), so that the API surface feels intentional and grep-friendly.

## Why this matters
- `compute_kernel_matrix` and `apply_kernel_operator` already exist as single-word concepts; the multi-word names obscure that.
- The `clamp_tensor` vs `stable_clip` duplication is dissolved into one `clamp`.
- `set_seed` / `get_rng_states` / `set_rng_states` become `seed` / `snapshot` / `restore` — a single-word vocabulary for RNG control.

## Acceptance Criteria
- [ ] Every module-level function in `xaker/` has a single-word name.
- [ ] `clamp_tensor` and `stable_clip` collapse into one `clamp(t, lo=None, hi=None, eps=None)`.
- [ ] `set_seed` becomes `seed`; `get_rng_states` becomes `snapshot`; `set_rng_states` becomes `restore`.
- [ ] All imports across `xaker/`, `tests/`, `examples/`, `docs/` updated.

## Out of Scope
- Class names (Phase D), method names (Phase F).
- Deleting v1-only functions (Phase B covers).

## Technical Checklist (atoms)
- [ ] E01 `reshape_to_heads` → `heads`
- [ ] E02 `reshape_from_heads` → `merge`
- [ ] E03 `broadcast_mask` → `broadcast`
- [ ] E04 `apply_mask` → `mask`
- [ ] E05 `compute_kernel_matrix` → `kernel`
- [ ] E06 `apply_kernel_operator` → `op`
- [ ] E07 `pcg_solve` → `pcg`
- [ ] E08 `richardson_solve` → `richardson`
- [ ] E09 `create_causal_mask` → `causal`
- [ ] E10 `create_padding_mask` → `padding`
- [ ] E11 `verify_tensor_shapes` → `shape`
- [ ] E12 `set_seed` → `seed`
- [ ] E13 `get_rng_states` → `snapshot`
- [ ] E14 `set_rng_states` → `restore`
- [ ] E15 `check_finite` → `finite`
- [ ] E16 `clamp_tensor` → `clamp` (merge with `stable_clip`)
- [ ] E17 DELETE `stable_clip` (use `clamp` everywhere)
- [ ] E18 `label_smoothing_cross_entropy` → `ce`
- [ ] E19 `create_dummy_data` → `dummy`
- [ ] E20 `compute_kernel_condition_number` → `cond`
- [ ] E21 DELETE `compute_conditioning_metrics` (replaced by typed driver)
- [ ] E22 DELETE `long_context_benchmark` (replaced)
- [ ] E23 DELETE `runtime_profile` (replaced)
- [ ] E24 DELETE `profile_iterations` (v1)

## Definition of Done
- `git grep -nE '\bdef (reshape_to_heads|reshape_from_heads|broadcast_mask|apply_mask|compute_kernel_matrix|apply_kernel_operator|pcg_solve|richardson_solve|create_causal_mask|create_padding_mask|verify_tensor_shapes|set_seed|get_rng_states|set_rng_states|check_finite|clamp_tensor|label_smoothing_cross_entropy|create_dummy_data|compute_kernel_condition_number|long_context_benchmark|runtime_profile|profile_iterations|compute_conditioning_metrics)\b' xaker/ tests/ examples/` returns 0
- `pytest tests/ -q` green
