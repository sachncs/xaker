# Epic G: Single-word Config dataclass fields

## User Story
As a user configuring a `Laker` model, I want every Config field to have a single-word name (`dim`, `heads`, `lam`, `pcg`, `tol`), so that YAML configs, CLI flags, and code are greppable and a typo on `dim` doesn't match `d_model`.

## Why this matters
- `d_model`, `num_heads`, `head_dim` carry the redundant `num_` / `_model` / `_dim` qualifiers that disappear once the field is single-word.
- `xsa_mode` → `mode`, `kernel_type` → `kernel`, `preconditioner_type` → `precond` — same role, single word.
- `use_fused`, `seed`, `num_iterations`, `clip_abs`, and the `effective_pcg_iters` property are deleted: their own docstrings admit nothing reads them.

## Acceptance Criteria
- [ ] Every field of `Config` has a single-word name.
- [ ] `use_fused`, `seed`, `num_iterations`, `clip_abs` are deleted.
- [ ] `effective_pcg_iters` property is deleted.
- [ ] Categorical fields (`kernel`, `mode`, `precond`) accept the new short literals (`"exp"`, `"subtract"`, `"cccp"`, ...).
- [ ] `__post_init__` validation passes for every documented literal.
- [ ] `pylint xaker/ --fail-under=9.5` passes; `mypy xaker/` 0 errors.

## Out of Scope
- Training config (`Fit`) field renames (Phase H).
- Adding new config fields.

## Technical Checklist (atoms)
- [ ] G01 `d_model` → `dim`
- [ ] G02 `num_heads` → `heads`
- [ ] G03 `head_dim` → `headdim`
- [ ] G04 `dropout` → `drop`
- [ ] G05 `eps` (unchanged)
- [ ] G06 `lambda_init` → `lam`
- [ ] G07 `kernel_type` → `kernel` (string enum: `"exp"|"rbf"|"linear"|"cosine"`)
- [ ] G08 `xsa_mode` → `mode` (string enum: `"subtract"|"zero"|"mask"`)
- [ ] G09 DELETE field `use_fused`
- [ ] G10 DELETE field `seed`
- [ ] G11 `preconditioner_type` → `precond` (string enum: `"cccp"|"fast"|"diagonal"|"identity"`)
- [ ] G12 `preconditioner_rank` → `rank`
- [ ] G13 `cccp_num_directions` → `directions`
- [ ] G14 `cccp_max_iterations` → `iters`
- [ ] G15 `cccp_gamma` → `gamma`
- [ ] G16 `cccp_shrinkage_rho` → `rho`
- [ ] G17 `cccp_shrinkage_eps` → `eps_shrink`
- [ ] G18 `pcg_max_iterations` → `pcg`
- [ ] G19 `pcg_tolerance` → `tol`
- [ ] G20 DELETE field `num_iterations`
- [ ] G21 `precond_update_frequency` → `freq`
- [ ] G22 `kernel_temperature` → `temp`
- [ ] G23 `kernel_symmetric` → `symmetric`
- [ ] G24 `kernel_normalize_qk` → `normalize`
- [ ] G25 DELETE field `clip_abs`
- [ ] G26 DELETE property `effective_pcg_iters`

## Definition of Done
- `git grep -nE 'd_model|num_heads|head_dim|dropout|lambda_init|kernel_type|xsa_mode|use_fused|preconditioner_type|preconditioner_rank|cccp_num_directions|cccp_max_iterations|cccp_gamma|cccp_shrinkage|pcg_max_iterations|pcg_tolerance|num_iterations|precond_update_frequency|kernel_temperature|kernel_symmetric|kernel_normalize_qk|clip_abs|effective_pcg_iters' xaker/config.py tests/ examples/` returns 0 (excluding G-explanatory comments)
- `pylint xaker/ --fail-under=9.5` passes
- `mypy xaker/ --no-implicit-optional --warn-unused-ignores` 0 errors
- `pytest tests/test_config.py -q` green after full rename
