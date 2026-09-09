# Epic O: Tests renames + invariant tests

## User Story
As a maintainer, I want every test name to be a single word (`test_shape`, `test_grad`, `test_lam`, `test_zero`) and every polymorphism invariant to have a test, so that future regressions are caught by name and by behavior.

## Why this matters
- Test names like `test_laker_attention_gradient_flow` and `test_compute_kernel_condition_number` violate the single-word rule.
- Tests for v1 classes need to go (Phase B covers most); tests for new invariants (scale always trainable, three dispatch tables, `Cache` state-dict, `pcg` `Solve` return, fallback convergence, finiteness, `clamp`, RNG round-trip, `Ctx`, `cond`, bench driver) need to be added.

## Acceptance Criteria
- [ ] Every test method name is a single word (or `test_<one_concept>` for compound concepts that cannot be expressed in one word, e.g. `test_grad_block_params`).
- [ ] Every polymorphism invariant has a test.
- [ ] Tests cover PCG `Solve` convergence, fallback, finiteness.
- [ ] Tests cover `Make(config)`, `BLOCK[config.kind]`, `XsaStrategy(config, scale)`.
- [ ] Test coverage is ≥ 85%.

## Out of Scope
- Property-based tests beyond the listed invariants.

## Technical Checklist (atoms)
- [ ] O01 Rename test methods to single-word per rename map
- [ ] O02 Rename test classes per rename map
- [ ] O03 Delete tests for `LakerAttentionLayer`, `effective_pcg_iters`, v1 classes
- [ ] O04 Add `tests/test_property_attention.py`
- [ ] O05 Add `tests/test_convergence.py`
- [ ] O06 Add `tests/test_rubric.py`
- [ ] O07 Add `tests/test_bench.py`
- [ ] O08 Add `tests/test_examples.py`
- [ ] O09 Add `tests/test_repro.py`
- [ ] O10 Add `tests/test_dispatch.py`
- [ ] O11 Test `Xsa.xsa_scale` is always `nn.Parameter` regardless of `mode`
- [ ] O12 Test `BLOCK["standard"]`, `BLOCK["xsa"]`, `BLOCK["fused"]` are correct classes
- [ ] O13 Test `Make(config)` returns correct strategy
- [ ] O14 Test `XsaStrategy(config, scale)` returns correct strategy
- [ ] O15 Test `Cache` state-dict behavior for `Fast` and `Cccp`
- [ ] O16 Test `pcg` returns `Solve` with all fields
- [ ] O17 Test `Laker.attend` checks `solve.converged` and falls back
- [ ] O18 Test `Laker.attend` finite-output property
- [ ] O19 Test `clamp(t)` and `clamp(t, lo, hi)`
- [ ] O20 Test `seed(s)`, `snapshot()`, `restore(snapshot)` round-trip
- [ ] O21 Test `Ctx` construction
- [ ] O22 Test `cond(kernel)` returns per-(batch, head) SVD ratio
- [ ] O23 Test bench driver `run(spec)` produces valid `Result`

## Definition of Done
- `git grep -nE 'def test_[a-z]+_[a-z_]+ ?\(' tests/` returns 0
- `pytest tests/ -q --cov=xaker --cov-fail-under=85` passes
- `pylint tests/ --fail-under=9.5` passes
- `mypy xaker/` 0 errors
