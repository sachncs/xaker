# Epic T: Verification gates — final Definition of Done

## User Story
As a paper author, I want a documented set of greps and test runs that prove every claim in the README and rubric, so that I can defend the publication against reviewer skepticism.

## Why this matters
- Without gates, the project cannot claim paper-worthy.
- Grep gates are cheap, fast, and catch drift.
- Test gates are the substantive content.

## Acceptance Criteria
- [ ] All T01-T23 grep gates return expected counts.
- [ ] `pytest -q --cov-fail-under=85` passes.
- [ ] `pylint xaker/ --fail-under=9.5` passes.
- [ ] `mypy xaker/ --warn-unused-ignores --no-implicit-optional` 0 errors.
- [ ] `xaker-validate --min-total 14 --repo-root .` exits 0.
- [ ] `python -m examples.run_paper_experiment --spec examples/specs/rubric.yaml --check` exits 0.
- [ ] Library LOC ≤ 2,200.

## Out of Scope
- Performance benchmarks as gate (run as part of M phase).

## Technical Checklist (atoms)
- [ ] T01 `git grep -rn 'laker_xsa|xaker|XAKER' .` returns 0 (excl. CHANGELOG history)
- [ ] T02 `git grep -nE '\bdef [a-z]+_[a-z_]+ ?\(' xaker/ tests/ examples/` returns 0
- [ ] T03 `git grep -nE '\bclass [A-Z][a-zA-Z]*[A-Z]' xaker/` returns 0
- [ ] T04 `git grep -nE '\b_[a-zA-Z]' xaker/` returns only dunders
- [ ] T05 `git grep -rn 'if.*mode ==|elif.*mode ==' xaker/` returns 0
- [ ] T06 `git grep -rn 'Precond as|noqa.*F401|as PrecondModule' .` returns 0
- [ ] T07 `pytest tests/ -q --strict-markers` green
- [ ] T08 `pylint xaker/ --fail-under=9.5` passes
- [ ] T09 `mypy xaker/ --warn-unused-ignores --no-implicit-optional` 0 errors
- [ ] T10 `xaker-validate --min-total 14 --repo-root .` exits 0
- [ ] T11 `python -m examples.run_paper_experiment --spec examples/specs/rubric.yaml --check` exits 0
- [ ] T12 Library LOC ≤ 2,200
- [ ] T13 Coverage ≥ 85%
- [ ] T14 `paper_runs/baseline.json` generated with all required metrics
- [ ] T15 Update `MANIFEST.in` to ship `examples/specs/*.yaml`
- [ ] T16 Update `.gitignore`: `paper_runs/*.json` ignored, `*.md` kept
- [ ] T17 `git grep -nE '\bclass [A-Z][a-z]+[A-Z]' xaker/` returns 0
- [ ] T18 `git grep -nE '\bdef [a-z]+_[a-z_]+ ?\(' xaker/ tests/ examples/` returns 0
- [ ] T19 `git grep -rn 'attention_type ==|if.*kind ==|if.*mode ==|if.*precond ==' xaker/` returns 0
- [ ] T20 `git grep -rn 'LakerAttentionLayer|KernelAttentionRegression|FusedXSALAKERAttention|LearnedPreconditioner|KernelFunction|effective_pcg_iters|use_fused' .` returns 0
- [ ] T21 `pytest tests/ --cov=xaker --cov-fail-under=85 -q` passes
- [ ] T22 `pylint xaker/ --fail-under=9.5 --disable=missing-docstring` passes
- [ ] T23 `mypy xaker/ --no-implicit-optional --warn-unused-ignores` 0 errors

## Definition of Done
- T01-T23 all pass
- `paper_runs/rubric_summary.md` exists and lists `total >= 14`
