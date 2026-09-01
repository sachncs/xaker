# Epic S: CI enforces single-word + rubric + coverage

## User Story
As a maintainer, I want CI to fail any PR that violates the single-word rule, drops rubric score below 14, or drops coverage below 85%, so that the repository cannot drift away from paper-worthy.

## Why this matters
- A repository without enforcement cannot stay clean.
- Rubric enforcement is what makes the published claims defensible.
- Coverage threshold prevents accidental regression in critical paths.

## Acceptance Criteria
- [ ] CI runs single-word enforcement grep.
- [ ] CI runs `xaker-validate --min-total 14`.
- [ ] CI runs `pytest -m rubric`.
- [ ] CI runs `pytest --cov-fail-under=85`.
- [ ] CI runs `pylint xaker/ --fail-under=9.5`.
- [ ] CI runs `mypy xaker/ --warn-unused-ignores --no-implicit-optional`.
- [ ] Slow tests run on a nightly job.

## Out of Scope
- Adding code coverage services.

## Technical Checklist (atoms)
- [ ] S01 Add single-word enforcement grep in `.github/workflows/ci.yml`
- [ ] S02 Update CI to run `xaker-validate --min-total 14`
- [ ] S03 Update CI to run `pytest -m rubric`
- [ ] S04 Update CI to run `pytest --cov-fail-under=85`
- [ ] S05 Update CI to run `pylint xaker/ --fail-under=9.5`
- [ ] S06 Update CI to run `mypy xaker/ --warn-unused-ignores --no-implicit-optional`
- [ ] S07 Add `--strict-markers` and `--randomly-seed=1234` to pytest invocation
- [ ] S08 Add `BOUND` exposure check (single global constant)
- [ ] S09 Add `pytest -m "not slow"` for default CI run; `pytest -m slow` for nightly

## Definition of Done
- CI badge in README reflects green build
- All CI steps pass on the post-refactor tree
- Any single-word violation fails CI
