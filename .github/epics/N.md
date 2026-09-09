# Epic N: Paper-worthiness rubric (`xaker.rubric`)

## User Story
As a paper author, I want a six-dimension rubric (Novelty, Reproducibility, Correctness, Efficiency, Stability, Usability) that scores the codebase 0-3 per dimension and gates CI on `total >= 14` and `no dim < 2`, so that the published claims are defensible from evidence inside the repo.

## Why this matters
- Without a rubric, paper claims reduce to "trust me" — the published RESULTS.md says fused shows no accuracy benefit, but the numbers come from v1. That is a paper-worthy defect.
- A reproducible grader that CI enforces is the only way to keep the rubric honest over time.

## Acceptance Criteria
- [ ] `xaker.rubric.dimensions` defines `Novelty`, `Repro`, `Correctness`, `Efficiency`, `Stability`, `Usability`.
- [ ] `xaker.rubric.grader.grade(repo_root)` returns a `RubricResult`.
- [ ] `xaker.rubric.reporting` renders Markdown and JSON.
- [ ] `xaker.rubric.plugin` is a pytest plugin providing `@pytest.mark.rubric`.
- [ ] `xaker/cli/validate.py` exposes `xaker-validate`.
- [ ] `tests/conftest.py` enables the plugin.
- [ ] `xaker-validate --repo-root . --min-total 14` exits 0.

## Out of Scope
- Scoring weights beyond the six dimensions.

## Technical Checklist (atoms)
- [ ] N01 Create `xaker/rubric/dimensions.py`
- [ ] N02 Create `xaker/rubric/rubric.py`
- [ ] N03 Create `xaker/rubric/grader.py`
- [ ] N04 Create `xaker/rubric/reporting.py`
- [ ] N05 Create `xaker/rubric/plugin.py`
- [ ] N06 Create `xaker/cli/validate.py`
- [ ] N07 Wire `xaker-validate` into `pyproject.toml [project.scripts]`
- [ ] N08 Add `tests/test_rubric.py`
- [ ] N09 Add `tests/conftest.py` with `pytest_plugins = ["xaker.rubric.plugin"]`
- [ ] N10 Add shared `rubric_result` fixture in `conftest.py`

## Definition of Done
- `xaker-validate --repo-root . --min-total 14` exits 0
- `pytest -m rubric -q` green
- `paper_runs/rubric_summary.md` regenerated
- `pylint xaker/rubric/ --fail-under=9.5` passes
