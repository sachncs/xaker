# Epic U: Misc bookkeeping (MANIFEST, .gitignore, optional deps)

## User Story
As a maintainer, I want `MANIFEST.in`, `.gitignore`, and `pyproject.toml` to reflect the new repo state, so that wheel builds ship the right files and the git tree stays clean.

## Why this matters
- `MANIFEST.in` does not yet ship `examples/specs/*.yaml`; the typed driver breaks in a wheel install without them.
- `.gitignore` does not yet exclude raw `paper_runs/*.json`.
- `pyproject.toml` needs `[project.optional-dependencies].paper` for `matplotlib`, `pandas`, `pyyaml`.
- Cached `__pycache__` directories should not be tracked.

## Acceptance Criteria
- [ ] `MANIFEST.in` ships `examples/specs/*.yaml` and `examples/run_paper_experiment.py`.
- [ ] `.gitignore` excludes `paper_runs/*.json`; `*.md` and `.gitkeep` tracked.
- [ ] `pyproject.toml [project.optional-dependencies].paper` lists `matplotlib`, `pandas`, `pyyaml`.
- [ ] `pyproject.toml [project.optional-dependencies].pytest` lists `pytest-randomly`, `pytest-cov`.
- [ ] `paper_runs/.gitkeep` exists.
- [ ] Tracked `__pycache__` directories removed.

## Out of Scope
- Adding pre-commit hooks.

## Technical Checklist (atoms)
- [ ] U01 Add `paper_runs/` to `.gitignore` for `*.json` only
- [ ] U02 Update `MANIFEST.in` to ship `examples/specs/*.yaml` and `examples/run_paper_experiment.py`
- [ ] U03 Add `[project.optional-dependencies].paper` for `matplotlib`, `pandas`, `pyyaml`
- [ ] U04 Add `[project.optional-dependencies].pytest` for `pytest-randomly`, `pytest-cov`
- [ ] U05 Add `paper_runs/.gitkeep`
- [ ] U06 Remove tracked `__pycache__` directories

## Definition of Done
- `cat MANIFEST.in | grep examples/specs` matches
- `cat .gitignore | grep paper_runs` matches
- `pip install -e ".[paper]"` succeeds
- `git ls-files | grep __pycache__` returns 0
