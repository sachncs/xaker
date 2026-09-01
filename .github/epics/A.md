# Epic A: Brand alignment — rebrand xaker → xaker

## User Story
As a research engineer adopting XAKER in another codebase, I want the package to be named consistently with its scientific identity, so that the import path is short, the CLI is grep-friendly, and the citation reflects the underlying method rather than a hyphenated workaround.

## Why this matters
- The current `laker_xsa` name is a paper-arc acronym; the new `xaker` name fuses XSA + LAKER into a single searchable brand.
- Shorter import paths reduce noise in user code, papers, and CI logs.
- Aligned CLI binaries (`xaker-train`, `xaker-eval`, `xaker-bench`, `xaker-validate`) make the package feel intentional.
- The repo is currently in pre-release; this is the natural moment to rename.

## Acceptance Criteria
- [ ] Every `from xaker...` import in the repo is replaced with `from xaker...`.
- [ ] `pip install xaker` works in a fresh venv.
- [ ] `from xaker import Laker, Xsa, Config` resolves.
- [ ] `xaker-train --help` runs without errors.
- [ ] `git grep -rn 'laker_xsa|xaker|XAKER' .` returns 0 hits (excluding the `CHANGELOG.md` history entry that records the rename).
- [ ] `pyproject.toml` `name = "xaker"`.
- [ ] `[project.scripts]` lists `xaker-train`, `xaker-eval`, `xaker-bench`, `xaker-validate`.
- [ ] `[project.urls]` points at `sachn-cs/xaker`.
- [ ] `__version__` is bumped to `0.4.0`.
- [ ] `README.md` title is `# XAKER`.
- [ ] `CITATION.cff` updated with the new repo URL and name.
- [ ] `CHANGELOG.md` has a `[0.4.0] BREAKING` entry that records the rename.

## Out of Scope
- Renaming the `Laker` class (kept; brand is package, class is scientific entity).
- Internal module file structure (covered by Phase C).
- v1 legacy removal (covered by Phase B).

## Technical Checklist (atoms)
- [ ] A01 Rename top-level package directory `laker_xsa/` → `xaker/`
- [ ] A02 Update `pyproject.toml` `name = "xaker"`
- [ ] A03 Update `pyproject.toml` `[project.scripts]` to `xaker-train`, `xaker-eval`, `xaker-bench`, `xaker-validate`
- [ ] A04 Update `pyproject.toml` `[project.urls]`
- [ ] A05 Update `pyproject.toml` `[tool.setuptools.packages.find]`
- [ ] A06 Rewrite `README.md` title and headers
- [ ] A07 Replace all `from xaker.` imports repo-wide
- [ ] A08 Replace all `import xaker` repo-wide
- [ ] A09 Replace `laker_xsa-` and `xaker` strings in user-visible text
- [ ] A10 Update `CITATION.cff`
- [ ] A11 Update `CHANGELOG.md` with `[0.4.0] BREAKING` entry
- [ ] A12 Update `CONTRIBUTING.md`
- [ ] A13 Bump `__version__` to `0.4.0`
- [ ] A14 Confirm CLI binary names in `[project.scripts]`

## Definition of Done
- `git grep -rn 'laker_xsa|xaker|XAKER' .` returns 0 (excluding CHANGELOG history)
- `pip install -e .` is clean
- `python -c "from xaker import Laker, Xsa, Config"` exits 0
- `xaker-train --help` exits 0
- `pylint xaker/ --fail-under=9.5` passes (after Phase T)
