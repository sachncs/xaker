# Epic R: Docs reflect post-refactor state

## User Story
As an external reader, I want every doc (`README.md`, `docs/*.md`) to describe the post-refactor tree, the single-word API, the polymorphism patterns, and the paper-worthiness rubric — so that I can use the library and judge its claims without reverse-engineering the source.

## Why this matters
- Several docs (BENCHMARK_STATUS, QUANTIFIED_IMPROVEMENTS, QUANTITATIVE_SUMMARY, IMPLEMENTATION_AUDIT, FINAL_SUMMARY) describe implementation states that no longer exist.
- `architecture.md` shows the pre-refactor tree.
- `api.md` does not exist.
- `paper_rubric.md` does not exist.
- `math.md` needs the PCG and CCCP derivations.

## Acceptance Criteria
- [ ] Obsolete docs deleted.
- [ ] `README.md` updated for xaker brand, single-word API, paper rubric.
- [ ] `docs/architecture.md` shows `xaker/` tree.
- [ ] `docs/math.md` includes PCG + CCCP derivations.
- [ ] `docs/paper_rubric.md` exists.
- [ ] `docs/api.md` exists with single-symbol documentation.
- [ ] `docs/getting-started.md` shows `pip install xaker`.
- [ ] `docs/faq.md` has a "why xaker?" entry.

## Out of Scope
- Translation to other languages.

## Technical Checklist (atoms)
- [ ] R01 Rewrite `docs/architecture.md`
- [ ] R02 Rewrite `docs/math.md` for XAKER
- [ ] R03 Update `docs/getting-started.md`
- [ ] R04 Update `docs/faq.md` (add "why xaker?" entry)
- [ ] R05 Update `docs/limitations.md`
- [ ] R06 Delete `docs/BENCHMARK_STATUS.md`
- [ ] R07 Delete `docs/QUANTIFIED_IMPROVEMENTS.md`
- [ ] R08 Delete `docs/QUANTITATIVE_SUMMARY.md`
- [ ] R09 Move `docs/benchmark_report.md` content to `paper_runs/report.md`
- [ ] R10 Add `docs/paper_rubric.md`
- [ ] R11 Add `docs/api.md`
- [ ] R12 Update `README.md` API table
- [ ] R13 Update `RESULTS.md` to point at `paper_runs/rubric_summary.md`
- [ ] R14 Delete `docs/IMPLEMENTATION_AUDIT.md`
- [ ] R15 Delete `docs/FINAL_SUMMARY.md`
- [ ] R16 Rewrite `docs/api.md` listing every public symbol

## Definition of Done
- `git ls-files | grep -E 'BENCHMARK_STATUS|QUANTIFIED_IMPROVEMENTS|QUANTITATIVE_SUMMARY|IMPLEMENTATION_AUDIT|FINAL_SUMMARY'` returns 0
- `docs/architecture.md` shows the new `xaker/` tree
- `docs/api.md` lists every public symbol
- `docs/paper_rubric.md` exists and is referenced by README
- `README.md` install line is `pip install xaker`
