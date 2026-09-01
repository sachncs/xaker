# Epic Q: One typed experiment driver

## User Story
As a researcher running paper-grade experiments, I want one `xaker-run-paper-experiment` driver plus a small set of YAML specs, so that every published result traces back to a single config file.

## Why this matters
- The four current `examples/*.py` scripts (3,205 LOC) each re-implement their own training loop and emit ad-hoc JSON.
- A single driver with YAML specs is one place to extend.
- `--check` mode runs at PR time in < 30s; full runs produce the canonical JSON for `paper_runs/`.

## Acceptance Criteria
- [ ] `examples/run_paper_experiment.py` is the single CLI driver.
- [ ] Five YAML specs in `examples/specs/` cover baseline, ablation, scaling, stability, rubric.
- [ ] YAML field names match `Spec` dataclass (single-word).
- [ ] `paper_runs/` directory created with `.gitkeep`; JSON ignored, MD kept.
- [ ] `python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml --check` exits 0 in CI.

## Out of Scope
- Per-experiment custom drivers.

## Technical Checklist (atoms)
- [ ] Q01 Delete `examples/comparative_analysis.py`
- [ ] Q02 Delete `examples/hard_benchmark.py`
- [ ] Q03 Delete `examples/long_sequence_benchmark.py`
- [ ] Q04 Delete `examples/nlp_sentiment_benchmark.py`
- [ ] Q05 Delete `examples/run_benchmarks.py`
- [ ] Q06 Delete `examples/run_forward.py`
- [ ] Q07 Create `examples/run_paper_experiment.py`
- [ ] Q08 Create `examples/specs/baseline.yaml`
- [ ] Q09 Create `examples/specs/ablation.yaml`
- [ ] Q10 Create `examples/specs/scaling.yaml`
- [ ] Q11 Create `examples/specs/stability.yaml`
- [ ] Q12 Create `examples/specs/rubric.yaml`
- [ ] Q13 Update `examples/__init__.py`
- [ ] Q14 Update `examples/minimal_training.py` to new API
- [ ] Q15 Create `paper_runs/.gitkeep`
- [ ] Q16-Q20 YAML field names match `Spec` (single-word)

## Definition of Done
- `examples/run_paper_experiment.py --spec examples/specs/baseline.yaml --check` exits 0
- `git grep -rn 'attention-type\|d_model\|num_heads\|xsa_mode\|preconditioner_type' examples/specs/` returns 0
- `paper_runs/*.json` ignored by `.gitignore`; `*.md` and `.gitkeep` tracked
