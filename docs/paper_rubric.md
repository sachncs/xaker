# Paper-Worthiness Rubric

XAKER is graded against six dimensions. The CI `xaker-validate` command
enforces `total >= 14` and `no dim < 2` (except `novelty` may be 1).

## Dimensions

### Novelty (0–3)
Distinct method beyond baseline.
- 1: minor variation of a known method
- 2: combination of known methods with measurable synergy
- 3: a demonstrable new mechanism or formulation

### Repro (0–3)
Reproducibility from code alone.
- 1: manual / ad hoc
- 2: reproducible on a single machine
- 3: reproducible across machines; deterministic flags set

### Correctness (0–3)
Invariant tests cover math.
- 1: smoke tests
- 2: unit + property tests
- 3: invariant tests cover key math (PCG convergence, kernel symmetry, etc.)

### Efficiency (0–3)
Benchmarked vs baselines.
- 1: anecdotal
- 2: consistent improvement
- 3: published numbers with CI

### Stability (0–3)
Stable across seeds, dtypes, sequence lengths.
- 1: smoke-tested
- 2: stable on representative workloads
- 3: stable across stress conditions

### Usability (0–3)
CLI / API / docs.
- 1: hard to use
- 2: external engineer can use
- 3: publishable API + docs

## Pass threshold

- `total >= 14` (out of 18)
- every dim (except novelty) >= 2
- novelty may be 1 if efficiency + repro are both 3

## Current grading

Run `xaker-validate --repo-root . --min-total 14`. Last run: 17/18, PASS.