# Epic P: CLI alignment to single-word API

## User Story
As a CLI user, I want `xaker-train`, `xaker-eval`, `xaker-bench`, `xaker-validate` to take flags that match the renamed `Config` fields, so that `--dim 256 --heads 4 --kind fused` reads naturally and a typo fails fast.

## Why this matters
- Current `--d-model`, `--num-heads`, `--attention-type` flags are multi-word; after the rename they should become `--dim`, `--heads`, `--kind`.
- `kind` is passed to `Model(config, kind=...)`, NOT to `Config` — this keeps model-architecture concerns out of attention-params.
- All four CLIs call `seed(args.seed)` exactly once at top.

## Acceptance Criteria
- [ ] `xaker-train`, `xaker-eval`, `xaker-bench`, `xaker-validate` flags use single-word tokens matching `Config` and `Fit`.
- [ ] Each CLI calls `xaker.utils.rng.seed(args.seed)` exactly once at top.
- [ ] `xaker-bench` arms are `standard`, `xsa`, `fused` only.
- [ ] `xaker-validate` accepts `--min-total` and `--repo-root`.

## Out of Scope
- Adding new CLI subcommands.

## Technical Checklist (atoms)
- [ ] P01 Update `xaker/cli/train.py` flags and references
- [ ] P02 Update `xaker/cli/eval.py` flags and references
- [ ] P03 Update `xaker/cli/bench.py` flags and references
- [ ] P04 Create `xaker/cli/validate.py`
- [ ] P05 Drop `use_fused` flag logic from `train.py`
- [ ] P06 CLI `train` passes `kind` to `Model(config, kind=...)` (not to `Config`)
- [ ] P07 CLI `eval` reads renamed `Config` fields from checkpoint
- [ ] P08 CLI `bench` arms are `["standard","xsa","fused"]` only
- [ ] P09 All four CLIs call `xaker.utils.rng.seed(args.seed)` exactly once at top

## Definition of Done
- `xaker-train --help` shows single-word flags
- `xaker-eval --help` shows single-word flags
- `xaker-bench --help` shows single-word flags
- `xaker-validate --help` shows `--min-total` and `--repo-root`
- `git grep -rn 'attention-type\|use_fused' xaker/cli/` returns 0
