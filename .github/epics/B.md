# Epic B: Hard-delete v1 legacy attention/preconditioner

## User Story
As a maintainer of the `xaker` package, I want every deprecated v1 module and shim file removed from the codebase, so that no external reader is misled into using outdated classes and the published benchmark numbers finally describe the v2 (LAKER) implementation that the README advertises.

## Why this matters
- The current benchmarks measure `FusedXSALAKERAttention` (v1), not `LakerAttention` (v2). The published numbers cannot defend the v2 claim.
- Every `_legacy.py` and `*_attention.py` shim is a back-compat tax that forces new readers to choose between two coexisting APIs.
- v1 uses Richardson iteration; v2 uses PCG. Different math, different convergence. There is no gradual migration path; the v1 path is gone.
- Hard delete now prevents drift between v1 and v2 in tests, docs, and CI.

## Acceptance Criteria
- [ ] Files `xaker/attention/_legacy.py`, `xaker/attention/attention_kernel.py`, `xaker/attention/standard_attention.py`, `xaker/attention/xsa_attention.py`, `xaker/attention/kernel_attention.py`, `xaker/attention/fused_attention_v2.py`, `xaker/solver/preconditioner.py` no longer exist.
- [ ] No import in `xaker/`, `tests/`, `examples/`, or `docs/` references `KernelAttentionRegression`, `FusedXSALAKERAttention`, `LearnedPreconditioner`, `KernelFunction`, `LakerAttentionLayer`, or any re-export from the v1 modules.
- [ ] `xaker/solver/__init__.py` no longer re-exports `LearnedPreconditioner`.
- [ ] `xaker/__init__.py` and `xaker/attention/__init__.py` no longer re-export any v1 symbol.
- [ ] All `pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")` lines in tests are removed (no deprecation warnings left to suppress).
- [ ] Tests that imported v1 classes are rewritten to use v2 equivalents or deleted.
- [ ] `pylint xaker/ --fail-under=9.5` still passes after deletion.
- [ ] `pytest tests/ -q` green.

## Out of Scope
- Adding a deprecation path for users with v1 checkpoints (none exist).
- Renaming remaining v2 modules (Phase C).

## Technical Checklist (atoms)
- [ ] B01 Delete `xaker/attention/_legacy.py`
- [ ] B02 Delete `xaker/attention/attention_kernel.py`
- [ ] B03 Delete `xaker/attention/standard_attention.py`
- [ ] B04 Delete `xaker/attention/xsa_attention.py`
- [ ] B05 Delete `xaker/attention/kernel_attention.py`
- [ ] B06 Delete `xaker/attention/fused_attention_v2.py`
- [ ] B07 Delete `xaker/solver/preconditioner.py`
- [ ] B08 Remove `LearnedPreconditioner` re-export from `xaker/solver/__init__.py`
- [ ] B09 Remove `_legacy` re-exports from `xaker/__init__.py` and `xaker/attention/__init__.py`
- [ ] B10 Remove `pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")` from all tests
- [ ] B11 Delete tests for v1 classes; rewrite to v2

## Definition of Done
- `git ls-files | grep -E '_legacy|attention_kernel|standard_attention|xsa_attention|kernel_attention|fused_attention_v2'` returns 0
- `git grep -rn 'KernelAttentionRegression|FusedXSALAKERAttention|LearnedPreconditioner|KernelFunction|LakerAttentionLayer' .` returns 0
- `pytest tests/ -q` green
- `pylint xaker/ --fail-under=9.5` passes
