# Epic L: Intentional public API surface

## User Story
As a new external user, I want to read `xaker/__init__.py` and immediately understand what I can import: a small, ordered, intentional public surface.

## Why this matters
- The current `xaker/__init__.py` re-exports v1 classes (`KernelAttentionRegression`, `FusedXSALAKERAttention`) by accident-of-history.
- After Phase D, the canonical surface is much smaller: `Laker`, `Xsa`, `Standard`, `Kernel`, `Block`, `Model`, `Mlp`, `Trainer`, `Fit`, `Config`, `pcg`, `richardson`, `Solve`, `kernel`, `op`, `mask`, `merge`, `heads`, `broadcast`, `causal`, `padding`, `shape`, `clamp`, `finite`, `seed`, `snapshot`, `restore`, `Ctx`, `Rubric`, `Score`, `Dimension`, `grade`, plus the strategy classes.
- Order matters: kernel first, then attention, model, training, solver, bench, rubric, utils.

## Acceptance Criteria
- [ ] `xaker/__init__.py` exports only the canonical public surface.
- [ ] `__all__` is exhaustive and matches the actual exports.
- [ ] No v1 re-exports.
- [ ] `xaker/bench/__init__.py` re-exports the typed-driver surface.

## Out of Scope
- Internal helper reorganization.

## Technical Checklist (atoms)
- [ ] L01 Rewrite `xaker/__init__.py` with new public surface
- [ ] L02 Re-export `Config`, `Standard`, `Xsa`, `Laker`, `Kernel`, `Projection`, `Zero`, `Mask`, `Strategy`, `Base`, `Qkv`
- [ ] L03 Re-export `Block`, `Model`, `Mlp`, `Trainer`, `Fit`, `ce`
- [ ] L04 Re-export `pcg`, `richardson`, `Solve`, `op`, `kernel`, `mask`, `merge`, `heads`, `broadcast`, `clamp`
- [ ] L05 Re-export `causal`, `padding`, `shape`, `finite`, `seed`, `snapshot`, `restore`, `Ctx`
- [ ] L06 Re-export rubric surface: `Rubric`, `Score`, `Dimension`, `grade`
- [ ] L07 Re-export `Cache`, `PrecondProto`, `Solve`
- [ ] L08 Re-export `zerodiag`, `rms`
- [ ] L09 Re-export `Spec`, `Result`, `Metrics` from `xaker/bench/__init__.py`
- [ ] L10 Document `__all__` ordering
- [ ] L11 Re-export `Make` and preconditioner strategies
- [ ] L12 Set `__version__ = "0.4.0"`

## Definition of Done
- `python -c "import xaker; print(sorted(xaker.__all__))"` matches the documented canonical list
- `from xaker import Laker, Xsa, Standard, Kernel, Block, Model, Mlp, Trainer, Fit, Config, pcg, ce, seed, snapshot, restore` exits 0
- `git grep -rn 'KernelAttentionRegression|FusedXSALAKERAttention|LearnedPreconditioner|KernelFunction' xaker/__init__.py` returns 0
- `pytest tests/ -q` green
