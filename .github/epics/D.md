# Epic D: Single-word class renames

## User Story
As a reader of the API, I want every public class to have a single-word name (`Laker`, `Xsa`, `Block`, `Model`), so that `from xaker import Laker` reads as fluently as `from torch import nn` and I can refer to a class in conversation using one syllable.

## Why this matters
- Multi-word PascalCase classes (`XSALAKERTransformer`, `StandardMultiHeadAttention`) are verbose; their role is clear but their name isn't.
- The compiler/type checker doesn't care, but a paper reviewer, a stack-overflow answer, and a code review all suffer from verbose names.
- Single-word classes also let single-word strategy names (`Projection`, `Zero`, `Mask`, `Identity`, `Cccp`, `Fast`, `Diagonal`) fit alongside.

## Acceptance Criteria
- [ ] Every public class in `xaker/` has a single-word name.
- [ ] No multi-word PascalCase class name survives in the codebase.
- [ ] Class references in tests, examples, docs are updated.
- [ ] The single-word `Laker` class is kept (it is the LAKER paper's contribution and the brand's namesake).
- [ ] `LakerAttentionLayer` is deleted (no-op wrapper, never used).
- [ ] `LakerPreconditioner` is split into four strategy classes (`Identity`, `Diagonal`, `Fast`, `Cccp`) and the factory function `Make(config)`.

## Out of Scope
- v1 classes (already deleted in Phase B).
- File renames (Phase C).
- Strategy pattern implementation (Phase J).

## Technical Checklist (atoms)
- [ ] D01 `XSA_LAKER_Config` → `Config`
- [ ] D02 `BaseMultiHeadAttention` → `Base`
- [ ] D03 `QKVProjection` → `Qkv`
- [ ] D04 `StandardMultiHeadAttention` → `Standard`
- [ ] D05 `ExclusiveSelfAttention` → `Xsa`
- [ ] D06 `AttentionKernel` → `Kernel`
- [ ] D07 `LakerAttention` → `Laker`
- [ ] D08 DELETE `LakerAttentionLayer`
- [ ] D09 `XSAStrategy` → `XsaMode` Protocol
- [ ] D10 `XSAProjectionRemoval` → `Projection`
- [ ] D11 `XSAZeroDiagonal` → `Zero`
- [ ] D12 `XSALAKERTransformerBlock` → `Block`
- [ ] D13 `XSALAKERTransformer` → `Model`
- [ ] D14 `MLP` → `Mlp`
- [ ] D15 DELETE `LakerPreconditioner` (split into 4 strategy classes + `Make(config)`)
- [ ] D16 `TrainingConfig` → `Fit`
- [ ] D17 DELETE `LearnedPreconditioner` (v1, already removed by B07)
- [ ] D18 DELETE `KernelFunction` (v1)
- [ ] D19 DELETE `KernelAttentionRegression` (v1)
- [ ] D20 DELETE `FusedXSALAKERAttention` (v1)
- [ ] D21 `Trainer` (already single-word, unchanged)

## Definition of Done
- `git grep -rn 'XSA_LAKER_Config|StandardMultiHeadAttention|ExclusiveSelfAttention|LakerAttention\b|AttentionKernel|BaseMultiHeadAttention|QKVProjection|XSAStrategy|XSAProjectionRemoval|XSAZeroDiagonal|model\b.*Block|XSALAKERTransformer|MLP|LakerPreconditioner|TrainingConfig|LearnedPreconditioner|KernelFunction|KernelAttentionRegression|FusedXSALAKERAttention|LakerAttentionLayer' xaker/ tests/ examples/` returns 0
- `pytest tests/ -q` green
- `pylint xaker/ --fail-under=9.5` passes
