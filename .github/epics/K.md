# Epic K: Reproducibility and explicit dependency context

## User Story
As a researcher reproducing a benchmark, I want every CLI to call `seed(s)` exactly once at the top, every tensor to flow through a typed `Ctx` (device + dtype), and every benchmark to record its `git_sha`, so that a run is bit-reproducible from the JSON output alone.

## Why this matters
- `device = "cuda" if ... else "cpu"` literals appear in 11+ places; centralizing in `Ctx` removes drift.
- `torch.manual_seed` is scattered across 8+ files; centralizing in `xaker.utils.rng.seed` makes ownership explicit.
- `TENSOR_CLIP_ABS` belongs in `xaker.utils.ops` as `BOUND`.
- `Trainer.__init__` should not call `torch.manual_seed`; it should accept an externally seeded `rng` snapshot.

## Acceptance Criteria
- [ ] `xaker/utils/ctx.py` exports `Ctx` dataclass + `to_ctx(t, ctx)`.
- [ ] `BOUND = 1e6` lives in `xaker/utils/ops.py`; `clamp(t)` uses it.
- [ ] No `device = torch.device(...)` literal in `xaker/cli/*.py` or `xaker/bench/*.py`.
- [ ] `Trainer.__init__` no longer calls `torch.manual_seed`.
- [ ] `xaker-bench`, `xaker-train`, `xaker-eval`, `xaker-validate` each call `seed(args.seed)` exactly once at top.
- [ ] `xaker-validate --repo-root . --min-total 14` exits 0.

## Out of Scope
- Reproducibility beyond the package's reach (cuDNN autotuner, hardware).

## Technical Checklist (atoms)
- [ ] K01 Add `xaker/utils/ctx.py` with `Ctx` dataclass
- [ ] K02 Replace `device = ...` literals with `Ctx(device=...)`
- [ ] K03 Remove `torch.manual_seed` from `Trainer.__init__`; accept `rng` arg
- [ ] K04 Update `examples/*.py` to call `xaker.utils.rng.seed` once at top
- [ ] K05 Add `to_ctx(t, ctx)` helper in `xaker/utils/ctx.py`
- [ ] K06 Add `BOUND = 1e6` constant in `xaker/utils/ops.py`
- [ ] K07 Use `clamp(t)` (with `BOUND`) in `Laker.attend`
- [ ] K08 Add `Ctx` dataclass: `device`, `dtype = torch.float32`
- [ ] K09 Add `to_ctx(t, ctx)` helper
- [ ] K10 Replace all `device = "cuda" if ... else "cpu"` literals

## Definition of Done
- `git grep -rn 'device = torch.device\|torch.cuda.is_available() and "cuda"' xaker/cli/ xaker/bench/ examples/` returns 0
- `git grep -rn 'torch.manual_seed' xaker/ examples/` returns only `xaker/utils/rng.py`
- `pytest tests/ -q --randomly-seed=1234` deterministic
- `xaker-validate --repo-root . --min-total 14` exits 0
