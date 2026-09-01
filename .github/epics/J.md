# Epic J: Replace if/elif mode chains with polymorphism

## User Story
As a future contributor adding a new attention variant, preconditioner strategy, or XSA mode, I want behavior selection to happen through named dispatch tables and concrete strategy classes, so that my addition is a one-line edit and I never have to touch existing if/elif branches.

## Why this matters
- The current `LakerPreconditioner.compute_preconditioner` and `.apply_preconditioner` have four `if mode == ...` branches each. Each is a missed polymorphism.
- `XSALAKERTransformerBlock.__init__` has five `if attention_type == ...` branches. A `BLOCK` registry makes this one line.
- `XsaStrategy` already has three strategies (`Projection`, `Zero`, `Mask`); the if/elif remains only at the call site. Move it to the factory.
- Adding a new variant should be one class + one entry in a dict.

## Acceptance Criteria
- [ ] `xaker/solver/precond.py` exports `Make(config)` factory + four concrete strategy classes (`Identity`, `Diagonal`, `Fast`, `Cccp`) + `Cache` dataclass + `PrecondProto` Protocol. No `if self.mode == ...` chains.
- [ ] `xaker/attention/xsa.py` exports `XsaStrategy(config, scale)` factory + `Projection`, `Zero`, `Mask` concrete classes + `XSA_MODE` dispatch. No `if config.mode == ...` chains inside `Xsa.__init__` or `Xsa.attend`.
- [ ] `xaker/attention/__init__.py` exports `BLOCK = {"standard": Standard, "xsa": Xsa, "fused": Laker}` registry.
- [ ] `xaker/model/block.py` `Block.__init__(config, attention)` takes attention via DI; no if/elif on `attention_type`.
- [ ] `xaker/model/model.py` `Model.__init__` uses `BLOCK[config.kind](config)` to construct the attention; `kind` is a `Model` kwarg, not on `Config`.
- [ ] `xaker/attention/laker.py` `Laker.attend` calls `self.xsa.apply(...)`; no `if config.mode == ...` chains.
- [ ] `pcg` returns a `Solve` dataclass with `iters`, `converged`, `res`, `history`; `Laker.attend` checks `solve.converged` and falls back only on `not converged and not finite`.
- [ ] `git grep -rn 'if.*mode ==|elif.*mode ==|if.*attention_type ==' xaker/` returns 0.
- [ ] `git grep -rn 'as PrecondModule|noqa.*F401'` returns 0.

## Out of Scope
- Adding new strategy variants.

## Technical Checklist (atoms)
- [ ] J01 Refactor `xaker/solver/precond.py`: split into `Identity`/`Diagonal`/`Fast`/`Cccp` concrete `nn.Module` strategies; add `Make(config)` factory, `Cache` dataclass, `PrecondProto` Protocol
- [ ] J02 Refactor `xaker/attention/xsa.py`: split into `Projection`/`Zero`/`Mask` concrete classes; add `XsaStrategy(config, scale)` factory, `XSA_MODE` dispatch
- [ ] J03 Refactor `xaker/attention/laker.py`: use `Make(config)` and `XsaStrategy(config, self.xsa_scale)`; remove `clean_self_projection`/`zero_diagonal`/`rms_normalize` methods; remove if/elif mode chains
- [ ] J04 Add `BLOCK` registry in `xaker/attention/__init__.py` mapping `"standard"/"xsa"/"fused"` to classes
- [ ] J05 Refactor `xaker/model/block.py`: `Block(config, attention)` DI
- [ ] J06 Refactor `xaker/model/model.py`: use `BLOCK[config.kind](config)` and pass to each `Block`
- [ ] J07 Refactor `xaker/solver/cg.py`: `pcg` returns `Solve` dataclass (`x`, `iters`, `converged`, `res`, `history`); drop `apply_kernel_operator` re-export shim
- [ ] J08 Refactor `xaker/attention/laker.py`: check `solve.converged`; fall back only on `not converged and not finite`
- [ ] J09 Add `xaker/attention/ops.py` with free functions `zerodiag(kernel)` and `rms(x, eps)`

## Definition of Done
- `git grep -rn 'if.*mode ==|elif.*mode ==|if.*attention_type ==' xaker/` returns 0
- `git grep -rn 'as PrecondModule|noqa.*F401'` returns 0
- `pytest tests/ -q` green
- `pylint xaker/ --fail-under=9.5` passes
- `mypy xaker/ --no-implicit-optional --warn-unused-ignores` 0 errors
