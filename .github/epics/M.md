# Epic M: Typed bench driver (`xaker.bench`)

## User Story
As a researcher measuring the v2 LAKER attention, I want one typed driver that emits a schema-stable JSON containing throughput, latency, peak memory, convergence iterations, residual history, variance across seeds, and `git_sha` + environment — so that published numbers are reproducible from the JSON alone.

## Why this matters
- The current `cli/benchmark.py` and `benchmarks/long_context.py` measure the deprecated v1 path; the published numbers do not describe the v2 implementation.
- A typed driver with one `Spec`, one `Result`, one `Metrics` dataclass is easier to extend and easier to validate.
- Single-word field names match the rest of the API.

## Acceptance Criteria
- [ ] `xaker/bench/bench.py` exports `Spec`, `Result`, `Metrics`, `run`, `time`, `memory`, `converge`, `write`.
- [ ] `Spec` accepts `lengths`, `heads`, `dim`, `warmup`, `runs`, `seeds`, `kinds`, `dtype`, `device`.
- [ ] `Result` contains `git_sha`, `torch.__version__`, `cuda`, `device_name`, `cudnn_deterministic`, and `results: dict[(kind, length), Metrics]`.
- [ ] `Metrics` records `forward_ms_mean`, `forward_ms_std`, `backward_ms_mean`, `backward_ms_std`, `memory_mib`, `iters_mean`, `iters_std`.
- [ ] `run(spec)` returns `Result`; `write(result, path)` validates schema.
- [ ] `examples/specs/*.yaml` match `Spec` field names.

## Out of Scope
- Multi-process benchmarking.

## Technical Checklist (atoms)
- [ ] M01 Create `Spec` dataclass
- [ ] M02 Create `Result` dataclass
- [ ] M03 Create `Metrics` dataclass
- [ ] M04 Implement `time(module, x, *, runs, warmup, ctx)`
- [ ] M05 Implement `memory(module, x, *, ctx)`
- [ ] M06 Implement `converge(attn, x, *, ctx, iters)`
- [ ] M07 Implement `run(spec)` returning `Result`
- [ ] M08 Implement `write(result, path)` JSON with schema validation
- [ ] M09 Add `git_sha`, `torch.__version__`, `cuda`, `device_name`, `cudnn_deterministic` to JSON
- [ ] M10 Create `xaker/bench/__init__.py` re-exporting the surface
- [ ] M11 `Spec` field renames: `seq_lens` → `lengths`, `d_model` → `dim`, `num_heads` → `heads`, `num_warmup` → `warmup`, `num_runs` → `runs`, `num_layers` → `layers`, `num_trials` → `trials`, `attention_kinds` → `kinds`
- [ ] M12 `Result`/`Metrics` field renames matching `Spec`
- [ ] M13 `Spec.kind` is a `Literal["standard","xsa","fused"]` enum
- [ ] M14 `Result.results` keyed by `(kind, length) -> Metrics`
- [ ] M15 `run(spec)` returns `Result` with environment block
- [ ] M16 `write(result, path)` validates schema before write

## Definition of Done
- `python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml --check` exits 0
- `pytest tests/test_bench.py -q` green
- Two runs with same `seeds` produce bit-identical JSON
- `xaker/bench/Result` JSON validates against documented schema
