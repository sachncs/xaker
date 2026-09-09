---
layout: page
title: "Limitations"
description: "The numeric guards and dtype frontier for every preconditioner, kernel, and XSA mode."
permalink: "/limitations/"
---

**Audience: anyone considering xaker for a production workload.**

**Time: 5 minutes.**

This document lists the places where xaker's behaviour deviates from
the textbook, the dtype, or the operating environment. Use it as a
checklist before reporting a bug.

## Numerical

- The PCG iterative solver assumes the regularised kernel matrix
  `(K + lam I)` is positive-definite. The package does not enforce
  this; random Gaussian kernels often fail to converge and fall
  back to `torch.linalg.solve` (a single dense solve).
- The CCCP preconditioner is O(n^3) per iteration; for sequences
  beyond ~512 tokens, prefer `precond="fast"` or `precond="diagonal"`.
- Gradient backpropagation through the PCG loop is not implemented
  as a custom `torch.autograd.Function`; gradients flow to the
  kernel and preconditioner parameters only via the direct-solve
  fallback path.

## dtype contract

The package ships a per-dtype saturation map for every numeric
guard. Use the table below to predict the behaviour of a forward
pass at a given precision; test suites that exercise these guards
live under `tests/test_numerics.py`.

| Guard | Where | fp16 frontier | bf16 frontier | Notes |
|---|---|---|---|---|
| `elu(x) + 1` saturation | `xaker/attention/linear.py` | `x < -14` | `x < -30` | `Linear` clamps inputs to `[-sat, +sat]` before `elu`; see `Linear.feature_clamp`. |
| `exp(clamp(x, -100, 100))` | `xaker/attention/func.py` | `exp(50)` overflows | `exp(80)` overflows | Pure fp16 path is therefore safe up to ~50; bf16 up to ~80. |
| `BOUND = 1e6` clamp | `xaker/solver/precond.py` | safe (max ~6.5e4) | safe | Reducing `BOUND` to `1e4` is recommended if you switch to fp16+. |
| `softplus(diag + lam)` | `xaker/solver/precond.py` | always finite | always finite | Safe across all four supported dtypes. |
| `eps = 1e-6` floor | `xaker/config.py` | invisible | invisible | One ulp above fp16 precision; bump to `1e-4` for fp16 work. |

Callers that need fp16-grade accuracy should select `dtype =
float32` on the `Ctx` they pass into the bench driver; the model
itself is dtype-agnostic and will move with `module.to(ctx.dtype)`.

## Compatibility

- The v1 attention classes (e.g. `KernelAttentionRegression`) have
  been hard-deleted; checkpoints saved against them cannot be
  loaded directly through `Model.load_state_dict`.
- The `Linear` (Katharopoulos et al.) baseline cannot represent
  positional structure; it fails on tasks that require position
  awareness (LRA copy at length=32 yields 14% accuracy vs 87-91%
  for `Standard`, `Xsa`, `Fused`). Use `Standard` / `Xsa` /
  `Fused` whenever the task needs positional recall.
- On Apple Silicon (MPS) the PyTorch linalg kernels have shape
  bugs for batched 4-D `linalg.solve`, `linalg.lu_solve`, and
  `linalg.eigh`. The benchmark suite defaults to CPU; override
  with `XAKER_DEVICE=cuda` on a CUDA host.

## Performance

- The benchmark suite runs on CPU with `dim=64, heads=4` for
  reproducibility. Real workloads benefit from larger dims and a
  CUDA-capable GPU; the relative ordering across attention
  variants holds at all sizes tested but absolute wall-clock
  numbers will differ.

## Reproducibility

- The `Trainer` does not seed PyTorch's global RNG. Call
  `xaker.utils.rng.seed(N)` before constructing both the model
  and the trainer when initialisation reproducibility is
  required; the bundled CLI does this for you.
- The fast preconditioner's payload is cached between
  `build` calls when `freq > 1` and the learnable factors are
  untouched since the last build. Mutation is detected via
  `Tensor.data_ptr()` identity; for safety, call
  `Fast.reset_cache()` after an explicit optimiser step when
  you want to guarantee a fresh build.



## Next steps
- [Troubleshooting](/xaker/troubleshooting/) — symptom-driven fixes for the rare mistakes a new user hits.
- [Mathematical foundations](/xaker/math/) — derivations that motivate every numeric guard listed here.
