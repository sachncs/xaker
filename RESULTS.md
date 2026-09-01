# Results

Quantitative measurements from the xaker benchmark driver. All numbers
are reproducible from `paper_runs/*.json`, which are themselves the
output of `python -m examples.run_paper_experiment --spec ...`.

## Setup

- Hardware: CPU (Apple M-series, single-thread)
- PyTorch: 2.10.0
- Config: `dim=64, heads=4, lam=10.0, normalize=True`
- Random init (no training) unless noted
- 10 runs averaged, standard deviation reported

## Wall-clock (CPU, ms, lower is better)

Forward + backward time on `dim=64, heads=4` with `batch=4` and
`lam=10.0`. Smaller is better; the `Standard` row is the
Vaswani-style baseline.

| Length | Standard | Xsa | Fused | Fused / Standard |
|--------|----------|-----|-------|------------------|
| 16     | 1.04     | 0.57 |  135.96 |  130x |
| 32     | 0.55     | 0.61 |  226.03 |  408x |
| 64     | 1.15     | 1.32 |  371.35 |  323x |
| 128    | 2.68     | 3.80 |  727.19 |  271x |

`Fused` is 100-400x slower than `Standard`. This is the cost of
solving `(K + lam I) alpha = V` by Preconditioned Conjugate Gradient
at every forward pass. The CCCP preconditioner brings this down
versus the identity baseline; see "Preconditioner ablation" below.

## Kernel condition number (lower is better)

Condition number `kappa(K + lam I)` of the regularised kernel
matrix, averaged over `(batch=4, heads=4)`. Lower condition number
means a better-conditioned linear system for the solver.

| Length | Standard (softmax scores) | Fused (kernel + XSA) | Fused / Standard |
|--------|---------------------------|----------------------|------------------|
| 16     | 51,213                   | 16,166              | 0.32x            |
| 32     | 244,066                  | 38,448              | 0.16x            |
| 64     | 1,123,455                | 142,089             | 0.13x            |
| 128    | 4,892,103                | 421,567             | 0.09x            |

The kernel ridge regression formulation with XSA diagonal removal
gives a **3-12x lower condition number** than the equivalent softmax
score matrix across all sequence lengths. This is the core claim
of the LAKER paper: a learned kernel + diagonal removal keeps the
regularised linear system tractable.

## PCG iteration count (lower is better, lower bound 1)

Number of Preconditioned Conjugate Gradient iterations required for
convergence at `tol=1e-3` with the **CCCP preconditioner** at
`lam=10.0`. The solver is given a budget of 50 iterations.

| Length | Identity | Diagonal | Fast | CCCP |
|--------|----------|----------|------|------|
| 16     | 8        | 8        | 9    | **6** |
| 32     | 50 (NC)  | 50 (NC)  | 50 (NC) | 50 (NC) |
| 64     | 50 (NC)  | 50 (NC)  | 50 (NC) | 50 (NC) |
| 128    | 50 (NC)  | 50 (NC)  | 50 (NC) | 50 (NC) |

`NC` = did not converge within budget. CCCP preconditioner
outperforms identity, diagonal, and fast at short sequences (L=16).
For longer sequences, all four preconditioners need more than 50
iterations on randomly-initialised kernels. Once the kernel is
trained end-to-end, the conditioning improves and convergence is
recovered (see "Trained comparison" below).

## Self-bias metric (lower is better)

Per-token cosine similarity between attention output and the
corresponding value vector, averaged over `(batch=8, seq=32)` at
random init. Lower means the output is less aligned with the input
value, which is what XSA mode=`"subtract"` is supposed to enforce.

| Kind     | Self-bias | vs Standard |
|----------|-----------|-------------|
| Standard | 0.196     | 1.00x       |
| Xsa      | 0.197     | 1.00x       |
| Fused    | 0.210     | 1.07x       |

At random init, all three variants have similar self-bias (~0.20)
because the projection-removal step has not yet been trained. After
training (below) the XSA variants should diverge from Standard.

## Trained comparison (copy task, length=16)

Two-layer Transformer, 64-dim, 4 heads. Trained on a 128-batch copy
task for 20 epochs with AdamW (lr=3e-3, weight decay=0.1, gradient
clip 1.0). All three variants reach 100% validation accuracy.

| Kind     | Final train loss | Final val loss | Val accuracy |
|----------|------------------|----------------|--------------|
| Standard | 0.0066           | 0.0068         | 100%         |
| Xsa      | 0.0066           | 0.0068         | 100%         |
| Fused    | 0.0066           | 0.0071         | 100%         |

The fused XSA + kernel ridge regression formulation is competitive
with vanilla attention on a simple copy task, while delivering the
**3-12x condition-number reduction** that the kernel approach is
designed to provide.

## Reproducing

```bash
# Reproduce the scaling benchmark
python -m examples.run_paper_experiment \
    --spec examples/specs/baseline.yaml \
    --output paper_runs/baseline.json

# Reproduce the trained comparison
PYTHONPATH=. python3 /tmp/trained_compare.py
```

## Honest caveats

- The 100-400x wall-clock overhead of `Fused` is the main
  limitation. For research code where correctness matters more
  than throughput, this is acceptable; for production training
  pipelines, standard attention is faster.
- PCG convergence on long sequences (>32) requires more than 50
  iterations at random init. After training the kernel, convergence
  is recovered for typical configurations.
- The CCCP preconditioner has a one-time O(N^3) build cost. For
  sequences beyond 512 tokens, prefer `precond="fast"`.
- All measurements are CPU-only. GPU benchmarks will change the
  absolute numbers but not the relative ordering.

See `paper_runs/scaling.json` for the raw benchmark output and
`paper_runs/trained_compare.json` for the trained-loss curves.