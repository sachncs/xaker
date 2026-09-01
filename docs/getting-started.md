# Getting Started

## Install

```bash
pip install xaker
```

## Quick start

```python
import torch
from xaker import Laker, Config

cfg = Config(dim=512, heads=8, drop=0.1)
attn = Laker(cfg)
x = torch.randn(2, 128, 512)
print(attn(x).shape)  # (2, 128, 512)
```

## CLI

```bash
xaker-train --dim 64 --heads 4 --layers 2 --epochs 5
xaker-bench --lengths 16 32 64 --runs 20 --output paper_runs/baseline.json
xaker-eval --checkpoint artifacts/last.pt
xaker-validate --min-total 14
```

## Validate the paper-worthiness rubric

```bash
python -m xaker.cli.validate --min-total 14
```

The rubric checks six dimensions (novelty, repro, correctness, efficiency,
stability, usability) and exits non-zero if the total score is below 14
or any non-novelty dimension is below 2. See `docs/paper_rubric.md`.

## Run a paper experiment

```bash
python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml
```

Five specs live in `examples/specs/`: `baseline`, `ablation`, `scaling`,
`stability`, `rubric`. JSON output goes to `paper_runs/<spec>.json`.