# Getting Started

## Install

```bash
pip install xaker
```

Or from source:

```bash
git clone https://github.com/sachncs/xaker.git
cd xaker
pip install -e '.[dev]'
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

## Build a Transformer

```python
import torch
from xaker import Config, Model, Trainer, Fit

cfg = Config(dim=64, heads=4, drop=0.1, precond="fast")
fit = Fit(epochs=2, lr=1e-3, decay=0.1)
model = Model(cfg, num_layers=2, vocab_size=100, max_seq_len=16, attention_type="fused")

x = torch.randint(0, 100, (4, 16))
y = x.clone()
trainer = Trainer(model, fit, torch.device("cpu"))
metrics = trainer.epoch([(x, y)])
print(metrics)
```

## CLI

```bash
# Train a small model on synthetic data
xaker-train --dim 64 --heads 4 --layers 2 --epochs 5

# Run the typed benchmark driver and write JSON
xaker-bench --lengths 16 32 64 --runs 20 --output paper_runs/baseline.json

# Load a checkpoint and run a smoke forward pass
xaker-eval --checkpoint artifacts/last.pt

# Run the paper-worthiness rubric
xaker-validate --min-total 14
```

## Validate the paper-worthiness rubric

```bash
python -m xaker.cli.validate --min-total 14
```

The rubric checks six dimensions (novelty, repro, correctness,
efficiency, stability, usability) and exits non-zero if the total
score is below 14 or any non-novelty dimension is below 2. See
`docs/paper_rubric.md` for the rubric description.

## Run a paper experiment

```bash
python -m examples.run_paper_experiment --spec examples/specs/baseline.yaml
```

Five specs live in `examples/specs/`: `baseline`, `ablation`,
`scaling`, `stability`, `rubric`. JSON output goes to
`paper_runs/<spec>.json`. Add `--check` to use a tiny config for a
smoke test.

## Test the public surface

```bash
python3 -c "
from xaker import Laker, Xsa, Standard, Config, BLOCK, BLOCK as b
cfg = Config(dim=64, heads=4)
print('BLOCK keys:', sorted(BLOCK.keys()))
print('Laker:', BLOCK['fused'](cfg))
print('Xsa:', BLOCK['xsa'](cfg))
print('Standard:', BLOCK['standard'](cfg))
"
```