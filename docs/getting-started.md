---
layout: page
title: "Getting started"
description: "Install xaker, run the four-line quickstart, save and load a checkpoint, validate the paper rubric."
permalink: "/getting-started/"
---
This page covers the install path that works today, the most
common first-call patterns, and the recovery commands a new
contributor needs to know. See [Installation](/xaker/installation/)
for the long-form story (PyTorch CUDA wheels, environment
isolation, troubleshooting) and the
[Tutorial](/xaker/tutorial/) for a guided walk-through of a
four-block Transformer.

**Audience: first-time xaker users who already have PyTorch installed and want a forward pass within the next ten minutes.**

**Time: five minutes for install + first forward pass; fifteen more for the train / save / load round-trip.**

## Install from source

`xaker` is published from source on GitHub. PyPI mirroring is
planned once the package name can be reserved; until then, the
working install is the editable source install:

```bash
git clone https://github.com/sachncs/xaker.git
cd xaker
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
pip install -e '.[dev]'
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA version
```

The `.` in `.[dev]` is intentional: it means "install this
package and also the dev extras." The square brackets are part of
the command, not punctuation.

## Quick start

### Command line

```bash
xaker-validate                                # paper-worthiness rubric (17/18 by default)
xaker-train --dim 256 --heads 4 --layers 4 --epochs 5 --kind fused
xaker-bench --dim 512 --heads 8 --kinds standard xsa fused linear \
            --runs 10 --output paper_runs/baseline.json
```

### Python

```python
import torch
from xaker import Config, Model

config = Config(dim=512, heads=8, kernel="exp", mode="subtract", precond="fast")
model = Model(config, num_layers=6, vocab_size=32000, max_seq_len=512, attention_type="fused")

batch = torch.randint(0, 32000, (2, 128))
logits = model(batch)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"logits shape: {logits.shape}")
```

Or build just the attention block:

```python
from xaker import BLOCK
block = BLOCK["fused"](config)
x = torch.randn(2, 128, 512)
out = block(x)
```

## Build a Transformer and train it

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

## Save and load a checkpoint

`xaker-train` writes a checkpoint when given `--out <path>`;
`xaker-eval` loads that file and runs a smoke forward pass:

```bash
xaker-train --dim 32 --heads 4 --layers 2 --epochs 1 --out artifacts/last.pt
xaker-eval  --checkpoint artifacts/last.pt --kind fused --dim 32 --heads 4
```

The CLI round-trip is a real load-and-fuse path; the file is a
plain `state_dict` produced by `torch.save`.

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
smoke test. Unknown YAML kinds raise `ValueError` at load time so
typos never reach the solver.

## Test the public surface

```bash
python3 -c "
from xaker import Fused, Xsa, Standard, Linear, Config, BLOCK
cfg = Config(dim=64, heads=4)
print('BLOCK keys:', sorted(BLOCK.keys()))
print('Fused:', BLOCK['fused'](cfg))
print('Xsa:', BLOCK['xsa'](cfg))
print('Standard:', BLOCK['standard'](cfg))
print('Linear:', BLOCK['linear'](cfg))
"
```

[install]: installation.md


## Next steps

- [Installation](/xaker/installation/) — long-form install guide with PyTorch CUDA wheels and troubleshooting.
- [Tutorial](/xaker/tutorial/) — multi-step walk-through of a four-block Transformer.
- [Recipes](/xaker/recipes/) — concrete patterns for tinkerers.
