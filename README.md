# Layer Analysis for Qwen3 Models

Extract and analyze **every transformer layer's outputs** from Qwen3-family models.

## Quick Start

```bash
./run.sh main.py -p "The capital of France is" --pool last_token --dtype float16
```

Output lands in `output/` (or `output/<name>/` with `-n`):
```
output/
├── report.json              # all stats + config + generated answer
└── vecs/
    ├── stacked.pt            # full (37, 2560) tensor — torch.load()
    ├── embedding.json        # [0.012, -0.034, ...]  — 2560 floats
    ├── layer_0.json
    └── ...
```

## Core API

```python
from layer_analysis import LayerAnalyzer

analyzer = LayerAnalyzer("Qwen/Qwen3-4B-Base")

# Get stacked tensor (one forward pass)
states = analyzer.get_hidden_states("hello", pool="last_token")
# → torch.Tensor  shape (37, 2560)  [embedding + 36 layers]

# Full run with generation
results = analyzer.run_all("hello", pool="mean_content", temperature=0.7)
save_results(results, "output/")
```

## CLI

```bash
./run.sh main.py [OPTIONS]

  -m, --model        HF model id                     [default: Qwen/Qwen3-4B-Base]
  -p, --prompt       Input text
  --pool             last_token | mean_content | mean_all
  --chat             Wrap in chat template
  --thinking / --no-thinking   Enable/disable thinking mode
  -t, --max-new-tokens  Tokens to generate           [default: 50]
  --temperature      Sampling temperature            [default: 0.7]
  --dtype            float32 | float16 | bfloat16    [default: float32]
  -o, --output-dir   Output directory                [default: output/]
  -n, --name         Subfolder name for this run
```

## Batch + Analysis

```bash
# 1. Extract all prompts (80 prompts in prompts.py)
./run.sh python batch_run.py -t 50 --dtype float16

# 2. PCA visualization
./run.sh python analysis.py pca output/batch/topic* -o pca_output

# 3. Sapir-Whorf RGB analysis
./run.sh python analysis.py sw -o pca_output
```

## Project Structure

```
mine/
├── main.py                  # CLI extraction tool
├── batch_run.py             # Batch runner
├── prompts.py               # Multi-lingual prompt collection
├── analysis.py              # PCA + Sapir-Whorf analysis
├── nix_gpu_fix.py           # NixOS GPU workaround
├── run.sh                   # Launcher (handles NixOS env)
├── layer_analysis/
│   ├── extractor.py         # Core extraction engine
│   └── serializer.py        # Save to disk (JSON + .pt)
└── README.md
```

## Pooling Modes

| `pool=` | Returns |
|---------|---------|
| `last_token` | Vector at final token position |
| `mean_content` | Mean over content span (excludes chat template) |
| `mean_all` | Mean over all tokens |

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, transformers ≥ 4.45, accelerate
- GPU: 8GB+ VRAM recommended (use `--dtype float16`)
