#!/usr/bin/env python3
"""
Extract hidden states from every transformer layer.

Usage:
    uv run python scripts/extract.py -p "hello" --pool last_token --dtype float16
    uv run python scripts/extract.py -p "hello" -n my_run -t 512
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nix_gpu_fix  # noqa: E402 — must be before torch on NixOS

import argparse
import torch
from src.layer_analysis import LayerAnalyzer, save_results


def main():
    p = argparse.ArgumentParser(description="Qwen3 layer-by-layer output analysis")
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--prompt", "-p", default="The meaning of life is")
    p.add_argument("--chat", action="store_true", help="Wrap in chat template")
    p.add_argument("--thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", action="store_false", dest="thinking")
    p.add_argument("--pool", default="mean_content",
                   choices=["last_token", "mean_content", "mean_all"])
    p.add_argument("--output-dir", "-o", default="output")
    p.add_argument("--name", "-n", default=None, help="Subfolder under output-dir")
    p.add_argument("--max-new-tokens", "-t", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--no-extract", action="store_true", help="Skip extraction, only generate")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])

    results = analyzer.run_all(
        text=args.prompt, pool=args.pool, chat=args.chat,
        thinking=args.thinking, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, extract=not args.no_extract,
    )

    out_dir = Path(args.output_dir)
    if args.name:
        out_dir = out_dir / args.name

    save_results(results, output_dir=str(out_dir))


if __name__ == "__main__":
    main()
