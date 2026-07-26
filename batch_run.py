#!/usr/bin/env python3
"""
Batch run: extract layer outputs for all prompts in prompts.py.

Usage:
    python batch_run.py -t 50           # run all with generation
    python batch_run.py --dry-run        # list what would run
    python batch_run.py -t 0             # extraction only (fast)
"""

from __future__ import annotations

import nix_gpu_fix  # noqa: E402,F401

import argparse, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import flatten_prompts

OUT_DIR = Path("output/batch")
VENV_PYTHON = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")
MAIN = str(Path(__file__).resolve().parent / "main.py")

ENV = {**os.environ,
       "LD_LIBRARY_PATH": "/run/opengl-driver/lib:"
       "/nix/store/hngmi01i8wgi25a0byrxcn4ysz5j79mw-gcc-15.2.0-lib/lib:"
       "/nix/store/dbz6pb9g67kpgpl95k8d85kzpxm1c32p-zlib-1.3.2/lib"}


def main():
    p = argparse.ArgumentParser(description="Batch layer extraction")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-new-tokens", "-t", type=int, default=50)
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    prompts = flatten_prompts()
    to_run = []
    skipped = 0
    for entry in prompts:
        run_dir = OUT_DIR / entry["name"]
        if (run_dir / "vecs" / "stacked.pt").exists():
            skipped += 1; continue
        to_run.append(entry)

    print(f"Total: {len(prompts)}  Already done: {skipped}  To run: {len(to_run)}")

    if args.dry_run:
        for entry in to_run:
            print(f"  [{entry['name']:22s}] {entry['lang']} {entry['type']} | {entry['text'][:60]}")
        return

    for i, entry in enumerate(to_run):
        run_dir = OUT_DIR / entry["name"]
        print(f"\n[{i+1}/{len(to_run)}] {entry['name']}  — {entry['text'][:60]}")
        cmd = [VENV_PYTHON, MAIN, "-p", entry["text"], "--pool", "last_token",
               "-o", str(OUT_DIR), "-n", entry["name"], "-t", str(args.max_new_tokens),
               "--dtype", args.dtype]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=ENV)
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr[-200:]}")
        else:
            print(f"  OK → {run_dir}")

    print(f"\nDone. {len(to_run)} runs.")


if __name__ == "__main__":
    main()
