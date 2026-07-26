"""
Serialize layer analysis results to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_results(
    results: dict[str, Any],
    output_dir: str | Path = "output",
    *,
    save_tensors: bool = True,
    save_json: bool = True,
    indent: int = 2,
) -> Path:
    """Save analysis results to disk.

    Files produced:
        report.json          — all stats (norms, means, config)
        vecs/stacked.pt      — (num_layers+1, hidden_dim) tensor
        vecs/{name}.json     — per-layer float arrays
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON report (exclude heavy tensor data)
    if save_json:
        report = _strip_tensors(results)
        json_path = out / "report.json"
        json_path.write_text(json.dumps(report, indent=indent, default=str))
        print(f"  ✓  report  → {json_path}")

    # hidden states
    hs = results.get("hidden_states", {})
    stacked = hs.get("_stacked") if isinstance(hs, dict) else None
    if stacked is not None:
        vecs_dir = out / "vecs"
        vecs_dir.mkdir(parents=True, exist_ok=True)
        num_layers = stacked.shape[0] - 1
        labels = ["embedding"] + [f"layer_{i}" for i in range(num_layers)]

        if save_tensors:
            torch.save(stacked, vecs_dir / "stacked.pt")
            print(f"  ✓  stacked  → {vecs_dir / 'stacked.pt'}  shape={tuple(stacked.shape)}")

        for i, label in enumerate(labels):
            (vecs_dir / f"{label}.json").write_text(json.dumps(stacked[i].tolist()))
        print(f"  ✓  vecs     → {vecs_dir}/  ({len(labels)} JSON files)")

    print(f"\nAll results saved to: {out.resolve()}")
    return out


def _strip_tensors(obj: Any) -> Any:
    """Recursively remove keys starting with '_' from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_tensors(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_tensors(v) for v in obj]
    return obj
