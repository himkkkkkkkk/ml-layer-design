#!/usr/bin/env python3
"""
Iterative generation test: does the middle-layer meaning space determine the answer?

1. Start with initial prompt
2. Generate answer, use as next prompt
3. Repeat N times
4. Extract hidden states at each step
5. Track middle-layer similarity between consecutive iterations

If middle layers encode stable meaning, consecutive answers should stay
in the same semantic region. If they drift, meaning is unstable.

Usage:
    ./run.sh python scripts/iterate_test.py --dtype float16 -o results/iterate
"""

from __future__ import annotations

import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import nix_gpu_fix  # noqa
from src.layer_analysis import LayerAnalyzer, extract_hidden_states, tokenize_with_content

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description="Iterative generation test")
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--prompt", default="the sun rises in the east")
    p.add_argument("--iterations", "-n", type=int, default=20)
    p.add_argument("--max-new-tokens", "-t", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out-dir", "-o", default="results/iterate")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])
    N = analyzer.num_layers
    H = analyzer.hidden_size

    # ── Iterative generation ──
    texts = [args.prompt]
    states_list = []  # each: (num_layers+1, hidden)

    print(f"Starting: {args.prompt!r}\n")
    for i in range(args.iterations):
        current = texts[-1]
        # Extract hidden states for current prompt
        inputs = tokenize_with_content(analyzer.tokenizer, current)
        states = extract_hidden_states(analyzer.model, inputs, pool="last_token")
        states_list.append(states.cpu().numpy())

        # Generate continuation
        answer = analyzer.generate(current, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature)
        if not answer.strip():
            print(f"  [{i}] empty answer, stopping")
            break
        texts.append(answer.strip())
        print(f"  [{i}] → {answer[:100]}")

    print(f"\nGenerated {len(texts)-1} iterations")

    # ── Analysis ──
    num_layers = N + 1
    stacked = np.stack(states_list)  # (iterations, num_layers, hidden)
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]

    # 1. Consecutive similarity per layer
    fig, ax = plt.subplots(figsize=(15, 6))
    for li in range(num_layers):
        sims = []
        for i in range(len(states_list) - 1):
            sims.append(cosine_similarity(
                states_list[i][li].reshape(1, -1),
                states_list[i+1][li].reshape(1, -1))[0, 0])
        alpha = 1.0 if li in [0, N//3, N//2, 2*N//3, N] else 0.3
        lw = 2 if li in [0, N//3, N//2, 2*N//3, N] else 0.5
        label = xlabels[li] if li in [0, N//3, N//2, 2*N//3, N] else None
        ax.plot(range(len(sims)), sims, linewidth=lw, alpha=alpha, label=label)
    ax.set_xlabel("Iteration step"); ax.set_ylabel("Cosine similarity (consecutive)")
    ax.set_title("Consecutive-iteration similarity by layer"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(); fig.savefig(out_dir / "consecutive_sim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'consecutive_sim.png'}")

    # 2. Self-similarity matrix (all iterations vs all, middle layer)
    mid = N // 2
    X_mid = stacked[:, mid, :]
    sim_matrix = cosine_similarity(X_mid)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Iteration")
    ax.set_title(f"Self-similarity matrix — {xlabels[mid]} (middle layer)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(states_list)):
        for j in range(len(states_list)):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.tight_layout(); fig.savefig(out_dir / "self_sim_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'self_sim_matrix.png'}")

    # 3. Similarity to ORIGINAL prompt across iterations
    fig, ax = plt.subplots(figsize=(14, 6))
    for li in [0, N//3, N//2, 2*N//3, N]:
        sims = [1.0]  # iteration 0 = original
        for i in range(1, len(states_list)):
            sims.append(cosine_similarity(
                states_list[0][li].reshape(1, -1),
                states_list[i][li].reshape(1, -1))[0, 0])
        ax.plot(range(len(sims)), sims, linewidth=2, label=xlabels[li])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cosine similarity to original prompt")
    ax.set_title("Drift from original prompt across iterations")
    ax.legend(); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1)
    plt.tight_layout(); fig.savefig(out_dir / "drift_from_original.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'drift_from_original.png'}")

    # Save data
    with open(out_dir / "texts.json", "w") as f:
        json.dump({"prompt": args.prompt, "iterations": texts}, f, indent=2, ensure_ascii=False)
    torch.save(torch.tensor(stacked), out_dir / "states.pt")
    print(f"  ✓  {out_dir / 'texts.json'}")
    print(f"  ✓  {out_dir / 'states.pt'}")

    print(f"\nDone → {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
