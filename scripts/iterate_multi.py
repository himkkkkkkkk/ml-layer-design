#!/usr/bin/env python3
"""
Multi-topic iterative generation: cluster by topic across iterations.

8 topics, 20 iterations each. If the late-layer gist is stable,
same-topic iterations should cluster together regardless of elaboration.

Usage:
    ./run.sh python scripts/iterate_multi.py --dtype float16 -o results/iterate_multi
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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 8 topic statements
TOPICS = [
    "the sun rises in the east and sets in the west",
    "water boils at 100 degrees celsius",
    "the earth revolves around the sun",
    "plants need sunlight to grow",
    "humans need oxygen to survive",
    "ice melts when heated",
    "most birds can fly in the sky",
    "light travels faster than sound",
]
TOPIC_NAMES = ["Sun", "Water", "Earth", "Plants", "Humans", "Ice", "Birds", "Light"]
TOPIC_COLORS = plt.cm.tab10(np.linspace(0, 1, len(TOPICS)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--iterations", "-n", type=int, default=20)
    p.add_argument("--max-new-tokens", "-t", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out-dir", "-o", default="results/iterate_multi")
    p.add_argument("--skip-extract", action="store_true")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        stacked = torch.load(out_dir / "states.pt", weights_only=True).float().numpy()
        with open(out_dir / "texts.json") as f:
            all_texts = json.load(f)
    else:
        analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])
        N = analyzer.num_layers
        H = analyzer.hidden_size

        all_states = []  # list of (iterations, num_layers, hidden) per topic
        all_texts = []   # list of [prompt, a0, a1, ...] per topic

        for t, prompt in enumerate(TOPICS):
            print(f"\n── Topic {t}: {TOPIC_NAMES[t]} ──")
            texts = [prompt]
            states_list = []

            for i in range(args.iterations):
                current = texts[-1]
                inputs = tokenize_with_content(analyzer.tokenizer, current)
                states = extract_hidden_states(analyzer.model, inputs, pool="last_token")
                states_list.append(states.cpu().numpy())

                answer = analyzer.generate(current, max_new_tokens=args.max_new_tokens,
                                           temperature=args.temperature)
                if not answer.strip(): break
                texts.append(answer.strip())
                print(f"  [{i}] → {answer[:80]}")

            all_states.append(np.stack(states_list))
            all_texts.append(texts)

        # Pad to same iteration count
        min_iter = min(s.shape[0] for s in all_states)
        all_states = [s[:min_iter] for s in all_states]
        stacked = np.stack(all_states)  # (8 topics, iterations, num_layers, hidden)
        print(f"\nShape: {stacked.shape}  (topics × iterations × layers × hidden)")

        torch.save(torch.tensor(stacked), out_dir / "states.pt")
        with open(out_dir / "texts.json", "w") as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)

    T, I, L, H = stacked.shape
    topic_arr = np.repeat(np.arange(T), I)  # [0,0,...,1,1,...]
    iter_arr = np.tile(np.arange(I), T)      # [0,1,2,...,0,1,2,...]
    X_full = stacked.reshape(T * I, L, H)    # (T*I, L, H)
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, L)]

    # ══════════════════════════════════════════════════════════════════
    # ALL layers: PCA colored by topic
    # ══════════════════════════════════════════════════════════════════
    cols, rows = 7, (L + 6) // 7
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()
    for li in range(L):
        ax = axes[li]
        X = X_full[:, li, :]; Xc = X - X.mean(axis=0, keepdims=True)
        X2 = PCA(2).fit_transform(Xc)
        for t in range(T):
            m = topic_arr == t
            if m.any():
                ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], s=12, alpha=0.8)
        label = "emb" if li == 0 else f"L{li-1}"
        ax.set_title(label, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for li in range(L, len(axes)): axes[li].set_visible(False)
    handles = [Patch(color=TOPIC_COLORS[t], label=TOPIC_NAMES[t]) for t in range(T)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("PCA by Topic — 8 topics × 20 iterations each", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_topic_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_topic_all.png'}")

    # ══════════════════════════════════════════════════════════════════
    # Within-topic stability: consecutive similarity per layer (avg over topics)
    # ══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(15, 6))
    for li in range(L):
        all_cons = []
        for t in range(T):
            Xt = stacked[t, :, li, :]  # (I, H)
            for i in range(I - 1):
                all_cons.append(cosine_similarity(
                    Xt[i].reshape(1,-1), Xt[i+1].reshape(1,-1))[0,0])
        alpha = 1.0 if li in [0, L//3, L//2, 2*L//3, L-1] else 0.3
        lw = 2 if li in [0, L//3, L//2, 2*L//3, L-1] else 0.5
        label = xlabels[li] if li in [0, L//3, L//2, 2*L//3, L-1] else None
        ax.plot(range(I-1), [np.mean([all_cons[j] for t in range(T)]) for j in range(I-1)],
                linewidth=lw, alpha=alpha, label=label)
    ax.set_xlabel("Iteration step"); ax.set_ylabel("Cosine similarity (consecutive)")
    ax.set_title("Consecutive-iteration similarity by layer (avg over 8 topics)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.tight_layout(); fig.savefig(out_dir / "consecutive_sim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'consecutive_sim.png'}")

    # ══════════════════════════════════════════════════════════════════
    # Within-topic vs cross-topic similarity across layers
    # ══════════════════════════════════════════════════════════════════
    within_topic, cross_topic = [], []
    for li in range(L):
        wt, ct = [], []
        for t in range(T):
            Xt = stacked[t, :, li, :]  # (I, H)
            # all pairs within same topic
            wt.append(cosine_similarity(Xt).mean())
        for t1 in range(T):
            for t2 in range(t1+1, T):
                ct.append(cosine_similarity(
                    stacked[t1, :, li, :], stacked[t2, :, li, :]).mean())
        within_topic.append(np.mean(wt))
        cross_topic.append(np.mean(ct))

    fig, ax = plt.subplots(figsize=(14, 6))
    xs = range(L)
    ax.plot(xs, within_topic, color="#e74c3c", linewidth=2.5, label="Within same topic (should be high)")
    ax.plot(xs, cross_topic, color="#3498db", linewidth=2.5, label="Cross different topics (should be low)")
    for i in range(L):
        if within_topic[i] > cross_topic[i]: ax.axvspan(i-0.5, i+0.5, alpha=0.06, color="#e74c3c")
    ax.set_xticks(range(0, L, 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, L, 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_xlabel("Layer"); ax.set_ylim(0, 1)
    ax.set_title("Within-topic vs Cross-topic similarity across iterations", fontsize=14)
    ax.legend(fontsize=11, loc="lower right"); ax.grid(True, alpha=0.25)
    plt.tight_layout(); fig.savefig(out_dir / "within_vs_cross.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'within_vs_cross.png'}")

    print(f"\nDone → {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
