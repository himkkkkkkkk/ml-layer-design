#!/usr/bin/env python3
"""
Pragmatic mode test: does illocutionary force matter?

Tests whether STATEMENT vs QUESTION vs ORDER/COMMAND
(the "action" implied by the utterance) affects the interlingua.

8 facts × 3 modes × 4 variants = 96 prompts (English only)
Extract hidden states, run PCA colored by TOPIC vs MODE.

Usage:
    ./run.sh python scripts/prag_test.py --dtype float16 -o results/prag_test
"""

from __future__ import annotations

import sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nix_gpu_fix  # noqa: E402
from src.layer_analysis import LayerAnalyzer

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets.prag_prompts import PRAG_PROMPTS, TOPIC_NAMES, MODE_NAMES, flatten

TOPIC_COLORS = plt.cm.tab10(np.linspace(0, 1, len(TOPIC_NAMES)))
MODE_COLORS = {"statement": "#2980b9", "question": "#e74c3c", "order": "#2ecc71"}
MODE_MARKERS = {"statement": "o", "question": "^", "order": "s"}


def main():
    p = argparse.ArgumentParser(description="Pragmatic mode test")
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out-dir", "-o", default="results/prag_test")
    p.add_argument("--skip-extract", action="store_true")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = flatten()
    print(f"Prompts: {len(prompts)}  ({len(TOPIC_NAMES)} topics × {len(MODE_NAMES)} modes × 4 variants)")

    # ── Extract ──
    if not args.skip_extract:
        analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])
        N, H = analyzer.num_layers, analyzer.hidden_size
        vecs_dir = out_dir / "vecs"; vecs_dir.mkdir(exist_ok=True)
        for i, pr in enumerate(prompts):
            path = vecs_dir / f"{pr['name']}.pt"
            if not path.exists():
                states = analyzer.get_hidden_states(pr["text"], pool="last_token")
                torch.save(states.cpu(), path)
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{len(prompts)}")
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"model_id": analyzer.model_id, "num_layers": N, "hidden_size": H}, f)
        print(f"  Done: {len(prompts)} vectors")
    else:
        vecs_dir = out_dir / "vecs"
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
        N, H = meta["num_layers"], meta["hidden_size"]

    # ── Load ──
    tensors = {}
    for pr in prompts:
        tensors[pr["name"]] = torch.load(vecs_dir / f"{pr['name']}.pt", weights_only=True).float().numpy()
    num_layers = N + 1
    topic_arr = np.array([pr["topic"] for pr in prompts])
    mode_arr = np.array([MODE_NAMES.index(pr["mode"]) for pr in prompts])
    mode_labels = np.array([pr["mode"] for pr in prompts])
    stacked = np.stack([tensors[pr["name"]] for pr in prompts])
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]
    print(f"  Shape: {stacked.shape}")

    # ══════════════════════════════════════════════════════════════════
    # 1. ALL layers: TOPIC-colored and MODE-colored
    # ══════════════════════════════════════════════════════════════════
    cols, rows = 7, (num_layers + 6) // 7

    for color_by, label_arr, color_names, cmap, word, fn in [
        ("topic", topic_arr, TOPIC_NAMES, TOPIC_COLORS, "Topic", "pca_topic_all"),
        ("mode", mode_arr, MODE_NAMES,
         np.array([MODE_COLORS[m] for m in MODE_NAMES]), "Mode", "pca_mode_all"),
    ]:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        for idx in range(num_layers):
            ax = axes[idx]
            X = stacked[:, idx, :]; Xc = X - X.mean(axis=0, keepdims=True)
            X2 = PCA(2).fit_transform(Xc)
            for c in range(len(color_names)):
                m = label_arr == c
                if m.any():
                    marker = MODE_MARKERS.get(color_names[c], "o") if word == "Mode" else "o"
                    ax.scatter(X2[m, 0], X2[m, 1], c=[cmap[c]], marker=marker, s=15, alpha=0.8)
            label = "emb" if idx == 0 else f"L{idx-1}"
            ax.set_title(label, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for idx in range(num_layers, len(axes)): axes[idx].set_visible(False)
        handles = [Patch(color=cmap[c] if isinstance(cmap, np.ndarray) else cmap[color_names[c]],
                         label=color_names[c]) for c in range(len(color_names))]
        fig.legend(handles=handles, loc="lower center", ncol=len(color_names), fontsize=7, frameon=False)
        fig.suptitle(f"PCA by {word} — 8 facts × 3 modes × 4 variants", fontsize=14, y=1.01)
        plt.tight_layout(); fig.savefig(out_dir / f"{fn}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✓  {out_dir / f'{fn}.png'}")

    # ── Combined: color=TOPIC, shape=MODE ──
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten()
    for idx in range(num_layers):
        ax = axes[idx]
        X = stacked[:, idx, :]; Xc = X - X.mean(axis=0, keepdims=True)
        X2 = PCA(2).fit_transform(Xc)
        for mi, mode in enumerate(MODE_NAMES):
            for ti in range(len(TOPIC_NAMES)):
                m = (mode_arr == mi) & (topic_arr == ti)
                if m.any():
                    ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[ti]],
                              marker=MODE_MARKERS[mode], s=18, alpha=0.85,
                              edgecolors="white", linewidths=0.2)
        label = "emb" if idx == 0 else f"L{idx-1}"
        ax.set_title(label, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for idx in range(num_layers, len(axes)): axes[idx].set_visible(False)
    from matplotlib.lines import Line2D
    leg1 = [Patch(color=TOPIC_COLORS[t], label=TOPIC_NAMES[t]) for t in range(len(TOPIC_NAMES))]
    leg2 = [Line2D([0],[0], color="gray", marker=MODE_MARKERS[m], linestyle="",
                   markersize=8, label=m) for m in MODE_NAMES]
    l1 = fig.legend(handles=leg1, loc="lower center", ncol=4, fontsize=7, frameon=False,
                    title="Topic (color)", title_fontsize=8)
    fig.add_artist(l1)
    l2 = fig.legend(handles=leg2, loc="lower center", ncol=3, fontsize=7, frameon=False,
                    title="Mode (shape)", title_fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("PCA: Color=Topic  Shape=Mode — 8 facts × 3 modes × 4 variants", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_combined_all.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_combined_all.png'}")

    # ══════════════════════════════════════════════════════════════════
    # 2. RGB-style: same-topic-diff-mode vs same-mode-diff-topic
    # ══════════════════════════════════════════════════════════════════
    red_mean, green_mean, blue_mean = [], [], []
    for layer in range(num_layers):
        vecs = {}
        for i, pr in enumerate(prompts):
            vecs[(pr["topic"], pr["mode"])] = stacked[i, layer, :]

        # RED: same topic, different mode
        red = []
        for t in range(len(TOPIC_NAMES)):
            for i, m1 in enumerate(MODE_NAMES):
                for m2 in MODE_NAMES[i+1:]:
                    red.append(cosine_similarity(
                        vecs[(t, m1)].reshape(1, -1), vecs[(t, m2)].reshape(1, -1))[0, 0])

        # GREEN: same mode, different topic
        green = []
        for m in MODE_NAMES:
            for i, t1 in enumerate(range(len(TOPIC_NAMES))):
                for t2 in range(i+1, len(TOPIC_NAMES)):
                    green.append(cosine_similarity(
                        vecs[(t1, m)].reshape(1, -1), vecs[(t2, m)].reshape(1, -1))[0, 0])

        # BLUE: different topic, different mode
        blue = []
        for t1 in range(len(TOPIC_NAMES)):
            for t2 in range(len(TOPIC_NAMES)):
                if t1 == t2: continue
                for m1 in MODE_NAMES:
                    for m2 in MODE_NAMES:
                        if m1 == m2: continue
                        blue.append(cosine_similarity(
                            vecs[(t1, m1)].reshape(1, -1), vecs[(t2, m2)].reshape(1, -1))[0, 0])

        red_mean.append(np.mean(red))
        green_mean.append(np.mean(green))
        blue_mean.append(np.mean(blue))

    fig, ax = plt.subplots(figsize=(15, 6))
    xs = range(num_layers)
    ax.plot(xs, red_mean, color="#e74c3c", linewidth=2.5, label="Same topic, different mode (RED)")
    ax.plot(xs, green_mean, color="#27ae60", linewidth=2.5, label="Same mode, different topic (GREEN)")
    ax.plot(xs, blue_mean, color="#3498db", linewidth=2.5, label="Different topic, different mode (BLUE)")
    for i in range(num_layers):
        if red_mean[i] > green_mean[i]: ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="#e74c3c")

    red_win = [red_mean[i] > green_mean[i] for i in range(num_layers)]
    phases, prev, start = [], red_win[0], 0
    for i in range(1, num_layers):
        if red_win[i] != prev: phases.append((start, i-1, prev)); start = i; prev = red_win[i]
    phases.append((start, num_layers-1, prev))
    for s, e, is_red in phases:
        if e - s >= 2:
            label = "TOPIC WINS\n(semantics > pragmatics)" if is_red else "MODE WINS\n(pragmatics > semantics)"
            y = max(red_mean[s:e+1] + green_mean[s:e+1]) + 0.04
            ax.annotate(label, ((s+e)/2, y), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                        ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xticks(range(0, num_layers, 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_xlabel("Layer"); ax.set_ylim(0, 1)
    ax.set_title("Pragmatics vs Semantics: Does statement/question/order matter?", fontsize=14)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
    n_red = sum(red_win)
    ax.text(0.02, 0.97, f"RED > GREEN in {n_red}/{num_layers} layers ({n_red/num_layers:.0%})\n"
            f"8 facts × 3 modes (statement/question/order) × 4 variants",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    plt.tight_layout(); fig.savefig(out_dir / "prag_rgb_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'prag_rgb_curves.png'}")

    # ══════════════════════════════════════════════════════════════════
    # 3. Within-mode RGB: for each mode, same-topic vs diff-topic
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, mode in zip(axes, MODE_NAMES):
        mode_vecs = {}
        for i, pr in enumerate(prompts):
            if pr["mode"] == mode:
                mode_vecs.setdefault(pr["topic"], []).append(stacked[i])

        same_topic, diff_topic = [], []
        for layer in range(num_layers):
            st, dt = [], []
            for t in range(len(TOPIC_NAMES)):
                vt = [v[layer] for v in mode_vecs.get(t, [])]
                for i, v1 in enumerate(vt):
                    for v2 in vt[i+1:]:
                        st.append(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0,0])
            for t1 in range(len(TOPIC_NAMES)):
                for t2 in range(t1+1, len(TOPIC_NAMES)):
                    for v1 in mode_vecs.get(t1, []):
                        for v2 in mode_vecs.get(t2, []):
                            dt.append(cosine_similarity(v1[layer].reshape(1,-1), v2[layer].reshape(1,-1))[0,0])
            same_topic.append(np.mean(st) if st else 0)
            diff_topic.append(np.mean(dt) if dt else 0)

        ax.plot(xs, same_topic, color="#e74c3c", linewidth=2, label="Same topic")
        ax.plot(xs, diff_topic, color="#3498db", linewidth=2, label="Different topic")
        for i in range(num_layers):
            if same_topic[i] > diff_topic[i]: ax.axvspan(i-0.5, i+0.5, alpha=0.06, color="#e74c3c")
        ax.set_title(f"{mode.upper()}", fontsize=12)
        ax.set_xticks(range(0, num_layers, 6))
        ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 6)], fontsize=7)
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.suptitle("Within-Mode Topic Clustering: Statement vs Question vs Order", fontsize=14, y=1.02)
    plt.tight_layout(); fig.savefig(out_dir / "prag_per_mode.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'prag_per_mode.png'}")

    print(f"\n{'='*60}")
    print(f"  Done  →  {out_dir.resolve()}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
