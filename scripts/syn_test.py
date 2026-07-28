#!/usr/bin/env python3
"""
Syntactic variation test: does grammar affect the interlingua?

Tests whether different syntactic forms of the SAME fact cluster together
(topic dominates) or separate by structure (syntax dominates).

Method:
  - 8 facts × 6 syntactic forms = 48 prompts (English only)
  - Extract hidden states, run PCA colored by TOPIC
  - Also compute: same-topic-diff-struct vs same-struct-diff-topic similarity

Usage:
    ./run.sh python scripts/syn_test.py --dtype float16 -o results/syn_test
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets.syn_prompts import SYN_PROMPTS, TOPIC_NAMES, STRUCT_NAMES, flatten

TOPIC_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
STRUCT_COLORS = plt.cm.Set2(np.linspace(0, 1, len(STRUCT_NAMES)))
STRUCT_MARKERS = ['o', '^', 's', 'D', 'v', 'P']  # distinct shape per structure


def main():
    p = argparse.ArgumentParser(description="Syntactic variation test")
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out-dir", "-o", default="results/syn_test")
    p.add_argument("--skip-extract", action="store_true")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = flatten()
    print(f"Prompts: {len(prompts)}  ({len(TOPIC_NAMES)} topics × {len(STRUCT_NAMES)} structures)")

    # ── Extract ──
    if not args.skip_extract:
        analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])
        N = analyzer.num_layers
        H = analyzer.hidden_size
        vecs_dir = out_dir / "vecs"
        vecs_dir.mkdir(exist_ok=True)

        for i, pr in enumerate(prompts):
            path = vecs_dir / f"{pr['name']}.pt"
            if path.exists():
                continue
            states = analyzer.get_hidden_states(pr["text"], pool="last_token")
            torch.save(states.cpu(), path)
            if (i + 1) % 12 == 0:
                print(f"  {i+1}/{len(prompts)}")

        meta = {"model_id": analyzer.model_id, "num_layers": N, "hidden_size": H}
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        print(f"  Done: {len(prompts)} vectors")
    else:
        vecs_dir = out_dir / "vecs"
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
        N = meta["num_layers"]
        H = meta["hidden_size"]

    # ── Load tensors ──
    tensors = {}
    for pr in prompts:
        path = vecs_dir / f"{pr['name']}.pt"
        tensors[pr["name"]] = torch.load(path, weights_only=True).float().numpy()

    num_layers = N + 1
    topic_arr = np.array([pr["topic"] for pr in prompts])
    struct_arr = np.array([STRUCT_NAMES.index(pr["struct"]) for pr in prompts])
    stacked = np.stack([tensors[pr["name"]] for pr in prompts])
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]

    print(f"  Shape: {stacked.shape}  (prompts × layers × hidden)")

    # ══════════════════════════════════════════════════════════════════
    # 1. ALL layers PCA colored by TOPIC
    # ══════════════════════════════════════════════════════════════════
    cols = 7; rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for idx in range(num_layers):
        ax = axes[idx]
        X = stacked[:, idx, :]; Xc = X - X.mean(axis=0, keepdims=True)
        X2 = PCA(2).fit_transform(Xc)
        for t in range(len(TOPIC_NAMES)):
            m = topic_arr == t
            if m.any():
                ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], s=15, alpha=0.8)
        label = "emb" if idx == 0 else f"L{idx-1}"
        ax.set_title(label, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for idx in range(num_layers, len(axes)): axes[idx].set_visible(False)
    handles = [Patch(color=TOPIC_COLORS[t], label=TOPIC_NAMES[t]) for t in range(len(TOPIC_NAMES))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("PCA by TOPIC — 8 facts × 6 syntactic forms (English only)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_topic_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_topic_all.png'}")

    # 2. ALL layers PCA: color=TOPIC, shape=STRUCTURE (combined)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten()
    for idx in range(num_layers):
        ax = axes[idx]
        X = stacked[:, idx, :]; Xc = X - X.mean(axis=0, keepdims=True)
        X2 = PCA(2).fit_transform(Xc)
        for s in range(len(STRUCT_NAMES)):
            for t in range(len(TOPIC_NAMES)):
                m = (struct_arr == s) & (topic_arr == t)
                if m.any():
                    ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], marker=STRUCT_MARKERS[s],
                              s=18, alpha=0.85, edgecolors="white", linewidths=0.2)
        label = "emb" if idx == 0 else f"L{idx-1}"
        ax.set_title(label, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    for idx in range(num_layers, len(axes)): axes[idx].set_visible(False)
    # two legends
    from matplotlib.lines import Line2D
    leg1 = [Patch(color=TOPIC_COLORS[t], label=TOPIC_NAMES[t]) for t in range(len(TOPIC_NAMES))]
    leg2 = [Line2D([0],[0], color="gray", marker=STRUCT_MARKERS[s], linestyle="",
                   markersize=8, label=STRUCT_NAMES[s]) for s in range(len(STRUCT_NAMES))]
    l1 = fig.legend(handles=leg1, loc="lower center", ncol=4, fontsize=7, frameon=False,
                    title="Topic (color)", title_fontsize=8)
    fig.add_artist(l1)
    l2 = fig.legend(handles=leg2, loc="lower center", ncol=6, fontsize=7, frameon=False,
                    title="Structure (shape)", title_fontsize=8,
                    bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("PCA: Color=Topic  Shape=Structure — 8 facts × 6 forms", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_combined_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_combined_all.png'}")

    # ══════════════════════════════════════════════════════════════════
    # 3. RGB-style: same-topic-diff-struct vs same-struct-diff-topic
    # ══════════════════════════════════════════════════════════════════
    red_mean, green_mean, blue_mean = [], [], []
    for layer in range(num_layers):
        vecs = {}
        for i, pr in enumerate(prompts):
            vecs[(pr["topic"], pr["struct"])] = stacked[i, layer, :]

        # RED: same topic, different structure
        red = []
        for t in range(len(TOPIC_NAMES)):
            for i, s1 in enumerate(STRUCT_NAMES):
                for s2 in STRUCT_NAMES[i+1:]:
                    red.append(cosine_similarity(
                        vecs[(t, s1)].reshape(1, -1), vecs[(t, s2)].reshape(1, -1))[0, 0])

        # GREEN: same structure, different topic
        green = []
        for s in STRUCT_NAMES:
            for i, t1 in enumerate(range(len(TOPIC_NAMES))):
                for t2 in range(i+1, len(TOPIC_NAMES)):
                    green.append(cosine_similarity(
                        vecs[(t1, s)].reshape(1, -1), vecs[(t2, s)].reshape(1, -1))[0, 0])

        # BLUE: different topic, different structure
        blue = []
        for t1 in range(len(TOPIC_NAMES)):
            for t2 in range(len(TOPIC_NAMES)):
                if t1 == t2: continue
                for s1 in STRUCT_NAMES:
                    for s2 in STRUCT_NAMES:
                        if s1 == s2: continue
                        blue.append(cosine_similarity(
                            vecs[(t1, s1)].reshape(1, -1), vecs[(t2, s2)].reshape(1, -1))[0, 0])

        red_mean.append(np.mean(red))
        green_mean.append(np.mean(green))
        blue_mean.append(np.mean(blue))

    fig, ax = plt.subplots(figsize=(15, 6))
    xs = range(num_layers)
    ax.plot(xs, red_mean, color="#e74c3c", linewidth=2.5, label="Same topic, different structure (RED)")
    ax.plot(xs, green_mean, color="#27ae60", linewidth=2.5, label="Same structure, different topic (GREEN)")
    ax.plot(xs, blue_mean, color="#3498db", linewidth=2.5, label="Different topic, different structure (BLUE)")
    for i in range(num_layers):
        if red_mean[i] > green_mean[i]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.07, color="#e74c3c")

    red_win = [red_mean[i] > green_mean[i] for i in range(num_layers)]
    phases, prev, start = [], red_win[0], 0
    for i in range(1, num_layers):
        if red_win[i] != prev: phases.append((start, i-1, prev)); start = i; prev = red_win[i]
    phases.append((start, num_layers-1, prev))
    for s, e, is_red in phases:
        if e - s >= 2:
            label = "TOPIC WINS\n(semantics > syntax)" if is_red else "SYNTAX WINS\n(structure > meaning)"
            y = max(red_mean[s:e+1] + green_mean[s:e+1]) + 0.04
            ax.annotate(label, ((s+e)/2, y), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                        ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xticks(range(0, num_layers, 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_xlabel("Layer"); ax.set_ylim(0, 1)
    ax.set_title("Syntax vs Semantics: Does grammar affect the interlingua?", fontsize=14)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
    n_red = sum(red_win)
    ax.text(0.02, 0.97, f"RED > GREEN in {n_red}/{num_layers} layers ({n_red/num_layers:.0%})\n"
            f"8 facts × 6 syntactic forms (English only)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    plt.tight_layout()
    fig.savefig(out_dir / "syn_rgb_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_dir / 'syn_rgb_curves.png'}")

    # ══════════════════════════════════════════════════════════════════
    # 4. Key layers: TOPIC vs STRUCTURE side by side
    # ══════════════════════════════════════════════════════════════════
    keys = {"emb": 0, "L01": 1, "L06": 6, "L12": 12, "L15": 15, "L18": 18, "L24": 24, f"L{N-1}": N}
    for color_by, color_arr, color_names, cmap, markers, title_word, fn in [
        ("topic", topic_arr, TOPIC_NAMES, TOPIC_COLORS, None, "Topic", "key_topic"),
        ("struct", struct_arr, STRUCT_NAMES, STRUCT_COLORS, STRUCT_MARKERS, "Structure", "key_struct"),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
        for ax, (title, li) in zip(axes, keys.items()):
            X = stacked[:, li, :]; Xc = X - X.mean(axis=0, keepdims=True)
            X2 = PCA(2).fit_transform(Xc)
            for c in range(len(color_names)):
                m = color_arr == c
                if m.any():
                    kw = {"marker": markers[c]} if markers else {}
                    ax.scatter(X2[m, 0], X2[m, 1], c=[cmap[c]], s=35, alpha=0.85,
                              edgecolors="white", linewidths=0.3, **kw)
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        for i in range(len(keys), len(axes)): axes[i].set_visible(False)
        if markers:
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=cmap[c], marker=markers[c], linestyle="",
                              markersize=8, label=color_names[c]) for c in range(len(color_names))]
        else:
            handles = [Patch(color=cmap[c], label=color_names[c]) for c in range(len(color_names))]
        fig.legend(handles=handles, loc="lower center", ncol=len(color_names), fontsize=7, frameon=False)
        fig.suptitle(f"PCA by {title_word} — Key Layers", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"pca_{fn}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {out_dir / f'pca_{fn}.png'}")

    elapsed = time.time()
    print(f"\n{'='*60}")
    print(f"  Done  →  {out_dir.resolve()}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
