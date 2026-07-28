#!/usr/bin/env python3
"""
Sapir-Whorf Reproduction: Does language matter in middle layers?

Reproduces the analysis from:
  "LLM Neuroanatomy III: Do LLMs Break the Sapir-Whorf Hypothesis?"
  https://dnhkng.github.io/posts/sapir-whorf/

Method:
  For each layer, compute cosine similarity between hidden states:
    RED:   same topic, different language  (e.g. EN-"sun rises" vs ZH-"太阳升起")
    GREEN: same language, different topic  (e.g. EN-"sun rises" vs EN-"water boils")
    BLUE:  different topic, different language (baseline)

  If RED > GREEN in middle layers → language is erased, topic dominates
  → the model has a language-agnostic "interlingua" semantic space.

Usage:
    ./run.sh python scripts/run.py --dtype float16 -o results/sapir_whorf
"""

from __future__ import annotations

import sys, json, time, argparse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nix_gpu_fix  # noqa: E402
from src.layer_analysis import LayerAnalyzer, extract_hidden_states, tokenize_with_content
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets.prompts import PROMPTS

# ── config ───────────────────────────────────────────────────────────
LANGS = list(PROMPTS[0].keys())
TOPICS = list(range(len(PROMPTS)))
TOPIC_NAMES = ["Sun rises", "Water 100°C", "Earth orbits", "Plants light",
               "Humans O₂", "Ice melts", "Birds fly", "Light>sound"]
LANG_COLORS = {"EN": "#e74c3c", "ZH": "#3498db", "DE": "#2ecc71", "FR": "#f39c12", "JA": "#9b59b6"}
TYPE_COLORS = {"S": "#2980b9", "Q": "#e74c3c"}

# ── helpers ──────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Extract hidden states for all prompts
# ══════════════════════════════════════════════════════════════════════

def extract_all(analyzer: LayerAnalyzer, out_dir: Path) -> dict:
    """Extract hidden states for all prompts, return {name: (num_layers+1, hidden) tensor}."""
    _sep("Phase 1: Extraction")
    tensors = {}
    vecs_dir = out_dir / "vecs"
    vecs_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for t in TOPICS:
        for lang in PROMPTS[t]:
            for typ in ["S", "Q"]:
                name = f"topic{t}_{lang}_{typ}"
                text = PROMPTS[t][lang][0 if typ == "S" else 1]
                configs.append({"topic": t, "lang": lang, "type": typ, "name": name, "text": text})

    print(f"Extracting {len(configs)} prompts ...")
    for i, cfg in enumerate(configs):
        path = vecs_dir / f"{cfg['name']}.pt"
        if path.exists():
            tensors[cfg["name"]] = torch.load(path, weights_only=True)
            continue
        states = analyzer.get_hidden_states(cfg["text"], pool="last_token")
        torch.save(states.cpu(), path)
        tensors[cfg["name"]] = states.cpu()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(configs)}")

    # Save metadata
    meta = {
        "model_id": analyzer.model_id,
        "num_layers": analyzer.num_layers,
        "hidden_size": analyzer.hidden_size,
        "num_prompts": len(configs),
        "languages": LANGS,
        "num_topics": len(TOPICS),
        "configs": [{k: v for k, v in cfg.items() if k != "text"} for cfg in configs],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done: {len(configs)} vectors saved to {vecs_dir}/")
    return tensors


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Sapir-Whorf RGB analysis
# ══════════════════════════════════════════════════════════════════════

def run_sw_analysis(tensors: dict, meta: dict, out_dir: Path):
    _sep("Phase 2: Sapir-Whorf RGB analysis")
    num_layers = meta["num_layers"] + 1
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]

    def load(t, l, typ):
        return tensors[f"topic{t}_{l}_{typ}"].float().numpy()

    def compute_rgb(typ):
        rm, gm, bm, rs, gs, bs = [], [], [], [], [], []
        for layer in range(num_layers):
            vecs = {(t, l): load(t, l, typ)[layer] for t in TOPICS for l in LANGS}
            red, green, blue = [], [], []
            for t in TOPICS:
                for i, l1 in enumerate(LANGS):
                    for l2 in LANGS[i+1:]:
                        red.append(cosine_similarity(vecs[(t,l1)].reshape(1,-1), vecs[(t,l2)].reshape(1,-1))[0,0])
            for l in LANGS:
                for i, t1 in enumerate(TOPICS):
                    for t2 in TOPICS[i+1:]:
                        green.append(cosine_similarity(vecs[(t1,l)].reshape(1,-1), vecs[(t2,l)].reshape(1,-1))[0,0])
            for t1 in TOPICS:
                for t2 in TOPICS:
                    if t1 == t2: continue
                    for l1 in LANGS:
                        for l2 in LANGS:
                            if l1 == l2: continue
                            blue.append(cosine_similarity(vecs[(t1,l1)].reshape(1,-1), vecs[(t2,l2)].reshape(1,-1))[0,0])
            rm.append(np.mean(red)); rs.append(np.std(red))
            gm.append(np.mean(green)); gs.append(np.std(green))
            bm.append(np.mean(blue)); bs.append(np.std(blue))
        return {"rm": rm, "rs": rs, "gm": gm, "gs": gs, "bm": bm, "bs": bs}

    def plot(rgb, suffix, fn):
        xs = range(num_layers)
        fig, ax = plt.subplots(figsize=(15, 6))
        for vals, stds, color, label in [
            (rgb["rm"], rgb["rs"], "#e74c3c", "Same topic, diff lang (RED)"),
            (rgb["gm"], rgb["gs"], "#27ae60", "Same lang, diff topic (GREEN)"),
            (rgb["bm"], rgb["bs"], "#3498db", "Diff topic, diff lang (BLUE)"),
        ]:
            ax.plot(xs, vals, color=color, linewidth=2.2, label=label)
            ax.fill_between(xs, np.array(vals)-np.array(stds), np.array(vals)+np.array(stds), alpha=0.12, color=color)
        for i in range(num_layers):
            if rgb["rm"][i] > rgb["gm"][i]: ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="#e74c3c")
        # phase labels
        red_win = [rgb["rm"][i] > rgb["gm"][i] for i in range(num_layers)]
        phases, prev, start = [], red_win[0], 0
        for i in range(1, num_layers):
            if red_win[i] != prev: phases.append((start, i-1, prev)); start = i; prev = red_win[i]
        phases.append((start, num_layers-1, prev))
        for s, e, is_red in phases:
            if e - s >= 2:
                label = "TOPIC SPACE" if is_red else "LANGUAGE LAYER"
                y = max(rgb["rm"][s:e+1] + rgb["gm"][s:e+1]) + 0.04
                ax.annotate(label, ((s+e)/2, y), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                            ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        ax.set_xticks(range(0, num_layers, 3))
        ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
        ax.set_ylabel("Cosine similarity"); ax.set_ylim(0, 1)
        ax.set_title(f"Cosine similarity — {meta['model_id']} {suffix}", fontsize=14)
        ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
        n_red = sum(red_win)
        ax.text(0.02, 0.97, f"RED > GREEN in {n_red}/{num_layers} layers ({n_red/num_layers:.0%})",
                transform=ax.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
        plt.tight_layout(); fig.savefig(out_dir / fn, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✓  {out_dir / fn}")

    rgb_s = compute_rgb("S")
    rgb_q = compute_rgb("Q")
    plot(rgb_s, "(Statements)", "sw_main_statements.png")
    plot(rgb_q, "(Questions)", "sw_main_questions.png")

    # centered
    for rgb, suffix, fn in [(rgb_s, "Statements", "sw_centered_statements.png")]:
        xs = range(num_layers)
        rc = np.array(rgb["rm"]) - np.array(rgb["bm"])
        gc = np.array(rgb["gm"]) - np.array(rgb["bm"])
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(xs, rc, color="#e74c3c", linewidth=2.2, label="RED centered (topic signal)")
        ax.fill_between(xs, rc-rgb["rs"], rc+rgb["rs"], alpha=0.12, color="#e74c3c")
        ax.plot(xs, gc, color="#27ae60", linewidth=2.2, label="GREEN centered (language signal)")
        ax.fill_between(xs, gc-rgb["gs"], gc+rgb["gs"], alpha=0.12, color="#27ae60")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for i in range(num_layers):
            if rc[i] > gc[i]: ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="#e74c3c")
        red_win = [rc[i] > gc[i] for i in range(num_layers)]
        phases, prev, start = [], red_win[0], 0
        for i in range(1, num_layers):
            if red_win[i] != prev: phases.append((start, i-1, prev)); start = i; prev = red_win[i]
        phases.append((start, num_layers-1, prev))
        for s, e, is_red in phases:
            if e - s >= 2:
                label = "SEMANTIC SPACE" if is_red else "LANGUAGE LAYER"
                y = max(rc[s:e+1].max(), gc[s:e+1].max()) + 0.02
                ax.annotate(label, ((s+e)/2, y), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                            ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        ax.set_xticks(range(0, num_layers, 3))
        ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
        ax.set_ylabel("Centered cosine similarity"); ax.set_title(f"Centered similarity {suffix}", fontsize=14)
        ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
        plt.tight_layout(); fig.savefig(out_dir / fn, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✓  {out_dir / fn}")

    # Per-language-pair
    pair_colors = {("EN","ZH"):"#e74c3c", ("EN","DE"):"#3498db", ("EN","FR"):"#2ecc71", ("EN","JA"):"#9b59b6",
                   ("ZH","DE"):"#f39c12", ("ZH","FR"):"#1abc9c", ("ZH","JA"):"#e67e22",
                   ("DE","FR"):"#2980b9", ("DE","JA"):"#8e44ad", ("FR","JA"):"#16a085"}
    fig, ax = plt.subplots(figsize=(15, 6))
    for (l1, l2), c in pair_colors.items():
        scores = []
        for layer in range(num_layers):
            sims = [cosine_similarity(load(t,l1.lower(),"S")[layer].reshape(1,-1), load(t,l2.lower(),"S")[layer].reshape(1,-1))[0,0] for t in TOPICS]
            scores.append(np.mean(sims))
        ax.plot(range(num_layers), scores, color=c, linewidth=1.8, alpha=0.8, label=f"{l1}↔{l2}")
    ax.set_xticks(range(0, num_layers, 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_title("Cross-language convergence by language pair", fontsize=14)
    ax.legend(fontsize=8, ncol=5, loc="lower right"); ax.grid(True, alpha=0.25)
    plt.tight_layout(); fig.savefig(out_dir / "sw_per_lang_pair.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'sw_per_lang_pair.png'}")

    # Topic heatmap
    matrix = np.zeros((len(TOPICS), num_layers))
    for t in TOPICS:
        for layer in range(num_layers):
            sims = [cosine_similarity(load(t,l1,"S")[layer].reshape(1,-1), load(t,l2,"S")[layer].reshape(1,-1))[0,0]
                    for i, l1 in enumerate(LANGS) for l2 in LANGS[i+1:]]
            matrix[t, layer] = np.mean(sims)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(0, num_layers, 2))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 2)], fontsize=7, rotation=45)
    ax.set_yticks(range(len(TOPICS))); ax.set_yticklabels(TOPIC_NAMES, fontsize=9)
    ax.set_title("Cross-language semantic convergence by topic", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean cross-language cosine sim")
    plt.tight_layout(); fig.savefig(out_dir / "sw_topic_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'sw_topic_heatmap.png'}")

    # Same-type vs diff-type curves
    _plot_topic_type_curves(load, num_layers, out_dir)

    # Save SW summary
    sw_summary = {
        "red_win_layers": int(sum(1 for i in range(num_layers) if rgb_s["rm"][i] > rgb_s["gm"][i])),
        "total_layers": num_layers,
        "peak_red_green_gap": float(max(np.array(rgb_s["rm"]) - np.array(rgb_s["gm"]))),
        "peak_layer": int(np.argmax(np.array(rgb_s["rm"]) - np.array(rgb_s["gm"]))),
    }
    with open(out_dir / "sw_summary.json", "w") as f:
        json.dump(sw_summary, f, indent=2)


def _plot_topic_type_curves(load, num_layers, out_dir):
    """Same-topic: same-type (S↔S, Q↔Q) vs diff-type (S↔Q) cross-language."""
    all_blue = np.zeros(num_layers)
    all_red = np.zeros(num_layers)
    for t in TOPICS:
        bc, rc = [], []
        for layer in range(num_layers):
            bv, rv = [], []
            for typ in ["S", "Q"]:
                for i, l1 in enumerate(LANGS):
                    for l2 in LANGS[i+1:]:
                        bv.append(cosine_similarity(load(t,l1,typ)[layer].reshape(1,-1), load(t,l2,typ)[layer].reshape(1,-1))[0,0])
            for l1 in LANGS:
                for l2 in LANGS:
                    if l1 == l2: continue
                    rv.append(cosine_similarity(load(t,l1,"S")[layer].reshape(1,-1), load(t,l2,"Q")[layer].reshape(1,-1))[0,0])
            for l in LANGS:
                rv.append(cosine_similarity(load(t,l,"S")[layer].reshape(1,-1), load(t,l,"Q")[layer].reshape(1,-1))[0,0])
            bc.append(np.mean(bv)); rc.append(np.mean(rv))
        all_blue += np.array(bc); all_red += np.array(rc)
    all_blue /= len(TOPICS); all_red /= len(TOPICS)

    xs = range(num_layers)
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(xs, all_blue, color="#3498db", linewidth=2.5, label="Same type (S↔S + Q↔Q cross-lang)")
    ax.plot(xs, all_red, color="#e74c3c", linewidth=2.5, label="Diff type (S↔Q cross-lang)")
    ab, ar = np.array(all_blue), np.array(all_red)
    ax.fill_between(xs, ab, ar, where=(ab>ar), alpha=0.1, color="#3498db")
    ax.fill_between(xs, ab, ar, where=(ab<=ar), alpha=0.1, color="#e74c3c")
    bw = [ab[i] > ar[i] for i in range(len(xs))]
    phases, prev, start = [], bw[0], 0
    for i in range(1, len(xs)):
        if bw[i] != prev: phases.append((start, i-1, prev)); start = i; prev = bw[i]
    phases.append((start, len(xs)-1, prev))
    for s, e, b in phases:
        if e - s >= 2:
            ax.annotate("S ≠ Q (type matters)" if not b else "S ≈ Q (type erased)",
                        ((s+e)/2, max(ab[s:e+1].max(), ar[s:e+1].max()) + 0.03),
                        fontsize=9, color="#e74c3c" if not b else "#3498db", ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xticks(range(0, len(xs), 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, len(xs), 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_xlabel("Layer")
    ax.set_title("Same-Topic Cross-Language: Same-Type vs Different-Type (S/Q)", fontsize=14)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1)
    plt.tight_layout(); fig.savefig(out_dir / "sw_topic_type_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'sw_topic_type_curve.png'}")


# ══════════════════════════════════════════════════════════════════════
# Phase 3: PCA / LDA / MDS
# ══════════════════════════════════════════════════════════════════════

def run_pca_analysis(tensors: dict, meta: dict, out_dir: Path):
    _sep("Phase 3: PCA / LDA / MDS")
    num_layers = meta["num_layers"] + 1

    # Build stacked array: (num_prompts, num_layers, hidden)
    configs = meta.get("configs", [])
    if not configs:
        # reconstruct from tensors
        for name in tensors:
            parts = name.split("_")
            if len(parts) >= 3:
                configs.append({"name": name, "topic": int(parts[0][5:]), "lang": parts[1], "type": parts[2]})

    stacked = np.stack([tensors[c["name"]].float().numpy() for c in configs])  # (P, L, H)
    P, L, H = stacked.shape
    meta_list = [{"lang": c["lang"].upper(), "type": c["type"].upper(), "prompt": c["name"]} for c in configs]
    print(f"  Shape: {stacked.shape}  (prompts × layers × hidden)")

    keys = {"emb": 0, "L01": 1, "L06": 6, "L12": 12, "L18": 18, "L24": 24, "L30": 30, f"L{num_layers-2}": num_layers-1}

    # Centered PCA by language
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
    for ax, (title, li) in zip(axes, keys.items()):
        X = stacked[:, li, :]; Xc = X - X.mean(axis=0, keepdims=True)
        X2 = PCA(2).fit_transform(Xc)
        for lang, c in LANG_COLORS.items():
            m = [i for i, mt in enumerate(meta_list) if mt["lang"] == lang]
            if m: ax.scatter(X2[m, 0], X2[m, 1], c=c, s=20, alpha=0.7, label=lang)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(keys), len(axes)): axes[i].set_visible(False)
    handles = [Patch(color=c, label=l) for l, c in LANG_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Centered PCA by Language", fontsize=13, y=1.01)
    plt.tight_layout(); fig.savefig(out_dir / "pca_centered.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_centered.png'}")

    # LDA by language
    lang_list = sorted(set(m["lang"] for m in meta_list))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
    for ax, (title, li) in zip(axes, keys.items()):
        X = stacked[:, li, :]
        y = np.array([lang_list.index(m["lang"]) for m in meta_list])
        try:
            X2 = LDA(n_components=2).fit_transform(X, y)
            for lang, c in LANG_COLORS.items():
                m = [i for i, mt in enumerate(meta_list) if mt["lang"] == lang]
                if m: ax.scatter(X2[m, 0], X2[m, 1], c=c, s=20, alpha=0.7)
        except Exception: ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(keys), len(axes)): axes[i].set_visible(False)
    fig.suptitle("LDA by Language", fontsize=13, y=1.01)
    plt.tight_layout(); fig.savefig(out_dir / "pca_lda_lang.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_lda_lang.png'}")

    # LDA by type (S vs Q)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
    for ax, (title, li) in zip(axes, keys.items()):
        X = stacked[:, li, :]
        y = np.array([0 if m["type"] == "S" else 1 for m in meta_list])
        try:
            X2 = LDA(n_components=1).fit_transform(X, y)
            for typ, c in TYPE_COLORS.items():
                mm = [i for i, mt in enumerate(meta_list) if mt["type"] == typ]
                ax.scatter(X2[mm], [0]*len(mm), c=c, marker="o" if typ=="S" else "^", s=20, alpha=0.7)
        except Exception: ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9); ax.set_yticks([])
    for i in range(len(keys), len(axes)): axes[i].set_visible(False)
    handles = [Patch(color=TYPE_COLORS["S"], label="S"), Patch(color=TYPE_COLORS["Q"], label="Q")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("LDA: Statement vs Question", fontsize=13, y=1.01)
    plt.tight_layout(); fig.savefig(out_dir / "pca_lda_type.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_lda_type.png'}")

    # Distance heatmap (final layer)
    X = stacked[:, -1, :]
    dist = cosine_distances(X)
    order = sorted(range(len(meta_list)), key=lambda i: (meta_list[i]["lang"], meta_list[i]["type"]))
    labels = [f"{meta_list[i]['lang']}-{meta_list[i]['type']}" for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(dist[np.ix_(order, order)], cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Cosine distance matrix — final layer", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
    fig.savefig(out_dir / "pca_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_dir / 'pca_heatmap.png'}")

    # ── Topic-colored: statements only, then questions only ──
    TOPIC_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
    # build topic array from configs (same order as stacked)
    topic_arr = np.array([c["topic"] for c in configs])
    type_arr = np.array([c["type"] for c in configs])

    for typ, typ_name in [("S", "Statements"), ("Q", "Questions")]:
        mask = type_arr == typ
        X_typ = stacked[mask]
        top_typ = topic_arr[mask]
        total = num_layers

        # ── ALL layers grid ──
        cols = 7; rows = (total + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        for idx in range(total):
            ax = axes[idx]
            X = X_typ[:, idx, :]; Xc = X - X.mean(axis=0, keepdims=True)
            X2 = PCA(2).fit_transform(Xc)
            for t in range(len(TOPICS)):
                m = top_typ == t
                if m.any():
                    ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], s=15, alpha=0.8)
            label_name = "embedding" if idx == 0 else f"L{idx-1}"
            ax.set_title(label_name, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for idx in range(total, len(axes)): axes[idx].set_visible(False)
        handles = [Patch(color=TOPIC_COLORS[t], label=TOPIC_NAMES[t]) for t in range(len(TOPICS))]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"Centered PCA by Topic — ALL Layers ({typ_name})", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"pca_topic_all_{typ.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {out_dir / f'pca_topic_all_{typ.lower()}.png'}")

        # ── Key layers only (larger) ──
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
        for ax, (title, li) in zip(axes, keys.items()):
            X = X_typ[:, li, :]; Xc = X - X.mean(axis=0, keepdims=True)
            X2 = PCA(2).fit_transform(Xc)
            for t in range(len(TOPICS)):
                m = top_typ == t
                if m.any():
                    ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], s=35, alpha=0.85,
                              edgecolors="white", linewidths=0.3)
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        for i in range(len(keys), len(axes)): axes[i].set_visible(False)
        fig.suptitle(f"Centered PCA by Topic — {typ_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"pca_topic_{typ.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {out_dir / f'pca_topic_{typ.lower()}.png'}")

        # LDA by topic (key layers)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
        for ax, (title, li) in zip(axes, keys.items()):
            X = X_typ[:, li, :]
            try:
                X2 = LDA(n_components=2).fit_transform(X, top_typ)
                for t in range(len(TOPICS)):
                    m = top_typ == t
                    if m.any():
                        ax.scatter(X2[m, 0], X2[m, 1], c=[TOPIC_COLORS[t]], s=35, alpha=0.85,
                                  edgecolors="white", linewidths=0.3)
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        for i in range(len(keys), len(axes)): axes[i].set_visible(False)
        fig.suptitle(f"LDA by Topic — {typ_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / f"lda_topic_{typ.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {out_dir / f'lda_topic_{typ.lower()}.png'}")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Unified layer analysis pipeline")
    p.add_argument("--model", "-m", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out-dir", "-o", default="results/run_01")
    p.add_argument("--skip-extract", action="store_true", help="Skip extraction, only analyze existing vecs/")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if not args.skip_extract:
        analyzer = LayerAnalyzer(model_id=args.model, dtype=dtype_map[args.dtype])
        tensors = extract_all(analyzer, out_dir)
        meta = {
            "model_id": analyzer.model_id,
            "num_layers": analyzer.num_layers,
            "hidden_size": analyzer.hidden_size,
        }
    else:
        print("Loading existing vectors ...")
        vecs_dir = out_dir / "vecs"
        tensors = {}
        for pt in sorted(vecs_dir.glob("*.pt")):
            tensors[pt.stem] = torch.load(pt, weights_only=True)
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
        print(f"  Loaded {len(tensors)} vectors")

    meta["num_layers"] = tensors[list(tensors.keys())[0]].shape[0] - 1 if "num_layers" not in meta else meta["num_layers"]

    run_sw_analysis(tensors, meta, out_dir)
    run_pca_analysis(tensors, meta, out_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s  →  {out_dir.resolve()}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
