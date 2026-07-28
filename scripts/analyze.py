#!/usr/bin/env python3
"""
Layer output analysis: PCA visualization + Sapir-Whorf test.

Usage:
    uv run python scripts/analyze.py pca output/batch/topic* -o pca_output
    uv run python scripts/analyze.py sw output/batch/topic* -o pca_output
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets.prompts import PROMPTS


BATCH_DIR = Path("output/batch")

def _get_batch_dir(dirs: list[Path] | None) -> Path:
    """If dirs given, use parent of first; else default."""
    if dirs:
        return dirs[0].parent if dirs[0].parent.name == "batch" else dirs[0].parent
    return BATCH_DIR
LANG_COLORS = {"EN": "#e74c3c", "ZH": "#3498db", "DE": "#2ecc71", "FR": "#f39c12", "JA": "#9b59b6"}
TYPE_COLORS = {"S": "#2980b9", "Q": "#e74c3c"}

# ══════════════════════════════════════════════════════════════════════
# data loading
# ══════════════════════════════════════════════════════════════════════

def load_vec(topic: int, lang: str, typ: str, batch_dir: Path = None) -> np.ndarray:
    bd = batch_dir or BATCH_DIR
    name = f"topic{topic}_{lang.lower()}_{typ}"
    return torch.load(bd / name / "vecs" / "stacked.pt", weights_only=True).float().numpy()

def load_runs(run_dirs: list[Path]):
    stacked, configs, dirs = [], [], []
    for d in sorted(run_dirs):
        pt = d / "vecs" / "stacked.pt"
        if not pt.exists(): continue
        stacked.append(torch.load(pt, weights_only=True).float().numpy())
        with open(d / "report.json") as f: configs.append(json.load(f)["config"])
        dirs.append(d)
    return stacked, configs, dirs

def parse_meta(configs, dirs):
    meta = []
    for c, d in zip(configs, dirs):
        parts = d.name.split("_")
        if len(parts) >= 3:
            lang, typ = parts[1].upper(), parts[2].upper()
        else:
            prompt = c["prompt"]
            has_zh = any('\u4e00' <= ch <= '\u9fff' for ch in prompt)
            has_ja = any('\u3040' <= ch <= '\u30ff' for ch in prompt)
            lang = "JA" if has_ja else ("ZH" if has_zh else "EN")
            is_q = ("?" in prompt or "吗" in prompt or prompt.lower().startswith(("is ","are ","do ","does ")))
            typ = "Q" if is_q else "S"
        meta.append({"lang": lang, "type": typ, "prompt": c["prompt"][:60]})
    return meta

# ══════════════════════════════════════════════════════════════════════
# plotting helpers (improved over raw PCA)
# ══════════════════════════════════════════════════════════════════════

def _plot_centered_pca(ax, X, meta, title):
    """PCA after subtracting the global mean (removes dominant shared direction)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    X2 = PCA(2).fit_transform(Xc)
    for lang, c in LANG_COLORS.items():
        m = [i for i, mt in enumerate(meta) if mt["lang"] == lang]
        if m: ax.scatter(X2[m, 0], X2[m, 1], c=c, s=30, alpha=0.8, label=lang)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def _plot_lda_lang(ax, X, meta, title):
    """Linear Discriminant Analysis — supervised separation by language."""
    lang_list = sorted(set(m["lang"] for m in meta))
    y = np.array([lang_list.index(m["lang"]) for m in meta])
    try:
        X2 = LDA(n_components=2).fit_transform(X, y)
        for lang, c in LANG_COLORS.items():
            m = [i for i, mt in enumerate(meta) if mt["lang"] == lang]
            if m: ax.scatter(X2[m, 0], X2[m, 1], c=c, s=30, alpha=0.8, label=lang)
    except Exception:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def _plot_cosine_mds(ax, X, meta, title):
    """MDS on cosine distance matrix — preserves local structure better."""
    D = cosine_distances(X)
    try:
        X2 = MDS(n_components=2, metric="precomputed", random_state=42,
                 max_iter=300, n_init=1).fit_transform(D)
        for lang, c in LANG_COLORS.items():
            m = [i for i, mt in enumerate(meta) if mt["lang"] == lang]
            if m: ax.scatter(X2[m, 0], X2[m, 1], c=c, s=30, alpha=0.8, label=lang)
    except Exception:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def _plot_topic_type_curves(stacked, meta, num_layers, out):
    """Same-topic cross-language: same-type (S↔S, Q↔Q) vs diff-type (S↔Q)."""
    from datasets.prompts import PROMPTS as P
    langs = list(P[0].keys())
    topics = list(range(len(P)))
    # Build index: (topic, lang, type) -> row in stacked
    idx = {}
    for i, m in enumerate(meta):
        parts = Path(str(i)).name  # won't work; use meta ordering
    # Reconstruct from dirs
    flat = flatten_prompts()
    # meta is aligned with stacked; find which topic/lang/type each row is
    # Use the dirs from load_runs
    # Actually, simpler: reload directly from BATCH_DIR
    import torch as _torch
    bd = BATCH_DIR

    all_blue = np.zeros(num_layers + 1)
    all_red = np.zeros(num_layers + 1)

    for t in topics:
        bc, rc = [], []
        for layer in range(num_layers + 1):
            bv, rv = [], []
            for typ in ["S", "Q"]:
                for i, l1 in enumerate(langs):
                    for l2 in langs[i+1:]:
                        v1 = _torch.load(bd / f"topic{t}_{l1}_{typ}" / "vecs" / "stacked.pt",
                                         weights_only=True).float().numpy()[layer]
                        v2 = _torch.load(bd / f"topic{t}_{l2}_{typ}" / "vecs" / "stacked.pt",
                                         weights_only=True).float().numpy()[layer]
                        bv.append(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0,0])
            for l1 in langs:
                for l2 in langs:
                    if l1 == l2: continue
                    v1 = _torch.load(bd / f"topic{t}_{l1}_S" / "vecs" / "stacked.pt",
                                     weights_only=True).float().numpy()[layer]
                    v2 = _torch.load(bd / f"topic{t}_{l2}_Q" / "vecs" / "stacked.pt",
                                     weights_only=True).float().numpy()[layer]
                    rv.append(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0,0])
            for l in langs:
                v1 = _torch.load(bd / f"topic{t}_{l}_S" / "vecs" / "stacked.pt",
                                 weights_only=True).float().numpy()[layer]
                v2 = _torch.load(bd / f"topic{t}_{l}_Q" / "vecs" / "stacked.pt",
                                 weights_only=True).float().numpy()[layer]
                rv.append(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0,0])
            bc.append(np.mean(bv)); rc.append(np.mean(rv))
        all_blue += np.array(bc); all_red += np.array(rc)

    all_blue /= len(topics); all_red /= len(topics)

    xs = range(num_layers + 1)
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers + 1)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(xs, all_blue, color="#3498db", linewidth=2.5, label="Same type (S↔S + Q↔Q cross-lang)")
    ax.plot(xs, all_red, color="#e74c3c", linewidth=2.5, label="Diff type (S↔Q cross-lang)")

    # fill between
    ab, ar = np.array(all_blue), np.array(all_red)
    ax.fill_between(xs, ab, ar, where=(ab > ar), alpha=0.1, color="#3498db")
    ax.fill_between(xs, ab, ar, where=(ab <= ar), alpha=0.1, color="#e74c3c")

    # phase labels
    bw = [ab[i] > ar[i] for i in range(len(xs))]
    phases, prev, start = [], bw[0], 0
    for i in range(1, len(xs)):
        if bw[i] != prev: phases.append((start, i-1, prev)); start = i; prev = bw[i]
    phases.append((start, len(xs)-1, prev))
    for s, e, b in phases:
        if e - s >= 2:
            label = "S ≠ Q (type matters)" if not b else "S ≈ Q (type erased)"
            color = "#e74c3c" if not b else "#3498db"
            mid = (s + e) / 2
            y = max(ab[s:e+1].max(), ar[s:e+1].max()) + 0.03
            ax.annotate(label, (mid, y), fontsize=9, color=color, ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xticks(range(0, len(xs), 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, len(xs), 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title("Same-Topic Cross-Language: Same-Type vs Different-Type (S/Q)", fontsize=14)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out / "topic_type_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out / 'topic_type_curve.png'}")


# ══════════════════════════════════════════════════════════════════════
# PCA plots
# ══════════════════════════════════════════════════════════════════════

def pca_command(args):
    stacked, configs, dirs = load_runs(args.run_dirs)
    meta = parse_meta(configs, dirs)
    num_layers = stacked[0].shape[0] - 1
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    print(f"Runs: {len(stacked)}  Layers: {num_layers}  Hidden: {stacked[0].shape[1]}")

    # ═══ key layers: 3 methods (centered PCA, LDA, cosine MDS) ═══
    keys = {"emb": 0, "L01": 1, "L06": 6, "L12": 12, "L18": 18, "L24": 24, "L30": 30, f"L{num_layers-1}": num_layers}

    for method_name, method_fn in [
        ("centered_pca", _plot_centered_pca),
        ("lda_lang", _plot_lda_lang),
        ("cosine_mds", _plot_cosine_mds),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
        for ax, (title, li) in zip(axes, keys.items()):
            X = np.array([s[li] for s in stacked])
            method_fn(ax, X, meta, title)
        for i in range(len(keys), len(axes)): axes[i].set_visible(False)
        fig.suptitle(f"{method_name} — Qwen3-4B-Instruct", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out / f"{method_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {out / f'{method_name}.png'}")

    # ── LDA by type (S vs Q) ──
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
    for ax, (title, li) in zip(axes, keys.items()):
        X = np.array([s[li] for s in stacked])
        y = np.array([0 if m["type"] == "S" else 1 for m in meta])
        try:
            X2 = LDA(n_components=1).fit_transform(X, y)
            for typ, c in TYPE_COLORS.items():
                mm = [i for i, mt in enumerate(meta) if mt["type"] == typ]
                ax.scatter(X2[mm], [0]*len(mm), c=c, marker="o" if typ=="S" else "^",
                          s=40, alpha=0.8, label=typ)
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(title, fontsize=9); ax.set_yticks([])
    for i in range(len(keys), len(axes)): axes[i].set_visible(False)
    handles = [Patch(color=TYPE_COLORS["S"], label="Statement"), Patch(color=TYPE_COLORS["Q"], label="Question")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("LDA: Statement vs Question", fontsize=13, y=1.01)
    plt.tight_layout(); fig.savefig(out / "lda_type.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'lda_type.png'}")

    # ── same-topic: same-type vs diff-type across languages ──
    _plot_topic_type_curves(stacked, meta, num_layers, out)

    # ── distance heatmap (final layer) ──
    X = np.array([s[num_layers] for s in stacked])
    dist = cosine_distances(X)
    order = sorted(range(len(meta)), key=lambda i: (meta[i]["lang"], meta[i]["type"]))
    labels = [f"{meta[i]['lang']}-{meta[i]['type']}" for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(dist[np.ix_(order, order)], cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"Cosine distance matrix — final layer", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
    fig.savefig(out / "pca_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'pca_heatmap.png'}")

    # ── separation evolution ──
    langs_list = sorted(set(m["lang"] for m in meta))
    lang_groups = {l: [i for i, m in enumerate(meta) if m["lang"] == l] for l in langs_list}
    type_groups = {"S": [i for i, m in enumerate(meta) if m["type"] == "S"],
                   "Q": [i for i, m in enumerate(meta) if m["type"] == "Q"]}
    lang_sep, type_sep = [], []
    for li in range(num_layers + 1):
        Xl = np.array([s[li] for s in stacked])
        ld = [cosine_distances(Xl[lang_groups[l1]], Xl[lang_groups[l2]]).mean()
              for i, l1 in enumerate(langs_list) for l2 in langs_list[i+1:]]
        lang_sep.append(np.mean(ld) if ld else 0)
        type_sep.append(cosine_distances(Xl[type_groups["S"]], Xl[type_groups["Q"]]).mean())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(num_layers+1), lang_sep, color="#2c3e50", linewidth=2)
    ax1.fill_between(range(num_layers+1), 0, lang_sep, alpha=0.15, color="#2c3e50")
    ax1.set_title("Cross-Language Separation"); ax1.set_xlabel("Layer"); ax1.grid(True, alpha=0.3)
    ax2.plot(range(num_layers+1), type_sep, color="#c0392b", linewidth=2)
    ax2.fill_between(range(num_layers+1), 0, type_sep, alpha=0.15, color="#c0392b")
    ax2.set_title("Statement vs Question Separation"); ax2.set_xlabel("Layer"); ax2.grid(True, alpha=0.3)
    fig.suptitle("Separation Score Evolution", fontsize=13); plt.tight_layout()
    fig.savefig(out / "pca_separation.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'pca_separation.png'}")
    print(f"\nDone → {out.resolve()}/")


# ══════════════════════════════════════════════════════════════════════
# Sapir-Whorf (RGB) analysis
# ══════════════════════════════════════════════════════════════════════

def sw_command(args):
    batch_dir = _get_batch_dir(args.run_dirs) if hasattr(args, 'run_dirs') and args.run_dirs else BATCH_DIR
    topics = list(range(len(PROMPTS)))
    langs = list(PROMPTS[0].keys())
    num_layers = 37
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    print(f"Batch dir: {batch_dir}")

    def compute_rgb(typ="S"):
        rm, gm, bm, rs, gs, bs = [], [], [], [], [], []
        for layer in range(num_layers):
            vecs = {(t, l): load_vec(t, l, typ, batch_dir)[layer] for t in topics for l in langs}
            red, green, blue = [], [], []
            for t in topics:
                for i, l1 in enumerate(langs):
                    for l2 in langs[i+1:]:
                        red.append(cosine_similarity(vecs[(t,l1)].reshape(1,-1), vecs[(t,l2)].reshape(1,-1))[0,0])
            for l in langs:
                for i, t1 in enumerate(topics):
                    for t2 in topics[i+1:]:
                        green.append(cosine_similarity(vecs[(t1,l)].reshape(1,-1), vecs[(t2,l)].reshape(1,-1))[0,0])
            for t1 in topics:
                for t2 in topics:
                    if t1 == t2: continue
                    for l1 in langs:
                        for l2 in langs:
                            if l1 == l2: continue
                            blue.append(cosine_similarity(vecs[(t1,l1)].reshape(1,-1), vecs[(t2,l2)].reshape(1,-1))[0,0])
            rm.append(np.mean(red)); rs.append(np.std(red))
            gm.append(np.mean(green)); gs.append(np.std(green))
            bm.append(np.mean(blue)); bs.append(np.std(blue))
        return {"rm": rm, "rs": rs, "gm": gm, "gs": gs, "bm": bm, "bs": bs}

    def plot_rgb(rgb, title_suffix, filename):
        xs = range(num_layers)
        xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(xs, rgb["rm"], color="#e74c3c", linewidth=2.2, label="Same topic, diff lang (RED)")
        ax.fill_between(xs, np.array(rgb["rm"])-np.array(rgb["rs"]), np.array(rgb["rm"])+np.array(rgb["rs"]), alpha=0.12, color="#e74c3c")
        ax.plot(xs, rgb["gm"], color="#27ae60", linewidth=2.2, label="Same lang, diff topic (GREEN)")
        ax.fill_between(xs, np.array(rgb["gm"])-np.array(rgb["gs"]), np.array(rgb["gm"])+np.array(rgb["gs"]), alpha=0.12, color="#27ae60")
        ax.plot(xs, rgb["bm"], color="#3498db", linewidth=2.2, label="Diff topic, diff lang (BLUE)")
        ax.fill_between(xs, np.array(rgb["bm"])-np.array(rgb["bs"]), np.array(rgb["bm"])+np.array(rgb["bs"]), alpha=0.12, color="#3498db")
        for i in range(num_layers):
            if rgb["rm"][i] > rgb["gm"][i]: ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="#e74c3c")
        # phase labels
        red_win = [rgb["rm"][i] > rgb["gm"][i] for i in range(num_layers)]
        phases, prev, start = [], red_win[0], 0
        for i in range(1, num_layers):
            if red_win[i] != prev: phases.append((start, i-1, prev)); start = i; prev = red_win[i]
        phases.append((start, num_layers-1, prev))
        for s, e, is_red in phases:
            label = "TOPIC SPACE\n(topic > language)" if is_red else "LANGUAGE\n(format > meaning)"
            y_pos = max(rgb["rm"][s:e+1] + rgb["gm"][s:e+1]) + 0.04
            if e - s >= 2:
                ax.annotate(label, ((s+e)/2, y_pos), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                            ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        ax.set_xticks(range(0, num_layers, 3))
        ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
        ax.set_ylabel("Cosine similarity"); ax.set_ylim(0, 1.0)
        ax.set_title(f"Cosine similarity across layers — Qwen3-4B-Instruct {title_suffix}", fontsize=14)
        ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
        n_red = sum(red_win)
        ax.text(0.02, 0.97, f"RED > GREEN in {n_red}/{num_layers} layers ({n_red/num_layers:.0%})\n"
                f"Model: Qwen3-4B-Instruct | {len(topics)} topics × {len(langs)} languages",
                transform=ax.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
        plt.tight_layout(); fig.savefig(out / filename, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✓  {out / filename}")

    # centered version
    def plot_centered(rgb, title_suffix, filename):
        xs = range(num_layers)
        xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]
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
            label = "SEMANTIC SPACE" if is_red else "LANGUAGE LAYER"
            y_pos = max(rc[s:e+1].max(), gc[s:e+1].max()) + 0.02
            if e - s >= 2:
                ax.annotate(label, ((s+e)/2, y_pos), fontsize=9, color="#e74c3c" if is_red else "#27ae60",
                            ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        ax.set_xticks(range(0, num_layers, 3))
        ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
        ax.set_ylabel("Centered cosine similarity"); ax.set_title(f"Centered cosine similarity {title_suffix}", fontsize=14)
        ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25)
        plt.tight_layout(); fig.savefig(out / filename, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  ✓  {out / filename}")

    print("\n── RGB curves (statements) ──")
    rgb_s = compute_rgb("S")
    plot_rgb(rgb_s, "(Statements)", "sw_main_statements.png")
    plot_centered(rgb_s, "(Statements)", "sw_centered_statements.png")

    print("\n── RGB curves (questions) ──")
    rgb_q = compute_rgb("Q")
    plot_rgb(rgb_q, "(Questions)", "sw_main_questions.png")

    # per-language-pair
    print("\n── Per language pair ──")
    pair_colors = {("EN","ZH"):"#e74c3c", ("EN","DE"):"#3498db", ("EN","FR"):"#2ecc71", ("EN","JA"):"#9b59b6",
                   ("ZH","DE"):"#f39c12", ("ZH","FR"):"#1abc9c", ("ZH","JA"):"#e67e22",
                   ("DE","FR"):"#2980b9", ("DE","JA"):"#8e44ad", ("FR","JA"):"#16a085"}
    fig, ax = plt.subplots(figsize=(15, 6))
    for (l1, l2), c in pair_colors.items():
        scores = []
        for layer in range(num_layers):
            sims = [cosine_similarity(load_vec(t,l1,"S",batch_dir)[layer].reshape(1,-1), load_vec(t,l2,"S",batch_dir)[layer].reshape(1,-1))[0,0]
                    for t in topics]
            scores.append(np.mean(sims))
        ax.plot(range(num_layers), scores, color=c, linewidth=1.8, alpha=0.8, label=f"{l1}↔{l2}")
    xlabels = ["emb"] + [f"L{i-1}" for i in range(1, num_layers)]
    ax.set_xticks(range(0, num_layers, 3))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 3)], fontsize=8)
    ax.set_ylabel("Cosine similarity"); ax.set_title("Cross-language semantic convergence by language pair", fontsize=14)
    ax.legend(fontsize=8, ncol=5, loc="lower right"); ax.grid(True, alpha=0.25)
    plt.tight_layout(); fig.savefig(out / "sw_per_lang_pair.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'sw_per_lang_pair.png'}")

    # topic heatmap
    print("\n── Topic heatmap ──")
    topic_names = ["Sun rises", "Water 100°C", "Earth orbits", "Plants light", "Humans O₂", "Ice melts", "Birds fly", "Light>sound"]
    matrix = np.zeros((len(topics), num_layers))
    for t in topics:
        for layer in range(num_layers):
            sims = [cosine_similarity(load_vec(t,l1,"S",batch_dir)[layer].reshape(1,-1), load_vec(t,l2,"S",batch_dir)[layer].reshape(1,-1))[0,0]
                    for i, l1 in enumerate(langs) for l2 in langs[i+1:]]
            matrix[t, layer] = np.mean(sims)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(0, num_layers, 2))
    ax.set_xticklabels([xlabels[i] for i in range(0, num_layers, 2)], fontsize=7, rotation=45)
    ax.set_yticks(range(len(topics))); ax.set_yticklabels(topic_names, fontsize=9)
    ax.set_title("Cross-language semantic convergence by topic", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean cross-language cosine sim")
    plt.tight_layout(); fig.savefig(out / "sw_topic_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'sw_topic_heatmap.png'}")

    # statement vs question comparison
    print("\n── Statement vs Question ──")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, rgb, title in [(ax1, rgb_s, "Statements"), (ax2, rgb_q, "Questions")]:
        xs = range(num_layers)
        ax.plot(xs, rgb["rm"], color="#e74c3c", linewidth=2, label="RED")
        ax.plot(xs, rgb["gm"], color="#27ae60", linewidth=2, label="GREEN")
        ax.plot(xs, rgb["bm"], color="#3498db", linewidth=2, label="BLUE")
        for i in range(num_layers):
            if rgb["rm"][i] > rgb["gm"][i]: ax.axvspan(i-0.5, i+0.5, alpha=0.07, color="#e74c3c")
        ax.set_title(title, fontsize=13); ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.suptitle("Statement vs Question: does grammar affect the interlingua?", fontsize=14)
    plt.tight_layout(); fig.savefig(out / "sw_stmt_vs_q.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out / 'sw_stmt_vs_q.png'}")

    print(f"\nDone → {out.resolve()}/")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Layer output analysis")
    sp = p.add_subparsers(dest="command", required=True)
    pca_p = sp.add_parser("pca", help="PCA visualization")
    pca_p.add_argument("run_dirs", nargs="+", type=Path)
    pca_p.add_argument("--output-dir", "-o", type=Path, default=Path("pca_output"))
    sw_p = sp.add_parser("sw", help="Sapir-Whorf RGB analysis")
    sw_p.add_argument("run_dirs", nargs="*", type=Path,
                       help="Batch directories (e.g. output/batch/topic0*). Uses output/batch/ if omitted.")
    sw_p.add_argument("--output-dir", "-o", type=Path, default=Path("pca_output"))
    args = p.parse_args()
    if args.command == "pca": pca_command(args)
    elif args.command == "sw": sw_command(args)

if __name__ == "__main__":
    main()
