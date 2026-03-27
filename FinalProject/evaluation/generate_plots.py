"""
evaluation/generate_plots.py
============================
Generate all Phase 1 visualisations (Configs A, B, C) from the saved CSVs.

Usage (from FinalProject/):
    python evaluation/generate_plots.py
    python evaluation/generate_plots.py --base datasets/evaluations/raw_results
    python evaluation/generate_plots.py --base datasets/evaluations/phase1
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_BASE = "datasets/evaluations/raw_results"

CONFIGS = {
    "Config A\n(short q + passages)":  "config_a",
    "Config B\n(title q + full docs)": "config_b",
    "Config C\n(body q + full docs)":  "config_c",
}
RETRIEVER_ORDER = [
    "Classic", "BM25",
    "MinHashDBSCAN", "MinHashLSH",
    "ProbMinHashDBSCAN", "ProbMinHashLSH",
]
PALETTE = {
    "Classic":           "#4C72B0",
    "BM25":              "#DD8452",
    "MinHashDBSCAN":     "#55A868",
    "MinHashLSH":        "#C44E52",
    "ProbMinHashDBSCAN": "#8172B2",
    "ProbMinHashLSH":    "#937860",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _colors(rets):
    return [PALETTE.get(r, "#888888") for r in rets]

def _ordered(df):
    df = df.copy()
    df["retriever"] = pd.Categorical(df["retriever"], categories=RETRIEVER_ORDER, ordered=True)
    return df.sort_values("retriever")

def _save(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()

def _get(df, retriever, col):
    row = df[df["retriever"] == retriever]
    return float(row[col].iloc[0]) if not row.empty and col in row.columns else 0.0


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_grouped_bar(summaries, metric, ylabel, title, out):
    """Grouped bar: one metric across all configs side-by-side."""
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = list(summaries.keys())
    x, width = np.arange(len(RETRIEVER_ORDER)), 0.13
    for i, lbl in enumerate(labels):
        df = _ordered(summaries[lbl])
        vals = [_get(df, r, metric) for r in RETRIEVER_ORDER]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=lbl.replace("\n", " "),
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_ORDER, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(out)


def plot_heatmap(df, title, out):
    """Metrics × retrievers heatmap for one config."""
    df = _ordered(df)
    metrics = [c for c in ["hit_rate_at_all_mean", "mrr_mean", "map",
                             "recall_at_all_mean", "f1_at_all_mean"] if c in df.columns]
    labels  = ["Hit Rate", "MRR", "MAP", "Recall", "F1"][:len(metrics)]
    mat = df.set_index("retriever")[metrics].reindex(
        [r for r in RETRIEVER_ORDER if r in df["retriever"].values]
    ).astype(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mat.values.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat)))
    ax.set_xticklabels(mat.index, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(mat)):
        for j in range(len(metrics)):
            v = mat.values[i, j]
            ax.text(i, j, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(out)


def plot_per_config_bars(data_dict, col, ylabel, title, out, std_col=None):
    """One bar-chart subplot per config for a given column."""
    fig, axes = plt.subplots(1, len(data_dict), figsize=(5 * len(data_dict), 5), sharey=False)
    axes = [axes] if len(data_dict) == 1 else list(axes)
    for ax, (lbl, df) in zip(axes, data_dict.items()):
        df = _ordered(df)
        rets = [r for r in RETRIEVER_ORDER if r in df["retriever"].values]
        vals = [_get(df, r, col) for r in rets]
        errs = [_get(df, r, std_col) for r in rets] if std_col else None
        ax.bar(range(len(rets)), vals, yerr=errs, capsize=4,
               color=_colors(rets), alpha=0.85, edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(rets)))
        ax.set_xticklabels(rets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(out)


def plot_update_latency(updates, out):
    """Log-scale median + p95 update latency per config."""
    if not updates:
        print("  No update data — skipping")
        return
    combined = pd.concat(
        [df.assign(config=lbl.replace("\n", " ")) for lbl, df in updates.items()],
        ignore_index=True,
    )
    configs = combined["config"].unique()
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), sharey=False)
    axes = [axes] if len(configs) == 1 else list(axes)
    for ax, cfg in zip(axes, configs):
        sub  = combined[combined["config"] == cfg]
        rets = [r for r in RETRIEVER_ORDER if r in sub["retriever"].values]
        med  = [sub[sub["retriever"] == r]["update_latency_s"].median() for r in rets]
        p95  = [sub[sub["retriever"] == r]["update_latency_s"].quantile(0.95) for r in rets]
        x    = np.arange(len(rets))
        ax.bar(x - 0.18, med, 0.34, label="Median", color=_colors(rets), alpha=0.8, edgecolor="black")
        ax.bar(x + 0.18, p95, 0.34, label="p95",    color=_colors(rets), alpha=0.45,
               edgecolor="black", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(rets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Latency (s)")
        ax.set_title(cfg, fontsize=10)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3g}"))
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Update Latency — Median & p95 (log scale)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(out)


def plot_accuracy_vs_memory(summaries, out):
    """MRR vs peak memory scatter per config."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 5))
    axes = [axes] if len(summaries) == 1 else list(axes)
    for ax, (lbl, df) in zip(axes, summaries.items()):
        df = _ordered(df)
        for r in RETRIEVER_ORDER:
            if df[df["retriever"] == r].empty:
                continue
            mem   = _get(df, r, "memory_mb_mean")
            mrr_v = _get(df, r, "mrr_mean")
            ax.scatter(mem, mrr_v, color=PALETTE.get(r, "#888"), s=120,
                       zorder=3, edgecolors="black", linewidths=0.8)
            ax.annotate(r, (mem, mrr_v), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=PALETTE.get(r, "#888"))
        ax.set_xlabel("Peak Memory (MB)")
        ax.set_ylabel("MRR")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MRR vs Memory Usage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(out)


def plot_mrr_violin(accuracy, out):
    """Per-query MRR distribution violin per config."""
    if not accuracy:
        return
    fig, axes = plt.subplots(1, len(accuracy), figsize=(6 * len(accuracy), 5), sharey=True)
    axes = [axes] if len(accuracy) == 1 else list(axes)
    for ax, (lbl, df) in zip(axes, accuracy.items()):
        rets = [r for r in RETRIEVER_ORDER if not df[df["retriever"] == r].empty]
        data = [df[df["retriever"] == r]["mrr"].dropna().values for r in rets]
        if not data:
            continue
        parts = ax.violinplot(data, positions=range(len(rets)), showmedians=True, showextrema=False)
        for pc, r in zip(parts["bodies"], rets):
            pc.set_facecolor(PALETTE.get(r, "#888"))
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(rets)))
        ax.set_xticklabels(rets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("MRR")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-Query MRR Distribution (violin)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(out)


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 evaluation plots")
    parser.add_argument(
        "--base", default=DEFAULT_BASE,
        help=f"Root results directory (default: {DEFAULT_BASE})",
    )
    args = parser.parse_args()

    base = args.base
    out  = os.path.join(base, "plots")
    os.makedirs(out, exist_ok=True)

    summaries, accuracy, memory, updates = {}, {}, {}, {}
    for lbl, cfg in CONFIGS.items():
        d = os.path.join(base, cfg)
        if os.path.exists(f"{d}/summary.csv"):
            summaries[lbl] = pd.read_csv(f"{d}/summary.csv")
        if os.path.exists(f"{d}/accuracy.csv"):
            accuracy[lbl]  = pd.read_csv(f"{d}/accuracy.csv")
        if os.path.exists(f"{d}/memory.csv"):
            memory[lbl]    = pd.read_csv(f"{d}/memory.csv")
        if os.path.exists(f"{d}/update_time.csv"):
            updates[lbl]   = pd.read_csv(f"{d}/update_time.csv")

    print(f"Loaded configs: {[k.replace(chr(10), ' ') for k in summaries]}")
    print(f"Saving plots → {out}/\n")

    # 1 — Hit Rate @all across all configs
    plot_grouped_bar(
        summaries, "hit_rate_at_all_mean", "Hit Rate @all",
        "Hit Rate @all — All Configs",
        f"{out}/hit_rate_all_configs.png",
    )
    # 2 — MRR across all configs
    plot_grouped_bar(
        summaries, "mrr_mean", "Mean Reciprocal Rank (MRR)",
        "MRR — All Configs",
        f"{out}/mrr_all_configs.png",
    )
    # 3 — Per-config accuracy heatmap
    for lbl, cfg in CONFIGS.items():
        if lbl in summaries:
            plot_heatmap(
                summaries[lbl],
                f"Accuracy Metrics — {lbl.replace(chr(10), ' ')}",
                f"{out}/heatmap_{cfg}.png",
            )
    # 4 — Memory comparison
    plot_per_config_bars(
        summaries, "memory_mb_mean", "Peak Memory (MB)",
        "Peak Query Memory Usage (MB)",
        f"{out}/memory_comparison.png",
    )
    # 5 — Update latency
    plot_update_latency(updates, f"{out}/update_latency.png")
    # 6 — MRR vs memory scatter
    plot_accuracy_vs_memory(summaries, f"{out}/accuracy_vs_memory.png")
    # 7 — Per-query MRR violin
    plot_mrr_violin(accuracy, f"{out}/mrr_violin.png")
    # 8 — Result set size
    plot_per_config_bars(
        accuracy, "n_results", "Mean # Results Returned",
        "Mean Result Set Size per Retriever",
        f"{out}/n_results.png",
    )

    saved = [f for f in os.listdir(out) if f.endswith(".png")]
    print(f"\nAll plots saved to: {out}/")
    print(f"Files ({len(saved)}):")
    for f in sorted(saved):
        print(f"  {f}")


if __name__ == "__main__":
    main()
