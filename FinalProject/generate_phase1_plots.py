"""
generate_phase1_plots.py
========================
Generate visualisations for the Phase 1 evaluation results
(Configs A, B, C) from the saved CSVs.

Usage:
    cd FinalProject
    python generate_phase1_plots.py
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = "datasets/evaluations/phase1"
OUT    = "datasets/evaluations/phase1/plots"
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
os.makedirs(OUT, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
summaries: dict[str, pd.DataFrame] = {}
accuracy:  dict[str, pd.DataFrame] = {}
memory:    dict[str, pd.DataFrame] = {}
updates:   dict[str, pd.DataFrame] = {}

for label, cfg in CONFIGS.items():
    d = f"{BASE}/{cfg}"
    if os.path.exists(f"{d}/summary.csv"):
        summaries[label] = pd.read_csv(f"{d}/summary.csv")
    if os.path.exists(f"{d}/accuracy.csv"):
        accuracy[label] = pd.read_csv(f"{d}/accuracy.csv")
    if os.path.exists(f"{d}/memory.csv"):
        memory[label] = pd.read_csv(f"{d}/memory.csv")
    if os.path.exists(f"{d}/update_time.csv"):
        updates[label] = pd.read_csv(f"{d}/update_time.csv")

print(f"Loaded configs: {list(summaries.keys())}")

# ─────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────
def _colors(rets):
    return [PALETTE.get(r, "#888888") for r in rets]

def _order(df):
    df = df.copy()
    df["retriever"] = pd.Categorical(
        df["retriever"], categories=RETRIEVER_ORDER, ordered=True
    )
    return df.sort_values("retriever")


# =========================================================================
# Plot 1 — Hit-rate @ all  across all configs  (grouped bar)
# =========================================================================
def plot_hit_rate_all_configs():
    fig, ax = plt.subplots(figsize=(12, 5))
    cfg_labels = list(summaries.keys())
    n_ret = len(RETRIEVER_ORDER)
    n_cfg = len(cfg_labels)
    width = 0.13
    x = np.arange(n_ret)

    for i, cfg_lbl in enumerate(cfg_labels):
        df = summaries[cfg_lbl]
        df = _order(df)
        vals = []
        for r in RETRIEVER_ORDER:
            row = df[df["retriever"] == r]
            vals.append(float(row["hit_rate_at_all_mean"].iloc[0]) if not row.empty else 0.0)
        offset = (i - n_cfg / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=cfg_lbl.replace("\n", " "),
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Hit Rate @all")
    ax.set_ylim(0, 1.18)
    ax.set_title("Hit Rate @all — All Configs", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{OUT}/hit_rate_all_configs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 2 — MRR across all configs  (grouped bar)
# =========================================================================
def plot_mrr_all_configs():
    fig, ax = plt.subplots(figsize=(12, 5))
    cfg_labels = list(summaries.keys())
    n_ret = len(RETRIEVER_ORDER)
    n_cfg = len(cfg_labels)
    width = 0.13
    x = np.arange(n_ret)

    for i, cfg_lbl in enumerate(cfg_labels):
        df = summaries[cfg_lbl]
        df = _order(df)
        vals = []
        for r in RETRIEVER_ORDER:
            row = df[df["retriever"] == r]
            vals.append(float(row["mrr_mean"].iloc[0]) if not row.empty else 0.0)
        offset = (i - n_cfg / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=cfg_lbl.replace("\n", " "),
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_ylim(0, 1.15)
    ax.set_title("MRR — All Configs", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{OUT}/mrr_all_configs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 3 — Per-config summary heatmap (metrics × retrievers)
# =========================================================================
def plot_summary_heatmap(cfg_lbl: str, cfg_key: str):
    if cfg_lbl not in summaries:
        return
    df = _order(summaries[cfg_lbl])
    df = df[df["retriever"].isin(RETRIEVER_ORDER)]

    metrics = ["hit_rate_at_all_mean", "mrr_mean", "map", "recall_at_all_mean", "f1_at_all_mean"]
    metric_labels = ["Hit Rate", "MRR", "MAP", "Recall", "F1"]
    metrics = [m for m in metrics if m in df.columns]
    metric_labels = metric_labels[:len(metrics)]

    mat = df.set_index("retriever")[metrics].reindex(
        [r for r in RETRIEVER_ORDER if r in df["retriever"].values]
    ).astype(float)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mat.values.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.index)))
    ax.set_xticklabels(mat.index, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    for i in range(len(mat.index)):
        for j in range(len(metrics)):
            v = mat.values[i, j]
            ax.text(i, j, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Accuracy Metrics — {cfg_lbl.replace(chr(10), ' ')}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/heatmap_{cfg_key}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 4 — Memory usage comparison (per config)
# =========================================================================
def plot_memory_comparison():
    fig, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 5), sharey=False)
    if len(summaries) == 1:
        axes = [axes]

    for ax, (cfg_lbl, df) in zip(axes, summaries.items()):
        df = _order(df)
        rets = [r for r in RETRIEVER_ORDER if r in df["retriever"].values]
        vals = [float(df[df["retriever"] == r]["memory_mb_mean"].iloc[0]) for r in rets]
        colors = _colors(rets)
        bars = ax.bar(range(len(rets)), vals, color=colors, alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(rets)))
        ax.set_xticklabels(rets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title(cfg_lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Peak Query Memory Usage (MB)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/memory_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 5 — Update latency (only configs that have update_time.csv)
# =========================================================================
def plot_update_latency():
    if not updates:
        print("  No update data available — skipping")
        return

    # Combine all configs that have update data (usually only config_a & config_c)
    frames = []
    for lbl, df in updates.items():
        df = df.copy()
        df["config"] = lbl.replace("\n", " ")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    configs = combined["config"].unique()
    rets = [r for r in RETRIEVER_ORDER if r in combined["retriever"].values]

    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), sharey=False)
    if len(configs) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        sub = combined[combined["config"] == cfg]
        medians, p95s = [], []
        valid_rets = []
        for r in rets:
            r_data = sub[sub["retriever"] == r]["update_latency_s"]
            if r_data.empty:
                continue
            valid_rets.append(r)
            medians.append(float(r_data.median()))
            p95s.append(float(r_data.quantile(0.95)))
        colors = _colors(valid_rets)
        x = np.arange(len(valid_rets))
        ax.bar(x - 0.18, medians, 0.34, label="Median", color=colors, alpha=0.8, edgecolor="black")
        ax.bar(x + 0.18, p95s,    0.34, label="p95",    color=colors, alpha=0.45, edgecolor="black", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_rets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Update Latency (s)")
        ax.set_title(cfg, fontsize=10)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3g}"))
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle("Update Latency — Median & p95 (log scale)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/update_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 6 — Accuracy vs Memory scatter (per config)
# =========================================================================
def plot_accuracy_vs_memory():
    fig, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 5), sharey=False)
    if len(summaries) == 1:
        axes = [axes]

    for ax, (cfg_lbl, df) in zip(axes, summaries.items()):
        df = _order(df)
        for r in RETRIEVER_ORDER:
            row = df[df["retriever"] == r]
            if row.empty:
                continue
            mem = float(row["memory_mb_mean"].iloc[0])
            mrr = float(row["mrr_mean"].iloc[0])
            ax.scatter(mem, mrr, color=PALETTE.get(r, "#888"), s=120, zorder=3,
                       edgecolors="black", linewidths=0.8)
            ax.annotate(r, (mem, mrr), textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=PALETTE.get(r, "#888"))

        ax.set_xlabel("Peak Memory (MB)")
        ax.set_ylabel("MRR")
        ax.set_title(cfg_lbl.replace("\n", " "), fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MRR vs Memory Usage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/accuracy_vs_memory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 7 — Per-query MRR distribution (violin) for Config B & C
# =========================================================================
def plot_mrr_violin():
    configs_to_plot = {k: v for k, v in accuracy.items() if k in accuracy}
    if not configs_to_plot:
        return

    fig, axes = plt.subplots(1, len(configs_to_plot), figsize=(6 * len(configs_to_plot), 5), sharey=True)
    if len(configs_to_plot) == 1:
        axes = [axes]

    for ax, (cfg_lbl, df) in zip(axes, configs_to_plot.items()):
        data_by_ret = []
        labels = []
        for r in RETRIEVER_ORDER:
            sub = df[df["retriever"] == r]["mrr"].dropna()
            if not sub.empty:
                data_by_ret.append(sub.values)
                labels.append(r)
        if not data_by_ret:
            continue
        parts = ax.violinplot(data_by_ret, positions=range(len(labels)),
                              showmedians=True, showextrema=False)
        for i, (pc, lbl) in enumerate(zip(parts["bodies"], labels)):
            pc.set_facecolor(PALETTE.get(lbl, "#888"))
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("MRR")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(cfg_lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Per-Query MRR Distribution (violin)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/mrr_violin.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Plot 8 — N results returned  (how many docs each retriever returns)
# =========================================================================
def plot_n_results():
    if not accuracy:
        return
    fig, axes = plt.subplots(1, len(accuracy), figsize=(5 * len(accuracy), 5), sharey=False)
    if len(accuracy) == 1:
        axes = [axes]

    for ax, (cfg_lbl, df) in zip(axes, accuracy.items()):
        means, stds, labels = [], [], []
        for r in RETRIEVER_ORDER:
            sub = df[df["retriever"] == r]["n_results"].dropna()
            if not sub.empty:
                means.append(float(sub.mean()))
                stds.append(float(sub.std()))
                labels.append(r)
        colors = _colors(labels)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean # Results Returned")
        ax.set_title(cfg_lbl.replace("\n", " "), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean Result Set Size per Retriever", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/n_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print(f"\nGenerating Phase 1 plots → {OUT}/\n")
    plot_hit_rate_all_configs()
    plot_mrr_all_configs()
    for lbl, cfg in CONFIGS.items():
        plot_summary_heatmap(lbl, cfg)
    plot_memory_comparison()
    plot_update_latency()
    plot_accuracy_vs_memory()
    plot_mrr_violin()
    plot_n_results()

    print(f"\nAll plots saved to: {OUT}/")
    print("Files:")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(f"  {f}")

