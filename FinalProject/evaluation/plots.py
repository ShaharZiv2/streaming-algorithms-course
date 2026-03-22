"""
plots.py – Visualisation helpers for retriever benchmark results.

All functions accept the DataFrame produced by evaluation.benchmark.run_benchmark
and save figures to `save_path`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent colour palette – one colour per retriever name (7 entries for up to 6 sketches)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Bar chart – mean metric comparison across retrievers
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    results_df: pd.DataFrame,
    top_k: int = 10,
    save_path: str = "datasets/evaluations",
) -> None:
    """Bar chart comparing mean ± std of every IR metric per retriever."""
    _ensure_dir(save_path)

    metric_cols = [f"precision@{top_k}", f"recall@{top_k}", f"ndcg@{top_k}", "mrr", "ap"]
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    retrievers = sorted(results_df["retriever"].unique())
    colors = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(retrievers)}

    # Build per-retriever mean/std dict
    stats: dict[str, dict] = {}
    for ret in retrievers:
        sub = results_df[results_df["retriever"] == ret]
        stats[ret] = {m: (float(sub[m].mean()), float(sub[m].std(ddof=0))) for m in metric_cols}

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        means = [stats[r][metric][0] for r in retrievers]
        stds  = [stats[r][metric][1] for r in retrievers]
        x = np.arange(len(retrievers))
        ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
               color=[colors[r] for r in retrievers],
               alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(metric.upper(), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Retriever Metric Comparison (mean ± std)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{save_path}/metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Latency distribution – box/violin plot
# ---------------------------------------------------------------------------

def plot_latency_distribution(
    results_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Box-plot of per-query latency (seconds) for each retriever."""
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=results_df, x="retriever", y="latency_s",
                hue="retriever", palette=PALETTE, legend=False,
                ax=ax, width=0.5, fliersize=3)
    ax.set_xlabel("Retriever", fontsize=11)
    ax.set_ylabel("Latency (s)", fontsize=11)
    ax.set_title("Query Latency Distribution", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/latency_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Latency vs metric scatter (speed-quality trade-off)
# ---------------------------------------------------------------------------

def plot_latency_vs_metric(
    results_df: pd.DataFrame,
    metric: str = "mrr",
    save_path: str = "datasets/evaluations",
) -> None:
    """Scatter: mean latency (x) vs mean metric (y) – one point per retriever."""
    _ensure_dir(save_path)
    if metric not in results_df.columns:
        print(f"Warning: metric '{metric}' not in results, skipping latency_vs_metric plot")
        return

    grouped = results_df.groupby("retriever").agg(
        mean_latency=("latency_s", "mean"),
        mean_metric=(metric, "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in enumerate(grouped.itertuples(index=False)):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(float(row.mean_latency), float(row.mean_metric),
                   s=150, color=color, zorder=3, label=str(row.retriever))
        ax.annotate(str(row.retriever),
                    (float(row.mean_latency), float(row.mean_metric)),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Mean Latency (s)", fontsize=11)
    ax.set_ylabel(f"Mean {metric.upper()}", fontsize=11)
    ax.set_title(f"Speed–Quality Trade-off ({metric.upper()})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/latency_vs_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Memory usage comparison
# ---------------------------------------------------------------------------

def plot_memory_comparison(
    results_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Bar chart of mean peak memory (MB) per retriever during retrieval."""
    _ensure_dir(save_path)

    grouped = results_df.groupby("retriever")["memory_mb"].agg(["mean", "std"]).reset_index()
    retrievers = grouped["retriever"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(retrievers))
    ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, width=0.5,
           color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax.set_title("Peak Retrieval Memory per Query", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/memory_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Per-query metric heatmap (retriever × query)
# ---------------------------------------------------------------------------

def plot_per_query_heatmap(
    results_df: pd.DataFrame,
    metric: str = "mrr",
    max_queries: int = 100,
    save_path: str = "datasets/evaluations",
) -> None:
    """Heatmap of per-query metric values, rows = retrievers, cols = queries."""
    _ensure_dir(save_path)
    if metric not in results_df.columns:
        print(f"Warning: metric '{metric}' not in results, skipping heatmap")
        return

    pivot = results_df.pivot_table(index="retriever", columns="query_id", values=metric)
    pivot = pivot.iloc[:, :max_queries]  # cap columns

    fig, ax = plt.subplots(figsize=(min(20, pivot.shape[1] * 0.2 + 2), 4))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={"label": metric.upper()},
                xticklabels=False)
    ax.set_title(f"Per-query {metric.upper()} (first {max_queries} queries)", fontsize=12)
    ax.set_xlabel("Query", fontsize=10)
    ax.set_ylabel("Retriever", fontsize=10)
    plt.tight_layout()
    out = f"{save_path}/per_query_{metric}_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Corpus-size scaling experiment plots
# ---------------------------------------------------------------------------

def plot_scaling_metrics(
    scaling_df: pd.DataFrame,
    metric: str = "mrr",
    save_path: str = "datasets/evaluations",
) -> None:
    """Line plot: metric vs corpus_size for each retriever.

    `scaling_df` must contain columns: retriever, corpus_size, and the metric.
    """
    _ensure_dir(save_path)
    if metric not in scaling_df.columns:
        print(f"Warning: metric '{metric}' not in scaling_df, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ret in enumerate(scaling_df["retriever"].unique()):
        subset = scaling_df[scaling_df["retriever"] == ret].sort_values("corpus_size")
        ax.plot(subset["corpus_size"], subset[metric],
                marker="o", label=ret, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Corpus Size (# documents)", fontsize=11)
    ax.set_ylabel(f"Mean {metric.upper()}", fontsize=11)
    ax.set_title(f"{metric.upper()} vs Corpus Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/scaling_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_scaling_latency(
    scaling_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Line plot: mean latency vs corpus_size for each retriever."""
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ret in enumerate(scaling_df["retriever"].unique()):
        subset = scaling_df[scaling_df["retriever"] == ret].sort_values("corpus_size")
        ax.plot(subset["corpus_size"], subset["latency_s"],
                marker="s", label=ret, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Corpus Size (# documents)", fontsize=11)
    ax.set_ylabel("Mean Latency (s)", fontsize=11)
    ax.set_title("Query Latency vs Corpus Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/scaling_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Streaming / update-time experiment
# ---------------------------------------------------------------------------

def plot_update_times(
    update_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Bar chart of mean update latency per retriever.

    `update_df` must have columns: retriever, update_latency_s
    """
    _ensure_dir(save_path)

    grouped = update_df.groupby("retriever")["update_latency_s"].agg(["mean", "std"]).reset_index()
    retrievers = grouped["retriever"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(retrievers))
    ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, width=0.5,
           color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Update Latency (s)", fontsize=11)
    ax.set_title("Streaming Update Latency per Document", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{save_path}/update_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 4-panel retriever comparison: retrieve time, update time, credibility, distance
# ---------------------------------------------------------------------------

def plot_retriever_comparison(
    results_df: pd.DataFrame,
    update_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """4-panel side-by-side comparison of all retrievers on MS MARCO.

    Panels
    ------
    1. Mean retrieve time (s) per retriever  – lower is better
    2. Mean update time (s) per retriever    – lower is better
    3. Mean credibility score per retriever  – higher is better
    4. Mean distance score per retriever     – lower is better (closer match)

    Parameters
    ----------
    results_df : DataFrame produced by run_benchmark()
                 (columns: retriever, latency_s, credibility_score, distance_score, …)
    update_df  : DataFrame produced by run_update_experiment()
                 (columns: retriever, update_latency_s)
    save_path  : directory where the PNG will be written
    """
    _ensure_dir(save_path)

    retrievers = sorted(results_df["retriever"].unique())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]
    x = np.arange(len(retrievers))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Retriever Comparison on MS MARCO\n"
        "(retrieve time, update time, credibility score, distance score)",
        fontsize=13, y=1.03,
    )

    def _bar(ax, values, stds, title, ylabel, invert_note=""):
        ax.bar(x, values, yerr=stds, capsize=5, width=0.55,
               color=colors, alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}\n{invert_note}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate values on top of bars
        for xi, (v, s) in enumerate(zip(values, stds)):
            ax.text(xi, v + s + max(values) * 0.01, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.5)

    # --- Panel 1: Retrieve time ---
    ret_means = [float(results_df[results_df["retriever"] == r]["latency_s"].mean()) for r in retrievers]
    ret_stds  = [float(results_df[results_df["retriever"] == r]["latency_s"].std(ddof=0)) for r in retrievers]
    _bar(axes[0], ret_means, ret_stds,
         title="Retrieve Time", ylabel="Mean latency (s)", invert_note="↓ lower is better")

    # --- Panel 2: Update time ---
    upd_retrievers = update_df["retriever"].unique() if update_df is not None else []
    upd_means, upd_stds = [], []
    for r in retrievers:
        if r in upd_retrievers:
            sub = update_df[update_df["retriever"] == r]["update_latency_s"]
            upd_means.append(float(sub.mean()))
            upd_stds.append(float(sub.std(ddof=0)))
        else:
            upd_means.append(0.0)
            upd_stds.append(0.0)
    _bar(axes[1], upd_means, upd_stds,
         title="Update Time", ylabel="Mean update latency (s)", invert_note="↓ lower is better")

    # --- Panel 3: Credibility score ---
    if "credibility_score" in results_df.columns:
        cred_means = [float(results_df[results_df["retriever"] == r]["credibility_score"].mean()) for r in retrievers]
        cred_stds  = [float(results_df[results_df["retriever"] == r]["credibility_score"].std(ddof=0)) for r in retrievers]
        _bar(axes[2], cred_means, cred_stds,
             title="Credibility Score", ylabel="Score (0–100)", invert_note="↑ higher is better")
        from evaluation.metrics import MISINFORMATION_CREDIBILITY_THRESHOLD
        axes[2].axhline(MISINFORMATION_CREDIBILITY_THRESHOLD, color="orange",
                        linestyle="--", linewidth=1.2,
                        label=f"Threshold ({MISINFORMATION_CREDIBILITY_THRESHOLD})")
        axes[2].legend(fontsize=7)
        axes[2].set_ylim(0, 110)
    else:
        axes[2].set_visible(False)

    # --- Panel 4: Distance score ---
    if "distance_score" in results_df.columns:
        dist_means = [float(results_df[results_df["retriever"] == r]["distance_score"].mean()) for r in retrievers]
        dist_stds  = [float(results_df[results_df["retriever"] == r]["distance_score"].std(ddof=0)) for r in retrievers]
        _bar(axes[3], dist_means, dist_stds,
             title="Distance Score", ylabel="Cosine distance (0–1)", invert_note="↓ lower is better")
        from evaluation.metrics import HALLUCINATION_DISTANCE_THRESHOLD
        axes[3].axhline(HALLUCINATION_DISTANCE_THRESHOLD, color="red",
                        linestyle="--", linewidth=1.2,
                        label=f"Hallucination threshold ({HALLUCINATION_DISTANCE_THRESHOLD})")
        axes[3].legend(fontsize=7)
        axes[3].set_ylim(0, 1.1)
    else:
        axes[3].set_visible(False)

    plt.tight_layout()
    out = f"{save_path}/retriever_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Sketch vs Baseline comparison: retrieve time, memory, baseline precision
# ---------------------------------------------------------------------------

def plot_sketch_vs_baseline(
    results_df: pd.DataFrame,
    baseline_names: tuple[str, ...] = ("Classic", "BM25"),
    save_path: str = "datasets/evaluations",
) -> None:
    """3-panel bar chart comparing sketch retrievers to the Classic+BM25 baseline.

    Panels
    ------
    1. Mean retrieve time (s)       – all retrievers, lower is better
    2. Mean peak memory (MB)        – all retrievers, lower is better
    3. Baseline precision           – fraction of each retriever's results that
                                      overlap with the combined Classic+BM25 result
                                      set. Baseline retrievers are always 1.0.
                                      Sketch retrievers show how well they
                                      approximate the baseline.

    Parameters
    ----------
    results_df     : DataFrame from run_benchmark() — must contain
                     latency_s, memory_mb, baseline_precision columns
    baseline_names : retriever names that form the baseline (shown in grey)
    save_path      : output directory
    """
    _ensure_dir(save_path)
    if "baseline_precision" not in results_df.columns:
        print("Warning: baseline_precision not in results, skipping sketch_vs_baseline plot")
        return

    retrievers = sorted(results_df["retriever"].unique())
    # Baselines grey, sketch retrievers coloured
    colors = []
    sketch_color_iter = iter(PALETTE)
    for r in retrievers:
        if r in baseline_names:
            colors.append("#AAAAAA")
        else:
            colors.append(next(sketch_color_iter, "#333333"))

    x = np.arange(len(retrievers))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Sketch Retrievers vs Baseline (Classic + BM25)\n"
        "on Retrieve Time, Memory, and Baseline Precision",
        fontsize=13, y=1.03,
    )

    def _bar(ax, values, stds, title, ylabel, note=""):
        bars = ax.bar(x, values, yerr=stds, capsize=5, width=0.55,
                      color=colors, alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}\n{note}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, (v, s) in enumerate(zip(values, stds)):
            ax.text(xi, v + s + max(values) * 0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.5)

    # Panel 1: retrieve time
    lat_means = [float(results_df[results_df["retriever"] == r]["latency_s"].mean()) for r in retrievers]
    lat_stds  = [float(results_df[results_df["retriever"] == r]["latency_s"].std(ddof=0)) for r in retrievers]
    _bar(axes[0], lat_means, lat_stds, "Retrieve Time", "Mean latency (s)", "↓ lower is better")

    # Panel 2: memory
    mem_means = [float(results_df[results_df["retriever"] == r]["memory_mb"].mean()) for r in retrievers]
    mem_stds  = [float(results_df[results_df["retriever"] == r]["memory_mb"].std(ddof=0)) for r in retrievers]
    _bar(axes[1], mem_means, mem_stds, "Peak Memory", "Mean memory (MB)", "↓ lower is better")

    # Panel 3: baseline precision
    bp_means = [float(results_df[results_df["retriever"] == r]["baseline_precision"].mean()) for r in retrievers]
    bp_stds  = [float(results_df[results_df["retriever"] == r]["baseline_precision"].std(ddof=0)) for r in retrievers]
    _bar(axes[2], bp_means, bp_stds, "Baseline Precision",
         "Fraction overlapping with Classic+BM25", "↑ higher = closer to baseline")
    axes[2].set_ylim(0, 1.15)

    # Legend: grey = baseline, colour = sketch
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#AAAAAA", edgecolor="black", label="Baseline (Classic / BM25)")]
    axes[2].legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = f"{save_path}/sketch_vs_baseline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# MinHash vs Baseline focused comparison
# ---------------------------------------------------------------------------

def plot_minhash_vs_baseline(
    results_df: pd.DataFrame,
    sketch_names: tuple[str, ...] = ("MinHashDBSCAN", "MinHashLSH", "ProbMinHashDBSCAN", "ProbMinHashLSH"),
    baseline_names: tuple[str, ...] = ("Classic", "BM25"),
    save_path: str = "datasets/evaluations",
) -> None:
    """4-panel bar chart: MinHash family vs Classic+BM25 baseline.

    Panels
    ------
    1. Mean retrieval time (s)      – lower is better
    2. Mean peak memory (MB)        – lower is better
    3. Precision vs Classic         – fraction of each retriever's results that
                                      also appear in Classic's result set.
                                      Classic itself is skipped (would be 1.0).
    4. Precision vs BM25            – same, but against BM25's result set.
                                      BM25 itself is skipped.

    LSH retrievers (MinHashLSH, ProbMinHashLSH) return ALL results without
    a fixed top_k — their precision/recall metrics are computed over the full
    returned set, so they are marked with an asterisk (*) in the x-axis labels.
    """
    _ensure_dir(save_path)

    LSH_NAMES = {"MinHashLSH", "ProbMinHashLSH"}

    # Order: baselines first (grey), then sketches (coloured)
    all_retrievers = sorted(results_df["retriever"].unique())
    ordered = (
        [r for r in all_retrievers if r in baseline_names] +
        [r for r in all_retrievers if r in sketch_names] +
        [r for r in all_retrievers if r not in baseline_names and r not in sketch_names]
    )

    bar_colors = []
    sketch_palette = iter(PALETTE)
    for r in ordered:
        if r in baseline_names:
            bar_colors.append("#AAAAAA")
        else:
            bar_colors.append(next(sketch_palette, "#333333"))

    # X-tick labels: mark LSH retrievers with * to indicate "all results"
    x_labels = [f"{r}*" if r in LSH_NAMES else r for r in ordered]

    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(
        "MinHash Sketch Retrievers vs Baseline  —  "
        "Retrieval Time  |  Peak Memory  |  Precision vs Classic  |  Precision vs BM25\n"
        "(*LSH retrievers return all matching docs — metrics evaluated on full result set)",
        fontsize=12, y=1.03,
    )

    def _bar_panel(ax, values, stds, title, ylabel, note="", ylim=None, skip_label="N/A (self)"):
        """Draw a bar panel; NaN values (self-comparison) shown as hatched."""
        for xi, (v, s, c) in enumerate(zip(values, stds, bar_colors)):
            if np.isnan(v):
                ax.bar(xi, 0.05, width=0.55, color="white", edgecolor="black",
                       hatch="////", alpha=0.5)
                ax.text(xi, 0.07, skip_label, ha="center", va="bottom",
                        fontsize=6.5, color="grey", style="italic")
            else:
                ax.bar(xi, v, yerr=s, capsize=5, width=0.55,
                       color=c, alpha=0.87, edgecolor="black")
                max_v = max((vv for vv in values if not np.isnan(vv)), default=0)
                label_y = v + s + max_v * 0.03
                ax.text(xi, label_y, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}\n{note}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    # Panel 1: Retrieval time
    lat_means = [float(results_df[results_df["retriever"] == r]["latency_s"].mean()) for r in ordered]
    lat_stds  = [float(results_df[results_df["retriever"] == r]["latency_s"].std(ddof=0)) for r in ordered]
    _bar_panel(axes[0], lat_means, lat_stds, "Retrieval Time", "Mean latency (s)", "↓ lower is better")

    # Panel 2: Peak memory
    mem_means = [float(results_df[results_df["retriever"] == r]["memory_mb"].mean()) for r in ordered]
    mem_stds  = [float(results_df[results_df["retriever"] == r]["memory_mb"].std(ddof=0)) for r in ordered]
    _bar_panel(axes[1], mem_means, mem_stds, "Peak Memory", "Mean memory (MB)", "↓ lower is better")

    # Panel 3: Precision vs Classic
    def _prec_col(col, r):
        if col not in results_df.columns:
            return float("nan"), 0.0
        sub = results_df[results_df["retriever"] == r][col].dropna()
        return (float(sub.mean()), float(sub.std(ddof=0))) if len(sub) else (float("nan"), 0.0)

    classic_means = [_prec_col("vs_Classic", r)[0] for r in ordered]
    classic_stds  = [_prec_col("vs_Classic", r)[1] for r in ordered]
    _bar_panel(axes[2], classic_means, classic_stds,
               "Precision vs Classic", "Fraction overlapping Classic results",
               "↑ higher = closer to Classic", ylim=(0, 1.25))

    # Panel 4: Precision vs BM25
    bm25_means = [_prec_col("vs_BM25", r)[0] for r in ordered]
    bm25_stds  = [_prec_col("vs_BM25", r)[1] for r in ordered]
    _bar_panel(axes[3], bm25_means, bm25_stds,
               "Precision vs BM25", "Fraction overlapping BM25 results",
               "↑ higher = closer to BM25", ylim=(0, 1.25))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#AAAAAA", edgecolor="black", label="Baseline retriever"),
        Patch(facecolor=PALETTE[0], edgecolor="black", label="Sketch retriever"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="Self (skipped)"),
    ]
    axes[3].legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = f"{save_path}/minhash_vs_baseline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Convenience: generate all standard plots at once
# ---------------------------------------------------------------------------

def generate_all_plots(
    results_df: pd.DataFrame,
    top_k: int = 10,
    save_path: str = "datasets/evaluations",
    update_df: pd.DataFrame | None = None,
) -> None:
    """Call every standard plot function on the benchmark results DataFrame."""
    plot_metrics_comparison(results_df, top_k=top_k, save_path=save_path)
    plot_latency_distribution(results_df, save_path=save_path)
    plot_latency_vs_metric(results_df, metric="mrr", save_path=save_path)
    plot_latency_vs_metric(results_df, metric=f"ndcg@{top_k}", save_path=save_path)
    plot_memory_comparison(results_df, save_path=save_path)
    plot_per_query_heatmap(results_df, metric="mrr", save_path=save_path)
    # Focused MinHash vs Baseline comparison (time, memory, baseline precision)
    plot_minhash_vs_baseline(
        results_df,
        sketch_names=("MinHashDBSCAN", "MinHashLSH", "ProbMinHashDBSCAN", "ProbMinHashLSH"),
        save_path=save_path,
    )
    if update_df is not None and not update_df.empty:
        plot_retriever_comparison(results_df, update_df, save_path=save_path)


# ---------------------------------------------------------------------------
# LLM evaluation plots
# ---------------------------------------------------------------------------

def plot_llm_metrics(
    llm_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Bar charts for TinyLlama-as-judge metrics per retriever.

    Plots:
      - Mean LLM-relevant fraction (how often the LLM found retrieved passages useful)
      - Mean grounding rate (how often the generated answer is grounded in passages)
    """
    _ensure_dir(save_path)

    retrievers = sorted(llm_df["retriever"].unique())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]
    x = np.arange(len(retrievers))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 1. LLM-relevant fraction
    rel_means = [float(llm_df[llm_df["retriever"] == r]["llm_relevant_frac"].mean()) for r in retrievers]
    rel_stds  = [float(llm_df[llm_df["retriever"] == r]["llm_relevant_frac"].std(ddof=0)) for r in retrievers]
    axes[0].bar(x, rel_means, yerr=rel_stds, capsize=5, width=0.5,
                color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[0].set_ylabel("Fraction of passages judged relevant", fontsize=10)
    axes[0].set_title("TinyLlama Relevance Judge\n(mean ± std)", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)

    # 2. Grounding rate
    grnd_means = [float(llm_df[llm_df["retriever"] == r]["grounded"].mean()) for r in retrievers]
    grnd_stds  = [float(llm_df[llm_df["retriever"] == r]["grounded"].std(ddof=0)) for r in retrievers]
    axes[1].bar(x, grnd_means, yerr=grnd_stds, capsize=5, width=0.5,
                color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[1].set_ylabel("Fraction of answers grounded", fontsize=10)
    axes[1].set_title("TinyLlama Answer Grounding Rate\n(mean ± std)", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("LLM-as-Judge Evaluation (TinyLlama)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{save_path}/llm_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Distance Score plots
# ---------------------------------------------------------------------------

def plot_distance_scores(
    results_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Two-panel plot for the Distance Score metric.

    Panel 1 – Bar chart: mean distance score per retriever (lower = better).
    Panel 2 – Bar chart: hallucination flag rate per retriever.

    The distance score measures cosine distance between the query embedding
    and the top-1 retrieved passage embedding.  A score above the threshold
    (~0.7) means the passage is semantically far from the query and may
    produce a hallucinated answer.
    """
    _ensure_dir(save_path)
    if "distance_score" not in results_df.columns:
        print("Warning: distance_score not in results, skipping distance plot")
        return

    retrievers = sorted(results_df["retriever"].unique())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]
    x = np.arange(len(retrievers))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: mean distance score
    means = [float(results_df[results_df["retriever"] == r]["distance_score"].mean()) for r in retrievers]
    stds  = [float(results_df[results_df["retriever"] == r]["distance_score"].std(ddof=0)) for r in retrievers]
    bars = axes[0].bar(x, means, yerr=stds, capsize=5, width=0.5,
                       color=colors, alpha=0.85, edgecolor="black")
    from evaluation.metrics import HALLUCINATION_DISTANCE_THRESHOLD
    axes[0].axhline(HALLUCINATION_DISTANCE_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.5, label=f"Hallucination threshold ({HALLUCINATION_DISTANCE_THRESHOLD})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[0].set_ylabel("Cosine Distance (lower = more similar)", fontsize=10)
    axes[0].set_title("Distance Score\n(query ↔ top-1 passage)", fontsize=11)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Panel 2: hallucination flag rate
    flag_rates = [float(results_df[results_df["retriever"] == r]["hallucination_flagged"].mean()) for r in retrievers]
    axes[1].bar(x, flag_rates, width=0.5, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[1].set_ylabel("Fraction of queries flagged", fontsize=10)
    axes[1].set_title("Hallucination Flag Rate\n(distance > threshold)", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distance Score – Hallucination Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{save_path}/distance_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Credibility Score plots
# ---------------------------------------------------------------------------

def plot_credibility_scores(
    results_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Two-panel plot for the Credibility Score metric.

    Panel 1 – Bar chart: mean credibility score per retriever (higher = better).
    Panel 2 – Bar chart: misinformation flag rate per retriever.

    The credibility score is a lexical heuristic (0-100) measuring how much
    the top-1 retrieved passage resembles text from trusted academic or news
    sources vs known misinformation patterns.
    """
    _ensure_dir(save_path)
    if "credibility_score" not in results_df.columns:
        print("Warning: credibility_score not in results, skipping credibility plot")
        return

    retrievers = sorted(results_df["retriever"].unique())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(retrievers))]
    x = np.arange(len(retrievers))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: mean credibility
    means = [float(results_df[results_df["retriever"] == r]["credibility_score"].mean()) for r in retrievers]
    stds  = [float(results_df[results_df["retriever"] == r]["credibility_score"].std(ddof=0)) for r in retrievers]
    axes[0].bar(x, means, yerr=stds, capsize=5, width=0.5,
                color=colors, alpha=0.85, edgecolor="black")
    from evaluation.metrics import MISINFORMATION_CREDIBILITY_THRESHOLD
    axes[0].axhline(MISINFORMATION_CREDIBILITY_THRESHOLD, color="orange", linestyle="--",
                    linewidth=1.5, label=f"Misinformation threshold ({MISINFORMATION_CREDIBILITY_THRESHOLD})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[0].set_ylabel("Credibility Score (higher = more credible)", fontsize=10)
    axes[0].set_title("Credibility Score\n(top-1 retrieved passage)", fontsize=11)
    axes[0].set_ylim(0, 110)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Panel 2: misinformation flag rate
    flag_rates = [float(results_df[results_df["retriever"] == r]["misinformation_flagged"].mean()) for r in retrievers]
    axes[1].bar(x, flag_rates, width=0.5, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(retrievers, rotation=15, ha="right", fontsize=10)
    axes[1].set_ylabel("Fraction of queries flagged", fontsize=10)
    axes[1].set_title("Misinformation Flag Rate\n(credibility < threshold)", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Credibility Score – Misinformation Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{save_path}/credibility_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Running-Time Complexity plot
# ---------------------------------------------------------------------------

def plot_complexity_curve(
    complexity_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """Log-log plot of retrieval latency vs corpus size with O(n) reference line.

    This plot visually proves the sub-linear o(n) retrieval time of the ANN
    retriever vs the linear O(n) Classic retriever.

    `complexity_df` must have columns: retriever, corpus_size,
    mean_latency_s, std_latency_s.
    """
    _ensure_dir(save_path)
    if complexity_df.empty:
        print("Warning: complexity_df is empty, skipping complexity plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    retrievers = sorted(complexity_df["retriever"].unique())

    for ax, use_log in zip(axes, [False, True]):
        for i, ret in enumerate(retrievers):
            sub = complexity_df[complexity_df["retriever"] == ret].sort_values("corpus_size")
            ax.errorbar(
                sub["corpus_size"], sub["mean_latency_s"] * 1000,
                yerr=sub["std_latency_s"] * 1000,
                marker="o", label=ret, color=PALETTE[i % len(PALETTE)],
                linewidth=2, capsize=4,
            )

        # O(n) reference line based on the Classic retriever's first point
        classic_sub = complexity_df[complexity_df["retriever"].str.contains("Classic", case=False)]
        if not classic_sub.empty:
            classic_sub = classic_sub.sort_values("corpus_size")
            n0 = float(classic_sub["corpus_size"].iloc[0])
            t0 = float(classic_sub["mean_latency_s"].iloc[0]) * 1000
            ns = np.array(sorted(complexity_df["corpus_size"].unique()), dtype=float)
            ax.plot(ns, t0 * ns / n0, "k--", linewidth=1.2, alpha=0.6, label="O(n) reference")

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Running-Time Complexity (log-log scale)", fontsize=11)
        else:
            ax.set_title("Running-Time Complexity (linear scale)", fontsize=11)

        ax.set_xlabel("Corpus Size (n documents)", fontsize=10)
        ax.set_ylabel("Mean Query Latency (ms)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Retrieval Latency vs Corpus Size\n"
        "Classic = O(n)  |  ANN = sub-linear o(n)",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()
    out = f"{save_path}/complexity_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Dimensionality Reduction Tradeoff Plot
# ---------------------------------------------------------------------------

def plot_dim_reduction_tradeoff(
    dim_df: pd.DataFrame,
    save_path: str = "datasets/evaluations",
) -> None:
    """4-panel figure: sketch quality/speed/memory vs num_perm (sketch dimension).

    Panels
    ------
    1. Jaccard estimation error vs num_perm  (accuracy; ↓ better; theory: 1/√k)
    2. Baseline precision vs BM25 vs num_perm (retrieval quality; ↑ better)
    3. Mean retrieval latency vs num_perm     (speed; ↓ better)
    4. Memory per document vs num_perm        (efficiency; ↓ better)

    Expected DataFrame columns
    --------------------------
    retriever, num_perm, jaccard_error, prec_vs_bm25, latency_s, memory_mb_per_doc
    """
    _ensure_dir(save_path)

    retrievers = sorted(dim_df["retriever"].unique())
    colors = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(retrievers)}
    markers = ["o", "s", "^", "D", "v", "P"]
    marker_map = {r: markers[i % len(markers)] for i, r in enumerate(retrievers)}

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        "MinHash Sketching — Dimensionality Reduction Tradeoff\n"
        "num_perm controls sketch size: higher = more accurate but slower & larger",
        fontsize=12, y=1.04,
    )

    panels = [
        ("jaccard_error",      "Jaccard Estimation Error\n(|estimated − true|, ↓ better)",
         "Mean |Ĵ − J|",         False, True),
        ("prec_vs_bm25",       "Retrieval Quality vs BM25\n(baseline precision, ↑ better)",
         "Fraction overlapping BM25",   False, False),
        ("latency_s",          "Query Latency\n(↓ better)",
         "Mean latency (s)",            False, False),
        ("memory_mb_per_doc",  "Memory per Document\n(↓ better)",
         "MB per document",             False, False),
    ]

    for ax, (col, title, ylabel, logx, show_theory) in zip(axes, panels):
        for ret in retrievers:
            sub = dim_df[dim_df["retriever"] == ret].sort_values("num_perm")
            if col not in sub.columns or sub[col].isna().all():
                continue
            ax.plot(sub["num_perm"], sub[col],
                    color=colors[ret], marker=marker_map[ret],
                    linewidth=1.8, markersize=7, label=ret)

        # Theoretical 1/√k curve for Jaccard error panel
        if show_theory and "jaccard_error" in dim_df.columns:
            perms = np.array(sorted(dim_df["num_perm"].unique()))
            theory = 1.0 / np.sqrt(perms)
            # Scale to match observed magnitude
            obs = dim_df.groupby("num_perm")["jaccard_error"].mean()
            if len(obs) > 0:
                scale = obs.iloc[0] / (1.0 / np.sqrt(perms[0])) if perms[0] > 0 else 1.0
            else:
                scale = 0.5
            ax.plot(perms, theory * scale, "k--", linewidth=1.2,
                    alpha=0.6, label="Theory: 1/√k")

        ax.set_xlabel("num_perm (sketch dimension)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sorted(dim_df["num_perm"].unique()))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"{save_path}/dim_reduction_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

