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

# Consistent colour palette – one colour per retriever name
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


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
# Convenience: generate all standard plots at once
# ---------------------------------------------------------------------------

def generate_all_plots(
    results_df: pd.DataFrame,
    top_k: int = 10,
    save_path: str = "datasets/evaluations",
) -> None:
    """Call every standard plot function on the benchmark results DataFrame."""
    plot_metrics_comparison(results_df, top_k=top_k, save_path=save_path)
    plot_latency_distribution(results_df, save_path=save_path)
    plot_latency_vs_metric(results_df, metric="mrr", save_path=save_path)
    plot_latency_vs_metric(results_df, metric=f"ndcg@{top_k}", save_path=save_path)
    plot_memory_comparison(results_df, save_path=save_path)
    plot_per_query_heatmap(results_df, metric="mrr", save_path=save_path)


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






