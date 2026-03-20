"""
evaluate_retrievers.py – Main evaluation entry-point for the FinalProject.

Mirrors the HW2 evaluate_performance.py structure:

  1. Build each retriever (ClassicRetriever, ANNRetriever, SketchRetriever)
  2. Load queries + qrels ground truth from the MS MARCO dataset
  3. Run the benchmark → DataFrame of per-(retriever, query) metrics
  4. (Optional) Run scaling experiment over different corpus sizes
  5. (Optional) Run streaming/update-time experiment
  6. Save results CSV + all plots + a text summary report

Run from the FinalProject/ directory:
  python evaluate_retrievers.py [--quick] [--corpus-sizes 1000 5000 10000]
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from evaluation.benchmark import (
    load_queries_and_qrels_from_qrels_jsonl,
    run_benchmark, run_complexity_experiment,
)
from evaluation.plots import (
    generate_all_plots, plot_scaling_latency, plot_scaling_metrics,
    plot_update_times, plot_llm_metrics,
    plot_distance_scores, plot_credibility_scores, plot_complexity_curve,
)
from logic.constants import QRELS_JSONL, COLLECTION_JSONL
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.classic_retriever import ClassicRetriever
from retrievers.min_hash_retriever import MinHashRetriever
from retrievers.minhashLSH_retriever import MinHashLSHRetriever

EVAL_DIR = "datasets/evaluations"
TOP_K = 10


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_summary_report(results_df: pd.DataFrame, top_k: int = TOP_K, save_path: str = EVAL_DIR) -> None:
    """Write a human-readable text summary, mirroring HW2's summary_report."""
    os.makedirs(save_path, exist_ok=True)

    metric_cols = [f"precision@{top_k}", f"recall@{top_k}", f"ndcg@{top_k}", "mrr", "ap"]
    metric_cols = [c for c in metric_cols if c in results_df.columns]
    score_cols  = ["distance_score", "credibility_score",
                   "hallucination_flagged", "misinformation_flagged"]
    score_cols  = [c for c in score_cols if c in results_df.columns]
    all_cols = metric_cols + score_cols + ["latency_s", "memory_mb"]

    retrievers = sorted(results_df["retriever"].unique())
    # Build per-retriever stats as plain dicts to avoid multi-level indexing
    stats: dict[str, dict] = {}
    for ret in retrievers:
        sub = results_df[results_df["retriever"] == ret]
        stats[ret] = {col: (float(sub[col].mean()), float(sub[col].std(ddof=0))) for col in all_cols}

    report_path = f"{save_path}/summary_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RETRIEVER EVALUATION SUMMARY  –  MS MARCO\n")
        f.write("=" * 80 + "\n\n")

        for ret in retrievers:
            f.write(f"Retriever: {ret}\n")
            f.write("-" * 40 + "\n")
            for metric in metric_cols:
                mean_val, std_val = stats[ret][metric]
                f.write(f"  {metric:<22}: {mean_val:.4f} ± {std_val:.4f}\n")
            # Distance / Credibility / flags
            for sc in score_cols:
                mean_val, std_val = stats[ret][sc]
                f.write(f"  {sc:<22}: {mean_val:.4f} ± {std_val:.4f}\n")
            lat_mean, lat_std = stats[ret]["latency_s"]
            mem_mean, mem_std = stats[ret]["memory_mb"]
            f.write(f"  {'latency_s':<22}: {lat_mean:.4f} ± {lat_std:.4f} s\n")
            f.write(f"  {'memory_mb':<22}: {mem_mean:.4f} ± {mem_std:.4f} MB\n")
            f.write("\n")

        # Best per metric
        f.write("=" * 80 + "\n")
        f.write("Best Retrievers\n")
        f.write("-" * 80 + "\n")
        for metric in metric_cols:
            best = max(retrievers, key=lambda r: stats[r][metric][0])
            val = stats[best][metric][0]
            f.write(f"  Best {metric:<14}: {best}  ({val:.4f})\n")
        fastest = min(retrievers, key=lambda r: stats[r]["latency_s"][0])
        f.write(f"  Fastest              : {fastest}  ({stats[fastest]['latency_s'][0]:.4f} s)\n")
        f.write("\n")

    print(f"Saved summary report → {report_path}")


# ---------------------------------------------------------------------------
# Scaling experiment
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    queries: list[dict],
    qrels: dict[str, set[str]],
    corpus_sizes: list[int],
    max_queries: int = 50,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Re-build each retriever at increasing corpus sizes and measure quality + speed."""
    rows = []
    for size in corpus_sizes:
        print(f"\n[scaling] corpus_size={size}")

        corpus_docs, size_qrels, size_queries = build_evaluation_corpus(
            qrels=qrels, queries=queries, corpus_size=size, max_queries=max_queries)

        # Build retrievers fresh for each size
        retrievers: dict = {}
        try:
            classic = ClassicRetriever(top_k=top_k, num_initial_documents=0)
            classic.build_corpus_from_docs(corpus_docs)
            retrievers["Classic"] = classic
        except Exception as e:
            print(f"  ClassicRetriever build failed: {e}")

        try:
            bm25 = BM25Retriever(top_k=top_k, num_initial_documents=0)
            bm25.build_corpus_from_docs(corpus_docs)
            retrievers["BM25"] = bm25
        except Exception as e:
            print(f"  BM25Retriever build failed: {e}")

        try:
            mh = MinHashRetriever(dbscan_eps=0.85, corpus_initial_size=0)
            mh.build_corpus_from_docs(corpus_docs)
            retrievers["MinHash"] = mh
        except Exception as e:
            print(f"  MinHashRetriever build failed: {e}")

        bench_df = run_benchmark(retrievers, size_queries, size_qrels,
                                 top_k=top_k, max_queries=len(size_queries))
        bench_df["corpus_size"] = size

        # Aggregate to one row per retriever per size
        agg = bench_df.groupby("retriever").agg(
            latency_s=("latency_s", "mean"),
            mrr=("mrr", "mean"),
        ).reset_index()
        agg["corpus_size"] = size
        rows.append(agg)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Streaming update experiment
# ---------------------------------------------------------------------------

def run_update_experiment(
    retrievers: dict[str, BaseRetriever],
    new_documents: list[dict],
    n_updates: int = 20,
) -> pd.DataFrame:
    """Measure how long each retriever takes per .update() call."""
    rows = []
    docs_to_add = new_documents[:n_updates]

    for name, retriever in retrievers.items():
        for doc in docs_to_add:
            t0 = time.perf_counter()
            try:
                retriever.update(doc)
            except Exception as exc:
                print(f"  [{name}] update failed: {exc}")
            elapsed = time.perf_counter() - t0
            rows.append({"retriever": name, "update_latency_s": elapsed})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------

def run_llm_evaluation(
    retrievers: dict[str, BaseRetriever],
    queries: list[dict],
    corpus_docs: list[dict],
    top_k: int = 5,
    max_queries: int = 20,
    save_path: str = EVAL_DIR,
) -> pd.DataFrame:
    """Run TinyLlama as a relevance judge and answer generator over retrieved results.

    For each retriever × query:
      - Retrieve top_k passages
      - Ask TinyLlama: is each passage relevant?  (LLM-as-judge recall)
      - Ask TinyLlama: generate an answer from the context
      - Score grounding (does answer contain passage n-grams?)

    Returns a DataFrame with one row per (retriever, query).
    """
    from evaluation.llm_evaluator import TinyLlamaEvaluator

    # Build a doc_id → text lookup from the corpus
    doc_lookup: dict[str, str] = {d["key"]: d["data"] for d in corpus_docs}

    evaluator = TinyLlamaEvaluator(n_gpu_layers=1, verbose=False)

    eval_queries = queries[:max_queries]
    rows = []

    print(f"\n[LLM eval] Evaluating {len(eval_queries)} queries × {len(retrievers)} retrievers…")

    for ret_name, retriever in retrievers.items():
        print(f"  → {ret_name}")
        for i, query in enumerate(eval_queries):
            query_text = query["data"].strip()

            # Retrieve
            try:
                result = retriever.retrieve(query_text, top_k=top_k)
            except Exception as exc:
                print(f"    [!] retrieve failed: {exc}")
                result = []

            # Resolve doc IDs to texts
            if result and isinstance(result[0], tuple):
                retrieved_pairs = [(doc_id, doc_lookup.get(doc_id, "")) for doc_id, _ in result]
            else:
                retrieved_pairs = [(doc_id, doc_lookup.get(doc_id, "")) for doc_id in result]

            # LLM judge
            try:
                rag_eval = evaluator.evaluate_rag(query_text, retrieved_pairs, top_k_for_answer=3)
            except Exception as exc:
                print(f"    [!] LLM eval failed: {exc}")
                rag_eval = {
                    "llm_relevant_count": 0,
                    "llm_relevant_frac": 0.0,
                    "llm_relevant_ids": [],
                    "generated_answer": "",
                    "grounded": False,
                }

            rows.append({
                "retriever": ret_name,
                "query_id": query["key"],
                "query_text": query_text,
                "llm_relevant_count": rag_eval["llm_relevant_count"],
                "llm_relevant_frac": rag_eval["llm_relevant_frac"],
                "grounded": int(rag_eval["grounded"]),
                "generated_answer": rag_eval["generated_answer"],
            })

            print(f"\r    {i+1}/{len(eval_queries)}", end="", flush=True)
        print()

    df = pd.DataFrame(rows)
    out = f"{save_path}/llm_evaluation_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved LLM evaluation → {out}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FinalProject retrievers")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast mode: 500 corpus docs, 30 eval queries, skip scaling"
    )
    parser.add_argument(
        "--corpus-size", type=int, default=10_000,
        help="Number of documents to build the initial corpus (default: 10000)"
    )
    parser.add_argument(
        "--max-queries", type=int, default=200,
        help="Max number of queries to evaluate (default: 200)"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Retrieval depth (default: 10)"
    )
    parser.add_argument(
        "--skip-sketch", action="store_true",
        help="Skip SketchRetriever (slow to build)"
    )
    parser.add_argument(
        "--scaling", action="store_true",
        help="Run corpus-size scaling experiment"
    )
    parser.add_argument(
        "--scaling-sizes", type=int, nargs="+",
        default=[1_000, 5_000, 10_000, 50_000],
        help="Corpus sizes for scaling experiment"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Run LLM evaluation (TinyLlama)"
    )
    return parser.parse_args()


def build_evaluation_corpus(
    qrels: dict[str, set[str]],
    queries: list[dict],
    corpus_size: int,
    max_queries: int,
) -> tuple[list[dict[str, str]], dict[str, set[str]], list[dict]]:
    """Build a realistic evaluation corpus with guaranteed relevant passage coverage.

    Strategy:
      1. Take the first `max_queries` queries that have qrel entries.
      2. Collect their relevant passage IDs (must be in corpus).
      3. Fill remaining corpus slots with OTHER qrel passage IDs as topically
         coherent distractors (much harder than random padding).
      4. Scan the collection once to load all needed passages.
      5. Filter to queries whose relevant passage was actually found locally.

    Returns
    -------
    corpus_docs      : list of {"key": passage_id, "data": text}
    filtered_qrels   : qrels restricted to queries with local coverage
    filtered_queries : queries restricted to those with local coverage
    """
    from logic.processing.file_utils import load_jsonl

    # 1. Eval query pool
    eval_queries_pool = [q for q in queries if q["key"] in qrels][:max_queries]

    # 2. Relevant passage IDs for eval queries (priority — must be in corpus)
    priority_ids: set[str] = set()
    for q in eval_queries_pool:
        priority_ids.update(qrels[q["key"]])

    # 3. Distractor pool: all other qrel passages not in priority set
    distractor_ids: set[str] = set()
    for doc_ids in qrels.values():
        distractor_ids.update(doc_ids)
    distractor_ids -= priority_ids

    # Combined target: priority first, then distractors up to corpus_size
    n_distractors = max(0, corpus_size - len(priority_ids))
    target_ids = priority_ids | set(sorted(distractor_ids)[:n_distractors])

    print(f"  Target corpus: {len(priority_ids)} relevant + {min(n_distractors, len(distractor_ids))} distractor passages")

    # 4. Single pass over collection to load all target passages
    found_passages: dict[str, str] = {}
    for doc in load_jsonl(COLLECTION_JSONL):
        key = doc["key"]
        if key in target_ids:
            found_passages[key] = doc["data"].strip()
        if len(found_passages) == len(target_ids):
            break

    found_relevant = len(priority_ids & found_passages.keys())
    print(f"  Found {found_relevant}/{len(priority_ids)} relevant + "
          f"{len(found_passages) - found_relevant} distractor passages in local collection")

    # 5. Filter to queries whose relevant passage was actually found
    filtered_qrels: dict[str, set[str]] = {}
    filtered_queries: list[dict] = []
    for q in eval_queries_pool:
        qid = q["key"]
        local_relevant = qrels[qid] & found_passages.keys()
        if local_relevant:
            filtered_qrels[qid] = local_relevant
            filtered_queries.append(q)

    print(f"  Queries with local coverage: {len(filtered_queries)}/{len(eval_queries_pool)}")

    corpus_docs = [{"key": k, "data": v} for k, v in found_passages.items()]
    print(f"  Final corpus size: {len(corpus_docs)} documents")

    return corpus_docs, filtered_qrels, filtered_queries


def main() -> None:
    args = parse_args()

    if args.quick:
        args.corpus_size = 500
        args.max_queries = 30
        args.skip_sketch = True
        args.scaling = False

    os.makedirs(EVAL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load ground truth — derive both queries AND qrels from qrels.jsonl
    #    (queries.jsonl uses different IDs than qrels.jsonl, so we source
    #     everything from qrels.jsonl for a consistent evaluation set)
    # ------------------------------------------------------------------
    print("[main] Loading queries and qrels from qrels.jsonl…")
    queries, qrels = load_queries_and_qrels_from_qrels_jsonl(QRELS_JSONL)
    print(f"  {len(queries)} queries loaded, {len(qrels)} with relevance judgements")

    # ------------------------------------------------------------------
    # 2. Build corpus (seeded with relevant passages)
    # ------------------------------------------------------------------
    print(f"\n[main] Building evaluation corpus (target size={args.corpus_size})…")
    corpus_docs, eval_qrels, eval_queries = build_evaluation_corpus(
        qrels=qrels,
        queries=queries,
        corpus_size=args.corpus_size,
        max_queries=args.max_queries,
    )

    # ------------------------------------------------------------------
    # 3. Build retrievers from the pre-loaded corpus
    # ------------------------------------------------------------------
    print(f"\n[main] Indexing retrievers over {len(corpus_docs)} documents…")
    retrievers: dict = {}

    classic = ClassicRetriever(top_k=args.top_k, num_initial_documents=0)
    classic.build_corpus_from_docs(corpus_docs)
    retrievers["Classic"] = classic

    try:
        bm25 = BM25Retriever(top_k=args.top_k, num_initial_documents=0)
        bm25.build_corpus_from_docs(corpus_docs)
        retrievers["BM25"] = bm25
    except Exception as e:
        print(f"  BM25Retriever build failed (skipping): {e}")

    try:
        minhash = MinHashLSHRetriever(num_initial_documents=0, top_k=args.top_k)
        minhash.build_corpus_from_docs(corpus_docs)
        retrievers["MinHashLSH"] = minhash
    except Exception as e:
        print(f"  MinHashLSHRetriever build failed (skipping): {e}")

    try:
        minhash_dbscan = MinHashRetriever(dbscan_eps=0.85, corpus_initial_size=0)
        minhash_dbscan.build_corpus_from_docs(corpus_docs)
        retrievers["MinHash"] = minhash_dbscan
    except Exception as e:
        print(f"  MinHashRetriever build failed (skipping): {e}")

    if not args.skip_sketch:
        try:
            from retrievers.sketch_retriever import SketchRetriever
            sketch = SketchRetriever(dbscan_eps=0.85, num_initial_documents=0)
            sketch.build_corpus_from_docs(corpus_docs)
            retrievers["Sketch"] = sketch
        except Exception as e:
            print(f"  SketchRetriever build failed (skipping): {e}")

    # ------------------------------------------------------------------
    # 4. Benchmark
    # ------------------------------------------------------------------
    print(f"\n[main] Running benchmark ({len(eval_queries)} queries, top_k={args.top_k})…")
    results_df = run_benchmark(
        retrievers=retrievers,
        queries=eval_queries,
        qrels=eval_qrels,
        top_k=args.top_k,
        max_queries=len(eval_queries),
    )

    results_df.to_csv(f"{EVAL_DIR}/evaluation_results.csv", index=False)
    print(f"\nSaved results → {EVAL_DIR}/evaluation_results.csv  ({len(results_df)} rows)")

    # ------------------------------------------------------------------
    # 4. Streaming update experiment
    # ------------------------------------------------------------------
    print("\n[main] Running streaming update experiment…")
    # Use documents that come *after* the initial corpus as new arrivals
    from logic.processing.file_utils import load_jsonl
    from logic.constants import COLLECTION_JSONL
    new_docs = []
    for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
        if i < args.corpus_size:
            continue
        new_docs.append(doc)
        if len(new_docs) >= 20:
            break

    update_df = run_update_experiment(retrievers, new_docs, n_updates=20)
    update_df.to_csv(f"{EVAL_DIR}/update_latency.csv", index=False)
    print(f"Saved update timings → {EVAL_DIR}/update_latency.csv")

    # ------------------------------------------------------------------
    # 5. Scaling experiment (optional)
    # ------------------------------------------------------------------
    if args.scaling:
        print("\n[main] Running scaling experiment…")
        scaling_df = run_scaling_experiment(
            queries=queries,
            qrels=qrels,
            corpus_sizes=args.scaling_sizes,
            max_queries=min(args.max_queries, 50),
            top_k=args.top_k,
        )
        scaling_df.to_csv(f"{EVAL_DIR}/scaling_results.csv", index=False)
        plot_scaling_metrics(scaling_df, metric="mrr",                   save_path=EVAL_DIR)
        plot_scaling_metrics(scaling_df, metric=f"ndcg@{args.top_k}",   save_path=EVAL_DIR)
        plot_scaling_latency(scaling_df,                                  save_path=EVAL_DIR)

    # ------------------------------------------------------------------
    # 6. Running-Time Complexity experiment
    # ------------------------------------------------------------------
    print("\n[main] Running complexity experiment…")
    from retrievers.minhashLSH_retriever import MinHashLSHRetriever as _MHR
    from retrievers.sketch_retriever import SketchRetriever as _SKR

    def _make_classic(docs, k):
        r = ClassicRetriever(top_k=k, num_initial_documents=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_bm25(docs, k):
        r = BM25Retriever(top_k=k, num_initial_documents=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_minhash(docs, k):
        r = MinHashRetriever(dbscan_eps=0.85, corpus_initial_size=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_minhash_lsh(docs, k):
        r = _MHR(num_initial_documents=0, top_k=k)
        r.build_corpus_from_docs(docs)
        return r

    def _make_sketch(docs, k):
        r = _SKR(dbscan_eps=0.85, num_initial_documents=0)
        r.build_corpus_from_docs(docs)
        return r

    complexity_factories = {
        "Classic (O(n))": _make_classic,
        "BM25 (O(n))": _make_bm25,
        "MinHash (sub-linear)": _make_minhash,
        "MinHashLSH (sub-linear)": _make_minhash_lsh,
        "Sketch (sub-linear)": _make_sketch,
    }
    complexity_sizes = [100, 500, 1000, 2000, 5000, min(len(corpus_docs), 10000)]
    complexity_sizes = sorted(set(s for s in complexity_sizes if s <= len(corpus_docs)))
    test_query = eval_queries[0]["data"] if eval_queries else "what is the capital of France"

    complexity_df = run_complexity_experiment(
        retriever_factories=complexity_factories,
        corpus_docs=corpus_docs,
        test_query=test_query,
        corpus_sizes=complexity_sizes,
        top_k=args.top_k,
        n_trials=5,
    )
    complexity_df.to_csv(f"{EVAL_DIR}/complexity_results.csv", index=False)
    plot_complexity_curve(complexity_df, save_path=EVAL_DIR)
    print(f"Saved complexity results → {EVAL_DIR}/complexity_results.csv")

    # ------------------------------------------------------------------
    # 7. LLM evaluation (optional)
    # ------------------------------------------------------------------
    if args.llm:
        print("\n[main] Running LLM evaluation…")
        llm_df = run_llm_evaluation(
            retrievers=retrievers,
            queries=queries,
            corpus_docs=corpus_docs,
            top_k=min(5, args.top_k),
            max_queries=min(args.max_queries, 50),
        )
        llm_df.to_csv(f"{EVAL_DIR}/llm_evaluation_results.csv", index=False)
        plot_llm_metrics(llm_df, save_path=EVAL_DIR)

    # ------------------------------------------------------------------
    # 8. Plots & summary
    # ------------------------------------------------------------------
    print("\n[main] Generating plots…")
    generate_all_plots(results_df, top_k=args.top_k, save_path=EVAL_DIR, update_df=update_df)
    plot_update_times(update_df, save_path=EVAL_DIR)
    plot_distance_scores(results_df, save_path=EVAL_DIR)
    plot_credibility_scores(results_df, save_path=EVAL_DIR)

    print("\n[main] Writing summary report…")
    generate_summary_report(results_df, top_k=args.top_k, save_path=EVAL_DIR)

    print("\n" + "=" * 60)
    print("Evaluation complete!  All outputs saved to:", EVAL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()













