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
    plot_sketch_vs_baseline, plot_minhash_vs_baseline, plot_dim_reduction_tradeoff,
)
from logic.constants import QRELS_JSONL, COLLECTION_JSONL
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.classic_retriever import ClassicRetriever
from retrievers.min_hash_dbscan_retriever import MinHashDbscanRetriever
from retrievers.min_hash_lsh_retriever import MinHashLshRetriever
from retrievers.prob_min_hash_dbscan_retriever import ProbMinHashDbscanRetriever
from retrievers.prob_min_hash_lsh_retriever import ProbMinHashLshRetriever

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
    score_cols  = ["baseline_precision", "vs_Classic", "vs_BM25",
                   "distance_score", "credibility_score",
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

        # ------------------------------------------------------------------
        # Focused: MinHash family vs Baseline (the 3 core comparison metrics)
        # ------------------------------------------------------------------
        SKETCH_NAMES = ("MinHashDBSCAN", "MinHashLSH", "ProbMinHashDBSCAN", "ProbMinHashLSH")
        BASELINE_NAMES = ("Classic", "BM25")
        sketch_retrievers  = [r for r in retrievers if r in SKETCH_NAMES]
        baseline_retrievers = [r for r in retrievers if r in BASELINE_NAMES]

        if sketch_retrievers and baseline_retrievers:
            f.write("=" * 80 + "\n")
            f.write("MinHash Sketch Retrievers vs Baseline (Classic + BM25)\n")
            f.write("Focus: Retrieval Time  |  Peak Memory  |  Prec vs Classic  |  Prec vs BM25\n")
            f.write("=" * 80 + "\n\n")

            header = (f"  {'Retriever':<18}  {'Latency (s)':>14}  {'Memory (MB)':>14}"
                      f"  {'vs Classic':>12}  {'vs BM25':>12}\n")
            f.write(header)
            f.write("  " + "-" * (len(header) - 3) + "\n")

            for group_label, group in [("Baseline", baseline_retrievers), ("Sketch", sketch_retrievers)]:
                f.write(f"  --- {group_label} ---\n")
                for ret in group:
                    lat_mean, lat_std = stats[ret]["latency_s"]
                    mem_mean, mem_std = stats[ret]["memory_mb"]

                    def _fmt_prec(col):
                        if col in stats[ret]:
                            m, s = stats[ret][col]
                            return f"{m:.3f}±{s:.3f}" if not (m != m) else "  N/A (self)"
                        return "     N/A"

                    f.write(
                        f"  {ret:<18}  "
                        f"{lat_mean:>7.4f}±{lat_std:<5.4f}  "
                        f"{mem_mean:>7.4f}±{mem_std:<5.4f}  "
                        f"{_fmt_prec('vs_Classic'):>14}  "
                        f"{_fmt_prec('vs_BM25'):>14}\n"
                    )
            f.write("\n")

            # Speed-up / memory reduction ratios vs each baseline
            for baseline in baseline_retrievers:
                b_lat = stats[baseline]["latency_s"][0]
                b_mem = stats[baseline]["memory_mb"][0]
                f.write(f"  Ratios vs {baseline}:\n")
                for sketch in sketch_retrievers:
                    s_lat = stats[sketch]["latency_s"][0]
                    s_mem = stats[sketch]["memory_mb"][0]
                    col = f"vs_{baseline}"
                    s_prec = stats[sketch].get(col, (float("nan"), 0.0))[0]
                    speedup = b_lat / s_lat if s_lat > 0 else float("inf")
                    mem_ratio = s_mem / b_mem if b_mem > 0 else float("inf")
                    prec_str = f"{s_prec:.4f}" if s_prec == s_prec else "N/A"
                    f.write(
                        f"    {sketch:<16}: "
                        f"speed-up={speedup:.2f}x  "
                        f"mem_ratio={mem_ratio:.2f}x  "
                        f"prec_vs_{baseline}={prec_str}\n"
                    )
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
            mh = MinHashDbscanRetriever(dbscan_eps=0.85, corpus_initial_size=0)
            mh.build_corpus_from_docs(corpus_docs)
            retrievers["MinHashDBSCAN"] = mh
        except Exception as e:
            print(f"  MinHashDbscanRetriever build failed: {e}")

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



# ---------------------------------------------------------------------------
# Dimensionality Reduction Experiment
# ---------------------------------------------------------------------------

def run_dim_reduction_experiment(
    corpus_docs: list[dict],
    eval_queries: list[dict],
    eval_qrels: dict[str, set[str]],
    baseline_retrievers: dict,          # pre-built Classic + BM25
    num_perms: list[int] | None = None,
    top_k: int = TOP_K,
    n_jaccard_pairs: int = 500,
) -> pd.DataFrame:
    """Rebuild MinHash/ProbMinHash retrievers at each num_perm and measure tradeoffs.

    For each num_perm value, this function:
      1. Computes the true Jaccard similarity for n_jaccard_pairs random document pairs
         using the full vocabulary (exact ground truth).
      2. Rebuilds MinHashLSH and ProbMinHashLSH with that num_perm.
      3. Estimates Jaccard via MinHash dot product (hashvalues agreement fraction).
      4. Records: jaccard_error, baseline precision vs BM25, latency, memory/doc.

    This directly shows the dimension→accuracy→speed tradeoff that is the core
    theoretical contribution of MinHash sketching as dimensionality reduction.

    Parameters
    ----------
    corpus_docs        : list of {"key": str, "data": str} dicts (same as main eval)
    eval_queries       : list of {"key": str, "data": str} query dicts
    eval_qrels         : dict of qid → set of relevant passage IDs
    baseline_retrievers: dict with keys "BM25" and "Classic" already built
    num_perms          : sketch dimensions to sweep; default [16, 32, 64, 128, 256, 512]
    top_k              : retrieval depth for benchmark
    n_jaccard_pairs    : number of random doc pairs to measure Jaccard error on

    Returns
    -------
    DataFrame with columns: retriever, num_perm, jaccard_error, prec_vs_bm25,
                             latency_s, memory_mb_per_doc, vocab_dim, reduction_factor
    """
    import random
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from datasketch import MinHash, MinHashLSH
    from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
    from logic.constants import SEED
    from logic.prob_min_hash import ProbMinHash4
    from evaluation.benchmark import run_benchmark

    if num_perms is None:
        num_perms = [16, 32, 64, 128, 256, 512]

    texts = [d["data"].strip() for d in corpus_docs]
    doc_ids = [d["key"] for d in corpus_docs]
    n_docs = len(texts)

    print(f"\n[dim_reduction] Corpus: {n_docs} docs, sweeping num_perm={num_perms}")

    # ------------------------------------------------------------------
    # Step 1: Compute exact Jaccard ground truth (full TF-IDF vocabulary)
    # for n_jaccard_pairs random pairs — this is the reference we compare
    # sketches against at each num_perm level.
    # ------------------------------------------------------------------
    print(f"  Building exact vocab for Jaccard ground truth…")
    exact_vec = CountVectorizer(
        ngram_range=(1, 2),
        tokenizer=StemmingTokenizer(),
        stop_words=stemmed_stop_words(),
        binary=True,
    )
    exact_mat = exact_vec.fit_transform(texts)   # (n_docs × vocab), binary sparse
    vocab_dim = exact_mat.shape[1]
    print(f"  Vocabulary dimension: {vocab_dim:,} features")

    rng = random.Random(SEED)
    pair_indices = [(rng.randint(0, n_docs - 1), rng.randint(0, n_docs - 1))
                    for _ in range(n_jaccard_pairs)]
    # Filter out same-doc pairs
    pair_indices = [(i, j) for i, j in pair_indices if i != j][:n_jaccard_pairs]

    def _exact_jaccard(i: int, j: int) -> float:
        a = exact_mat[i]
        b = exact_mat[j]
        inter = float(a.minimum(b).sum())
        union = float(a.maximum(b).sum())
        return inter / union if union > 0 else 0.0

    print(f"  Computing {len(pair_indices)} exact Jaccard pairs…")
    exact_jaccards = [_exact_jaccard(i, j) for i, j in pair_indices]

    # ------------------------------------------------------------------
    # Step 2: Also build TF-IDF for ProbMinHash ground truth
    # ------------------------------------------------------------------
    tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        tokenizer=StemmingTokenizer(),
        stop_words=stemmed_stop_words(),
    )
    tfidf_mat = tfidf_vec.fit_transform(texts)

    # Pre-compute BM25 baseline results for prec_vs_bm25 calculation
    bm25_results: dict[str, list[str]] = {}
    if "BM25" in baseline_retrievers:
        bm25_ret = baseline_retrievers["BM25"]
        for q in eval_queries:
            try:
                res = bm25_ret.retrieve(q["data"].strip(), top_k=top_k)
                if res and isinstance(res[0], tuple):
                    bm25_results[q["key"]] = [d for d, _ in res]
                else:
                    bm25_results[q["key"]] = list(res)
            except Exception:
                bm25_results[q["key"]] = []

    rows = []

    for num_perm in num_perms:
        print(f"\n  [num_perm={num_perm}] Building MinHash + ProbMinHash signatures…")

        # --- MinHash signatures (CountVectorizer n-grams) ---
        mh_sigs = []
        for text in texts:
            try:
                data = [text]
                cv = CountVectorizer(
                    ngram_range=(1, 2),
                    tokenizer=StemmingTokenizer(),
                    stop_words=stemmed_stop_words(),
                )
                cv.fit_transform(data)
                ngrams = cv.get_feature_names_out()
                m = MinHash(num_perm=num_perm, seed=SEED)
                for gram in ngrams:
                    m.update(gram.encode("utf-8"))
                mh_sigs.append(m.hashvalues.copy())
            except Exception:
                mh_sigs.append(np.zeros(num_perm, dtype=np.uint64))
        mh_sigs = np.array(mh_sigs, dtype=np.float64)

        # --- ProbMinHash signatures (TF-IDF weighted) ---
        pmh_sigs = []
        for i in range(n_docs):
            start, end = tfidf_mat.indptr[i], tfidf_mat.indptr[i + 1]
            keys = tfidf_mat.indices[start:end]
            weights = tfidf_mat.data[start:end]
            pmh = ProbMinHash4(num_perm=num_perm, seed=SEED)
            if len(keys) > 0:
                pmh.fit(keys, weights)
                pmh_sigs.append(np.array(pmh.hashvalues, dtype=np.float64))
            else:
                pmh_sigs.append(np.zeros(num_perm, dtype=np.float64))
        pmh_sigs = np.array(pmh_sigs, dtype=np.float64)

        # --- Jaccard estimation error ---
        # MinHash estimated Jaccard = fraction of matching hash values
        mh_estimated = [
            float(np.mean(mh_sigs[i] == mh_sigs[j]))
            for i, j in pair_indices
        ]
        mh_errors = [abs(est - exact) for est, exact in zip(mh_estimated, exact_jaccards)]
        mh_jaccard_error = float(np.mean(mh_errors))

        pmh_estimated = [
            float(np.mean(pmh_sigs[i] == pmh_sigs[j]))
            for i, j in pair_indices
        ]
        pmh_errors = [abs(est - exact) for est, exact in zip(pmh_estimated, exact_jaccards)]
        pmh_jaccard_error = float(np.mean(pmh_errors))

        # --- Memory per doc ---
        # Each signature is num_perm × 8 bytes (uint64)
        bytes_per_doc = num_perm * 8
        mb_per_doc = bytes_per_doc / (1024 ** 2)

        # --- Build LSH index and measure retrieval latency + baseline precision ---
        for retriever_name, sigs, jaccard_error in [
            ("MinHashLSH", mh_sigs, mh_jaccard_error),
            ("ProbMinHashLSH", pmh_sigs, pmh_jaccard_error),
        ]:
            print(f"    Building {retriever_name} index (threshold=0.15)…")
            # Build LSH index
            lsh_index = MinHashLSH(threshold=0.15, num_perm=num_perm)
            with lsh_index.insertion_session() as session:
                for doc_id, sig in zip(doc_ids, sigs):
                    try:
                        mh_obj = MinHash(num_perm=num_perm)
                        mh_obj.hashvalues = sig.astype(np.uint64)
                        session.insert(doc_id, mh_obj)
                    except Exception:
                        pass

            # Measure retrieval latency + baseline precision on eval queries
            import tracemalloc
            latencies = []
            precisions_vs_bm25 = []

            n_eval = min(len(eval_queries), 100)  # cap at 100 queries for speed
            for q in eval_queries[:n_eval]:
                query_text = q["data"].strip()
                qid = q["key"]

                # Build query MinHash
                if retriever_name == "MinHashLSH":
                    try:
                        cv = CountVectorizer(
                            ngram_range=(1, 2),
                            tokenizer=StemmingTokenizer(),
                            stop_words=stemmed_stop_words(),
                        )
                        cv.fit_transform([query_text])
                        ngrams = cv.get_feature_names_out()
                        q_mh = MinHash(num_perm=num_perm, seed=SEED)
                        for gram in ngrams:
                            q_mh.update(gram.encode("utf-8"))
                    except Exception:
                        q_mh = MinHash(num_perm=num_perm, seed=SEED)
                else:  # ProbMinHashLSH
                    try:
                        q_vec = tfidf_vec.transform([query_text])
                        q_keys = q_vec[0].indices
                        q_weights = q_vec[0].data
                        pmh_q = ProbMinHash4(num_perm=num_perm, seed=SEED)
                        if len(q_keys) > 0:
                            pmh_q.fit(q_keys, q_weights)
                        q_mh = MinHash(num_perm=num_perm)
                        q_mh.hashvalues = np.array(pmh_q.hashvalues, dtype=np.uint64)
                    except Exception:
                        q_mh = MinHash(num_perm=num_perm, seed=SEED)

                # Time the query
                tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    retrieved = lsh_index.query(q_mh)
                except Exception:
                    retrieved = []
                elapsed = time.perf_counter() - t0
                tracemalloc.stop()
                latencies.append(elapsed)

                # Baseline precision vs BM25
                bm25_res = bm25_results.get(qid, [])
                if retrieved and bm25_res:
                    bm25_set = set(bm25_res)
                    prec = sum(1 for d in retrieved if d in bm25_set) / len(retrieved)
                else:
                    prec = 0.0
                precisions_vs_bm25.append(prec)

            row = {
                "retriever":        retriever_name,
                "num_perm":         num_perm,
                "vocab_dim":        vocab_dim,
                "reduction_factor": round(vocab_dim / num_perm, 1),
                "jaccard_error":    jaccard_error,
                "prec_vs_bm25":     float(np.mean(precisions_vs_bm25)),
                "latency_s":        float(np.mean(latencies)),
                "memory_mb_per_doc": mb_per_doc,
            }
            rows.append(row)
            print(f"      jaccard_error={jaccard_error:.4f}  "
                  f"prec_vs_bm25={row['prec_vs_bm25']:.4f}  "
                  f"latency={row['latency_s']*1000:.2f}ms  "
                  f"mem={mb_per_doc*1024:.2f}KB/doc  "
                  f"reduction={row['reduction_factor']}×")

    return pd.DataFrame(rows)


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
    parser.add_argument(
        "--dim-reduction", action="store_true",
        help="Run dimensionality reduction tradeoff experiment (vary num_perm)"
    )
    parser.add_argument(
        "--dim-reduction-perms", type=int, nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="num_perm values to sweep in the dim-reduction experiment (default: 16 32 64 128 256 512)"
    )
    return parser.parse_args()


def _fetch_passages_by_id(
    collection_path: str,
    target_ids: set[str],
) -> dict[str, str]:
    """Fetch specific passages from collection.jsonl by their numeric ID.

    MS MARCO collection.jsonl has key == line_number (0-indexed).
    We do a single forward scan collecting every hit — no early break so all
    passages within the file are reachable regardless of ID magnitude.

    Returns dict: passage_id -> text
    """
    import json as _json

    if not target_ids:
        return {}

    target_set = set(target_ids)
    found: dict[str, str] = {}

    with open(collection_path, "r", encoding="utf-8") as f:
        for line in f:
            # Fast key extraction without full JSON parse on non-target lines
            # Collection format: {"key": "N", "data": "..."}
            try:
                obj = _json.loads(line)
                key = obj["key"]
                if key in target_set:
                    found[key] = obj["data"].strip()
                    if len(found) == len(target_set):
                        break
            except Exception:
                pass

    return found


def build_evaluation_corpus(
    qrels: dict[str, set[str]],
    queries: list[dict],
    corpus_size: int,
    max_queries: int,
    max_passage_id: int = 500_000,
) -> tuple[list[dict[str, str]], dict[str, set[str]], list[dict]]:
    """Build a realistic evaluation corpus with guaranteed relevant passage coverage.

    The local MS MARCO collection contains passages 0–500,000. Qrels reference
    passage IDs up to ~8.8M, so we pre-filter to queries whose relevant passages
    fall within the local collection before doing anything else.

    Strategy:
      1. Pre-filter qrels to passage IDs ≤ max_passage_id (local collection range).
      2. Take the first `max_queries` queries that have locally-reachable relevant passages.
      3. Collect those relevant passage IDs (must be in corpus).
      4. Fill remaining corpus slots with other in-range qrel passages as distractors.
      5. Single forward scan of the collection to load all target passages.

    Returns
    -------
    corpus_docs      : list of {"key": passage_id, "data": text}
    filtered_qrels   : qrels restricted to queries with local coverage
    filtered_queries : queries restricted to those with local coverage
    """
    # 1. Pre-filter qrels to locally available passage IDs
    local_qrels: dict[str, set[str]] = {}
    for qid, pids in qrels.items():
        local_pids = {p for p in pids if int(p) <= max_passage_id}
        if local_pids:
            local_qrels[qid] = local_pids

    print(f"  Queries with locally-available relevant passages: "
          f"{len(local_qrels)}/{len(qrels)} "
          f"(collection covers passages 0–{max_passage_id:,})")

    # 2. Eval query pool — queries that have local coverage
    eval_queries_pool = [q for q in queries if q["key"] in local_qrels][:max_queries]

    # 3. Relevant passage IDs for eval queries
    priority_ids: set[str] = set()
    for q in eval_queries_pool:
        priority_ids.update(local_qrels[q["key"]])

    # 4. Distractor pool: other in-range qrel passages
    distractor_ids: set[str] = set()
    for pids in local_qrels.values():
        distractor_ids.update(pids)
    distractor_ids -= priority_ids

    n_distractors = max(0, corpus_size - len(priority_ids))
    sorted_distractors = sorted(distractor_ids, key=lambda x: int(x))[:n_distractors]
    target_ids = priority_ids | set(sorted_distractors)

    print(f"  Target corpus: {len(priority_ids)} relevant + {len(sorted_distractors)} distractor passages")

    # 5. Forward scan of collection
    found_passages = _fetch_passages_by_id(COLLECTION_JSONL, target_ids)

    found_relevant = len(priority_ids & found_passages.keys())
    print(f"  Found {found_relevant}/{len(priority_ids)} relevant + "
          f"{len(found_passages) - found_relevant} distractor passages")

    # 6. Final filter
    filtered_qrels: dict[str, set[str]] = {}
    filtered_queries: list[dict] = []
    for q in eval_queries_pool:
        qid = q["key"]
        local_relevant = local_qrels[qid] & found_passages.keys()
        if local_relevant:
            filtered_qrels[qid] = local_relevant
            filtered_queries.append(q)

    print(f"  Queries with corpus coverage: {len(filtered_queries)}/{len(eval_queries_pool)}")

    corpus_docs = [{"key": k, "data": v} for k, v in found_passages.items()]
    print(f"  Final corpus size: {len(corpus_docs)} documents")

    return corpus_docs, filtered_qrels, filtered_queries


def main() -> None:
    args = parse_args()

    if args.quick:
        args.corpus_size = 500
        args.max_queries = 30
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
        minhash_dbscan = MinHashDbscanRetriever(dbscan_eps=0.85, corpus_initial_size=0)
        minhash_dbscan.build_corpus_from_docs(corpus_docs)
        retrievers["MinHashDBSCAN"] = minhash_dbscan
    except Exception as e:
        print(f"  MinHashDbscanRetriever build failed (skipping): {e}")

    try:
        minhash_lsh = MinHashLshRetriever(min_hash_lsh_eps=0.85, corpus_initial_size=0)
        minhash_lsh.build_corpus_from_docs(corpus_docs)
        retrievers["MinHashLSH"] = minhash_lsh
    except Exception as e:
        print(f"  MinHashLshRetriever build failed (skipping): {e}")

    try:
        prob_minhash_dbscan = ProbMinHashDbscanRetriever(dbscan_eps=0.85, corpus_initial_size=0)
        prob_minhash_dbscan.build_corpus_from_docs(corpus_docs)
        retrievers["ProbMinHashDBSCAN"] = prob_minhash_dbscan
    except Exception as e:
        print(f"  ProbMinHashDbscanRetriever build failed (skipping): {e}")

    try:
        prob_minhash_lsh = ProbMinHashLshRetriever(min_hash_lsh_eps=0.85, corpus_initial_size=0)
        prob_minhash_lsh.build_corpus_from_docs(corpus_docs)
        retrievers["ProbMinHashLSH"] = prob_minhash_lsh
    except Exception as e:
        print(f"  ProbMinHashLshRetriever build failed (skipping): {e}")


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
        lsh_names=("MinHashLSH", "ProbMinHashLSH"),
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

    def _make_classic(docs, k):
        r = ClassicRetriever(top_k=k, num_initial_documents=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_bm25(docs, k):
        r = BM25Retriever(top_k=k, num_initial_documents=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_minhash_dbscan(docs, k):
        r = MinHashDbscanRetriever(dbscan_eps=0.85, corpus_initial_size=0)
        r.build_corpus_from_docs(docs)
        return r

    def _make_minhash_lsh(docs, k):
        r = MinHashLshRetriever(min_hash_lsh_eps=0.85, corpus_initial_size=0)
        r.build_corpus_from_docs(docs)
        return r

    complexity_factories = {
        "Classic (O(n))": _make_classic,
        "BM25 (O(n))": _make_bm25,
        "MinHashDBSCAN (sub-linear)": _make_minhash_dbscan,
        "MinHashLSH (sub-linear)": _make_minhash_lsh,
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
    # 6b. Dimensionality Reduction Experiment (optional)
    # ------------------------------------------------------------------
    dim_df = pd.DataFrame()
    if args.dim_reduction:
        print("\n[main] Running dimensionality reduction experiment…")
        print(f"  Sweeping num_perm={args.dim_reduction_perms}")
        print(f"  Corpus: {len(corpus_docs)} docs  |  Queries: {min(len(eval_queries), 100)} sampled")
        dim_df = run_dim_reduction_experiment(
            corpus_docs=corpus_docs,
            eval_queries=eval_queries,
            eval_qrels=eval_qrels,
            baseline_retrievers={k: v for k, v in retrievers.items() if k in ("BM25", "Classic")},
            num_perms=args.dim_reduction_perms,
            top_k=args.top_k,
        )
        dim_df.to_csv(f"{EVAL_DIR}/dim_reduction_results.csv", index=False)
        print(f"Saved dim-reduction results → {EVAL_DIR}/dim_reduction_results.csv")
        print("\n  Dimensionality Reduction Summary:")
        print(f"  {'Retriever':20s} {'num_perm':>8} {'vocab_dim':>10} {'reduction':>10} "
              f"{'J_error':>10} {'prec_BM25':>10} {'lat_ms':>8} {'KB/doc':>8}")
        print("  " + "-" * 88)
        for _, row in dim_df.iterrows():
            print(f"  {row['retriever']:20s} {int(row['num_perm']):>8} "
                  f"{int(row['vocab_dim']):>10,} {row['reduction_factor']:>9.0f}× "
                  f"{row['jaccard_error']:>10.4f} {row['prec_vs_bm25']:>10.4f} "
                  f"{row['latency_s']*1000:>7.2f}ms {row['memory_mb_per_doc']*1024:>7.2f}KB")

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
    plot_sketch_vs_baseline(results_df, save_path=EVAL_DIR)
    # Focused comparison: MinHash family vs Classic + BM25
    # on the 3 core metrics: retrieval time, memory, baseline precision
    plot_minhash_vs_baseline(
        results_df,
        sketch_names=("MinHashDBSCAN", "MinHashLSH", "ProbMinHashDBSCAN", "ProbMinHashLSH"),
        save_path=EVAL_DIR,
    )
    if not dim_df.empty:
        plot_dim_reduction_tradeoff(dim_df, save_path=EVAL_DIR)

    print("\n[main] Writing summary report…")
    generate_summary_report(results_df, top_k=args.top_k, save_path=EVAL_DIR)

    print("\n" + "=" * 60)
    print("Evaluation complete!  All outputs saved to:", EVAL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()













