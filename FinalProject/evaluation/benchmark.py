"""
benchmark.py – Run retrievers against a set of queries and collect metrics.

Usage
-----
    from evaluation.benchmark import run_benchmark

    results_df = run_benchmark(
        retrievers={"Classic": classic_r, "ANN": ann_r, "Sketch": sketch_r},
        queries=queries,        # list of {"key": qid, "data": query_text}
        qrels=qrels,            # dict  qid -> set of relevant doc_ids
        top_k=10,
    )

Each row in the returned DataFrame represents one (retriever, query) pair and
contains latency, memory delta, and all IR metrics.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import (
    compute_all_metrics,
    baseline_precision,
    credibility_score,
    distance_score,
    hallucination_flagged,
    misinformation_flagged,
    HALLUCINATION_DISTANCE_THRESHOLD,
    MISINFORMATION_CREDIBILITY_THRESHOLD,
)
from retrievers.base_retriever import BaseRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single_query(
    retriever: BaseRetriever,
    query_text: str,
    top_k: int,
) -> tuple[list[str], float, float]:
    """Invoke retriever.retrieve() and measure wall-clock latency + peak memory.

    Returns
    -------
    doc_ids   : ordered list of retrieved doc ids
    latency_s : wall-clock seconds
    memory_mb : peak memory increase in MB (tracemalloc)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = retriever.retrieve(query_text, top_k=top_k)
    latency_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # retrieve() may return list[str] or list[tuple[str, float]]
    if result and isinstance(result[0], tuple):
        doc_ids = [doc_id for doc_id, _ in result]
    else:
        doc_ids = list(result)

    memory_mb = peak / (1024 * 1024)
    return doc_ids, latency_s, memory_mb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(
    retrievers: dict[str, BaseRetriever],
    queries: list[dict[str, str]],
    qrels: dict[str, set[str]],
    top_k: int = 10,
    max_queries: int | None = None,
    baseline_names: tuple[str, ...] = ("Classic", "BM25"),
) -> pd.DataFrame:
    """Evaluate every retriever over every query.

    Parameters
    ----------
    retrievers      : mapping of name → retriever instance
    queries         : list of {"key": qid, "data": query_text}
    qrels           : dict of qid → set of relevant doc_ids
    top_k           : number of results to request per query
    max_queries     : optional cap on number of queries
    baseline_names  : retriever names treated as the baseline for
                      baseline_precision computation (default: Classic + BM25)

    Returns
    -------
    pd.DataFrame with columns:
        retriever, query_id, latency_s, memory_mb,
        precision@k, recall@k, ndcg@k, mrr, ap,
        baseline_precision, distance_score, credibility_score, …
    """
    rows: list[dict[str, Any]] = []

    # Filter to queries that have relevance judgements
    eval_queries = [q for q in queries if q["key"] in qrels]
    if max_queries is not None:
        eval_queries = eval_queries[:max_queries]

    total = len(eval_queries)
    print(f"[benchmark] Evaluating {total} queries × {len(retrievers)} retrievers…")

    # ------------------------------------------------------------------
    # Pre-compute per-baseline results for every query.
    # per_baseline_results[baseline_name][qid] = list of doc_ids
    # This lets us compute vs_Classic and vs_BM25 precision separately
    # for every retriever (including the cross-baseline comparison).
    # ------------------------------------------------------------------
    per_baseline_results: dict[str, dict[str, list[str]]] = {}
    for bname in baseline_names:
        if bname not in retrievers:
            continue
        retriever = retrievers[bname]
        print(f"\n  [baseline] Pre-computing results for {bname}…")
        per_baseline_results[bname] = {}
        for query in eval_queries:
            qid = query["key"]
            try:
                result = retriever.retrieve(query["data"].strip(), top_k=top_k)
                if result and isinstance(result[0], tuple):
                    doc_ids = [d for d, _ in result]
                else:
                    doc_ids = list(result)
            except Exception:
                doc_ids = []
            per_baseline_results[bname][qid] = doc_ids

    # Keep the combined view for backward compatibility
    baseline_results: dict[str, list[str]] = {}
    for bname, bq in per_baseline_results.items():
        for qid, doc_ids in bq.items():
            baseline_results.setdefault(qid, [])
            baseline_results[qid].extend(d for d in doc_ids if d not in baseline_results[qid])

    # ------------------------------------------------------------------
    # Main benchmark loop
    # ------------------------------------------------------------------
    for retriever_name, retriever in retrievers.items():
        print(f"\n  → {retriever_name}")

        vectorizer = getattr(retriever, "vectorizer", None)
        svd        = getattr(retriever, "svd", None)
        is_baseline = retriever_name in baseline_names

        for i, query in enumerate(eval_queries):
            qid = query["key"]
            query_text = query["data"].strip()
            relevant = qrels[qid]

            try:
                doc_ids, latency_s, memory_mb = _run_single_query(retriever, query_text, top_k)
                metrics = compute_all_metrics(doc_ids, relevant, k=top_k)
            except Exception as exc:
                print(f"    [!] query {qid} failed: {exc}")
                doc_ids, latency_s, memory_mb = [], 0.0, 0.0
                metrics = compute_all_metrics([], relevant, k=top_k)

            # ----------------------------------------------------------
            # Per-baseline precision: how much overlap does this retriever
            # have with each individual baseline's results?
            #
            # For a baseline retriever (e.g. Classic), we compute its
            # overlap with the OTHER baseline (e.g. BM25), giving a
            # meaningful cross-baseline agreement score instead of 1.0.
            # ----------------------------------------------------------
            per_bl_precision: dict[str, float] = {}
            for bname in baseline_names:
                if bname not in per_baseline_results:
                    continue
                bl_doc_ids = per_baseline_results[bname].get(qid, [])
                if retriever_name == bname:
                    # This IS that baseline — skip (would be trivially 1.0)
                    per_bl_precision[f"vs_{bname}"] = float("nan")
                else:
                    per_bl_precision[f"vs_{bname}"] = baseline_precision(doc_ids, bl_doc_ids)

            # Keep the combined baseline_precision for backward compat
            bl_precision = baseline_precision(doc_ids, baseline_results.get(qid, []))
            if retriever_name in baseline_names:
                # For a baseline, combined includes own results → recompute
                # against only the OTHER baselines
                others = [d for bn, bq in per_baseline_results.items()
                          if bn != retriever_name
                          for d in bq.get(qid, [])]
                bl_precision = baseline_precision(doc_ids, others) if others else float("nan")

            # ----------------------------------------------------------
            # Distance Score
            # ----------------------------------------------------------
            top1_distance = 1.0
            top1_hallucination = True
            if doc_ids and vectorizer is not None:
                try:
                    q_vec = vectorizer.transform([query_text])
                    if svd is not None:
                        q_emb = svd.transform(q_vec)
                        doc_idx = retriever.doc_ids.index(doc_ids[0])
                        p_emb = retriever.reduced_matrix[doc_idx:doc_idx+1]
                    else:
                        q_emb = q_vec.toarray()
                        doc_idx = retriever.doc_ids.index(doc_ids[0])
                        p_emb = retriever.doc_matrix[doc_idx].toarray()
                    top1_distance = distance_score(
                        np.array(q_emb).flatten(),
                        np.array(p_emb).flatten(),
                    )
                    top1_hallucination = hallucination_flagged(top1_distance)
                except Exception:
                    pass

            # ----------------------------------------------------------
            # Credibility Score
            # ----------------------------------------------------------
            top1_credibility = 50.0
            top1_misinformation = False
            if doc_ids and hasattr(retriever, "doc_ids") and hasattr(retriever, "doc_texts"):
                try:
                    doc_idx = retriever.doc_ids.index(doc_ids[0])
                    top1_credibility = credibility_score(retriever.doc_texts[doc_idx])
                    top1_misinformation = misinformation_flagged(top1_credibility)
                except Exception:
                    pass

            row = {
                "retriever": retriever_name,
                "query_id": qid,
                "latency_s": latency_s,
                "memory_mb": memory_mb,
                "baseline_precision": bl_precision,
                **per_bl_precision,
                "distance_score": top1_distance,
                "hallucination_flagged": int(top1_hallucination),
                "credibility_score": top1_credibility,
                "misinformation_flagged": int(top1_misinformation),
                **metrics,
            }
            rows.append(row)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"\r    {i + 1}/{total} queries done", end="", flush=True)

        print()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Running-Time Complexity Experiment
# ---------------------------------------------------------------------------

def run_complexity_experiment(
    retriever_factories: dict[str, Any],
    corpus_docs: list[dict[str, str]],
    test_query: str,
    corpus_sizes: list[int],
    top_k: int = 10,
    n_trials: int = 5,
) -> pd.DataFrame:
    """Measure retrieval latency at increasing corpus sizes.

    This produces the empirical running-time complexity curve that can be
    compared against the theoretical O(n) (Classic) vs sub-linear (ANN)
    complexity predictions.

    Parameters
    ----------
    retriever_factories : dict of name → callable(corpus_docs, top_k) → retriever
    corpus_docs         : full pool of documents to sample from
    test_query          : fixed query used for every measurement
    corpus_sizes        : list of corpus sizes to test (n values)
    top_k               : retrieval depth
    n_trials            : number of timed repetitions per (retriever, size) point

    Returns
    -------
    DataFrame with columns: retriever, corpus_size, mean_latency_s, std_latency_s
    """
    rows = []
    for size in corpus_sizes:
        subset = corpus_docs[:size]
        if len(subset) < size:
            print(f"  [complexity] Warning: only {len(subset)} docs available for size={size}")

        for ret_name, factory in retriever_factories.items():
            # Build retriever on the subset
            try:
                retriever = factory(subset, top_k)
            except Exception as exc:
                print(f"  [complexity] Build failed {ret_name}@{size}: {exc}")
                continue

            # Time n_trials retrievals
            latencies = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                try:
                    retriever.retrieve(test_query, top_k=top_k)
                except Exception:
                    pass
                latencies.append(time.perf_counter() - t0)

            rows.append({
                "retriever": ret_name,
                "corpus_size": size,
                "mean_latency_s": float(np.mean(latencies)),
                "std_latency_s": float(np.std(latencies, ddof=0)),
            })
            print(f"  [complexity] {ret_name:10s}  n={size:6d}  "
                  f"mean={rows[-1]['mean_latency_s']*1000:.2f} ms")

    return pd.DataFrame(rows)


def load_qrels(qrels_path: str) -> dict[str, set[str]]:
    """Parse the MS MARCO top1000.dev qrels file (JSONL or TSV format).

    JSONL format (qrels.jsonl):
        {"key": "<passage_id>", "data": "<query_id>\\t<query_text>\\t<passage_text>\\n"}

    Returns dict: query_id -> set of relevant passage_ids.
    """
    import json as _json
    qrels: dict[str, set[str]] = {}

    with open(qrels_path) as f:
        first_line = f.readline().strip()
    is_jsonl = first_line.startswith("{")

    with open(qrels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                obj = _json.loads(line)
                passage_id = obj["key"]
                parts = obj["data"].strip().split("\t")
                if len(parts) < 1:
                    continue
                query_id = parts[0]
            else:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                passage_id = parts[0]
                query_id = parts[1]

            qrels.setdefault(query_id, set()).add(passage_id)

    return qrels


def load_queries_and_qrels_from_qrels_jsonl(
    qrels_jsonl_path: str,
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    """Single-source loader that derives BOTH queries and qrels from qrels.jsonl.

    qrels.jsonl format: {"key": "<passage_id>", "data": "<query_id>\\t<query_text>\\t<passage_text>"}

    Because queries.jsonl uses different query IDs than qrels.jsonl, this is
    the authoritative way to get a consistent (queries, qrels) pair where
    every query is guaranteed to have at least one relevant passage.

    Returns
    -------
    queries : list of {"key": query_id, "data": query_text}  (deduplicated)
    qrels   : dict of query_id -> set of relevant passage_ids
    """
    import json as _json
    seen_qids: set[str] = set()
    queries: list[dict[str, str]] = []
    qrels: dict[str, set[str]] = {}

    with open(qrels_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            passage_id = obj["key"]
            parts = obj["data"].strip().split("\t")
            if len(parts) < 2:
                continue
            query_id, query_text = parts[0], parts[1]
            qrels.setdefault(query_id, set()).add(passage_id)
            if query_id not in seen_qids:
                seen_qids.add(query_id)
                queries.append({"key": query_id, "data": query_text.strip()})

    return queries, qrels


def load_queries_jsonl(queries_jsonl_path: str) -> list[dict[str, str]]:
    """Load queries from JSONL file. Each line: {"key": qid, "data": text}."""
    import json as _json
    queries = []
    with open(queries_jsonl_path) as f:
        for line in f:
            obj = _json.loads(line)
            obj["data"] = obj["data"].strip()
            queries.append(obj)
    return queries


