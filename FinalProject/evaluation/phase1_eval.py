"""
evaluation/phase1_eval.py
=========================
Phase 1 evaluation: Memory, Update Time, and Accuracy across three data configs.

Config A – Short queries  + Short documents  (passages ~52 words)
Config B – Short queries  + Full documents   (fulldocs titles ~5 words as query)
Config C – Long queries   + Full corpus      (fulldocs body ~545 words as query)

All accuracy metrics are computed over the FULL returned result set (@all).
Classic uses cosine >= COSINE_THRESHOLD; BM25 uses score > mean_positive_per_query.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import (
    average_precision,
    baseline_precision,
    f1_at_all,
    hit_rate_at_all,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from logic.constants import (
    COLLECTION_JSONL,
    FULLDOCS_TSV_GZ,
    QRELS_JSONL,
)
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever, _tokenize
from retrievers.classic_retriever import ClassicRetriever, _jaccard_scores

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COSINE_THRESHOLD = 0.05   # Classic: return docs with Jaccard-cosine >= this
CORPUS_SIZE_A    = 100_000
CORPUS_SIZE_BC   = 100_000
QUERY_SAMPLE_BC  = 10_000   # number of fulldoc titles/bodies used as queries
UPDATE_CORPUS_N  = 5_000    # smaller corpus used for the update-time experiment
N_UPDATE_DOCS    = 100
SEED             = 42


# ---------------------------------------------------------------------------
# Full-result-set retrievers (bypass top-k)
# ---------------------------------------------------------------------------

def _retrieve_all_classic(retriever: ClassicRetriever, query_text: str) -> list[str]:
    """Return all doc_ids with Jaccard-cosine similarity >= COSINE_THRESHOLD."""
    if retriever.vectorizer is None or retriever.doc_matrix is None:
        return []
    try:
        query_vec = retriever.vectorizer.transform([query_text])
    except Exception:
        return []
    scores = _jaccard_scores(query_vec, retriever.doc_matrix)   # (n,)
    mask = scores >= COSINE_THRESHOLD
    if not mask.any():
        return []
    indices = np.where(mask)[0]
    order = np.argsort(scores[indices])[::-1]
    return [retriever.doc_ids[indices[i]] for i in order]


def _retrieve_all_bm25(retriever: BM25Retriever, query_text: str) -> list[str]:
    """Return all doc_ids with BM25 score > mean of all positive scores."""
    if retriever._bm25 is None:
        return []
    tokens = _tokenize(query_text)
    if not tokens:
        return []
    scores = np.array(retriever._bm25.get_scores(tokens))
    positives = scores[scores > 0]
    if len(positives) == 0:
        return []
    threshold = float(positives.mean())
    mask = scores > threshold
    if not mask.any():
        return []
    indices = np.where(mask)[0]
    order = np.argsort(scores[indices])[::-1]
    return [retriever.doc_ids[indices[i]] for i in order]


def _retrieve_full(retriever: BaseRetriever, retriever_name: str, query_text: str) -> list[str]:
    """Dispatch to the correct full-result-set retrieve function."""
    if isinstance(retriever, ClassicRetriever):
        return _retrieve_all_classic(retriever, query_text)
    if isinstance(retriever, BM25Retriever):
        return _retrieve_all_bm25(retriever, query_text)
    # Sketch retrievers already return their full natural result set
    result = retriever.retrieve(query_text)
    if not result:
        return []
    if isinstance(result[0], tuple):
        return [d for d, _ in result]
    return [str(d) for d in result]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def run_query_with_memory(
    retriever: BaseRetriever,
    retriever_name: str,
    query_text: str,
) -> tuple[list[str], float]:
    """Run one full-result-set retrieve call; return (doc_ids, peak_memory_mb)."""
    tracemalloc.start()
    try:
        doc_ids = _retrieve_full(retriever, retriever_name, query_text)
    except Exception:
        doc_ids = []
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return doc_ids, peak_bytes / (1024 * 1024)


def run_update_experiment(
    retriever: BaseRetriever,
    retriever_name: str,
    update_docs: list[dict],
    n: int = N_UPDATE_DOCS,
    corpus_offset: int = 0,
) -> list[dict]:
    """Time n sequential update() calls. Returns list of row dicts."""
    rows = []
    for idx, doc in enumerate(update_docs[:n]):
        t0 = time.perf_counter()
        try:
            retriever.update(doc)
        except Exception as exc:
            print(f"      update() failed [{retriever_name}] doc {idx}: {exc}")
        elapsed = time.perf_counter() - t0
        rows.append({
            "retriever":              retriever_name,
            "doc_id":                 doc["key"],
            "update_index":           idx + 1,
            "corpus_size_at_update":  corpus_offset + idx + 1,
            "update_latency_s":       elapsed,
        })
        if (idx + 1) % 10 == 0:
            print(f"\r      {idx + 1}/{n} updates done", end="", flush=True)
    print()
    return rows


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_config_a(
    corpus_size: int = CORPUS_SIZE_A,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], dict[str, set[str]], list[dict]]:
    """Config A: short queries (~6 words) + short passage corpus (~52 words).

    Returns: (corpus_docs, eval_queries, qrels, update_docs)
    """
    print(f"[Config A] Loading passage corpus (first {corpus_size:,} docs)…")
    corpus_docs: list[dict] = []
    with open(COLLECTION_JSONL) as f:
        for line in f:
            if len(corpus_docs) >= corpus_size:
                break
            obj = json.loads(line)
            corpus_docs.append({"key": obj["key"], "data": obj["data"].strip()})

    corpus_id_set = {d["key"] for d in corpus_docs}
    print(f"  Corpus loaded: {len(corpus_docs):,} passages.")

    print("[Config A] Loading qrels and deriving queries…")
    qrels: dict[str, set[str]] = {}
    queries_map: dict[str, str] = {}
    with open(QRELS_JSONL) as f:
        for line in f:
            obj = json.loads(line)
            passage_id = obj["key"]
            parts = obj["data"].strip().split("\t")
            if len(parts) < 2:
                continue
            query_id, query_text = parts[0], parts[1]
            qrels.setdefault(query_id, set()).add(passage_id)
            if query_id not in queries_map:
                queries_map[query_id] = query_text

    # filter qrels to passages present in corpus
    filtered_qrels: dict[str, set[str]] = {}
    eval_queries: list[dict] = []
    for qid, pids in qrels.items():
        in_corpus = pids & corpus_id_set
        if in_corpus:
            filtered_qrels[qid] = in_corpus
            eval_queries.append({"key": qid, "data": queries_map[qid]})

    # shuffle for reproducibility and limit for speed
    rng = random.Random(seed)
    rng.shuffle(eval_queries)
    print(f"  Eval queries: {len(eval_queries):,} (all with ≥1 relevant passage in corpus)")

    # update docs: next N_UPDATE_DOCS passages from collection (beyond corpus_size)
    print("[Config A] Loading update docs…")
    update_docs: list[dict] = []
    with open(COLLECTION_JSONL) as f:
        for i, line in enumerate(f):
            if i < corpus_size:
                continue
            obj = json.loads(line)
            update_docs.append({"key": obj["key"], "data": obj["data"].strip()})
            if len(update_docs) >= N_UPDATE_DOCS:
                break
    print(f"  Update docs: {len(update_docs)}")

    return corpus_docs, eval_queries, filtered_qrels, update_docs


def _load_fulldocs(n: int = CORPUS_SIZE_BC) -> list[dict]:
    """Stream first n rows of fulldocs.tsv.gz → list[{key, title, data}]."""
    docs: list[dict] = []
    with gzip.open(FULLDOCS_TSV_GZ, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            docs.append({"key": parts[0], "title": parts[1], "data": parts[2]})
            if len(docs) >= n:
                break
    return docs


def load_configs_bc(
    corpus_size: int = CORPUS_SIZE_BC,
    n_queries: int   = QUERY_SAMPLE_BC,
    seed: int        = SEED,
) -> tuple[list[dict], list[dict], list[dict], dict[str, set[str]], list[dict]]:
    """Load data shared by Config B and Config C.

    Returns: (corpus_docs, queries_b, queries_c, qrels, update_docs)
      corpus_docs : {"key": url, "data": body}  — the retriever-visible corpus
      queries_b   : {"key": url, "data": title} — short title queries (Config B)
      queries_c   : {"key": url, "data": body}  — long body queries  (Config C)
      qrels       : {url: {url}}                 — self-referential ground truth
      update_docs : next N_UPDATE_DOCS docs beyond corpus_size
    """
    print(f"[Config B/C] Loading {corpus_size:,} full docs from fulldocs.tsv.gz…")
    all_docs = _load_fulldocs(corpus_size + N_UPDATE_DOCS)

    corpus_raw = all_docs[:corpus_size]
    update_raw = all_docs[corpus_size: corpus_size + N_UPDATE_DOCS]

    # corpus_docs in retriever-compatible format (body is the "data")
    corpus_docs = [{"key": d["key"], "data": d["data"]} for d in corpus_raw]
    print(f"  Corpus: {len(corpus_docs):,} full docs.")

    # random sample for queries
    rng = random.Random(seed)
    query_sample = rng.sample(corpus_raw, min(n_queries, len(corpus_raw)))

    queries_b = [{"key": d["key"], "data": d["title"]} for d in query_sample]
    queries_c = [{"key": d["key"], "data": d["data"]}  for d in query_sample]

    # self-referential: the answer to a title/body query IS that document
    qrels = {d["key"]: {d["key"]} for d in query_sample}

    update_docs = [{"key": d["key"], "data": d["data"]} for d in update_raw]
    print(f"  Query sample: {len(queries_b):,}  Update docs: {len(update_docs)}")

    return corpus_docs, queries_b, queries_c, qrels, update_docs


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    accuracy_rows: list[dict],
    memory_rows:   list[dict],
    update_rows:   list[dict],
    retrievers:    list[str],
) -> pd.DataFrame:
    acc_df = pd.DataFrame(accuracy_rows)
    mem_df = pd.DataFrame(memory_rows)

    rows = []
    for ret in retrievers:
        a = acc_df[acc_df["retriever"] == ret]
        m = mem_df[mem_df["retriever"] == ret]

        row: dict[str, Any] = {"retriever": ret}

        for col in ["precision_at_all", "recall_at_all", "f1_at_all",
                    "ndcg_at_all", "mrr", "ap", "hit_rate_at_all",
                    "vs_Classic", "vs_BM25", "n_results"]:
            if col in a.columns:
                row[f"{col}_mean"] = float(a[col].mean(skipna=True))

        row["map"] = float(a["ap"].mean(skipna=True)) if "ap" in a.columns else float("nan")
        row["memory_mb_mean"] = float(m["memory_mb"].mean()) if not m.empty else float("nan")
        row["memory_mb_std"]  = float(m["memory_mb"].std())  if not m.empty else float("nan")

        if update_rows:
            u = pd.DataFrame(update_rows)
            u = u[u["retriever"] == ret]
            if not u.empty:
                lats = u["update_latency_s"].values
                row["update_latency_mean_s"] = float(lats.mean())
                row["update_latency_std_s"]  = float(lats.std())
                row["update_latency_p50_s"]  = float(np.percentile(lats, 50))
                row["update_latency_p95_s"]  = float(np.percentile(lats, 95))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_phase1(
    retrievers:   dict[str, BaseRetriever],
    corpus_docs:  list[dict],
    eval_queries: list[dict],
    qrels:        dict[str, set[str]],
    config_name:  str,
    output_dir:   str,
    update_docs:  list[dict] | None = None,
    max_queries:  int | None = None,
    run_updates:  bool = True,
) -> None:
    """Orchestrate accuracy + memory + update-time for one config. Saves 4 CSVs."""
    save_dir = os.path.join(output_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Build all retrievers ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" Phase 1 — {config_name}  ({len(corpus_docs):,} corpus docs)")
    print(f"{'='*65}")
    for name, retriever in retrievers.items():
        print(f"\n  ► Building {name}…")
        t0 = time.perf_counter()
        retriever.build_corpus_from_docs(corpus_docs)
        print(f"    Done in {time.perf_counter()-t0:.1f}s")

    # ── 2. Filter / cap queries ───────────────────────────────────────────
    runnable = [q for q in eval_queries if q["key"] in qrels]
    if max_queries:
        runnable = runnable[:max_queries]
    total_q = len(runnable)
    print(f"\n  Queries to evaluate: {total_q:,}")

    # ── 3. Pre-compute Classic + BM25 full results ─────────────────────────
    classic_full: dict[str, list[str]] = {}
    bm25_full:    dict[str, list[str]] = {}
    classic_ret = retrievers.get("Classic")
    bm25_ret    = retrievers.get("BM25")

    if classic_ret:
        print("  Pre-computing Classic full results…")
        for q in runnable:
            classic_full[q["key"]] = _retrieve_all_classic(classic_ret, q["data"])
    if bm25_ret:
        print("  Pre-computing BM25 full results…")
        for q in runnable:
            bm25_full[q["key"]] = _retrieve_all_bm25(bm25_ret, q["data"])

    # ── 4. Accuracy + Memory loop ─────────────────────────────────────────
    accuracy_rows: list[dict] = []
    memory_rows:   list[dict] = []

    for ret_name, retriever in retrievers.items():
        print(f"\n  ► Evaluating {ret_name} ({total_q:,} queries)…")
        t_start = time.perf_counter()

        for i, q in enumerate(runnable):
            qid      = q["key"]
            qtext    = q["data"]
            relevant = qrels.get(qid, set())

            try:
                doc_ids, mem_mb = run_query_with_memory(retriever, ret_name, qtext)
            except Exception as exc:
                print(f"\n    [!] query {qid} failed: {exc}")
                doc_ids, mem_mb = [], 0.0

            k = len(doc_ids)
            P    = precision_at_k(doc_ids, relevant, k) if k else 0.0
            R    = recall_at_k(doc_ids, relevant, k)    if k else 0.0
            F1   = f1_at_all(doc_ids, relevant)
            NDCG = ndcg_at_k(doc_ids, relevant, k)      if k else 0.0
            MRR  = mrr(doc_ids, relevant)
            AP   = average_precision(doc_ids, relevant)
            HR   = hit_rate_at_all(doc_ids, relevant)

            vs_c = (baseline_precision(doc_ids, classic_full.get(qid, []))
                    if ret_name != "Classic" else float("nan"))
            vs_b = (baseline_precision(doc_ids, bm25_full.get(qid, []))
                    if ret_name != "BM25" else float("nan"))

            accuracy_rows.append({
                "retriever":       ret_name,
                "query_id":        qid,
                "n_results":       k,
                "precision_at_all": P,
                "recall_at_all":   R,
                "f1_at_all":       F1,
                "ndcg_at_all":     NDCG,
                "mrr":             MRR,
                "ap":              AP,
                "hit_rate_at_all": HR,
                "vs_Classic":      vs_c,
                "vs_BM25":         vs_b,
            })
            memory_rows.append({
                "retriever": ret_name,
                "query_id":  qid,
                "memory_mb": mem_mb,
            })

            if (i + 1) % 500 == 0 or (i + 1) == total_q:
                elapsed = time.perf_counter() - t_start
                print(f"\r    {i+1:>6}/{total_q}  ({elapsed:.0f}s)", end="", flush=True)
        print()

    # ── 5. Update time (runs LAST – mutates index) ────────────────────────
    update_rows: list[dict] = []
    if run_updates and update_docs:
        print(f"\n  ► Update-time experiment ({N_UPDATE_DOCS} sequential updates)…")
        print(f"    NOTE: update() is run on the already-built {len(corpus_docs):,}-doc index.")
        for ret_name, retriever in retrievers.items():
            print(f"    {ret_name}…")
            rows = run_update_experiment(
                retriever, ret_name, update_docs,
                n=N_UPDATE_DOCS, corpus_offset=len(corpus_docs),
            )
            update_rows.extend(rows)

    # ── 6. Save CSVs ──────────────────────────────────────────────────────
    acc_path = os.path.join(save_dir, "accuracy.csv")
    mem_path = os.path.join(save_dir, "memory.csv")
    upd_path = os.path.join(save_dir, "update_time.csv")
    sum_path = os.path.join(save_dir, "summary.csv")

    pd.DataFrame(accuracy_rows).to_csv(acc_path, index=False)
    pd.DataFrame(memory_rows).to_csv(mem_path, index=False)
    if update_rows:
        pd.DataFrame(update_rows).to_csv(upd_path, index=False)

    summary_df = _build_summary(
        accuracy_rows, memory_rows, update_rows,
        list(retrievers.keys()),
    )
    summary_df.to_csv(sum_path, index=False)

    # ── 7. Print summary table ────────────────────────────────────────────
    _print_summary(summary_df, config_name)
    print(f"\n  Saved to {save_dir}/")


def _print_summary(df: pd.DataFrame, config_name: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  Summary — {config_name}")
    print(f"{'─'*65}")
    cols = ["retriever", "hit_rate_at_all_mean", "mrr_mean", "map",
            "recall_at_all_mean", "f1_at_all_mean",
            "memory_mb_mean", "update_latency_mean_s"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

