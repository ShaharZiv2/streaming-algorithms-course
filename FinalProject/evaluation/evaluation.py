"""
evaluation/evaluation.py
========================
Phase 1 evaluation: Memory, Update Time, and Accuracy across three data configs.

Config A – Short queries  + Short documents  (passages ~52 words)
Config B – Short queries  + Full documents   (fulldoc titles ~5 words as query)
Config C – Long  queries  + Full corpus      (fulldoc body  ~545 words as query)

All accuracy metrics are computed over the FULL returned result set (@all).

Run directly:
    # quick smoke-test (~1–2 min, 1 000-doc corpus, 10 queries, no updates)
    python evaluation/evaluation.py --quick

    # full runs
    python evaluation/evaluation.py --all
    python evaluation/evaluation.py --config-a --max-queries 500
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Allow `python evaluation/evaluation.py` to be run from the project root
# by ensuring the project root (parent of this file's directory) is on sys.path.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import csv
import gzip
import json
import os
import random
import sys
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

COSINE_THRESHOLD = 0.05
CORPUS_SIZE_A    = 100_000
CORPUS_SIZE_BC   = 100_000
QUERY_SAMPLE_BC  = 10_000
UPDATE_CORPUS_N  = 5_000
N_UPDATE_DOCS    = 100
SEED             = 42

# Quick-run overrides (used when --quick is passed)
QUICK_CORPUS_SIZE = 1_000
QUICK_MAX_QUERIES = 10

DEFAULT_OUTPUT       = "datasets/evaluations/raw_results"
DEFAULT_QUICK_OUTPUT = "datasets/evaluations/quick"


# ---------------------------------------------------------------------------
# Full-result-set retrievers (bypass top-k)
# ---------------------------------------------------------------------------

def _retrieve_all_classic(retriever: ClassicRetriever, query_text: str) -> list[str]:
    """Return all doc_ids with TF-IDF cosine similarity >= COSINE_THRESHOLD."""
    if retriever.vectorizer is None or retriever.doc_matrix is None:
        return []
    try:
        query_vec = retriever.vectorizer.transform([query_text])
    except Exception:
        return []

    from sklearn.preprocessing import normalize as sk_normalize
    query_norm = sk_normalize(query_vec, norm="l2")

    if hasattr(retriever, "doc_matrix_norm") and retriever.doc_matrix_norm is not None:
        doc_norm = retriever.doc_matrix_norm
    else:
        doc_norm = sk_normalize(retriever.doc_matrix, norm="l2", copy=True)

    scores = doc_norm.dot(query_norm.T).toarray().ravel()
    mask = scores >= COSINE_THRESHOLD
    if not mask.any():
        return []
    indices = np.where(mask)[0]
    order = np.argsort(scores[indices])[::-1]
    return [retriever.doc_ids[indices[i]] for i in order]


def _retrieve_all_bm25(retriever: BM25Retriever, query_text: str) -> list[str]:
    """Return all doc_ids with BM25 score > mean+std of positive scores."""
    if retriever._bm25 is None:
        return []
    tokens = _tokenize(query_text)
    if not tokens:
        return []
    scores = np.array(retriever._bm25.get_scores(tokens))
    positives = scores[scores > 0]
    if len(positives) == 0:
        return []
    threshold = float(positives.mean() + positives.std())
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
    """Run one full-result-set retrieve; return (doc_ids, peak_memory_mb)."""
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
            "retriever":             retriever_name,
            "doc_id":                doc["key"],
            "update_index":          idx + 1,
            "corpus_size_at_update": corpus_offset + idx + 1,
            "update_latency_s":      elapsed,
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
    """Config A: short queries + short passage corpus.

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

    filtered_qrels: dict[str, set[str]] = {}
    eval_queries: list[dict] = []
    for qid, pids in qrels.items():
        in_corpus = pids & corpus_id_set
        if in_corpus:
            filtered_qrels[qid] = in_corpus
            eval_queries.append({"key": qid, "data": queries_map[qid]})

    rng = random.Random(seed)
    rng.shuffle(eval_queries)
    print(f"  Eval queries: {len(eval_queries):,}")

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
    """
    print(f"[Config B/C] Loading {corpus_size:,} full docs from fulldocs.tsv.gz…")
    all_docs = _load_fulldocs(corpus_size + N_UPDATE_DOCS)

    corpus_raw = all_docs[:corpus_size]
    update_raw = all_docs[corpus_size: corpus_size + N_UPDATE_DOCS]

    corpus_docs = [{"key": d["key"], "data": d["data"]} for d in corpus_raw]
    print(f"  Corpus: {len(corpus_docs):,} full docs.")

    rng = random.Random(seed)
    query_sample = rng.sample(corpus_raw, min(n_queries, len(corpus_raw)))

    queries_b = [{"key": d["key"], "data": d["title"]} for d in query_sample]
    queries_c = [{"key": d["key"], "data": d["data"]}  for d in query_sample]
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


def _print_summary(df: pd.DataFrame, config_name: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  Summary — {config_name}")
    print(f"{'─'*65}")
    cols = ["retriever", "hit_rate_at_all_mean", "mrr_mean", "map",
            "recall_at_all_mean", "f1_at_all_mean",
            "memory_mb_mean", "update_latency_mean_s"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Mini update experiment
# ---------------------------------------------------------------------------

def _run_mini_update_experiment(
    corpus_docs: list[dict],
    update_docs: list[dict],
    mini_size: int = UPDATE_CORPUS_N,
) -> list[dict]:
    """Build fresh retrievers on a small corpus slice; time N_UPDATE_DOCS updates."""
    mini_corpus = corpus_docs[:mini_size]
    print(f"\n  ► Update-time experiment ({N_UPDATE_DOCS} updates on {mini_size:,}-doc mini-corpus)…")

    mini_retrievers = _make_retrievers()
    print(f"    Building mini-corpus retrievers ({mini_size:,} docs)…")
    for name, ret in mini_retrievers.items():
        t0 = time.perf_counter()
        ret.build_corpus_from_docs(mini_corpus)
        print(f"      {name}: {time.perf_counter()-t0:.1f}s")

    all_rows: list[dict] = []
    for ret_name, ret in mini_retrievers.items():
        print(f"    {ret_name} updates…", end="", flush=True)
        rows = run_update_experiment(
            ret, ret_name, update_docs,
            n=N_UPDATE_DOCS, corpus_offset=mini_size,
        )
        all_rows.extend(rows)
    return all_rows


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
    skip_build:   bool = False,
) -> None:
    """Orchestrate accuracy + memory + update-time for one config. Saves CSVs."""
    save_dir = os.path.join(output_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f" Phase 1 — {config_name}  ({len(corpus_docs):,} corpus docs)")
    print(f"{'='*65}")

    if skip_build:
        print("  (Skipping build — reusing retrievers from previous config)")
    else:
        for name, retriever in retrievers.items():
            print(f"\n  ► Building {name}…")
            t0 = time.perf_counter()
            retriever.build_corpus_from_docs(corpus_docs)
            print(f"    Done in {time.perf_counter()-t0:.1f}s")

    runnable = [q for q in eval_queries if q["key"] in qrels]
    if max_queries:
        runnable = runnable[:max_queries]
    total_q = len(runnable)
    print(f"\n  Queries to evaluate: {total_q:,}")

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
            accuracy_rows.append({
                "retriever":        ret_name,
                "query_id":         qid,
                "n_results":        k,
                "precision_at_all": precision_at_k(doc_ids, relevant, k) if k else 0.0,
                "recall_at_all":    recall_at_k(doc_ids, relevant, k)    if k else 0.0,
                "f1_at_all":        f1_at_all(doc_ids, relevant),
                "ndcg_at_all":      ndcg_at_k(doc_ids, relevant, k)      if k else 0.0,
                "mrr":              mrr(doc_ids, relevant),
                "ap":               average_precision(doc_ids, relevant),
                "hit_rate_at_all":  hit_rate_at_all(doc_ids, relevant),
                "vs_Classic":       (baseline_precision(doc_ids, classic_full.get(qid, []))
                                     if ret_name != "Classic" else float("nan")),
                "vs_BM25":          (baseline_precision(doc_ids, bm25_full.get(qid, []))
                                     if ret_name != "BM25" else float("nan")),
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

    update_rows: list[dict] = []
    if run_updates and update_docs:
        update_rows = _run_mini_update_experiment(
            corpus_docs=corpus_docs,
            update_docs=update_docs,
            mini_size=UPDATE_CORPUS_N,
        )

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
    _print_summary(summary_df, config_name)
    print(f"\n  Saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def _make_retrievers() -> dict[str, BaseRetriever]:
    """Instantiate all 6 retrievers in lazy mode (no auto-build)."""
    from retrievers.bm25_retriever import BM25Retriever
    from retrievers.classic_retriever import ClassicRetriever
    from retrievers.min_hash_dbscan_retriever import MinHashDbscanRetriever
    from retrievers.min_hash_lsh_retriever import MinHashLshRetriever
    from retrievers.prob_min_hash_dbscan_retriever import ProbMinHashDbscanRetriever
    from retrievers.prob_min_hash_lsh_retriever import ProbMinHashLshRetriever

    return {
        "Classic":           ClassicRetriever(top_k=10, num_initial_documents=0),
        "BM25":              BM25Retriever(top_k=10, num_initial_documents=0),
        "MinHashDBSCAN":     MinHashDbscanRetriever(corpus_initial_size=0),
        "MinHashLSH":        MinHashLshRetriever(corpus_initial_size=0),
        "ProbMinHashDBSCAN": ProbMinHashDbscanRetriever(corpus_initial_size=0),
        "ProbMinHashLSH":    ProbMinHashLshRetriever(corpus_initial_size=0),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluation.py --quick             # smoke-test, ~1–2 min
  python evaluation/evaluation.py --all               # full run, all 3 configs
  python evaluation/evaluation.py --config-a --max-queries 500
        """,
    )
    parser.add_argument("--config-a",    action="store_true", help="Run Config A")
    parser.add_argument("--config-b",    action="store_true", help="Run Config B")
    parser.add_argument("--config-c",    action="store_true", help="Run Config C")
    parser.add_argument("--all",         action="store_true", help="Run all 3 configs")
    parser.add_argument(
        "--quick", action="store_true",
        help=(
            f"Smoke-test: {QUICK_CORPUS_SIZE:,}-doc corpus, "
            f"{QUICK_MAX_QUERIES} queries, no updates. "
            f"Saves to {DEFAULT_QUICK_OUTPUT}/ (never overwrites {DEFAULT_OUTPUT}/)."
        ),
    )
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Cap number of eval queries per config")
    parser.add_argument("--no-updates",  action="store_true",
                        help="Skip the update-time experiment")
    parser.add_argument("--output-dir",  default=None,
                        help=f"Output root (default: {DEFAULT_OUTPUT}; "
                             f"{DEFAULT_QUICK_OUTPUT} when --quick)")
    args = parser.parse_args()

    if args.all:
        args.config_a = args.config_b = args.config_c = True

    # --quick: safe defaults — tiny corpus, separate output dir, no updates
    if args.quick:
        if not (args.config_a or args.config_b or args.config_c):
            args.config_a = True              # default to Config A
        if args.max_queries is None:
            args.max_queries = QUICK_MAX_QUERIES
        args.no_updates = True
        if args.output_dir is None:
            args.output_dir = DEFAULT_QUICK_OUTPUT

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT

    if not (args.config_a or args.config_b or args.config_c):
        parser.error("Specify at least one of --config-a/b/c, or use --all / --quick")

    os.makedirs(args.output_dir, exist_ok=True)

    corpus_a  = QUICK_CORPUS_SIZE if args.quick else CORPUS_SIZE_A
    corpus_bc = QUICK_CORPUS_SIZE if args.quick else CORPUS_SIZE_BC

    t_global = time.perf_counter()
    shared_bc = None

    # ── Config A ─────────────────────────────────────────────────────────────
    if args.config_a:
        print("\n" + "="*70)
        print(" CONFIG A  —  Short queries  +  Short passage corpus")
        if args.quick:
            print(f"  [QUICK: {corpus_a:,}-doc corpus, "
                  f"max {args.max_queries} queries, no updates]")
        print("="*70)

        corpus_docs, eval_queries, qrels, update_docs = load_config_a(
            corpus_size=corpus_a
        )
        run_phase1(
            retrievers   = _make_retrievers(),
            corpus_docs  = corpus_docs,
            eval_queries = eval_queries,
            qrels        = qrels,
            config_name  = "config_a",
            output_dir   = args.output_dir,
            update_docs  = update_docs,
            max_queries  = args.max_queries,
            run_updates  = not args.no_updates,
        )

    # ── Config B / C shared data load ────────────────────────────────────────
    if args.config_b or args.config_c:
        print("\nLoading shared full-docs data (Config B/C)…")
        if args.quick:
            print(f"  [QUICK: {corpus_bc:,}-doc corpus]")
        shared_bc = load_configs_bc(corpus_size=corpus_bc)

    # ── Config B ─────────────────────────────────────────────────────────────
    if args.config_b:
        print("\n" + "="*70)
        print(" CONFIG B  —  Short queries (doc titles)  +  Full document corpus")
        print("="*70)

        corpus_docs, queries_b, queries_c, qrels, update_docs = shared_bc
        retrievers_b = _make_retrievers()
        run_phase1(
            retrievers   = retrievers_b,
            corpus_docs  = corpus_docs,
            eval_queries = queries_b,
            qrels        = qrels,
            config_name  = "config_b",
            output_dir   = args.output_dir,
            update_docs  = update_docs,
            max_queries  = args.max_queries,
            run_updates  = False,
        )
        # pass built retrievers to Config C to avoid rebuilding
        shared_bc = (corpus_docs, queries_b, queries_c, qrels, update_docs, retrievers_b)

    # ── Config C ─────────────────────────────────────────────────────────────
    if args.config_c:
        print("\n" + "="*70)
        print(" CONFIG C  —  Long queries (doc bodies)   +  Full document corpus")
        print("="*70)

        if shared_bc and len(shared_bc) == 6:
            corpus_docs, queries_b, queries_c, qrels, update_docs, retrievers_c = shared_bc
            skip_build = True
        elif shared_bc:
            corpus_docs, queries_b, queries_c, qrels, update_docs = shared_bc
            retrievers_c = _make_retrievers()
            skip_build = False
        else:
            corpus_docs, queries_b, queries_c, qrels, update_docs = load_configs_bc(
                corpus_size=corpus_bc
            )
            retrievers_c = _make_retrievers()
            skip_build = False

        run_phase1(
            retrievers   = retrievers_c,
            corpus_docs  = corpus_docs,
            eval_queries = queries_c,
            qrels        = qrels,
            config_name  = "config_c",
            output_dir   = args.output_dir,
            update_docs  = update_docs,
            max_queries  = args.max_queries,
            run_updates  = not args.no_updates,
            skip_build   = skip_build,
        )

    elapsed = time.perf_counter() - t_global
    print(f"\n{'='*70}")
    print(f" Done in {elapsed/60:.1f} min  |  Results: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



