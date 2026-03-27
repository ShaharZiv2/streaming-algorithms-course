"""
run_phase1.py  –  Phase 1 evaluation runner

Usage
-----
    python run_phase1.py --all
    python run_phase1.py --config-a --config-b
    python run_phase1.py --config-c --max-queries 2000 --no-updates

Flags
-----
  --config-a          Run Config A (short queries + short passages)
  --config-b          Run Config B (doc titles   + full docs)
  --config-c          Run Config C (doc bodies   + full docs)
  --all               Equivalent to --config-a --config-b --config-c
  --max-queries N     Cap number of eval queries per config  (default: all)
  --no-updates        Skip the update-time experiment
  --output-dir PATH   Root output directory (default: datasets/evaluations/phase1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure we can import project modules regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.phase1_eval import (
    N_UPDATE_DOCS,
    UPDATE_CORPUS_N,
    load_config_a,
    load_configs_bc,
    run_phase1,
)
from retrievers.bm25_retriever import BM25Retriever
from retrievers.classic_retriever import ClassicRetriever
from retrievers.min_hash_dbscan_retriever import MinHashDbscanRetriever
from retrievers.min_hash_lsh_retriever import MinHashLshRetriever
from retrievers.prob_min_hash_dbscan_retriever import ProbMinHashDbscanRetriever
from retrievers.prob_min_hash_lsh_retriever import ProbMinHashLshRetriever


DEFAULT_OUTPUT = "datasets/evaluations/phase1"


def _make_retrievers() -> dict:
    """Instantiate all 6 retrievers in lazy mode (no auto-build)."""
    return {
        "Classic":          ClassicRetriever(top_k=10, num_initial_documents=0),
        "BM25":             BM25Retriever(top_k=10, num_initial_documents=0),
        "MinHashDBSCAN":    MinHashDbscanRetriever(corpus_initial_size=0),
        "MinHashLSH":       MinHashLshRetriever(corpus_initial_size=0),
        "ProbMinHashDBSCAN":ProbMinHashDbscanRetriever(corpus_initial_size=0),
        "ProbMinHashLSH":   ProbMinHashLshRetriever(corpus_initial_size=0),
    }


def run_config_a(args: argparse.Namespace) -> None:
    print("\n" + "="*70)
    print(" CONFIG A  —  Short queries  +  Short passage corpus")
    print("="*70)

    corpus_docs, eval_queries, qrels, update_docs = load_config_a()

    retrievers = _make_retrievers()
    run_phase1(
        retrievers   = retrievers,
        corpus_docs  = corpus_docs,
        eval_queries = eval_queries,
        qrels        = qrels,
        config_name  = "config_a",
        output_dir   = args.output_dir,
        update_docs  = update_docs,
        max_queries  = args.max_queries,
        run_updates  = not args.no_updates,
    )


def run_config_b(args: argparse.Namespace, shared=None) -> tuple:
    print("\n" + "="*70)
    print(" CONFIG B  —  Short queries (doc titles)  +  Full document corpus")
    print("="*70)

    if shared is None:
        corpus_docs, queries_b, queries_c, qrels, update_docs = load_configs_bc()
    else:
        corpus_docs, queries_b, queries_c, qrels, update_docs = shared

    retrievers = _make_retrievers()
    run_phase1(
        retrievers   = retrievers,
        corpus_docs  = corpus_docs,
        eval_queries = queries_b,
        qrels        = qrels,
        config_name  = "config_b",
        output_dir   = args.output_dir,
        update_docs  = update_docs,
        max_queries  = args.max_queries,
        run_updates  = False,          # updates run once after Config C
    )
    # Return retrievers (already built) so Config C can reuse them
    return corpus_docs, queries_b, queries_c, qrels, update_docs, retrievers


def run_config_c(args: argparse.Namespace, shared=None) -> None:
    print("\n" + "="*70)
    print(" CONFIG C  —  Long queries (doc bodies)   +  Full document corpus")
    print("="*70)

    if shared is None:
        corpus_docs, queries_b, queries_c, qrels, update_docs = load_configs_bc()
        retrievers = _make_retrievers()
        skip_build = False
    else:
        # shared may include pre-built retrievers from Config B
        if len(shared) == 6:
            corpus_docs, queries_b, queries_c, qrels, update_docs, retrievers = shared
            skip_build = True
        else:
            corpus_docs, queries_b, queries_c, qrels, update_docs = shared
            retrievers = _make_retrievers()
            skip_build = False

    run_phase1(
        retrievers   = retrievers,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Evaluation Runner")
    parser.add_argument("--config-a",    action="store_true")
    parser.add_argument("--config-b",    action="store_true")
    parser.add_argument("--config-c",    action="store_true")
    parser.add_argument("--all",         action="store_true", help="Run all configs")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Cap number of eval queries per config")
    parser.add_argument("--no-updates",  action="store_true",
                        help="Skip update-time experiment")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.all:
        args.config_a = args.config_b = args.config_c = True

    if not (args.config_a or args.config_b or args.config_c):
        parser.error("Specify at least one config flag, or --all")

    os.makedirs(args.output_dir, exist_ok=True)
    t_global = time.perf_counter()

    shared_bc = None

    if args.config_a:
        run_config_a(args)

    if args.config_b or args.config_c:
        # load fulldocs once — reuse for both B and C
        print("\nLoading shared full-docs data (used for Config B and C)…")
        shared_bc = load_configs_bc()

    if args.config_b:
        shared_bc = run_config_b(args, shared=shared_bc)
        # shared_bc now contains pre-built retrievers (6-tuple)

    if args.config_c:
        run_config_c(args, shared=shared_bc)

    elapsed = time.perf_counter() - t_global
    print(f"\n{'='*70}")
    print(f" All done in {elapsed/60:.1f} min.  Results in: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



