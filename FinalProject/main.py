"""
main.py  –  Phase 1 evaluation entry point

Usage
-----
    python main.py --all
    python main.py --config-a --config-b
    python main.py --config-c --max-queries 2000 --no-updates
    python main.py --quick            # smoke-test, ~1–2 min

Flags
-----
  --config-a          Run Config A (short queries + short passages)
  --config-b          Run Config B (doc titles   + full docs)
  --config-c          Run Config C (doc bodies   + full docs)
  --all               Equivalent to --config-a --config-b --config-c
  --quick             Smoke-test: 1 000-doc corpus, 10 queries, no updates
  --max-queries N     Cap number of eval queries per config  (default: all)
  --no-updates        Skip the update-time experiment
  --output-dir PATH   Root output directory (default: datasets/evaluations/phase1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project modules are importable when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluation import (
    DEFAULT_OUTPUT,
    DEFAULT_QUICK_OUTPUT,
    QUICK_CORPUS_SIZE,
    QUICK_MAX_QUERIES,
    CORPUS_SIZE_A,
    CORPUS_SIZE_BC,
    _make_retrievers,
    load_config_a,
    load_configs_bc,
    run_phase1,
)


# ---------------------------------------------------------------------------
# Per-config runners
# ---------------------------------------------------------------------------

def run_config_a(args: argparse.Namespace, corpus_size: int) -> None:
    print("\n" + "="*70)
    print(" CONFIG A  —  Short queries  +  Short passage corpus")
    if args.quick:
        print(f"  [QUICK: {corpus_size:,}-doc corpus, "
              f"max {args.max_queries} queries, no updates]")
    print("="*70)

    corpus_docs, eval_queries, qrels, update_docs = load_config_a(
        corpus_size=corpus_size
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


def run_config_b(args: argparse.Namespace, shared: tuple) -> tuple:
    print("\n" + "="*70)
    print(" CONFIG B  —  Short queries (doc titles)  +  Full document corpus")
    print("="*70)

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
        run_updates  = False,   # updates run once after Config C
    )
    # Return built retrievers so Config C can reuse them (skip rebuild)
    return corpus_docs, queries_b, queries_c, qrels, update_docs, retrievers


def run_config_c(args: argparse.Namespace, shared: tuple) -> None:
    print("\n" + "="*70)
    print(" CONFIG C  —  Long queries (doc bodies)   +  Full document corpus")
    print("="*70)

    if len(shared) == 6:
        # Pre-built retrievers available from Config B
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                          # smoke-test (~1 min)
  python main.py --all                            # full run, all 3 configs
  python main.py --config-a --max-queries 500
  python main.py --config-b --config-c --no-updates
        """,
    )
    parser.add_argument("--config-a",    action="store_true", help="Run Config A")
    parser.add_argument("--config-b",    action="store_true", help="Run Config B")
    parser.add_argument("--config-c",    action="store_true", help="Run Config C")
    parser.add_argument("--all",         action="store_true", help="Run all configs")
    parser.add_argument("--quick",       action="store_true",
                        help=f"Smoke-test: {QUICK_CORPUS_SIZE:,}-doc corpus, "
                             f"{QUICK_MAX_QUERIES} queries, no updates. "
                             f"Saves to {DEFAULT_QUICK_OUTPUT}/")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Cap number of eval queries per config")
    parser.add_argument("--no-updates",  action="store_true",
                        help="Skip update-time experiment")
    parser.add_argument("--output-dir",  default=None,
                        help=f"Output root (default: {DEFAULT_OUTPUT}; "
                             f"{DEFAULT_QUICK_OUTPUT} when --quick)")
    args = parser.parse_args()

    if args.all:
        args.config_a = args.config_b = args.config_c = True

    # --quick sets safe defaults
    if args.quick:
        if not (args.config_a or args.config_b or args.config_c):
            args.config_a = True
        if args.max_queries is None:
            args.max_queries = QUICK_MAX_QUERIES
        args.no_updates = True
        if args.output_dir is None:
            args.output_dir = DEFAULT_QUICK_OUTPUT

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT

    if not (args.config_a or args.config_b or args.config_c):
        parser.error("Specify at least one of --config-a/b/c, --all, or --quick")

    os.makedirs(args.output_dir, exist_ok=True)

    corpus_a  = QUICK_CORPUS_SIZE if args.quick else CORPUS_SIZE_A
    corpus_bc = QUICK_CORPUS_SIZE if args.quick else CORPUS_SIZE_BC

    t_global = time.perf_counter()
    shared_bc = None

    if args.config_a:
        run_config_a(args, corpus_size=corpus_a)

    if args.config_b or args.config_c:
        print("\nLoading shared full-docs data (Config B/C)…")
        shared_bc = load_configs_bc(corpus_size=corpus_bc)

    if args.config_b:
        shared_bc = run_config_b(args, shared=shared_bc)

    if args.config_c:
        run_config_c(args, shared=shared_bc)

    elapsed = time.perf_counter() - t_global
    print(f"\n{'='*70}")
    print(f" All done in {elapsed/60:.1f} min.  Results in: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

