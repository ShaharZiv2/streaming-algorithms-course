# Final Project — Sketch-Based Document Retrieval

Benchmarking **streaming-algorithm sketches** (MinHash, ProbMinHash) as the
backbone of a document retriever, evaluated against classical TF-IDF and BM25
baselines on the [MS MARCO](https://microsoft.github.io/msmarco/) passage and
document corpora.

---

## Overview

| Question | Answer |
|---|---|
| Dataset | MS MARCO (passages + full documents) |
| Corpus size | 100 000 documents (1 000 for `--quick`) |
| Retrievers | 6 (see below) |
| Evaluation configs | 3 (A / B / C) |
| Metrics | Hit Rate, MRR, MAP, P/R/F1@all, NDCG, memory, update latency |

---

## Retrievers

| Name | Description |
|---|---|
| **Classic** | TF-IDF (unigram + bigram) + DBSCAN clustering; Jaccard similarity at query time |
| **BM25** | BM25Okapi with the same stemming tokeniser as Classic |
| **MinHashDBSCAN** | MinHash sketches (128 permutations) + DBSCAN for candidate pruning |
| **MinHashLSH** | MinHash sketches + Locality-Sensitive Hashing bucket lookup |
| **ProbMinHashDBSCAN** | TF-IDF-weighted ProbMinHash (ProbMinHash4) + DBSCAN |
| **ProbMinHashLSH** | TF-IDF-weighted ProbMinHash (ProbMinHash4) + LSH bucket lookup |

All retrievers share the same **SnowballStemmer tokeniser** and the same
`update()` / `retrieve()` interface (`BaseRetriever`).

---

## Evaluation Configs

| Config | Query source | Document corpus |
|---|---|---|
| **A** | Short passage queries (~5 words) | Passages (~52 words) — 100 k docs |
| **B** | Document *titles* (~5 words) | Full documents (~545 words) — 100 k docs |
| **C** | Document *bodies* (~545 words) | Full documents (~545 words) — 100 k docs |

Each config measures:
- **Accuracy** — Hit Rate @all, MRR, MAP, P/R/F1/NDCG @all  
- **Memory** — peak RSS tracked with `tracemalloc` (MB)  
- **Update latency** — time to insert 100 new documents one-by-one (s)

---

## Project Structure

```
FinalProject/
├── main.py                        # Evaluation entry point
├── pyproject.toml                 # Poetry dependencies
│
├── evaluation/
│   ├── evaluation.py              # Core runner: load data, run_phase1(), metrics loop
│   ├── metrics.py                 # P@k, R@k, MRR, NDCG, AP, Hit Rate, F1
│   ├── benchmark.py               # Timing / memory utilities
│   └── generate_plots.py          # Generate all 10 plots from saved CSVs
│
├── retrievers/
│   ├── base_retriever.py          # Abstract base (build_corpus / update / retrieve)
│   ├── classic_retriever.py       # TF-IDF + DBSCAN + Jaccard
│   ├── bm25_retriever.py          # BM25Okapi
│   ├── min_hash_lsh_retriever.py  # MinHash + LSH
│   ├── min_hash_dbscan_retriever.py
│   ├── prob_min_hash_lsh_retriever.py
│   └── prob_min_hash_dbscan_retriever.py
│
├── logic/
│   ├── constants.py               # All file paths
│   ├── prob_min_hash.py           # ProbMinHash4 implementation
│   ├── tf_idf.py                  # TF-IDF build / load helpers
│   ├── stemming_utils.py          # SnowballStemmer tokeniser + stop words
│   └── preprocessing_utils.py
│
└── datasets/
    ├── jsonl/                     # collection.jsonl, queries.jsonl, qrels.jsonl
    ├── tsv/                       # fulldocs.tsv.gz (full document corpus)
    └── evaluations/
        ├── experiment_results/    # ★ Data & plots used in the final report
        │   └── raw_results/
        │       ├── config_a/      # accuracy.csv, memory.csv, summary.csv, update_time.csv
        │       ├── config_b/
        │       ├── config_c/
        │       └── plots/         # 10 PNG visualisations
        └── raw_results/           # Re-runnable output (same structure as above)
```

---

## Runtime

All timings are on a 100 000-document corpus (Apple M-series / equivalent x86 hardware).

### Overall wall-clock estimates

| Command | Corpus | Queries | Approx. time |
|---|---|---|---|
| `--quick` | 1 000 docs | 10 | ~1–2 min |
| `--config-a` | 100 000 passages | 1 000 | ~20–40 min |
| `--config-b` | 100 000 full docs | 1 000 | ~30–60 min |
| `--config-c` | 100 000 full docs | 1 000 | ~60–90 min |
| `--all` | both corpora | 3 × 1 000 | ~2–4 hours |

> **Tip:** running `--config-b --config-c` together shares the index build,
> saving ~30–60 min vs running each independently.

### Measured update latency (100-doc experiment)

Average time to insert **one** new document into a live index (measured from
`datasets/evaluations/raw_results/`):

#### Config A — passage corpus (~52 words / doc)

| Retriever | Mean / doc | 100-doc total |
|---|---|---|
| ProbMinHashDBSCAN | < 1 ms | < 0.1 s |
| MinHashLSH | ~1 ms | ~0.1 s |
| ProbMinHashLSH | ~4 ms | ~0.4 s |
| BM25 | ~22 ms | ~2 s |
| MinHashDBSCAN | ~147 ms | ~15 s |
| Classic | ~770 ms | ~77 s |

#### Config C — full-document corpus (~545 words / doc)

| Retriever | Mean / doc | 100-doc total |
|---|---|---|
| ProbMinHashDBSCAN | < 1 ms | < 0.1 s |
| MinHashLSH | ~5 ms | ~0.5 s |
| ProbMinHashLSH | ~12 ms | ~1 s |
| MinHashDBSCAN | ~137 ms | ~14 s |
| BM25 | ~434 ms | ~43 s |
| Classic | ~16.8 s | ~28 min |

> Classic's update cost scales with corpus size because it rebuilds the TF-IDF
> index and DBSCAN clusters on every insertion. Sketch-based retrievers
> (MinHashLSH, ProbMinHashLSH) maintain roughly constant update time regardless
> of document length.

---

## Setup

### 1. Install dependencies

```bash
pip install poetry
poetry install
```

### 2. Dataset

The MS MARCO files are **not** included in the repo (too large). Download them:

```bash
# Passages (Config A)
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz -P datasets/tsv/
tar -xf datasets/tsv/collection.tar.gz -C datasets/tsv/

# Full documents (Configs B/C)
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz -P datasets/tsv/ # (already in tsv dir)

# Queries and qrels
# Place collection.jsonl, queries.jsonl, qrels.jsonl in datasets/jsonl/
```

---

## Running Experiments

All commands should be run from the `FinalProject/` directory.

### Quick smoke-test (~1–2 min)

```bash
python main.py --quick
```

Runs Config A with a 1 000-doc corpus, 10 queries, no update experiment.
Results saved to `datasets/evaluations/phase1_quick/`.

### Full evaluation

```bash
# All three configs
python main.py --all

# Individual configs
python main.py --config-a
python main.py --config-b --config-c

# Cap queries and skip update experiment
python main.py --all --max-queries 500 --no-updates

# Custom output directory
python main.py --all --output-dir datasets/evaluations/my_run
```

**Config B and C share the same corpus** — running them together avoids
rebuilding the index twice.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--config-a` | off | Run Config A |
| `--config-b` | off | Run Config B |
| `--config-c` | off | Run Config C |
| `--all` | off | Equivalent to `--config-a --config-b --config-c` |
| `--quick` | off | Smoke-test (1 k docs, 10 queries, no updates) |
| `--max-queries N` | all | Cap eval queries per config |
| `--no-updates` | off | Skip update-latency experiment |
| `--output-dir PATH` | `datasets/evaluations/raw_results` | Output root |

---

## Output Format

> **Report data:** the CSVs and plots used in the final report are committed at
> `datasets/evaluations/experiment_results/raw_results/`.
> To regenerate the report plots from that data:
> ```bash
> python evaluation/generate_plots.py --base datasets/evaluations/experiment_results/raw_results
> ```

Each config writes four CSVs to `<output-dir>/config_{a,b,c}/`:

| File | Contents |
|---|---|
| `summary.csv` | Per-retriever aggregate metrics (mean ± std for all measures) |
| `accuracy.csv` | Per-query rows: retriever, query\_id, mrr, hit\_rate, n\_results, … |
| `memory.csv` | Per-retriever peak memory (MB) per query |
| `update_time.csv` | Per-document update latency (s) |

---

## Generating Plots

```bash
# Default: reads from datasets/evaluations/raw_results/
python evaluation/generate_plots.py

# Custom base directory
python evaluation/generate_plots.py --base datasets/evaluations/my_run
```

Produces 10 PNGs in `<base>/plots/`:

| Plot | File |
|---|---|
| Hit Rate @all — all configs grouped | `hit_rate_all_configs.png` |
| MRR — all configs grouped | `mrr_all_configs.png` |
| Accuracy heatmap (Config A) | `heatmap_config_a.png` |
| Accuracy heatmap (Config B) | `heatmap_config_b.png` |
| Accuracy heatmap (Config C) | `heatmap_config_c.png` |
| Peak memory usage | `memory_comparison.png` |
| Update latency (median + p95, log scale) | `update_latency.png` |
| MRR vs memory scatter | `accuracy_vs_memory.png` |
| Per-query MRR distribution (violin) | `mrr_violin.png` |
| Mean result-set size | `n_results.png` |

---

## Running Tests

```bash
poetry run pytest
```

Unit tests for individual retrievers live in `tests/`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | TF-IDF, DBSCAN, cosine similarity |
| `datasketch` | MinHash, MinHashLSH |
| `rank-bm25` | BM25Okapi |
| `nltk` | SnowballStemmer |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` | Plotting |
| `xxhash` | Fast hashing for ProbMinHash |
| `joblib` | Parallel preprocessing |




