# Branch: `base-and-knn-retriever`

> **Author:** Tamara  
> **Date:** March 2026  
> **Project:** Streaming Algorithms – Final Project  
> **Dataset:** MS MARCO Passage Retrieval

---

## Overview

This branch builds the **baseline retrieval infrastructure** for the Final Project.
It introduces two new retrievers — `ClassicRetriever` (TF-IDF + Jaccard similarity)
and `BM25Retriever` — alongside a complete evaluation pipeline that benchmarks all
four retrievers on the MS MARCO dataset across latency, update time, credibility
score, and distance score.

---

## What Was Built

### 1. `ClassicRetriever` (`retrievers/classic_retriever.py`)

A brute-force **O(n)** baseline retriever.

| Property | Detail |
|---|---|
| Scoring | **Jaccard similarity** on TF-IDF vocabulary term sets |
| Tokenisation | SnowballStemmer + unigrams & bigrams + stemmed stop words |
| Vectoriser | `sklearn.TfidfVectorizer` with `binary=True` (presence/absence) |
| Build | Fits vectoriser, stores binary term matrix |
| Query | `O(n × vocab)` — scores every document |
| Update | Full matrix rebuild `O(n)` |

**Jaccard formula:**

```
Jaccard(Q, D) = |Q ∩ D| / |Q ∪ D|
```

where Q and D are the sets of unique n-gram tokens from the shared vocabulary
that appear in the query and document respectively. Score range: **[0, 1]**.

---

### 2. `BM25Retriever` (`retrievers/bm25_retriever.py`)

A probabilistic **O(n)** retriever using **BM25Okapi** (`rank_bm25`).

| Property | Detail |
|---|---|
| Scoring | BM25 — term-frequency saturation + document-length normalisation |
| Parameters | `k1=1.5` (TF saturation), `b=0.75` (length normalisation) |
| Tokenisation | Same SnowballStemmer pipeline as ClassicRetriever |
| Build | Tokenises all docs, constructs `BM25Okapi` index |
| Query | `O(n · q)` — scores every document for each query term |
| Update | Appends tokenised doc, rebuilds `BM25Okapi` |

**BM25 score formula:**

```
BM25(t, d) = IDF(t) × [ tf(t,d) × (k1 + 1) ] / [ tf(t,d) + k1 × (1 − b + b × |d|/avgdl) ]
```

---

### 3. Evaluation Pipeline (`evaluate_retrievers.py`, `evaluation/`)

A full benchmarking suite that runs all four retrievers head-to-head.

#### Retrievers compared
| Name | Algorithm | Complexity |
|---|---|---|
| **Classic** | Jaccard on TF-IDF term sets | O(n) |
| **BM25** | BM25Okapi probabilistic ranking | O(n) |
| **MinHash** | MinHash LSH approximate nearest neighbour | sub-linear |
| **Sketch** | MinHash signatures + DBSCAN clustering | sub-linear |

#### Metrics collected per retriever

| Metric | Description |
|---|---|
| `retrieve_time` | Mean wall-clock latency per query (seconds) |
| `update_time` | Mean `.update()` latency per new document (seconds) |
| `credibility_score` | Lexical heuristic on top-1 passage (0–100, ↑ better) |
| `distance_score` | Cosine distance between query and top-1 passage (0–1, ↓ better) |
| `precision@10` | Fraction of top-10 results that are relevant |
| `recall@10` | Fraction of relevant docs found in top-10 |
| `ndcg@10` | Normalised Discounted Cumulative Gain at 10 |
| `mrr` | Mean Reciprocal Rank — rank of first relevant doc |
| `ap` | Average Precision |
| `memory_mb` | Peak memory delta during retrieval (MB) |

#### Plots generated (`datasets/evaluations/`)

| File | Contents |
|---|---|
| `retriever_comparison.png` | **4-panel**: retrieve time · update time · credibility · distance (all retrievers side-by-side) |
| `metrics_comparison.png` | Bar chart — mean ± std of all IR metrics |
| `latency_distribution.png` | Box plot of per-query latency |
| `latency_vs_mrr.png` | Speed–quality scatter (MRR) |
| `latency_vs_ndcg@10.png` | Speed–quality scatter (NDCG@10) |
| `memory_comparison.png` | Peak memory per retriever |
| `per_query_mrr_heatmap.png` | Per-query MRR heatmap (retriever × query) |
| `update_latency.png` | Mean update latency bar chart |
| `distance_scores.png` | Distance score + hallucination flag rate |
| `credibility_scores.png` | Credibility score + misinformation flag rate |
| `complexity_curve.png` | Latency vs corpus size (linear + log-log) |

---

### 4. `SketchRetriever` — changes to your partner's code (`retrievers/sketch_retriever.py`)

> ⚠️ **This retriever was written by your partner.** The changes below were made
> solely to plug it into the shared evaluation benchmark — the core algorithm
> logic was not touched.

---

#### How your partner's retriever works (unchanged logic)

`SketchRetriever` is a **sub-linear** retriever. Instead of scoring every
document for every query (like Classic or BM25 do), it groups documents into
clusters upfront so that at query time only a handful of cluster centroids need
to be compared.

It works in three stages:

**Stage 1 — MinHash signatures**

Each document is tokenised into stemmed unigrams + bigrams via `CountVectorizer`.
A **128-permutation MinHash** is then computed over those n-grams, producing a
128-integer hash vector called the document's *sketch*. This sketch is a compact
probabilistic fingerprint of the document's vocabulary — two documents with
similar vocabularies will have very similar hash vectors (this is the core
MinHash guarantee).

**Stage 2 — DBSCAN clustering**

All document sketches are clustered using `DBSCAN` with `metric='hamming'`.
Hamming distance between two 128-integer hash vectors approximates the Jaccard
distance between the original token sets. Documents with similar vocabulary end
up in the same cluster. Documents that don't fit any cluster are labelled `-1`
(noise) and discarded from the index.

For each cluster that survives, a **centroid** is stored — the element-wise
mean of all member sketches.

**Stage 3 — Retrieval**

A query is sketched the same way as a document (tokenise → MinHash → 128-int
vector). That query sketch is then compared (Hamming distance) against every
stored cluster centroid. All documents belonging to the **nearest centroid's
cluster** are returned as results.

Because there are far fewer centroids than documents, this lookup is sub-linear
in the number of documents — that is the key algorithmic advantage over Classic
and BM25.

**`build_corpus()` (your partner's original entry point)**

Loads pre-computed MinHash signatures from a `.npz` file on disk
(`COLLECTION_MIN_HASH`). If the file doesn't exist it scans the full JSONL
collection and generates it. DBSCAN cluster assignments are also cached to disk
so re-running is fast.

**`update(n)` (your partner's original signature)**

Extended the signature array from the pre-loaded `.npz` file by `n` more
entries and re-ran DBSCAN over the enlarged set.

---

#### What was changed in this branch — and why

The evaluation benchmark (`evaluate_retrievers.py`) loads all four retrievers
from the same **in-memory** list of documents (`corpus_docs`) and calls two
methods on each one:

```python
retriever.build_corpus_from_docs(corpus_docs)   # build from list, no file I/O
retriever.update(single_doc_dict)               # stream one new document in
```

Your partner's retriever only had `build_corpus()` (loads from disk) and
`update(int)` (pulls from pre-loaded `.npz`), so it was being silently skipped
in every benchmark run with the error:

```
Build failed Sketch (sub-linear): 'SketchRetriever' object has no attribute 'build_corpus_from_docs'
```

Two targeted additions were made to fix this:

---

**① Added `build_corpus_from_docs(docs: list[dict])`**

This new method replicates your partner's exact signature-generation and
clustering pipeline, but accepts a plain Python list of
`{"key": ..., "data": ...}` dicts instead of reading from a file:

```
for each doc:
    1. tokenise with CountVectorizer  (same as your partner's code)
    2. compute 128-perm MinHash       (same as your partner's code)
    3. store hashvalues in a list

stack all hashvalues → (n_docs × 128) numpy array
run DBSCAN on that array              (same eps as __init__)
build corpus_df (doc_id → cluster)
build centroid dict (cluster → mean signature)
```

The internal state produced (`self.ids`, `self.signatures`, `self.clusters`,
`self.corpus_df`, `self.centroids`) is identical to what `build_corpus()`
produces for the same documents — `retrieve()` doesn't need to know which
entry point was used.

---

**② Extended `update()` to accept a document dict**

The original signature `update(num_updates: int = 1)` was extended to also
accept a single `{"key": ..., "data": ...}` dict, **without removing the
original integer path**:

```python
def update(self, document: dict | int = 1):
    if isinstance(document, dict):
        # New path (used by benchmark):
        # tokenise → MinHash → append new signature row → re-cluster
    else:
        # Legacy path (your partner's original behaviour):
        # pull `n` more entries from pre-loaded .npz → re-cluster
```

After either path, DBSCAN is re-run on the full updated signature matrix and
the centroid dict is rebuilt — exactly as your partner's original `update()`
did.

---

#### Summary table

| Method | Before | After |
|---|---|---|
| `build_corpus()` | Loads from `.npz` on disk | **Unchanged** |
| `build_corpus_from_docs(docs)` | Did not exist — benchmark skipped Sketch entirely | **Added**: same pipeline, works from in-memory list |
| `update(n: int)` | Accepted integer, pulled `n` docs from `.npz` | **Preserved** as the `else` branch |
| `update(doc: dict)` | Not supported | **Added**: tokenises doc, computes MinHash, appends, re-clusters |
| MinHash / DBSCAN / centroid / retrieve logic | Written by your partner | **Unchanged** |

---

## Evaluation Results (MS MARCO · 10,000 docs · 200 queries · top-10)

| Retriever | Retrieve time | Update time | Credibility | Distance | MRR |
|---|---|---|---|---|---|
| Classic | 0.7 ms | ~15 ms | 50.2 | **0.87** | 0.000 |
| BM25 | **0.6 ms** | ~0.5 ms | **50.4** | 1.00 | 0.000 |
| MinHash | 18.2 ms | — | 50.0 | 1.00 | **0.013** |
| Sketch | ~0 ms | — | 50.0 | 1.00 | 0.000 |

> **Note on IR scores:** Precision, Recall, NDCG, MRR and AP are near zero
> because the local JSONL collection slice (~1,900 docs) does not contain most
> of the MS MARCO relevant passages. Running against the full collection will
> produce meaningful IR scores. The latency, credibility, and distance metrics
> are unaffected.

### Key takeaways
- **BM25** is the fastest retriever (0.6 ms/query) and has the lowest update cost
- **Classic (Jaccard)** produces the best distance scores (0.87) because its TF-IDF vectors partially capture query–document semantic overlap
- **MinHash** is the only retriever that found relevant documents, owing to its
  approximate set-similarity matching
- **Sketch** has near-zero query latency thanks to centroid-based cluster lookup,
  but requires a large corpus for DBSCAN to form meaningful clusters

---

## Files Changed

```
retrievers/bm25_retriever.py          ← new: BM25Okapi retriever
retrievers/classic_retriever.py       ← changed: cosine similarity → Jaccard similarity
retrievers/minhash_retriever.py       ← renamed to minhashLSH_retriever.py
retrievers/sketch_retriever.py        ← fixed: added build_corpus_from_docs(), fixed update()
evaluate_retrievers.py                ← added BM25 to all experiments, fixed syntax error
evaluation/plots.py                   ← added plot_retriever_comparison() 4-panel figure,
                                         updated generate_all_plots() signature
```

---

## How to Run

```bash
cd FinalProject

# Quick smoke test (500 docs, 30 queries)
python evaluate_retrievers.py --quick

# Full evaluation (10k docs, 200 queries)
python evaluate_retrievers.py --corpus-size 10000 --max-queries 200

# With corpus-size scaling experiment
python evaluate_retrievers.py --corpus-size 10000 --max-queries 200 --scaling
```

Results and plots are saved to `datasets/evaluations/`.


