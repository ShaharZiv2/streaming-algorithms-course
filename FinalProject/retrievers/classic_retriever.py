"""
ClassicRetriever – TF-IDF bag-of-words retriever with DBSCAN clustering.

Uses the same tokenization as SketchRetriever for a fair comparison:
  - SnowballStemmer tokenizer
  - unigram + bigram (ngram_range=(1, 2))
  - stemmed sklearn stop words

Pipeline:
  build_corpus()  – fit TF-IDF, run DBSCAN on the document vectors to form
                    topical clusters, compute a dense centroid per cluster.
  retrieve()      – transform query, find nearest centroid (O(k_clusters)),
                    score only documents in that cluster with Jaccard similarity.
                    Falls back to full O(n) scan when no clusters exist.

DBSCAN clustering makes retrieval sub-linear in practice:
  - Each cluster groups topically similar documents.
  - At query time only O(n / k_clusters) documents are scored instead of all n.
  - Noise points (label=-1) form a catch-all "unclustered" bucket that is used
    as fallback when the nearest cluster contains fewer than top_k results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from logic.constants import COLLECTION_JSONL
from logic.processing.file_utils import load_jsonl
from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
from retrievers.base_retriever import BaseRetriever


def _make_vectorizer() -> TfidfVectorizer:
    """Return a TfidfVectorizer with the same tokenization as SketchRetriever.

    Both retrievers share:
      - SnowballStemmer tokenizer
      - unigrams + bigrams  (ngram_range=(1, 2))
      - stemmed sklearn stop-word list
      - vocabulary capped at 50 000 features for memory efficiency
    """
    return TfidfVectorizer(
        tokenizer=StemmingTokenizer(),
        ngram_range=(1, 2),
        stop_words=stemmed_stop_words(),
        max_features=50_000,
        sublinear_tf=True,
        binary=True,   # presence/absence only – Jaccard works on sets
    )


def _jaccard_scores(query_vec, doc_matrix) -> np.ndarray:
    """Jaccard similarity between query_vec and every row of doc_matrix (binary sparse).

    Jaccard(Q, D) = |Q ∩ D| / |Q ∪ D| = dot(Q,D) / (|Q| + |D| - dot(Q,D))

    Returns a 1-D float array of length n_docs.
    """
    # intersection size: dot product of binary vectors
    intersection = doc_matrix.dot(query_vec.T).toarray().ravel().astype(float)  # (n,)

    query_len = float(query_vec.nnz)                      # |Q|
    doc_lens  = np.diff(doc_matrix.indptr).astype(float)  # |D| for each doc

    union = query_len + doc_lens - intersection
    # avoid division by zero (empty doc or empty query)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(union > 0, intersection / union, 0.0)
    return scores


class ClassicRetriever(BaseRetriever):
    """TF-IDF retriever accelerated by DBSCAN clustering.

    Parameters
    ----------
    top_k                 : number of results to return
    num_initial_documents : documents to auto-load on init; 0 = lazy
    dbscan_eps            : DBSCAN epsilon (cosine distance); controls cluster
                            tightness. Lower = tighter, more clusters.
    dbscan_min_samples    : minimum cluster size; None = auto (max(2, n//200))
    """

    def __init__(
        self,
        top_k: int = 10,
        num_initial_documents: int = 10_000,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int | None = None,
    ):
        self.top_k = top_k
        self.num_initial_documents = num_initial_documents
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # populated by build_corpus / build_corpus_from_docs
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None          # sparse binary  (n × vocab)
        self.doc_matrix_norm = None     # L2-normalised dense for centroid search

        # DBSCAN outputs
        self.clusters: np.ndarray | None = None   # (n,) cluster label per doc
        self.corpus_df: pd.DataFrame | None = None
        self.centroids: dict[int, np.ndarray] = {}  # cluster_label → dense centroid

        super().__init__(lazy=(self.num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None):
        """Load the first `num_initial_documents` entries from the collection
        and build the binary term matrix."""
        n = num_initial_documents if num_initial_documents is not None else self.num_initial_documents
        print(f"[ClassicRetriever] Loading {n} documents…")
        self.doc_ids, self.doc_texts = [], []
        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= n:
                break
            self.doc_ids.append(doc["key"])
            self.doc_texts.append(doc["data"].strip())
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{n} loaded", end="", flush=True)
        print(f"\n[ClassicRetriever] Fitting vectoriser over {len(self.doc_texts)} docs…")
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        self._fit_dbscan()
        print("[ClassicRetriever] Corpus ready.")

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the corpus from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[ClassicRetriever] Indexing {len(docs)} pre-loaded documents…")
        self.doc_ids   = [d["key"] for d in docs]
        self.doc_texts = [d["data"].strip() for d in docs]
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        self._fit_dbscan()
        print("[ClassicRetriever] Corpus ready.")

    def update(self, document: dict):
        """Append a single document and rebuild the index."""
        self.doc_ids.append(document["key"])
        self.doc_texts.append(document["data"].strip())
        self._rebuild_matrix()

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs using DBSCAN-accelerated Jaccard search.

        1. Transform query into TF-IDF binary vector.
        2. Find the nearest cluster centroid (cosine similarity, O(k_clusters)).
        3. Score only documents in that cluster with exact Jaccard (O(cluster_size)).
        4. If results < top_k, extend with noise-bucket documents, then fall back
           to full scan as a last resort.
        """
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("Corpus not built. Call build_corpus() first.")

        k = top_k if top_k is not None else self.top_k
        query_vec = self.vectorizer.transform([query])   # (1 × vocab) sparse binary

        # ------------------------------------------------------------------
        # Fast path: DBSCAN cluster search
        # ------------------------------------------------------------------
        if self.centroids:
            best_cluster = self._find_best_cluster(query_vec)

            # Gather candidate indices: best cluster first, then noise
            candidate_idxs = self._cluster_indices(best_cluster)
            if len(candidate_idxs) < k:
                noise_idxs = self._cluster_indices(-1)
                candidate_idxs = np.concatenate([candidate_idxs, noise_idxs])

            if len(candidate_idxs) >= k:
                sub_matrix = self.doc_matrix[candidate_idxs]
                scores = _jaccard_scores(query_vec, sub_matrix)
                top_local = np.argsort(scores)[::-1][:k]
                return [(self.doc_ids[candidate_idxs[i]], float(scores[i]))
                        for i in top_local]

        # ------------------------------------------------------------------
        # Fallback: full O(n) scan (when no clusters formed or corpus tiny)
        # ------------------------------------------------------------------
        scores = _jaccard_scores(query_vec, self.doc_matrix)
        top_indices = (np.argsort(scores)[::-1][:k]
                       if k >= len(scores)
                       else np.argpartition(scores, -k)[-k:][
                           np.argsort(scores[np.argpartition(scores, -k)[-k:]])[::-1]])
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # DBSCAN helpers
    # ------------------------------------------------------------------

    def _fit_dbscan(self):
        """Run DBSCAN on L2-normalised sparse TF-IDF vectors using cosine metric.

        Works directly on the sparse matrix (no .toarray()) so memory stays low
        even for large vocabularies.  sklearn's brute-force algorithm supports
        sparse inputs with cosine metric natively.
        """
        n = len(self.doc_ids)
        min_samples = self.dbscan_min_samples if self.dbscan_min_samples is not None \
                      else max(2, n // 200)

        # Normalise sparse matrix in-place (stays sparse)
        self.doc_matrix_norm = normalize(self.doc_matrix, norm="l2", copy=True)

        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=min_samples,
            metric="cosine",
            algorithm="brute",   # only algorithm that supports sparse cosine
            n_jobs=-1,
        )

        print(f"[ClassicRetriever] Running DBSCAN (eps={self.dbscan_eps}, "
              f"min_samples={min_samples}, n={n})…")
        self.clusters = dbscan.fit_predict(self.doc_matrix_norm)
        self.corpus_df = pd.DataFrame({"doc_id": self.doc_ids, "cluster": self.clusters})

        # Compute sparse centroid per cluster, store as dense 1-D array
        self.centroids = {}
        for label in np.unique(self.clusters):
            if label == -1:
                continue
            mask = self.clusters == label
            # mean of sparse rows → dense centroid vector
            centroid = np.asarray(self.doc_matrix_norm[mask].mean(axis=0)).ravel()
            self.centroids[label] = centroid

        n_clusters = len(self.centroids)
        n_noise    = int(np.sum(self.clusters == -1))
        print(f"[ClassicRetriever] DBSCAN done: {n_clusters} clusters, "
              f"{n_noise} noise points.")

    def _find_best_cluster(self, query_vec) -> int:
        """Return the cluster label whose centroid is most similar to query_vec."""
        # L2-normalise the sparse query vector, then convert just that 1 row to dense
        q_norm = normalize(query_vec, norm="l2")
        q_dense = np.asarray(q_norm.todense()).ravel()   # shape (vocab,)
        best_label, best_sim = -1, -1.0
        for label, centroid in self.centroids.items():
            sim = float(np.dot(q_dense, centroid))   # cosine (both L2-normed)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label

    def _cluster_indices(self, label: int) -> np.ndarray:
        """Return row indices in doc_matrix that belong to *label*."""
        return np.where(self.clusters == label)[0]

    def _rebuild_matrix(self):
        """Re-fit vectoriser, rebuild matrix, re-run DBSCAN."""
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        self._fit_dbscan()
