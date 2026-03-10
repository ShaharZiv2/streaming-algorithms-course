"""
KnnRetriever – sub-linear retrieval using TF-IDF + TruncatedSVD + BallTree KNN.

Pipeline:
  build_corpus()  – load documents, fit TF-IDF, reduce dimensionality via
                    TruncatedSVD (Latent Semantic Analysis), then build a
                    sklearn NearestNeighbors BallTree index.
  update()        – buffer new documents; rebuild the full index every
                    `rebuild_threshold` insertions.
  retrieve()      – project the query into the low-dim space, query the
                    BallTree for the top-k neighbours, and merge with any
                    un-indexed documents in the buffer via brute-force.
                    Returns list[(doc_id, score)] sorted by descending score.

Complexity:
  Retrieval is O(log n) in the BallTree + O(buffer_size) for the overflow
  buffer, versus O(n) for ClassicRetriever.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from logic.constants import COLLECTION_JSONL
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever


class KnnRetriever(BaseRetriever):
    """TF-IDF + LSA + BallTree KNN retriever."""

    def __init__(
        self,
        top_k: int = 10,
        n_components: int = 150,
        algorithm: str = "ball_tree",
        rebuild_threshold: int = 500,
    ):
        """
        Args:
            top_k:              default number of results to return.
            n_components:       SVD dimensionality (100-300 is typical).
            algorithm:          NearestNeighbors algorithm – "ball_tree",
                                "kd_tree", or "brute".
            rebuild_threshold:  how many buffered updates before a full
                                index rebuild is triggered.
        """
        self.top_k = top_k
        self.n_components = n_components
        self.algorithm = algorithm
        self.rebuild_threshold = rebuild_threshold

        # Core data
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

        # Fitted transformers
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

        # KNN index & reduced matrix (only for indexed docs)
        self.nn_index: NearestNeighbors | None = None
        self.reduced_matrix: np.ndarray | None = None  # (n_indexed × n_components)
        self._n_indexed: int = 0  # how many docs are in the NN index

        # Buffer for docs added after last rebuild
        self._buffer_ids: list[str] = []
        self._buffer_texts: list[str] = []

        # BaseRetriever.__init__ calls self.build_corpus()
        super().__init__()

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int = 10_000):
        """Load `num_initial_documents` from the collection and build the
        full TF-IDF → SVD → BallTree pipeline."""
        print(f"[KnnRetriever] Loading {num_initial_documents} documents…")
        self.doc_ids = []
        self.doc_texts = []

        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= num_initial_documents:
                break
            self.doc_ids.append(doc["key"])
            self.doc_texts.append(doc["data"].strip())
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{num_initial_documents} loaded", end="", flush=True)

        print(f"\n[KnnRetriever] Building index over {len(self.doc_texts)} docs…")
        self._rebuild_index()
        print("[KnnRetriever] Corpus ready.")

    def update(self, document: dict):
        """Add a single {"key": ..., "data": ...} document.

        Documents are buffered and the index is rebuilt once the buffer
        reaches `rebuild_threshold`.
        """
        self.doc_ids.append(document["key"])
        self.doc_texts.append(document["data"].strip())
        self._buffer_ids.append(document["key"])
        self._buffer_texts.append(document["data"].strip())

        if len(self._buffer_ids) >= self.rebuild_threshold:
            print(f"[KnnRetriever] Buffer full ({len(self._buffer_ids)} docs) – rebuilding index…")
            self._rebuild_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs for *query*.

        Queries the BallTree index for indexed docs, then does a small
        brute-force scan over the un-indexed buffer, and merges results.
        """
        if self.vectorizer is None or self.nn_index is None:
            raise RuntimeError("Corpus not built. Call build_corpus() first.")

        k = top_k if top_k is not None else self.top_k

        # --- Transform query ---
        query_tfidf = self.vectorizer.transform([query])       # (1 × vocab) sparse
        query_reduced = self.svd.transform(query_tfidf)        # (1 × n_components) dense

        # --- KNN search on the indexed portion ---
        n_indexed = self.reduced_matrix.shape[0]
        knn_k = min(k, n_indexed)
        distances, indices = self.nn_index.kneighbors(query_reduced, n_neighbors=knn_k)
        # distances are cosine distances → similarity = 1 - distance
        knn_results: list[tuple[str, float]] = [
            (self.doc_ids[idx], 1.0 - dist)
            for dist, idx in zip(distances[0], indices[0])
        ]

        # --- Brute-force over the un-indexed buffer ---
        buffer_results: list[tuple[str, float]] = []
        if self._buffer_texts:
            buf_tfidf = self.vectorizer.transform(self._buffer_texts)
            buf_reduced = self.svd.transform(buf_tfidf)
            scores = cosine_similarity(query_reduced, buf_reduced)[0]
            for i, score in enumerate(scores):
                buffer_results.append((self._buffer_ids[i], float(score)))

        # --- Merge & return top-k ---
        merged = knn_results + buffer_results
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        """Fit TF-IDF vectoriser → TruncatedSVD → NearestNeighbors from scratch."""
        # 1) TF-IDF
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents="unicode",
            max_features=50_000,
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        # 2) Dimensionality reduction (LSA)
        actual_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.reduced_matrix = self.svd.fit_transform(tfidf_matrix)  # (n × components) dense

        # 3) BallTree KNN index
        self.nn_index = NearestNeighbors(
            n_neighbors=self.top_k,
            algorithm=self.algorithm,
            metric="cosine",
        )
        self.nn_index.fit(self.reduced_matrix)
        self._n_indexed = len(self.doc_texts)

        # 4) Clear buffer
        self._buffer_ids.clear()
        self._buffer_texts.clear()

