"""
ANNRetriever – sub-linear retrieval using TF-IDF + TruncatedSVD + BallTree ANN.

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


class ANNRetriever(BaseRetriever):
    """TF-IDF + LSA + BallTree ANN retriever."""

    def __init__(
        self,
        top_k: int = 10,
        n_components: int = 150,
        algorithm: str = "ball_tree",
        rebuild_threshold: int = 500,
        num_initial_documents: int = 10_000,
    ):
        """
        Args:
            top_k:                  default number of results to return.
            n_components:           SVD dimensionality (100-300 is typical).
            algorithm:              NearestNeighbors algorithm – "ball_tree",
                                    "kd_tree", or "brute".
            rebuild_threshold:      how many buffered updates before a full
                                    index rebuild is triggered.
            num_initial_documents:  corpus size to load on init.
        """
        self.top_k = top_k
        self.n_components = n_components
        self.algorithm = algorithm
        self.rebuild_threshold = rebuild_threshold
        self.num_initial_documents = num_initial_documents

        # Core data
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

        # Fitted transformers
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

        # ANN index & reduced matrix (only for indexed docs)
        self.nn_index: NearestNeighbors | None = None
        self.reduced_matrix: np.ndarray | None = None  # (n_indexed × n_components)
        self._n_indexed: int = 0  # how many docs are in the NN index

        # Buffer for docs added after last rebuild
        self._buffer_ids: list[str] = []
        self._buffer_texts: list[str] = []

        # BaseRetriever.__init__ calls self.build_corpus() unless lazy=True
        super().__init__(lazy=(self.num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None):
        """Load `num_initial_documents` from the collection and build the
        full TF-IDF → SVD → BallTree pipeline."""
        n = num_initial_documents if num_initial_documents is not None else self.num_initial_documents
        print(f"[ANNRetriever] Loading {n} documents…")
        self.doc_ids = []
        self.doc_texts = []

        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= n:
                break
            self.doc_ids.append(doc["key"])
            self.doc_texts.append(doc["data"].strip())
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{n} loaded", end="", flush=True)

        print(f"\n[ANNRetriever] Building index over {len(self.doc_texts)} docs…")
        self._rebuild_index()
        print("[ANNRetriever] Corpus ready.")

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the TF-IDF → SVD → BallTree index from a pre-loaded list of docs.

        Use this instead of build_corpus() when you want to control exactly
        which documents are in the corpus (e.g. seeded with relevant passages).
        """
        print(f"[ANNRetriever] Indexing {len(docs)} pre-loaded documents…")
        self.doc_ids   = [d["key"] for d in docs]
        self.doc_texts = [d["data"].strip() for d in docs]
        self._rebuild_index()
        print("[ANNRetriever] Corpus ready.")

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
            print(f"[ANNRetriever] Buffer full ({len(self._buffer_ids)} docs) – rebuilding index…")
            self._rebuild_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs for *query*.

        Queries the BallTree index for indexed docs, then does a small
        brute-force scan over the un-indexed buffer, and merges results.
        """
        if self.vectorizer is None or self.nn_index is None:
            raise RuntimeError("Corpus not built. Call build_corpus() first.")

        k = top_k if top_k is not None else self.top_k

        # --- Transform query and L2-normalise (must match index vectors) ---
        query_tfidf = self.vectorizer.transform([query])       # (1 × vocab) sparse
        query_reduced = self.svd.transform(query_tfidf)        # (1 × n_components) dense
        q_norm = np.linalg.norm(query_reduced)
        if q_norm > 0:
            query_reduced = query_reduced / q_norm

        # --- ANN search on the indexed portion ---
        n_indexed = self.reduced_matrix.shape[0]
        ann_k = min(k, n_indexed)
        distances, indices = self.nn_index.kneighbors(query_reduced, n_neighbors=ann_k)
        # euclidean dist on unit vectors → cosine_sim = 1 - dist²/2
        ann_results: list[tuple[str, float]] = [
            (self.doc_ids[idx], 1.0 - (dist ** 2) / 2.0)
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
        merged = ann_results + buffer_results
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        """Fit TF-IDF vectoriser → TruncatedSVD → L2-normalise → NearestNeighbors from scratch.

        ball_tree does not support cosine metric directly, so we L2-normalise
        the reduced vectors first. Euclidean distance on unit vectors is
        monotonically related to cosine similarity:
            cosine_sim = 1 - (euclidean_dist² / 2)
        """
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
        reduced = self.svd.fit_transform(tfidf_matrix)  # (n × components) dense

        # 3) L2-normalise so euclidean ≡ cosine
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.reduced_matrix = reduced / norms  # unit vectors

        # 4) BallTree ANN index on normalised vectors with euclidean metric
        self.nn_index = NearestNeighbors(
            n_neighbors=min(self.top_k, len(self.doc_texts)),
            algorithm=self.algorithm,
            metric="euclidean",
        )
        self.nn_index.fit(self.reduced_matrix)
        self._n_indexed = len(self.doc_texts)

        # 5) Clear buffer
        self._buffer_ids.clear()
        self._buffer_texts.clear()

