"""
ClassicRetriever – O(n) brute-force retrieval using TF-IDF bag-of-words vectors.

Pipeline:
  build_corpus()  – load the first `num_initial_documents` documents from the
                    JSONL collection and fit a TF-IDF vectoriser over them.
  update()        – append one new document and re-transform the matrix.
  retrieve()      – score every document in the corpus via cosine similarity
                    with the query vector and return the top-k (doc_id, score)
                    pairs.  This is a true O(n) linear scan – no index.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logic.constants import COLLECTION_JSONL
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever


class ClassicRetriever(BaseRetriever):
    """Brute-force TF-IDF retriever – O(n) cosine similarity search."""

    def __init__(self, top_k: int = 10, num_initial_documents: int = 10_000):
        self.top_k = top_k
        self.num_initial_documents = num_initial_documents

        # populated by build_corpus()
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None          # sparse (n_docs × vocab) TF-IDF matrix

        # BaseRetriever.__init__ calls self.build_corpus() unless lazy=True
        super().__init__(lazy=(self.num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None):
        """Load the first `num_initial_documents` entries from the collection
        and build the TF-IDF matrix."""
        n = num_initial_documents if num_initial_documents is not None else self.num_initial_documents
        print(f"[ClassicRetriever] Loading {n} documents…")
        self.doc_ids = []
        self.doc_texts = []

        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= n:
                break
            self.doc_ids.append(doc["key"])
            self.doc_texts.append(doc["data"].strip())
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{n} loaded", end="", flush=True)

        print(f"\n[ClassicRetriever] Fitting TF-IDF vectoriser over {len(self.doc_texts)} docs…")
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents="unicode",
            max_features=50_000,        # cap vocabulary for memory efficiency
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print("[ClassicRetriever] Corpus ready.")

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the TF-IDF corpus from a pre-loaded list of {"key":…,"data":…} dicts.

        Use this instead of build_corpus() when you want to control exactly
        which documents are in the corpus (e.g. seeded with relevant passages).
        """
        print(f"[ClassicRetriever] Indexing {len(docs)} pre-loaded documents…")
        self.doc_ids   = [d["key"] for d in docs]
        self.doc_texts = [d["data"].strip() for d in docs]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents="unicode",
            max_features=50_000,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print("[ClassicRetriever] Corpus ready.")

    def update(self, document: dict):
        """Append a single document dict {"key": ..., "data": ...} to the corpus
        and rebuild the TF-IDF matrix.

        Note: rebuilding the matrix on every update is O(n).  For batch
        insertions prefer collecting documents first and calling
        `_rebuild_matrix()` once.
        """
        self.doc_ids.append(document["key"])
        self.doc_texts.append(document["data"].strip())
        self._rebuild_matrix()

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs for *query*.

        Complexity: O(n × vocab) – every document is scored.
        """
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("Corpus not built. Call build_corpus() first.")

        k = top_k if top_k is not None else self.top_k

        query_vec = self.vectorizer.transform([query])           # (1 × vocab) sparse
        scores = cosine_similarity(query_vec, self.doc_matrix)  # (1 × n_docs)
        scores = scores[0]                                       # flatten to (n_docs,)

        # Partial sort – get the indices of the top-k highest scores (O(n))
        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # np.argpartition gives O(n) partial sort, then we sort just k items
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_matrix(self):
        """Re-fit the vectoriser and rebuild the TF-IDF matrix from scratch."""
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents="unicode",
            max_features=50_000,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

