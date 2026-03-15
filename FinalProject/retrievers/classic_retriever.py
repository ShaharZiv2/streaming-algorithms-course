"""
ClassicRetriever – O(n) brute-force retrieval using TF-IDF bag-of-words vectors
with Jaccard similarity scoring.

Uses the same tokenization as SketchRetriever for a fair comparison:
  - SnowballStemmer tokenizer
  - unigram + bigram (ngram_range=(1, 2))
  - stemmed sklearn stop words

Jaccard similarity measures the overlap between the *set* of vocabulary terms
present in the query and in each document:

    Jaccard(Q, D) = |Q ∩ D| / |Q ∪ D|

where Q and D are the sets of unique n-gram tokens (drawn from the shared
TF-IDF vocabulary) that appear in the query and the document respectively.
The score is always in [0, 1]:  0 = no shared terms, 1 = identical term sets.

Pipeline:
  build_corpus()  – load the first `num_initial_documents` documents from the
                    JSONL collection, fit a TF-IDF vectoriser to build the
                    shared vocabulary, then store a binary term-set per doc.
  update()        – append one new document and rebuild the vocabulary / sets.
  retrieve()      – score every document via Jaccard and return the top-k
                    (doc_id, score) pairs.  This is a true O(n) linear scan.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """Compute Jaccard similarity between *query_vec* and every row of *doc_matrix*.

    Both inputs are binary sparse matrices (values are 0 or 1).

    Jaccard(Q, D) = |Q ∩ D| / |Q ∪ D|
                 = dot(Q, D) / (|Q| + |D| - dot(Q, D))

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
    """Brute-force TF-IDF retriever – O(n) Jaccard similarity search."""

    def __init__(self, top_k: int = 10, num_initial_documents: int = 10_000):
        self.top_k = top_k
        self.num_initial_documents = num_initial_documents

        # populated by build_corpus()
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None   # sparse binary (n_docs × vocab) matrix

        super().__init__(lazy=(self.num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None):
        """Load the first `num_initial_documents` entries from the collection
        and build the binary term matrix."""
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

        print(f"\n[ClassicRetriever] Fitting vectoriser over {len(self.doc_texts)} docs…")
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print("[ClassicRetriever] Corpus ready.")

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build the corpus from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[ClassicRetriever] Indexing {len(docs)} pre-loaded documents…")
        self.doc_ids   = [d["key"] for d in docs]
        self.doc_texts = [d["data"].strip() for d in docs]
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print("[ClassicRetriever] Corpus ready.")

    def update(self, document: dict):
        """Append a single document and rebuild the term matrix (O(n))."""
        self.doc_ids.append(document["key"])
        self.doc_texts.append(document["data"].strip())
        self._rebuild_matrix()

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs for *query*.

        Scoring: Jaccard similarity on the shared TF-IDF vocabulary term sets.
        Complexity: O(n × vocab) – every document is scored.
        """
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("Corpus not built. Call build_corpus() first.")

        k = top_k if top_k is not None else self.top_k

        query_vec = self.vectorizer.transform([query])   # (1 × vocab) sparse binary
        scores = _jaccard_scores(query_vec, self.doc_matrix)  # (n_docs,)

        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_matrix(self):
        """Re-fit the vectoriser and rebuild the binary term matrix from scratch."""
        self.vectorizer = _make_vectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

