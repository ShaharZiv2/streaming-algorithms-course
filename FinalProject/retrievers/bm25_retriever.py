"""
bm25_retriever.py – BM25 retriever for fair comparison with SketchRetriever.

Uses the same tokenization as ClassicRetriever and SketchRetriever:
  - SnowballStemmer tokenizer
  - stemmed sklearn stop words

BM25 (Best Match 25) is a probabilistic ranking function that improves on
TF-IDF by:
  1. Term-frequency saturation  – extra occurrences of a term give diminishing
     returns (controlled by k1, default 1.5).
  2. Document-length normalisation – penalises long documents that happen to
     contain many query terms just because they are long (controlled by b,
     default 0.75).

Score formula for term t in document d:
    BM25(t,d) = IDF(t) * tf(t,d) * (k1+1)
                         ─────────────────────────────────
                         tf(t,d) + k1*(1 - b + b*|d|/avgdl)

Where:
  IDF(t)  = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
  tf(t,d) = raw term count in d
  |d|     = token count of d
  avgdl   = average document length across corpus
  N       = total number of documents

Complexity
----------
Build  : O(n · d)  – tokenise n documents
Query  : O(n · q)  – score every document for q query terms  (same O(n) as Classic)
Update : O(n · d)  – full index rebuild (BM25 IDF depends on all docs)
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from logic.constants import COLLECTION_JSONL
from logic.processing.file_utils import load_jsonl
from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer
from retrievers.base_retriever import BaseRetriever

# shared tokenizer – same as ClassicRetriever and SketchRetriever
_tokenizer = StemmingTokenizer()
_stop_words = set(stemmed_stop_words())


def _tokenize(text: str) -> list[str]:
    """Stem + remove stop words, matching Classic/Sketch tokenization."""
    return [t for t in _tokenizer(text) if t not in _stop_words]


class BM25Retriever(BaseRetriever):
    """BM25Okapi retriever – O(n) probabilistic ranking, same tokenization as Sketch."""

    def __init__(self, top_k: int = 10, num_initial_documents: int = 10_000,
                 k1: float = 1.5, b: float = 0.75):
        self.top_k = top_k
        self.num_initial_documents = num_initial_documents
        self.k1 = k1
        self.b = b

        # populated by build_corpus / build_corpus_from_docs
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self._tokenized: list[list[str]] = []  # tokenized docs for BM25
        self._bm25: BM25Okapi | None = None

        super().__init__(lazy=(num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None) -> None:
        n = num_initial_documents if num_initial_documents is not None else self.num_initial_documents
        print(f"[BM25Retriever] Loading {n} documents…")
        docs = []
        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= n:
                break
            docs.append(doc)
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{n} loaded", end="", flush=True)
        print()
        self._index_docs(docs)

    def build_corpus_from_docs(self, docs: list[dict]) -> None:
        """Build BM25 index from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[BM25Retriever] Indexing {len(docs)} pre-loaded documents…")
        self._index_docs(docs)

    def update(self, document: dict) -> None:
        """Append one document and rebuild the BM25 index (IDF must be recomputed)."""
        self.doc_ids.append(document["key"])
        text = document["data"].strip()
        self.doc_texts.append(text)
        self._tokenized.append(_tokenize(text))
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return top-k (doc_id, bm25_score) pairs for *query*.

        Complexity: O(n · q) – scores every document for each query term.
        """
        if self._bm25 is None:
            return []

        k = top_k if top_k is not None else self.top_k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)  # ndarray shape (n,)

        if k >= len(scores):
            import numpy as np
            top_indices = np.argsort(scores)[::-1]
        else:
            import numpy as np
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_docs(self, docs: list[dict]) -> None:
        self.doc_ids = [d["key"] for d in docs]
        self.doc_texts = [d["data"].strip() for d in docs]
        self._tokenized = [_tokenize(t) for t in self.doc_texts]
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)
        print(f"[BM25Retriever] Index ready ({len(self.doc_ids)} docs, k1={self.k1}, b={self.b}).")

