"""
minhashLSH_retriever.py - Approximate nearest-neighbour retriever using MinHash LSH.

Why was MinHashLSH slower than MinHash before?
  1. lsh_threshold=0.1 is extremely tight -- LSH buckets were nearly always empty
     so EVERY query fell through to the O(n) Python fallback.
  2. The O(n) fallback iterated over a Python list of datasketch MinHash objects
     calling sig.hashvalues per document -- very slow Python-level loops.
Fixes applied:
  - Default threshold raised to 0.5 (tune lower if recall matters more).
  - All hashvalues stored as a pre-built numpy uint64 matrix at index time.
  - Fallback uses one numpy broadcast over the whole matrix: O(n) at BLAS speed.
  - Re-rank of LSH candidates also uses the numpy matrix, not .jaccard() calls.
"""

from __future__ import annotations
import re
import numpy as np
from datasketch import MinHash, MinHashLSH
from logic.constants import COLLECTION_JSONL, SEED
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever

_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 3) -> set:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        t = text.lower()
        return {t[i:i + 3].encode("utf-8") for i in range(max(1, len(t) - 2))}
    return {" ".join(words[i:i + k]).encode("utf-8") for i in range(len(words) - k + 1)}


def _make_minhash(text: str, num_perm: int = 128, k: int = 3) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=SEED)
    for shingle in _shingles(text, k=k):
        m.update(shingle)
    return m


class MinHashLSHRetriever(BaseRetriever):
    """Approximate k-NN retriever using MinHash LSH.

    Parameters
    ----------
    num_initial_documents : 0 = lazy (call build_corpus_from_docs manually)
    top_k                 : default retrieval depth
    num_perm              : number of MinHash permutations
    lsh_threshold         : Jaccard threshold for LSH buckets.
                            Was 0.1 which caused constant O(n) fallback. Default now 0.5.
    shingle_k             : word n-gram size for shingling
    """

    def __init__(
        self,
        num_initial_documents: int = 10_000,
        top_k: int = 10,
        num_perm: int = 128,
        lsh_threshold: float = 0.5,
        shingle_k: int = 3,
    ):
        self.num_initial_documents = num_initial_documents
        self.top_k = top_k
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.shingle_k = shingle_k
        self.ids: list = []
        self.doc_texts: dict = {}
        self.lsh_index = None
        # Pre-built numpy uint64 matrix (n x num_perm) -- avoids Python loops at query time
        self._sig_matrix: np.ndarray = np.empty((0, num_perm), dtype=np.uint64)
        self._id_to_idx: dict = {}
        super().__init__(lazy=(num_initial_documents == 0))

    def build_corpus(self) -> None:
        n = self.num_initial_documents
        print(f"[MinHashLSHRetriever] Loading {n} documents...")
        docs = []
        for i, doc in enumerate(load_jsonl(COLLECTION_JSONL)):
            if i >= n:
                break
            docs.append(doc)
            if (i + 1) % 1_000 == 0:
                print(f"\r  {i + 1}/{n} loaded", end="", flush=True)
        print()
        self._index_docs(docs)

    def build_corpus_from_docs(self, docs: list) -> None:
        print(f"[MinHashLSHRetriever] Indexing {len(docs)} pre-loaded documents...")
        self._index_docs(docs)

    def update(self, document: dict) -> None:
        doc_id = document["key"]
        text = document["data"].strip()
        m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
        idx = len(self.ids)
        self.ids.append(doc_id)
        self.doc_texts[doc_id] = text
        self._id_to_idx[doc_id] = idx
        self._sig_matrix = np.vstack([self._sig_matrix, m.hashvalues.reshape(1, -1)])
        if self.lsh_index is not None:
            try:
                self.lsh_index.insert(doc_id, m)
            except ValueError:
                pass

    def retrieve(self, query: str, top_k=None) -> list:
        """Return top-k (doc_id, jaccard_score) pairs.

        Fast path : LSH bucket lookup + numpy vectorised re-rank.
        Fallback  : single numpy broadcast over full sig matrix -- O(n) at BLAS speed.
        """
        if self.lsh_index is None or not self.ids:
            return []
        k = top_k if top_k is not None else self.top_k
        query_mh = _make_minhash(query, num_perm=self.num_perm, k=self.shingle_k)
        q_hash = query_mh.hashvalues  # shape (num_perm,)

        candidates = self.lsh_index.query(query_mh)

        if len(candidates) >= k:
            # Re-rank LSH candidates with vectorised numpy -- no per-object .jaccard()
            idxs = [self._id_to_idx[cid] for cid in candidates if cid in self._id_to_idx]
            if not idxs:
                return []
            cand_sigs = self._sig_matrix[idxs]
            scores = np.mean(cand_sigs == q_hash, axis=1)
            order = np.argsort(scores)[::-1][:k]
            return [(candidates[i], float(scores[i])) for i in order]

        # Vectorised fallback: single (n, num_perm) == (num_perm,) broadcast
        scores_all = np.mean(self._sig_matrix == q_hash, axis=1)
        top_idxs = np.argsort(scores_all)[::-1][:k]
        return [(self.ids[int(i)], float(scores_all[i])) for i in top_idxs]

    def _index_docs(self, docs: list) -> None:
        self.ids = []
        self.doc_texts = {}
        self._id_to_idx = {}
        sig_rows = []
        self.lsh_index = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        for i, doc in enumerate(docs):
            doc_id = doc["key"]
            text = doc["data"].strip()
            m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
            self._id_to_idx[doc_id] = i
            self.ids.append(doc_id)
            self.doc_texts[doc_id] = text
            sig_rows.append(m.hashvalues)
            try:
                self.lsh_index.insert(doc_id, m)
            except ValueError:
                pass
        self._sig_matrix = (
            np.array(sig_rows, dtype=np.uint64) if sig_rows
            else np.empty((0, self.num_perm), dtype=np.uint64)
        )
        print(
            f"[MinHashLSHRetriever] LSH index ready "
            f"({len(self.ids)} docs, threshold={self.lsh_threshold}, num_perm={self.num_perm})."
        )
