"""
minhash_retriever.py – Approximate nearest-neighbour retriever based on
MinHash Locality-Sensitive Hashing (LSH).

Algorithm
---------
1. Tokenise each document into k-word shingles.
2. Compute a 128-permutation MinHash signature per document.
3. Index all signatures in a datasketch.MinHashLSH table.
4. At query time:
   a. Compute the query's MinHash signature.
   b. Query the LSH index → candidate set (sub-linear in corpus size).
   c. Re-rank candidates by exact Jaccard similarity of MinHash signatures.
   d. Return top-k (doc_id, jaccard_score) pairs sorted by score.

Complexity
----------
Build  : O(n · d)  where d = document length
Query  : O(1) expected LSH lookup + O(|candidates|) re-rank  → sub-linear o(n)
Update : O(d)  – single insert into LSH index, no full rebuild
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from datasketch import MinHash, MinHashLSH

from logic.constants import COLLECTION_JSONL, SEED
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 3) -> set[bytes]:
    """Return the set of k-word shingles from *text* as UTF-8 bytes."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        # fallback: character 3-grams for very short texts
        t = text.lower()
        return {t[i:i + 3].encode("utf-8") for i in range(max(1, len(t) - 2))}
    return {" ".join(words[i:i + k]).encode("utf-8") for i in range(len(words) - k + 1)}


def _make_minhash(text: str, num_perm: int = 128, k: int = 3) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=SEED)
    for shingle in _shingles(text, k=k):
        m.update(shingle)
    return m


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MinHashRetriever(BaseRetriever):
    """Approximate k-NN retriever using MinHash LSH.

    Parameters
    ----------
    top_k               : default number of results to return
    num_perm            : MinHash permutations (128 is a good accuracy/speed tradeoff)
    lsh_threshold       : Jaccard similarity threshold for LSH bucket grouping;
                          lower → wider buckets → higher recall, more candidates
    shingle_k           : word-shingle size
    num_initial_documents: 0 = lazy (call build_corpus_from_docs manually)
    """

    def __init__(
        self,
        top_k: int = 10,
        num_perm: int = 128,
        lsh_threshold: float = 0.1,
        shingle_k: int = 3,
        num_initial_documents: int = 10_000,
    ):
        self.top_k = top_k
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.shingle_k = shingle_k
        self.num_initial_documents = num_initial_documents

        # populated by build_corpus / build_corpus_from_docs
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self._signatures: list[MinHash] = []
        self._lsh: Optional[MinHashLSH] = None

        super().__init__(lazy=(self.num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self, num_initial_documents: int | None = None) -> None:
        n = num_initial_documents if num_initial_documents is not None else self.num_initial_documents
        print(f"[MinHashRetriever] Loading {n} documents…")
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
        """Build LSH index from a pre-loaded list of {"key":…, "data":…} dicts."""
        print(f"[MinHashRetriever] Indexing {len(docs)} pre-loaded documents…")
        self._index_docs(docs)

    def update(self, document: dict) -> None:
        """Add a single document to the LSH index (O(d) – no full rebuild)."""
        doc_id = document["key"]
        text   = document["data"].strip()
        m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
        self.doc_ids.append(doc_id)
        self.doc_texts.append(text)
        self._signatures.append(m)
        if self._lsh is not None:
            try:
                self._lsh.insert(doc_id, m)
            except ValueError:
                pass  # duplicate key – skip

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return top-k (doc_id, jaccard_score) pairs for *query*.

        Sub-linear query time: LSH bucket lookup + small candidate re-rank.
        Falls back to full Hamming scan when LSH returns too few candidates.
        """
        if self._lsh is None or not self.doc_ids:
            return []

        k = top_k if top_k is not None else self.top_k
        query_mh = _make_minhash(query, num_perm=self.num_perm, k=self.shingle_k)

        # 1. Sub-linear LSH candidate lookup
        candidates: list[str] = self._lsh.query(query_mh)

        # 2. Fallback: Hamming-based scan when corpus is very small
        if len(candidates) < k:
            q_hash = query_mh.hashvalues
            scores_all = [
                (did, float(np.mean(sig.hashvalues == q_hash)))
                for did, sig in zip(self.doc_ids, self._signatures)
            ]
            scores_all.sort(key=lambda x: x[1], reverse=True)
            return scores_all[:k]

        # 3. Re-rank candidates by exact MinHash Jaccard estimate
        id_to_sig: dict[str, MinHash] = {
            did: sig for did, sig in zip(self.doc_ids, self._signatures)
        }
        scored = []
        for cid in candidates:
            sig = id_to_sig.get(cid)
            if sig is None:
                continue
            scored.append((cid, float(query_mh.jaccard(sig))))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_docs(self, docs: list[dict]) -> None:
        self.doc_ids = []
        self.doc_texts = []
        self._signatures = []

        self._lsh = MinHashLSH(
            threshold=self.lsh_threshold,
            num_perm=self.num_perm,
        )

        for doc in docs:
            doc_id = doc["key"]
            text   = doc["data"].strip()
            m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
            self.doc_ids.append(doc_id)
            self.doc_texts.append(text)
            self._signatures.append(m)
            try:
                self._lsh.insert(doc_id, m)
            except ValueError:
                pass  # duplicate key – skip

        print(f"[MinHashRetriever] LSH index ready ({len(self.doc_ids)} docs, "
              f"threshold={self.lsh_threshold}, num_perm={self.num_perm}).")

