"""
minhashLSH_retriever.py – Approximate nearest-neighbour retriever based on
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

import numpy as np
from datasketch import MinHash, MinHashLSH

from logic.constants import COLLECTION_JSONL, SEED
from logic.processing.file_utils import load_jsonl
from retrievers.base_retriever import BaseRetriever


# ---------------------------------------------------------------------------
# Tokenisation helpers  (mirrors SketchRetriever's vectorizer/ngram approach)
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

class MinHashLSHRetriever(BaseRetriever):
    """Approximate k-NN retriever using MinHash LSH.

    Constructor mirrors SketchRetriever:
        MinHashLSHRetriever(num_initial_documents=10_000, ...)

    Parameters
    ----------
    num_initial_documents : docs to auto-load from COLLECTION_JSONL on init;
                            0 = lazy – call build_corpus_from_docs() manually
    top_k                 : default number of results to return
    num_perm              : MinHash permutations (128 is a good tradeoff)
    lsh_threshold         : Jaccard threshold for LSH bucket grouping;
                            lower → wider buckets → higher recall
    shingle_k             : word-shingle size for tokenisation
    """

    def __init__(
        self,
        num_initial_documents: int = 10_000,   # first arg – matches SketchRetriever
        top_k: int = 10,
        num_perm: int = 128,
        lsh_threshold: float = 0.1,
        shingle_k: int = 3,
    ):
        self.num_initial_documents = num_initial_documents
        self.top_k = top_k
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.shingle_k = shingle_k

        # attribute names mirror SketchRetriever  (ids / signatures / doc_texts)
        self.ids: list[str] = []
        self.signatures: list[MinHash] = []
        self.doc_texts: dict[str, str] = {}     # doc_id → text  (same as Sketch)
        self.lsh_index: MinHashLSH | None = None

        super().__init__(lazy=(num_initial_documents == 0))

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def build_corpus(self) -> None:
        """Load first num_initial_documents docs from COLLECTION_JSONL and index."""
        n = self.num_initial_documents
        print(f"[MinHashLSHRetriever] Loading {n} documents…")
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
        print(f"[MinHashLSHRetriever] Indexing {len(docs)} pre-loaded documents…")
        self._index_docs(docs)

    def update(self, document: dict) -> None:
        """Add a single document dict to the LSH index (O(d) – no full rebuild)."""
        doc_id = document["key"]
        text   = document["data"].strip()
        m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
        self.ids.append(doc_id)
        self.signatures.append(m)
        self.doc_texts[doc_id] = text
        if self.lsh_index is not None:
            try:
                self.lsh_index.insert(doc_id, m)
            except ValueError:
                pass  # duplicate key – skip

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return top-k (doc_id, jaccard_score) pairs for *query*.

        Sub-linear query time: LSH bucket lookup + small candidate re-rank.
        Falls back to full Hamming scan when LSH returns too few candidates
        (mirrors SketchRetriever's centroid fallback for small corpora).
        """
        if self.lsh_index is None or not self.ids:
            return []

        k = top_k if top_k is not None else self.top_k
        query_mh = _make_minhash(query, num_perm=self.num_perm, k=self.shingle_k)

        # 1. Sub-linear LSH candidate lookup
        candidates: list[str] = self.lsh_index.query(query_mh)

        # 2. Fallback: Hamming similarity scan (mirrors Sketch's centroid distance)
        if len(candidates) < k:
            q_hash = query_mh.hashvalues
            scores_all = [
                (did, float(np.mean(sig.hashvalues == q_hash)))
                for did, sig in zip(self.ids, self.signatures)
            ]
            scores_all.sort(key=lambda x: x[1], reverse=True)
            return scores_all[:k]

        # 3. Re-rank candidates by exact MinHash Jaccard estimate
        id_to_sig: dict[str, MinHash] = dict(zip(self.ids, self.signatures))
        scored = [
            (cid, float(query_mh.jaccard(id_to_sig[cid])))
            for cid in candidates
            if cid in id_to_sig
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_docs(self, docs: list[dict]) -> None:
        """Compute MinHash signatures and build the LSH index."""
        self.ids = []
        self.signatures = []
        self.doc_texts = {}

        self.lsh_index = MinHashLSH(
            threshold=self.lsh_threshold,
            num_perm=self.num_perm,
        )

        for doc in docs:
            doc_id = doc["key"]
            text   = doc["data"].strip()
            m = _make_minhash(text, num_perm=self.num_perm, k=self.shingle_k)
            self.ids.append(doc_id)
            self.signatures.append(m)
            self.doc_texts[doc_id] = text
            try:
                self.lsh_index.insert(doc_id, m)
            except ValueError:
                pass  # duplicate key – skip

        print(f"[MinHashLSHRetriever] LSH index ready ({len(self.ids)} docs, "
              f"threshold={self.lsh_threshold}, num_perm={self.num_perm}).")

