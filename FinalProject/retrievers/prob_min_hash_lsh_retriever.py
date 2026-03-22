from typing import List

from logic import tf_idf
from logic.constants import SEED, PROB_MIN_HASH_LSH_CORPUS_DIR, COLLECTION_PROB_MIN_HASH
from logic.prob_min_hash import ProbMinHash4
from datasketch import MinHash as DMinHash

import numpy as np

from retrievers.base_min_hash_lsh_retriever import BaseMinHashLshRetriever


class ProbMinHashLshRetriever(BaseMinHashLshRetriever):

    def __init__(self, min_hash_lsh_eps: float = 0.85, corpus_initial_size: int = 100_000):
        if corpus_initial_size > 0:
            self.tfidf_vectorizer, self.tfidf_matrix = tf_idf.load_or_build_tfidf(corpus_initial_size)
        else:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        super().__init__(min_hash_lsh_eps, corpus_initial_size)

    @property
    def _collection_path(self) -> str:
        return COLLECTION_PROB_MIN_HASH

    @property
    def _corpus_dir(self) -> str:
        return PROB_MIN_HASH_LSH_CORPUS_DIR

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build ProbMinHash LSH index from a pre-loaded list of {"key":…,"data":…} dicts."""
        from datasketch import MinHashLSH
        from sklearn.feature_extraction.text import TfidfVectorizer
        from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer

        print(f"[ProbMinHashLshRetriever] Building TF-IDF vectoriser over {len(docs)} documents…")
        texts = [d["data"].strip() for d in docs]
        doc_ids = [d["key"] for d in docs]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            tokenizer=StemmingTokenizer(),
            stop_words=stemmed_stop_words(),
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.corpus_size = len(docs)

        print(f"[ProbMinHashLshRetriever] Building ProbMinHash signatures…")
        sketches = []
        for i in range(self.corpus_size):
            start, end = tfidf_matrix.indptr[i], tfidf_matrix.indptr[i + 1]
            doc_keys = tfidf_matrix.indices[start:end]
            doc_weights = tfidf_matrix.data[start:end]
            pmh = ProbMinHash4(num_perm=128, seed=SEED)
            pmh.fit(doc_keys, doc_weights)
            sketches.append(pmh.hashvalues)

        self.ids = doc_ids
        self.signatures = np.array(sketches)
        self.min_hash_cursor = self.corpus_size

        print(f"[ProbMinHashLshRetriever] Building LSH index (threshold={1 - self.min_hash_lsh_eps:.2f})…")
        self.index = MinHashLSH(threshold=1 - self.min_hash_lsh_eps, num_perm=128)
        with self.index.insertion_session() as session:
            for doc_id, sig in zip(self.ids, self.signatures):
                session.insert(doc_id, DMinHash(num_perm=128, hashvalues=sig))
        print(f"[ProbMinHashLshRetriever] LSH index ready ({len(doc_ids)} docs).")

    def retrieve(self, query, **kwargs) -> List[np.str_]:
        query_mat = self.tfidf_vectorizer.transform([query])
        query_keys = query_mat[0].indices
        query_weights = query_mat[0].data
        if len(query_keys) == 0:
            return []
        prob_min_hash = ProbMinHash4(num_perm=128, seed=SEED)
        prob_min_hash.fit(query_keys, query_weights)
        # Build the MinHash object directly from hashvalues — avoids the extra
        # object allocation and copy inside ProbMinHash4.to_min_hash().
        query_minhash = DMinHash(num_perm=128, hashvalues=prob_min_hash.hashvalues)
        return self.index.query(query_minhash)

    def _generate_docs(self):
        sketches = []
        for i in range(self.corpus_size):
            print(f'\rDocument: {i}, {100 * i / self.corpus_size:2f}% done', end='', flush=True)
            start, end = self.tfidf_matrix.indptr[i], self.tfidf_matrix.indptr[i + 1]
            document_keys = self.tfidf_matrix.indices[start:end]
            document_weights = self.tfidf_matrix.data[start:end]

            prob_min_hash = ProbMinHash4(num_perm=128, seed=SEED)
            prob_min_hash.fit(document_keys, document_weights)
            sketches.append(prob_min_hash.hashvalues)


        np.savez_compressed(self._collection_path, doc_ids=range(self.corpus_size), signatures=np.array(sketches))
        return np.load(self._collection_path)

    def update(self, document: dict | int = 1):
        """Append a single document to the LSH index.

        Note: TF-IDF vocab is fixed at build time so new tokens are ignored.
        """
        if isinstance(document, dict) and self.tfidf_vectorizer is not None:
            query_mat = self.tfidf_vectorizer.transform([document["data"].strip()])
            doc_keys = query_mat[0].indices
            doc_weights = query_mat[0].data
            if len(doc_keys) == 0:
                return
            pmh = ProbMinHash4(num_perm=128, seed=SEED)
            pmh.fit(doc_keys, doc_weights)
            doc_id = document["key"]
            self.ids.append(doc_id)
            new_sig = np.array(pmh.hashvalues).reshape(1, -1)
            self.signatures = np.vstack([self.signatures, new_sig])
            if self.index is not None:
                try:
                    self.index.insert(doc_id, DMinHash(num_perm=128, hashvalues=pmh.hashvalues))
                except Exception:
                    pass
