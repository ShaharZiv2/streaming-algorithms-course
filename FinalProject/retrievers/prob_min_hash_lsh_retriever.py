from typing import List

from logic import tf_idf
from logic.constants import SEED, PROB_MIN_HASH_LSH_CORPUS_DIR, COLLECTION_PROB_MIN_HASH
from logic.prob_min_hash import ProbMinHash4

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

    def retrieve(self, query, **kwargs) -> List[np.str_]:
        query_mat = self.tfidf_vectorizer.transform([query])
        query_keys = query_mat[0].indices
        query_weights = query_mat[0].data
        if len(query_keys) == 0:
            return []
        prob_min_hash = ProbMinHash4(num_perm=128, seed=SEED)
        prob_min_hash.fit(query_keys, query_weights)

        return self.index.query(prob_min_hash.to_min_hash())

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
        pass
