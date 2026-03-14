from logic import tf_idf
from logic.constants import COLLECTION_PROB_MIN_HASH, SEED, PROB_MIN_HASH_CORPUS_DIR, \
    TF_IDF_VECTORIZER, TF_IDF_MATRIX
from logic.prob_min_hash import ProbMinHash4
from logic.processing.file_utils import load_processed_file

from retrievers.base_min_hash_retriever import BaseMinHashRetriever
import numpy as np


class ProbMinHashRetriever(BaseMinHashRetriever):

    def __init__(self, dbscan_eps: float = 0.85, corpus_initial_size: int = 100_000):
        self.tfidf_vectorizer = load_processed_file(TF_IDF_VECTORIZER) or tf_idf.build_vectorizer(corpus_initial_size)
        self.tfidf_matrix = load_processed_file(TF_IDF_MATRIX)
        super().__init__(dbscan_eps, corpus_initial_size)

    @property
    def _collection_path(self) -> str:
        return COLLECTION_PROB_MIN_HASH

    @property
    def _corpus_dir(self) -> str:
        return PROB_MIN_HASH_CORPUS_DIR

    def update(self, document):
        pass

    def retrieve(self, query):
        query_mat = self.tfidf_vectorizer.transform([query])
        query_keys = query_mat[0].indices
        query_weights = query_mat[0].data
        prob_min_hash = ProbMinHash4(num_perm=128, seed=SEED)
        prob_min_hash.fit(query_keys, query_weights)
        hash_values = prob_min_hash.hashvalues

        best_cluster = self._find_best_cluster(hash_values)
        return self._get_cluster_doc_ids(best_cluster)


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

        np.savez_compressed(COLLECTION_PROB_MIN_HASH, doc_ids=range(self.corpus_size), signatures=np.array(sketches))
        return np.load(COLLECTION_PROB_MIN_HASH)
