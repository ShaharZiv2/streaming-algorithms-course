from logic import tf_idf
from logic.constants import COLLECTION_PROB_MIN_HASH, SEED, PROB_MIN_HASH_CORPUS_DIR, \
    TF_IDF_VECTORIZER, TF_IDF_MATRIX
from logic.prob_min_hash import ProbMinHash4
from logic.processing.file_utils import load_processed_file
from logic.stemming_utils import stemmed_stop_words, StemmingTokenizer

from retrievers.base_min_hash_retriever import BaseMinHashRetriever
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class ProbMinHashRetriever(BaseMinHashRetriever):

    def __init__(self, dbscan_eps: float = 0.85, corpus_initial_size: int = 100_000):
        # Only load/build the TF-IDF vectorizer when we're actually going to
        # use build_corpus() from disk.  When corpus_initial_size=0 we expect
        # build_corpus_from_docs() to be called instead, which builds its own.
        if corpus_initial_size > 0:
            self.tfidf_vectorizer = load_processed_file(TF_IDF_VECTORIZER) or tf_idf.build_vectorizer(corpus_initial_size)
            self.tfidf_matrix = load_processed_file(TF_IDF_MATRIX)
        else:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        super().__init__(dbscan_eps, corpus_initial_size)

    @property
    def _collection_path(self) -> str:
        return COLLECTION_PROB_MIN_HASH

    @property
    def _corpus_dir(self) -> str:
        return PROB_MIN_HASH_CORPUS_DIR

    def build_corpus_from_docs(self, docs: list[dict]):
        """Build ProbMinHash index from a pre-loaded list of {"key":…,"data":…} dicts."""
        print(f"[ProbMinHashRetriever] Building TF-IDF vectoriser over {len(docs)} documents…")
        texts = [d["data"].strip() for d in docs]
        self.doc_ids_list = [d["key"] for d in docs]

        # Build TF-IDF on the provided docs
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            tokenizer=StemmingTokenizer(),
            stop_words=stemmed_stop_words(),
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.corpus_size = len(docs)

        print(f"[ProbMinHashRetriever] Building ProbMinHash signatures…")
        sketches = []
        for i in range(self.corpus_size):
            start, end = tfidf_matrix.indptr[i], tfidf_matrix.indptr[i + 1]
            doc_keys = tfidf_matrix.indices[start:end]
            doc_weights = tfidf_matrix.data[start:end]
            pmh = ProbMinHash4(num_perm=128, seed=SEED)
            pmh.fit(doc_keys, doc_weights)
            sketches.append(pmh.hashvalues)

        self.ids = self.doc_ids_list
        self.signatures = np.array(sketches)
        self.min_hash_cursor = self.corpus_size

        print(f"[ProbMinHashRetriever] Running DBSCAN clustering…")
        min_samples = max(2, self.corpus_size // 100_000) if self.corpus_size >= 2 else 1
        self.dbscan.set_params(min_samples=min_samples)
        self.clusters = self.dbscan.fit_predict(self.signatures)
        self.corpus_df = pd.DataFrame({"doc_id": self.ids, "cluster": self.clusters})
        self._compute_centroids()
        print(f"[ProbMinHashRetriever] Index ready ({self.corpus_size} docs, "
              f"{len(self.centroids)} clusters).")

    def update(self, document: dict | int = 1):
        """Update is not supported for ProbMinHashRetriever (TF-IDF vocab is fixed at build time)."""
        pass

    def retrieve(self, query, **kwargs):
        query_mat = self.tfidf_vectorizer.transform([query])
        query_keys = query_mat[0].indices
        query_weights = query_mat[0].data
        if len(query_keys) == 0:
            return []
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



